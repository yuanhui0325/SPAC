
import os, argparse, json, struct
import torch
from spac_ref.dataset import PLYDataset
from spac_ref.models.spac_sparse import SPACCodecSparse, HAS_ME
from spac_ref.models.spac_models import SPACCodec as PointCodec
from spac_ref.utils.color import yuv_to_rgb, rgb_to_yuv
from spac_ref.utils.ply_io import read_ply_xyzrgb
from spac_ref.entropy.ans import encode_laplace_stream, encode_gaussian_stream

def flatten_by_channel(y, C):
    # y: (N, C) -> list of (Ni,) per channel to improve locality (optional)
    return [y[:, c].contiguous().view(-1) for c in range(C)]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ply", required=True)
    ap.add_argument("--ckpt", required=True, help="Trained checkpoint (weights must match decoder)")
    ap.add_argument("--out_bitstream", required=True, help="Output .spac file (binary)")
    ap.add_argument("--backend", choices=["sparse","point"], default="sparse")
    ap.add_argument("--Qy", type=int, default=255)
    ap.add_argument("--Qz", type=int, default=255)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

def main():
    args = parse_args()
    device = args.device
    # Load single sample
    xyz, rgb = read_ply_xyzrgb(args.in_ply)
    import numpy as np
    from spac_ref.utils.color import rgb_to_yuv
    yuv = rgb_to_yuv(rgb)
    from spac_ref.utils.normals import estimate_normals
    normals = estimate_normals(xyz, k=16)
    from spac_ref.fs import FrequencySampler
    fs = FrequencySampler(group_size=64, q=0.6)
    # build indices (tri-stage)
    import numpy as np
    P1 = np.arange(len(xyz))
    P2, R1 = fs.sample(yuv)
    P3_sub, R2_sub = fs.sample(yuv[P2])
    P3 = P2[P3_sub]; R2 = P2[R2_sub]
    P4_sub, R3_sub = fs.sample(yuv[P3])
    P4 = P3[P4_sub]; R3 = P3[R3_sub]
    idx = {"P1": P1, "P2": P2, "P3": P3, "P4": P4, "R1": R1, "R2": R2, "R3": R3}

    xyz_t = torch.from_numpy(xyz).float().to(device)
    yuv_t = torch.from_numpy(yuv).float().to(device)
    nrm_t = torch.from_numpy(normals).float().to(device)
    idx_t = {k: torch.from_numpy(v).long().to(device) for k,v in idx.items()}

    # Build model (ctx/hsq disabled to ensure decodability)
    if args.backend=="sparse":
        if not HAS_ME:
            raise RuntimeError("MinkowskiEngine not available")
        from spac_ref.models.spac_sparse import SPACCodecSparse
        model = SPACCodecSparse(base_dim=96, use_attn=True, use_hyper=True, use_ctx=False, use_hsq=False).to(device)
    else:
        model = PointCodec(base_dim=128).to(device)
        # NOTE: PointCodec currently has ctx/hsq enabled in EntropyModel; for bitstream we require ctx=False, hsq=False.
        # In practice, prefer sparse backend here.

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    # Extract features y1..y4 using encoder path (no quantization)
    if args.backend=="sparse":
        y_layers = model.extract_features(xyz_t, yuv_t, nrm_t, idx_t)  # y1..y4 (float)
        # Hyperprior from y4
        from spac_ref.models.spac_sparse import EntropyModel
        em = EntropyModel(dim=y_layers[-1].shape[1], z_dim=y_layers[-1].shape[1], use_hyper=True, use_ctx=False, use_hsq=False).to(device)
    else:
        raise NotImplementedError("Real bitstream path is provided for sparse backend.")

    with torch.no_grad():
        y1,y2,y3,y4 = y_layers
        # zL = HyperEncoder(y4)
        zL = em.hyper_enc(y4)
        # Quantize to integers
        z_int = torch.round(zL).to(torch.int32).view(-1)
        # Encode z with fixed Gaussian N(0,1)
        z_bytes = encode_gaussian_stream(z_int, sigma=1.0, Q=args.Qz)

        # Decode global template to get (mu_t, b_t)
        glob = em.hyper_dec(zL.mean(dim=0, keepdim=True))
        mu_t, b_t = glob.chunk(2, dim=-1)
        b_t = torch.nn.functional.softplus(b_t) + 1e-3
        # Broadcast per-layer
        params = []
        for y in [y1,y2,y3,y4]:
            # Per-channel params
            C = y.shape[1]
            mu = mu_t[0].unsqueeze(0).expand(y.shape[0], -1).contiguous().view(-1)
            b  = b_t[0].unsqueeze(0).expand(y.shape[0], -1).contiguous().view(-1)
            # Quantize y to integers
            y_int = torch.round(y).to(torch.int32).view(-1)
            # Encode layer stream with Laplace(mu,b)
            y_bytes = encode_laplace_stream(y_int, mu, b, Q=args.Qy)
            params.append((y_bytes, y.shape))

    # Pack to a binary file with a simple JSON header
    header = {
        "backend": args.backend,
        "Qy": args.Qy, "Qz": args.Qz,
        "shapes": {
            "y1": [int(y1.shape[0]), int(y1.shape[1])],
            "y2": [int(y2.shape[0]), int(y2.shape[1])],
            "y3": [int(y3.shape[0]), int(y3.shape[1])],
            "y4": [int(y4.shape[0]), int(y4.shape[1])],
            "z":  [int(zL.shape[0]), int(zL.shape[1])]
        },
        "idx": {k: v.tolist() for k,v in idx.items()}
    }
    with open(args.out_bitstream, "wb") as f:
        hb = json.dumps(header).encode("utf-8")
        f.write(struct.pack("<I", len(hb)))
        f.write(hb)
        # write z stream
        f.write(struct.pack("<I", len(z_bytes)))
        f.write(z_bytes)
        # write y streams (y1..y4)
        for (yb, shp) in params:
            f.write(struct.pack("<I", len(yb)))
            f.write(yb)
    print(f"[OK] Wrote bitstream to {args.out_bitstream}")

if __name__ == "__main__":
    main()
