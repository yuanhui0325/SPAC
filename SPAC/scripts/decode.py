
import os, argparse, json, struct
import torch
import numpy as np
from spac_ref.models.spac_sparse import SPACCodecSparse, HAS_ME
from spac_ref.models.spac_models import SPACCodec as PointCodec
from spac_ref.entropy.ans import decode_laplace_stream, decode_gaussian_stream
from spac_ref.utils.ply_io import write_ply_xyzrgb, read_ply_xyzrgb
from spac_ref.utils.color import yuv_to_rgb

def parse_args():
    ap = argparse.ArgumentParser(description="Decode SPAC bitstream and reconstruct attributes. Optionally provide original XYZ to write a full-geometry PLY.")
    ap.add_argument("--bitstream", required=True, help="Input .spac file produced by scripts/encode.py")
    ap.add_argument("--ckpt", required=True, help="Checkpoint used during encoding")
    ap.add_argument("--out_ply", required=True, help="Output reconstructed PLY")
    ap.add_argument("--xyz_ply", type=str, default=None, help="Path to original geometry PLY (xyz will be read, colors ignored)")
    ap.add_argument("--xyz_npy", type=str, default=None, help="Path to original geometry .npy file storing float32 array of shape (N,3)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

def load_xyz_from_args(args, N_expected):
    if args.xyz_ply:
        xyz, _ = read_ply_xyzrgb(args.xyz_ply)  # returns (xyz, rgb)
        if xyz.shape[0] != N_expected:
            raise ValueError(f"xyz_ply has {xyz.shape[0]} points but bitstream expects {N_expected}.")
        return xyz.astype(np.float32)
    if args.xyz_npy:
        xyz = np.load(args.xyz_npy)
        if xyz.shape[0] != N_expected or xyz.shape[1] != 3:
            raise ValueError(f"xyz_npy has shape {xyz.shape}, expected ({N_expected},3).")
        return xyz.astype(np.float32)
    # Fallback: zeros with warning
    print("[WARN] No xyz provided. Using zeros. Provide --xyz_ply or --xyz_npy for real-geometry PLY.")
    return np.zeros((N_expected,3), dtype=np.float32)

def main():
    args = parse_args()
    device = args.device
    with open(args.bitstream, "rb") as f:
        (hlen,) = struct.unpack("<I", f.read(4))
        header = json.loads(f.read(hlen).decode("utf-8"))
        (zlen,) = struct.unpack("<I", f.read(4))
        z_bytes = f.read(zlen)
        y_bytes = []
        for _ in range(4):
            (l,) = struct.unpack("<I", f.read(4))
            y_bytes.append(f.read(l))

    backend = header["backend"]
    Qy = header["Qy"]; Qz = header["Qz"]
    shapes = header["shapes"]
    idx = {k: torch.tensor(v, dtype=torch.long, device=device) for k,v in header["idx"].items()}

    # Build model / entropy modules
    if backend=="sparse":
        if not HAS_ME:
            raise RuntimeError("MinkowskiEngine not available")
        from spac_ref.models.spac_sparse import SPACCodecSparse, EntropyModel
        model = SPACCodecSparse(base_dim=96, use_attn=True, use_hyper=True, use_ctx=False, use_hsq=False).to(device)
        em = model.entropy
    else:
        model = PointCodec(base_dim=128).to(device)
        raise NotImplementedError("Real bitstream path currently supports sparse backend.")

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    # Decode z
    N_z = shapes["z"][0] * shapes["z"][1]
    z_dec = decode_gaussian_stream(N_z, sigma=1.0, bitstream=z_bytes, Q=Qz, device="cpu").to(torch.float32)
    zL = z_dec.view(shapes["z"][0], shapes["z"][1]).to(device)
    glob = em.hyper_dec(zL.mean(dim=0, keepdim=True))
    mu_t, b_t = glob.chunk(2, dim=-1)
    b_t = torch.nn.functional.softplus(b_t) + 1e-3

    # Decode y1..y4
    y_list = []
    for li, key in enumerate(["y1","y2","y3","y4"]):
        N, C = shapes[key]
        mu = mu_t[0].unsqueeze(0).expand(N, -1).contiguous().view(-1).to("cpu")
        b  = b_t[0].unsqueeze(0).expand(N, -1).contiguous().view(-1).to("cpu")
        y_int = decode_laplace_stream(mu, b, y_bytes[li], Q=Qy)
        y = y_int.view(N, C).to(torch.float32).to(device)
        y_list.append(y)

    # Prepare geometry
    N_all = int(idx["P1"].numel())
    xyz_np = load_xyz_from_args(args, N_all)
    xyz_t = torch.from_numpy(xyz_np).to(device)

    # Reconstruct attributes
    yuv_hat = model.decode_reconstruct(y_list, xyz_t, idx)
    rgb_hat = yuv_to_rgb(yuv_hat.detach().cpu().numpy())

    # Write PLY with real geometry
    write_ply_xyzrgb(args.out_ply, xyz_np, rgb_hat)
    print(f"[OK] Wrote reconstructed PLY to {args.out_ply}")

if __name__ == "__main__":
    main()
