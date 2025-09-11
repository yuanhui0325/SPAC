
import os, argparse, json, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from spac_ref.dataset import PLYDataset
from spac_ref.models.spac_sparse import SPACCodecSparse, HAS_ME
from spac_ref.models.spac_models import SPACCodec as PointCodec
from spac_ref.train_utils import rd_loss

ABLATIONS = {
    "full_sparse": {"backend":"sparse", "use_attn":True, "use_hyper":True, "use_ctx":True, "use_hsq":True, "use_normals":True},
    "no_normals": {"backend":"sparse", "use_attn":True, "use_hyper":True, "use_ctx":True, "use_hsq":True, "use_normals":False},
    "no_attn": {"backend":"sparse", "use_attn":False, "use_hyper":True, "use_ctx":True, "use_hsq":True, "use_normals":True},
    "no_hyper": {"backend":"sparse", "use_attn":True, "use_hyper":False, "use_ctx":True, "use_hsq":True, "use_normals":True},
    "no_ctx": {"backend":"sparse", "use_attn":True, "use_hyper":True, "use_ctx":False, "use_hsq":True, "use_normals":True},
    "no_hsq": {"backend":"sparse", "use_attn":True, "use_hyper":True, "use_ctx":True, "use_hsq":False, "use_normals":True},
    "point_backend": {"backend":"point", "use_attn":True, "use_hyper":True, "use_ctx":True, "use_hsq":True, "use_normals":True},
}

def psnr(mse):
    import math
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)

def run_eval(ds_glob, ckpt=None, variant="full_sparse", device=None, lambda1=400.0, lambda2=1.0):
    args = ABLATIONS[variant]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = PLYDataset(ds_glob, shuffle=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    if args["backend"]=="sparse":
        if not HAS_ME:
            raise RuntimeError("MinkowskiEngine is not installed for sparse backend")
        model = SPACCodecSparse(base_dim=96, use_attn=args["use_attn"], use_hyper=args["use_hyper"], use_ctx=args["use_ctx"], use_hsq=args["use_hsq"]).to(device)
    else:
        from spac_ref.models.spac_models import SPACCodec
        model = SPACCodec(base_dim=128).to(device)

    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"], strict=False)

    model.eval()
    out = {"variant": variant, "results": []}
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {variant}"):
            xyz = batch["xyz"][0].to(device)
            yuv = batch["yuv"][0].to(device)
            normals = batch["normals"][0].to(device) if args["use_normals"] else torch.zeros_like(batch["normals"][0]).to(device)
            idx = {k: v[0].to(device) for k,v in batch["idxs"].items()}
            if args["backend"]=="sparse":
                yhat_list, losses = model.forward_encode(xyz, yuv, normals, idx, train=False)
                yuv_hat = model.decode_reconstruct(yhat_list, xyz, idx)
            else:
                yhat_list, losses = model.encode_features(xyz.cpu().numpy(), yuv.cpu().numpy(), normals.cpu().numpy(), 
                                                         {k: v.cpu().numpy() for k,v in idx.items()}, train=False)
                yuv_hat = model.decode_and_reconstruct(yhat_list, idx, N=xyz.shape[0])
            mse = F.mse_loss(yuv_hat[:, :1], yuv[:, :1]).item()
            entropy = losses["entropy_nll"].item() if isinstance(losses["entropy_nll"], torch.Tensor) else losses["entropy_nll"]
            hyper = losses["hyper_nll"].item() if isinstance(losses["hyper_nll"], torch.Tensor) else losses["hyper_nll"]
            rd = mse + lambda1*entropy + lambda2*hyper
            out["results"].append({"mse": mse, "psnr": psnr(mse), "entropy_proxy": entropy, "hyper_proxy": hyper, "rd_loss_proxy": rd})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_glob", type=str, required=True)
    ap.add_argument("--variants", type=str, nargs="+", default=list(ABLATIONS.keys()))
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_json", type=str, default="ablation_results.json")
    ap.add_argument("--lambda1", type=float, default=400.0)
    ap.add_argument("--lambda2", type=float, default=1.0)
    args = ap.parse_args()

    all_res = {}
    for v in args.variants:
        res = run_eval(args.val_glob, ckpt=args.ckpt, variant=v, device=args.device, lambda1=args.lambda1, lambda2=args.lambda2)
        all_res[v] = res
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)
    print(f"[OK] Saved ablation to {args.out_json}")

if __name__ == "__main__":
    main()
