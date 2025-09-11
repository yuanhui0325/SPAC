\
import os, argparse, json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from spac_ref.dataset import PLYDataset
from spac_ref.models.spac_models import SPACCodec
from spac_ref.train_utils import rd_loss

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_glob", type=str, required=True, help="Glob to training PLY files")
    ap.add_argument("--val_glob", type=str, required=True, help="Glob to validation PLY files")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda1", type=float, default=400.0, help="Entropy weight")
    ap.add_argument("--lambda2", type=float, default=1.0, help="Hyperprior weight")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", type=str, default="runs/spac")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train_ds = PLYDataset(args.train_glob, group=64, q=0.6)
    val_ds = PLYDataset(args.val_glob, group=64, q=0.6, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = SPACCodec(base_dim=128).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = 1e9
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        tr_loss = 0.0
        for batch in pbar:
            xyz = batch["xyz"][0].to(args.device)  # bs=1 for variable N
            yuv = batch["yuv"][0].to(args.device)
            normals = batch["normals"][0].to(args.device)
            idx = {k: v[0].to(args.device) for k,v in batch["idxs"].items()}
            # Encode
            yhat_list, losses = model.encode_features(xyz.cpu().numpy(), yuv.cpu().numpy(), normals.cpu().numpy(), 
                                                     {k: v.cpu().numpy() for k,v in idx.items()}, train=True)
            # Decode & reconstruct
            yuv_hat = model.decode_and_reconstruct(yhat_list, idx, N=xyz.shape[0])
            # Distortion MSE (Eq. 21 approximated on full cloud)
            d_loss = F.mse_loss(yuv_hat[:, :1], yuv[:, :1])  # optimize luminance (Y) MSE as in common practice
            loss = rd_loss(d_loss, losses["entropy_nll"], losses["hyper_nll"], args.lambda1, args.lambda2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "d": f"{d_loss.item():.4f}"})
        # Simple val
        model.eval()
        with torch.no_grad():
            vd = 0.0; n=0
            for batch in val_loader:
                xyz = batch["xyz"][0].to(args.device)
                yuv = batch["yuv"][0].to(args.device)
                normals = batch["normals"][0].to(args.device)
                idx = {k: v[0].to(args.device) for k,v in batch["idxs"].items()}
                yhat_list, losses = model.encode_features(xyz.cpu().numpy(), yuv.cpu().numpy(), normals.cpu().numpy(), 
                                                         {k: v.cpu().numpy() for k,v in idx.items()}, train=False)
                yuv_hat = model.decode_and_reconstruct(yhat_list, idx, N=xyz.shape[0])
                d_loss = F.mse_loss(yuv_hat[:, :1], yuv[:, :1])  # optimize luminance (Y) MSE as in common practice
                vd += d_loss.item(); n+=1
            vd = vd / max(1,n)
        ckpt_path = os.path.join(args.save_dir, f"epoch{epoch:04d}_valMSE{vd:.6f}.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)
        if vd < best:
            best = vd
            torch.save({"model": model.state_dict(), "epoch": epoch}, os.path.join(args.save_dir, "best.pt"))

if __name__ == "__main__":
    main()
