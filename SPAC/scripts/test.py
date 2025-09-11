\
import os, argparse
import torch
from spac_ref.dataset import PLYDataset
from spac_ref.models.spac_models import SPACCodec
from spac_ref.utils.ply_io import write_ply_xyzrgb
from spac_ref.utils.color import yuv_to_rgb

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to checkpoint best.pt")
    ap.add_argument("--in_ply", type=str, required=True, help="Input PLY (*.ply)")
    ap.add_argument("--out_ply", type=str, required=True, help="Output reconstructed PLY (*.ply)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

def main():
    args = parse_args()
    ds = PLYDataset([args.in_ply], shuffle=False)
    sample = ds[0]
    xyz = sample["xyz"].numpy()
    yuv = sample["yuv"].numpy()
    normals = sample["normals"].numpy()
    idx = {k: v.numpy() for k,v in sample["idxs"].items()}
    model = SPACCodec(base_dim=128)
    ckpt = torch.load(args.model, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    with torch.no_grad():
        yhat_list, _ = model.encode_features(xyz, yuv, normals, idx, train=False)
        yuv_hat = model.decode_and_reconstruct(yhat_list, {k: torch.from_numpy(v).long() for k,v in idx.items()}, N=xyz.shape[0])
    rgb_hat = yuv_to_rgb(yuv_hat.cpu().numpy())
    write_ply_xyzrgb(args.out_ply, xyz, rgb_hat)

if __name__ == "__main__":
    main()
