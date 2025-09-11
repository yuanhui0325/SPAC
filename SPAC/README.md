SPAC
===============================================================

1) Environment Setup
--------------------
Requirements:
- Python 3.9–3.11
- PyTorch >= 2.1 (GPU with CUDA 11.8/12.x recommended)
- Packages: numpy, scipy, tqdm, compressai (optional), torchac (optional),
            trimesh, plyfile, scikit-learn


Install 
  python -m venv .venv
  source .venv/bin/activate
  pip install -U pip
  pip install -r requirements.txt
  pip install -e .


Verify ME:
  python - << 'PY'
import MinkowskiEngine as ME
print("MinkowskiEngine OK")
PY

Check your torch/CUDA (to select the right ME wheel):
  python - << 'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
PY


2) Training
-----------
  python scripts/train.py     --train_glob "/path/to/train/*.ply"     --val_glob   "/path/to/val/*.ply"     --epochs 100 --bs 1 --lr 1e-4     --lambda1 400 --lambda2 1     --save_dir runs/point

Notes:
- lambda1 controls the rate–distortion trade-off.
- Keep --bs 1 if memory is tight .


3) Testing / Reconstruction
---------------------------
3.1 Point Backend
  python scripts/test.py     --model  runs/point/best.pt     --in_ply /path/to/single_input.ply     --out_ply /path/to/recon_output.ply

3.2 Sparse Convolution Backend
Edit scripts/test.py: replace the model import with the sparse version.

Then run the same command as in 3.1 to export the reconstructed PLY.
