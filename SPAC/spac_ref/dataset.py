\
import os, glob, random
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils.ply_io import read_ply_xyzrgb
from .utils.color import rgb_to_yuv
from .utils.normals import estimate_normals
from .fs import FrequencySampler

class PLYDataset(Dataset):
    def __init__(self, paths, group=64, q=0.6, layers=4, shuffle=True):
        self.paths = sorted(paths) if isinstance(paths, (list,tuple)) else sorted(glob.glob(paths))
        self.fs = FrequencySampler(group_size=group, q=q)
        self.layers = layers
        self.shuffle = shuffle

    def __len__(self): return len(self.paths)

    def _multi_stage_fs(self, xyz, yuv):
        N = len(xyz)
        # Stage 1: P1 -> split to P2 (HF) and R1 (residual)
        P1_idx = np.arange(N)
        P2_idx, R1_idx = self.fs.sample(yuv)
        # Stage 2: P2 -> P3 and R2
        P2_rgb = yuv[P2_idx]
        P3_sub, R2_sub = self.fs.sample(P2_rgb)
        P3_idx = P2_idx[P3_sub]; R2_idx = P2_idx[R2_sub]
        # Stage 3: P3 -> P4 and R3
        P3_rgb = yuv[P3_idx]
        P4_sub, R3_sub = self.fs.sample(P3_rgb)
        P4_idx = P3_idx[P4_sub]; R3_idx = P3_idx[R3_sub]
        return {
            "P1": P1_idx, "P2": P2_idx, "P3": P3_idx, "P4": P4_idx,
            "R1": R1_idx, "R2": R2_idx, "R3": R3_idx
        }

    def __getitem__(self, i):
        xyz, rgb = read_ply_xyzrgb(self.paths[i])
        yuv = rgb_to_yuv(rgb)
        if self.shuffle:
            idx = np.random.permutation(len(xyz))
            xyz, rgb = xyz[idx], rgb[idx]
        idxs = self._multi_stage_fs(xyz, yuv)
        normals = estimate_normals(xyz, k=16)
        data = {
            "xyz": torch.from_numpy(xyz).float(),
            "yuv": torch.from_numpy(yuv).float(),
            "normals": torch.from_numpy(normals).float(),
            "idxs": {k: torch.from_numpy(v).long() for k,v in idxs.items()}
        }
        return data
