\
import numpy as np
from sklearn.neighbors import NearestNeighbors

def estimate_normals(xyz: np.ndarray, k: int = 16):
    """
    PCA-based normal estimation. Returns (N,3) normals with unit length.
    """
    nbrs = NearestNeighbors(n_neighbors=min(k, len(xyz))).fit(xyz)
    _, idx = nbrs.kneighbors(xyz)
    normals = np.zeros_like(xyz, dtype=np.float32)
    for i in range(len(xyz)):
        pts = xyz[idx[i]]
        cov = np.cov(pts.T)
        # smallest eigenvector is normal
        w, v = np.linalg.eigh(cov)
        n = v[:, np.argmin(w)]
        # Orient consistently using viewpoint at infinity along +z
        if n[2] < 0: n = -n
        normals[i] = n / (np.linalg.norm(n) + 1e-8)
    return normals
