
import numpy as np

# BT.601 (approx.) YUV with full-range RGB in [0,1]; U,V roughly in [-0.5, 0.5]
def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    yuv = np.stack([y, u, v], axis=-1).astype(np.float32)
    return yuv

def yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    y = yuv[..., 0]
    u = yuv[..., 1]
    v = yuv[..., 2]
    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)
