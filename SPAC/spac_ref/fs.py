\
import numpy as np
from scipy.signal.windows import hamming

class FrequencySampler:
    """
    Frequency Sampling (FS) module as in SPAC.
    - Groups Ω points (by storage order)
    - Apply Hamming window + FFT on color channels
    - Keep coefficients with magnitude <= q% of max; zero the rest (Eq. 13)
    - IFFT -> spatial; select positions where IFFT != 0 as high-frequency samples
    Returns:
      high_idx: indices of sampled "high-frequency" points
      residual_idx: complementary indices
    """
    def __init__(self, group_size: int = 64, q: float = 0.6):
        self.group = int(group_size)
        self.q = float(q)

    def sample(self, colors: np.ndarray):
        """
        colors: (N,3) in YUV (Y in [0,1], U/V roughly [-0.5,0.5])
        """
        N = colors.shape[0]
        g = self.group
        win = hamming(g, sym=False).astype(np.float32)
        hf_mask = np.zeros(N, dtype=bool)
        # Process per group (last group can be shorter)
        for start in range(0, N, g):
            end = min(start + g, N)
            seg = colors[start:end]
            L = end - start
            if L <= 1:
                continue
            w = hamming(L, sym=False).astype(np.float32)
            for ch in range(3):
                c = seg[:, ch]
                cwin = w * c
                F = np.fft.fft(cwin)
                mag = np.abs(F)
                thr = self.q * np.max(mag) if mag.size else 0.0
                keep = mag >= thr
                Fzp = F * keep
                rec = np.fft.ifft(Fzp).real
                # mark non-zeros as high-freq
                mask = np.abs(rec) > 1e-8
                hf_mask[start:end] |= mask
        high_idx = np.where(hf_mask)[0]
        residual_idx = np.where(~hf_mask)[0]
        return high_idx, residual_idx
