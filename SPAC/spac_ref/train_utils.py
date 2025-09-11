\
import torch

def add_uniform_noise(x, delta):
    # training-time noise U(-2/delta, 2/delta) following HSQ scaling idea
    # If delta==0: return x
    eps = (2.0 / (delta.clamp(min=1e-6))).unsqueeze(1).expand_as(x)
    u = torch.rand_like(x) * 2*eps - eps
    return x + u

def quantize(x, delta):
    # test-time quantization with scaling factor delta
    s = delta.clamp(min=1e-6)
    q = torch.round(x / s) * s
    return q

def rd_loss(d_loss, entropy_loss, hyper_loss, lam1, lam2):
    return d_loss + lam1 * entropy_loss + lam2 * hyper_loss
