
import torch
import torch.nn.functional as F
try:
    import torchac
except Exception as e:
    raise ImportError("torchac is required for ANS coding. Please install torchac>=0.9.3") from e


def discrete_laplace_cdf(symbols, mu, b, Q=255):
    """
    Build float-CDF for each symbol in `symbols` given Laplace params mu, b.
    - symbols: (N,) int tensor in [-Q, Q]
    - mu, b:   (N,) tensors (per-symbol), b>0
    Returns:
      cdf: (N, 2*Q+2) float tensor in [0,1], monotonically increasing, cdf[...,0]=0, cdf[...,-1]=1
      sym: (N,) int tensor shifted to [0, 2Q] for torchac
    """
    device = symbols.device
    N = symbols.numel()
    # bin edges v: [-Q-0.5, ..., Q+0.5], length L = 2Q+2
    v = torch.linspace(-Q-0.5, Q+0.5, 2*Q+2, device=device)
    # continuous Laplace CDF at edges
    # F(x) = 0.5 + 0.5*sign(x-mu)*(1 - exp(-|x-mu|/b))
    # vectorize per sample using broadcasting
    x = v.unsqueeze(0).expand(N, -1)  # (N,L)
    mu = mu.view(-1,1)
    b = torch.clamp(b.view(-1,1), min=1e-6)
    diff = x - mu
    sgn = torch.sign(diff)
    Fedge = 0.5 + 0.5*sgn * (1.0 - torch.exp(-torch.abs(diff) / b))
    Fedge = torch.clamp(Fedge, 0.0, 1.0)
    # ensure boundary values are exact
    Fedge[:,0] = 0.0
    Fedge[:,-1] = 1.0
    # symbols to range [0, 2Q]
    sym = torch.clamp(symbols, -Q, Q) + Q
    return Fedge.contiguous(), sym.contiguous()

def discrete_gaussian_cdf(symbols, sigma=1.0, Q=255):
    """
    Standard normal N(0, sigma^2) discretized on Z with unit bins.
    Returns cdf and shifted symbols for torchac.
    """
    device = symbols.device
    N = symbols.numel()
    v = torch.linspace(-Q-0.5, Q+0.5, 2*Q+2, device=device)
    x = v.unsqueeze(0).expand(N, -1)
    # Gaussian CDF via error function
    s = sigma
    Fedge = 0.5 * (1.0 + torch.erf(x / (math.sqrt(2)*s)))
    Fedge = torch.clamp(Fedge, 0.0, 1.0)
    Fedge[:,0] = 0.0
    Fedge[:,-1] = 1.0
    sym = torch.clamp(symbols, -Q, Q) + Q
    return Fedge.contiguous(), sym.contiguous()

def ans_encode_float_cdf(cdf, symbols):
    """
    cdf: (N, L) float, symbols: (N,) int in [0, L-2]
    Returns bytes.
    """
    # torchac wants [B, L] and [B] int
    bs = torchac.encode_float_cdf(cdf.cpu(), symbols.cpu())
    return bs

def ans_decode_float_cdf(cdf, bitstream):
    """
    cdf: (N, L) float
    Returns decoded (N,) int tensor.
    """
    dec = torchac.decode_float_cdf(cdf.cpu(), bitstream)
    return dec


def encode_laplace_stream(y_int, mu, b, Q=255):
    """
    Encode integer y with per-symbol Laplace params.
    Returns bytes.
    """
    cdf, syms = discrete_laplace_cdf(y_int, mu, b, Q=Q)
    return ans_encode_float_cdf(cdf, syms)

def decode_laplace_stream(mu, b, bitstream, Q=255):
    """
    Decode stream given per-symbol Laplace params.
    Returns y_int.
    """
    # we need the same CDF as encoder
    # Build dummy symbols to shape the CDF; torchac ignores symbol values here
    N = mu.numel()
    tmp_symbols = torch.zeros(N, dtype=torch.int32, device=mu.device)
    cdf, _ = discrete_laplace_cdf(tmp_symbols, mu, b, Q=Q)
    dec = ans_decode_float_cdf(cdf, bitstream)
    return dec.to(torch.int32) - Q

def encode_gaussian_stream(z_int, sigma=1.0, Q=255):
    cdf, syms = discrete_gaussian_cdf(z_int, sigma=sigma, Q=Q)
    return ans_encode_float_cdf(cdf, syms)

def decode_gaussian_stream(N, sigma, bitstream, Q=255, device="cpu"):
    tmp = torch.zeros(N, dtype=torch.int32, device=device)
    cdf, _ = discrete_gaussian_cdf(tmp, sigma=sigma, Q=Q)
    dec = ans_decode_float_cdf(cdf, bitstream)
    return (dec.to(torch.int32) - Q)
