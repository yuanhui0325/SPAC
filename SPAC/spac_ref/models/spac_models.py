\
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import MinkowskiEngine as ME
    HAS_ME = True
except Exception:
    HAS_ME = False

class OffsetAttention(nn.Module):
    """Simplified offset-attention block capturing global dependencies."""
    def __init__(self, dim, heads=4):
        super().__init__()
        self.dim = dim
        self.h = heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x: (N, C)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # scaled dot-product
        d = x.shape[-1]
        att = torch.softmax(q @ k.t() / (d ** 0.5), dim=-1)
        out = att @ v
        return self.proj(out) + x

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth=2):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(depth-1):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)

class FNetPoint(nn.Module):
    """Point-based fallback when MinkowskiEngine is not available."""
    def __init__(self, in_dim=6, base=128, depth=1, use_normals=True):
        super().__init__()
        feats = in_dim + (3 if use_normals else 0)
        blocks = []
        for i in range(depth):
            blocks += [MLP(feats if i==0 else base, base, base, depth=2), nn.ReLU(inplace=True), OffsetAttention(base)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        # x: (N, D) features (includes normals if used)
        return self.net(x)

class ReFNetPoint(nn.Module):
    def __init__(self, dim=128, depth=2):
        super().__init__()
        blocks = []
        for i in range(depth):
            blocks += [MLP(dim, dim, dim, depth=2), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x): return self.net(x)

class ReconNetPoint(nn.Module):
    def __init__(self, dim=128, out_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.attn = OffsetAttention(dim)
        self.fc2 = nn.Linear(dim, out_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x), inplace=True)
        h = self.attn(h)
        return torch.sigmoid(self.fc2(h))  # predict color in [0,1]

class LaplaceLL(nn.Module):
    """Negative log-likelihood for Laplace with predicted mean/scale."""
    def __init__(self):
        super().__init__()

    def forward(self, y, mu, b):
        # b: scale >0
        b = torch.clamp(b, min=1e-6)
        return torch.log(2*b) + torch.abs(y - mu) / b

class HSQ(nn.Module):
    """Scaling factor predictor for additive uniform noise/quantization."""
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, 1), nn.Softplus()  # positive
        )

    def forward(self, y):
        return self.net(y).squeeze(-1)

class HyperEncoder(nn.Module):
    def __init__(self, in_dim=128, hidden=256, z_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, z_dim)
        )

    def forward(self, y):  # y from bottom layer
        return self.net(y)

class HyperDecoder(nn.Module):
    def __init__(self, z_dim=128, hidden=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim*2)  # to predict mu and scale (per-channel template)
        )

    def forward(self, z):
        h = self.net(z)
        return h  # caller splits into templates for mu/scale

class ContextModel(nn.Module):
    """Autoregressive context: here we approximate via a 1-layer MLP that conditions on local mean."""
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim*2)  # mu and scale per channel
        )

    def forward(self, y):
        # y: (N, C) latent; we compute simple global context by mean pooling as a proxy
        ctx = y.mean(dim=0, keepdim=True).expand_as(y)
        return self.net(ctx)

class EntropyModel(nn.Module):
    """
    Laplace likelihood with global hyperprior derived from bottom layer, + HSQ quantization.
    This models yl for all layers; bottom-layer yL is used to produce zL via HyperEncoder.
    """
    def __init__(self, dim=128, z_dim=128):
        super().__init__()
        self.hyper_enc = HyperEncoder(in_dim=dim, z_dim=z_dim)
        self.hyper_dec = HyperDecoder(z_dim=z_dim, out_dim=dim)
        self.ctx = ContextModel(dim=dim)
        self.hsq = HSQ(dim=dim)
        self.ll = LaplaceLL()

    def forward(self, y_layers, train=True):
        """
        y_layers: list of tensors [y1, y2, y3, y4] with y4=bottom.
        Returns: quantized yhat list, losses dict
        """
        y1, y2, y3, y4 = y_layers
        zL = self.hyper_enc(y4)
        # Factorized entropy for zL ~ N(0, sigma^2) (approx): compute -log prob
        # Use simple Gaussian NLL with unit variance for illustration
        z_nll = 0.5 * (zL**2 + 1.837877)  # -log N(0,1) up to constant
        # Decode global hyperprior template
        glob = self.hyper_dec(zL.mean(dim=0, keepdim=True))  # (1, 2*dim)
        mu_t, b_t = glob.chunk(2, dim=-1)
        b_t = F.softplus(b_t) + 1e-3

        losses = {}
        yhat_list, nll_sum = [], 0.0
        for y in [y1, y2, y3, y4]:
            # Context-predicted (mu, b) offset from global template
            ctx_out = self.ctx(y)
            mu_c, b_c = ctx_out.chunk(2, dim=-1)
            b_c = F.softplus(b_c) + 1e-3
            mu = mu_t + mu_c
            b = b_t + b_c
            # HSQ
            delta = self.hsq(y).clamp(min=1e-3, max=10.0)
            if train:
                y_tilde = y + (torch.rand_like(y)-0.5) * (4.0/delta).unsqueeze(1)
            else:
                y_tilde = torch.round(y * delta.unsqueeze(1)) / delta.unsqueeze(1)
            # NLL
            nll = self.ll(y_tilde, mu, b).mean()
            nll_sum = nll_sum + nll
            yhat_list.append(y_tilde)
        losses["entropy_nll"] = nll_sum
        losses["hyper_nll"] = z_nll.mean()
        return yhat_list, losses


class FNetBlock(nn.Module):
    def __init__(self, in_dim=6, base=128, depth=1, use_normals=True):
        super().__init__()
        self.fnet = FNetPoint(in_dim=in_dim, base=base, depth=depth, use_normals=use_normals)

    def forward(self, feats):
        return self.fnet(feats)

class ReFNet(nn.Module):
    def __init__(self, dim=128, depth=2):
        super().__init__()
        self.ref = ReFNetPoint(dim=dim, depth=depth)

    def forward(self, y): return self.ref(y)

class ReconNet(nn.Module):
    def __init__(self, dim=128, out_dim=3):
        super().__init__()
        self.rec = ReconNetPoint(dim=dim, out_dim=out_dim)

    def forward(self, y): return self.rec(y)

class SPACCodec(nn.Module):
    """
    Four-layer progressive attribute codec following the paper.
    Inputs:
      xyz: (N,3), rgb: (N,3) in [0,1], normals: (N,3)
    """
    def __init__(self, base_dim=128):
        super().__init__()
        # FNet depths per layer: shallow->deep (1..4)
        self.f1 = FNetBlock(in_dim=6, base=base_dim, depth=1, use_normals=True)
        self.f2 = FNetBlock(in_dim=6, base=base_dim, depth=2, use_normals=True)
        self.f3 = FNetBlock(in_dim=6, base=base_dim, depth=3, use_normals=True)
        self.f4 = FNetBlock(in_dim=6, base=base_dim, depth=4, use_normals=True)
        self.entropy = EntropyModel(dim=base_dim, z_dim=base_dim)
        self.ref1 = ReFNet(dim=base_dim, depth=2)
        self.ref2 = ReFNet(dim=base_dim, depth=2)
        self.ref3 = ReFNet(dim=base_dim, depth=2)
        self.ref4 = ReFNet(dim=base_dim, depth=2)
        self.rec = ReconNet(dim=base_dim, out_dim=3)

    def encode_features(self, xyz, yuv, normals, idx_layers, train=True):
        """
        idx_layers: dict with keys 'P2','R1','P3','R2','P4','R3' as indices into N
        """
        feats = torch.from_numpy(np.concatenate([yuv, normals], axis=1)).float()
        # Extract per residual
        y1 = self.f1(feats[idx_layers["R1"]])
        y2 = self.f2(feats[idx_layers["R2"]])
        y3 = self.f3(feats[idx_layers["R3"]])
        y4 = self.f4(feats[idx_layers["P4"]])  # bottom layer uses P4 points
        # Entropy model (with hyperprior + HSQ)
        yhat_list, losses = self.entropy([y1, y2, y3, y4], train=train)
        return yhat_list, losses

    def decode_and_reconstruct(self, yhat_list, idx_layers, N):
        # Decode per layer and reconstruct attributes progressively
        y1h, y2h, y3h, y4h = yhat_list
        r1 = self.rec(self.ref1(y1h))
        r2 = self.rec(self.ref2(y2h))
        r3 = self.rec(self.ref3(y3h))
        p4 = self.rec(self.ref4(y4h))
        # Assemble final by progressive sum: P̂l = P̂4 + sum(P̂i,res) (Eq. 20)
        yuv_hat = torch.zeros((N,3), dtype=torch.float32, device=r1.device)
        yuv_hat[idx_layers["P4"]] = p4
        yuv_hat[idx_layers["R1"]] += r1
        yuv_hat[idx_layers["R2"]] += r2
        yuv_hat[idx_layers["R3"]] += r3
        return torch.clamp(yuv_hat, 0.0, 1.0)

