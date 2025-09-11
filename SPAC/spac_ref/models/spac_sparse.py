
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import MinkowskiEngine as ME
    HAS_ME = True
except Exception:
    HAS_ME = False

class OffsetAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        d = x.shape[-1]
        att = torch.softmax(q @ k.t() / (d ** 0.5), dim=-1)
        out = att @ v
        return self.proj(out) + x

class SparseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, heads=0, use_attn=False):
        super().__init__()
        assert HAS_ME, "MinkowskiEngine is required for SparseBlock"
        self.conv = ME.MinkowskiConvolution(
            in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=stride,
            dimension=3)
        self.bn = ME.MinkowskiBatchNorm(out_ch)
        self.act = ME.MinkowskiReLU(inplace=True)
        self.use_attn = use_attn
        if use_attn:
            self.attn = OffsetAttention(out_ch)

    def forward(self, x: "ME.SparseTensor"):
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        if self.use_attn:
            # Apply attention in feature space per batch separately
            feats = y.F
            feats = self.attn(feats)
            y = ME.SparseTensor(features=feats, coordinate_map_key=y.coordinate_map_key, coordinate_manager=y.coordinate_manager)
        return y

class SparseFNet(nn.Module):
    """
    Sparse FNet operating on 2x2x2 child cells at the octree second-to-last level.
    Input: coords (M,4 int32 [b,x,y,z]), feats (M,Cin), and a list 'child_to_points'
           mapping each child cell to a 1D LongTensor of original point indices.
    Output: per-point latent y (N_points, C)
    """
    def __init__(self, in_ch=6, base=96, depth=3, use_attn=True):
        super().__init__()
        assert HAS_ME, "MinkowskiEngine required"
        ch = base
        blocks = [SparseBlock(in_ch, ch, k=2, use_attn=False)]
        for i in range(depth-1):
            blocks += [SparseBlock(ch, ch, k=3, use_attn=use_attn)]
        self.net = nn.Sequential(*blocks)

    def forward(self, coords, feats, child_to_points, n_points_layer):
        st = ME.SparseTensor(features=feats, coordinates=coords)
        y = st
        for blk in self.net:
            y = blk(y)
        # y.F: (M, C). Map back to points by broadcasting child feature to all its points
        child_feats = y.F
        C = child_feats.shape[1]
        device = child_feats.device
        y_pts = torch.zeros((n_points_layer, C), device=device, dtype=child_feats.dtype)
        for i, pts in enumerate(child_to_points):
            if pts.numel() == 0: 
                continue
            y_pts[pts] = child_feats[i].unsqueeze(0).expand(pts.numel(), C)
        return y_pts

class SparseReFNet(nn.Module):
    def __init__(self, dim=96, depth=2, use_attn=False):
        super().__init__()
        assert HAS_ME, "MinkowskiEngine required"
        blocks = [SparseBlock(dim, dim, k=3, use_attn=use_attn) for _ in range(depth)]
        self.net = nn.Sequential(*blocks)

    def forward(self, coords, feats, child_to_points, n_points_layer):
        st = ME.SparseTensor(features=feats, coordinates=coords)
        y = st
        for blk in self.net:
            y = blk(y)
        child_feats = y.F
        C = child_feats.shape[1]
        device = child_feats.device
        y_pts = torch.zeros((n_points_layer, C), device=device, dtype=child_feats.dtype)
        for i, pts in enumerate(child_to_points):
            if pts.numel() == 0: 
                continue
            y_pts[pts] = child_feats[i].unsqueeze(0).expand(pts.numel(), C)
        return y_pts

class ReconNetPoint(nn.Module):
    def __init__(self, dim=96, out_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, out_dim)

    def forward(self, x):
        return torch.sigmoid(self.fc2(F.relu(self.fc1(x), inplace=True)))

class LaplaceLL(nn.Module):
    def forward(self, y, mu, b):
        b = torch.clamp(b, min=1e-6)
        return torch.log(2*b) + torch.abs(y - mu) / b

class HSQ(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, 1), nn.Softplus()
        )

    def forward(self, y):
        return self.net(y).squeeze(-1)

class HyperEncoder(nn.Module):
    def __init__(self, in_dim=96, hidden=192, z_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, z_dim)
        )

    def forward(self, y): return self.net(y)

class HyperDecoder(nn.Module):
    def __init__(self, z_dim=96, hidden=192, out_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim*2)
        )
    def forward(self, z): return self.net(z)

class ContextModel(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim*2)
        )
    def forward(self, y):
        ctx = y.mean(dim=0, keepdim=True).expand_as(y)
        return self.net(ctx)

class EntropyModel(nn.Module):
    def __init__(self, dim=96, z_dim=96, use_hyper=True, use_ctx=True, use_hsq=True):
        super().__init__()
        self.use_hyper = use_hyper
        self.use_ctx = use_ctx
        self.use_hsq = use_hsq
        self.ll = LaplaceLL()
        if use_hyper:
            self.hyper_enc = HyperEncoder(in_dim=dim, z_dim=z_dim)
            self.hyper_dec = HyperDecoder(z_dim=z_dim, out_dim=dim)
        if use_ctx:
            self.ctx = ContextModel(dim=dim)
        if use_hsq:
            self.hsq = HSQ(dim=dim)

    def forward(self, y_layers, train=True):
        y1,y2,y3,y4 = y_layers
        if self.use_hyper:
            zL = self.hyper_enc(y4)
            z_nll = 0.5 * (zL**2 + 1.837877)
            glob = self.hyper_dec(zL.mean(dim=0, keepdim=True))
            mu_t, b_t = glob.chunk(2, dim=-1)
            b_t = F.softplus(b_t) + 1e-3
        else:
            z_nll = torch.zeros_like(y4[:, :1]).mean()
            mu_t = torch.zeros((1, y4.shape[1]), device=y4.device)
            b_t = torch.ones((1, y4.shape[1]), device=y4.device) * 0.5

        losses = {}
        yhat_list, nll_sum = [], 0.0
        for y in [y1,y2,y3,y4]:
            if self.use_ctx:
                mu_c, b_c = self.ctx(y).chunk(2, dim=-1)
                b_c = F.softplus(b_c) + 1e-3
                mu = mu_t + mu_c
                b = b_t + b_c
            else:
                mu = mu_t.expand_as(y)
                b = b_t.expand_as(y)

            if self.use_hsq:
                delta = self.hsq(y).clamp(min=1e-3, max=10.0)
                if train:
                    y_tilde = y + (torch.rand_like(y)-0.5) * (4.0/delta).unsqueeze(1)
                else:
                    y_tilde = torch.round(y * delta.unsqueeze(1)) / delta.unsqueeze(1)
            else:
                # Straight-through rounding proxy
                if train:
                    y_tilde = y + torch.empty_like(y).uniform_(-0.5, 0.5)
                else:
                    y_tilde = torch.round(y)

            nll = self.ll(y_tilde, mu, b).mean()
            nll_sum = nll_sum + nll
            yhat_list.append(y_tilde)
        losses["entropy_nll"] = nll_sum
        losses["hyper_nll"] = z_nll.mean()
        return yhat_list, losses

# ----------------- Helper to build sparse tensors from octree cells -----------
def build_sparse_from_octree(xyz, feats, octree_nodes):
    """
    Args:
      xyz: (N,3) tensor (selected layer points)
      feats: (N,Cin) tensor
      octree_nodes: list of nodes, each with .children where each child has .points_idx, .bbox_min, .bbox_max
    Returns:
      coords (M,4), feats_cell (M,Cin), child_to_points (list of LongTensor)
    """
    device = feats.device
    coords = []
    feats_cell = []
    child_to_points = []
    b = 0
    for node in octree_nodes:
        # map each child to code in [0..7]
        for ch in node.children:
            idxs = torch.as_tensor(ch.points_idx, device=device, dtype=torch.long)
            if idxs.numel() == 0:
                continue
            # Determine child code by comparing bbox min to node bbox min (bit=1 if min>)
            code = 0
            for d in range(3):
                bit = 1 if (ch.bbox_min[d] > node.bbox_min[d]) else 0
                code = (code << 1) | bit
            x = (code >> 2) & 1
            y = (code >> 1) & 1
            z = code & 1
            coords.append([b, x, y, z])
            feats_cell.append(feats[idxs].mean(dim=0))
            child_to_points.append(idxs)
        b += 1
    if len(coords) == 0:
        return torch.zeros((0,4), dtype=torch.int32, device=device), torch.zeros((0, feats.shape[1]), device=device), []
    coords = torch.as_tensor(coords, dtype=torch.int32, device=device)
    feats_cell = torch.stack(feats_cell, dim=0)
    return coords, feats_cell, child_to_points


class SPACCodecSparse(nn.Module):
    def __init__(self, base_dim=96, use_attn=True, use_hyper=True, use_ctx=True, use_hsq=True):
        super().__init__()
        assert HAS_ME, "MinkowskiEngine is required for SPACCodecSparse"
        self.base = base_dim
        self.f1 = SparseFNet(in_ch=6, base=base_dim, depth=2, use_attn=use_attn)
        self.f2 = SparseFNet(in_ch=6, base=base_dim, depth=3, use_attn=use_attn)
        self.f3 = SparseFNet(in_ch=6, base=base_dim, depth=4, use_attn=use_attn)
        self.f4 = SparseFNet(in_ch=6, base=base_dim, depth=4, use_attn=use_attn)
        self.entropy = EntropyModel(dim=base_dim, z_dim=base_dim, use_hyper=use_hyper, use_ctx=use_ctx, use_hsq=use_hsq)
        self.ref1 = SparseReFNet(dim=base_dim, depth=2, use_attn=False)
        self.ref2 = SparseReFNet(dim=base_dim, depth=2, use_attn=False)
        self.ref3 = SparseReFNet(dim=base_dim, depth=2, use_attn=False)
        self.ref4 = SparseReFNet(dim=base_dim, depth=2, use_attn=False)
        self.rec = ReconNetPoint(dim=base_dim, out_dim=3)

    def _build_nodes(self, xyz_np, max_leaf_points=8, max_depth=8):
        # local inline octree builder (reuse logic from spac_ref.octree if available)
        from spac_ref.octree import build_octree, collect_second_to_last_level
        root = build_octree(xyz_np, max_leaf_points=max_leaf_points, max_depth=max_depth)
        nodes = collect_second_to_last_level(root)
        return nodes

    def _encode_layer(self, xyz, yuv, normals):
        # xyz/rgb/normals: tensors for the current layer (N,3)
        device = xyz.device
        feats = torch.cat([yuv, normals], dim=1)
        nodes = self._build_nodes(xyz.cpu().numpy())
        coords, feats_cell, child_to_points = build_sparse_from_octree(xyz, feats, nodes)
        if coords.shape[0] == 0:
            # fallback: single cell with average
            coords = torch.tensor([[0,0,0,0]], dtype=torch.int32, device=device)
            feats_cell = feats.mean(dim=0, keepdim=True)
            child_to_points = [torch.arange(xyz.shape[0], device=device, dtype=torch.long)]
        return coords, feats_cell, child_to_points

    def extract_features(self, xyz_all, yuv_all, normals_all, idx_layers):
        # Build per-layer sparse inputs and run SparseFNet to obtain y_l
        ys = []
        for key, fnet in zip(["R1","R2","R3","P4"], [self.f1,self.f2,self.f3,self.f4]):
            ids = idx_layers[key]
            xyz = xyz_all[ids]
            rgb = yuv_all[ids]
            nrm = normals_all[ids]
            coords, feats_cell, child_to_points = self._encode_layer(xyz, yuv, nrm)
            y_pts = fnet(coords, feats_cell, child_to_points, n_points_layer=xyz.shape[0])
            ys.append(y_pts)
        return ys  # y1,y2,y3,y4

    def forward_encode(self, xyz_all, yuv_all, normals_all, idx_layers, train=True):
        y_layers = self.extract_features(xyz_all, yuv_all, normals_all, idx_layers)
        yhat_list, losses = self.entropy(y_layers, train=train)
        return yhat_list, losses

    def decode_reconstruct(self, yhat_list, xyz_all, idx_layers):
        # Run sparse ReFNet per layer, then ReconNet (pointwise) and merge
        yuv_hat = torch.zeros((xyz_all.shape[0], 3), device=xyz_all.device, dtype=torch.float32)
        for key, ref, yhat in zip(["R1","R2","R3","P4"], [self.ref1,self.ref2,self.ref3,self.ref4], yhat_list):
            ids = idx_layers[key]
            xyz = xyz_all[ids]
            # rebuild sparse graph for decoder (same coordinates mapping)
            coords, feats_cell, child_to_points = self._encode_layer(xyz, yuv_hat[ids]*0, xyz*0)  # geometry only for coords
            yref = ref(coords, feats_cell.new_zeros((feats_cell.shape[0], yhat.shape[1])).scatter_(1, torch.zeros_like(feats_cell[:, :1], dtype=torch.long), 0.0) + \
                       torch.stack([yhat[pts].mean(dim=0) for pts in child_to_points], dim=0),
                       child_to_points, n_points_layer=xyz.shape[0])
            pred = self.rec(yref)
            yuv_hat[ids] += pred
        return torch.clamp(yuv_hat, 0.0, 1.0)
