
import numpy as np

class OctreeNode:
    __slots__ = ("points_idx", "bbox_min", "bbox_max", "children", "depth")
    def __init__(self, points_idx, bbox_min, bbox_max, depth):
        self.points_idx = np.asarray(points_idx, dtype=np.int64)
        self.bbox_min = np.asarray(bbox_min, dtype=np.float32)
        self.bbox_max = np.asarray(bbox_max, dtype=np.float32)
        self.children = []
        self.depth = depth

def _bbox(points):
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    mx = np.maximum(mx, mn + 1e-6)
    return mn, mx

def build_octree(xyz: np.ndarray, max_leaf_points: int = 8, max_depth: int = 8):
    mn, mx = _bbox(xyz)
    root = OctreeNode(np.arange(len(xyz)), mn, mx, 0)
    stack = [root]
    while stack:
        node = stack.pop()
        if node.depth >= max_depth or len(node.points_idx) <= max_leaf_points:
            continue
        mid = 0.5 * (node.bbox_min + node.bbox_max)
        pts = xyz[node.points_idx]
        codes = (pts > mid).astype(np.int32)
        child_codes = codes[:,0]*4 + codes[:,1]*2 + codes[:,2]*1
        for c in range(8):
            idx_local = np.where(child_codes == c)[0]
            if idx_local.size == 0:
                continue
            sel = node.points_idx[idx_local]
            cmin = node.bbox_min.copy()
            cmax = node.bbox_max.copy()
            # Decode bits to locate child bbox
            bits = [(c>>2)&1, (c>>1)&1, c&1]
            for d in range(3):
                if bits[d] == 0:
                    cmax[d] = mid[d]
                else:
                    cmin[d] = mid[d]
            child = OctreeNode(sel, cmin, cmax, node.depth+1)
            node.children.append(child)
            stack.append(child)
    return root

def collect_second_to_last_level(root: OctreeNode):
    out = []
    stack = [root]
    while stack:
        node = stack.pop()
        if not node.children:
            continue
        if all(len(ch.children)==0 for ch in node.children):
            out.append(node)
        else:
            stack.extend(node.children)
    return out
