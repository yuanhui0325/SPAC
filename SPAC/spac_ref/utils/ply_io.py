\
import numpy as np
from plyfile import PlyData, PlyElement

def read_ply_xyzrgb(path: str):
    ply = PlyData.read(path)
    v = ply['vertex']
    x = np.asarray(v['x'], dtype=np.float32)
    y = np.asarray(v['y'], dtype=np.float32)
    z = np.asarray(v['z'], dtype=np.float32)
    cols = []
    for c in ['red','green','blue']:
        if c in v.data.dtype.names:
            cols.append(np.asarray(v[c], dtype=np.float32))
        else:
            cols.append(np.zeros_like(x))
    rgb = np.stack(cols, axis=1) / 255.0
    xyz = np.stack([x,y,z], axis=1)
    return xyz, rgb

def write_ply_xyzrgb(path: str, xyz: np.ndarray, rgb: np.ndarray):
    xyz = np.asarray(xyz).astype(np.float32)
    rgb = np.clip(np.asarray(rgb),0,1)
    r = (rgb[:,0]*255).astype(np.uint8)
    g = (rgb[:,1]*255).astype(np.uint8)
    b = (rgb[:,2]*255).astype(np.uint8)
    verts = np.empty(xyz.shape[0], dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    verts['x'],verts['y'],verts['z'] = xyz[:,0],xyz[:,1],xyz[:,2]
    verts['red'],verts['green'],verts['blue'] = r,g,b
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el], text=True).write(path)
