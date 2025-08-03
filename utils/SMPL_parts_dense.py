# ==================== CODE WATERMARK ========================
# ⚠️ This source file is authored by Ming Li.
# Unauthorized reproduction, distribution, or modification
# without explicit permission is strictly prohibited.
# ============================================================

import sys
import numpy as np
from os.path import exists, join, dirname, basename, splitext
import pickle as pkl
from lib.generate_colors import generate_colors

ROOT = dirname(dirname(__file__))
sys.path.append(ROOT)
SMPL_PARTS_DENSE = join(ROOT, 'assets', 'smpl_parts_dense.pkl')
assert exists(SMPL_PARTS_DENSE)


def smpl_parts_colors(parts_np: np.ndarray) -> np.ndarray:
    parts_color = generate_colors(14)
    with open(SMPL_PARTS_DENSE, 'rb') as f:
        dat = pkl.load(f, encoding='latin-1')
    smpl_parts = np.zeros((6890, 1))
    for n, k in enumerate(dat):
        smpl_parts[dat[k]] = n
    colors = np.array([parts_color[l] for l in parts_np])
    return colors


def smpl_vertices_colors():
    parts_color = generate_colors(14)
    with open(SMPL_PARTS_DENSE, 'rb') as f:
        dat = pkl.load(f, encoding='latin-1')
    smpl_parts = np.zeros(6890, dtype=np.int32)
    for n, k in enumerate(dat):
        smpl_parts[dat[k]] = n
    smpl_parts = np.clip(smpl_parts, 0, 13)
    colors = parts_color[smpl_parts]
    return colors, smpl_parts
