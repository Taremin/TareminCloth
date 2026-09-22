"""Rust核とPythonフォールバックのparity検証 (Blender非依存・GPU不要)"""
import os
import sys
import unittest

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from taremin_cloth.brush import math as brush_math

try:
    import taremin_cloth_core as _core

    _RUST_OK = all(
        hasattr(_core, n)
        for n in (
            "brush_build_adjacency_csr",
            "brush_smooth_targets",
            "brush_radial_expand_targets",
        )
    )
except Exception:
    _core = None
    _RUST_OK = False


def _random_case(seed, n=40, e_extra=30, b=12):
    rng = np.random.default_rng(seed)
    pos = (rng.random((n, 3)).astype(np.float32) - 0.5) * 0.2
    # グリッド辺 + ランダム辺の混合グラフ
    side = int(np.ceil(np.sqrt(n)))
    edges = set()
    for i in range(n):
        if (i + 1) % side != 0 and i + 1 < n:
            edges.add((min(i, i + 1), max(i, i + 1)))
        if i + side < n:
            edges.add((min(i, i + side), max(i, i + side)))
    while len(edges) < n + e_extra:
        a, c = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != c:
            edges.add((min(a, c), max(a, c)))
    edges = np.array(sorted(edges), dtype=np.uint32)
    brush_idx = np.array(sorted(rng.choice(n, size=b, replace=False)), dtype=np.uint32)
    weights = rng.random(b).astype(np.float32)
    strength = float(rng.random())
    return pos, edges, brush_idx, weights, strength


@unittest.skipUnless(_RUST_OK, "Rust brush API未ビルドのためスキップ")
class TestSmoothParity(unittest.TestCase):
    def test_csr_parity(self):
        for seed in range(5):
            pos, edges, _, _, _ = _random_case(seed)
            n = len(pos)
            off_py, idx_py = brush_math.build_adjacency_csr(n, edges)
            off_rs, idx_rs = _core.brush_build_adjacency_csr(n, np.ascontiguousarray(edges))
            self.assertEqual(list(np.asarray(off_rs)), list(off_py.tolist()))
            self.assertEqual(list(np.asarray(idx_rs)), list(idx_py.tolist()))

    def test_smooth_parity(self):
        for seed in range(5):
            pos, edges, b_idx, w, s = _random_case(seed)
            n = len(pos)
            off_py, idx_py = brush_math.build_adjacency_csr(n, edges)
            excl = [int(b_idx[0]), int(b_idx[-1])]
            tgt_py = brush_math.laplacian_smooth_targets(
                pos, off_py, idx_py, b_idx, w, s, exclude=excl)
            tgt_rs = np.asarray(
                _core.brush_smooth_targets(
                    np.ascontiguousarray(pos), off_py, idx_py,
                    np.ascontiguousarray(b_idx), np.ascontiguousarray(w),
                    s, excl),
                dtype=np.float32)
            np.testing.assert_allclose(tgt_rs, tgt_py, atol=1e-4)

    def test_expand_parity(self):
        for seed in range(5):
            pos, _, b_idx, w, s = _random_case(seed)
            center = pos[int(b_idx[len(b_idx) // 2])]
            tgt_py = brush_math.radial_expand_targets(pos, center, b_idx, w, s)
            tgt_rs = np.asarray(
                _core.brush_radial_expand_targets(
                    np.ascontiguousarray(pos), center,
                    np.ascontiguousarray(b_idx), np.ascontiguousarray(w), s),
                dtype=np.float32)
            np.testing.assert_allclose(tgt_rs, tgt_py, atol=1e-4)

    def test_error_parity(self):
        pos = np.zeros((3, 3), dtype=np.float32)
        off, idx = brush_math.build_adjacency_csr(3, np.empty((0, 2), dtype=np.uint32))
        with self.assertRaises(ValueError):
            brush_math.laplacian_smooth_targets(pos, off, idx, [0], [1.0, 0.5], 1.0)
        with self.assertRaises(ValueError):
            _core.brush_smooth_targets(pos, off, idx,
                                       np.array([0], dtype=np.uint32),
                                       np.array([1.0, 0.5], dtype=np.float32), 1.0, [])


if __name__ == "__main__":
    unittest.main()
