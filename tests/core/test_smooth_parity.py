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


@unittest.skipUnless(_RUST_OK, "Rust brush API未ビルドのためスキップ")
class TestBrushMathParity(unittest.TestCase):
    """math.py (NumPyフォールバック) と Rust核のparity検証"""

    def _numpy(self, fn, *args, **kwargs):
        from unittest import mock

        with mock.patch.object(brush_math, "_RUST_MATH", False):
            return fn(*args, **kwargs)

    def test_falloff_all_shapes(self):
        rng = np.random.default_rng(11)
        d = (rng.random(30).astype(np.float32) * 0.2).tolist()
        for shape, sid in (('SMOOTH', 0), ('SPHERE', 1), ('SHARP', 2),
                           ('LINEAR', 3), ('CONSTANT', 4)):
            for radius in (0.05, 0.1):
                w_py = self._numpy(brush_math.falloff_weights, d, radius, shape)
                w_rs = np.asarray(
                    _core.brush_falloff_weights(
                        np.ascontiguousarray(d, dtype=np.float32), radius, sid),
                    dtype=np.float32)
                np.testing.assert_allclose(w_rs, w_py, atol=1e-5)

    def test_verts_in_brush(self):
        for seed in range(3):
            pos, _, _, _, _ = _random_case(seed)
            center = pos[3].tolist()
            for shape, sid in (('SMOOTH', 0), ('SPHERE', 1), ('CONSTANT', 4)):
                f_py, w_py = self._numpy(brush_math.verts_in_brush, pos, center, 0.08, shape)
                f_rs, w_rs = _core.brush_verts_in_brush(
                    np.ascontiguousarray(pos), center, 0.08, sid)
                self.assertEqual(list(np.asarray(f_rs)), list(f_py.tolist()))
                np.testing.assert_allclose(np.asarray(w_rs), w_py, atol=1e-5)

    def test_depth_keep_mask(self):
        rng = np.random.default_rng(12)
        pts = (rng.random((25, 3)).astype(np.float32) - 0.5).tolist()
        o, d = [0.0, 0.0, 5.0], [0.0, 0.0, -1.0]
        for hit_t, eps in ((4.0, 0.1), (None, 0.1), (5.0, 0.0)):
            m_py = self._numpy(brush_math.depth_keep_mask, pts, o, d, hit_t, eps)
            m_rs = np.asarray(
                _core.brush_depth_keep_mask(
                    np.ascontiguousarray(pts, dtype=np.float32), o, d, hit_t, eps),
                dtype=bool)
            self.assertEqual(m_rs.tolist(), np.asarray(m_py, dtype=bool).tolist())

    def test_select_grab_pins(self):
        rng = np.random.default_rng(13)
        for trial in range(5):
            n = int(rng.integers(0, 20))
            found = list(range(n))
            w = rng.random(n).astype(np.float32).tolist()
            th = float(rng.random() * 0.1)
            i_py, w_py = self._numpy(brush_math.select_grab_pins, found, w, th)
            i_rs, w_rs = _core.brush_select_grab_pins(found, w, th)
            self.assertEqual(list(i_rs), i_py)
            np.testing.assert_allclose(np.asarray(w_rs), np.asarray(w_py), atol=1e-6)
        # 長さ不一致はNumPy側のzip切詰め仕様を維持 (Rustは未使用)
        i_py, _ = self._numpy(brush_math.select_grab_pins, [0, 1], [0.5], 0.01)
        self.assertEqual(i_py, [0])

    def test_radial_adjust(self):
        for args in ((0.05, 200.0, 0.005, 0.5), (0.05, -100.0, 0.005, 0.5),
                     (0.5, 10000.0, 0.005, 0.5), (0.005, -10000.0, 0.005, 0.5),
                     (0.08, 37.0, 0.005, 0.5)):
            v_py = self._numpy(brush_math.radial_adjust, *args)
            v_rs = _core.brush_radial_adjust(*args)
            self.assertAlmostEqual(v_rs, v_py, places=9)

    def test_grab_drag_targets(self):
        rng = np.random.default_rng(14)
        for trial in range(5):
            n = int(rng.integers(1, 15))
            init = (rng.random((n, 3)).astype(np.float32) - 0.5).tolist()
            delta = (rng.random(3).astype(np.float32) - 0.5).tolist()
            w = rng.random(n).astype(np.float32).tolist()
            s = float(rng.random())
            t_py = self._numpy(brush_math.grab_drag_targets, init, delta, w, s)
            t_rs = np.asarray(
                _core.brush_grab_drag_targets(
                    np.ascontiguousarray(init, dtype=np.float32), delta, w, s),
                dtype=np.float32)
            np.testing.assert_allclose(t_rs, t_py, atol=1e-5)
        with self.assertRaises(ValueError):
            self._numpy(brush_math.grab_drag_targets, [[0, 0, 0]], [0, 0, 0], [1.0, 0.5], 1.0)
        with self.assertRaises(ValueError):
            _core.brush_grab_drag_targets(
                np.zeros((1, 3), dtype=np.float32), [0, 0, 0], [1.0, 0.5], 1.0)

    def test_ray_plane_hit(self):
        cases = [
            ((0, 0, 5), (0, 0, -1), (0, 0, 0), (0, 0, 1), [0, 0, 0]),
            ((1, 2, 3), (0, 1, 0), (0, 0, 0), (0, 1, 0), [1, 0, 3]),
            ((0, 0, 5), (1, 0, 0), (0, 0, 0), (0, 0, 1), None),
        ]
        for o, d, pp, pn, _ in cases:
            h_py = self._numpy(brush_math.ray_plane_hit, o, d, pp, pn)
            h_rs = _core.brush_ray_plane_hit(o, d, pp, pn)
            if h_py is None:
                self.assertIsNone(h_rs)
            else:
                np.testing.assert_allclose(np.asarray(h_rs), np.asarray(h_py), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
