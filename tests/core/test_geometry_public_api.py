"""Rust公開幾何APIとPython実装のparity検証 (Blender非依存・GPU不要)"""
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

try:
    import taremin_cloth_core as _core

    _RUST_OK = all(
        hasattr(_core, n)
        for n in (
            "compute_areal_inv_masses",
            "color_edge_pairs",
            "tri_tri_intersect",
            "find_triangle_intersections",
            "find_proximity_violations",
            "audit_pairs",
        )
    )
except Exception:
    _core = None
    _RUST_OK = False


def _numpy_areal_reference(positions, faces, pin_weights, areal_density=0.15):
    """Rust化前のNumPy実装と同一式のリファレンス (フォールバック相当)。"""
    n_verts = len(positions)
    masses = np.zeros(n_verts, dtype=np.float64)
    if faces is not None and len(faces) > 0 and areal_density > 0.0:
        tris = np.asarray(faces, dtype=np.int64)
        valid = (
            (tris[:, 0] != tris[:, 1])
            & (tris[:, 1] != tris[:, 2])
            & (tris[:, 2] != tris[:, 0])
            & (tris >= 0).all(axis=1)
            & (tris < n_verts).all(axis=1)
        )
        tris = tris[valid]
        if len(tris) > 0:
            p0 = np.asarray(positions, dtype=np.float64)[tris[:, 0]]
            p1 = np.asarray(positions, dtype=np.float64)[tris[:, 1]]
            p2 = np.asarray(positions, dtype=np.float64)[tris[:, 2]]
            areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
            tri_mass = areas * float(areal_density) / 3.0
            np.add.at(masses, tris[:, 0], tri_mass)
            np.add.at(masses, tris[:, 1], tri_mass)
            np.add.at(masses, tris[:, 2], tri_mass)
    masses = np.maximum(masses, 1e-9)
    inv = (1.0 / masses).astype(np.float32)
    if pin_weights is not None and len(pin_weights) == n_verts:
        w = np.clip(np.asarray(pin_weights, dtype=np.float32), 0.0, 1.0)
        inv = (inv * (1.0 - w)).astype(np.float32)
    return inv


def _python_coloring_reference(edges_arr, num_vertices):
    """log_tools旧実装と同一式のリファレンス。"""
    n = len(edges_arr)
    adj = [[] for _ in range(num_vertices)]
    for idx, (v0, v1) in enumerate(edges_arr):
        if v0 < num_vertices:
            adj[v0].append(idx)
        if v1 < num_vertices:
            adj[v1].append(idx)
    order = list(range(n))
    order.sort(
        key=lambda i: len(adj[edges_arr[i][0]]) + len(adj[edges_arr[i][1]]),
        reverse=True,
    )
    colors = [-1] * n
    for i in order:
        v0, v1 = (int(edges_arr[i][0]), int(edges_arr[i][1]))
        used = set()
        for adj_i in adj[v0]:
            if adj_i != i and colors[adj_i] != -1:
                used.add(colors[adj_i])
        for adj_i in adj[v1]:
            if adj_i != i and colors[adj_i] != -1:
                used.add(colors[adj_i])
        c = 0
        while c in used:
            c += 1
        colors[i] = c
    return colors


@unittest.skipUnless(_RUST_OK, "Rust公開幾何API未ビルドのためスキップ")
class TestGeometryPublicApi(unittest.TestCase):
    def test_areal_parity_random(self):
        rng = np.random.default_rng(0)
        for seed in range(5):
            rng = np.random.default_rng(seed)
            n = 40
            pos = (rng.random((n, 3)).astype(np.float32) - 0.5).astype(np.float32)
            faces = []
            side = 6
            for y in range(side - 1):
                for x in range(side - 1):
                    a = y * side + x
                    b = a + 1
                    c = a + side
                    d = c + 1
                    faces.append([a, b, c])
                    faces.append([b, d, c])
            faces = np.array(faces, dtype=np.uint32)
            pins = (rng.random(n) < 0.1).astype(np.float32)
            got = np.asarray(
                _core.compute_areal_inv_masses(
                    np.ascontiguousarray(pos, dtype=np.float32),
                    np.ascontiguousarray(faces, dtype=np.uint32),
                    np.ascontiguousarray(pins, dtype=np.float32),
                    0.15,
                ),
                dtype=np.float32,
            )
            want = _numpy_areal_reference(pos, faces, pins, 0.15)
            self.assertEqual(got.shape, want.shape)
            np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)

    def test_areal_edge_cases(self):
        pos = np.zeros((3, 3), dtype=np.float32)
        # 面なしでも有限値を返す
        got = np.asarray(
            _core.compute_areal_inv_masses(pos, None, None, 0.15), dtype=np.float32
        )
        self.assertEqual(got.shape, (3,))
        self.assertTrue(np.all(np.isfinite(got)))
        # 完全固定ピンは0.0
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        pins = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        got = np.asarray(
            _core.compute_areal_inv_masses(pos, faces, pins, 0.15), dtype=np.float32
        )
        self.assertAlmostEqual(float(got[0]), 0.0, places=6)

    def test_coloring_parity_and_validity(self):
        rng = np.random.default_rng(1)
        for seed in range(5):
            n = 30
            m = 60
            a = rng.integers(0, n, size=m)
            b = rng.integers(0, n, size=m)
            mask = a != b
            a, b = a[mask], b[mask]
            edges = np.stack(
                [np.minimum(a, b), np.maximum(a, b)], axis=1
            ).astype(np.uint32)
            got = list(_core.color_edge_pairs(int(n), np.ascontiguousarray(edges)))
            want = _python_coloring_reference(edges, int(n))
            self.assertEqual(got, want)
            # 妥当性: 頂点共有拘束は異色
            vert_to = {}
            for idx, (v0, v1) in enumerate(edges.tolist()):
                vert_to.setdefault(int(v0), []).append(idx)
                vert_to.setdefault(int(v1), []).append(idx)
            for lst in vert_to.values():
                seen = [got[i] for i in lst]
                self.assertEqual(len(seen), len(set(seen)))

    def test_coloring_empty(self):
        got = list(
            _core.color_edge_pairs(10, np.empty((0, 2), dtype=np.uint32))
        )
        self.assertEqual(got, [])


def _grid_mesh(side=6, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = side * side
    pos = np.zeros((n, 3), dtype=np.float32)
    for y in range(side):
        for x in range(side):
            pos[y * side + x] = [x * 0.05, y * 0.05, 0.0]
    if noise > 0.0:
        pos += (rng.random(pos.shape).astype(np.float32) - 0.5) * noise
    faces = []
    for y in range(side - 1):
        for x in range(side - 1):
            a = y * side + x
            b = a + 1
            c = a + side
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    return pos, np.array(faces, dtype=np.uint32)


@unittest.skipUnless(_RUST_OK, "Rust公開幾何API未ビルドのためスキップ")
class TestIntersectionParity(unittest.TestCase):
    def test_single_pair_parity(self):
        from taremin_cloth import analysis as _an

        rng = np.random.default_rng(3)
        for _ in range(20):
            tris = (rng.random((2, 3, 3)).astype(np.float64) - 0.5).astype(np.float32)
            old = _an._RUST_GEOMETRY
            try:
                _an._RUST_GEOMETRY = False
                want = _an.tri_tri_intersection_moller(
                    tris[0][0], tris[0][1], tris[0][2],
                    tris[1][0], tris[1][1], tris[1][2],
                )
            finally:
                _an._RUST_GEOMETRY = old
            got = bool(
                _core.tri_tri_intersect(
                    tris[0][0].tolist(), tris[0][1].tolist(), tris[0][2].tolist(),
                    tris[1][0].tolist(), tris[1][1].tolist(), tris[1][2].tolist(),
                )
            )
            self.assertEqual(got, want)

    def test_intersections_parity_noisy(self):
        from taremin_cloth import analysis as _an

        for seed in range(4):
            pos, faces = _grid_mesh(noise=0.03, seed=seed)
            old = _an._RUST_GEOMETRY
            try:
                _an._RUST_GEOMETRY = False
                want = _an.find_triangle_intersections(pos, faces)
            finally:
                _an._RUST_GEOMETRY = old
            got = _an.find_triangle_intersections(pos, faces)
            self.assertEqual(got, want)

    def test_intersections_known(self):
        from taremin_cloth import analysis as _an

        # 貫通する2枚 (各1三角形)
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.2, -1.0, -1.0],
                [0.2, 2.0, -1.0],
                [0.2, 0.5, 2.0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
        self.assertEqual(_an.find_triangle_intersections(pos, faces), [(0, 1)])
        # 離間時は空
        pos2 = pos.copy()
        pos2[3:, 2] += 10.0
        self.assertEqual(_an.find_triangle_intersections(pos2, faces), [])
        # 辺共有ペアは除外/検出の切替
        pos3 = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        faces3 = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
        self.assertEqual(_an.find_triangle_intersections(pos3, faces3, True), [])
        old = _an._RUST_GEOMETRY
        try:
            _an._RUST_GEOMETRY = False
            want = _an.find_triangle_intersections(pos3, faces3, False)
        finally:
            _an._RUST_GEOMETRY = old
        got = _an.find_triangle_intersections(pos3, faces3, False)
        self.assertEqual(got, want)

    def test_proximity_parity(self):
        from taremin_cloth import analysis as _an

        for seed in range(3):
            pos, faces = _grid_mesh(noise=0.004, seed=seed)
            old = _an._RUST_GEOMETRY
            try:
                _an._RUST_GEOMETRY = False
                want = _an.find_proximity_violations(pos, faces, 0.005, True)
            finally:
                _an._RUST_GEOMETRY = old
            got = _an.find_proximity_violations(pos, faces, 0.005, True)
            self.assertEqual(len(got), len(want))
            for (gv, gf, gd), (wv, wf, wd) in zip(got, want):
                self.assertEqual((gv, gf), (wv, wf))
                self.assertAlmostEqual(gd, wd, places=6)


def _two_layer_case(dz=0.002):
    """近接2層クアッド。A層(z=0)とB層(z=dz)は辺を共有しない。"""
    a = np.array(
        [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.05, 0.05, 0.0]],
        dtype=np.float64,
    )
    b = a + np.array([0.01, 0.01, dz], dtype=np.float64)
    pos = np.vstack([a, b])
    faces = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6], [5, 7, 6]], dtype=np.uint32)
    edges = np.array(
        [[0, 1], [1, 2], [2, 0], [1, 3], [3, 2],
         [4, 5], [5, 6], [6, 4], [5, 7], [7, 6]],
        dtype=np.uint32,
    )
    return pos, faces, edges


@unittest.skipUnless(_RUST_OK, "Rust公開幾何API未ビルドのためスキップ")
class TestAuditParity(unittest.TestCase):
    def _both(self, *args, **kw):
        from taremin_cloth import pair_audit as _pa

        old = _pa._RUST_AUDIT
        try:
            _pa._RUST_AUDIT = False
            want = _pa.audit_frame(*args, **kw)
        finally:
            _pa._RUST_AUDIT = old
        got = _pa.audit_frame(*args, **kw)
        return got, want

    def _assert_rep_equal(self, got, want):
        self.assertEqual(set(got.keys()), set(want.keys()))
        for k in want:
            if isinstance(want[k], float):
                self.assertAlmostEqual(got[k], want[k], places=9, msg=k)
            else:
                self.assertEqual(got[k], want[k], msg=k)

    def test_empty_pairs(self):
        pos, faces, edges = _two_layer_case()
        kw = dict(
            vt_pairs=np.empty((0, 4), dtype=np.uint32),
            ee_pairs=np.empty((0, 4), dtype=np.uint32),
            vt_count=0, ee_count=0, faces=faces, edges=edges,
            rest_lengths=None, thickness=0.005, safety_margin=0.005,
            horizon_scale=1.3, max_horizon=0.02, margin_mode=1,
            dt_frame=1.0 / 60.0, max_vt_pairs=65536, max_ee_pairs=65536,
        )
        got, want = self._both(pos, None, pos, **kw)
        self._assert_rep_equal(got, want)
        self.assertGreater(want["vt_ground"], 0)
        self.assertGreater(want["ee_ground"], 0)

    def test_cached_hit_and_stale(self):
        pos, faces, edges = _two_layer_case()
        prev = pos.copy()
        prev[4:, 2] -= 0.001  # B層が接近中
        vel = np.zeros_like(pos)
        vel[4:, 2] = -0.06  # -60mm/s
        # 実在ペアをキャッシュ: B層頂点4とA層面0、辺ペア(0,5)
        vt = np.array([[4, 0, 1, 2], [5, 1, 3, 2]], dtype=np.uint32)
        ee = np.array([[0, 1, 4, 5]], dtype=np.uint32)
        kw = dict(
            vt_pairs=vt, ee_pairs=ee, vt_count=2, ee_count=1,
            faces=faces, edges=edges, rest_lengths=None, thickness=0.005,
            safety_margin=0.005, horizon_scale=1.3, max_horizon=0.02,
            margin_mode=1, dt_frame=1.0 / 60.0, max_vt_pairs=65536,
            max_ee_pairs=65536,
        )
        got, want = self._both(prev, vel, pos, **kw)
        self._assert_rep_equal(got, want)
        self.assertGreaterEqual(want["vt_hit"], 1)

    def test_margin_fixed_and_dropped(self):
        pos, faces, edges = _two_layer_case(dz=0.05)  # 遠方層
        kw = dict(
            vt_pairs=np.empty((0, 4), dtype=np.uint32),
            ee_pairs=np.empty((0, 4), dtype=np.uint32),
            vt_count=10, ee_count=7, faces=faces, edges=edges,
            rest_lengths=None, thickness=0.005, safety_margin=0.005,
            horizon_scale=1.3, max_horizon=0.02, margin_mode=0,
            dt_frame=1.0 / 60.0, max_vt_pairs=4, max_ee_pairs=3,
        )
        got, want = self._both(pos, None, pos, **kw)
        self._assert_rep_equal(got, want)
        self.assertEqual(want["vt_dropped"], 10)
        self.assertEqual(want["ee_dropped"], 7)


if __name__ == "__main__":
    unittest.main()
