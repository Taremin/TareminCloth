"""シワ影響範囲表示と骨半径解決の単体テスト (Blender不要・GPU不要)。"""
import math
import unittest

import numpy as np


class TestResolveBoneRadius(unittest.TestCase):
    def test_explicit_wins(self):
        from taremin_cloth.engine.wrinkle_field import resolve_bone_radius
        self.assertAlmostEqual(resolve_bone_radius(0.09, 0.05, 0.05, 0.4), 0.09)

    def test_head_tail_average(self):
        from taremin_cloth.engine.wrinkle_field import resolve_bone_radius
        self.assertAlmostEqual(resolve_bone_radius(0.0, 0.04, 0.06, 0.4), 0.05)

    def test_length_fallback(self):
        from taremin_cloth.engine.wrinkle_field import resolve_bone_radius
        self.assertAlmostEqual(resolve_bone_radius(0.0, 0.0, 0.0, 0.4), 0.06)


class TestDeriveInfluenceSpans(unittest.TestCase):
    def test_full_circle_reports_closed(self):
        from taremin_cloth.engine.wrinkle_field import derive_wrinkle_influence_spans
        th = np.linspace(0.0, 2.0 * math.pi, 48)
        pts = np.stack([0.06 * np.cos(th), 0.06 * np.sin(th), np.full_like(th, 0.05)], axis=-1)
        span = derive_wrinkle_influence_spans(pts, [0, 0, 0], [0, 0, 1], [1, 0, 0])
        self.assertIsNotNone(span)
        self.assertTrue(span["is_closed"])
        self.assertAlmostEqual(span["theta_max"] - span["theta_min"], 2.0 * math.pi)
        self.assertAlmostEqual(span["target_r"], 0.06, delta=1e-6)

    def test_partial_arc_span(self):
        from taremin_cloth.engine.wrinkle_field import derive_wrinkle_influence_spans
        th = np.linspace(math.pi * 0.6, math.pi * 1.4, 20)
        pts = np.stack([0.06 * np.cos(th), 0.06 * np.sin(th), np.full_like(th, 0.05)], axis=-1)
        span = derive_wrinkle_influence_spans(pts, [0, 0, 0], [0, 0, 1], [1, 0, 0])
        self.assertIsNotNone(span)
        self.assertFalse(span["is_closed"])
        self.assertGreater(span["theta_max"] - span["theta_min"], 1.0)

    def test_explicit_target_radius(self):
        from taremin_cloth.engine.wrinkle_field import derive_wrinkle_influence_spans
        th = np.linspace(0.0, 2.0 * math.pi, 24)
        pts = np.stack([0.06 * np.cos(th), 0.06 * np.sin(th), np.full_like(th, 0.05)], axis=-1)
        span = derive_wrinkle_influence_spans(
            pts, [0, 0, 0], [0, 0, 1], [1, 0, 0], explicit_target_radius=0.09)
        self.assertAlmostEqual(span["target_r"], 0.09)

    def test_too_few_points_returns_none(self):
        from taremin_cloth.engine.wrinkle_field import derive_wrinkle_influence_spans
        self.assertIsNone(derive_wrinkle_influence_spans(
            np.zeros((1, 3)), [0, 0, 0], [0, 0, 1]))


class TestInfluenceMesh(unittest.TestCase):
    def test_valley_shell_bounds(self):
        from taremin_cloth.utils.drawing import compute_wrinkle_influence_mesh
        tris, lines = compute_wrinkle_influence_mesh(
            [0, 0, 0], [0, 0, 1], [1, 0, 0],
            0.0, 2.0 * math.pi, 0.0, 0.1, 0.06, 0.0, 0.015,
            n_theta=16, n_z=2,
        )
        self.assertGreater(len(tris), 0)
        radii = [float(np.linalg.norm(np.array(p[:2]))) for p in tris]
        self.assertAlmostEqual(min(radii), 0.06, delta=1e-6)
        self.assertAlmostEqual(max(radii), 0.075, delta=1e-6)

    def test_crest_shell_bounds(self):
        from taremin_cloth.utils.drawing import compute_wrinkle_influence_mesh
        tris, _ = compute_wrinkle_influence_mesh(
            [0, 0, 0], [0, 0, 1], [1, 0, 0],
            0.0, math.pi, 0.0, 0.1, 0.07, 0.020, 0.0,
            n_theta=8, n_z=1,
        )
        radii = [float(np.linalg.norm(np.array(p[:2]))) for p in tris]
        self.assertAlmostEqual(min(radii), 0.05, delta=1e-6)
        self.assertAlmostEqual(max(radii), 0.07, delta=1e-6)

    def test_build_meshes_end_to_end(self):
        from taremin_cloth.utils.drawing import build_wrinkle_influence_meshes
        th = np.linspace(0.0, 2.0 * math.pi, 32)
        pts = [[float(0.06 * np.cos(t)), float(0.06 * np.sin(t)), 0.05] for t in th]
        meshes = build_wrinkle_influence_meshes(
            [("root", pts, 0.03, 0.0)],
            [0, 0, 0], [0, 0, 1], [1, 0, 0], 0.05,
        )
        self.assertEqual(len(meshes), 1)
        self.assertEqual(meshes[0][0], "root")
        self.assertGreater(len(meshes[0][1]), 0)

    def test_mesh_follows_build_frame(self):
        """影響範囲の長軸は構築座標系の骨軸に一致すること。
        固定ボーン座標系での描画が屈曲時に垂直化した回帰検出用。"""
        from taremin_cloth.engine.wrinkle_field import derive_wrinkle_influence_spans
        from taremin_cloth.utils.drawing import compute_wrinkle_influence_mesh
        # 90度屈曲した先のボーンB周りのリング
        th = np.linspace(0.0, 2.0 * math.pi, 32)
        pts = np.stack(
            [[0.4, 0.06 * math.cos(t), 0.3 + 0.06 * math.sin(t)] for t in th]
        ).astype(np.float32)
        span = derive_wrinkle_influence_spans(
            pts, [0.3, 0, 0.3], [1, 0, 0], [0, 0, 1])
        self.assertTrue(span["is_closed"])
        tris, _ = compute_wrinkle_influence_mesh(
            [0.3, 0, 0.3], [1, 0, 0], [0, 0, 1],
            span["theta_min"], span["theta_max"],
            span["z_min"], span["z_max"], span["target_r"], 0.0, 0.015,
            n_theta=16, n_z=1,
        )
        p = np.array(tris).reshape(-1, 3)
        _, _, vt = np.linalg.svd(p - p.mean(0), full_matrices=False)
        long_axis = vt[np.argmin(np.linalg.svd(p - p.mean(0), compute_uv=False))]
        self.assertGreater(abs(float(np.dot(long_axis, [1, 0, 0]))), 0.99)
        # 旧ボーン座標系では帯が伸びて不一致になることの記録
        stale = derive_wrinkle_influence_spans(pts, [0, 0, 0], [0, 0, 1], [1, 0, 0])
        self.assertGreater(stale["z_max"] - stale["z_min"],
                           span["z_max"] - span["z_min"])


if __name__ == "__main__":
    unittest.main()
