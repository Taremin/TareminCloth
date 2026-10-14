# -*- coding: utf-8 -*-
"""Milestone 1: Rust 2D CDT + Botsch-Kobbelt 等方性リメッシュの単体テスト (Blender非依存・GPU不要)."""
import math
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

    _HAS_REMESH_2D = hasattr(_core, "remesh_pattern_2d")
except Exception:
    _core = None
    _HAS_REMESH_2D = False


def _polygon_signed_area_2d(pts: np.ndarray) -> float:
    """Shoelace公式による2D閉ポリゴンの符号付き面積 (CCW > 0)."""
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _triangle_signed_areas_2d(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """各三角形の2D符号付き面積配列 [F] を返す."""
    p0 = verts[tris[:, 0]]
    p1 = verts[tris[:, 1]]
    p2 = verts[tris[:, 2]]
    return 0.5 * (
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    )


def _triangle_min_angles_deg(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """各三角形の最小内角 (度) 配列 [F] を返す."""
    p0 = verts[tris[:, 0]]
    p1 = verts[tris[:, 1]]
    p2 = verts[tris[:, 2]]
    a = np.linalg.norm(p2 - p1, axis=1)
    b = np.linalg.norm(p0 - p2, axis=1)
    c = np.linalg.norm(p1 - p0, axis=1)
    eps = 1e-12
    cos_a = np.clip((b * b + c * c - a * a) / np.maximum(2.0 * b * c, eps), -1.0, 1.0)
    cos_b = np.clip((a * a + c * c - b * b) / np.maximum(2.0 * a * c, eps), -1.0, 1.0)
    cos_c = np.clip((a * a + b * b - c * c) / np.maximum(2.0 * a * b, eps), -1.0, 1.0)
    ang_a = np.degrees(np.arccos(cos_a))
    ang_b = np.degrees(np.arccos(cos_b))
    ang_c = np.degrees(np.arccos(cos_c))
    return np.minimum(np.minimum(ang_a, ang_b), ang_c)


def _make_sampled_rectangle_loop(width: float, height: float, step: float) -> np.ndarray:
    """反時計回り(CCW)に等間隔サンプリングされた矩形境界ループ [B, 2] を生成する."""
    nx = max(2, int(round(width / step)))
    ny = max(2, int(round(height / step)))
    pts = []
    for i in range(nx):
        pts.append([i * (width / nx), 0.0])
    for j in range(ny):
        pts.append([width, j * (height / ny)])
    for i in range(nx):
        pts.append([width - i * (width / nx), height])
    for j in range(ny):
        pts.append([0.0, height - j * (height / ny)])
    return np.array(pts, dtype=np.float32)


def _make_concave_bodice_loop(step: float = 0.015) -> np.ndarray:
    """凹んだ襟ぐり(U字ネック)を持つ身頃型紙の反時計回り2D境界ループを生成する."""
    # 外形: 幅 0.30m, 高さ 0.40m, 上辺中央に深さ 0.08m のU字凹み
    corners = [
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([0.30, 0.0], dtype=np.float32),
        np.array([0.30, 0.40], dtype=np.float32),
        np.array([0.22, 0.40], dtype=np.float32),
    ]
    pts = []
    # 下辺・右辺・右肩
    for k in range(3):
        p0, p1 = corners[k], corners[k + 1]
        dist = float(np.linalg.norm(p1 - p0))
        n = max(2, int(round(dist / step)))
        for i in range(n):
            pts.append((p0 + (p1 - p0) * (i / n)).tolist())

    # 襟ぐり円弧: (0.22, 0.40) -> (0.15, 0.32) -> (0.08, 0.40)
    n_arc = max(6, int(round(0.22 / step)))
    for i in range(n_arc):
        theta = math.pi * (i / n_arc)  # 0 -> pi
        u = 0.15 + 0.07 * math.cos(theta)
        v = 0.40 - 0.08 * math.sin(theta)
        pts.append([u, v])

    # 左肩・左辺
    tail_corners = [
        np.array([0.08, 0.40], dtype=np.float32),
        np.array([0.0, 0.40], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    ]
    for k in range(2):
        p0, p1 = tail_corners[k], tail_corners[k + 1]
        dist = float(np.linalg.norm(p1 - p0))
        n = max(2, int(round(dist / step)))
        for i in range(n):
            pts.append((p0 + (p1 - p0) * (i / n)).tolist())

    return np.array(pts, dtype=np.float32)


@unittest.skipUnless(_HAS_REMESH_2D, "taremin_cloth_core.remesh_pattern_2d 未実装のためスキップ")
class TestPatternRemesh2D(unittest.TestCase):
    def test_boundary_vertices_preserved_verbatim(self):
        """先頭 0..B の境界頂点座標と順序が一切変更されないこと（縫合ペア1対1対応の不変条件）."""
        target_len = 0.020
        boundary = _make_sampled_rectangle_loop(0.20, 0.30, target_len)
        b_count = len(boundary)

        verts_2d, tris = _core.remesh_pattern_2d(
            np.ascontiguousarray(boundary, dtype=np.float32),
            None,
            target_edge_length=target_len,
            iterations=6,
        )
        verts_2d = np.asarray(verts_2d, dtype=np.float32)
        tris = np.asarray(tris, dtype=np.uint32)

        self.assertGreater(len(verts_2d), b_count, "内部スタイナー点が追加されていること")
        self.assertGreater(len(tris), 0)
        # 境界頂点 0..B の座標が完全一致すること
        np.testing.assert_allclose(
            verts_2d[:b_count],
            boundary,
            atol=1e-6,
            err_msg="Boundary vertices 0..B must remain strictly locked in order and position",
        )

    def test_ccw_orientation_and_area_conservation_on_concave_pattern(self):
        """凹形状（U字襟ぐり）の外部三角形が確実に除去され、全三角形がCCWかつ総面積が一致すること."""
        target_len = 0.015
        boundary = _make_concave_bodice_loop(step=target_len)
        expected_area = _polygon_signed_area_2d(boundary)
        self.assertGreater(expected_area, 0.0)

        verts_2d, tris = _core.remesh_pattern_2d(
            np.ascontiguousarray(boundary, dtype=np.float32),
            None,
            target_edge_length=target_len,
            iterations=6,
        )
        verts_2d = np.asarray(verts_2d, dtype=np.float32)
        tris = np.asarray(tris, dtype=np.uint32)

        tri_areas = _triangle_signed_areas_2d(verts_2d, tris)
        self.assertTrue(
            np.all(tri_areas > 1e-8),
            f"All triangles must have positive CCW area (min={tri_areas.min():.3e})",
        )

        total_tri_area = float(np.sum(tri_areas))
        self.assertAlmostEqual(
            total_tri_area,
            expected_area,
             delta=expected_area * 0.005,
            msg="Total triangle area must match concave polygon area (no exterior triangles in neckline cutout)",
        )

        # 襟ぐり凹部の中央空間 (0.15, 0.37) に三角形が存在しないことを確認
        centroids = (verts_2d[tris[:, 0]] + verts_2d[tris[:, 1]] + verts_2d[tris[:, 2]]) / 3.0
        dists_to_cutout = np.linalg.norm(
            centroids - np.array([0.15, 0.37], dtype=np.float32), axis=1
        )
        self.assertGreater(
            float(np.min(dists_to_cutout)),
            0.025,
            "Concave neckline region must remain empty",
        )

    def test_isotropic_quality_no_slivers_and_edge_length_convergence(self):
        """Botsch-Kobbelt等方性リメッシュにより極小スリバー三角形が排除され、辺長が目標値付近に収束すること."""
        target_len = 0.015
        boundary = _make_sampled_rectangle_loop(0.24, 0.24, target_len)

        verts_2d, tris = _core.remesh_pattern_2d(
            np.ascontiguousarray(boundary, dtype=np.float32),
            None,
            target_edge_length=target_len,
            iterations=8,
        )
        verts_2d = np.asarray(verts_2d, dtype=np.float32)
        tris = np.asarray(tris, dtype=np.uint32)

        min_angles = _triangle_min_angles_deg(verts_2d, tris)
        self.assertGreaterEqual(
            float(np.min(min_angles)),
            18.0,
            f"Sliver triangles detected: min angle {np.min(min_angles):.2f} deg < 18.0 deg",
        )
        self.assertGreaterEqual(
            float(np.percentile(min_angles, 50)),
            45.0,
            "Median minimum angle should be >= 45 deg for isotropic triangulation",
        )

        # 全エッジ長の平均が target_len の ±20% 以内に収束していること
        e01 = np.linalg.norm(verts_2d[tris[:, 1]] - verts_2d[tris[:, 0]], axis=1)
        e12 = np.linalg.norm(verts_2d[tris[:, 2]] - verts_2d[tris[:, 1]], axis=1)
        e20 = np.linalg.norm(verts_2d[tris[:, 0]] - verts_2d[tris[:, 2]], axis=1)
        mean_len = float(np.mean(np.concatenate([e01, e12, e20])))
        self.assertAlmostEqual(mean_len, target_len, delta=target_len * 0.20)

    def test_resolution_scaling_draft_vs_bake(self):
        """Draft粒度(20mm)とBake粒度(5mm)で三角形数が面積比(約16倍)に沿ってスケーリングすること."""
        loop_draft = _make_sampled_rectangle_loop(0.20, 0.20, 0.020)
        loop_bake = _make_sampled_rectangle_loop(0.20, 0.20, 0.005)

        _, tris_draft = _core.remesh_pattern_2d(loop_draft, None, target_edge_length=0.020, iterations=5)
        _, tris_bake = _core.remesh_pattern_2d(loop_bake, None, target_edge_length=0.005, iterations=5)

        ratio = len(tris_bake) / max(1, len(tris_draft))
        self.assertGreater(ratio, 10.0, f"Bake mesh should have >10x more triangles than draft (got {ratio:.1f}x)")
        self.assertLess(ratio, 24.0)


if __name__ == "__main__":
    unittest.main()
