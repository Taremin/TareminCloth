# -*- coding: utf-8 -*-
"""geometry パッケージの単体テスト (Blender非依存・GPU不要)."""

import math
import os
import sys
import unittest
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from taremin_cloth.pattern.geometry import (
    make_segment_controls,
    compute_auto_handles,
    evaluate_cubic,
    subdivide_cubic,
    build_arc_table,
    point_at_arc_length,
    closest_point_on_table,
    compute_seam_subdivision_count,
    sample_edge_range_by_count,
    signed_area,
    ensure_ccw,
    point_in_polygon,
    has_self_intersection,
    Affine2D,
    compute_bounding_box,
)


class TestPatternGeometry(unittest.TestCase):
    """幾何・ベジェ・サンプリング・多角形判定の検証テスト."""

    def test_golden_circle_arc_length(self):
        """ゴールデンテスト: 半径1mの4区間ベジェ円の総弧長が2*pi (6.283185) と0.1%以内で一致すること."""
        # 単位円を4区間ベジェで近似する標準定数 kappa = 4/3 * (sqrt(2) - 1)
        k = 4.0 / 3.0 * (math.sqrt(2.0) - 1.0)  # 約 0.5522847

        quadrants = [
            # 第1象限: (1, 0) -> (0, 1)
            ((1.0, 0.0), (1.0, k), (k, 1.0), (0.0, 1.0)),
            # 第2象限: (0, 1) -> (-1, 0)
            ((0.0, 1.0), (-k, 1.0), (-1.0, k), (-1.0, 0.0)),
            # 第3象限: (-1, 0) -> (0, -1)
            ((-1.0, 0.0), (-1.0, -k), (-k, -1.0), (0.0, -1.0)),
            # 第4象限: (0, -1) -> (1, 0)
            ((0.0, -1.0), (k, -1.0), (1.0, -k), (1.0, 0.0)),
        ]

        total_computed = 0.0
        for c0, c1, c2, c3 in quadrants:
            table = build_arc_table(c0, c1, c2, c3, tolerance=0.00001)
            total_computed += table.total_length

        expected_perimeter = 2.0 * math.pi
        error = abs(total_computed - expected_perimeter)
        rel_error = error / expected_perimeter

        # 0.1% (0.001) 以内の誤差であること
        self.assertLess(rel_error, 0.001, f"Relative error {rel_error:.6f} exceeded 0.1%")
        self.assertAlmostEqual(total_computed, expected_perimeter, delta=0.005)

    def test_make_segment_controls_straight(self):
        """ハンドル未指定時の直線3等分制御点の構築."""
        p0 = (0.0, 0.0)
        p1 = (3.0, 6.0)
        c0, c1, c2, c3 = make_segment_controls(p0, None, None, p1)
        self.assertEqual(c0, (0.0, 0.0))
        self.assertAlmostEqual(c1[0], 1.0)
        self.assertAlmostEqual(c1[1], 2.0)
        self.assertAlmostEqual(c2[0], 2.0)
        self.assertAlmostEqual(c2[1], 4.0)
        self.assertEqual(c3, (3.0, 6.0))

        # 直線ベジェの弧長が弦長と一致すること
        table = build_arc_table(c0, c1, c2, c3)
        expected_len = math.hypot(3.0, 6.0)
        self.assertAlmostEqual(table.total_length, expected_len, places=5)

    def test_closest_point_on_table(self):
        """弧長テーブルに対する最近点探索."""
        p0 = (0.0, 0.0)
        p1 = (10.0, 0.0)
        c0, c1, c2, c3 = make_segment_controls(p0, None, None, p1)
        table = build_arc_table(c0, c1, c2, c3)

        query = (5.0, 3.0)
        best_pt, best_t, dist = closest_point_on_table(table, query)
        self.assertAlmostEqual(best_pt[0], 5.0, places=4)
        self.assertAlmostEqual(best_pt[1], 0.0, places=4)
        self.assertAlmostEqual(best_t, 0.5, places=4)
        self.assertAlmostEqual(dist, 3.0, places=4)

    def test_subdivide_cubic(self):
        """3次ベジェの中点分割."""
        c0 = (0.0, 0.0)
        c1 = (0.0, 1.0)
        c2 = (1.0, 1.0)
        c3 = (1.0, 0.0)
        left, right = subdivide_cubic(c0, c1, c2, c3, 0.5)

        # left の終点と right の始点が一致すること
        self.assertAlmostEqual(left[3][0], right[0][0])
        self.assertAlmostEqual(left[3][1], right[0][1])

        # 中点での直接評価と一致すること
        mid_eval = evaluate_cubic(c0, c1, c2, c3, 0.5)
        self.assertAlmostEqual(left[3][0], mid_eval[0])
        self.assertAlmostEqual(left[3][1], mid_eval[1])

    def test_sampling_and_seam_subdivision(self):
        """弧長サンプリングと分割数計算."""
        count = compute_seam_subdivision_count(0.30, 0.30, spacing=0.015)
        self.assertEqual(count, 20)

        # 範囲外の spacing は例外
        with self.assertRaises(ValueError):
            compute_seam_subdivision_count(0.30, 0.30, spacing=0.001)

        c0 = (0.0, 0.0)
        c3 = (1.0, 0.0)
        ctrls = make_segment_controls(c0, None, None, c3)
        tbl = build_arc_table(*ctrls)

        sampled = sample_edge_range_by_count([tbl], 4)
        self.assertEqual(len(sampled), 5)
        np.testing.assert_allclose(sampled[:, 0], [0.0, 0.25, 0.5, 0.75, 1.0], atol=1e-5)

        # 部分サンプリング [0.2, 0.8]
        sub_sampled = sample_edge_range_by_count([tbl], 2, t0=0.2, t1=0.8)
        self.assertEqual(len(sub_sampled), 3)
        np.testing.assert_allclose(sub_sampled[:, 0], [0.2, 0.5, 0.8], atol=1e-5)

    def test_polygon_signed_area_and_ccw(self):
        """符号付き面積とCCW正規化."""
        # 反時計回り正方形 (1x1)
        ccw_pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        self.assertAlmostEqual(signed_area(ccw_pts), 1.0)
        self.assertEqual(ensure_ccw(ccw_pts), ccw_pts)

        # 時計回り正方形
        cw_pts = list(reversed(ccw_pts))
        self.assertAlmostEqual(signed_area(cw_pts), -1.0)
        self.assertEqual(ensure_ccw(cw_pts), ccw_pts)

    def test_point_in_polygon(self):
        """点・多角形の内外判定."""
        poly = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        self.assertTrue(point_in_polygon((1.0, 1.0), poly))
        self.assertFalse(point_in_polygon((3.0, 1.0), poly))
        self.assertFalse(point_in_polygon((-0.5, 1.0), poly))
        # 境界線上
        self.assertTrue(point_in_polygon((0.0, 1.0), poly))
        self.assertTrue(point_in_polygon((1.0, 0.0), poly))

    def test_has_self_intersection(self):
        """自己交差判定."""
        # 正常な凸多角形
        poly_ok = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        self.assertFalse(has_self_intersection(poly_ok))

        # 蝶ネクタイ型自己交差
        poly_bow = [(0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0)]
        self.assertTrue(has_self_intersection(poly_bow))

    def test_affine2d_transform(self):
        """アフィン変換の正逆一致."""
        aff = Affine2D(tx=5.0, ty=-3.0, rotation_rad=math.pi / 2.0, sx=2.0, sy=2.0)
        pt = (1.0, 2.0)
        transformed = aff.apply(pt)
        inverted = aff.inverse_apply(transformed)
        self.assertAlmostEqual(inverted[0], pt[0], places=5)
        self.assertAlmostEqual(inverted[1], pt[1], places=5)

        # NumPy 配列適用
        arr = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32)
        arr_t = aff.apply_array(arr)
        p0_t = aff.apply(arr[0])
        p1_t = aff.apply(arr[1])
        np.testing.assert_allclose(arr_t[0], p0_t, atol=1e-5)
        np.testing.assert_allclose(arr_t[1], p1_t, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
