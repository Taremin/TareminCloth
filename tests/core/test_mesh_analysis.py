"""
taremin_cloth.analysis モジュールのユニットテスト
三角形交差判定、変位スパイク検出、反転面検出、OBJエクスポートの検証
"""

import os
import sys
import tempfile
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from unittest import mock
from taremin_cloth import analysis as analysis_mod
from taremin_cloth.analysis import (
    tri_tri_intersection_sat,
    tri_tri_intersection_segment,
    find_triangle_intersections,
    find_triangle_intersection_segments,
    find_mesh_collider_intersections,
    find_displacement_spikes,
    detect_inverted_faces,
    export_obj,
)


class TestMeshAnalysis(unittest.TestCase):
    def test_tri_tri_intersection_sat(self):
        """Möller / SAT による2三角形交差判定の検証"""
        # 1. 明らかに交差する2つの三角形（十字に交差）
        p0 = np.array([0.0, -1.0, 0.0])
        p1 = np.array([0.0, 1.0, 0.0])
        p2 = np.array([0.0, 0.0, 1.0])

        q0 = np.array([-1.0, 0.0, 0.5])
        q1 = np.array([1.0, 0.0, 0.5])
        q2 = np.array([0.0, 0.0, -0.5])

        self.assertTrue(tri_tri_intersection_sat(p0, p1, p2, q0, q1, q2))

        # 2. 離れている2つの三角形
        q0_far = q0 + np.array([5.0, 0.0, 0.0])
        q1_far = q1 + np.array([5.0, 0.0, 0.0])
        q2_far = q2 + np.array([5.0, 0.0, 0.0])
        self.assertFalse(tri_tri_intersection_sat(p0, p1, p2, q0_far, q1_far, q2_far))

        # 3. 平行で同一平面でない三角形
        p0_z = np.array([0.0, 0.0, 0.0])
        p1_z = np.array([1.0, 0.0, 0.0])
        p2_z = np.array([0.0, 1.0, 0.0])
        q0_z = np.array([0.0, 0.0, 1.0])
        q1_z = np.array([1.0, 0.0, 1.0])
        q2_z = np.array([0.0, 1.0, 1.0])
        self.assertFalse(tri_tri_intersection_sat(p0_z, p1_z, p2_z, q0_z, q1_z, q2_z))

        # 4. 同一平面上でエッジを共有・隣接している三角形（交差していない）
        q0_coplanar_adj = np.array([1.0, 0.0, 0.0])
        q1_coplanar_adj = np.array([1.0, 1.0, 0.0])
        q2_coplanar_adj = np.array([0.0, 1.0, 0.0])
        self.assertFalse(tri_tri_intersection_sat(p0_z, p1_z, p2_z, q0_coplanar_adj, q1_coplanar_adj, q2_coplanar_adj))

        # 5. 同一平面上で互いに重なり合っている三角形（真の交差）
        q0_coplanar_overlap = np.array([0.2, 0.2, 0.0])
        q1_coplanar_overlap = np.array([1.2, 0.2, 0.0])
        q2_coplanar_overlap = np.array([0.2, 1.2, 0.0])
        self.assertTrue(tri_tri_intersection_sat(p0_z, p1_z, p2_z, q0_coplanar_overlap, q1_coplanar_overlap, q2_coplanar_overlap))

    def test_find_triangle_intersections_in_mesh(self):
        """メッシュ内の空間ハッシュを用いた交差検出の検証"""
        # 2枚の四角形（それぞれ2三角形、計4三角形）
        # 四角形A (XY平面 z=0): [0,1,2], [0,2,3]
        # 四角形B (XZ平面 y=0, z=-0.5~0.5): [4,5,6], [4,6,7]
        verts = np.array([
            [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
            [-0.5, 0.0, -0.5], [0.5, 0.0, -0.5], [0.5, 0.0, 0.5], [-0.5, 0.0, 0.5],
        ], dtype=np.float32)

        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # 四角形Aの面 0, 1
            [4, 5, 6], [4, 6, 7],  # 四角形Bの面 2, 3
        ], dtype=np.uint32)

        intersections = find_triangle_intersections(verts, faces, ignore_adjacent=True)
        self.assertTrue(len(intersections) > 0)
        # 面 0 または 1 と、面 2 または 3 が交差していること
        for f0, f1 in intersections:
            self.assertTrue((f0 in (0, 1) and f1 in (2, 3)) or (f0 in (2, 3) and f1 in (0, 1)))

    def test_find_displacement_spikes(self):
        """変位スパイク検出の検証"""
        prev_pos = np.zeros((10, 3), dtype=np.float32)
        curr_pos = np.zeros((10, 3), dtype=np.float32)

        # 頂点 3 を 10mm (0.01m) 動かす
        curr_pos[3, 0] = 0.010
        # 頂点 7 を 2mm (0.002m) 動かす
        curr_pos[7, 1] = 0.002

        spikes = find_displacement_spikes(prev_pos, curr_pos, threshold_mm=5.0)
        self.assertEqual(len(spikes), 1)
        self.assertEqual(spikes[0]["vertex_idx"], 3)
        self.assertAlmostEqual(spikes[0]["displacement_mm"], 10.0, places=3)

    def test_detect_inverted_faces(self):
        """法線反転面の検出検証"""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)

        # 参照法線 (上向き [0, 0, 1])
        ref_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        # 正常な場合（反転なし）
        inv_faces = detect_inverted_faces(positions, faces, reference_normals=ref_normals)
        self.assertEqual(len(inv_faces), 0)

        # 頂点順序を逆（時計回り）にして法線を下向きにする
        faces_inverted = np.array([[0, 2, 1]], dtype=np.uint32)
        inv_faces = detect_inverted_faces(positions, faces_inverted, reference_normals=ref_normals)
        self.assertEqual(len(inv_faces), 1)
        self.assertEqual(inv_faces[0], 0)

    def test_export_obj(self):
        """OBJエクスポートの検証"""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)

        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = os.path.join(tmpdir, "test.obj")
            export_obj(obj_path, positions, faces)
            self.assertTrue(os.path.exists(obj_path))
            with open(obj_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("v 0.000000 0.000000 0.000000", content)
            self.assertIn("f 1 2 3", content)

    def test_small_parallel_and_moller_denominator_both_backends(self):
        """1cm微小平行離反ペアとMöller斜め交差・頂点平面上ケースをRust/NumPy双方で検証"""
        v0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        sv0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        sv1 = np.array([0.01, 0.0, 0.0], dtype=np.float32)
        sv2 = np.array([0.0, 0.01, 0.0], dtype=np.float32)
        su0 = np.array([0.0, 0.0, 0.002], dtype=np.float32)
        su1 = np.array([0.01, 0.0, 0.002], dtype=np.float32)
        su2 = np.array([0.0, 0.01, 0.002], dtype=np.float32)
        so0 = np.array([0.002, 0.002, 0.0], dtype=np.float32)
        so1 = np.array([0.012, 0.002, 0.0], dtype=np.float32)
        so2 = np.array([0.002, 0.012, 0.0], dtype=np.float32)

        u0 = np.array([1.2, 0.2, 0.5], dtype=np.float32)
        u1 = np.array([1.2, 0.25, 0.5], dtype=np.float32)
        u2 = np.array([-0.5, 0.2, -1.0], dtype=np.float32)
        nu0 = np.array([2.0, 0.2, 0.5], dtype=np.float32)
        nu1 = np.array([2.0, 0.25, 0.5], dtype=np.float32)
        nu2 = np.array([0.5, 0.2, -1.0], dtype=np.float32)
        p_in0 = np.array([0.2, 0.2, 0.0], dtype=np.float32)
        p_in1 = np.array([0.3, 0.3, 1.0], dtype=np.float32)
        p_in2 = np.array([0.4, 0.2, -1.0], dtype=np.float32)

        def assert_cases():
            self.assertFalse(tri_tri_intersection_sat(sv0, sv1, sv2, su0, su1, su2))
            self.assertTrue(tri_tri_intersection_sat(sv0, sv1, sv2, so0, so1, so2))
            self.assertTrue(tri_tri_intersection_sat(v0, v1, v2, u0, u1, u2))
            self.assertTrue(tri_tri_intersection_sat(v0, v1, v2, u0, u2, u1))
            self.assertFalse(tri_tri_intersection_sat(v0, v1, v2, nu0, nu1, nu2))
            self.assertTrue(tri_tri_intersection_sat(v0, v1, v2, p_in0, p_in1, p_in2))

        # 現在のデフォルト経路 (Rust利用可能時はRust経路)
        assert_cases()
        # Python/NumPy フォールバック経路
        with mock.patch.object(analysis_mod, "_RUST_GEOMETRY", False):
            assert_cases()

    def test_intersection_segments_and_mesh_collider(self):
        """3D交差線分抽出と布-コライダー交差判定の検証 (Rust/NumPy双方)"""
        verts = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.2, -1.0, -1.0], [0.2, 2.0, -1.0], [0.2, 0.5, 2.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
        collider_tris = np.array([
            [0.3, -1.0, -1.0, 0.3, 2.0, -1.0, 0.3, 0.5, 2.0],
        ], dtype=np.float32)

        def check_all():
            seg = tri_tri_intersection_segment(
                verts[0], verts[1], verts[2], verts[3], verts[4], verts[5]
            )
            self.assertIsNotNone(seg)
            p_a, p_b = seg
            self.assertAlmostEqual(float(p_a[0]), 0.2, places=4)
            self.assertAlmostEqual(float(p_b[0]), 0.2, places=4)
            self.assertAlmostEqual(float(p_a[2]), 0.0, places=4)
            self.assertAlmostEqual(float(p_b[2]), 0.0, places=4)

            pairs, segs = find_triangle_intersection_segments(verts, faces)
            self.assertEqual(pairs, [(0, 1)])
            self.assertEqual(segs.shape, (1, 2, 3))

            c_pairs, c_segs = find_mesh_collider_intersections(verts, faces, collider_tris)
            self.assertTrue(len(c_pairs) >= 1)
            self.assertEqual(c_segs.shape, (len(c_pairs), 2, 3))
            self.assertEqual(c_pairs[0][0], 0)

        check_all()
        with mock.patch.object(analysis_mod, "_RUST_GEOMETRY_SEGMENTS", False):
            check_all()

    def test_shallow_curved_surface_separated_not_false_positive(self):
        """滑らかな曲面上で0.01mm未満の浮きと5mmの横離間を持つ非交差ペアを誤検出しないこと"""
        # 5mm四方の近傍2面（曲率によりわずかに傾き、duは全正 +0.006〜+0.017mm、面同士は5mm離間）
        v0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        v1 = np.array([0.005, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 0.005, 0.0], dtype=np.float32)

        u0 = np.array([0.006, 0.0, 6e-6], dtype=np.float32)
        u1 = np.array([0.011, 0.0, 1.0e-5], dtype=np.float32)
        u2 = np.array([0.006, 0.005, 1.7e-5], dtype=np.float32)

        # わずかに相手平面を跨ぐ（du ≈ [-1.6e-5, +0.8e-5, -0.3e-5]）が横に5mm離れた非交差ペア
        w0 = np.array([0.006, 0.0, -1.6e-5], dtype=np.float32)
        w1 = np.array([0.011, 0.0, 8.0e-6], dtype=np.float32)
        w2 = np.array([0.006, 0.005, -3.0e-6], dtype=np.float32)

        # 斜め同一平面上で重なり合う真の交差ペア (f32丸め込み)
        inv_sqrt2 = np.float32(1.0 / np.sqrt(2.0))
        c_v0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        c_v1 = np.array([0.01, 0.0, 0.01], dtype=np.float32)
        c_v2 = np.array([0.0, 0.01, 0.0], dtype=np.float32)
        c_u0 = np.array([0.002 * inv_sqrt2, 0.002, 0.002 * inv_sqrt2], dtype=np.float32)
        c_u1 = np.array([0.012 * inv_sqrt2, 0.002, 0.012 * inv_sqrt2], dtype=np.float32)
        c_u2 = np.array([0.002 * inv_sqrt2, 0.012, 0.002 * inv_sqrt2], dtype=np.float32)

        def assert_all():
            self.assertFalse(tri_tri_intersection_sat(v0, v1, v2, u0, u1, u2))
            self.assertIsNone(tri_tri_intersection_segment(v0, v1, v2, u0, u1, u2))
            self.assertFalse(tri_tri_intersection_sat(v0, v1, v2, w0, w1, w2))
            self.assertIsNone(tri_tri_intersection_segment(v0, v1, v2, w0, w1, w2))
            self.assertTrue(tri_tri_intersection_sat(c_v0, c_v1, c_v2, c_u0, c_u1, c_u2))
            self.assertIsNotNone(tri_tri_intersection_segment(c_v0, c_v1, c_v2, c_u0, c_u1, c_u2))

        assert_all()
        with mock.patch.object(analysis_mod, "_RUST_GEOMETRY", False), \
                mock.patch.object(analysis_mod, "_RUST_GEOMETRY_SEGMENTS", False):
            assert_all()


if __name__ == "__main__":
    unittest.main()
