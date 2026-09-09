"""
taremin_cloth.analysis モジュールのユニットテスト
三角形交差判定、変位スパイク検出、反転面検出、OBJエクスポートの検証
"""

import os
import sys
import tempfile
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)

from taremin_cloth.analysis import (
    tri_tri_intersection_sat,
    find_triangle_intersections,
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


if __name__ == "__main__":
    unittest.main()
