import unittest
import bpy
import numpy as np
import sys
from pathlib import Path

# pythonディレクトリおよびプロジェクトルートをsys.pathに追加
root_dir = Path(__file__).parent.parent
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import taremin_cloth
from taremin_cloth.utils import topology


class TestDynamicDiagonal(unittest.TestCase):
    def setUp(self):
        taremin_cloth.register()
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def tearDown(self):
        taremin_cloth.unregister()
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def test_evaluate_quad_diagonal_by_strain_horizontal_compression(self):
        """
        水平方向の圧縮によるシワ（v0-v2対角線が縮み、v1-v3対角線が保たれる/伸びるケース）
        -> 稜線となる v1-v3 が選ばれる (戻り値 False)
        """
        # 正方形レスト (0,0), (1,0), (1,1), (0,1)
        p0_rest = np.array([0.0, 0.0, 0.0])
        p1_rest = np.array([1.0, 0.0, 0.0])
        p2_rest = np.array([1.0, 1.0, 0.0])
        p3_rest = np.array([0.0, 1.0, 0.0])

        # 変形後: X方向に圧縮され、p1とp2が左に寄る。さらにp0とp2間が縮む
        p0_curr = np.array([0.2, 0.0, 0.0])
        p1_curr = np.array([0.8, -0.2, 0.0])
        p2_curr = np.array([0.8, 0.8, 0.0])
        p3_curr = np.array([0.2, 1.0, 0.0])

        use_02 = topology.evaluate_quad_diagonal_by_strain(
            p0_rest, p1_rest, p2_rest, p3_rest,
            p0_curr, p1_curr, p2_curr, p3_curr
        )
        self.assertFalse(use_02, "歪み率が大きい（保たれている）v1-v3方向が選択されること")

    def test_evaluate_quad_diagonal_by_strain_vertical_compression(self):
        """
        垂直方向の圧縮によるシワ（v1-v3対角線が縮み、v0-v2対角線が保たれるケース）
        -> 稜線となる v0-v2 が選ばれる (戻り値 True)
        """
        p0_rest = np.array([0.0, 0.0, 0.0])
        p1_rest = np.array([1.0, 0.0, 0.0])
        p2_rest = np.array([1.0, 1.0, 0.0])
        p3_rest = np.array([0.0, 1.0, 0.0])

        # 変形後: Y方向に圧縮
        p0_curr = np.array([0.0, 0.2, 0.0])
        p1_curr = np.array([1.0, 0.5, 0.0])
        p2_curr = np.array([1.0, 0.8, 0.0])
        p3_curr = np.array([0.0, 0.5, 0.0])

        use_02 = topology.evaluate_quad_diagonal_by_strain(
            p0_rest, p1_rest, p2_rest, p3_rest,
            p0_curr, p1_curr, p2_curr, p3_curr
        )
        self.assertTrue(use_02, "歪み率が大きい（保たれている）v0-v2方向が選択されること")

    def test_apply_dynamic_diagonal_and_restore_quad(self):
        """動的対角線分割の適用とQuad復元ライフサイクル検証"""
        # 単一Quad平面 (4頂点, 1面)
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        obj = bpy.context.active_object
        self.assertIsNotNone(obj)

        self.assertEqual(len(obj.data.vertices), 4)
        self.assertEqual(len(obj.data.polygons), 1)

        # 動的対角線分割を適用
        success = topology.apply_dynamic_diagonal_triangulation(obj)
        self.assertTrue(success)

        # 頂点数は元の4頂点のまま！面数は1 -> 2 (三角形) に分割される
        self.assertEqual(len(obj.data.vertices), 4, "頂点数は増加しないこと")
        self.assertEqual(len(obj.data.polygons), 2, "1つのQuadが2つの三角形に分割されること")
        for poly in obj.data.polygons:
            self.assertEqual(len(poly.vertices), 3)

        # Quad復元 (Restore Quad) を実行
        res = topology.restore_original_quads(obj)
        self.assertTrue(res)

        # 元の4頂点、1つのQuadに戻る
        self.assertEqual(len(obj.data.vertices), 4)
        self.assertEqual(len(obj.data.polygons), 1)
        self.assertEqual(len(obj.data.polygons[0].vertices), 4)

    def test_dynamic_diagonal_operator_integration(self):
        """オペレータ経由での動的対角線分割と復元"""
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3, size=2.0)
        obj = bpy.context.active_object
        n_orig_faces = len(obj.data.polygons)
        n_orig_verts = len(obj.data.vertices)

        # オペレータ実行
        res = bpy.ops.taremin_cloth.apply_dynamic_diagonal()
        self.assertEqual(res, {'FINISHED'})

        # 頂点数は増加せず、面数は2倍（各四角面が2つの三角形に分割）
        self.assertEqual(len(obj.data.vertices), n_orig_verts, "頂点数は増加しないこと")
        self.assertEqual(len(obj.data.polygons), n_orig_faces * 2, "各Quadが2つの三角形に分割されること")

        # 復元オペレータ実行
        res_restore = bpy.ops.taremin_cloth.restore_quad_topology()
        self.assertEqual(res_restore, {'FINISHED'})

        # 元の四角面メッシュ構造に戻る
        self.assertEqual(len(obj.data.vertices), n_orig_verts)
        self.assertEqual(len(obj.data.polygons), n_orig_faces)


if __name__ == '__main__':
    unittest.main()
