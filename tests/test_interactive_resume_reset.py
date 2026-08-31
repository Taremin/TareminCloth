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
from taremin_cloth import operators
from taremin_cloth.utils import topology


class TestInteractiveResumeReset(unittest.TestCase):
    def setUp(self):
        taremin_cloth.register()
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def tearDown(self):
        taremin_cloth.unregister()
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def test_interactive_resume_and_reset_restores_initial_shape(self):
        """複数回のインタラクティブ変形後にResetを実行すると、真の初期レスト形状に完全復元されるかの検証"""
        bpy.ops.mesh.primitive_plane_add(size=2.0)
        obj = bpy.context.active_object
        self.assertIsNotNone(obj)

        # Taremin Cloth を有効化
        obj.taremin_cloth.is_cloth = True
        obj.taremin_cloth.enabled = True

        # 初期頂点座標を記録
        n_verts = len(obj.data.vertices)
        orig_coords = np.empty(n_verts * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", orig_coords)

        # 1回目のインタラクティブ開始（シミュレータ生成）
        sim1, coords1 = operators.get_or_create_simulator(obj)
        self.assertIsNotNone(sim1)
        self.assertTrue("_taremin_rest_positions" in obj)

        # 初回のエッジ0の自然長を記録
        orig_edge0_rest_len = sim1.get_edge_initial_rest_length(0)
        self.assertGreater(orig_edge0_rest_len, 0.0)

        # 頂点を下方に引っ張って変形（シミュレーション進行を模擬）
        deformed_coords = orig_coords.copy()
        deformed_coords[2] -= 1.0  # 頂点0のZを下げる
        deformed_coords[5] -= 1.5  # 頂点1のZを下げる
        obj.data.vertices.foreach_set("co", deformed_coords)
        obj.data.update()
        obj["_taremin_is_deformed"] = True

        # 停止 (Paused: シミュレータは保持されるか、破棄されてもCold Resumeされる)
        # ここでシミュレータを再生成する状況（Cold Resume）をテストするため、あえてキャッシュをクリア
        operators.clear_simulator_for_object(obj.name)

        # 2回目のインタラクティブ開始（Cold Resumeの検証）
        sim2, coords2 = operators.get_or_create_simulator(obj)
        self.assertIsNotNone(sim2)

        # 2回目のシミュレータでも、エッジ0の初期自然長（レスト長）が変形前の長さと完全に一致することを検証！
        new_edge0_rest_len = sim2.get_edge_initial_rest_length(0)
        self.assertAlmostEqual(orig_edge0_rest_len, new_edge0_rest_len, places=5)

        # さらに変形が進む
        deformed_coords2 = deformed_coords.copy()
        deformed_coords2[8] -= 2.0  # 頂点2も下げる
        obj.data.vertices.foreach_set("co", deformed_coords2)
        obj.data.update()

        # ここで Reset Cloth を実行！
        bpy.ops.taremin_cloth.reset_selected()

        # 頂点座標が初回開始前の原初座標と完全一致することを検証！
        restored_coords = np.empty(n_verts * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", restored_coords)
        np.testing.assert_allclose(restored_coords, orig_coords, atol=1e-6)
        self.assertFalse(obj.get("_taremin_is_deformed", False))

    def test_apply_rest_shape(self):
        """Apply Rest Shape を実行すると、現在の変形形状が新しい初期レスト形状として確定されるかの検証"""
        bpy.ops.mesh.primitive_plane_add(size=2.0)
        obj = bpy.context.active_object
        obj.taremin_cloth.is_cloth = True

        # 初期座標
        n_verts = len(obj.data.vertices)
        orig_coords = np.empty(n_verts * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", orig_coords)

        # 初回初期化
        operators.get_or_create_simulator(obj)

        # 変形
        deformed_coords = orig_coords.copy()
        deformed_coords[2] += 0.5
        obj.data.vertices.foreach_set("co", deformed_coords)
        obj.data.update()
        obj["_taremin_is_deformed"] = True

        # Apply Rest Shape を実行
        bpy.ops.taremin_cloth.apply_rest_shape()

        # is_deformed がリセットされ、新レスト座標が保存されていること
        self.assertFalse(obj.get("_taremin_is_deformed", False))
        cached_rest = np.frombuffer(obj["_taremin_rest_positions"], dtype=np.float32)
        np.testing.assert_allclose(cached_rest, deformed_coords, atol=1e-6)

        # さらに変形させた後に Reset を押すと、この「新レスト形状」に戻ることを検証
        further_coords = deformed_coords.copy()
        further_coords[5] += 1.0
        obj.data.vertices.foreach_set("co", further_coords)
        obj.data.update()
        obj["_taremin_is_deformed"] = True

        bpy.ops.taremin_cloth.reset_selected()

        restored_coords = np.empty(n_verts * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", restored_coords)
        np.testing.assert_allclose(restored_coords, deformed_coords, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
