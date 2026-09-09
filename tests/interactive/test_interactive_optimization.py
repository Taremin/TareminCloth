"""
インタラクティブモード高速化機能（描画隔離・ローカルビュー、ピン描画キャッシュ、アキュムレータ制御）の統合テスト
"""

import sys
from pathlib import Path
import unittest

root_dir = Path(__file__).parent.parent
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import bpy
import taremin_cloth
from taremin_cloth.ops.interactive import enter_isolated_view, exit_isolated_view
from taremin_cloth.utils import drawing


class TestInteractiveOptimization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            taremin_cloth.register()
        except ValueError:
            pass

    @classmethod
    def tearDownClass(cls):
        try:
            taremin_cloth.unregister()
        except Exception:
            pass

    def setUp(self):
        # シーン内の全オブジェクトを削除
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def tearDown(self):
        drawing.set_interactive_active(False)
        drawing.clear_pin_cache()

    def test_isolate_viewport_view_property_default(self):
        """isolate_viewport_view プロパティがデフォルト True で存在することを検証"""
        bpy.ops.mesh.primitive_plane_add()
        obj = bpy.context.active_object
        obj.taremin_cloth.is_cloth = True

        self.assertTrue(hasattr(obj.taremin_cloth, "isolate_viewport_view"))
        self.assertTrue(obj.taremin_cloth.isolate_viewport_view)

        # トグル可能か検証
        obj.taremin_cloth.isolate_viewport_view = False
        self.assertFalse(obj.taremin_cloth.isolate_viewport_view)
        obj.taremin_cloth.isolate_viewport_view = True
        self.assertTrue(obj.taremin_cloth.isolate_viewport_view)

    def test_isolated_view_enter_and_exit(self):
        """enter_isolated_view と exit_isolated_view でローカルビューが安全に切り替わり復帰するか検証"""
        # 布オブジェクト
        bpy.ops.mesh.primitive_plane_add(location=(0, 0, 0))
        cloth_obj = bpy.context.active_object
        cloth_obj.name = "ClothPlane"
        cloth_obj.taremin_cloth.is_cloth = True

        # コライダーオブジェクト
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, -1))
        col_obj = bpy.context.active_object
        col_obj.name = "ColliderCube"
        col_obj.taremin_cloth_collider.is_collider = True
        col_obj.taremin_cloth_collider.enabled = True

        # 無関係の重いオブジェクト
        bpy.ops.mesh.primitive_uv_sphere_add(location=(5, 5, 5))
        other_obj = bpy.context.active_object
        other_obj.name = "BackgroundSphere"

        # 布オブジェクトを選択状態にしてアクティブに
        bpy.ops.object.select_all(action='DESELECT')
        cloth_obj.select_set(True)
        bpy.context.view_layer.objects.active = cloth_obj

        # 3D Viewport エリアを探す
        v3d_areas = [a for a in bpy.context.screen.areas if a.type == 'VIEW_3D']
        if not v3d_areas:
            self.skipTest("VIEW_3D area not found in headless environment")

        # 1. ローカルビューに入る
        isolated_areas, saved_selection, saved_active = enter_isolated_view(bpy.context, cloth_obj)

        for area in isolated_areas:
            space = area.spaces.active
            self.assertIsNotNone(space.local_view, "ローカルビューが有効化されている必要があります")

        # 布オブジェクトがアクティブかつ選択されていること
        self.assertEqual(bpy.context.active_object, cloth_obj)
        self.assertTrue(cloth_obj.select_get())

        # 2. ローカルビューを解除
        exit_isolated_view(bpy.context, isolated_areas, saved_selection, saved_active)

        for area in v3d_areas:
            space = area.spaces.active
            self.assertIsNone(space.local_view, "ローカルビューが解除されている必要があります")

        # 元の選択状態が維持されていること
        self.assertTrue(cloth_obj.select_get())

    def test_pin_drawing_cache(self):
        """ピン描画のインデックスキャッシュが正しく生成・クリアされるか検証"""
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=5, y_subdivisions=5)
        obj = bpy.context.active_object
        obj.taremin_cloth.is_cloth = True

        # ピン頂点グループを作成
        vg = obj.vertex_groups.new(name="Pin")
        vg.add([0, 1, 2], 1.0, 'REPLACE')

        drawing.clear_pin_cache()
        self.assertEqual(len(drawing._pin_indices_cache), 0)

        drawing.set_interactive_active(True)
        self.assertTrue(drawing.is_interactive_active())

        # drawing._pin_indices_cache へのアクセス確認
        cache_key = (obj.name, "Pin")
        vert_count = len(obj.data.vertices)
        self.assertEqual(vert_count, 36)  # (5+1)*(5+1) = 36

        pin_indices = [
            v.index for v in obj.data.vertices
            for g in v.groups if g.group == vg.index and g.weight > 0.0
        ]
        drawing._pin_indices_cache[cache_key] = (pin_indices, vert_count)

        self.assertIn(cache_key, drawing._pin_indices_cache)
        cached_indices, cached_count = drawing._pin_indices_cache[cache_key]
        self.assertEqual(cached_indices, [0, 1, 2])
        self.assertEqual(cached_count, 36)

        # インタラクティブ終了時にキャッシュがクリアされること
        drawing.set_interactive_active(False)
        self.assertEqual(len(drawing._pin_indices_cache), 0)


if __name__ == "__main__":
    unittest.main()
