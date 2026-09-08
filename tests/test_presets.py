import unittest
import bpy
import sys
import shutil
from pathlib import Path

# pythonディレクトリをsys.pathに追加
addon_dir = Path(__file__).parent.parent / "python"
if str(addon_dir) not in sys.path:
    sys.path.insert(0, str(addon_dir))

import taremin_cloth
from taremin_cloth.presets import (
    BUILTIN_FABRIC_PRESETS,
    BUILTIN_SIMULATION_PRESETS,
    BUILTIN_COLLIDER_PRESETS,
    get_preset_base_dir,
    get_category_preset_dir,
    get_custom_presets,
    get_all_presets,
)


class TestPresets(unittest.TestCase):
    def setUp(self):
        taremin_cloth.register()
        bpy.ops.mesh.primitive_plane_add()
        self.obj = bpy.context.active_object
        self.created_custom_files = []

    def tearDown(self):
        # クリーンアップ: 作成されたテスト用カスタムプリセットを削除
        for p in self.created_custom_files:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

        if self.obj and self.obj.name in bpy.data.objects:
            bpy.data.objects.remove(self.obj, do_unlink=True)
        taremin_cloth.unregister()

    def test_builtin_presets_structure(self):
        """ビルトインプリセットが正しいデータ構造とパラメータを持っているか検証"""
        # Fabric
        self.assertIn("Silk (絹)", BUILTIN_FABRIC_PRESETS)
        self.assertIn("Cotton (木綿)", BUILTIN_FABRIC_PRESETS)
        self.assertIn("Leather (革)", BUILTIN_FABRIC_PRESETS)
        for name, params in BUILTIN_FABRIC_PRESETS.items():
            self.assertIn("tension_stiffness", params)
            self.assertIn("bending_stiffness", params)
            self.assertIn("air_damping", params)
            self.assertIn("thickness", params)
            self.assertGreater(params["tension_stiffness"], 0)
            self.assertGreater(params["thickness"], 0)

        # Simulation
        self.assertIn("Fast (Realtime)", BUILTIN_SIMULATION_PRESETS)
        self.assertIn("Balanced (Default)", BUILTIN_SIMULATION_PRESETS)
        self.assertIn("Best (High Res)", BUILTIN_SIMULATION_PRESETS)
        for name, params in BUILTIN_SIMULATION_PRESETS.items():
            self.assertIn("substeps", params)
            self.assertIn("solver_iterations", params)
            self.assertIn("enable_adaptive_substep", params)
            self.assertGreaterEqual(params["substeps"], 1)
            self.assertGreaterEqual(params["solver_iterations"], 1)

        # Collider
        self.assertIn("Skin (Human)", BUILTIN_COLLIDER_PRESETS)
        self.assertIn("Smooth / Metal", BUILTIN_COLLIDER_PRESETS)
        for name, params in BUILTIN_COLLIDER_PRESETS.items():
            self.assertIn("friction", params)
            self.assertIn("restitution", params)
            self.assertIn("thickness", params)
            self.assertGreaterEqual(params["friction"], 0.0)
            self.assertLessEqual(params["friction"], 1.0)

    def test_apply_fabric_preset(self):
        """布素材プリセットの適用オペレーターの動作確認"""
        self.obj.taremin_cloth.is_cloth = True

        # Silk を適用
        res = bpy.ops.taremin_cloth.apply_preset(category='fabric', preset_name="Silk (絹)")
        self.assertEqual(res, {'FINISHED'})
        self.assertEqual(self.obj.taremin_cloth.last_fabric_preset, "Silk (絹)")
        self.assertAlmostEqual(self.obj.taremin_cloth.bending_stiffness, 1.0)
        self.assertAlmostEqual(self.obj.taremin_cloth.tension_stiffness, 800.0)
        self.assertAlmostEqual(self.obj.taremin_cloth.thickness, 0.002)

        # Leather を適用
        res = bpy.ops.taremin_cloth.apply_preset(category='fabric', preset_name="Leather (革)")
        self.assertEqual(res, {'FINISHED'})
        self.assertEqual(self.obj.taremin_cloth.last_fabric_preset, "Leather (革)")
        self.assertAlmostEqual(self.obj.taremin_cloth.bending_stiffness, 200.0)
        self.assertAlmostEqual(self.obj.taremin_cloth.tension_stiffness, 8000.0)
        self.assertAlmostEqual(self.obj.taremin_cloth.thickness, 0.015)

    def test_apply_simulation_preset(self):
        """シミュレーション品質プリセットの適用オペレーターの動作確認"""
        self.obj.taremin_cloth.is_cloth = True

        # Fast を適用
        res = bpy.ops.taremin_cloth.apply_preset(category='simulation', preset_name="Fast (Realtime)")
        self.assertEqual(res, {'FINISHED'})
        self.assertEqual(self.obj.taremin_cloth.last_simulation_preset, "Fast (Realtime)")
        self.assertEqual(self.obj.taremin_cloth.substeps, 8)
        self.assertTrue(self.obj.taremin_cloth.enable_adaptive_substep)
        self.assertEqual(self.obj.taremin_cloth.solver_iterations, 1)

        # Best を適用
        res = bpy.ops.taremin_cloth.apply_preset(category='simulation', preset_name="Best (High Res)")
        self.assertEqual(res, {'FINISHED'})
        self.assertEqual(self.obj.taremin_cloth.last_simulation_preset, "Best (High Res)")
        self.assertEqual(self.obj.taremin_cloth.substeps, 40)
        self.assertFalse(self.obj.taremin_cloth.enable_adaptive_substep)
        self.assertEqual(self.obj.taremin_cloth.solver_iterations, 4)

    def test_apply_collider_preset(self):
        """コライダープリセットの適用オペレーターの動作確認"""
        self.obj.taremin_cloth_collider.is_collider = True

        # Skin を適用
        res = bpy.ops.taremin_cloth.apply_preset(category='collider', preset_name="Skin (Human)")
        self.assertEqual(res, {'FINISHED'})
        self.assertEqual(self.obj.taremin_cloth_collider.last_collider_preset, "Skin (Human)")
        self.assertAlmostEqual(self.obj.taremin_cloth_collider.friction, 0.6)
        self.assertAlmostEqual(self.obj.taremin_cloth_collider.restitution, 0.0)

        # Smooth / Metal を適用
        res = bpy.ops.taremin_cloth.apply_preset(category='collider', preset_name="Smooth / Metal")
        self.assertEqual(res, {'FINISHED'})
        self.assertEqual(self.obj.taremin_cloth_collider.last_collider_preset, "Smooth / Metal")
        self.assertAlmostEqual(self.obj.taremin_cloth_collider.friction, 0.1)
        self.assertAlmostEqual(self.obj.taremin_cloth_collider.restitution, 0.05)

    def test_custom_preset_save_load_and_delete(self):
        """カスタムプリセットの保存、再適用、削除の一連の動作確認"""
        self.obj.taremin_cloth.is_cloth = True
        self.obj.taremin_cloth.tension_stiffness = 5555.0
        self.obj.taremin_cloth.bending_stiffness = 77.0
        self.obj.taremin_cloth.thickness = 0.0123

        test_preset_name = "__UnitTest_Custom_Cloth__"

        # プリセット保存
        res = bpy.ops.taremin_cloth.save_preset(category='fabric', preset_name=test_preset_name)
        self.assertEqual(res, {'FINISHED'})

        cat_dir = get_category_preset_dir('fabric')
        preset_file = cat_dir / f"{test_preset_name}.json"
        self.created_custom_files.append(preset_file)
        self.assertTrue(preset_file.exists(), "カスタムプリセットJSONファイルが生成されていること")

        # 別のプリセットを適用して値を変更
        bpy.ops.taremin_cloth.apply_preset(category='fabric', preset_name="Silk (絹)")
        self.assertAlmostEqual(self.obj.taremin_cloth.tension_stiffness, 800.0)

        # 保存したカスタムプリセットを適用して値が復元されるか確認
        res = bpy.ops.taremin_cloth.apply_preset(category='fabric', preset_name=test_preset_name)
        self.assertEqual(res, {'FINISHED'})
        self.assertAlmostEqual(self.obj.taremin_cloth.tension_stiffness, 5555.0)
        self.assertAlmostEqual(self.obj.taremin_cloth.bending_stiffness, 77.0)
        self.assertAlmostEqual(self.obj.taremin_cloth.thickness, 0.0123)

        # プリセット削除
        res = bpy.ops.taremin_cloth.delete_preset(category='fabric', preset_name=test_preset_name)
        self.assertEqual(res, {'FINISHED'})
        self.assertFalse(preset_file.exists(), "カスタムプリセットJSONファイルが削除されていること")

    def test_batch_simulation_state(self):
        """シミュレーション有効/無効の一括操作オペレーターの動作検証"""
        self.obj.taremin_cloth.is_cloth = True
        self.obj.taremin_cloth.enabled = True

        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        cube.taremin_cloth_collider.is_collider = True
        cube.taremin_cloth_collider.enabled = True

        try:
            # 1. ALL_OFF
            bpy.ops.taremin_cloth.batch_simulation_state(action='ALL_OFF')
            self.assertFalse(self.obj.taremin_cloth.enabled)
            self.assertFalse(cube.taremin_cloth_collider.enabled)

            # 2. ALL_ON
            bpy.ops.taremin_cloth.batch_simulation_state(action='ALL_ON')
            self.assertTrue(self.obj.taremin_cloth.enabled)
            self.assertTrue(cube.taremin_cloth_collider.enabled)

            # 3. SOLO (cube がアクティブ)
            bpy.context.view_layer.objects.active = cube
            bpy.ops.taremin_cloth.batch_simulation_state(action='SOLO')
            self.assertFalse(self.obj.taremin_cloth.enabled)
            self.assertTrue(cube.taremin_cloth_collider.enabled)

            # 4. INVERT
            bpy.ops.taremin_cloth.batch_simulation_state(action='INVERT')
            self.assertTrue(self.obj.taremin_cloth.enabled)
            self.assertFalse(cube.taremin_cloth_collider.enabled)

            # 5. CLOTHS_ONLY
            bpy.ops.taremin_cloth.batch_simulation_state(action='CLOTHS_ONLY')
            self.assertTrue(self.obj.taremin_cloth.enabled)
            self.assertFalse(cube.taremin_cloth_collider.enabled)

            # 6. COLLIDERS_ONLY
            bpy.ops.taremin_cloth.batch_simulation_state(action='COLLIDERS_ONLY')
            self.assertFalse(self.obj.taremin_cloth.enabled)
            self.assertTrue(cube.taremin_cloth_collider.enabled)
        finally:
            if cube.name in bpy.data.objects:
                bpy.data.objects.remove(cube, do_unlink=True)

    def test_config_preset_save_apply_delete(self):
        """シミュレーション構成プリセットの保存・適用・削除の動作検証"""
        self.obj.taremin_cloth.is_cloth = True
        self.obj.taremin_cloth.enabled = True

        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        cube.taremin_cloth_collider.is_collider = True
        cube.taremin_cloth_collider.enabled = False

        preset_name = "__UnitTest_Scene_Config__"

        try:
            # プリセット保存
            res = bpy.ops.taremin_cloth.save_config_preset(preset_name=preset_name)
            self.assertEqual(res, {'FINISHED'})
            self.assertEqual(bpy.context.scene.taremin_cloth_active_config_preset, preset_name)

            # 状態を変更 (全OFF)
            bpy.ops.taremin_cloth.batch_simulation_state(action='ALL_OFF')
            self.assertFalse(self.obj.taremin_cloth.enabled)
            self.assertFalse(cube.taremin_cloth_collider.enabled)

            # プリセット適用 (保存時の状態に復元: Plane=True, Cube=False)
            res_apply = bpy.ops.taremin_cloth.apply_config_preset(preset_name=preset_name)
            self.assertEqual(res_apply, {'FINISHED'})
            self.assertTrue(self.obj.taremin_cloth.enabled)
            self.assertFalse(cube.taremin_cloth_collider.enabled)

            # ビルトインプリセット適用テスト
            res_builtin = bpy.ops.taremin_cloth.apply_config_preset(preset_name="All Enabled (全て有効)")
            self.assertEqual(res_builtin, {'FINISHED'})
            self.assertTrue(self.obj.taremin_cloth.enabled)
            self.assertTrue(cube.taremin_cloth_collider.enabled)

            # プリセット削除
            res_del = bpy.ops.taremin_cloth.delete_config_preset(preset_name=preset_name)
            self.assertEqual(res_del, {'FINISHED'})
        finally:
            if cube.name in bpy.data.objects:
                bpy.data.objects.remove(cube, do_unlink=True)


if __name__ == "__main__":
    unittest.main()

