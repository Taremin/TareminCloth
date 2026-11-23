"""
taremin_cloth 単一メッシュSDFコライダー (MESH_SDF) E2E統合テスト (Blender API依存)
アーマチュアを持たない一般メッシュ (Suzanne / Cylinder) に対する MESH_SDF コライダーのベイク、
布シミュレーションにおける接触・支持、オブジェクト移動追従の検証
"""

import unittest
import bpy
import numpy as np
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests
import taremin_cloth
from taremin_cloth.engine.collider import sync_colliders
from taremin_cloth.engine.sdf_baker import clear_all_cached_sdf


@unittest.skipUnless(getattr(tests, "_IS_REAL_BLENDER", False), "Blender実行環境が必要です")
class TestMeshSdfE2E(unittest.TestCase):
    def setUp(self):
        clear_all_cached_sdf()
        bpy.ops.wm.read_factory_settings(use_empty=True)
        taremin_cloth.register()

    def tearDown(self):
        taremin_cloth.unregister()
        clear_all_cached_sdf()

    def test_mesh_sdf_suzanne_collision(self):
        """アーマチュアなしのSuzanneにMESH_SDFを設定し、step_cloth_scene経由で頭部支持を検証するE2Eテスト"""
        # 1. コライダーメッシュ作成 (Suzanne、アーマチュア一切なし)
        bpy.ops.mesh.primitive_monkey_add(size=1.0, location=(0.0, 0.0, 0.0))
        suzanne = bpy.context.active_object
        suzanne.name = "Collider_Suzanne"

        suzanne.taremin_cloth_collider.is_collider = True
        suzanne.taremin_cloth_collider.enabled = True
        suzanne.taremin_cloth_collider.collider_type = 'MESH_SDF'
        suzanne.taremin_cloth_collider.mesh_sdf_voxel_size = 0.02  # 2cm
        suzanne.taremin_cloth_collider.thickness = 0.02
        suzanne.taremin_cloth_collider.friction = 0.4
        suzanne.taremin_cloth_collider.mesh_sdf_cache_enabled = True

        # 2. 布メッシュ作成 (Suzanneの上部 Z=1.0 に水平グリッドを配置)
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=5, y_subdivisions=5, size=0.6, location=(0.0, 0.0, 1.0))
        cloth_obj = bpy.context.active_object
        cloth_obj.name = "Cloth_Grid"
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        cloth_obj.taremin_cloth.is_cloth = True
        cloth_obj.taremin_cloth.layer_id = 0
        cloth_obj.taremin_cloth.thickness = 0.02
        cloth_obj.taremin_cloth.tension_stiffness = 5000.0
        cloth_obj.taremin_cloth.bending_stiffness = 20.0
        cloth_obj.taremin_cloth.substeps = 10

        # 3. 自由落下シミュレーション (30ステップ: step_cloth_scene 経由でコライダー同期+ステップ)
        scene = bpy.context.scene
        res = {}
        for _ in range(30):
            res = taremin_cloth.step_cloth_scene(scene, dt=0.016, update_mesh=True)

        _, out_pos = res[cloth_obj.name]
        sim_pos = out_pos.reshape((-1, 3))

        min_z = float(np.min(sim_pos[:, 2]))
        max_z = float(np.max(sim_pos[:, 2]))
        print(f"[Test Mesh SDF Suzanne] 布のZ座標範囲: min={min_z:.4f}, max={max_z:.4f}")

        self.assertGreater(min_z, 0.30, f"布がSuzanneの頭部を貫通して落下しています: min_z={min_z}")
        self.assertLess(max_z, 1.0, f"布がまったく落下していません: max_z={max_z}")

    def test_mesh_sdf_object_movement_follow(self):
        """MESH_SDFコライダー移動時、step_cloth_scene経由で布が押し上げられるリアルタイム追従検証"""
        # 1. 立方体コライダー (Z=0〜0.6)
        bpy.ops.mesh.primitive_cube_add(size=0.6, location=(0.0, 0.0, 0.3))
        cube = bpy.context.active_object
        cube.name = "Collider_Cube"

        cube.taremin_cloth_collider.is_collider = True
        cube.taremin_cloth_collider.enabled = True
        cube.taremin_cloth_collider.collider_type = 'MESH_SDF'
        cube.taremin_cloth_collider.mesh_sdf_voxel_size = 0.03
        cube.taremin_cloth_collider.thickness = 0.01

        # 2. 単一頂点布オブジェクト (立方体上面 Z=0.6 のすぐ上 Z=0.8)
        cloth_mesh = bpy.data.meshes.new("ClothPointMesh")
        cloth_mesh.from_pydata([[0.0, 0.0, 0.8]], [], [])
        cloth_mesh.update()
        cloth_obj = bpy.data.objects.new("ClothPointObj", cloth_mesh)
        bpy.context.scene.collection.objects.link(cloth_obj)
        cloth_obj.taremin_cloth.is_cloth = True
        cloth_obj.taremin_cloth.thickness = 0.01
        cloth_obj.taremin_cloth.substeps = 10

        bpy.context.view_layer.update()
        scene = bpy.context.scene

        # 落下させて立方体上面 (z ≈ 0.6 + 0.02 = 0.62) で静止させる
        res = {}
        for _ in range(30):
            res = taremin_cloth.step_cloth_scene(scene, dt=0.016, update_mesh=True)

        _, out_pos = res[cloth_obj.name]
        initial_rest_z = float(out_pos[2])
        print(f"[Test Movement Follow] 初期静止Z: {initial_rest_z:.4f}")
        self.assertAlmostEqual(initial_rest_z, 0.62, delta=0.03)

        # 3. コライダーオブジェクトを Z=+0.2 移動
        cube.location.z += 0.2
        bpy.context.view_layer.update()

        for _ in range(20):
            res = taremin_cloth.step_cloth_scene(scene, dt=0.016, update_mesh=True)

        _, out_pos = res[cloth_obj.name]
        moved_rest_z = float(out_pos[2])
        print(f"[Test Movement Follow] コライダー移動後Z: {moved_rest_z:.4f} (期待値 ≈ {initial_rest_z + 0.2:.4f})")

        self.assertGreater(moved_rest_z, initial_rest_z + 0.15, f"コライダーの移動に追従して布が押し上げられていません: {moved_rest_z}")

    def test_mesh_sdf_auto_vram_clamping_e2e(self):
        """Blenderオブジェクト上でMax VRAM予算（128MB）を設定し、step_cloth_scene経由で安全に動作するE2Eテスト"""
        bpy.ops.mesh.primitive_cube_add(size=0.8, location=(0.0, 0.0, 0.4))
        cube = bpy.context.active_object
        cube.name = "Collider_Cube_VramTest"

        cube.taremin_cloth_collider.is_collider = True
        cube.taremin_cloth_collider.enabled = True
        cube.taremin_cloth_collider.collider_type = 'MESH_SDF'
        cube.taremin_cloth_collider.mesh_sdf_voxel_size = 0.002  # 2mm
        cube.taremin_cloth_collider.mesh_sdf_max_vram_mb = 128   # 128MB上限
        cube.taremin_cloth_collider.mesh_sdf_auto_scale = True
        cube.taremin_cloth_collider.thickness = 0.01

        cloth_mesh = bpy.data.meshes.new("ClothVramPointMesh")
        cloth_mesh.from_pydata([[0.0, 0.0, 1.0]], [], [])
        cloth_mesh.update()
        cloth_obj = bpy.data.objects.new("ClothVramPointObj", cloth_mesh)
        bpy.context.scene.collection.objects.link(cloth_obj)
        cloth_obj.taremin_cloth.is_cloth = True
        cloth_obj.taremin_cloth.thickness = 0.01
        cloth_obj.taremin_cloth.substeps = 10

        scene = bpy.context.scene
        res = {}
        for _ in range(30):
            res = taremin_cloth.step_cloth_scene(scene, dt=0.016, update_mesh=True)

        _, out_pos = res[cloth_obj.name]
        print(f"[Test Auto VRAM E2E] 上限128MBでの静止Z: {out_pos[2]:.4f} (上面 z=0.8 + 0.02 = 0.82)")
        self.assertAlmostEqual(float(out_pos[2]), 0.82, delta=0.03)


if __name__ == "__main__":
    unittest.main()
