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

root_dir = Path(__file__).parent.parent
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import taremin_cloth
from taremin_cloth.engine.collider import sync_colliders
from taremin_cloth.engine.sdf_baker import clear_all_cached_sdf


class TestMeshSdfE2E(unittest.TestCase):
    def setUp(self):
        clear_all_cached_sdf()
        bpy.ops.wm.read_factory_settings(use_empty=True)
        taremin_cloth.register()

    def tearDown(self):
        taremin_cloth.unregister()
        clear_all_cached_sdf()

    def test_mesh_sdf_suzanne_collision(self):
        """アーマチュアなしのSuzanne（モンキー）にMESH_SDFを設定し、布が頭部で支持されるE2Eテスト"""
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

        mesh = cloth_obj.data
        n_verts = len(mesh.vertices)
        coords = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coords)
        pos = coords.reshape((n_verts, 3)) + np.array([0.0, 0.0, 1.0], dtype=np.float32)

        n_edges = len(mesh.edges)
        edge_indices = np.empty(n_edges * 2, dtype=np.uint32)
        mesh.edges.foreach_get("vertices", edge_indices)
        edges = edge_indices.reshape((n_edges, 2))

        tri_list = []
        mesh.calc_loop_triangles()
        for tri in mesh.loop_triangles:
            tri_list.append(tri.vertices)
        faces = np.array(tri_list, dtype=np.uint32)

        inv_masses = np.ones(n_verts, dtype=np.float32)

        import taremin_cloth_core
        sim = taremin_cloth_core.ClothSimulator(
            positions=pos,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            layer_id=0,
            thickness=0.02,
            stiffness=5000.0,
            bending_stiffness=20.0,
        )
        sim.set_gravity(0.0, 0.0, -9.8)

        # 3. コライダー同期 (MESH_SDF が自動ベイクされシミュレータに登録される)
        sync_colliders(sim, bpy.context.scene)

        # 4. 自由落下シミュレーション (30ステップ)
        for _ in range(30):
            sim.step(dt=0.016, substeps=10)

        out_pos = np.zeros(n_verts * 3, dtype=np.float32)
        sim.get_positions(out_pos)
        sim_pos = out_pos.reshape((-1, 3))

        min_z = float(np.min(sim_pos[:, 2]))
        max_z = float(np.max(sim_pos[:, 2]))
        print(f"[Test Mesh SDF Suzanne] 布のZ座標範囲: min={min_z:.4f}, max={max_z:.4f}")

        # Suzanneの頭頂部は z ≈ 0.4〜0.5。布厚み・コライダー厚み込みで、布は z > 0.35 で止まるはず（落下して奈落に落ちていない）
        self.assertGreater(min_z, 0.30, f"布がSuzanneの頭部を貫通して落下しています: min_z={min_z}")
        self.assertLess(max_z, 1.0, f"布がまったく落下していません: max_z={max_z}")

    def test_mesh_sdf_object_movement_follow(self):
        """MESH_SDFコライダーオブジェクトを移動させた際、布が押し上げられる（リアルタイム追従）検証"""
        # 1. 立方体コライダー (Z=0〜0.6)
        bpy.ops.mesh.primitive_cube_add(size=0.6, location=(0.0, 0.0, 0.3))
        cube = bpy.context.active_object
        cube.name = "Collider_Cube"

        cube.taremin_cloth_collider.is_collider = True
        cube.taremin_cloth_collider.enabled = True
        cube.taremin_cloth_collider.collider_type = 'MESH_SDF'
        cube.taremin_cloth_collider.mesh_sdf_voxel_size = 0.03
        cube.taremin_cloth_collider.thickness = 0.01

        # 2. 布頂点 (立方体の上面 Z=0.6 のすぐ上 Z=0.8 から落下)
        cloth_verts = np.array([[0.0, 0.0, 0.8]], dtype=np.float32)
        import taremin_cloth_core
        sim = taremin_cloth_core.ClothSimulator(
            positions=cloth_verts,
            edges=np.empty((0, 2), dtype=np.uint32),
            thickness=0.01,
        )
        sim.set_gravity(0.0, 0.0, -9.8)

        bpy.context.view_layer.update()
        sync_colliders(sim, bpy.context.scene)

        # 落下させて立方体上面（z ≈ 0.6 + 0.02 = 0.62）で静止させる
        for _ in range(30):
            sim.step(dt=0.016, substeps=10)

        out_pos = np.zeros(3, dtype=np.float32)
        sim.get_positions(out_pos)
        initial_rest_z = float(out_pos[2])
        print(f"[Test Movement Follow] 初期静止Z: {initial_rest_z:.4f}")
        self.assertAlmostEqual(initial_rest_z, 0.62, delta=0.03)

        # 3. コライダーオブジェクトを Z=+0.2 移動
        cube.location.z += 0.2
        bpy.context.view_layer.update()

        # sync_colliders を呼んで変換行列を更新し、シミュレーションを進める
        sync_colliders(sim, bpy.context.scene)
        for _ in range(20):
            sync_colliders(sim, bpy.context.scene)
            sim.step(dt=0.016, substeps=10)

        sim.get_positions(out_pos)
        moved_rest_z = float(out_pos[2])
        print(f"[Test Movement Follow] コライダー移動後Z: {moved_rest_z:.4f} (期待値 ≈ {initial_rest_z + 0.2:.4f})")

        # コライダーの移動に伴って布も Z ≈ 0.82 に押し上げられていること
        self.assertGreater(moved_rest_z, initial_rest_z + 0.15, f"コライダーの移動に追従して布が押し上げられていません: {moved_rest_z}")

    def test_mesh_sdf_auto_vram_clamping_e2e(self):
        """Blenderオブジェクト上でMax VRAM予算（128MB）を設定し、自動最適化されて安全にベイク・シミュレーションされるE2Eテスト"""
        # 0.8m 立方体コライダー (通常2mmで約320MB〜400MB)
        bpy.ops.mesh.primitive_cube_add(size=0.8, location=(0.0, 0.0, 0.4))
        cube = bpy.context.active_object
        cube.name = "Collider_Cube_VramTest"

        cube.taremin_cloth_collider.is_collider = True
        cube.taremin_cloth_collider.enabled = True
        cube.taremin_cloth_collider.collider_type = 'MESH_SDF'
        cube.taremin_cloth_collider.mesh_sdf_voxel_size = 0.002 # 2mm
        cube.taremin_cloth_collider.mesh_sdf_max_vram_mb = 128  # 128MB上限
        cube.taremin_cloth_collider.mesh_sdf_auto_scale = True
        cube.taremin_cloth_collider.thickness = 0.01

        cloth_verts = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        import taremin_cloth_core
        sim = taremin_cloth_core.ClothSimulator(
            positions=cloth_verts,
            edges=np.empty((0, 2), dtype=np.uint32),
            thickness=0.01,
        )
        sim.set_gravity(0.0, 0.0, -9.8)

        # sync_colliders で自動スケール調整されてベイク完了すること
        sync_colliders(sim, bpy.context.scene)

        # 落下シミュレーション (立方体上面 z=0.8 で停止)
        for _ in range(30):
            sim.step(dt=0.016, substeps=10)

        out_pos = np.zeros(3, dtype=np.float32)
        sim.get_positions(out_pos)
        print(f"[Test Auto VRAM E2E] 上限128MBでの静止Z: {out_pos[2]:.4f} (上面 z=0.8 + 0.02 = 0.82)")
        self.assertAlmostEqual(out_pos[2], 0.82, delta=0.03)


if __name__ == "__main__":
    unittest.main()
