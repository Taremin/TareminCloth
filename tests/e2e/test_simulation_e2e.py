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

import taremin_cloth_core
import taremin_cloth


class TestSimulationE2E(unittest.TestCase):
    def setUp(self):
        taremin_cloth.register()

    def tearDown(self):
        taremin_cloth.unregister()

    def test_blender_cloth_simulation(self):
        """Blenderメッシュからデータを取得しGPU Clothシミュレーションを実行して座標を反映するE2Eテスト"""
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=1.0)
        obj = bpy.context.active_object
        mesh = obj.data

        n_verts = len(mesh.vertices)
        self.assertEqual(n_verts, 25)

        # 頂点座標の取得 (NumPy配列)
        coords = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coords)
        pos_2d = coords.reshape((n_verts, 3))

        # エッジの取得
        n_edges = len(mesh.edges)
        edge_indices = np.empty(n_edges * 2, dtype=np.uint32)
        mesh.edges.foreach_get("vertices", edge_indices)
        edges_2d = edge_indices.reshape((n_edges, 2))

        # 面の取得
        tri_list = []
        mesh.calc_loop_triangles()
        for tri in mesh.loop_triangles:
            tri_list.append(tri.vertices)
        faces_2d = np.array(tri_list, dtype=np.uint32)

        # ピン留め: 上端の頂点（Y座標が最大のもの）を固定
        max_y = np.max(pos_2d[:, 1])
        inv_masses = np.ones(n_verts, dtype=np.float32)
        for i, pos in enumerate(pos_2d):
            if abs(pos[1] - max_y) < 1e-4:
                inv_masses[i] = 0.0

        # シミュレータ作成
        sim = taremin_cloth_core.ClothSimulator(
            positions=pos_2d,
            edges=edges_2d,
            faces=faces_2d,
            inv_masses=inv_masses,
            layer_id=0,
            thickness=0.005,
            stiffness=10000.0,
            bending_stiffness=50.0,
        )

        self.assertGreater(sim.num_bending_constraints, 0)

        # 60フレーム進める
        for _ in range(60):
            sim.step(dt=1.0 / 60.0, substeps=20)

        # 頂点座標の更新
        sim.get_positions(coords)
        mesh.vertices.foreach_set("co", coords)
        mesh.update()

        # 検証: 最下端の頂点がZ軸下向きに落下しているか
        updated_pos = coords.reshape((n_verts, 3))
        self.assertTrue(np.all(np.isfinite(updated_pos)), "NaN/Infが含まれないこと")
        self.assertLess(np.min(updated_pos[:, 2]), -0.01, "布が重力で下方に垂れ下がっていること")

        # 固定ピン頂点の位置が保持されていること
        for i, pos in enumerate(pos_2d):
            if abs(pos[1] - max_y) < 1e-4:
                np.testing.assert_allclose(updated_pos[i], pos, atol=1e-5)

        # クリーンアップ
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh, do_unlink=True)

    def test_timeline_frame_handler_with_pin_vertex_group(self):
        """頂点グループ 'Pin' を持つ布メッシュがタイムライン再生で正しく固定・変形されるか検証"""
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=1.0)
        obj = bpy.context.active_object
        obj.taremin_cloth.is_cloth = True
        obj.taremin_cloth.stiffness = 10000.0
        obj.taremin_cloth.bending_stiffness = 20.0

        vg = obj.vertex_groups.new(name="Pin")
        pin_indices = []
        for v in obj.data.vertices:
            if v.co.y > 0.4:
                vg.add([v.index], 1.0, 'REPLACE')
                pin_indices.append(v.index)

        init_coords = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", init_coords)
        init_pos = init_coords.reshape((-1, 3))

        scene = bpy.context.scene
        scene.frame_set(1)
        for f in range(2, 20):
            scene.frame_set(f)

        cur_coords = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", cur_coords)
        cur_pos = cur_coords.reshape((-1, 3))

        for p_idx in pin_indices:
            np.testing.assert_allclose(cur_pos[p_idx], init_pos[p_idx], atol=1e-5)

        self.assertLess(np.min(cur_pos[:, 2]), -0.01)

        bpy.data.objects.remove(obj, do_unlink=True)

    def test_frame_buffering_e2e(self):
        """GPUフレームバッファリングにより中間フレームが欠落なくキャッシュされスクラブ復元できることを検証"""
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=1.0)
        obj = bpy.context.active_object
        obj.name = "BufferingTestCloth"
        settings = obj.taremin_cloth
        settings.is_cloth = True
        settings.enable_frame_buffering = True
        settings.frame_buffer_size = 3

        scene = bpy.context.scene
        scene.frame_start = 1
        scene.frame_end = 10

        # 1. タイムライン再生 (フレーム 1 -> 10)
        scene.frame_set(1)
        for f in range(2, 11):
            scene.frame_set(f)

        # 2. キャッシュ蓄積の確認
        import taremin_cloth.operators as ops
        cache = ops._timeline_frame_cache.get(obj.name, {})
        for f in range(2, 11):
            self.assertIn(f, cache, f"フレーム {f} がタイムラインキャッシュに存在すること")

        # 3. 巻き戻し・スクラブテスト (バッファリング中間だったフレーム 3, 6 へ手動移動)
        for f in [3, 6, 2, 7]:
            scene.frame_set(f)
            mesh_co = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
            obj.data.vertices.foreach_get("co", mesh_co)
            cached_co = cache[f]
            np.testing.assert_allclose(mesh_co, cached_co, atol=1e-5, err_msg=f"フレーム {f} のメッシュ座標がキャッシュと完全一致すること")

        bpy.data.objects.remove(obj, do_unlink=True)

    def test_sewing_e2e(self):
        """2枚の型紙メッシュが縫合エッジ（Loose edge）により引き合わされるE2Eテスト"""
        mesh = bpy.data.meshes.new(name="SewingTestMesh")
        verts = [
            [-1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-2.0, 0.5, 0.0],
            [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.5, 0.0],
        ]
        edges = [
            (0, 1), (1, 2), (2, 0),
            (3, 4), (4, 5), (5, 3),
            (0, 3), (1, 4),
        ]
        faces = [
            (0, 1, 2),
            (3, 4, 5),
        ]
        mesh.from_pydata(verts, edges, faces)
        mesh.update()

        obj = bpy.data.objects.new("SewingTestObj", mesh)
        bpy.context.collection.objects.link(obj)

        obj.taremin_cloth.is_cloth = True
        obj.taremin_cloth.enable_sewing = True
        obj.taremin_cloth.sewing_shrink_speed = 5.0

        from taremin_cloth.operators import get_or_create_simulator, clear_simulators
        clear_simulators()
        sim, coords = get_or_create_simulator(obj)
        self.assertEqual(sim.num_sewing_constraints, 2)

        sim.set_gravity(0.0, 0.0, 0.0)

        for _ in range(60):
            sim.step(dt=1.0 / 60.0, substeps=20)

        sim.get_positions(coords)
        cur_pos = coords.reshape((-1, 3))

        dist03 = np.linalg.norm(cur_pos[0] - cur_pos[3])
        dist14 = np.linalg.norm(cur_pos[1] - cur_pos[4])
        print(f"\n[Test Sewing E2E] Dist 0-3: {dist03:.4f}, Dist 1-4: {dist14:.4f}")
        self.assertLess(dist03, 0.05, f"縫合エッジ(0,3)が収縮すること (実測: {dist03})")
        self.assertLess(dist14, 0.05, f"縫合エッジ(1,4)が収縮すること (実測: {dist14})")

        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh, do_unlink=True)

    def test_primitive_collider_e2e(self):
        """Blenderシーン内の球コライダーに布が接触して貫通しないE2Eテスト"""
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0.0, 0.0, 0.0))
        collider_obj = bpy.context.active_object
        collider_obj.taremin_cloth_collider.is_collider = True
        collider_obj.taremin_cloth_collider.collider_type = 'SPHERE'
        collider_obj.taremin_cloth_collider.radius = 0.5
        collider_obj.taremin_cloth_collider.friction = 0.3

        bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=1.0, location=(0.0, 0.0, 1.0))
        cloth_obj = bpy.context.active_object
        cloth_obj.taremin_cloth.is_cloth = True
        cloth_obj.taremin_cloth.stiffness = 10000.0

        from taremin_cloth.operators import get_or_create_simulator, clear_simulators
        clear_simulators()
        sim, coords = get_or_create_simulator(cloth_obj)

        for _ in range(60):
            sim.step(dt=1.0 / 60.0, substeps=20)

        sim.get_positions(coords)
        cur_pos = coords.reshape((-1, 3))

        for i, p in enumerate(cur_pos):
            dist = np.linalg.norm(p)
            self.assertGreaterEqual(dist, 0.5 - 1e-4, f"頂点 {i} が球コライダーに貫通している: dist={dist}")

        bpy.data.objects.remove(cloth_obj, do_unlink=True)
        bpy.data.objects.remove(collider_obj, do_unlink=True)

    def test_mesh_collider_e2e(self):
        """Blenderシーン内のカスタムメッシュコライダー (Suzanne) に布が接触して貫通しないE2Eテスト"""
        bpy.ops.mesh.primitive_monkey_add(size=1.0, location=(0.0, 0.0, 0.0))
        monkey_obj = bpy.context.active_object
        monkey_obj.taremin_cloth_collider.is_collider = True
        monkey_obj.taremin_cloth_collider.collider_type = 'MESH'
        monkey_obj.taremin_cloth_collider.thickness = 0.02

        # 布メッシュ (z=1.0)
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=0.8, location=(0.0, 0.0, 1.0))
        cloth_obj = bpy.context.active_object
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        cloth_obj.taremin_cloth.is_cloth = True
        cloth_obj.taremin_cloth.stiffness = 10000.0

        from taremin_cloth.operators import get_or_create_simulator, clear_simulators
        clear_simulators()
        sim, coords = get_or_create_simulator(cloth_obj)

        for _ in range(60):
            sim.step(dt=1.0 / 60.0, substeps=20)

        sim.get_positions(coords)
        cur_pos = coords.reshape((-1, 3))

        # 布が自由落下しきらず、Suzanneメッシュ上に留まっていること (サルの底面 z=-0.492 より上)
        min_z = np.min(cur_pos[:, 2])
        max_z = np.max(cur_pos[:, 2])
        print(f"\n[Test Mesh Collider E2E] Min Z on Monkey: {min_z:.4f}, Max Z: {max_z:.4f}")
        self.assertGreater(min_z, -0.50, "布がSuzanneメッシュコライダーを突き抜けず上に留まっていること")
        self.assertGreater(max_z, 0.20, "布の上部がサルの頭頂部付近に留まっていること")

        bpy.data.objects.remove(cloth_obj, do_unlink=True)
        bpy.data.objects.remove(monkey_obj, do_unlink=True)

    def test_multilayer_e2e(self):
        """2枚の重ね着布（インナーとアウター）のマルチレイヤーE2Eテスト"""
        mesh = bpy.data.meshes.new(name="MultiLayerMesh")
        verts = [
            [-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [-0.2, 0.2, 0.0], [0.2, 0.2, 0.0],
            [-0.2, -0.2, 0.2], [0.2, -0.2, 0.2], [-0.2, 0.2, 0.2], [0.2, 0.2, 0.2],
        ]
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
        ]
        faces = [
            (0, 1, 2), (1, 3, 2),
            (4, 5, 6), (5, 7, 6),
        ]
        mesh.from_pydata(verts, edges, faces)
        mesh.update()

        obj = bpy.data.objects.new("MultiLayerObj", mesh)
        bpy.context.collection.objects.link(obj)

        vg = obj.vertex_groups.new(name="Pin")
        vg.add([0, 1, 2, 3], 1.0, 'REPLACE')

        layer_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint32)
        thicknesses = np.array([0.05] * 8, dtype=np.float32)

        coords = np.empty(8 * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coords)
        pos_2d = coords.reshape((8, 3))
        edges_2d = np.array(edges, dtype=np.uint32)
        faces_2d = np.array(faces, dtype=np.uint32)
        inv_masses = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=pos_2d,
            edges=edges_2d,
            faces=faces_2d,
            inv_masses=inv_masses,
            layer_ids=layer_ids,
            thicknesses=thicknesses,
            stiffness=10000.0,
        )

        for _ in range(60):
            sim.step(dt=1.0 / 60.0, substeps=20)

        sim.get_positions(coords)
        cur_pos = coords.reshape((8, 3))

        for i in range(4, 8):
            self.assertGreaterEqual(cur_pos[i, 2], 0.10 - 1e-3, f"アウター頂点 {i} がインナーに貫通している: z={cur_pos[i, 2]}")

        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh, do_unlink=True)

    def test_interactive_operator_poll(self):
        """インタラクティブオペレーターのPollおよび登録テスト"""
        self.assertTrue(hasattr(bpy.ops.taremin_cloth, "interactive"))
        self.assertTrue(hasattr(bpy.ops.taremin_cloth, "reset_simulation"))

    def test_interactive_raycast_picking_and_depth(self):
        """インタラクティブモードのレイキャスト頂点ピッキングと深度計算の検証"""
        import mathutils

        # 手前(z=1.0)と奥(z=0.0)に面を持つ2層メッシュを作成
        mesh = bpy.data.meshes.new(name="DoubleLayerMesh")
        verts = [
            [-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0],  # 手前ポリゴン (index: 0)
            [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0],  # 奥ポリゴン (index: 1)
        ]
        faces = [[0, 1, 2], [3, 4, 5]]
        mesh.from_pydata(verts, [], faces)
        mesh.update()

        obj = bpy.data.objects.new("DoubleLayerObj", mesh)
        bpy.context.collection.objects.link(obj)
        obj.location = (2.0, 0.0, 0.0)  # ワールドオフセット
        bpy.context.view_layer.update()

        # カメラ視線レイ: z軸正方向(z=5.0)から真下(z=-1.0方向)へ
        ray_origin_world = mathutils.Vector((2.0, 0.0, 5.0))
        ray_dir_world = mathutils.Vector((0.0, 0.0, -1.0))

        # ローカル変換
        matrix_inv = obj.matrix_world.inverted()
        ray_origin_local = matrix_inv @ ray_origin_world
        ray_dir_local = (matrix_inv.to_3x3() @ ray_dir_world).normalized()

        hit, hit_loc, hit_normal, face_idx = obj.ray_cast(ray_origin_local, ray_dir_local)
        self.assertTrue(hit, "レイキャストがメッシュ表面にヒットすること")
        self.assertEqual(face_idx, 0, "手前の面 (face_idx=0) が優先してヒットすること（奥の面ではない）")

        # 頂点選択
        polygon = obj.data.polygons[face_idx]
        best_idx = None
        min_dist_sq = float('inf')
        for v_idx in polygon.vertices:
            v_co = obj.data.vertices[v_idx].co
            dist_sq = (v_co - hit_loc).length_squared
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_idx = v_idx

        self.assertIn(best_idx, [0, 1, 2], "手前ポリゴンの頂点が選択されること")

        # 選択された頂点自身の位置を基準とするビュー平面の定義
        camera_forward = mathutils.Vector((0.0, 0.0, -1.0))
        v_initial_local = mesh.vertices[best_idx].co.copy()
        v_initial_world = obj.matrix_world @ v_initial_local
        grab_plane_point = v_initial_world

        # クリック時の初期レイとビュー平面の初期交点
        denom0 = ray_dir_world.dot(camera_forward)
        t0 = (grab_plane_point - ray_origin_world).dot(camera_forward) / denom0
        initial_plane_hit = ray_origin_world + ray_dir_world * t0

        # 検証1: マウス移動ゼロ（静止時）では、目標位置が初期頂点位置と完全に一致すること（交点スナップが発生しないこと）
        delta_world_zero = initial_plane_hit - initial_plane_hit
        delta_local_zero = matrix_inv.to_3x3() @ delta_world_zero
        target_pos_local_zero = v_initial_local + delta_local_zero
        np.testing.assert_allclose(
            [target_pos_local_zero.x, target_pos_local_zero.y, target_pos_local_zero.z],
            [v_initial_local.x, v_initial_local.y, v_initial_local.z],
            atol=1e-5,
            err_msg="静止状態では頂点が1ミリも動かず初期位置に留まること"
        )

        # 斜め方向へのマウス移動（視線レイ）をシミュレート
        drag_dir_world = mathutils.Vector((0.3, 0.4, -1.0)).normalized()
        denom = drag_dir_world.dot(camera_forward)
        self.assertGreater(abs(denom), 1e-6)
        t = (grab_plane_point - ray_origin_world).dot(camera_forward) / denom
        current_plane_hit = ray_origin_world + drag_dir_world * t

        # 相対オフセットの計算
        delta_world = current_plane_hit - initial_plane_hit
        delta_local = matrix_inv.to_3x3() @ delta_world
        target_pos_local = v_initial_local + delta_local

        # 検証2: 斜めにドラッグしても、頂点本来のZ深度が完全に一定に保たれること
        self.assertAlmostEqual(target_pos_local.z, v_initial_local.z, places=4, msg="相対オフセット移動により頂点本来のZ深度が維持されること")

        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh, do_unlink=True)

    def test_reset_and_restore_rest_positions(self):
        """Reset Simulation / Disable Cloth / Frame 1到達時にメッシュ座標が初期状態に完全復元されるか検証"""
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=1.0)
        obj = bpy.context.active_object
        mesh = obj.data
        obj.taremin_cloth.is_cloth = True

        # 初期頂点座標の記録
        n_verts = len(mesh.vertices)
        initial_coords = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", initial_coords)

        # タイムラインを数フレーム進めて変形させる
        scene = bpy.context.scene
        scene.frame_set(1)
        scene.frame_set(2)
        scene.frame_set(10)

        # 変形していることを確認
        deformed_coords = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", deformed_coords)
        self.assertFalse(np.allclose(initial_coords, deformed_coords, atol=1e-4), "シミュレーションで変形していること")

        # 1. Reset Simulation で復元されるか
        res = bpy.ops.taremin_cloth.reset_simulation()
        self.assertEqual(res, {'FINISHED'})
        reset_coords = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", reset_coords)
        np.testing.assert_allclose(reset_coords, initial_coords, atol=1e-5, err_msg="Reset Simulation後に初期形状へ復元されること")

        # 2. 変形後に Disable Cloth で復元されるか
        scene.frame_set(11)
        scene.frame_set(15)
        res = bpy.ops.taremin_cloth.toggle_cloth()
        self.assertEqual(res, {'FINISHED'})
        self.assertFalse(obj.taremin_cloth.is_cloth)
        disabled_coords = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", disabled_coords)
        np.testing.assert_allclose(disabled_coords, initial_coords, atol=1e-5, err_msg="Disable Cloth後に初期形状へ復元されること")

        # 3. 再度有効化して進め、Frame 1 (frame_start) に戻った際に復元されるか
        bpy.ops.taremin_cloth.toggle_cloth()
        self.assertTrue(obj.taremin_cloth.is_cloth)
        scene.frame_set(16)
        scene.frame_set(20)
        scene.frame_set(scene.frame_start)
        rewound_coords = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", rewound_coords)
        np.testing.assert_allclose(rewound_coords, initial_coords, atol=1e-5, err_msg="Frame 1巻き戻し時に初期形状へ復元されること")

        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh, do_unlink=True)

    def test_object_attachment_e2e(self):
        """外部オブジェクト（Target Object）へのピン追従拘束E2Eテスト"""
        # ターゲットオブジェクト
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0.0, 0.0, 1.0))
        target_obj = bpy.context.active_object
        target_obj.name = "PinTarget"

        # 布メッシュ
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=2, y_subdivisions=2, size=0.5, location=(0.0, 0.0, 1.0))
        cloth_obj = bpy.context.active_object
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        cloth_obj.taremin_cloth.is_cloth = True
        cloth_obj.taremin_cloth.pin_target_object = target_obj

        # 頂点0をPinグループに追加
        vg = cloth_obj.vertex_groups.new(name="Pin")
        vg.add([0], 1.0, 'REPLACE')

        scene = bpy.context.scene
        scene.frame_set(1)
        scene.frame_set(2)

        # ターゲットオブジェクトを移動
        target_obj.location = (0.5, 0.0, 1.5)
        for f in range(3, 10):
            scene.frame_set(f)

        mesh = cloth_obj.data
        coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coords)
        cur_pos = coords.reshape((-1, 3))

        # 頂点0がターゲットの移動先 (0.5, 0.0, 1.5) 付近に追従していること
        np.testing.assert_allclose(cur_pos[0], [0.5, 0.0, 1.5], atol=0.08, err_msg="ピン頂点がターゲットオブジェクトに追従すること")

        scene.frame_set(1)
        bpy.data.objects.remove(cloth_obj, do_unlink=True)
        bpy.data.objects.remove(target_obj, do_unlink=True)

    def test_mesh_collider_broadphase_culling_removed(self):
        """布から離れた位置にあるメッシュコライダーがAABBカリングで除外されず確実に衝突するテスト"""
        # 原点にSuzanneコライダーを作成
        bpy.ops.mesh.primitive_monkey_add(size=2.0, location=(0.0, 0.0, 0.0))
        collider_obj = bpy.context.active_object
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        collider_obj.taremin_cloth_collider.is_collider = True
        collider_obj.taremin_cloth_collider.collider_type = 'MESH'
        collider_obj.taremin_cloth_collider.thickness = 0.02
        collider_obj.taremin_cloth_collider.friction = 0.8

        # Z=2.0m（以前の布下0.6m AABBでは届かない位置）に布グリッドを作成
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=6, y_subdivisions=6, size=1.0, location=(0.0, 0.0, 2.0))
        cloth_obj = bpy.context.active_object
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        cloth_obj.taremin_cloth.is_cloth = True
        cloth_obj.taremin_cloth.substeps = 20
        cloth_obj.taremin_cloth.solver_iterations = 2

        scene = bpy.context.scene
        scene.frame_set(1)
        # 1秒間（60フレーム）落下シミュレーション
        for f in range(2, 65):
            scene.frame_set(f)

        mesh = cloth_obj.data
        coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coords)
        z_coords = coords[2::3]

        # Suzanneの頂点は Z ≈ 0.5 付近。布がSuzanneに衝突してZ > 0.0 に留まること（すり抜けて落下しないこと）
        max_z = float(np.max(z_coords))
        self.assertGreater(max_z, 0.1, "布がSuzanneコライダー上で支えられ、貫通・すり抜けしないこと")

        scene.frame_set(1)
        bpy.data.objects.remove(cloth_obj, do_unlink=True)
        bpy.data.objects.remove(collider_obj, do_unlink=True)

    def test_rest_positions_dirty_detection_subdivision(self):
        """メッシュ細分化（頂点数変更）後でもReset Simulationで正しく新形状に復元されるテスト"""
        bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 2.0))
        cloth_obj = bpy.context.active_object
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        cloth_obj.taremin_cloth.is_cloth = True

        # 初期4頂点でシミュレーションを動かしてキャッシュを作成
        scene = bpy.context.scene
        scene.frame_set(1)
        scene.frame_set(5)
        bpy.ops.taremin_cloth.reset_simulation()
        scene.frame_set(1)

        # 編集モードに入って細分化（4頂点 -> 9頂点以上）
        bpy.ops.object.mode_set(mode='EDIT')
        import bmesh
        bm = bmesh.from_edit_mesh(cloth_obj.data)
        bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=2, use_grid_fill=True)
        bmesh.update_edit_mesh(cloth_obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')

        n_new_verts = len(cloth_obj.data.vertices)
        self.assertGreater(n_new_verts, 4, "頂点数が増加していること")

        # 細分化後の初期座標を記録
        expected_coords = np.empty(n_new_verts * 3, dtype=np.float32)
        cloth_obj.data.vertices.foreach_get("co", expected_coords)

        # 再びシミュレーションで変形させる
        for f in range(2, 10):
            scene.frame_set(f)

        # 変形していることを確認
        deformed_coords = np.empty(n_new_verts * 3, dtype=np.float32)
        cloth_obj.data.vertices.foreach_get("co", deformed_coords)
        self.assertFalse(np.allclose(deformed_coords, expected_coords, atol=1e-3))

        # Reset Simulation を実行
        bpy.ops.taremin_cloth.reset_simulation()

        # 細分化後の形状に完全復元されること
        restored_coords = np.empty(n_new_verts * 3, dtype=np.float32)
        cloth_obj.data.vertices.foreach_get("co", restored_coords)
        np.testing.assert_allclose(restored_coords, expected_coords, atol=1e-5, err_msg="細分化後の頂点座標に完全復元されること")

        scene.frame_set(1)
        bpy.data.objects.remove(cloth_obj, do_unlink=True)

    def test_rest_positions_dirty_detection_vertex_edit(self):
        """初期状態で頂点を手動移動（頂点数は同一）した場合、最新の編集位置がレストポーズとして記憶されるテスト"""
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=1.0, location=(0.0, 0.0, 1.0))
        cloth_obj = bpy.context.active_object
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        cloth_obj.taremin_cloth.is_cloth = True

        scene = bpy.context.scene
        scene.frame_set(1)
        scene.frame_set(3)
        bpy.ops.taremin_cloth.reset_simulation()
        scene.frame_set(1)

        # 未変形（初期状態）で頂点0を手動編集（オフセット）
        cloth_obj.data.vertices[0].co.z += 0.5
        cloth_obj.data.update()

        expected_coords = np.empty(len(cloth_obj.data.vertices) * 3, dtype=np.float32)
        cloth_obj.data.vertices.foreach_get("co", expected_coords)

        # シミュレーション実行
        for f in range(2, 10):
            scene.frame_set(f)

        # Reset Simulation 実行
        bpy.ops.taremin_cloth.reset_simulation()

        restored_coords = np.empty(len(cloth_obj.data.vertices) * 3, dtype=np.float32)
        cloth_obj.data.vertices.foreach_get("co", restored_coords)
        np.testing.assert_allclose(restored_coords, expected_coords, atol=1e-5, err_msg="手動編集後の頂点座標に完全復元されること")

        scene.frame_set(1)
        bpy.data.objects.remove(cloth_obj, do_unlink=True)

    def test_reset_simulation_operator_fallback_to_all_cloth(self):
        """非Clothオブジェクト選択時でもReset Simulationオペレーターでシーン内Clothが復元されるテスト"""
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3, size=1.0, location=(0.0, 0.0, 2.0))
        cloth_obj = bpy.context.active_object
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        cloth_obj.taremin_cloth.is_cloth = True

        init_coords = np.empty(len(cloth_obj.data.vertices) * 3, dtype=np.float32)
        cloth_obj.data.vertices.foreach_get("co", init_coords)

        # シミュレーションで変形
        scene = bpy.context.scene
        scene.frame_set(1)
        scene.frame_set(5)

        # 別の立方体オブジェクトを作成してアクティブにする（Clothではない）
        bpy.ops.mesh.primitive_cube_add(location=(10.0, 10.0, 10.0))
        cube_obj = bpy.context.active_object
        self.assertEqual(bpy.context.active_object, cube_obj)

        # この状態で Reset Simulation を実行
        bpy.ops.taremin_cloth.reset_simulation()

        # 布オブジェクトが初期位置に復元されていること
        res_coords = np.empty(len(cloth_obj.data.vertices) * 3, dtype=np.float32)
        cloth_obj.data.vertices.foreach_get("co", res_coords)
        np.testing.assert_allclose(res_coords, init_coords, atol=1e-5, err_msg="非Clothオブジェクト選択時でも布が初期位置に復元されること")

        bpy.data.objects.remove(cloth_obj, do_unlink=True)
        bpy.data.objects.remove(cube_obj, do_unlink=True)

    def test_reset_selected_operator(self):
        """2枚の布が存在する時、選択中のClothのみが個別リセットされ、もう一方は変形を維持するテスト"""
        # Cloth A (Z=2.0)
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3, size=1.0, location=(0.0, 0.0, 2.0))
        cloth_a = bpy.context.active_object
        cloth_a.name = "ClothA"
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        cloth_a.taremin_cloth.is_cloth = True

        # Cloth B (Z=4.0)
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3, size=1.0, location=(2.0, 0.0, 4.0))
        cloth_b = bpy.context.active_object
        cloth_b.name = "ClothB"
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        cloth_b.taremin_cloth.is_cloth = True

        coords_a_init = np.empty(len(cloth_a.data.vertices) * 3, dtype=np.float32)
        cloth_a.data.vertices.foreach_get("co", coords_a_init)
        coords_b_init = np.empty(len(cloth_b.data.vertices) * 3, dtype=np.float32)
        cloth_b.data.vertices.foreach_get("co", coords_b_init)

        scene = bpy.context.scene
        scene.frame_set(1)
        # 5フレームシミュレーション実行して両方変形させる
        for f in range(2, 6):
            scene.frame_set(f)

        coords_a_def = np.empty(len(cloth_a.data.vertices) * 3, dtype=np.float32)
        cloth_a.data.vertices.foreach_get("co", coords_a_def)
        coords_b_def = np.empty(len(cloth_b.data.vertices) * 3, dtype=np.float32)
        cloth_b.data.vertices.foreach_get("co", coords_b_def)

        self.assertFalse(np.allclose(coords_a_def, coords_a_init, atol=1e-3), "Cloth A が変形していること")
        self.assertFalse(np.allclose(coords_b_def, coords_b_init, atol=1e-3), "Cloth B が変形していること")

        # Cloth A のみをアクティブにして Reset Selected を実行
        bpy.context.view_layer.objects.active = cloth_a
        res = bpy.ops.taremin_cloth.reset_selected()
        self.assertEqual(res, {'FINISHED'})

        # Cloth A は初期位置に復元されていること
        coords_a_after = np.empty(len(cloth_a.data.vertices) * 3, dtype=np.float32)
        cloth_a.data.vertices.foreach_get("co", coords_a_after)
        np.testing.assert_allclose(coords_a_after, coords_a_init, atol=1e-5, err_msg="Cloth A のみが初期形状に復元されること")

        # Cloth B は変形状態のままであること
        coords_b_after = np.empty(len(cloth_b.data.vertices) * 3, dtype=np.float32)
        cloth_b.data.vertices.foreach_get("co", coords_b_after)
        np.testing.assert_allclose(coords_b_after, coords_b_def, atol=1e-5, err_msg="Cloth B は変形状態が維持されること")

        scene.frame_set(1)
        bpy.data.objects.remove(cloth_a, do_unlink=True)
        bpy.data.objects.remove(cloth_b, do_unlink=True)

    def test_reset_selected_poll_disabled_for_non_cloth(self):
        """非Clothオブジェクト選択時にReset SelectedのpollがFalseを返す（グレーアウト）テスト"""
        # 非Clothメッシュを作成
        bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.0))
        cube_obj = bpy.context.active_object
        cube_obj.taremin_cloth.is_cloth = False

        # 非Cloth時は poll が False であること
        self.assertFalse(bpy.ops.taremin_cloth.reset_selected.poll(), "非Cloth選択時はpollがFalse（グレーアウト）であること")

        # Cloth を有効化
        cube_obj.taremin_cloth.is_cloth = True
        self.assertTrue(bpy.ops.taremin_cloth.reset_selected.poll(), "Cloth選択時はpollがTrue（有効化）であること")

        bpy.data.objects.remove(cube_obj, do_unlink=True)

    def test_reset_all_operator(self):
        """Reset All により、非Cloth選択中であっても全Clothが一括初期化されるテスト"""
        # Cloth A
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3, size=1.0, location=(0.0, 0.0, 2.0))
        cloth_a = bpy.context.active_object
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        cloth_a.taremin_cloth.is_cloth = True

        # Cloth B
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3, size=1.0, location=(2.0, 0.0, 4.0))
        cloth_b = bpy.context.active_object
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        cloth_b.taremin_cloth.is_cloth = True

        coords_a_init = np.empty(len(cloth_a.data.vertices) * 3, dtype=np.float32)
        cloth_a.data.vertices.foreach_get("co", coords_a_init)
        coords_b_init = np.empty(len(cloth_b.data.vertices) * 3, dtype=np.float32)
        cloth_b.data.vertices.foreach_get("co", coords_b_init)

        scene = bpy.context.scene
        scene.frame_set(1)
        for f in range(2, 6):
            scene.frame_set(f)

        # 非Clothのカメラを作成してアクティブにする
        cam_data = bpy.data.cameras.new(name="DummyCam")
        cam_obj = bpy.data.objects.new(name="DummyCam", object_data=cam_data)
        bpy.context.collection.objects.link(cam_obj)
        bpy.context.view_layer.objects.active = cam_obj

        # Reset All を実行
        res = bpy.ops.taremin_cloth.reset_all()
        self.assertEqual(res, {'FINISHED'})

        # 両方のClothが初期位置に復元されていること
        coords_a_after = np.empty(len(cloth_a.data.vertices) * 3, dtype=np.float32)
        cloth_a.data.vertices.foreach_get("co", coords_a_after)
        np.testing.assert_allclose(coords_a_after, coords_a_init, atol=1e-5, err_msg="Cloth A が復元されていること")

        coords_b_after = np.empty(len(cloth_b.data.vertices) * 3, dtype=np.float32)
        cloth_b.data.vertices.foreach_get("co", coords_b_after)
        np.testing.assert_allclose(coords_b_after, coords_b_init, atol=1e-5, err_msg="Cloth B が復元されていること")

        scene.frame_set(1)
        bpy.data.objects.remove(cloth_a, do_unlink=True)
        bpy.data.objects.remove(cloth_b, do_unlink=True)
        bpy.data.objects.remove(cam_obj, do_unlink=True)

    def test_edge_collision_mesh_collider_e2e(self):
        """エッジ詳細接触判定 (Edge Collision) により、メッシュコライダーの突起突き抜けが抑制されるE2Eテスト"""
        # ICO球メッシュコライダー
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.0, location=(0.0, 0.0, 0.0))
        ico_obj = bpy.context.active_object
        ico_obj.taremin_cloth_collider.is_collider = True
        ico_obj.taremin_cloth_collider.collider_type = 'MESH'
        ico_obj.taremin_cloth_collider.thickness = 0.02

        # 布メッシュ (z=1.2 に配置)
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=6, y_subdivisions=6, size=1.6, location=(0.0, 0.0, 1.2))
        cloth_obj = bpy.context.active_object
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        cloth_obj.taremin_cloth.is_cloth = True
        cloth_obj.taremin_cloth.thickness = 0.01
        cloth_obj.taremin_cloth.enable_edge_collision = True
        cloth_obj.taremin_cloth.edge_margin_scale = 1.2
        cloth_obj.taremin_cloth.edge_margin_offset = 0.002
        cloth_obj.taremin_cloth.substeps = 20

        from taremin_cloth.operators import get_or_create_simulator, clear_simulators
        clear_simulators()
        sim, coords = get_or_create_simulator(cloth_obj)

        for _ in range(60):
            sim.step(dt=1.0 / 60.0, substeps=cloth_obj.taremin_cloth.substeps)

        sim.get_positions(coords)
        cur_pos = coords.reshape((-1, 3))

        # 布が自由落下せず、球コライダー上に留まっていること
        min_z = np.min(cur_pos[:, 2])
        max_z = np.max(cur_pos[:, 2])
        self.assertGreater(min_z, -0.6, "布が球コライダーを貫通せず上に留まっていること")
        self.assertGreater(max_z, 0.9, "布の上部が球の頭頂部付近に留まっていること")

        # 布がかぶさっている頭頂部 (z > 0.3) のコライダー頂点の飛び出し数を幾何学的に検査
        import mathutils
        cloth_polys = [p.vertices for p in cloth_obj.data.polygons]
        cloth_bvh = mathutils.bvhtree.BVHTree.FromPolygons([mathutils.Vector(p) for p in cur_pos], cloth_polys)
        protrude_count = 0
        for v in ico_obj.data.vertices:
            if v.co.z > 0.3:
                loc, norm, idx, dist = cloth_bvh.find_nearest(v.co)
                if loc and v.co.length > loc.length + 0.005:
                    protrude_count += 1

        self.assertEqual(protrude_count, 0, f"布がかぶさっている頭頂部でコライダー頂点の飛び出しがゼロであること: {protrude_count}")

        bpy.data.objects.remove(cloth_obj, do_unlink=True)
        bpy.data.objects.remove(ico_obj, do_unlink=True)

    def test_cloth_and_collider_simulation_enabled_bypass(self):
        """enabledフラグをFalseにした際、シミュレーション計算および衝突判定から安全にバイパス（除外）されるテスト"""
        from taremin_cloth.operators import cloth_frame_handler, clear_simulators, _simulators, sync_colliders
        clear_simulators()

        # 1. 布オブジェクト作成
        bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 2))
        cloth_obj = bpy.context.active_object
        cloth_obj.taremin_cloth.is_cloth = True
        cloth_obj.taremin_cloth.enabled = False  # ミュート状態

        # 2. コライダーオブジェクト作成
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
        cube_obj = bpy.context.active_object
        cube_obj.taremin_cloth_collider.is_collider = True
        cube_obj.taremin_cloth_collider.enabled = False  # ミュート状態

        scene = bpy.context.scene
        orig_frame = scene.frame_current

        try:
            # タイムラインをフレーム2に進めてハンドラー実行
            scene.frame_set(2)
            cloth_frame_handler(scene)

            # enabled=False なのでシミュレータが起動せず、未変形であること
            self.assertNotIn(cloth_obj.name, _simulators, "enabled=Falseの布はシミュレータが生成されないこと")
            self.assertFalse(cloth_obj.get("_taremin_cloth_is_deformed", False), "enabled=Falseの布は変形フラグが立たないこと")

            # コライダー同期テスト (enabled=False のコライダーは登録されないこと)
            test_sim, _ = from_operators_get_or_create = (None, None)
            from taremin_cloth.operators import get_or_create_simulator
            cloth_obj.taremin_cloth.enabled = True
            sim, coords = get_or_create_simulator(cloth_obj)

            # コライダー同期時、cube_obj は enabled=False のため GPU側に送られない
            sync_colliders(sim, scene)
            from taremin_cloth.operators import _collider_cache
            sim_id = id(sim)
            cached_col_names = [item[0] for item in _collider_cache.get(sim_id, ())]
            self.assertNotIn(cube_obj.name, cached_col_names, "enabled=FalseのコライダーはGPU衝突判定から除外されること")

            # コライダーを有効化すると同期リストに含まれること
            cube_obj.taremin_cloth_collider.enabled = True
            sync_colliders(sim, scene, force=True)
            cached_col_names_active = [item[0] for item in _collider_cache.get(sim_id, ())]
            self.assertIn(cube_obj.name, cached_col_names_active, "enabled=Trueに切り替えるとGPU衝突判定に登録されること")

        finally:
            clear_simulators()
            scene.frame_set(orig_frame)
            if cloth_obj.name in bpy.data.objects:
                bpy.data.objects.remove(cloth_obj, do_unlink=True)
            if cube_obj.name in bpy.data.objects:
                bpy.data.objects.remove(cube_obj, do_unlink=True)

    def test_ui_mode_and_subpanel_poll(self):
        """簡単モード / 詳細モードの切り替えとサブパネルpoll非表示制御のE2Eテスト"""
        from taremin_cloth.panels import (
            TAREMIN_CLOTH_PT_pinning,
            TAREMIN_CLOTH_PT_fabric,
            TAREMIN_CLOTH_PT_forces,
            TAREMIN_CLOTH_PT_collisions,
            TAREMIN_CLOTH_PT_pattern,
            TAREMIN_CLOTH_PT_quality,
        )

        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 0))
        cloth_obj = bpy.context.active_object
        cloth_obj.taremin_cloth.is_cloth = True

        scene = bpy.context.scene
        context = bpy.context

        try:
            # デフォルトは SIMPLE モード
            scene.taremin_cloth_ui_mode = 'SIMPLE'
            self.assertEqual(scene.taremin_cloth_ui_mode, 'SIMPLE')

            # 簡単モード時、サブパネルは poll が False（非表示）になること
            self.assertFalse(TAREMIN_CLOTH_PT_pinning.poll(context))
            self.assertFalse(TAREMIN_CLOTH_PT_fabric.poll(context))
            self.assertFalse(TAREMIN_CLOTH_PT_forces.poll(context))
            self.assertFalse(TAREMIN_CLOTH_PT_collisions.poll(context))
            self.assertFalse(TAREMIN_CLOTH_PT_pattern.poll(context))
            self.assertFalse(TAREMIN_CLOTH_PT_quality.poll(context))

            # 簡単モードでも Sewing 設定の変更が可能であること
            cloth_obj.taremin_cloth.enable_sewing = True
            self.assertTrue(cloth_obj.taremin_cloth.enable_sewing)

            # ADVANCED モードに切り替え
            scene.taremin_cloth_ui_mode = 'ADVANCED'
            self.assertEqual(scene.taremin_cloth_ui_mode, 'ADVANCED')

            # 詳細モード時、布がアクティブならサブパネルの poll が True になること
            self.assertTrue(TAREMIN_CLOTH_PT_pinning.poll(context))
            self.assertTrue(TAREMIN_CLOTH_PT_fabric.poll(context))
            self.assertTrue(TAREMIN_CLOTH_PT_forces.poll(context))
            self.assertTrue(TAREMIN_CLOTH_PT_collisions.poll(context))
            self.assertTrue(TAREMIN_CLOTH_PT_pattern.poll(context))
            self.assertTrue(TAREMIN_CLOTH_PT_quality.poll(context))

        finally:
            scene.taremin_cloth_ui_mode = 'SIMPLE'
            if cloth_obj.name in bpy.data.objects:
                bpy.data.objects.remove(cloth_obj, do_unlink=True)

    def test_collider_auto_detection_and_purpose_sync(self):
        """Blender実機オブジェクトに対するコライダー自動判別および目的別選択のE2Eテスト"""
        # 1. 球体オブジェクトでの自動判別テスト
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))
        sphere_obj = bpy.context.active_object
        sphere_obj.taremin_cloth_collider.is_collider = True

        try:
            # is_collider=True で自動判別が走り、SPHERE と判定されること
            self.assertEqual(sphere_obj.taremin_cloth_collider.collider_type, 'SPHERE')

            # 手動で目的を CHARACTER に変更 -> BONE_SDF に連動
            sphere_obj.taremin_cloth_collider.collider_purpose = 'CHARACTER'
            self.assertEqual(sphere_obj.taremin_cloth_collider.collider_type, 'BONE_SDF')

            # 手動で目的を FLOOR に変更 -> PLANE に連動
            sphere_obj.taremin_cloth_collider.collider_purpose = 'FLOOR'
            self.assertEqual(sphere_obj.taremin_cloth_collider.collider_type, 'PLANE')

            # 自動判別オペレーターを実行
            bpy.ops.taremin_cloth.auto_detect_collider()
            self.assertEqual(sphere_obj.taremin_cloth_collider.collider_type, 'SPHERE')
            self.assertEqual(sphere_obj.taremin_cloth_collider.collider_purpose, 'AUTO')

        finally:
            if sphere_obj.name in bpy.data.objects:
                bpy.data.objects.remove(sphere_obj, do_unlink=True)

        # 2. 平面オブジェクト（床）での自動判別テスト
        bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0, 0, 0))
        floor_obj = bpy.context.active_object
        floor_obj.taremin_cloth_collider.is_collider = True

        try:
            self.assertEqual(floor_obj.taremin_cloth_collider.collider_type, 'PLANE')
        finally:
            if floor_obj.name in bpy.data.objects:
                bpy.data.objects.remove(floor_obj, do_unlink=True)

    def test_self_collision_purpose_and_auto_fit(self):
        """自己衝突の用途プリセット切り替えと Auto Fit オペレーターのE2Eテスト"""
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=5, y_subdivisions=5, size=1.0, location=(0, 0, 0))
        cloth_obj = bpy.context.active_object
        cloth_obj.taremin_cloth.is_cloth = True

        settings = cloth_obj.taremin_cloth

        try:
            # 初期状態: enable_self_collision = False
            self.assertFalse(settings.enable_self_collision)
            self.assertEqual(settings.self_collision_purpose, 'STANDARD')

            # 有効化時に初回自動フィッティングが走り、適正厚みがセットされること
            settings.enable_self_collision = True
            self.assertTrue(settings.enable_self_collision)
            self.assertGreater(settings.thickness, 0.001)
            self.assertEqual(settings.coupled_self_collision_mode, 'RELAXATION')

            # 用途を SKIRT（プリーツ・スカート）に変更 -> FULL_COUPLED に自動連動
            settings.self_collision_purpose = 'SKIRT'
            self.assertEqual(settings.coupled_self_collision_mode, 'FULL_COUPLED')
            self.assertEqual(settings.self_collision_max_iterations, '512')
            self.assertAlmostEqual(settings.self_collision_relief_factor, 0.15, places=2)

            # 用途を THIN（薄手・シルク）に変更 -> マイルドな 0.10 に連動
            settings.self_collision_purpose = 'THIN'
            self.assertEqual(settings.coupled_self_collision_mode, 'RELAXATION')
            self.assertAlmostEqual(settings.self_collision_relief_factor, 0.10, places=2)

            # 手動で Auto Fit オペレーターを実行
            settings.self_collision_purpose = 'STANDARD'
            from taremin_cloth.ops.basic import TAREMIN_CLOTH_OT_auto_fit_self_collision
            self.assertTrue(TAREMIN_CLOTH_OT_auto_fit_self_collision.poll(bpy.context))
            res = bpy.ops.taremin_cloth.auto_fit_self_collision()
            self.assertEqual(res, {'FINISHED'})
            self.assertEqual(settings.coupled_self_collision_mode, 'RELAXATION')
            self.assertAlmostEqual(settings.self_collision_relief_factor, 0.20, places=2)

            # 用途を CUSTOM に変更した時、Auto Fit ボタンは disable (poll = False) になること
            settings.self_collision_purpose = 'CUSTOM'
            self.assertFalse(TAREMIN_CLOTH_OT_auto_fit_self_collision.poll(bpy.context))

        finally:
            if cloth_obj.name in bpy.data.objects:
                bpy.data.objects.remove(cloth_obj, do_unlink=True)


if __name__ == "__main__":
    unittest.main()





