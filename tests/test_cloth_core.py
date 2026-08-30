import unittest
import numpy as np


class TestClothCoreBasic(unittest.TestCase):
    def test_import_module(self):
        """taremin_cloth_core モジュールがインポートできることを確認"""
        import taremin_cloth_core
        self.assertTrue(hasattr(taremin_cloth_core, "is_gpu_available"))
        self.assertTrue(hasattr(taremin_cloth_core, "get_gpu_device_name"))
        self.assertTrue(hasattr(taremin_cloth_core, "ClothSimulator"))

    def test_gpu_availability(self):
        """GPUが利用可能でありデバイス名が取得できることを確認"""
        import taremin_cloth_core
        is_avail = taremin_cloth_core.is_gpu_available()
        self.assertTrue(is_avail, "GPU Contextが利用可能であるべき")

        device_name = taremin_cloth_core.get_gpu_device_name()
        print(f"\n[Test] Detected GPU Device: {device_name}")
        self.assertIsInstance(device_name, str)
        self.assertGreater(len(device_name), 0)

    def test_gpu_devices_and_backend_selection(self):
        """GPUデバイスの列挙およびバックエンド切り替えAPIの動作テスト"""
        import taremin_cloth_core

        # 1. デバイス一覧の列挙
        devices = taremin_cloth_core.get_available_gpu_devices()
        self.assertIsInstance(devices, list)
        self.assertGreater(len(devices), 0, "少なくとも1つ以上のGPUデバイスが検出されるべき")
        for dev in devices:
            self.assertIn("index", dev)
            self.assertIn("name", dev)
            self.assertIn("backend", dev)
            self.assertIn("device_type", dev)

        # 2. 現在のデバイス情報取得
        current = taremin_cloth_core.get_current_gpu_device()
        self.assertIsInstance(current, dict)
        self.assertIn("name", current)
        self.assertIn("backend", current)

        # 3. バックエンド指定再初期化（Auto / DX12等）
        taremin_cloth_core.set_gpu_device("auto", None)
        reinit_info = taremin_cloth_core.get_current_gpu_device()
        self.assertEqual(current["name"], reinit_info["name"])

    def test_cloth_simulator_grid(self):
        """Python経由でのClothSimulatorの物理演算・健全性テスト"""
        import taremin_cloth_core

        nx, ny = 4, 4
        dx = 0.1
        positions = []
        inv_masses = []

        for y in range(ny):
            for x in range(nx):
                positions.append([x * dx, y * dx, 0.0])
                if y == 0 and (x == 0 or x == nx - 1):
                    inv_masses.append(0.0)
                else:
                    inv_masses.append(1.0)

        positions_np = np.array(positions, dtype=np.float32)
        inv_masses_np = np.array(inv_masses, dtype=np.float32)

        edges = []
        for y in range(ny):
            for x in range(nx):
                idx = y * nx + x
                if x + 1 < nx:
                    edges.append([idx, idx + 1])
                if y + 1 < ny:
                    edges.append([idx, idx + nx])
        edges_np = np.array(edges, dtype=np.uint32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions_np,
            edges=edges_np,
            inv_masses=inv_masses_np,
            layer_id=0,
            thickness=0.005,
            stiffness=10000.0,
        )

        self.assertEqual(sim.num_vertices, nx * ny)
        self.assertEqual(sim.num_distance_constraints, len(edges))

        for _ in range(60):
            sim.step(dt=1.0 / 60.0, substeps=30)

        out_coords = np.zeros(nx * ny * 3, dtype=np.float32)
        sim.get_positions(out_coords)
        out_positions = out_coords.reshape((nx * ny, 3))

        # 1. NaN / Inf の完全不在
        self.assertTrue(np.all(np.isfinite(out_positions)), "頂点座標に NaN または Inf が含まれてはならない")

        # 2. 固定ピンが初期位置を維持
        np.testing.assert_allclose(out_positions[0], positions_np[0], atol=1e-5)
        np.testing.assert_allclose(out_positions[nx - 1], positions_np[nx - 1], atol=1e-5)

        # 3. 自由頂点が重力で落下していること
        self.assertLess(out_positions[-1, 2], -0.01, "布の下端が落下していること")

        # 4. エッジ長誤差が許容値（5%以内）
        max_error = 0.0
        for v0, v1 in edges:
            p0 = out_positions[v0]
            p1 = out_positions[v1]
            cur_len = np.linalg.norm(p0 - p1)

            orig_p0 = positions_np[v0]
            orig_p1 = positions_np[v1]
            rest_len = np.linalg.norm(orig_p0 - orig_p1)

            err = abs(cur_len - rest_len) / rest_len
            if err > max_error:
                max_error = err

        print(f"[Test] Max Edge Stretch Error: {max_error * 100:.2f}%")
        self.assertLess(max_error, 0.05, f"エッジ伸縮誤差が5%以内であるべき (実測: {max_error * 100:.2f}%)")

        # 5. リセット検証
        sim.reset()
        reset_coords = np.zeros(nx * ny * 3, dtype=np.float32)
        sim.get_positions(reset_coords)
        reset_positions = reset_coords.reshape((nx * ny, 3))
        np.testing.assert_allclose(reset_positions, positions_np, atol=1e-6)

    def test_bending_stiffness_effect(self):
        """曲げ剛性パラメータの大小による変形抑制効果の検証テスト"""
        import taremin_cloth_core

        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.5, 0.0],
            [1.0, 0.5, 0.0],
        ], dtype=np.float32)

        edges = np.array([
            [0, 1],
            [0, 2], [1, 2],
            [0, 3], [1, 3],
        ], dtype=np.uint32)

        faces = np.array([
            [0, 1, 2],
            [1, 0, 3],
        ], dtype=np.uint32)

        inv_masses = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        sim_soft = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            stiffness=10000.0,
            bending_stiffness=0.0,
        )
        self.assertEqual(sim_soft.num_bending_constraints, 1)

        for _ in range(60):
            sim_soft.step(dt=1.0 / 60.0, substeps=30)

        out_soft = np.zeros(4 * 3, dtype=np.float32)
        sim_soft.get_positions(out_soft)
        pos_soft = out_soft.reshape((4, 3))

        sim_stiff = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            stiffness=10000.0,
            bending_stiffness=500.0,
        )
        for _ in range(60):
            sim_stiff.step(dt=1.0 / 60.0, substeps=30)

        out_stiff = np.zeros(4 * 3, dtype=np.float32)
        sim_stiff.get_positions(out_stiff)
        pos_stiff = out_stiff.reshape((4, 3))

        drop_soft = pos_soft[3, 2]
        drop_stiff = pos_stiff[3, 2]
        print(f"\n[Test Bending] Drop Soft: {drop_soft:.4f}, Drop Stiff: {drop_stiff:.4f}")
        self.assertLess(drop_soft, drop_stiff, "曲げ剛性が高い方が垂れ下がりが抑制されるべき")

    def test_dynamic_pin_python(self):
        """Python経由での動的ピン操作 (set_pin, release_pin) の検証テスト"""
        import taremin_cloth_core

        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        edges = np.array([[0, 1]], dtype=np.uint32)
        inv_masses = np.array([0.0, 1.0], dtype=np.float32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            inv_masses=inv_masses,
            stiffness=10000.0,
        )

        target = [2.0, 1.0, 0.5]
        sim.set_pin(1, target, 1.0)

        for _ in range(30):
            sim.step(dt=1.0 / 60.0, substeps=20)

        out = np.zeros(2 * 3, dtype=np.float32)
        sim.get_positions(out)
        pos = out.reshape((2, 3))

        np.testing.assert_allclose(pos[1], target, atol=0.05)

        sim.release_pin(1)
        for _ in range(30):
            sim.step(dt=1.0 / 60.0, substeps=20)

        sim.get_positions(out)
        pos_after = out.reshape((2, 3))
        self.assertLess(pos_after[1, 2], target[2] - 0.05, "ピン解除後に落下すること")

    def test_sewing_python(self):
        """Python経由での縫合拘束 (Sewing Springs) の収縮検証テスト"""
        import taremin_cloth_core

        positions = np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        edges = np.empty((0, 2), dtype=np.uint32)
        sewing_springs = np.array([[0, 1]], dtype=np.uint32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            sewing_springs=sewing_springs,
            stiffness=10000.0,
            sewing_shrink_speed=5.0,
        )
        sim.set_gravity(0.0, 0.0, 0.0)
        self.assertEqual(sim.num_sewing_constraints, 1)

        for _ in range(60):
            sim.step(dt=1.0 / 60.0, substeps=20)

        out = np.zeros(2 * 3, dtype=np.float32)
        sim.get_positions(out)
        pos = out.reshape((2, 3))

        final_dist = np.linalg.norm(pos[0] - pos[1])
        print(f"\n[Test Sewing Python] Final Dist: {final_dist:.4f}")
        self.assertLess(final_dist, 0.05, f"縫合により距離が0付近に収縮すること (実測: {final_dist:.4f})")

    def test_collision_python(self):
        """Python経由での球・平面コライダーとの衝突・非貫通検証テスト"""
        import taremin_cloth_core

        positions = np.array([
            [-0.2, -0.2, 1.0],
            [0.2, -0.2, 1.0],
            [-0.2, 0.2, 1.0],
            [0.2, 0.2, 1.0],
        ], dtype=np.float32)
        edges = np.array([[0, 1], [0, 2], [1, 3], [2, 3]], dtype=np.uint32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            thickness=0.01,
            stiffness=10000.0,
        )

        sphere_center = [0.0, 0.0, 0.0]
        sphere_r = 0.5
        sim.add_sphere_collider(sphere_center, sphere_r, friction=0.3)
        sim.add_plane_collider([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], friction=0.2)

        for _ in range(60):
            sim.step(dt=1.0 / 60.0, substeps=20)

        out = np.zeros(4 * 3, dtype=np.float32)
        sim.get_positions(out)
        cur_pos = out.reshape((4, 3))

        for i, p in enumerate(cur_pos):
            dist = np.linalg.norm(p - np.array(sphere_center))
            self.assertGreaterEqual(dist, sphere_r - 1e-4, f"頂点 {i} が球内部に貫通している (dist={dist})")
            self.assertGreaterEqual(p[2], -1e-4, f"頂点 {i} が床平面を貫通している (z={p[2]})")

    def test_mesh_collider_python(self):
        """Python経由での三角形メッシュコライダー衝突テスト"""
        import taremin_cloth_core

        positions = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
        edges = np.empty((0, 2), dtype=np.uint32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            thickness=0.02,
            stiffness=10000.0,
        )

        # 水平な2枚の三角形メッシュコライダー (z=0)
        triangles = np.array([
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]],
            [[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
        ], dtype=np.float32)
        sim.set_mesh_collider_triangles(triangles, friction=0.2, thickness=0.02)

        for _ in range(60):
            sim.step(dt=1.0 / 60.0, substeps=20)

        out = np.zeros(1 * 3, dtype=np.float32)
        sim.get_positions(out)
        pos = out.reshape((1, 3))

        self.assertGreaterEqual(pos[0, 2], 0.04 - 1e-3, f"頂点がメッシュコライダー上に留まること (z={pos[0, 2]})")

    def test_mesh_collider_single_sided_escape(self):
        """メッシュコライダーのSingle Sided（片面衝突）により、裏側からの頂点脱出が許可されるテスト"""
        import taremin_cloth_core

        # 水平な2枚の三角形メッシュコライダー (z=0, 法線は+Z方向)
        triangles = np.array([
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]],
            [[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
        ], dtype=np.float32)

        # 1. Single Sided = True (片面判定): 裏側から表側へ素通りして上昇・脱出できること
        sim_single = taremin_cloth_core.ClothSimulator(
            positions=np.array([[0.0, 0.0, -0.02]], dtype=np.float32),
            edges=np.empty((0, 2), dtype=np.uint32),
            thickness=0.02,
            stiffness=10000.0,
        )
        sim_single.set_gravity(0.0, 0.0, 9.8)  # 上向き重力で脱出を試みる
        sim_single.set_mesh_collider_triangles(triangles, friction=0.2, thickness=0.02, single_sided=True)

        for _ in range(30):
            sim_single.step(dt=1.0 / 60.0, substeps=20)

        out_single = np.zeros(3, dtype=np.float32)
        sim_single.get_positions(out_single)
        self.assertGreater(out_single[2], 0.05, f"Single Sided有効時、裏側から表側へ脱出できること (z={out_single[2]})")

        # 2. Single Sided = True でも、表側からの落下は表面で確実に留まること
        sim_front = taremin_cloth_core.ClothSimulator(
            positions=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
            edges=np.empty((0, 2), dtype=np.uint32),
            thickness=0.02,
            stiffness=10000.0,
        )
        sim_front.set_gravity(0.0, 0.0, -9.8)  # 通常重力
        sim_front.set_mesh_collider_triangles(triangles, friction=0.2, thickness=0.02, single_sided=True)

        for _ in range(60):
            sim_front.step(dt=1.0 / 60.0, substeps=20)

        out_front = np.zeros(3, dtype=np.float32)
        sim_front.get_positions(out_front)
        self.assertGreaterEqual(out_front[2], 0.04 - 1e-3, f"Single Sided有効時でも、表側からの落下は表面で留まること (z={out_front[2]})")

    def test_multilayer_collision_python(self):
        """Python経由でのマルチレイヤー衝突（インナー vs アウター）非貫通テスト"""
        import taremin_cloth_core

        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.3]], dtype=np.float32)
        edges = np.empty((0, 2), dtype=np.uint32)
        inv_masses = np.array([0.0, 1.0], dtype=np.float32)
        layer_ids = np.array([0, 1], dtype=np.uint32)
        thicknesses = np.array([0.05, 0.05], dtype=np.float32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            inv_masses=inv_masses,
            layer_ids=layer_ids,
            thicknesses=thicknesses,
            stiffness=10000.0,
        )

        for _ in range(60):
            sim.step(dt=1.0 / 60.0, substeps=20)

        out = np.zeros(2 * 3, dtype=np.float32)
        sim.get_positions(out)
        pos = out.reshape((2, 3))

        dist = pos[1, 2] - pos[0, 2]
        print(f"\n[Test MultiLayer Python] Dist: {dist:.4f}")
        self.assertGreaterEqual(dist, 0.10 - 1e-3, f"アウターがインナーの上に留まること (実測: {dist})")

    def test_dynamic_parameters_python(self):
        """シミュレーション実行中の動的パラメータ変更 (剛性、重力、減衰) を検証"""
        import taremin_cloth_core

        positions = np.array([
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.2, 0.0, 1.0],
        ], dtype=np.float32)
        edges = np.array([[0, 1], [1, 2]], dtype=np.uint32)
        inv_masses = np.array([0.0, 1.0, 1.0], dtype=np.float32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            inv_masses=inv_masses,
            stiffness=100.0,
            bending_stiffness=1.0,
        )

        for _ in range(10):
            sim.step(dt=1.0 / 60.0, substeps=20)

        # 動的パラメータ変更APIのテスト
        sim.set_stiffness(50000.0, 200.0)
        sim.set_damping(0.1)
        sim.set_gravity(0.0, 0.0, -20.0)

        for _ in range(10):
            sim.step(dt=1.0 / 60.0, substeps=20)

        out = np.zeros(3 * 3, dtype=np.float32)
        sim.get_positions(out)
        self.assertTrue(np.all(np.isfinite(out)), "パラメータ変更後も有限値が維持されること")

    def test_solver_iterations_stretch_reduction(self):
        """ソルバー反復 (Solver Iterations) によるチェーン拘束の伸び抑制を検証"""
        import taremin_cloth_core

        # 10個の頂点が縦に連なった細長いチェーン（重力下で伸びやすい構造）
        n_verts = 10
        dx = 0.05
        positions = np.zeros((n_verts, 3), dtype=np.float32)
        positions[:, 2] = -np.arange(n_verts) * dx  # 縦に吊るす
        edges = np.array([[i, i + 1] for i in range(n_verts - 1)], dtype=np.uint32)
        inv_masses = np.ones(n_verts, dtype=np.float32)
        inv_masses[0] = 0.0  # 一番上を固定

        # 1. 反復回数 1 (iterations=1)
        sim1 = taremin_cloth_core.ClothSimulator(
            positions=positions.copy(),
            edges=edges,
            inv_masses=inv_masses,
            stiffness=1000.0,
        )
        sim1.set_solver_iterations(1)
        for _ in range(30):
            sim1.step(dt=1.0 / 60.0, substeps=10, solver_iterations=1)

        out1 = np.zeros(n_verts * 3, dtype=np.float32)
        sim1.get_positions(out1)
        pos1 = out1.reshape((-1, 3))
        total_len1 = -pos1[-1, 2]  # 最下端のZ座標

        # 2. 反復回数 4 (iterations=4, stiffness=10000.0)
        sim4 = taremin_cloth_core.ClothSimulator(
            positions=positions.copy(),
            edges=edges,
            inv_masses=inv_masses,
            stiffness=10000.0,
        )
        sim4.set_solver_iterations(4)
        for _ in range(30):
            sim4.step(dt=1.0 / 60.0, substeps=10, solver_iterations=4)

        out4 = np.zeros(n_verts * 3, dtype=np.float32)
        sim4.get_positions(out4)
        pos4 = out4.reshape((-1, 3))
        total_len4 = -pos4[-1, 2]

        init_len = (n_verts - 1) * dx
        print(f"\n[Test Solver Iterations] Init Len: {init_len:.4f}, Len(iter=1): {total_len1:.4f}, Len(iter=4): {total_len4:.4f}")
        self.assertLess(total_len4, total_len1, "反復回数を増やし剛性を高めることで伸びが抑制されること")
        self.assertAlmostEqual(total_len4, init_len, delta=init_len * 0.05, msg="iter=4 では初期長の 5% 以内の伸びに抑えられること")

    def test_blender_native_stiffness_and_damping_parameters(self):
        """Blenderネイティブ準拠の剛性4種 (引張・圧縮・せん断・曲げ) および減衰4種の動作検証"""
        import taremin_cloth_core

        positions = np.array([
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.0, 0.1, 1.0],
            [0.1, 0.1, 1.0],
        ], dtype=np.float32)
        edges = np.array([[0, 1], [1, 3], [3, 2], [2, 0], [0, 3]], dtype=np.uint32)
        faces = np.array([[0, 1, 3], [0, 3, 2]], dtype=np.uint32)
        inv_masses = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)

        # 1. 剛性4種を指定して初期化 (引張1000, 圧縮100, せん断50, 曲げ10)
        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            stiffness=1000.0,
            compression_stiffness=100.0,
            shear_stiffness=50.0,
            bending_stiffness=10.0,
        )

        for _ in range(10):
            sim.step(dt=1.0 / 60.0, substeps=10)

        # 2. 動的な4種剛性更新 API
        sim.set_stiffness_all(
            tension_stiffness=50000.0,    # 完全非伸縮
            compression_stiffness=20.0,   # シワが寄りやすい
            shear_stiffness=200.0,
            bending_stiffness=5.0,
        )

        # 3. 動的な4種減衰更新 API (引張・圧縮・せん断・曲げ減衰)
        sim.set_damping_all(
            tension_damp=5.0,
            compression_damp=5.0,
            shear_damp=5.0,
            bending_damp=0.5,
        )

        for _ in range(20):
            sim.step(dt=1.0 / 60.0, substeps=10)

        out = np.zeros(4 * 3, dtype=np.float32)
        sim.get_positions(out)
        self.assertTrue(np.all(np.isfinite(out)), "新パラメータ更新後も物理計算が安定して有限値を維持すること")

    def test_air_damping_settling(self):
        """空気減衰 (Air Damping) により布の振動が急速に整定・静止することを検証"""
        import taremin_cloth_core

        positions = np.array([
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.2, 0.0, 1.0],
        ], dtype=np.float32)
        edges = np.array([[0, 1], [1, 2]], dtype=np.uint32)
        inv_masses = np.array([0.0, 1.0, 1.0], dtype=np.float32)

        # 1. 減衰なし (damping=0.0)
        sim_free = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            faces=np.empty((0, 3), dtype=np.uint32),
            inv_masses=inv_masses,
            stiffness=1000.0,
        )
        sim_free.set_damping(0.0)

        for _ in range(60):
            sim_free.step(dt=1.0 / 60.0, substeps=20)
        p0_free = np.zeros(3 * 3, dtype=np.float32)
        sim_free.get_positions(p0_free)
        for _ in range(5):
            sim_free.step(dt=1.0 / 60.0, substeps=20)
        p1_free = np.zeros(3 * 3, dtype=np.float32)
        sim_free.get_positions(p1_free)
        vel_free = np.linalg.norm(p1_free - p0_free)

        # 2. 減衰あり (damping=5.0)
        sim_damped = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            faces=np.empty((0, 3), dtype=np.uint32),
            inv_masses=inv_masses,
            stiffness=1000.0,
        )
        sim_damped.set_damping(5.0)

        for _ in range(60):
            sim_damped.step(dt=1.0 / 60.0, substeps=20)
        p0_damped = np.zeros(3 * 3, dtype=np.float32)
        sim_damped.get_positions(p0_damped)
        for _ in range(5):
            sim_damped.step(dt=1.0 / 60.0, substeps=20)
        p1_damped = np.zeros(3 * 3, dtype=np.float32)
        sim_damped.get_positions(p1_damped)
        vel_damped = np.linalg.norm(p1_damped - p0_damped)

        print(f"\n[Test Air Damping Settling] Velocity Free: {vel_free:.4f}m, Damped: {vel_damped:.4f}m")
        self.assertLess(vel_damped, vel_free * 0.1, "減衰が効いている布は未減衰時と比べて速度が1/10以下に激減するべき")
        self.assertLess(vel_damped, 0.005, "減衰布の移動量は5ミリ未満に収まるべき")

    def test_edge_rest_length_scaling(self):
        """エッジ自然長の動的スケーリング (収縮/伸長) およびリセットのテスト"""
        import taremin_cloth_core

        # 2頂点 (0, 0, 0) と (1.0, 0, 0) を結ぶエッジ（初期長 1.0）
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        edges = np.array([[0, 1]], dtype=np.uint32)
        inv_masses = np.array([0.0, 1.0], dtype=np.float32) # 頂点0固定

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            inv_masses=inv_masses,
            stiffness=10000.0,
        )
        sim.set_gravity(0.0, 0.0, 0.0)

        # 初期長の確認
        init_len = sim.get_edge_initial_rest_length(0)
        self.assertAlmostEqual(init_len, 1.0, places=4)
        self.assertAlmostEqual(sim.get_edge_rest_length(0), 1.0, places=4)

        # エッジ0のスケールを 0.6 (目標長 0.6) に設定
        edge_indices = np.array([0], dtype=np.uint32)
        scales = np.array([0.6], dtype=np.float32)
        sim.set_edge_rest_length_scales(edge_indices, scales)

        self.assertAlmostEqual(sim.get_edge_rest_length(0), 0.6, places=4)

        # 30ステップシミュレーション
        for _ in range(30):
            sim.step(dt=1.0 / 60.0, substeps=20)

        out_coords = np.zeros(6, dtype=np.float32)
        sim.get_positions(out_coords)
        out_pos = out_coords.reshape((2, 3))
        cur_dist = np.linalg.norm(out_pos[1] - out_pos[0])

        print(f"\n[Test Edge Rest Length Scaling] Initial: 1.0, Target: 0.6, Current: {cur_dist:.4f}")
        self.assertLess(abs(cur_dist - 0.6), 0.03, f"エッジ長が目標長 0.6 に収束しているべき (実測: {cur_dist:.4f})")

        # リセット
        sim.reset_edge_rest_lengths()
        self.assertAlmostEqual(sim.get_edge_rest_length(0), 1.0, places=4)

    def test_self_collision_options(self):
        """自己衝突およびUntanglingオプションAPIのテスト"""
        import taremin_cloth_core

        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0],
        ], dtype=np.float32)
        edges = np.array([[0, 1], [1, 2], [2, 0], [1, 3], [3, 2]], dtype=np.uint32)
        faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.uint32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            faces=faces,
            thickness=0.02,
            stiffness=1000.0,
        )
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=0.15,
            max_displacement_ratio=0.25,
            exclude_neighbors=True,
            enable_normal_untangling=True,
        )

        for _ in range(15):
            sim.step(dt=1.0 / 60.0, substeps=10)

        out_coords = np.zeros(12, dtype=np.float32)
        sim.get_positions(out_coords)
        self.assertFalse(np.isnan(out_coords).any(), "シミュレーション座標にNaNが含まれてはならない")


if __name__ == "__main__":
    unittest.main()



