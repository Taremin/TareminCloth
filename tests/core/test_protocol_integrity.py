# -*- coding: utf-8 -*-
"""Python-Rust-wgpu間のエンドツーエンド プロトコル・バッファ整合性テスト.

Pythonから投入したパラメータや幾何データが、Rust経由でwgpuのGPUバッファ（Uniform/Storage）に
正しい型・レイアウト・値として書き込まれているかを実機リードバックで検証する。
"""

import json
import os
import sys
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)

import tests  # noqa: F401


class TestProtocolIntegrity(unittest.TestCase):
    def setUp(self):
        try:
            import taremin_cloth_core as core
            self.core = core
        except ImportError:
            try:
                from taremin_cloth import taremin_cloth_core as core
                self.core = core
            except ImportError:
                self.skipTest("taremin_cloth_core が利用できないためスキップします")

        from taremin_cloth.engine import simconfig as sc
        self.sc = sc

    def _create_simple_cloth_sim(self):
        """テスト用の最小三角形メッシュシミュレータを作成する."""
        pos = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ], dtype=np.float32)
        edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.uint32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        inv_masses = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        layer_ids = np.array([0, 1, 0], dtype=np.uint32)
        thicknesses = np.array([0.005, 0.008, 0.010], dtype=np.float32)

        sim = self.core.ClothSimulator(
            positions=pos,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            layer_ids=layer_ids,
            thicknesses=thicknesses,
            thickness=0.005,
            stiffness=2000.0,
            bending_stiffness=15.0,
        )
        return sim, pos, edges, faces, inv_masses, layer_ids, thicknesses

    def test_sim_params_gpu_integrity(self):
        """Pythonで設定したSimParamsパラメータがGPU Uniformバッファに正確に届いていること."""
        sim, _, _, _, _, _, _ = self._create_simple_cloth_sim()

        # 識別用の一意なパラメータ値を投入
        test_gravity = [1.25, -2.5, 3.75]
        test_damping = 0.42
        test_sewing_stiffness = 2500.0  # < 5000 なので compliance = 1 / 2500.0
        test_sewing_lock_dist = 0.035

        sim.set_gravity(test_gravity[0], test_gravity[1], test_gravity[2])
        sim.set_damping(test_damping)
        sim.set_sewing_stiffness(test_sewing_stiffness)
        sim.set_enable_sewing_lock(True)
        sim.set_sewing_lock_distance(test_sewing_lock_dist)

        dt = 0.02
        substeps = 8
        sim.step(dt=dt, substeps=substeps)

        # GPUバッファから実値をリードバック
        gpu_params = sim.inspect_sim_params()
        self.assertIsNotNone(gpu_params, "SimParams バッファの読戻しに成功すること")

        # 1. 重力ベクトルと substep_dt
        expected_substep_dt = dt / float(substeps)
        self.assertAlmostEqual(gpu_params["gravity"][0], test_gravity[0], places=5)
        self.assertAlmostEqual(gpu_params["gravity"][1], test_gravity[1], places=5)
        self.assertAlmostEqual(gpu_params["gravity"][2], test_gravity[2], places=5)
        self.assertAlmostEqual(gpu_params["substep_dt"], expected_substep_dt, places=6)

        # 2. 減衰およびステップ数
        self.assertAlmostEqual(gpu_params["damping"], test_damping, places=5)
        self.assertEqual(gpu_params["substeps"], substeps)
        self.assertEqual(gpu_params["num_vertices"], 3)
        self.assertEqual(gpu_params["num_distance_constraints"], 3)

        # 3. 縫合設定およびコンプライアンス変換
        expected_compliance = 1.0 / test_sewing_stiffness
        self.assertAlmostEqual(gpu_params["sewing_compliance"], expected_compliance, places=6)
        self.assertEqual(gpu_params["enable_sewing_lock"], 1.0)
        self.assertAlmostEqual(gpu_params["sewing_lock_distance"], test_sewing_lock_dist, places=5)

    def test_collision_params_gpu_integrity(self):
        """コライダー設定およびメッシュ三角形がGPU CollisionParamsバッファに正確に届いていること."""
        sim, _, _, _, _, _, _ = self._create_simple_cloth_sim()

        # コライダーを追加
        sim.add_sphere_collider([0.0, 0.0, -0.5], 0.25, friction=0.3, restitution=0.1)
        sim.add_capsule_collider([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.1, friction=0.4, restitution=0.2)

        # メッシュ三角形コライダーを追加
        tri_mesh = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ], dtype=np.float32)
        sim.set_mesh_collider_triangles(tri_mesh, friction=0.5, thickness=0.015, restitution=0.05, single_sided=True)

        # コライダーオプションを設定
        sim.set_edge_margin_scale(1.35)
        sim.set_edge_margin_offset(0.004)
        sim.set_collider_options(enable_cluster_culling=True, enable_single_sided_recovery=True, sweep_margin=0.075)

        dt = 0.016
        substeps = 10
        sim.step(dt=dt, substeps=substeps)

        # GPU CollisionParams バッファをリードバック
        gpu_col = sim.inspect_collision_params()
        self.assertIsNotNone(gpu_col, "CollisionParams バッファの読戻しに成功すること")

        self.assertEqual(gpu_col["num_vertices"], 3)
        self.assertEqual(gpu_col["num_colliders"], 2)  # 球 + カプセル
        self.assertEqual(gpu_col["num_mesh_triangles"], 2)  # 2三角形
        self.assertEqual(gpu_col["num_clusters"], 1)  # ceil(2 / 16) = 1
        self.assertAlmostEqual(gpu_col["dt"], dt / substeps, places=6)
        self.assertAlmostEqual(gpu_col["edge_margin_scale"], 1.35, places=5)
        self.assertAlmostEqual(gpu_col["edge_margin_offset"], 0.004, places=5)
        self.assertEqual(gpu_col["enable_cluster_culling"], 1)
        self.assertEqual(gpu_col["enable_single_sided_recovery"], 1)
        self.assertAlmostEqual(gpu_col["sweep_margin_offset"], 0.075, places=5)

    def test_self_collision_params_gpu_integrity(self):
        """自己衝突パラメータがGPU SelfCollisionParamsバッファに正確に届いていること."""
        sim, _, _, _, _, _, _ = self._create_simple_cloth_sim()

        test_relief = 0.38
        test_max_disp = 0.42
        test_exclude = False
        test_untangling = True
        test_max_iters = 512

        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=test_relief,
            max_displacement_ratio=test_max_disp,
            exclude_neighbors=test_exclude,
            enable_normal_untangling=test_untangling,
            max_iterations=test_max_iters,
        )

        sim.step(0.016, 5)

        gpu_sc = sim.inspect_self_collision_params()
        self.assertIsNotNone(gpu_sc, "SelfCollisionParams バッファの読戻しに成功すること")

        self.assertEqual(gpu_sc["num_vertices"], 3)
        self.assertEqual(gpu_sc["num_edges"], 3)
        self.assertAlmostEqual(gpu_sc["relief_factor"], test_relief, places=5)
        self.assertAlmostEqual(gpu_sc["max_displacement_ratio"], test_max_disp, places=5)
        self.assertEqual(gpu_sc["exclude_neighbors"], 1 if test_exclude else 0)
        self.assertEqual(gpu_sc["max_search_iterations"], test_max_iters)

    def test_gpu_vertex_data_integrity(self):
        """頂点データ（位置、逆質量、レイヤーID、厚み）がGPU Vertexバッファに正確に配置されていること."""
        sim, pos, _, _, inv_masses, layer_ids, thicknesses = self._create_simple_cloth_sim()

        # 各頂点をリードバックして検証
        for i in range(len(pos)):
            v = sim.inspect_gpu_vertex(i)
            self.assertIsNotNone(v, f"頂点 {i} の読戻しに成功すること")
            for axis in range(3):
                self.assertAlmostEqual(v["position"][axis], pos[i, axis], places=5)
                self.assertAlmostEqual(v["prev_pos"][axis], pos[i, axis], places=5)
            self.assertAlmostEqual(v["inv_mass"], inv_masses[i], places=5)
            self.assertEqual(v["layer_id"], layer_ids[i])
            self.assertAlmostEqual(v["thickness"], thicknesses[i], places=5)

    def test_gpu_collider_data_integrity(self):
        """プリミティブコライダーがGPU Colliderバッファに正確に配置されていること."""
        sim, _, _, _, _, _, _ = self._create_simple_cloth_sim()

        # 球コライダー (type 0)
        sphere_center = [0.1, 0.2, 0.3]
        sphere_radius = 0.5
        sim.add_sphere_collider(sphere_center, sphere_radius, friction=0.25, restitution=0.15)

        # カプセルコライダー (type 1)
        capsule_a = [1.0, 2.0, 3.0]
        capsule_b = [1.0, 2.0, 5.0]
        capsule_radius = 0.2
        sim.add_capsule_collider(capsule_a, capsule_b, capsule_radius, friction=0.45, restitution=0.0)

        cols = sim.inspect_gpu_colliders()
        self.assertEqual(len(cols), 2)

        # 球コライダーの検証
        c0 = cols[0]
        self.assertEqual(c0["collider_type"], 0)
        self.assertAlmostEqual(c0["radius"], sphere_radius, places=5)
        self.assertAlmostEqual(c0["friction"], 0.25, places=5)
        self.assertAlmostEqual(c0["restitution"], 0.15, places=5)
        for ax in range(3):
            self.assertAlmostEqual(c0["point_a"][ax], sphere_center[ax], places=5)

        # カプセルコライダーの検証
        c1 = cols[1]
        self.assertEqual(c1["collider_type"], 1)
        self.assertAlmostEqual(c1["radius"], capsule_radius, places=5)
        self.assertAlmostEqual(c1["friction"], 0.45, places=5)
        self.assertAlmostEqual(c1["restitution"], 0.0, places=5)
        for ax in range(3):
            self.assertAlmostEqual(c1["point_a"][ax], capsule_a[ax], places=5)
            self.assertAlmostEqual(c1["point_b"][ax], capsule_b[ax], places=5)

    def test_gpu_pins_data_integrity(self):
        """動的ピン拘束がGPU Pinバッファに正確に配置されていること."""
        sim, _, _, _, _, _, _ = self._create_simple_cloth_sim()

        target_pos = [0.55, 0.65, 0.75]
        target_weight = 0.85
        sim.set_pin(vertex_idx=2, target_pos=target_pos, weight=target_weight)

        pins = sim.inspect_gpu_pins()
        self.assertEqual(len(pins), 1)
        p = pins[0]
        self.assertEqual(p["vertex_idx"], 2)
        self.assertAlmostEqual(p["weight"], target_weight, places=5)
        for ax in range(3):
            self.assertAlmostEqual(p["target_pos"][ax], target_pos[ax], places=5)

        # 解除時の反映確認
        sim.release_pin(2)
        pins_after = sim.inspect_gpu_pins()
        self.assertEqual(len(pins_after), 0)


if __name__ == "__main__":
    unittest.main()

