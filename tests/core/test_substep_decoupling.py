# -*- coding: utf-8 -*-
"""自己衝突サブステップ・デカップリングの単体・統合テスト"""

import unittest
import numpy as np


class TestSubstepDecoupling(unittest.TestCase):
    """自己衝突ディスパッチ間隔（サブステップ・デカップリング）の機能検証テスト"""

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

    def create_test_cloth(self, nx=4, ny=4, dx=0.05):
        """テスト用の小さな平面布メッシュを生成する"""
        positions = []
        inv_masses = []
        for y in range(ny):
            for x in range(nx):
                positions.append([x * dx, y * dx, 0.0])
                # 最初の行をピン留め
                inv_masses.append(0.0 if y == 0 else 1.0)

        edges = []
        faces = []
        for y in range(ny):
            for x in range(nx):
                idx = y * nx + x
                if x + 1 < nx:
                    edges.append([idx, idx + 1])
                if y + 1 < ny:
                    edges.append([idx, idx + nx])
                if x + 1 < nx and y + 1 < ny:
                    faces.append([idx, idx + 1, idx + nx + 1])
                    faces.append([idx, idx + nx + 1, idx + nx])

        return (
            np.array(positions, dtype=np.float32),
            np.array(edges, dtype=np.uint32),
            np.array(faces, dtype=np.uint32),
            np.array(inv_masses, dtype=np.float32),
        )

    def test_interval_getter_setter(self):
        """サブステップ間隔のgetter/setterの動作テスト"""
        positions, edges, faces, inv_masses = self.create_test_cloth()
        sim = self.core.ClothSimulator(positions, edges, faces, inv_masses)

        # デフォルト値は 1
        self.assertEqual(sim.get_self_collision_substep_interval(), 1)

        # 2 に設定
        sim.set_self_collision_substep_interval(2)
        self.assertEqual(sim.get_self_collision_substep_interval(), 2)

        # 4 に設定
        sim.set_self_collision_substep_interval(4)
        self.assertEqual(sim.get_self_collision_substep_interval(), 4)

        # 0 または負の値（u32なので型制約あり）に対しては max(1) が適用される
        sim.set_self_collision_substep_interval(0)
        self.assertEqual(sim.get_self_collision_substep_interval(), 1)

    def test_simulation_step_with_interval_1_and_2(self):
        """interval=1 と interval=2 の両方でシミュレーションが正常にステップ進行することの検証"""
        positions, edges, faces, inv_masses = self.create_test_cloth(nx=6, ny=6)

        for interval in [1, 2, 3]:
            with self.subTest(interval=interval):
                sim = self.core.ClothSimulator(positions, edges, faces, inv_masses)
                sim.set_self_collision_substep_interval(interval)
                sim.set_enable_self_collision(True)

                out_pos = np.zeros(len(positions) * 3, dtype=np.float32)

                # 10フレームシミュレーションを進める (substeps=10)
                for _ in range(10):
                    sim.step(1.0 / 60.0, 10)
                    sim.get_positions(out_pos)

                    # NaN や Inf が発生していないことを確認
                    self.assertFalse(np.isnan(out_pos).any(), f"interval={interval} で NaN が発生しました")
                    self.assertFalse(np.isinf(out_pos).any(), f"interval={interval} で Inf が発生しました")

                # 重力によりピン留めされていない頂点が落下していることを確認
                pos_3d = out_pos.reshape(-1, 3)
                free_verts_z = pos_3d[6:, 2]  # ピン留め行以外のz座標
                self.assertTrue((free_verts_z < 0.0).all(), "自由頂点が重力方向に変位していません")


if __name__ == "__main__":
    unittest.main()
