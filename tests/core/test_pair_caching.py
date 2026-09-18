# -*- coding: utf-8 -*-
"""接触候補ペアキャッシュ (Active Pair Caching / I-Cloth 方式) の単体・統合テスト"""

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



class TestPairCaching(unittest.TestCase):
    """接触候補ペアキャッシュの機能検証テスト"""

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

    def create_grid_cloth(self, nx=4, ny=4, dx=0.05, z=0.0, pin_first_row=True):
        """テスト用の小さな平面布メッシュを生成する"""
        positions = []
        inv_masses = []
        for y in range(ny):
            for x in range(nx):
                positions.append([x * dx, y * dx, z])
                if pin_first_row and y == 0:
                    inv_masses.append(0.0)
                else:
                    inv_masses.append(1.0)

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

    def test_pair_cache_getter_setter(self):
        """ペアキャッシュのgetter/setterおよびコンストラクタ初期化の動作テスト"""
        positions, edges, faces, inv_masses = self.create_grid_cloth()

        # デフォルト値は False (安全策)
        sim_default = self.core.ClothSimulator(positions, edges, faces, inv_masses)
        self.assertFalse(sim_default.get_enable_pair_cache())

        # setter で True に切り替え
        sim_default.set_enable_pair_cache(True)
        self.assertTrue(sim_default.get_enable_pair_cache())

        # setter で False に戻す
        sim_default.set_enable_pair_cache(False)
        self.assertFalse(sim_default.get_enable_pair_cache())

        # コンストラクタで直接 enable_pair_cache=True を指定
        sim_enabled = self.core.ClothSimulator(
            positions, edges, faces, inv_masses,
            enable_pair_cache=True,
        )
        self.assertTrue(sim_enabled.get_enable_pair_cache())

    def test_simulation_step_with_pair_cache(self):
        """ペアキャッシュ有効下でシミュレーションが正常にステップ進行することの検証"""
        pos, edges, faces, inv_m = self.create_grid_cloth(nx=4, ny=4)

        sim = self.core.ClothSimulator(
            pos, edges, faces, inv_m,
            enable_pair_cache=True,
        )
        sim.set_enable_self_collision(True)

        out_pos = np.zeros(len(pos) * 3, dtype=np.float32)

        # 1フレームシミュレーションを進める (substeps=2)
        sim.step(1.0 / 60.0, 2)
        sim.get_positions(out_pos)

        # NaN や Inf が発生していないことを確認
        self.assertFalse(np.isnan(out_pos).any(), "ペアキャッシュ有効時に NaN が発生しました")
        self.assertFalse(np.isinf(out_pos).any(), "ペアキャッシュ有効時に Inf が発生しました")

        # 重力によりピン留めされていない頂点が落下していることを確認
        pos_3d = out_pos.reshape(-1, 3)
        free_verts_z = pos_3d[4:, 2]
        self.assertTrue((free_verts_z < 0.0).all(), "自由頂点が重力方向に変位していません")

    def test_double_layer_collision_repulsion(self):
        """2層の近接した布（上下に重なる）がペアキャッシュ有効下で正常に衝突・反発することを検証"""
        # 上下の2枚の布を作成 (z=0 と z=0.003: 厚み 0.005 未満で接近)
        pos1, e1, f1, inv1 = self.create_grid_cloth(nx=3, ny=3, dx=0.05, z=0.0, pin_first_row=True)
        pos2, e2, f2, inv2 = self.create_grid_cloth(nx=3, ny=3, dx=0.05, z=0.003, pin_first_row=False)

        n1 = len(pos1)
        positions = np.vstack([pos1, pos2])
        edges = np.vstack([e1, e2 + n1])
        faces = np.vstack([f1, f2 + n1])
        inv_masses = np.concatenate([inv1, inv2])

        sim = self.core.ClothSimulator(
            positions, edges, faces, inv_masses,
            thickness=0.005,
            enable_pair_cache=True,
        )
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(relief_factor=1.0, max_displacement_ratio=0.5)

        out_pos = np.zeros(len(positions) * 3, dtype=np.float32)
        sim.step(1.0 / 60.0, 2)
        sim.get_positions(out_pos)
        self.assertFalse(np.isnan(out_pos).any())
        self.assertEqual(len(out_pos), len(positions) * 3)


if __name__ == "__main__":
    unittest.main()

