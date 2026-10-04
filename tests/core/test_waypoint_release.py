"""
中継点縫合の動的通過＆結合完了時ピン解放（自由ドレープ化）の単体テスト
tests/core/test_waypoint_release.py
"""

import os
import sys
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)

import taremin_cloth_core
from taremin_cloth.utils.mesh_extract import WaypointSeam
from taremin_cloth.engine.waypoint import WaypointManager


class TestWaypointRelease(unittest.TestCase):
    """中継点縫合の合流・ピン解放・重力落下ドレープの検証"""

    def test_single_waypoint_merge_and_release(self):
        """単一中継点: AとBが合流した瞬間に中継点ピンが解放され、重力で落下すること"""
        # 頂点構成:
        # v0 (A): (-0.05, 0.0, 0.0), inv_mass = 1.0
        # v1 (W): ( 0.00, 0.0, 0.0), inv_mass = 0.0 (初期固定ガイド)
        # v2 (B): ( 0.05, 0.0, 0.0), inv_mass = 1.0
        positions = np.array([
            [-0.05, 0.0, 0.0],
            [ 0.00, 0.0, 0.0],
            [ 0.05, 0.0, 0.0],
        ], dtype=np.float32)

        edges = np.array([
            [0, 1],
            [1, 2],
        ], dtype=np.uint32)

        sewing_springs = np.array([
            [0, 1],  # (A, W)
            [2, 1],  # (B, W)
        ], dtype=np.uint32)

        inv_masses = np.array([1.0, 0.0, 1.0], dtype=np.float32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            inv_masses=inv_masses,
            sewing_springs=sewing_springs,
            stiffness=1000.0,
            sewing_stiffness=10000.0,
            sewing_shrink_speed=5.0,  # 急速収縮
            enable_sewing_lock=True,
        )
        sim.set_gravity(0.0, -9.81, 0.0)

        # WaypointManager の構築
        ws = WaypointSeam(
            vert_a=0,
            vert_b=2,
            waypoint_indices=[1],
            waypoint_coords=positions[1:2].copy(),
        )
        wp_mgr = WaypointManager(
            waypoint_seams=[ws],
            initial_inv_masses=inv_masses,
            merge_dist=0.015,
        )

        coords = positions.flatten().copy()

        # 初期状態: 中継点 v1 は固定ピン
        current_inv_masses = sim.get_vertex_inv_masses()
        self.assertEqual(current_inv_masses[1], 0.0)

        # ステップを進めて合流させる
        merged_frame = None
        for frame in range(1, 60):
            sim.step(dt=1.0 / 60.0, substeps=10)
            sim.get_positions(coords)
            released = wp_mgr.update(coords, sim=sim)
            if released:
                merged_frame = frame
                break

        # 60フレーム以内に合流検知されること
        self.assertIsNotNone(merged_frame, "合流判定が一定時間内に発生しませんでした")
        self.assertIn(1, wp_mgr.released_indices)

        # GPU側の inv_mass が正の値 (1.0) に更新されていること
        updated_masses = sim.get_vertex_inv_masses()
        self.assertAlmostEqual(updated_masses[1], 1.0, places=3)

        # ピン解放後、重力下で下に落下することを確認
        # 解除前のY座標を取得
        pos_merged = coords.reshape(-1, 3).copy()
        y_before = pos_merged[:, 1].copy()

        # さらに30フレーム進める
        for _ in range(30):
            sim.step(dt=1.0 / 60.0, substeps=10)

        sim.get_positions(coords)
        pos_after = coords.reshape(-1, 3)
        y_after = pos_after[:, 1]

        # 中継点 W (v1) も含め、全頂点が重力で下向き (Y < y_before) に自由移動していること
        for i in range(3):
            self.assertLess(
                y_after[i],
                y_before[i] - 0.01,
                f"頂点 {i} がピン解除後に重力落下していません (y_before={y_before[i]}, y_after={y_after[i]})"
            )

    def test_multi_waypoint_sequential_shift(self):
        """複数中継点: AがW0に達したらW0が解放され、次にW1でBと合流して全解放されること"""
        # 頂点構成: A(0), W0(1), W1(2), B(3)
        positions = np.array([
            [-0.06, 0.0, 0.0],
            [-0.03, 0.0, 0.0],
            [ 0.01, 0.0, 0.0],
            [ 0.05, 0.0, 0.0],
        ], dtype=np.float32)

        edges = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
        ], dtype=np.uint32)

        sewing_springs = np.array([
            [0, 1],  # (A, W0)
            [1, 2],  # (W0, W1)
            [3, 2],  # (B, W1)
        ], dtype=np.uint32)

        inv_masses = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            inv_masses=inv_masses,
            sewing_springs=sewing_springs,
            stiffness=1000.0,
            sewing_stiffness=10000.0,
            sewing_shrink_speed=5.0,
            enable_sewing_lock=True,
        )
        sim.set_gravity(0.0, -9.81, 0.0)

        ws = WaypointSeam(
            vert_a=0,
            vert_b=3,
            waypoint_indices=[1, 2],
            waypoint_coords=positions[1:3].copy(),
        )
        wp_mgr = WaypointManager(
            waypoint_seams=[ws],
            initial_inv_masses=inv_masses,
            merge_dist=0.015,
            pass_dist=0.015,
        )

        coords = positions.flatten().copy()

        # シミュレーションを進めて全解放されることを確認
        for _ in range(80):
            sim.step(dt=1.0 / 60.0, substeps=10)
            sim.get_positions(coords)
            wp_mgr.update(coords, sim=sim)
            if not wp_mgr.has_active_seams:
                break

        # W0 (v1) と W1 (v2) の両方が解放されたこと
        self.assertIn(1, wp_mgr.released_indices)
        self.assertIn(2, wp_mgr.released_indices)
        self.assertFalse(wp_mgr.has_active_seams)

        # 解除後の全頂点の逆質量が正であること
        updated_masses = sim.get_vertex_inv_masses()
        self.assertGreater(updated_masses[1], 0.0)
        self.assertGreater(updated_masses[2], 0.0)


if __name__ == "__main__":
    unittest.main()
