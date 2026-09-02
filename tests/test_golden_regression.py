import os
import unittest
import numpy as np
import taremin_cloth_core
from taremin_cloth import replayer
from taremin_cloth.replayer import ClothReplayer

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden_master")

class TestGoldenRegression(unittest.TestCase):
    """
    3160075（自然な挙動が保証されているバージョン）の全フレーム物理状態と
    完全一致・大域一致することを検証するゴールデンマスター・リグレッションテスト。
    """

    def test_case1_cloth_body_regression(self):
        """Case 1: 人体メッシュコライダー (10,518面) + 衣服メッシュ (3,024頂点)"""
        golden_file = os.path.join(GOLDEN_DIR, "case1_cloth_body.npz")
        self.assertTrue(os.path.exists(golden_file), f"Golden file not found: {golden_file}")
        data = np.load(golden_file)
        golden_pos = data["positions"]

        log_body = 'frame_logs/cloth_debug_20260901_182137_cloth_body.jsonl.gz'
        rep = ClothReplayer(log_body)
        f0 = replayer.get_frame(log_body, 0)
        pos0 = np.array(f0['positions'])
        sim = rep.create_simulator(pos0)
        out = np.empty(len(pos0) * 3, dtype=np.float32)

        for fi in range(1, len(golden_pos)):
            ff = replayer.get_frame(log_body, fi)
            rep.apply_frame_inputs(ff)
            dt = ff.get('dt', 1.0/60.0)
            substeps = ff.get('substeps', 20)
            solver_iters = ff.get('solver_iterations', 1)
            sim.step(dt, substeps, solver_iters)
            sim.get_positions(out)
            actual_pos = out.reshape(-1, 3)

            # 1. NaN チェック & 最大速度チェック
            self.assertFalse(np.isnan(actual_pos).any(), f"Case 1 contains NaN at Frame {fi}")
            disp = np.linalg.norm(actual_pos - golden_pos[fi-1], axis=1).max()
            vel = disp / dt
            self.assertLess(vel, 25.0, f"Case 1 velocity explosion at Frame {fi}: {vel:.2f} m/s")

            # 2. 短期厳密一致 (Frame 1..5: 100 substeps)
            diff = np.linalg.norm(actual_pos - golden_pos[fi], axis=1).max()
            if fi <= 5:
                self.assertLess(diff, 0.008, f"Case 1 diverged in short term at Frame {fi}: {diff*1000:.4f} mm")

            # 3. 中期大域統計 (重心位置の一致)
            center_actual = actual_pos.mean(axis=0)
            center_golden = golden_pos[fi].mean(axis=0)
            center_diff = np.linalg.norm(center_actual - center_golden)
            self.assertLess(center_diff, 0.010, f"Case 1 center of mass diverged at Frame {fi}: {center_diff*1000:.4f} mm")

    def test_case2_plane_sphere_regression(self):
        """Case 2: 高速突き上げ球体コライダー + 平面布 (2,178頂点)"""
        golden_file = os.path.join(GOLDEN_DIR, "case2_plane_sphere.npz")
        self.assertTrue(os.path.exists(golden_file), f"Golden file not found: {golden_file}")
        data = np.load(golden_file)
        golden_pos = data["positions"]

        log_plane = 'frame_logs/cloth_debug_20260902_025605_Plane.jsonl.gz'
        rep = ClothReplayer(log_plane)
        f0_p = replayer.get_frame(log_plane, 0)
        pos0_p = np.array(f0_p['positions'])
        sim = rep.create_simulator(pos0_p)
        out = np.empty(len(pos0_p) * 3, dtype=np.float32)

        for fi in range(1, len(golden_pos)):
            ff = replayer.get_frame(log_plane, fi)
            rep.apply_frame_inputs(ff)
            dt = ff.get('dt', 1.0/60.0)
            substeps = ff.get('substeps', 20)
            solver_iters = ff.get('solver_iterations', 1)
            sim.step(dt, substeps, solver_iters)
            sim.get_positions(out)
            actual_pos = out.reshape(-1, 3)

            self.assertFalse(np.isnan(actual_pos).any(), f"Case 2 contains NaN at Frame {fi}")
            
            # 短期厳密一致 (Frame 1..10)
            diff = np.linalg.norm(actual_pos - golden_pos[fi], axis=1).max()
            if fi <= 10:
                self.assertLess(diff, 0.001, f"Case 2 short-term diverged at Frame {fi}: {diff*1000:.4f} mm")

            # 大域重心一致
            center_actual = actual_pos.mean(axis=0)
            center_golden = golden_pos[fi].mean(axis=0)
            center_diff = np.linalg.norm(center_actual - center_golden)
            self.assertLess(center_diff, 0.015, f"Case 2 center of mass diverged at Frame {fi}: {center_diff*1000:.4f} mm")

    def test_case3_multi_collider_regression(self):
        """Case 3: カプセル2本 + 球体 + 床面 同時干渉"""
        golden_file = os.path.join(GOLDEN_DIR, "case3_multi_collider.npz")
        self.assertTrue(os.path.exists(golden_file), f"Golden file not found: {golden_file}")
        data = np.load(golden_file)
        golden_pos = data["positions"]
        pos_init = data["initial_pos"]
        edges = data["edges"]
        faces = data["faces"]

        sim = taremin_cloth_core.ClothSimulator(
            positions=pos_init,
            edges=edges,
            faces=faces,
            inv_masses=np.ones(len(pos_init), dtype=np.float32),
            thickness=0.005,
            stiffness=1000.0,
            bending_stiffness=10.0,
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.add_capsule_collider([-0.3, -0.3, 0.4], [0.3, -0.3, 0.5], radius=0.08, friction=0.4, restitution=0.0)
        sim.add_capsule_collider([-0.3, 0.3, 0.5], [0.3, 0.3, 0.4], radius=0.08, friction=0.4, restitution=0.0)
        sim.add_sphere_collider([0.0, 0.0, 0.3], radius=0.12, friction=0.5, restitution=0.0)
        sim.add_plane_collider([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], friction=0.6, restitution=0.0)

        out = np.empty(len(pos_init) * 3, dtype=np.float32)
        for fi in range(1, len(golden_pos)):
            sim.step(1.0/60.0, 20, 2)
            sim.get_positions(out)
            actual_pos = out.reshape(-1, 3)

            self.assertFalse(np.isnan(actual_pos).any(), f"Case 3 contains NaN at Frame {fi}")

            # 短期厳密一致 (Frame 1..10)
            diff = np.linalg.norm(actual_pos - golden_pos[fi], axis=1).max()
            if fi <= 10:
                self.assertLess(diff, 0.001, f"Case 3 short-term diverged at Frame {fi}: {diff*1000:.4f} mm")

            # 大域重心一致
            center_diff = np.linalg.norm(actual_pos.mean(axis=0) - golden_pos[fi].mean(axis=0))
            self.assertLess(center_diff, 0.015, f"Case 3 center of mass diverged at Frame {fi}: {center_diff*1000:.4f} mm")

    def test_case4_self_collision_drape_regression(self):
        """Case 4: 高密度多重折り畳み・セルフコリジョン"""
        golden_file = os.path.join(GOLDEN_DIR, "case4_self_collision_drape.npz")
        self.assertTrue(os.path.exists(golden_file), f"Golden file not found: {golden_file}")
        data = np.load(golden_file)
        golden_pos = data["positions"]
        pos_init = data["initial_pos"]
        edges = data["edges"]
        faces = data["faces"]

        sim = taremin_cloth_core.ClothSimulator(
            positions=pos_init,
            edges=edges,
            faces=faces,
            inv_masses=np.ones(len(pos_init), dtype=np.float32),
            thickness=0.005,
            stiffness=800.0,
            bending_stiffness=5.0,
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(relief_factor=0.2, max_displacement_ratio=0.2, exclude_neighbors=True, enable_normal_untangling=True, max_iterations=256)
        sim.add_sphere_collider([0.0, 0.0, 0.2], radius=0.20, friction=0.6, restitution=0.0)

        out = np.empty(len(pos_init) * 3, dtype=np.float32)
        for fi in range(1, len(golden_pos)):
            sim.step(1.0/60.0, 20, 2)
            sim.get_positions(out)
            actual_pos = out.reshape(-1, 3)

            self.assertFalse(np.isnan(actual_pos).any(), f"Case 4 contains NaN at Frame {fi}")

            # 短期厳密一致 (Frame 1..5)
            diff = np.linalg.norm(actual_pos - golden_pos[fi], axis=1).max()
            if fi <= 5:
                self.assertLess(diff, 0.005, f"Case 4 short-term diverged at Frame {fi}: {diff*1000:.4f} mm")

            # 大域重心一致
            center_diff = np.linalg.norm(actual_pos.mean(axis=0) - golden_pos[fi].mean(axis=0))
            self.assertLess(center_diff, 0.020, f"Case 4 center of mass diverged at Frame {fi}: {center_diff*1000:.4f} mm")

if __name__ == '__main__':
    unittest.main()
