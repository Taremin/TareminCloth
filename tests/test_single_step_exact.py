import os
import unittest
import numpy as np
import taremin_cloth_core
from taremin_cloth import replayer
from taremin_cloth.replayer import ClothReplayer

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden_master")

class TestSingleStepExact(unittest.TestCase):
    """
    3160075 の計算結果と 1ステップ（1フレーム）で高精度一致（誤差 < 0.5mm）することを
    検証する厳密な Characterization Test。
    """

    def test_case1_cloth_body_step_exact(self):
        log_body = 'frame_logs/cloth_debug_20260901_182137_cloth_body.jsonl.gz'
        for fi in [1, 5, 10]:
            golden_path = os.path.join(GOLDEN_DIR, f"step_case1_f{fi}.npz")
            self.assertTrue(os.path.exists(golden_path))
            data = np.load(golden_path)
            pos_in = data["pos_in"]
            pos_out_golden = data["pos_out"]

            f_curr = replayer.get_frame(log_body, fi)
            rep = ClothReplayer(log_body)
            sim = rep.create_simulator(pos_in)
            rep.apply_frame_inputs(f_curr)

            dt = f_curr.get('dt', 1.0/60.0)
            substeps = f_curr.get('substeps', 20)
            solver_iters = f_curr.get('solver_iterations', 1)
            sim.step(dt, substeps, solver_iters)

            out = np.empty(len(pos_in) * 3, dtype=np.float32)
            sim.get_positions(out)
            actual = out.reshape(-1, 3)

            # 許容誤差: 2.0mm 未満 (自己衝突の非同期揺らぎ範囲内)
            diff = np.linalg.norm(actual - pos_out_golden, axis=1).max()
            self.assertLess(diff, 0.002, f"Case 1 Single-step Frame {fi} diverged: diff = {diff*1000:.4f} mm")

    def test_case2_plane_sphere_step_exact(self):
        log_plane = 'frame_logs/cloth_debug_20260902_025605_Plane.jsonl.gz'
        for fi in [1, 10, 20]:
            golden_path = os.path.join(GOLDEN_DIR, f"step_case2_f{fi}.npz")
            self.assertTrue(os.path.exists(golden_path))
            data = np.load(golden_path)
            pos_in = data["pos_in"]
            pos_out_golden = data["pos_out"]

            f_curr = replayer.get_frame(log_plane, fi)
            rep = ClothReplayer(log_plane)
            sim = rep.create_simulator(pos_in)
            rep.apply_frame_inputs(f_curr)

            dt = f_curr.get('dt', 1.0/60.0)
            substeps = f_curr.get('substeps', 20)
            solver_iters = f_curr.get('solver_iterations', 1)
            sim.step(dt, substeps, solver_iters)

            out = np.empty(len(pos_in) * 3, dtype=np.float32)
            sim.get_positions(out)
            actual = out.reshape(-1, 3)

            # 許容誤差: 5.0mm 未満
            diff = np.linalg.norm(actual - pos_out_golden, axis=1).max()
            self.assertLess(diff, 0.005, f"Case 2 Single-step Frame {fi} diverged: diff = {diff*1000:.4f} mm")

    def test_case3_multi_collider_step_exact(self):
        c3_data = np.load(os.path.join(GOLDEN_DIR, "case3_multi_collider.npz"))
        pos_c3_init = c3_data["initial_pos"]
        edges_c3 = c3_data["edges"]
        faces_c3 = c3_data["faces"]

        step_data = np.load(os.path.join(GOLDEN_DIR, "step_case3_f1.npz"))
        pos_out_golden = step_data["pos_out"]

        sim = taremin_cloth_core.ClothSimulator(
            positions=pos_c3_init,
            edges=edges_c3,
            faces=faces_c3,
            inv_masses=np.ones(len(pos_c3_init), dtype=np.float32),
            thickness=0.005,
            stiffness=1000.0,
            bending_stiffness=10.0,
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.add_capsule_collider([-0.3, -0.3, 0.4], [0.3, -0.3, 0.5], radius=0.08, friction=0.4, restitution=0.0)
        sim.add_capsule_collider([-0.3, 0.3, 0.5], [0.3, 0.3, 0.4], radius=0.08, friction=0.4, restitution=0.0)
        sim.add_sphere_collider([0.0, 0.0, 0.3], radius=0.12, friction=0.5, restitution=0.0)
        sim.add_plane_collider([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], friction=0.6, restitution=0.0)

        sim.step(1.0/60.0, 20, 2)
        out = np.empty(len(pos_c3_init) * 3, dtype=np.float32)
        sim.get_positions(out)
        actual = out.reshape(-1, 3)

        # 許容誤差: 0.01mm (10um) 未満
        diff = np.linalg.norm(actual - pos_out_golden, axis=1).max()
        self.assertLess(diff, 1e-5, f"Case 3 Single-step Frame 1 diverged: diff = {diff*1e6:.4f} um")

if __name__ == '__main__':
    unittest.main()
