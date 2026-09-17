import os
import unittest
import numpy as np
import taremin_cloth_core

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "golden_master")

class TestSingleStepExact(unittest.TestCase):
    """
    3160075 の計算結果と 1ステップ（1フレーム）で高精度一致（誤差 < 0.5mm）することを
    検証する厳密な Characterization Test。
    自己完結したフィクスチャデータのみを用いて決定論的に検証する。
    """

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
