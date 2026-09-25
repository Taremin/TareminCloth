import os
import unittest
import numpy as np
import taremin_cloth_core

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures", "golden_master")

#: 物理質量の面密度 (再生成スクリプト scratch/regen_golden.py と同一値)
CASE3_AREAL_DENSITY = 0.15


def areal_inv_masses(pos, faces, density):
    """面積に応じて分配した物理単位の逆質量 (ゴールデン再現用)。"""
    masses = np.zeros(len(pos), dtype=np.float64)
    for f in np.asarray(faces, dtype=np.int64):
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        if a == b or b == c or c == a:
            continue
        ab = pos[b].astype(np.float64) - pos[a].astype(np.float64)
        ac = pos[c].astype(np.float64) - pos[a].astype(np.float64)
        share = 0.5 * float(np.linalg.norm(np.cross(ab, ac))) * density / 3.0
        masses[a] += share
        masses[b] += share
        masses[c] += share
    masses = np.maximum(masses, 1e-9)
    return (1.0 / masses).astype(np.float32)

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
            inv_masses=areal_inv_masses(pos_c3_init, faces_c3, CASE3_AREAL_DENSITY),
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
