"""
縫合収縮の時間整合性テスト
収縮速度 (m/s) は時間進行であり、自己衝突の有無・反復回数に依存しないことを検証する。
(旧実装では拘束解決ディスパッチ毎に収縮し、ON時は2倍速になっていた)
"""

import unittest
import numpy as np
import taremin_cloth_core

NX, NY, DX, GAP = 8, 8, 0.02, 0.30


def make_pair():
    pos, edges, faces, sew = [], [], [], []
    for patch in range(2):
        base = patch * NX * NY
        ox = patch * ((NX - 1) * DX + GAP)
        for y in range(NY):
            for x in range(NX):
                pos.append([ox + x * DX, 0.0, -y * DX])
        for y in range(NY):
            for x in range(NX):
                idx = base + y * NX + x
                if x + 1 < NX:
                    edges.append([idx, idx + 1])
                if y + 1 < NY:
                    edges.append([idx, idx + NX])
                if x + 1 < NX and y + 1 < NY:
                    faces.append([idx, idx + 1, idx + NX + 1])
                    faces.append([idx, idx + NX + 1, idx + NX])
    for y in range(NY):
        sew.append([y * NX + (NX - 1), NX * NY + y * NX + 0])
    return (np.array(pos, dtype=np.float32), np.array(edges, dtype=np.uint32),
            np.array(faces, dtype=np.uint32), np.array(sew, dtype=np.uint32))


def mean_rest(selfcol, iters, frames=10, shrink=0.5):
    pos, edges, faces, sew = make_pair()
    inv = np.ones(len(pos), dtype=np.float32)
    sim = taremin_cloth_core.ClothSimulator(
        positions=pos, edges=edges, faces=faces, inv_masses=inv,
        sewing_springs=sew, stiffness=1000.0,
        sewing_shrink_speed=shrink, sewing_stiffness=10000.0)
    sim.set_gravity(0, 0, 0)
    sim.set_enable_self_collision(selfcol)
    if selfcol:
        sim.set_coupled_self_collision_options(1, 2)
    for _ in range(frames):
        sim.step(dt=1.0 / 60.0, substeps=10, solver_iterations=iters)
    return float(np.mean(sim.get_sewing_current_rest_lengths()))


class TestSewingShrinkParity(unittest.TestCase):
    """収縮進行の自己衝突・反復非依存性"""

    def test_shrink_rate_matches_theory(self):
        """理論値 (gap - speed*time) と一致すること"""
        r = mean_rest(False, 2) * 1000.0
        expect = (GAP - 0.5 * (10.0 / 60.0)) * 1000.0
        self.assertAlmostEqual(r, expect, delta=1.0,
                               msg=f"収縮は時間進行であること (実測{r:.1f}mm, 理論{expect:.1f}mm)")

    def test_shrink_parity_on_off(self):
        """ON/OFFで収縮進行が一致すること"""
        r_off = mean_rest(False, 2)
        r_on = mean_rest(True, 2)
        self.assertAlmostEqual(r_off, r_on, delta=1e-4,
                               msg=f"ON/OFFで収縮速度が変わらないこと (OFF={r_off}, ON={r_on})")

    def test_shrink_independent_of_iterations(self):
        """反復回数を変えても収縮進行が一致すること"""
        r1 = mean_rest(False, 1)
        r4 = mean_rest(False, 4)
        self.assertAlmostEqual(r1, r4, delta=1e-4,
                               msg=f"反復数で収縮速度が変わらないこと (iter1={r1}, iter4={r4})")


if __name__ == "__main__":
    unittest.main()
