"""
内部エッジへの選択的 E-E CCD（G1 高相対速度・G2 低速近接クロスオーバー・折り畳み安定性）の検証テスト。
"""

import unittest
import numpy as np
import taremin_cloth_core


def _create_orthogonal_roof_mesh(*, z_b: float):
    """V-T衝突・境界エッジ衝突を排除し、内部エッジ同士のみが直交する8頂点屋根型メッシュを生成する。"""
    positions = np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.3, -0.5],
            [0.0, -0.3, -0.5],
            [0.0, -1.0, z_b],
            [0.0, 1.0, z_b],
            [0.3, 0.0, z_b + 0.5],
            [-0.3, 0.0, z_b + 0.5],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 0],
            [0, 3],
            [3, 1],
            [4, 5],
            [5, 6],
            [6, 4],
            [4, 7],
            [7, 5],
        ],
        dtype=np.uint32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 0, 3],
            [4, 5, 6],
            [5, 4, 7],
        ],
        dtype=np.uint32,
    )
    return positions, edges, faces


class TestInternalEdgeEdgeCCD(unittest.TestCase):
    """self_collision_ee_direct.wgsl の内部エッジ選択的 E-E CCD を検証する。"""

    def _run_crossing_case(self, *, z_b: float, vz: float, thickness: float = 0.005):
        pos, edges, faces = _create_orthogonal_roof_mesh(z_b=z_b)
        sim = taremin_cloth_core.ClothSimulator(
            positions=pos,
            edges=edges,
            faces=faces,
            thickness=thickness,
            stiffness=10000.0,
            workgroup_size=64,
        )
        sim.set_gravity(0.0, 0.0, 0.0)
        sim.set_enable_self_collision(True)
        sim.set_enable_pair_cache(False)

        vels = np.zeros_like(pos)
        vels[4:, 2] = vz
        sim.set_positions_and_velocities(pos, vels)
        sim.step(dt=1.0 / 60.0, substeps=1)

        out_coords = np.empty(len(pos) * 3, dtype=np.float32)
        sim.get_positions(out_coords)
        cur = out_coords.reshape((-1, 3))
        z_mid_a = 0.5 * float(cur[0, 2] + cur[1, 2])
        z_mid_b = 0.5 * float(cur[4, 2] + cur[5, 2])
        return z_mid_a, z_mid_b

    def test_high_speed_internal_ee_blocking(self):
        """G1: 厚みを1ステップで飛び越える高速内部エッジの貫通が遮断されること"""
        z_mid_a, z_mid_b = self._run_crossing_case(z_b=0.008, vz=-1.2)
        self.assertGreaterEqual(
            z_mid_b,
            z_mid_a - 1e-4,
            f"G1: 内部エッジ B が A を貫通してはならない (z_mid_a={z_mid_a:.6f}, z_mid_b={z_mid_b:.6f})",
        )

    def test_low_speed_close_crossover_blocking(self):
        """G2: 低速近接クロスオーバー時に離散反転押し出しではなく表側へ押し返されること"""
        z_mid_a, z_mid_b = self._run_crossing_case(z_b=0.002, vz=-0.24)
        self.assertGreater(
            z_mid_b,
            z_mid_a,
            f"G2: 内部エッジ B が A の裏側へ貫通してはならない (z_mid_a={z_mid_a:.6f}, z_mid_b={z_mid_b:.6f})",
        )


if __name__ == "__main__":
    unittest.main()
