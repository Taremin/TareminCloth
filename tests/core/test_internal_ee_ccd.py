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

    def test_ccd_not_diluted_by_pin_vt_discrete_contacts(self):
        """ピン固定三角形の離散V-T接触が混在しても真のE-E CCD変位が希釈されず貫通しないこと"""
        pos = np.array(
            [
                [-0.04, 0.0, 0.0],
                [0.04, 0.0, 0.0],
                [0.0, 0.02, -0.04],
                [0.0, -0.02, -0.04],
                [0.0, -0.04, 0.002],
                [0.0, 0.04, 0.002],
                [0.02, 0.0, 0.042],
                [-0.02, 0.0, 0.042],
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

        base_n = len(pos)
        pin_verts = []
        pin_edges = []
        pin_faces = []
        for y_sign in (-1.0, 1.0):
            yc = y_sign * 0.04
            for k in range(3):
                off = base_n + len(pin_verts)
                xp = 0.004 + k * 0.0005
                pin_verts.extend(
                    [
                        [xp, yc - 0.015, -0.03],
                        [xp, yc + 0.015, -0.03],
                        [xp, yc + 0.015, 0.01],
                        [xp, yc - 0.015, 0.01],
                    ]
                )
                pin_edges.extend(
                    [
                        [off, off + 1],
                        [off + 1, off + 2],
                        [off + 2, off + 3],
                        [off + 3, off],
                        [off, off + 2],
                    ]
                )
                pin_faces.extend(
                    [
                        [off, off + 1, off + 2],
                        [off, off + 2, off + 3],
                    ]
                )

        all_pos = np.vstack([pos, np.array(pin_verts, dtype=np.float32)])
        all_edges = np.vstack([edges, np.array(pin_edges, dtype=np.uint32)])
        all_faces = np.vstack([faces, np.array(pin_faces, dtype=np.uint32)])

        inv_masses = np.ones(len(all_pos), dtype=np.float32)
        inv_masses[base_n:] = 0.0

        sim = taremin_cloth_core.ClothSimulator(
            positions=all_pos,
            edges=all_edges,
            faces=all_faces,
            inv_masses=inv_masses,
            thickness=0.005,
            stiffness=10000.0,
            workgroup_size=64,
        )
        sim.set_gravity(0.0, 0.0, 0.0)
        sim.set_enable_self_collision(True)
        sim.set_enable_pair_cache(False)

        vels = np.zeros_like(all_pos)
        vels[4:8, 2] = -1.5
        sim.set_positions_and_velocities(all_pos, vels)
        sim.step(dt=1.0 / 60.0, substeps=1)

        out_coords = np.empty(len(all_pos) * 3, dtype=np.float32)
        sim.get_positions(out_coords)
        cur = out_coords.reshape((-1, 3))
        z_mid_a = 0.5 * float(cur[0, 2] + cur[1, 2])
        z_mid_b = 0.5 * float(cur[4, 2] + cur[5, 2])
        sep = z_mid_b - z_mid_a
        self.assertGreaterEqual(
            sep,
            0.0095,
            f"pin_vt 離散接触で真の E-E CCD が希釈されてはならない (z_mid_a={z_mid_a:.6f}, z_mid_b={z_mid_b:.6f}, sep={sep:.6f})",
        )

    def test_ccd_not_diluted_by_concurrent_double_ee(self):
        """多重同時 E-E CCD ヒット時に単純平均希釈で内部エッジが貫通しないこと"""
        z_b = 0.002
        # 空間ハッシュ 27 近傍内に確実に収まる実布スケールで、
        # A1 (y=-0.012, s=0.20) と A2 (y=+0.012, s=0.80) の 2 本へ B が同時交差する構成
        positions = np.array(
            [
                [-0.04, -0.012, 0.0],
                [0.04, -0.012, 0.0],
                [0.0, 0.008, -0.04],
                [0.0, -0.032, -0.04],
                [-0.04, 0.012, 0.0],
                [0.04, 0.012, 0.0],
                [0.0, 0.032, -0.04],
                [0.0, -0.008, -0.04],
                [0.0, -0.02, z_b],
                [0.0, 0.02, z_b],
                [0.02, 0.0, z_b + 0.04],
                [-0.02, 0.0, z_b + 0.04],
            ],
            dtype=np.float32,
        )
        edges = []
        faces = []
        for off in (0, 4, 8):
            edges.extend([[off + 0, off + 1], [off + 1, off + 2], [off + 2, off + 0], [off + 0, off + 3], [off + 3, off + 1]])
            faces.extend([[off + 0, off + 1, off + 2], [off + 1, off + 0, off + 3]])

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=np.array(edges, dtype=np.uint32),
            faces=np.array(faces, dtype=np.uint32),
            thickness=0.005,
            stiffness=10000.0,
            workgroup_size=64,
        )
        sim.set_gravity(0.0, 0.0, 0.0)
        sim.set_enable_self_collision(True)
        sim.set_enable_pair_cache(False)

        vels = np.zeros_like(positions)
        vels[8:12, 2] = -1.5
        sim.set_positions_and_velocities(positions, vels)
        sim.step(dt=1.0 / 60.0, substeps=1)

        out_coords = np.empty(len(positions) * 3, dtype=np.float32)
        sim.get_positions(out_coords)
        cur = out_coords.reshape((-1, 3))

        z_a1 = 0.5 * float(cur[0, 2] + cur[1, 2])
        z_b_at_a1 = 0.80 * float(cur[8, 2]) + 0.20 * float(cur[9, 2])
        z_a2 = 0.5 * float(cur[4, 2] + cur[5, 2])
        z_b_at_a2 = 0.20 * float(cur[8, 2]) + 0.80 * float(cur[9, 2])

        self.assertGreater(
            z_b_at_a1,
            z_a1 + 0.009,
            f"多重同時 E-E CCD (A1側) で貫通・希釈してはならない (z_a1={z_a1:.6f}, z_b={z_b_at_a1:.6f})",
        )
        self.assertGreater(
            z_b_at_a2,
            z_a2 + 0.009,
            f"多重同時 E-E CCD (A2側) で貫通・希釈してはならない (z_a2={z_a2:.6f}, z_b={z_b_at_a2:.6f})",
        )

    def test_ee_discrete_untangling_layer_order_symmetry(self):
        """離散E-Eパスでの多層布untangling判定が180度剛体回転(Mortonエッジ順序反転)に対称であること"""
        p_upright, e_raw, f_raw = _create_orthogonal_roof_mesh(z_b=-0.003)
        p_rotated = p_upright.copy()
        p_rotated[:, 1] *= -1.0
        p_rotated[:, 2] *= -1.0
        layers = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint32)

        def run_case(*, pos_init: np.ndarray, sign_z: float) -> tuple[float, float]:
            sim = taremin_cloth_core.ClothSimulator(
                positions=pos_init,
                edges=e_raw,
                faces=f_raw,
                layer_ids=layers,
                thickness=0.005,
                stiffness=10000.0,
                workgroup_size=64,
            )
            sim.set_gravity(0.0, 0.0, 0.0)
            sim.set_enable_self_collision(True)
            sim.set_enable_pair_cache(False)
            sim.set_self_collision_options(
                relief_factor=1.0,
                max_displacement_ratio=1.0,
                exclude_neighbors=True,
                enable_normal_untangling=True,
                max_iterations=128,
            )
            sim.step(dt=1.0 / 60.0, substeps=1)

            out_coords = np.empty(len(pos_init) * 3, dtype=np.float32)
            sim.get_positions(out_coords)
            cur = out_coords.reshape((-1, 3))
            z_inner = sign_z * 0.5 * float(cur[0, 2] + cur[1, 2])
            z_outer = sign_z * 0.5 * float(cur[4, 2] + cur[5, 2])
            return z_inner, z_outer

        z_in_up, z_out_up = run_case(pos_init=p_upright, sign_z=1.0)
        z_in_rot, z_out_rot = run_case(pos_init=p_rotated, sign_z=-1.0)

        self.assertAlmostEqual(
            z_out_rot,
            z_out_up,
            places=5,
            msg=f"剛体180度回転で離散 E-E untangling が非対称 (upright outer={z_out_up:.6f}, rotated outer={z_out_rot:.6f})",
        )
        self.assertAlmostEqual(
            z_in_rot,
            z_in_up,
            places=5,
            msg=f"剛体180度回転で内側エッジ反応が非対称 (upright inner={z_in_up:.6f}, rotated inner={z_in_rot:.6f})",
        )

    def test_ccd_envelope_no_overshoot_or_oscillation(self):
        """同方向多数重ね衝突および挟み込み条件で過剰押し出しやNaN・発振を起こさないこと"""
        # 1. 上下に同じ動的屋根エッジ 4 枚を重ね、落下屋根 1 枚が同時 CCD しても 1 枚分の厚みを大きく超えて跳ねないこと
        z_b = 0.002
        verts = []
        edges = []
        faces = []
        for k in range(4):
            off = len(verts)
            # 同一空間内でわずかに XY をズラした4枚の静的屋根（同方向 +Z 反発のみ）
            dx = (k - 1.5) * 0.001
            verts.extend(
                [
                    [-0.04, dx, 0.0],
                    [0.04, dx, 0.0],
                    [0.0, dx + 0.02, -0.04],
                    [0.0, dx - 0.02, -0.04],
                ]
            )
            edges.extend([[off, off + 1], [off + 1, off + 2], [off + 2, off], [off, off + 3], [off + 3, off + 1]])
            faces.extend([[off, off + 1, off + 2], [off + 1, off, off + 3]])

        b_off = len(verts)
        verts.extend(
            [
                [0.0, -0.04, z_b],
                [0.0, 0.04, z_b],
                [0.02, 0.0, z_b + 0.04],
                [-0.02, 0.0, z_b + 0.04],
            ]
        )
        edges.extend([[b_off, b_off + 1], [b_off + 1, b_off + 2], [b_off + 2, b_off], [b_off, b_off + 3], [b_off + 3, b_off + 1]])
        faces.extend([[b_off, b_off + 1, b_off + 2], [b_off + 1, b_off, b_off + 3]])

        pos = np.array(verts, dtype=np.float32)
        sim = taremin_cloth_core.ClothSimulator(
            positions=pos,
            edges=np.array(edges, dtype=np.uint32),
            faces=np.array(faces, dtype=np.uint32),
            thickness=0.005,
            stiffness=10000.0,
            workgroup_size=64,
        )
        sim.set_gravity(0.0, 0.0, 0.0)
        sim.set_enable_self_collision(True)
        sim.set_enable_pair_cache(False)

        vels = np.zeros_like(pos)
        vels[b_off:, 2] = -1.5
        sim.set_positions_and_velocities(pos, vels)
        sim.step(dt=1.0 / 60.0, substeps=1)

        out = np.empty(len(pos) * 3, dtype=np.float32)
        sim.get_positions(out)
        cur = out.reshape((-1, 3))
        self.assertFalse(np.isnan(cur).any(), "多重重なり CCD で NaN が発生してはならない")
        z_b_mid = 0.5 * float(cur[b_off, 2] + cur[b_off + 1, 2])
        self.assertGreater(z_b_mid, float(cur[0, 2]), "落下エッジが下側群を突き抜けてはならない")
        self.assertLess(z_b_mid, 0.025, f"多重同方向接触で過大な跳ね返りが発生してはならない (z_b_mid={z_b_mid:.6f})")


if __name__ == "__main__":
    unittest.main()
