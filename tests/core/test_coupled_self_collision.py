"""
Coupled自己衝突（RELAXATION / FULL_COUPLED）の単体・結合テスト
Python側からの設定切り替えおよびエッジ伸び抑制効果を自動検証する。
"""

import unittest
import numpy as np
import taremin_cloth_core


def create_test_pressing_cloth_pair(nx=20, ny=20, dx=0.04):
    """上下に対向して互いに押し付け合う2枚の布メッシュを生成"""
    n_per = nx * ny
    pos = []
    inv_m = []

    # 1. 下側の布（Z=0.0、外周を固定）
    for y in range(ny):
        for x in range(nx):
            pos.append([x * dx, y * dx, 0.0])
            if y == 0 or y == ny - 1 or x == 0 or x == nx - 1:
                inv_m.append(0.0)
            else:
                inv_m.append(1.0)

    # 2. 上側の布（Z=0.02、厚み thickness=0.02 に対して押し付け状態、外周固定）
    for y in range(ny):
        for x in range(nx):
            pos.append([x * dx, y * dx, 0.02])
            if y == 0 or y == ny - 1 or x == 0 or x == nx - 1:
                inv_m.append(0.0)
            else:
                inv_m.append(1.0)

    edges = []
    faces = []

    for offset in [0, n_per]:
        for y in range(ny):
            for x in range(nx):
                idx = offset + y * nx + x
                if x + 1 < nx:
                    edges.append([idx, idx + 1])
                if y + 1 < ny:
                    edges.append([idx, idx + nx])
                if x + 1 < nx and y + 1 < ny:
                    faces.append([idx, idx + 1, idx + nx + 1])
                    faces.append([idx, idx + nx + 1, idx + nx])

    return (
        np.array(pos, dtype=np.float32),
        np.array(edges, dtype=np.uint32),
        np.array(faces, dtype=np.uint32),
        np.array(inv_m, dtype=np.float32),
    )


class TestCoupledSelfCollision(unittest.TestCase):
    """Coupled自己衝突モードの動作検証"""

    def test_coupled_self_collision_stretch_suppression(self):
        """RELAXATION および FULL_COUPLED モードでエッジ伸びが有意に抑制されることを検証"""
        pos, edges, faces, inv_m = create_test_pressing_cloth_pair(20, 20)
        v0 = pos[edges[:, 0]]
        v1 = pos[edges[:, 1]]
        rest_lens = np.linalg.norm(v1 - v0, axis=1)

        def simulate_and_get_max_strain(mode: int, relax_iters: int) -> float:
            sim = taremin_cloth_core.ClothSimulator(
                positions=pos.copy(),
                edges=edges,
                faces=faces,
                inv_masses=inv_m,
                thickness=0.02,
                stiffness=1000.0,
                workgroup_size=64,
            )
            sim.set_gravity(0.0, 0.0, 0.0)
            sim.set_enable_self_collision(True)
            sim.set_self_collision_options(
                relief_factor=1.0,
                max_displacement_ratio=0.2,
                exclude_neighbors=True,
                enable_normal_untangling=False,
            )
            sim.set_coupled_self_collision_options(mode, relax_iters)

            for _ in range(30):
                sim.step(dt=1.0 / 60.0, substeps=10, solver_iterations=2)

            out_coords = np.empty(len(pos) * 3, dtype=np.float32)
            sim.get_positions(out_coords)
            cur_pos = out_coords.reshape((-1, 3))
            cur_v0 = cur_pos[edges[:, 0]]
            cur_v1 = cur_pos[edges[:, 1]]
            cur_lens = np.linalg.norm(cur_v1 - cur_v0, axis=1)
            strains = (cur_lens - rest_lens) / rest_lens
            return float(np.max(strains) * 100.0)

        # 1. モード0 (OFF / 従来方式)
        strain_off = simulate_and_get_max_strain(mode=0, relax_iters=0)

        # 2. モード1 (RELAXATION: Post-Relaxation 2 iters)
        strain_relax = simulate_and_get_max_strain(mode=1, relax_iters=2)

        # 3. モード3 (FULL_COUPLED: 反復内同調 + Post-Relaxation 1 iter)
        strain_coupled = simulate_and_get_max_strain(mode=3, relax_iters=1)

        print(f"\n[Test Coupled Self Collision] OFF Max Strain: {strain_off:.3f}%")
        print(f"[Test Coupled Self Collision] RELAXATION Max Strain: {strain_relax:.3f}%")
        print(f"[Test Coupled Self Collision] FULL_COUPLED Max Strain: {strain_coupled:.3f}%")

        # 従来の自己衝突では 2.5% 以上の伸びが発生
        self.assertGreater(strain_off, 2.5, "OFFモードで押し付けによる伸びが発生していること")

        # RELAXATION モードでは OFF よりも大幅に伸びが抑えられること (30%以上低減)
        self.assertLess(strain_relax, strain_off * 0.8, "RELAXATIONモードで伸びが有意に抑制されること")

        # FULL_COUPLED モードでも OFF よりも大幅に伸びが抑えられること
        self.assertLess(strain_coupled, strain_off * 0.8, "FULL_COUPLEDモードで伸びが有意に抑制されること")


if __name__ == "__main__":
    unittest.main()
