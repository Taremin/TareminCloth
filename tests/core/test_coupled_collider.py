# -*- coding: utf-8 -*-
"""
Coupled Collider (反復内同調) および Decoupled Collider (反復外1回解決) の単体・結合テスト
コライダー衝突の切り替え、非貫通保証、健全なエッジ歪み率を自動検証する。
"""

import unittest
import numpy as np
import taremin_cloth_core as core


class TestCoupledCollider(unittest.TestCase):
    """Coupled Collider オプションの動作検証"""

    def create_cloth_and_collider(self, nx=15, ny=15, size=0.5, z=0.25):
        """テスト用の布メッシュと三角形コライダーを生成"""
        dx = size / (nx - 1)
        dy = size / (ny - 1)
        pos = []
        for y in range(ny):
            for x in range(nx):
                pos.append([x * dx - size * 0.5, y * dy - size * 0.5, z])
        pos = np.array(pos, dtype=np.float32)

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
        edges = np.array(edges, dtype=np.uint32)
        faces = np.array(faces, dtype=np.uint32)

        # 面積に応じた物理逆質量
        masses = np.zeros(len(pos), dtype=np.float64)
        for f in faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            ab = pos[b] - pos[a]
            ac = pos[c] - pos[a]
            share = 0.5 * float(np.linalg.norm(np.cross(ab, ac))) * 0.15 / 3.0
            masses[a] += share
            masses[b] += share
            masses[c] += share
        masses = np.maximum(masses, 1e-9)
        inv_m = (1.0 / masses).astype(np.float32)

        # 水平なコライダー平面 (Z=0.0, 2つの三角形)
        col_tris = np.array([
            [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0]],
            [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
        ], dtype=np.float32)

        return pos, edges, faces, inv_m, col_tris

    def test_coupled_and_decoupled_collider_stability(self):
        """Coupled (True) と Decoupled (False) の両方で貫通がなく、伸びが健全範囲に収まることを検証"""
        for coupled in [True, False]:
            pos, edges, faces, inv_m, col_tris = self.create_cloth_and_collider()
            v0 = pos[edges[:, 0]]
            v1 = pos[edges[:, 1]]
            rest_lens = np.linalg.norm(v1 - v0, axis=1)

            sim = core.ClothSimulator(
                positions=pos.copy(),
                edges=edges,
                faces=faces,
                inv_masses=inv_m,
                thickness=0.01,
                stiffness=1000.0,
                workgroup_size=32,
            )
            sim.set_gravity(0.0, 0.0, -9.81)
            sim.set_enable_self_collision(False)
            sim.set_mesh_collider_triangles(
                col_tris, friction=0.3, thickness=0.01, restitution=0.0, single_sided=False, attributes=None
            )
            sim.set_coupled_collider(coupled)

            for _ in range(60):
                sim.step(dt=1.0 / 60.0, substeps=10)

            out_coords = np.empty(len(pos) * 3, dtype=np.float32)
            sim.get_positions(out_coords)
            final_pos = out_coords.reshape((-1, 3))

            # 1. 貫通検証: コライダー表面 (Z=0.0) - thickness (0.01) より上に布頂点が存在すること
            min_z = float(np.min(final_pos[:, 2]))
            self.assertGreater(
                min_z,
                -0.005,
                f"coupled={coupled}: コライダー表面下にめり込んでいないこと (min_z: {min_z:.4f}m)",
            )

            # 2. エッジ伸び率検証: 健全範囲 (< 8.0%) に収まること
            cur_v0 = final_pos[edges[:, 0]]
            cur_v1 = final_pos[edges[:, 1]]
            cur_lens = np.linalg.norm(cur_v1 - cur_v0, axis=1)
            max_strain = float(np.max((cur_lens - rest_lens) / rest_lens * 100.0))
            self.assertLess(
                max_strain,
                8.0,
                f"coupled={coupled}: 最大エッジ伸長率が8.0%未満であること (max_strain: {max_strain:.3f}%)",
            )


if __name__ == "__main__":
    unittest.main()
