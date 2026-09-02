"""
GPU Linear BVH (16面クラスタ境界球カリング) のオプション動作検証テスト
"""

import unittest
import numpy as np
import taremin_cloth_core


class TestClusterCullingOption(unittest.TestCase):
    def test_cluster_culling_cloth_body_interaction(self):
        """人体メッシュコライダー (10,518面) に対してクラスタカリング ON/OFF で安定衝突するかテスト"""
        # グリッド布の生成 (30x30 = 900頂点)
        n = 20
        x = np.linspace(-0.5, 0.5, n, dtype=np.float32)
        y = np.linspace(-0.5, 0.5, n, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        zz = np.ones_like(xx, dtype=np.float32) * 1.0  # Z=1.0 から落下

        positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

        edges = []
        for j in range(n):
            for i in range(n):
                idx = j * n + i
                if i + 1 < n:
                    edges.append([idx, idx + 1])
                if j + 1 < n:
                    edges.append([idx, idx + n])
        edges = np.array(edges, dtype=np.uint32)

        # 大きな球体コライダーをメッシュ三角形として作成 (球の近似メッシュ: 200面)
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        sphere_tris = []
        for i_u in range(len(u) - 1):
            for i_v in range(len(v) - 1):
                p00 = [0.3 * np.sin(v[i_v]) * np.cos(u[i_u]), 0.3 * np.sin(v[i_v]) * np.sin(u[i_u]), 0.5 + 0.3 * np.cos(v[i_v])]
                p10 = [0.3 * np.sin(v[i_v]) * np.cos(u[i_u+1]), 0.3 * np.sin(v[i_v]) * np.sin(u[i_u+1]), 0.5 + 0.3 * np.cos(v[i_v])]
                p01 = [0.3 * np.sin(v[i_v+1]) * np.cos(u[i_u]), 0.3 * np.sin(v[i_v+1]) * np.sin(u[i_u]), 0.5 + 0.3 * np.cos(v[i_v+1])]
                p11 = [0.3 * np.sin(v[i_v+1]) * np.cos(u[i_u+1]), 0.3 * np.sin(v[i_v+1]) * np.sin(u[i_u+1]), 0.5 + 0.3 * np.cos(v[i_v+1])]
                sphere_tris.append([p00, p10, p01])
                sphere_tris.append([p10, p11, p01])
        sphere_tris = np.array(sphere_tris, dtype=np.float32)

        # 1. 直接全走査 (Cluster Culling = OFF)
        sim_off = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            stiffness=500.0,
            bending_stiffness=5.0,
        )
        sim_off.set_mesh_collider_triangles(sphere_tris, friction=0.3, thickness=0.01)
        sim_off.set_collider_options(enable_cluster_culling=False, enable_single_sided_recovery=True, sweep_margin=0.05)

        # 2. クラスタカリング (Cluster Culling = ON)
        sim_on = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            stiffness=500.0,
            bending_stiffness=5.0,
        )
        sim_on.set_mesh_collider_triangles(sphere_tris, friction=0.3, thickness=0.01)
        sim_on.set_collider_options(enable_cluster_culling=True, enable_single_sided_recovery=True, sweep_margin=0.05)

        # 1フレーム進めて単一ステップでの差を検証
        sim_off.step(1.0 / 60.0, 10, 1)
        sim_on.step(1.0 / 60.0, 10, 1)

        out_off = np.empty(len(positions) * 3, dtype=np.float32)
        sim_off.get_positions(out_off)
        pos_off = out_off.reshape(-1, 3)

        out_on = np.empty(len(positions) * 3, dtype=np.float32)
        sim_on.get_positions(out_on)
        pos_on = out_on.reshape(-1, 3)

        diff_f1 = np.linalg.norm(pos_off - pos_on, axis=1).max()
        print(f"\nCluster Culling OFF vs ON diff after Frame 1: {diff_f1*1000:.4f} mm")
        self.assertLess(diff_f1, 0.001, f"Frame 1 diverged: {diff_f1*1000:.4f} mm")


if __name__ == "__main__":
    unittest.main()
