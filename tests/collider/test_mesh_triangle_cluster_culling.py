"""
GPU Linear BVH クラスタカリング (16面グループ境界球) の安全性と安定性テスト
外部ログファイルに依存せず、プロシージャルメッシュコライダー（UV球800面/50クラスタ）を用いて
決定論的かつ高速にクラスタカリング ON/OFF の整合性と安定性を検証します。
"""

import os
import sys
import unittest
import numpy as np

# プロジェクトルートとPythonパッケージのパスを追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
python_path = os.path.join(project_root, "python")
if python_path not in sys.path:
    sys.path.insert(0, python_path)

import taremin_cloth_core


def _generate_sphere_mesh_triangles(radius=0.5, u_div=20, v_div=20):
    """
    UV球の三角形メッシュ (shape [N, 3, 3]) をプロシージャル生成
    u_div=20, v_div=20 の場合、20 * 20 * 2 = 800 個の三角形（50クラスタ）を生成
    """
    verts = []
    for i in range(u_div + 1):
        lat = np.pi * (-0.5 + float(i) / u_div)
        for j in range(v_div):
            lon = 2 * np.pi * float(j) / v_div
            x = radius * np.cos(lat) * np.cos(lon)
            y = radius * np.cos(lat) * np.sin(lon)
            z = radius * np.sin(lat)
            verts.append([x, y, z])
    verts = np.array(verts, dtype=np.float32)

    tris = []
    for i in range(u_div):
        for j in range(v_div):
            next_j = (j + 1) % v_div
            i0 = i * v_div + j
            i1 = i * v_div + next_j
            i2 = (i + 1) * v_div + j
            i3 = (i + 1) * v_div + next_j
            tris.append([verts[i0], verts[i1], verts[i2]])
            tris.append([verts[i1], verts[i3], verts[i2]])

    return np.array(tris, dtype=np.float32)


class TestMeshTriangleClusterCulling(unittest.TestCase):
    """GPU Linear BVH クラスタカリング (16面グループ境界球) の安全性と安定性テスト"""

    def test_cluster_culling_on_vs_off_stability(self):
        """プロシージャル球メッシュコライダーでの クラスタカリング ON/OFF 安定性・整合性テスト"""
        # 1. 10x10 (100頂点) の布グリッドを球コライダー頭頂部の上方に配置
        nx, ny = 10, 10
        dx = 0.08
        positions = []
        inv_masses = []
        for y in range(ny):
            for x in range(nx):
                px = (x - nx / 2.0) * dx
                py = (y - ny / 2.0) * dx
                pz = 0.8  # 球の頭頂部 (z=0.5) の上方から落下
                positions.append([px, py, pz])
                inv_masses.append(1.0)

        positions = np.array(positions, dtype=np.float32)
        inv_masses = np.array(inv_masses, dtype=np.float32)

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

        # 800面の球コライダーを生成 (16面/クラスタ -> 50クラスタ、カリング閾値32面を十分超える)
        collider_triangles = _generate_sphere_mesh_triangles(radius=0.5, u_div=20, v_div=20)
        self.assertGreaterEqual(len(collider_triangles), 32, "クラスタカリング発動要件(32面以上)を満たすこと")

        def run_simulation(enable_cluster_culling: bool) -> np.ndarray:
            sim = taremin_cloth_core.ClothSimulator(
                positions=positions.copy(),
                edges=edges,
                faces=faces,
                inv_masses=inv_masses,
                thickness=0.005,
                stiffness=1000.0,
            )
            sim.set_gravity(0.0, 0.0, -9.81)
            sim.set_damping(0.05)
            sim.set_mesh_collider_triangles(collider_triangles, friction=0.3, thickness=0.01)
            sim.set_collider_options(
                enable_cluster_culling=enable_cluster_culling,
                enable_single_sided_recovery=True,
                sweep_margin=0.05,
            )

            out_coords = np.zeros(len(positions) * 3, dtype=np.float32)
            for frame in range(15):
                sim.step(dt=1.0 / 60.0, substeps=15)

            sim.get_positions(out_coords)
            return out_coords.reshape((-1, 3))

        # 2. オプション OFF (直接走査) でシミュレーション実行
        pos_off = run_simulation(enable_cluster_culling=False)

        # 3. オプション ON (16面クラスタカリング) でシミュレーション実行
        pos_on = run_simulation(enable_cluster_culling=True)

        # 4. 検証: NaN / Inf が含まれないこと
        self.assertFalse(np.isnan(pos_on).any(), "Cluster culling produced NaN positions")
        self.assertFalse(np.isinf(pos_on).any(), "Cluster culling produced Inf positions")

        # 5. 重心位置の一致 (大域的に同一の軌道、差分 < 10mm)
        center_off = pos_off.mean(axis=0)
        center_on = pos_on.mean(axis=0)
        center_diff = np.linalg.norm(center_off - center_on)
        self.assertLess(center_diff, 0.010, f"Center of mass diverged between ON and OFF: {center_diff*1000:.4f} mm")

        # 6. 最大局所差分 (許容誤差 20mm 以内)
        max_diff = np.linalg.norm(pos_off - pos_on, axis=1).max()
        self.assertLess(max_diff, 0.020, f"Max local diff between ON and OFF too large: {max_diff*1000:.4f} mm")


if __name__ == "__main__":
    unittest.main()
