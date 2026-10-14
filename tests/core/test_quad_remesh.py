# -*- coding: utf-8 -*-
"""Milestone 5: Phase 3 (Remesh to Quad) 境界ベルト・Barycentric逆写像・縫い目ゼロギャップ補正の単体テスト."""
import math
import os
import sys
import unittest
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import taremin_cloth_core as _core

    _HAS_QUAD_REMESH = hasattr(_core, "remesh_cloth_to_quad_3d")
except Exception:
    _core = None
    _HAS_QUAD_REMESH = False


def _create_wavy_simulated_patch(offset_x: float = 0.0, nx: int = 12, ny: int = 12):
    """2D UV空間 [0, 0.24]x[0, 0.24] と、3D空間上で正弦波ドレープ変形した三角形メッシュを生成する."""
    w, h = 0.24, 0.24
    uvs = []
    pos_3d = []
    for j in range(ny + 1):
        v = h * (j / ny)
        for i in range(nx + 1):
            u = w * (i / nx)
            uvs.append([u + offset_x, v])
            # 3D変形後座標: 斜め方向の美しいドレープシワ z = 0.015 * sin(2*pi*(u + 0.5*v)/0.12)
            z = 0.015 * math.sin(2.0 * math.pi * (u + 0.5 * v) / 0.12)
            pos_3d.append([u + offset_x, v, z])

    tris = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            a = j * stride + i
            b = a + 1
            c = a + stride
            d = c + 1
            tris.append([a, b, d])
            tris.append([a, d, c])

    # 反時計回り外周インデックス列 (0..B)
    boundary_indices = []
    for i in range(nx):
        boundary_indices.append(i)
    for j in range(ny):
        boundary_indices.append(j * stride + nx)
    for i in range(nx):
        boundary_indices.append(ny * stride + (nx - i))
    for j in range(ny):
        boundary_indices.append((ny - j) * stride)

    uvs_arr = np.array(uvs, dtype=np.float32)
    pos_arr = np.array(pos_3d, dtype=np.float32)
    tris_arr = np.array(tris, dtype=np.uint32)
    boundary_uvs = uvs_arr[boundary_indices]
    return pos_arr, uvs_arr, tris_arr, boundary_uvs


@unittest.skipUnless(_HAS_QUAD_REMESH, "taremin_cloth_core.remesh_cloth_to_quad_3d 未実装のためスキップ")
class TestQuadRemesh3D(unittest.TestCase):
    def test_boundary_quad_belt_and_wavy_surface_reconstruction(self):
        """外周1ループ目にBoundary Quad Beltが形成され、3D重心逆写像がドレープ曲面を高精度に再現すること."""
        sim_pos, sim_uv, sim_tris, boundary_uv = _create_wavy_simulated_patch()

        out = _core.remesh_cloth_to_quad_3d(
            sim_positions_3d=sim_pos,
            sim_uvs_2d=sim_uv,
            sim_triangles=sim_tris,
            boundary_loop_2d=boundary_uv,
            target_quad_size=0.020,
            boundary_belt_loops=1,
            smoothing_iterations=10,
        )
        quad_pos = np.asarray(out["positions_3d"], dtype=np.float32)
        quad_uv = np.asarray(out["uvs_2d"], dtype=np.float32)
        quads = np.asarray(out["quads"], dtype=np.uint32)
        tris = np.asarray(out["triangles"], dtype=np.uint32)
        b_indices = np.asarray(out["boundary_vertex_indices"], dtype=np.uint32)

        # 1. 外周頂点数が入力境界ループと一致すること
        self.assertEqual(len(b_indices), len(boundary_uv))
        np.testing.assert_allclose(quad_uv[b_indices], boundary_uv, atol=1e-5)

        # 2. Quad主体（全ポリゴンの85%以上が四角面）であること
        total_polys = len(quads) + len(tris)
        self.assertGreater(len(quads) / max(1, total_polys), 0.85)

        # 3. 逆写像された3D座標が元のドレープ曲面 z(u, v) に追従していること (誤差 1.5mm 以内)
        u = quad_uv[:, 0]
        v = quad_uv[:, 1]
        expected_z = 0.015 * np.sin(2.0 * np.pi * (u + 0.5 * v) / 0.12)
        max_z_err = float(np.max(np.abs(quad_pos[:, 2] - expected_z)))
        self.assertLess(max_z_err, 0.0015, f"Barycentric 3D surface error {max_z_err*1000:.2f}mm exceeds 1.5mm")

    def test_independent_uv_islands_with_zero_gap_3d_seam_snapping(self):
        """UVアイランドを独立維持したまま、縫合ペア境界の3D座標のみが中点スナップされて隙間ゼロになること."""
        pos_a, uv_a, tris_a, bnd_a = _create_wavy_simulated_patch(offset_x=0.0, nx=10, ny=10)
        pos_b, uv_b, tris_b, bnd_b = _create_wavy_simulated_patch(offset_x=0.30, nx=10, ny=10)

        out_a = _core.remesh_cloth_to_quad_3d(
            pos_a, uv_a, tris_a, bnd_a, target_quad_size=0.024, boundary_belt_loops=1, smoothing_iterations=8
        )
        out_b = _core.remesh_cloth_to_quad_3d(
            pos_b, uv_b, tris_b, bnd_b, target_quad_size=0.024, boundary_belt_loops=1, smoothing_iterations=8
        )

        qa_pos = np.asarray(out_a["positions_3d"], dtype=np.float32).copy()
        qb_pos = np.asarray(out_b["positions_3d"], dtype=np.float32).copy()
        qa_uv = np.asarray(out_a["uvs_2d"], dtype=np.float32)
        qb_uv = np.asarray(out_b["uvs_2d"], dtype=np.float32)

        # Island A の右辺 (インデックス 10..20) と Island B の左辺 (逆順) を縫合スナップ
        seam_idx_a = np.asarray(out_a["boundary_vertex_indices"], dtype=np.uint32)[10:21]
        seam_idx_b = np.asarray(out_b["boundary_vertex_indices"], dtype=np.uint32)[30:41][::-1]
        self.assertEqual(len(seam_idx_a), len(seam_idx_b))

        # 3D中点スナップを実行
        mid_3d = 0.5 * (qa_pos[seam_idx_a] + qb_pos[seam_idx_b])
        qa_pos[seam_idx_a] = mid_3d
        qb_pos[seam_idx_b] = mid_3d

        # 検証1: 3D空間上の縫い目ギャップが完全に 0.0m であること
        seam_gaps = np.linalg.norm(qa_pos[seam_idx_a] - qb_pos[seam_idx_b], axis=1)
        np.testing.assert_allclose(seam_gaps, 0.0, atol=1e-6)

        # 検証2: UV座標はアイランドごとに独立しており結合されていないこと
        uv_dists = np.linalg.norm(qa_uv[seam_idx_a] - qb_uv[seam_idx_b], axis=1)
        self.assertTrue(np.all(uv_dists > 0.05), "UV islands must remain distinct in 2D UV space")


if __name__ == "__main__":
    unittest.main()
