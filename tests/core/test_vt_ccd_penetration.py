"""
V-T CCD 非緩和（VT_CCD_RELAXATION = 1.0）による自己貫通遮断の回帰テスト。

Direct モードおよび Pair Cache モードにおいて、V-T CCD 変位を 0.5 倍へ緩和した際に
発生していた高相対速度衝突・パッチ落下・Grab 解放直後の裏抜け貫通が起きないことを検証する。
"""

import os
import sys
import unittest
from typing import Optional

import numpy as np
import taremin_cloth_core

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PYTHON_PKG = os.path.join(_REPO_ROOT, "python")
if _PYTHON_PKG not in sys.path:
    sys.path.insert(0, _PYTHON_PKG)

from taremin_cloth.analysis import find_triangle_intersections


def _create_grid_mesh(
    *,
    nx: int,
    ny: int,
    dx: float,
    z: float,
    pin_mode: str = "none",
):
    """中心原点の平面格子布メッシュとピン用固定フラグを生成する。"""
    positions = []
    fixed_flags = []
    for y in range(ny):
        for x in range(nx):
            positions.append([(x - (nx - 1) / 2.0) * dx, (y - (ny - 1) / 2.0) * dx, z])
            if pin_mode == "all":
                fixed_flags.append(True)
            elif pin_mode == "border":
                fixed_flags.append(x in (0, nx - 1) or y in (0, ny - 1))
            else:
                fixed_flags.append(False)

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

    return (
        np.array(positions, dtype=np.float32),
        np.array(edges, dtype=np.uint32),
        np.array(faces, dtype=np.uint32),
        np.array(fixed_flags, dtype=bool),
    )


def _compute_areal_inv_masses(
    pos: np.ndarray,
    faces: np.ndarray,
    *,
    density: float = 0.15,
    fixed_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """三角形面積から各頂点の物理質量を算出し、逆質量配列を返す。"""
    masses = np.zeros(len(pos), dtype=np.float64)
    for tri in np.asarray(faces, dtype=np.int64):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if a == b or b == c or c == a:
            continue
        ab = pos[b].astype(np.float64) - pos[a].astype(np.float64)
        ac = pos[c].astype(np.float64) - pos[a].astype(np.float64)
        share = 0.5 * float(np.linalg.norm(np.cross(ab, ac))) * density / 3.0
        masses[a] += share
        masses[b] += share
        masses[c] += share

    inv = (1.0 / np.maximum(masses, 1e-9)).astype(np.float32)
    if fixed_mask is not None:
        inv[np.asarray(fixed_mask, dtype=bool)] = 0.0
    return inv


def _read_positions(sim: taremin_cloth_core.ClothSimulator, num_verts: int) -> np.ndarray:
    """シミュレータから現在の頂点座標 (N, 3) を取得する。"""
    buf = np.empty(num_verts * 3, dtype=np.float32)
    sim.get_positions(buf)
    return buf.reshape((-1, 3))


class TestVtCcdPenetration(unittest.TestCase):
    """V-T CCD 非緩和 (VT_CCD_RELAXATION = 1.0) の自己貫通防止回帰テスト"""

    def test_direct_free_cloth_single_substep_blocking(self):
        """Direct モード: 自由布2枚の 1 サブステップ V-T 衝突で自己交差が発生しないこと"""
        pa, ea, fa, _ = _create_grid_mesh(nx=7, ny=7, dx=0.03, z=0.0, pin_mode="none")
        pb, eb, fb, _ = _create_grid_mesh(nx=5, ny=5, dx=0.025, z=0.004, pin_mode="none")
        na = len(pa)
        pos = np.vstack([pa, pb])
        edges = np.vstack([ea, eb + na])
        faces = np.vstack([fa, fb + na])
        inv_m = _compute_areal_inv_masses(pos, faces, density=0.15)

        sim = taremin_cloth_core.ClothSimulator(
            positions=pos.copy(),
            edges=edges,
            faces=faces,
            inv_masses=inv_m,
            thickness=0.005,
            stiffness=2000.0,
            bending_stiffness=15.0,
            enable_pair_cache=False,
        )
        sim.set_gravity(0.0, 0.0, 0.0)
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(relief_factor=1.0, max_displacement_ratio=0.5)

        vel = np.zeros_like(pos)
        vel[na:, 2] = -1.2
        sim.set_positions_and_velocities(pos, vel)

        sim.step(1.0 / 60.0, 1)
        p1 = _read_positions(sim, len(pos))
        self.assertFalse(np.isnan(p1).any(), "Frame 1 で NaN が発生してはならない")
        isect_1 = find_triangle_intersections(p1, faces)
        self.assertEqual(
            len(isect_1),
            0,
            f"Direct 1-substep V-T 衝突直後に自己交差が発生した: {len(isect_1)} ペア",
        )

        for _ in range(4):
            sim.step(1.0 / 60.0, 1)
        p5 = _read_positions(sim, len(pos))
        self.assertFalse(np.isnan(p5).any(), "Frame 5 で NaN が発生してはならない")
        isect_5 = find_triangle_intersections(p5, faces)
        self.assertEqual(
            len(isect_5),
            0,
            f"Direct 1-substep V-T 衝突の継続フレームで自己交差が残存した: {len(isect_5)} ペア",
        )

    def test_pair_cache_patch_drop_no_penetration(self):
        """Pair Cache モード: 5x5 パッチ落下で固定床への裏抜けと自己交差が起きないこと"""
        pa, ea, fa, fixed_a = _create_grid_mesh(nx=11, ny=11, dx=0.03, z=0.0, pin_mode="all")
        pb, eb, fb, fixed_b = _create_grid_mesh(nx=5, ny=5, dx=0.025, z=0.004, pin_mode="none")
        na = len(pa)
        pos = np.vstack([pa, pb])
        edges = np.vstack([ea, eb + na])
        faces = np.vstack([fa, fb + na])
        fixed = np.concatenate([fixed_a, fixed_b])
        inv_m = _compute_areal_inv_masses(pos, faces, density=0.15, fixed_mask=fixed)

        for substeps in (4, 8):
            with self.subTest(substeps=substeps):
                sim = taremin_cloth_core.ClothSimulator(
                    positions=pos.copy(),
                    edges=edges,
                    faces=faces,
                    inv_masses=inv_m.copy(),
                    thickness=0.005,
                    stiffness=2000.0,
                    bending_stiffness=15.0,
                    enable_pair_cache=True,
                )
                sim.set_gravity(0.0, 0.0, 0.0)
                sim.set_enable_self_collision(True)
                sim.set_enable_pair_cache_final_fallback(False)
                sim.set_self_collision_options(relief_factor=1.0, max_displacement_ratio=0.5)

                vel = np.zeros_like(pos)
                vel[na:, 2] = -1.2
                sim.set_positions_and_velocities(pos, vel)
                sim.step(1.0 / 60.0, substeps)

                p = _read_positions(sim, len(pos))
                self.assertFalse(np.isnan(p).any(), f"substeps={substeps} で NaN が発生")
                below_zero = int(np.sum(p[na:, 2] < 0.0))
                min_z_mm = float(p[na:, 2].min() * 1000.0)
                isect = find_triangle_intersections(p, faces)
                self.assertEqual(
                    below_zero,
                    0,
                    f"Pair Cache (substeps={substeps}) で固定床 Z=0 未満へ裏抜けた: below0={below_zero}, min_z={min_z_mm:.3f}mm",
                )
                self.assertEqual(
                    len(isect),
                    0,
                    f"Pair Cache (substeps={substeps}) で自己交差が発生した: {len(isect)} ペア",
                )

    def test_direct_grab_release_immediate_recovery(self):
        """Direct モード: Grab 押し込み解放直後に裏抜け頂点が残らず自己交差も解消されること"""
        pa, ea, fa, fixed_a = _create_grid_mesh(nx=9, ny=9, dx=0.03, z=0.0, pin_mode="border")
        pb, eb, fb, fixed_b = _create_grid_mesh(nx=9, ny=9, dx=0.03, z=0.02, pin_mode="none")
        na = len(pa)
        pos = np.vstack([pa, pb])
        edges = np.vstack([ea, eb + na])
        faces = np.vstack([fa, fb + na])
        fixed = np.concatenate([fixed_a, fixed_b])
        inv_m = _compute_areal_inv_masses(pos, faces, density=0.15, fixed_mask=fixed)

        # shear_stiffness 省略時の後方互換値 (stiffness * 0.5 = 750.0) は X字対称 Shear 下で過度に硬い膜となるため標準値を明示
        sim = taremin_cloth_core.ClothSimulator(
            positions=pos.copy(),
            edges=edges,
            faces=faces,
            inv_masses=inv_m,
            thickness=0.005,
            stiffness=1500.0,
            shear_stiffness=250.0,
            bending_stiffness=15.0,
            enable_pair_cache=False,
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(relief_factor=1.0, max_displacement_ratio=0.5)

        grab_v = na + 40
        for step_idx in range(1, 16):
            sim.set_pin(grab_v, [0.0, 0.0, 0.02 - 0.10 * (step_idx / 15.0)], 1.0)
            sim.step(1.0 / 60.0, 10)

        sim.release_pin(grab_v)

        # 解放直後 1 フレーム目で上布頂点が下布対応頂点より下へ抜けておらず、異布間交差がゼロであること
        sim.step(1.0 / 60.0, 10)
        p_r1 = _read_positions(sim, len(pos))
        self.assertFalse(np.isnan(p_r1).any(), "Grab 解放直後 (r1) で NaN が発生")
        min_rel_z_r1 = float((p_r1[na:, 2] - p_r1[:na, 2]).min())
        isect_r1 = find_triangle_intersections(p_r1, faces)
        n_faces_a = len(fa)
        cross_isect_r1 = [(a, b) for a, b in isect_r1 if (a < n_faces_a) != (b < n_faces_a)]
        self.assertGreaterEqual(
            min_rel_z_r1,
            0.0,
            f"Direct Grab 解放 1F 後に上布が下布より下へ裏抜けた: min(B_z - A_z)={min_rel_z_r1 * 1000.0:.2f}mm",
        )
        self.assertEqual(
            len(cross_isect_r1),
            0,
            f"Direct Grab 解放 1F 後に異布間の自己交差が発生した: {len(cross_isect_r1)} ペア",
        )

        # 解放 2 フレーム目で全自己交差と Z=0 未満の裏抜けが解消されていること
        sim.step(1.0 / 60.0, 10)
        p_r2 = _read_positions(sim, len(pos))
        self.assertFalse(np.isnan(p_r2).any(), "Grab 解放 2F 後 (r2) で NaN が発生")
        below_r2 = int(np.sum(p_r2[na:, 2] < 0.0))
        isect_r2 = find_triangle_intersections(p_r2, faces)
        self.assertEqual(
            below_r2,
            0,
            f"Direct Grab 解放 2F 後に裏抜け頂点が残存した: below0={below_r2}",
        )
        self.assertEqual(
            len(isect_r2),
            0,
            f"Direct Grab 解放 2F 後に自己交差が残存した: {len(isect_r2)} ペア",
        )

        # 解放 5 フレーム目でも引き続き自己交差と裏抜けがゼロであること
        for _ in range(3):
            sim.step(1.0 / 60.0, 10)
        p_r5 = _read_positions(sim, len(pos))
        self.assertFalse(np.isnan(p_r5).any(), "Grab 解放後 (r5) で NaN が発生")
        below_r5 = int(np.sum(p_r5[na:, 2] < 0.0))
        isect_r5 = find_triangle_intersections(p_r5, faces)
        self.assertEqual(
            below_r5,
            0,
            f"Direct Grab 解放 5F 後に裏抜け頂点が残存した: below0={below_r5}",
        )
        self.assertEqual(
            len(isect_r5),
            0,
            f"Direct Grab 解放 5F 後に自己交差が残存した: {len(isect_r5)} ペア",
        )


if __name__ == "__main__":
    unittest.main()
