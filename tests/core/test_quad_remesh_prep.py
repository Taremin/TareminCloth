# -*- coding: utf-8 -*-
"""Quadリメッシュ拡張コア・前処理・縫合結合の統合テスト (Blender不要)"""

import math
import os
import sys
import unittest

import numpy as np

# パス設定
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PYTHON_DIR = os.path.join(ROOT_DIR, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

import taremin_cloth_core as core
from taremin_cloth.utils.mesh_extract import extract_sewing_topology
from taremin_cloth.utils.quad_remesh_prep import (
    merge_island_outputs,
    prepare_quad_remesh_islands,
)


def _make_grid_panel(
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z: float,
    nx: int,
    ny: int,
    uv_x0: float,
    uv_x1: float,
    uv_y0: float,
    uv_y1: float,
):
    pos = []
    uv = []
    for j in range(ny + 1):
        ty = j / ny
        for i in range(nx + 1):
            tx = i / nx
            pos.append([x0 + tx * (x1 - x0), y0 + ty * (y1 - y0), z])
            uv.append([uv_x0 + tx * (uv_x1 - uv_x0), uv_y0 + ty * (uv_y1 - uv_y0)])
    tris = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            v0 = j * stride + i
            v1 = v0 + 1
            v2 = v0 + stride + 1
            v3 = v0 + stride
            tris.append([v0, v1, v2])
            tris.append([v0, v2, v3])
    pos_arr = np.array(pos, dtype=np.float32)
    uv_arr = np.array(uv, dtype=np.float32)
    tri_arr = np.array(tris, dtype=np.uint32)
    tri_uvs = uv_arr[tri_arr]
    return pos_arr, uv_arr, tri_arr, tri_uvs


def _poly_signed_area_2d(pts: np.ndarray) -> float:
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


class TestQuadRemeshPrepAndHybrid(unittest.TestCase):
    """前処理・縫合結合・拡張Hybrid充填コアの総合検証"""

    def test_prepare_uv_split_and_multiple_islands(self):
        # 3Dでは一部頂点を共有しているが UV では別アイランドに分かれた2パネル
        pos = np.array(
            [
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0], [2.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        tris = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]], dtype=np.uint32)
        # アイランドA (left [0.05..0.45]) と アイランドB (right [0.55..0.95])
        tri_uvs = np.array(
            [
                [[0.05, 0.05], [0.45, 0.05], [0.45, 0.95]],
                [[0.05, 0.05], [0.45, 0.95], [0.05, 0.95]],
                [[0.55, 0.05], [0.95, 0.05], [0.95, 0.95]],
                [[0.55, 0.05], [0.95, 0.95], [0.55, 0.95]],
            ],
            dtype=np.float32,
        )
        scene = prepare_quad_remesh_islands(pos, tris, triangle_uvs=tri_uvs)
        self.assertEqual(len(scene.islands), 2)
        for isl in scene.islands:
            self.assertEqual(len(isl.boundary_loop_2d), 4)
            self.assertEqual(len(isl.hole_loops_2d), 0)

    def test_prepare_hole_classification_and_nonmanifold_error(self):
        # 外周 0..3 に 1..2 の穴を持つメッシュ (4x4 のうち中央4マスをくり抜いた形状)
        xs = np.linspace(0.0, 4.0, 5, dtype=np.float32)
        ys = np.linspace(0.0, 4.0, 5, dtype=np.float32)
        pos = []
        uv = []
        for y in ys:
            for x in xs:
                pos.append([x, y, 0.0])
                uv.append([x / 4.0, y / 4.0])
        pos_arr = np.array(pos, dtype=np.float32)
        uv_arr = np.array(uv, dtype=np.float32)
        tris = []
        for j in range(4):
            for i in range(4):
                if 1 <= i <= 2 and 1 <= j <= 2:
                    continue
                v0 = j * 5 + i
                v1 = v0 + 1
                v2 = v0 + 6
                v3 = v0 + 5
                tris.append([v0, v1, v2])
                tris.append([v0, v2, v3])
        tri_arr = np.array(tris, dtype=np.uint32)
        tri_uvs = uv_arr[tri_arr]

        scene = prepare_quad_remesh_islands(pos_arr, tri_arr, triangle_uvs=tri_uvs)
        self.assertEqual(len(scene.islands), 1)
        isl = scene.islands[0]
        self.assertEqual(len(isl.boundary_loop_2d), 16)
        self.assertEqual(len(isl.hole_loops_2d), 1)
        self.assertEqual(len(isl.hole_loops_2d[0]), 8)

        # 非多様体（同じ向きの辺を共有する重複三角形）は ValueError
        bad_tris = np.vstack([tri_arr, tri_arr[:1]])
        bad_uvs = uv_arr[bad_tris]
        with self.assertRaises(ValueError):
            prepare_quad_remesh_islands(pos_arr, bad_tris, triangle_uvs=bad_uvs)

    def test_static_merged_and_pin_interpolation_with_normalized_uv(self):
        # 1m x 1m のパネル2枚（前後の身頃、z=0 と z=0.2）、正規化 UV 空間 [0.02..0.48], [0.52..0.98]
        p1, _, t1, uv1 = _make_grid_panel(
            x0=0.0, x1=1.0, y0=0.0, y1=1.0, z=0.0, nx=8, ny=8,
            uv_x0=0.02, uv_x1=0.48, uv_y0=0.02, uv_y1=0.48,
        )
        p2, _, t2, uv2 = _make_grid_panel(
            x0=0.0, x1=1.0, y0=0.0, y1=1.0, z=0.2, nx=8, ny=8,
            uv_x0=0.52, uv_x1=0.98, uv_y0=0.02, uv_y1=0.48,
        )
        n1 = len(p1)
        pos = np.vstack([p1, p2])
        tris = np.vstack([t1, t2 + np.uint32(n1)])
        tri_uvs = np.concatenate([uv1, uv2], axis=0)
        tri_pieces = np.concatenate([np.zeros(len(t1), dtype=np.int32), np.ones(len(t2), dtype=np.int32)])
        # 左右辺の頂点同士を縫合線 (Loose Edges) で接続
        sewing = []
        for j in range(9):
            left_1 = j * 9
            left_2 = n1 + j * 9
            right_1 = j * 9 + 8
            right_2 = n1 + j * 9 + 8
            sewing.append((left_1, left_2))
            sewing.append((right_1, right_2))

        # 上端 y=1.0 がピン重み 1.0、下端 y=0.0 が 0.0
        pins = pos[:, 1].astype(np.float32)

        scene = prepare_quad_remesh_islands(
            pos,
            tris,
            triangle_uvs=tri_uvs,
            loose_edges=sewing,
            triangle_piece_ids=tri_pieces,
            vertex_pin_weights=pins,
        )
        self.assertEqual(len(scene.islands), 2)

        outs = []
        for isl in scene.islands:
            out = core.remesh_cloth_to_quad_3d_ext(
                sim_positions_3d=isl.positions_3d,
                sim_uvs_2d=isl.uvs_2d,
                sim_triangles=isl.triangles.astype(np.uint32),
                boundary_loop_2d=isl.boundary_loop_2d,
                hole_loops_2d=isl.hole_loops_2d,
                target_quad_size=0.125,  # 1m / 8 = 0.125m -> 構造化 8x8 = 64 Quads になるはず
                boundary_belt_loops=1,
                smoothing_iterations=5,
                keep_structured_when_possible=True,
                target_size_in_3d=True,
            )
            outs.append(out)

        merged = merge_island_outputs(scene, outs, mode="STATIC_MERGED")
        self.assertEqual(len(merged.loose_edges), 0)
        self.assertEqual(len(merged.quads), 128)
        self.assertEqual(len(merged.triangles), 0)

        # 全シームが結合されたため、頂点数が (81 + 81 - 18) = 144 になっていること
        self.assertEqual(len(merged.positions_3d), 144)

        # コーナー UV はアイランド間で混ざっていないこと
        for qi in range(len(merged.quads)):
            isl_id = int(merged.quad_island_ids[qi])
            us = merged.quad_corner_uvs[qi, :, 0]
            if isl_id == 0:
                self.assertTrue(np.all(us <= 0.49))
            else:
                self.assertTrue(np.all(us >= 0.51))

        # ピンウェイト補間が 0..1 で連続保持されていること
        top_mask = merged.positions_3d[:, 1] > 0.99
        bot_mask = merged.positions_3d[:, 1] < 0.01
        self.assertTrue(np.allclose(merged.vertex_pin_weights[top_mask], 1.0, atol=1e-4))
        self.assertTrue(np.allclose(merged.vertex_pin_weights[bot_mask], 0.0, atol=1e-4))

    def test_keep_pattern_rebuilds_waypoint_seams(self):
        p1, _, t1, uv1 = _make_grid_panel(
            x0=0.0, x1=0.5, y0=0.0, y1=0.5, z=0.0, nx=4, ny=4,
            uv_x0=0.0, uv_x1=0.4, uv_y0=0.0, uv_y1=0.4,
        )
        p2, _, t2, uv2 = _make_grid_panel(
            x0=0.0, x1=0.5, y0=0.0, y1=0.5, z=0.4, nx=4, ny=4,
            uv_x0=0.5, uv_x1=0.9, uv_y0=0.0, uv_y1=0.4,
        )
        n1 = len(p1)
        n_mesh = n1 + len(p2)
        # BONE_WRAP 用の中継点 2個を追加
        waypoints = np.array([[0.55, 0.25, 0.13], [0.55, 0.25, 0.27]], dtype=np.float32)
        pos = np.vstack([p1, p2, waypoints])
        tris = np.vstack([t1, t2 + np.uint32(n1)])
        tri_uvs = np.concatenate([uv1, uv2], axis=0)

        # right_1 (row=2 -> v = 2*5 + 4 = 14) <-> wp0 -> wp1 <-> right_2 (n1 + 14)
        loose = [(14, n_mesh), (n_mesh, n_mesh + 1), (n_mesh + 1, n1 + 14)]

        scene = prepare_quad_remesh_islands(
            pos,
            tris,
            triangle_uvs=tri_uvs,
            loose_edges=loose,
        )
        self.assertEqual(len(scene.waypoint_seams), 1)

        outs = [
            core.remesh_cloth_to_quad_3d_ext(
                sim_positions_3d=isl.positions_3d,
                sim_uvs_2d=isl.uvs_2d,
                sim_triangles=isl.triangles.astype(np.uint32),
                boundary_loop_2d=isl.boundary_loop_2d,
                target_quad_size=0.125,
                target_size_in_3d=True,
            )
            for isl in scene.islands
        ]
        merged = merge_island_outputs(scene, outs, mode="KEEP_PATTERN")
        self.assertEqual(len(merged.loose_edges), 3)

        # 再構成された Loose Edge と面が再び WaypointSeam 1本として認識されること
        all_tris = merged.quads[:, :3] if len(merged.quads) > 0 else merged.triangles
        re_simple, re_wp = extract_sewing_topology(
            merged.loose_edges,
            faces_2d=all_tris,
            positions=merged.positions_3d,
        )
        self.assertEqual(len(re_simple), 0)
        self.assertEqual(len(re_wp), 1)
        self.assertEqual(len(re_wp[0].waypoint_indices), 2)

    def test_hybrid_robustness_concave_slit_and_invalid_inputs(self):
        # 1. 空配列・不正入力はパニックせず RuntimeError になること
        pos = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        empty_tris = np.zeros((0, 3), dtype=np.uint32)
        bnd = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        with self.assertRaises(RuntimeError):
            core.remesh_cloth_to_quad_3d(pos, uv, empty_tris, bnd, 0.1, 1, 5)

        # 2. 目標エッジ長相当でサンプルした8方向星型・U字・Tシャツ・円形穴付きで
        #    裏面ゼロ・NaNゼロ・角範囲・高いQuad率を維持すること
        pos, uv, tris, _ = _make_grid_panel(
            x0=0.0, x1=1.0, y0=0.0, y1=1.0, z=0.0, nx=4, ny=4,
            uv_x0=0.0, uv_x1=1.0, uv_y0=0.0, uv_y1=1.0,
        )

        def _resample_loop(raw_pts, step):
            out_pts = []
            m = len(raw_pts)
            for idx in range(m):
                a, b = raw_pts[idx], raw_pts[(idx + 1) % m]
                d = math.hypot(b[0] - a[0], b[1] - a[1])
                cnt = max(1, round(d / step))
                for k in range(cnt):
                    t = k / cnt
                    out_pts.append([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
            return np.array(out_pts, dtype=np.float32)

        star_raw = [
            [
                0.5 + (0.42 if k % 2 == 0 else 0.22) * math.cos(2.0 * math.pi * k / 16.0),
                0.5 + (0.42 if k % 2 == 0 else 0.22) * math.sin(2.0 * math.pi * k / 16.0),
            ]
            for k in range(16)
        ]
        star_arr = _resample_loop(star_raw, 0.06)

        u_arr = _resample_loop(
            [[0.1, 0.1], [0.38, 0.1], [0.38, 0.6], [0.62, 0.6], [0.62, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            0.04,
        )

        outer_c = np.array(
            [[0.5 + 0.45 * math.cos(2.0 * math.pi * k / 48.0), 0.5 + 0.45 * math.sin(2.0 * math.pi * k / 48.0)] for k in range(48)],
            dtype=np.float32,
        )
        inner_c = np.array(
            [[0.5 + 0.18 * math.cos(2.0 * math.pi * k / 20.0), 0.5 + 0.18 * math.sin(2.0 * math.pi * k / 20.0)] for k in range(20)],
            dtype=np.float32,
        )

        tshirt_raw = [[0.2, 0.0], [0.8, 0.0], [0.8, 0.45]]
        for k in range(1, 6):
            ang = -math.pi / 2.0 + k * (math.pi / 2.0) / 6.0
            tshirt_raw.append([0.8 + 0.15 * (1.0 - math.cos(ang)), 0.45 + 0.35 * (k / 6.0)])
        tshirt_raw.extend([[0.75, 0.90], [0.65, 0.92]])
        for k in range(1, 7):
            t = k / 7.0
            tshirt_raw.append([0.65 - t * 0.30, 0.92 - 0.12 * math.sin(math.pi * t)])
        tshirt_raw.extend([[0.35, 0.92], [0.25, 0.90]])
        for k in range(1, 6):
            t = k / 6.0
            tshirt_raw.append([0.2 - 0.15 * math.sin(math.pi * t) + 0.05 * (1.0 - t), 0.80 - 0.35 * t])
        tshirt_raw.append([0.2, 0.45])
        tshirt_arr = _resample_loop(tshirt_raw, 0.04)

        cases = [
            ("star", star_arr, None, 0.06, 0.72),
            ("tshirt", tshirt_arr, None, 0.04, 0.90),
            ("u_shape", u_arr, None, 0.04, 0.90),
            ("circular_skirt", outer_c, [inner_c], 0.05, 0.80),
        ]
        for label, bnd_arr, holes, sz, min_ratio in cases:
            out = core.remesh_cloth_to_quad_3d_ext(
                pos,
                uv,
                tris,
                bnd_arr,
                holes,
                target_quad_size=sz,
                boundary_belt_loops=1,
                smoothing_iterations=10,
                keep_structured_when_possible=True,
                target_size_in_3d=False,
            )
            out_uvs = out["uvs_2d"]
            quads = out["quads"]
            out_tris = out["triangles"]

            self.assertTrue(np.all(np.isfinite(out["positions_3d"])), label)
            total_f = len(quads) + len(out_tris)
            self.assertGreater(total_f, 0, label)
            self.assertGreaterEqual(len(quads) / total_f, min_ratio, label)

            for q in quads:
                pts = out_uvs[q]
                self.assertGreater(_poly_signed_area_2d(pts), 0.0, label)
                for k in range(4):
                    v1 = pts[(k - 1) % 4] - pts[k]
                    v2 = pts[(k + 1) % 4] - pts[k]
                    c = np.clip(
                        float(np.dot(v1, v2)) / (float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-12),
                        -1.0,
                        1.0,
                    )
                    deg = math.degrees(math.acos(c))
                    self.assertGreaterEqual(deg, 20.0, f"{label} min_ang={deg}")
                    self.assertLessEqual(deg, 160.0, f"{label} max_ang={deg}")
            for t in out_tris:
                pts = out_uvs[t]
                self.assertGreater(_poly_signed_area_2d(pts), 0.0, label)


if __name__ == "__main__":
    unittest.main()
