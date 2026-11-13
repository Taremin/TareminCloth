# -*- coding: utf-8 -*-
"""Quadリメッシュ用の前処理（UV-split・アイランド分割・境界抽出）および後処理（縫合統合・属性再構築）

Blender `bpy` に依存しない純粋 Python + NumPy 実装であり、
スタンドアロン単体テストおよび Blender オペレーターの両方から利用できる。
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

from .mesh_extract import WaypointSeam, build_seam_cluster_map, extract_sewing_topology


@dataclass
class PreparedQuadIsland:
    """UVアイランド単位に分解・境界抽出されたリメッシュ入力データ"""

    island_id: int
    positions_3d: np.ndarray        # (M, 3) float32
    uvs_2d: np.ndarray              # (M, 2) float32
    triangles: np.ndarray           # (T, 3) uint32
    boundary_loop_2d: np.ndarray    # (B, 2) float32
    hole_loops_2d: list[np.ndarray]
    outer_orig_vi: list[int]        # 長さ B: 各境界点に対応する元 3D 頂点インデックス (-1 は補間点)
    hole_orig_vis: list[list[int]]
    local_to_orig_vi: np.ndarray    # (M,) int32
    local_to_global_uv_vi: np.ndarray
    dominant_piece_index: int = -1
    dominant_cloth_layer: int = 0
    dominant_material_index: int = 0
    orig_pin_weights: Optional[np.ndarray] = None  # (M,) float32


@dataclass
class PreparedQuadScene:
    """全アイランドの前処理結果および縫合・UVシーム情報"""

    islands: list[PreparedQuadIsland]
    orig_positions_3d: np.ndarray
    simple_sewing_edges: list[tuple[int, int]]
    waypoint_seams: list[WaypointSeam]


@dataclass
class MergedQuadMeshResult:
    """全アイランド統合・縫合済み最終メッシュ構築用データ"""

    positions_3d: np.ndarray            # (V, 3) float32
    quads: np.ndarray                   # (Q, 4) uint32
    triangles: np.ndarray               # (K, 3) uint32
    quad_corner_uvs: np.ndarray         # (Q, 4, 2) float32
    tri_corner_uvs: np.ndarray          # (K, 3, 2) float32
    quad_island_ids: np.ndarray         # (Q,) int32
    tri_island_ids: np.ndarray          # (K,) int32
    quad_material_indices: np.ndarray   # (Q,) int32
    tri_material_indices: np.ndarray    # (K,) int32
    quad_piece_indices: np.ndarray      # (Q,) int32
    tri_piece_indices: np.ndarray       # (K,) int32
    quad_cloth_layers: np.ndarray       # (Q,) int32
    tri_cloth_layers: np.ndarray        # (K,) int32
    loose_edges: list[tuple[int, int]] = field(default_factory=list)
    vertex_pin_weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    vertex_piece_indices: list[set[int]] = field(default_factory=list)
    vertex_island_ids: list[set[int]] = field(default_factory=list)


def _polygon_signed_area_2d(pts: np.ndarray) -> float:
    """2D閉ループの符号付き面積を算出（反時計回りで正）"""
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _winding_number_2d(pt: tuple[float, float], poly: np.ndarray) -> int:
    """点 `pt` に対する閉多角形 `poly` の回転数 (Winding Number) を算出"""
    px, py = pt
    n = len(poly)
    if n < 3:
        return 0
    wn = 0
    for i in range(n):
        x1, y1 = float(poly[i, 0]), float(poly[i, 1])
        nxt = (i + 1) % n
        x2, y2 = float(poly[nxt, 0]), float(poly[nxt, 1])
        if y1 <= py:
            if y2 > py and ((x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)) > 0.0:
                wn += 1
        else:
            if y2 <= py and ((x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)) < 0.0:
                wn -= 1
    return wn


def _pick_loop_interior_probe(poly: np.ndarray) -> tuple[float, float]:
    """他ループへの包含判定に使う代表点（境界辺中点の微小内側点または頂点平均）を取得"""
    if len(poly) == 0:
        return (0.0, 0.0)
    # 頂点自体ではなく最初の有効辺の中点を使うと境界重なり誤爆を避けやすい
    return (float(poly[0, 0]), float(poly[0, 1]))


def _extract_boundary_loops_from_uv_triangles(
    uvs_2d: np.ndarray,
    tris: np.ndarray,
) -> tuple[list[list[int]], list[list[int]]]:
    """UVアイランドの三角形群から境界半辺トレースにより外周ループ群と穴ループ群を抽出する"""
    dir_counts: dict[tuple[int, int], int] = defaultdict(int)
    for tri in tris:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if a == b or b == c or c == a:
            continue
        dir_counts[(a, b)] += 1
        dir_counts[(b, c)] += 1
        dir_counts[(c, a)] += 1

    # 非多様体有向辺の検知
    dup_edges = [e for e, cnt in dir_counts.items() if cnt > 1]
    if dup_edges:
        raise ValueError(
            f"UVアイランド内に重複した有向辺（非多様体エッジ: {len(dup_edges)}本）が存在するため境界を確定できません"
        )

    # 反対向きの半辺が存在しない有向辺が境界辺
    boundary_edges: list[tuple[int, int]] = []
    out_edges: dict[int, list[int]] = defaultdict(list)
    in_degree: dict[int, int] = defaultdict(int)

    for (u, v) in dir_counts:
        if (v, u) not in dir_counts:
            boundary_edges.append((u, v))
            out_edges[u].append(v)
            in_degree[v] += 1

    if not boundary_edges:
        raise ValueError("閉じたUVメッシュ（境界ループなし）のため2D展開リメッシュを実行できません")

    # 開放端・次数不整合チェック（各頂点の入次数 == 出次数であること）
    all_bnd_verts = set(out_edges.keys()) | set(in_degree.keys())
    for v in all_bnd_verts:
        if len(out_edges[v]) != in_degree[v]:
            raise ValueError(
                f"境界ループが閉じていない、または非多様体頂点 (頂点 {v}: 出={len(out_edges[v])}, 入={in_degree[v]}) があります"
            )

    # 半辺単位で使用済みフラグ管理しつつ閉路をトレース
    unused_edges = set(boundary_edges)
    adj_remaining: dict[int, list[int]] = {v: list(nbrs) for v, nbrs in out_edges.items()}

    raw_loops: list[list[int]] = []
    while unused_edges:
        start_u, start_v = min(unused_edges)
        unused_edges.remove((start_u, start_v))
        adj_remaining[start_u].remove(start_v)

        loop = [start_u]
        prev = start_u
        curr = start_v

        max_steps = len(boundary_edges) + 4
        steps = 0
        while curr != start_u:
            steps += 1
            if steps > max_steps:
                raise ValueError("境界半辺の追跡が閉路に戻りません（トポロジー異常）")
            loop.append(curr)
            cands = adj_remaining.get(curr, [])
            if not cands:
                raise ValueError(f"境界ループ追跡中に頂点 {curr} で行き止まりが発生しました")

            if len(cands) == 1:
                nxt = cands[0]
            else:
                # バタフライ/ピンチ頂点: 直前方向 (curr -> prev) からの左回り角最小を選択
                cx, cy = float(uvs_2d[curr, 0]), float(uvs_2d[curr, 1])
                px, py = float(uvs_2d[prev, 0]), float(uvs_2d[prev, 1])
                back_ang = math.atan2(py - cy, px - cx)

                def _ccw_turn(v_id: int) -> float:
                    vx, vy = float(uvs_2d[v_id, 0]), float(uvs_2d[v_id, 1])
                    ang = math.atan2(vy - cy, vx - cx)
                    diff = (ang - back_ang) % (2.0 * math.pi)
                    return diff if diff > 1e-9 else 2.0 * math.pi

                nxt = min(cands, key=_ccw_turn)

            adj_remaining[curr].remove(nxt)
            unused_edges.remove((curr, nxt))
            prev = curr
            curr = nxt

        if len(loop) >= 3:
            pts = uvs_2d[np.array(loop, dtype=np.int64)]
            if abs(_polygon_signed_area_2d(pts)) > 1e-12:
                raw_loops.append(loop)

    if not raw_loops:
        raise ValueError("有効な面積を持つUV境界ループが見つかりませんでした")

    if len(raw_loops) == 1:
        return [raw_loops[0]], []

    # 面積とネスト深度（他ループ内包数）から外周と穴を分類
    loop_polys = [uvs_2d[np.array(lp, dtype=np.int64)] for lp in raw_loops]
    loop_areas = [abs(_polygon_signed_area_2d(poly)) for poly in loop_polys]

    outers: list[list[int]] = []
    holes: list[list[int]] = []

    for i, poly_i in enumerate(loop_polys):
        probe = _pick_loop_interior_probe(poly_i)
        depth = 0
        for j, poly_j in enumerate(loop_polys):
            if i == j:
                continue
            # 自分より厳密に大きいループにのみ内包され得る
            if loop_areas[j] <= loop_areas[i] * (1.0 - 1e-5):
                continue
            if _winding_number_2d(probe, poly_j) != 0:
                depth += 1
        if depth % 2 == 0:
            outers.append(raw_loops[i])
        else:
            holes.append(raw_loops[i])

    if not outers:
        # 数値誤差等で全て穴判定された場合のみ最大面積ループを外周とする
        max_idx = max(range(len(raw_loops)), key=lambda idx: loop_areas[idx])
        outers = [raw_loops[max_idx]]
        holes = [lp for k, lp in enumerate(raw_loops) if k != max_idx]
    elif len(outers) > 1:
        # 1連結UV成分内に複数外周がある異常時は最大面積ループを外周、残りを保持して処理
        max_idx = max(range(len(outers)), key=lambda idx: abs(_polygon_signed_area_2d(uvs_2d[np.array(outers[idx], dtype=np.int64)])))
        primary = outers[max_idx]
        outers = [primary]

    return outers, holes


def _ensure_min_4_boundary_points(
    loop_uvs: np.ndarray,
    loop_orig_vi: list[int],
) -> tuple[np.ndarray, list[int]]:
    """外周ループが3頂点のみの場合、最長辺の中点を補間挿入して4頂点以上にする"""
    if len(loop_uvs) >= 4:
        return loop_uvs, loop_orig_vi
    if len(loop_uvs) < 3:
        raise ValueError("外周ループの頂点数が 3 未満です")

    best_e = 0
    best_len2 = -1.0
    n = len(loop_uvs)
    for i in range(n):
        nxt = (i + 1) % n
        d = loop_uvs[nxt] - loop_uvs[i]
        l2 = float(d[0] * d[0] + d[1] * d[1])
        if l2 > best_len2:
            best_len2 = l2
            best_e = i

    nxt_e = (best_e + 1) % n
    mid_uv = 0.5 * (loop_uvs[best_e] + loop_uvs[nxt_e])
    new_uvs = np.insert(loop_uvs, best_e + 1, mid_uv, axis=0)
    new_orig = list(loop_orig_vi)
    new_orig.insert(best_e + 1, -1)
    return new_uvs, new_orig


def prepare_quad_remesh_islands(
    positions_3d: np.ndarray,
    triangles: np.ndarray,
    *,
    triangle_uvs: np.ndarray,
    loose_edges: Optional[list[tuple[int, int]]] = None,
    triangle_piece_ids: Optional[np.ndarray] = None,
    triangle_material_indices: Optional[np.ndarray] = None,
    vertex_cloth_layers: Optional[np.ndarray] = None,
    vertex_pin_weights: Optional[np.ndarray] = None,
    uv_quantize_tol: float = 1e-5,
) -> PreparedQuadScene:
    """三角化済みメッシュと面コーナー UV から UV-split・アイランド分割・境界抽出を行う。

    Args:
        positions_3d: (N, 3) 元3D頂点座標配列
        triangles: (F, 3) 元三角形の3D頂点インデックス配列
        triangle_uvs: (F, 3, 2) 各三角形コーナーのUV座標配列
        loose_edges: 縫合線・ウェイポイントを構成する孤立エッジ配列
        triangle_piece_ids: (F,) 各三角形の pattern_piece_index
        triangle_material_indices: (F,) 各三角形の material_index
        vertex_cloth_layers: (N,) 各元頂点の cloth_layer
        vertex_pin_weights: (N,) 各元頂点のピンウェイト
        uv_quantize_tol: UVシーム同一判定の丸め許容幅
    """
    pos_3d = np.asarray(positions_3d, dtype=np.float32)
    tris = np.asarray(triangles, dtype=np.int64)
    tri_uvs = np.asarray(triangle_uvs, dtype=np.float32)

    if len(pos_3d) == 0 or len(tris) == 0:
        raise ValueError("入力メッシュに頂点または面が存在しません")
    if tri_uvs.shape != (len(tris), 3, 2):
        raise ValueError(
            f"triangle_uvs の形状が不正です: 期待={(len(tris), 3, 2)}, 実際={tri_uvs.shape}"
        )

    # 1. コーナー UV に基づく UV-split 頂点化
    inv_tol = 1.0 / max(uv_quantize_tol, 1e-8)
    key_to_uv_vi: dict[tuple[int, int, int], int] = {}
    uv_pos_list: list[np.ndarray] = []
    uv_coord_list: list[np.ndarray] = []
    uv_to_orig: list[int] = []
    uv_pin_list: list[float] = []
    uv_layer_list: list[int] = []

    uv_tris = np.zeros_like(tris, dtype=np.int64)
    for fi in range(len(tris)):
        for ci in range(3):
            orig_vi = int(tris[fi, ci])
            u = float(tri_uvs[fi, ci, 0])
            v = float(tri_uvs[fi, ci, 1])
            key = (orig_vi, int(round(u * inv_tol)), int(round(v * inv_tol)))
            uv_vi = key_to_uv_vi.get(key)
            if uv_vi is None:
                uv_vi = len(uv_pos_list)
                key_to_uv_vi[key] = uv_vi
                uv_pos_list.append(pos_3d[orig_vi])
                uv_coord_list.append(np.array([u, v], dtype=np.float32))
                uv_to_orig.append(orig_vi)
                if vertex_pin_weights is not None and 0 <= orig_vi < len(vertex_pin_weights):
                    uv_pin_list.append(float(vertex_pin_weights[orig_vi]))
                if vertex_cloth_layers is not None and 0 <= orig_vi < len(vertex_cloth_layers):
                    uv_layer_list.append(int(vertex_cloth_layers[orig_vi]))
            uv_tris[fi, ci] = uv_vi

    # 2. UV 頂点空間での連結成分（UV アイランド）分割
    n_uv = len(uv_pos_list)
    vert_to_tris: list[list[int]] = [[] for _ in range(n_uv)]
    for fi in range(len(uv_tris)):
        a, b, c = int(uv_tris[fi, 0]), int(uv_tris[fi, 1]), int(uv_tris[fi, 2])
        if a == b or b == c or c == a:
            continue
        vert_to_tris[a].append(fi)
        vert_to_tris[b].append(fi)
        vert_to_tris[c].append(fi)

    visited_tri = [False] * len(uv_tris)
    islands: list[PreparedQuadIsland] = []

    for start_fi in range(len(uv_tris)):
        if visited_tri[start_fi]:
            continue
        a0, b0, c0 = int(uv_tris[start_fi, 0]), int(uv_tris[start_fi, 1]), int(uv_tris[start_fi, 2])
        if a0 == b0 or b0 == c0 or c0 == a0:
            visited_tri[start_fi] = True
            continue

        comp_tris: list[int] = []
        queue = deque([start_fi])
        visited_tri[start_fi] = True
        while queue:
            fi = queue.popleft()
            comp_tris.append(fi)
            for vi in uv_tris[fi]:
                for nb_fi in vert_to_tris[int(vi)]:
                    if not visited_tri[nb_fi]:
                        visited_tri[nb_fi] = True
                        queue.append(nb_fi)

        comp_tris.sort()
        global_uv_vis = sorted({int(v) for fi in comp_tris for v in uv_tris[fi]})
        g2l = {gv: lv for lv, gv in enumerate(global_uv_vis)}

        local_pos = np.array([uv_pos_list[gv] for gv in global_uv_vis], dtype=np.float32)
        local_uvs = np.array([uv_coord_list[gv] for gv in global_uv_vis], dtype=np.float32)
        local_tris = np.array(
            [[g2l[int(uv_tris[fi, 0])], g2l[int(uv_tris[fi, 1])], g2l[int(uv_tris[fi, 2])]] for fi in comp_tris],
            dtype=np.uint32,
        )
        local_to_orig = np.array([uv_to_orig[gv] for gv in global_uv_vis], dtype=np.int32)
        local_to_global_uv = np.array(global_uv_vis, dtype=np.int32)

        outers, holes = _extract_boundary_loops_from_uv_triangles(local_uvs, local_tris)
        outer_loop = outers[0]
        outer_uvs = local_uvs[np.array(outer_loop, dtype=np.int64)]
        outer_orig = [int(local_to_orig[lv]) for lv in outer_loop]
        outer_uvs, outer_orig = _ensure_min_4_boundary_points(outer_uvs, outer_orig)

        hole_uvs_list = [local_uvs[np.array(h, dtype=np.int64)] for h in holes]
        hole_orig_list = [[int(local_to_orig[lv]) for lv in h] for h in holes]

        dom_piece = -1
        if triangle_piece_ids is not None and len(comp_tris) > 0:
            vals = [int(triangle_piece_ids[fi]) for fi in comp_tris if int(triangle_piece_ids[fi]) >= 0]
            if vals:
                dom_piece = max(set(vals), key=vals.count)

        dom_mat = 0
        if triangle_material_indices is not None and len(comp_tris) > 0:
            mats = [int(triangle_material_indices[fi]) for fi in comp_tris]
            dom_mat = max(set(mats), key=mats.count)

        dom_layer = 0
        if uv_layer_list:
            layers = [uv_layer_list[gv] for gv in global_uv_vis]
            if layers:
                dom_layer = max(set(layers), key=layers.count)

        local_pins = None
        if uv_pin_list:
            local_pins = np.array([uv_pin_list[gv] for gv in global_uv_vis], dtype=np.float32)

        island_id = len(islands)
        islands.append(
            PreparedQuadIsland(
                island_id=island_id,
                positions_3d=local_pos,
                uvs_2d=local_uvs,
                triangles=local_tris,
                boundary_loop_2d=outer_uvs,
                hole_loops_2d=hole_uvs_list,
                outer_orig_vi=outer_orig,
                hole_orig_vis=hole_orig_list,
                local_to_orig_vi=local_to_orig,
                local_to_global_uv_vi=local_to_global_uv,
                dominant_piece_index=dom_piece if dom_piece >= 0 else island_id,
                dominant_cloth_layer=dom_layer,
                dominant_material_index=dom_mat,
                orig_pin_weights=local_pins,
            )
        )

    # 3. 既存の縫合トポロジー解析ユーティリティを利用して縫合線・ウェイポイントを抽出
    simple_sew: list[tuple[int, int]] = []
    wp_seams: list[WaypointSeam] = []
    if loose_edges:
        simple_sew, wp_seams = extract_sewing_topology(
            loose_edges,
            faces_2d=tris,
            positions=pos_3d,
        )

    return PreparedQuadScene(
        islands=islands,
        orig_positions_3d=pos_3d,
        simple_sewing_edges=simple_sew,
        waypoint_seams=wp_seams,
    )


def _interpolate_pin_weights_from_uv_mesh(
    target_uvs: np.ndarray,
    *,
    src_uvs: np.ndarray,
    src_tris: np.ndarray,
    src_pins: np.ndarray,
) -> np.ndarray:
    """出力 UV 頂点ごとに元アイランド三角形上の最近接重心座標から Pin ウェイトを補間する"""
    m = len(target_uvs)
    out = np.zeros(m, dtype=np.float32)
    if m == 0 or len(src_tris) == 0 or len(src_pins) == 0:
        return out

    tri_a = src_uvs[src_tris[:, 0]]  # (T, 2)
    tri_b = src_uvs[src_tris[:, 1]]
    tri_c = src_uvs[src_tris[:, 2]]
    ab = tri_b - tri_a
    ac = tri_c - tri_a
    det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    valid_det = np.abs(det) > 1e-12
    safe_det = np.where(valid_det, det, 1.0)

    pa = src_pins[src_tris[:, 0]]
    pb = src_pins[src_tris[:, 1]]
    pc = src_pins[src_tris[:, 2]]

    # 空間高速化のためチャンク単位で厳密包含チェック + フォールバック最近接頂点/投影
    src_min_u = float(np.min(src_uvs[:, 0]))
    src_min_v = float(np.min(src_uvs[:, 1]))
    src_span = max(
        float(np.max(src_uvs[:, 0]) - src_min_u),
        float(np.max(src_uvs[:, 1]) - src_min_v),
        1e-6,
    )
    cell = max(src_span / 32.0, 1e-5)

    t_min_u = np.minimum(np.minimum(tri_a[:, 0], tri_b[:, 0]), tri_c[:, 0])
    t_max_u = np.maximum(np.maximum(tri_a[:, 0], tri_b[:, 0]), tri_c[:, 0])
    t_min_v = np.minimum(np.minimum(tri_a[:, 1], tri_b[:, 1]), tri_c[:, 1])
    t_max_v = np.maximum(np.maximum(tri_a[:, 1], tri_b[:, 1]), tri_c[:, 1])

    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for ti in range(len(src_tris)):
        if not valid_det[ti]:
            continue
        gx0 = int(math.floor((float(t_min_u[ti]) - src_min_u) / cell))
        gx1 = int(math.ceil((float(t_max_u[ti]) - src_min_u) / cell))
        gy0 = int(math.floor((float(t_min_v[ti]) - src_min_v) / cell))
        gy1 = int(math.ceil((float(t_max_v[ti]) - src_min_v) / cell))
        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                grid[(gx, gy)].append(ti)

    eps = 1e-4
    for vi in range(m):
        u, v = float(target_uvs[vi, 0]), float(target_uvs[vi, 1])
        gx = int(math.floor((u - src_min_u) / cell))
        gy = int(math.floor((v - src_min_v) / cell))

        found = False
        for rad in range(0, 3):
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    cands = grid.get((gx + dx, gy + dy))
                    if not cands:
                        continue
                    for ti in cands:
                        ap_u = u - float(tri_a[ti, 0])
                        ap_v = v - float(tri_a[ti, 1])
                        lb = (ap_u * float(ac[ti, 1]) - ap_v * float(ac[ti, 0])) / float(safe_det[ti])
                        lc = (float(ab[ti, 0]) * ap_v - float(ab[ti, 1]) * ap_u) / float(safe_det[ti])
                        la = 1.0 - lb - lc
                        if la >= -eps and lb >= -eps and lc >= -eps:
                            wa = max(0.0, la)
                            wb = max(0.0, lb)
                            wc = max(0.0, lc)
                            w_sum = max(wa + wb + wc, 1e-12)
                            out[vi] = (wa * float(pa[ti]) + wb * float(pb[ti]) + wc * float(pc[ti])) / w_sum
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break

        if not found:
            # フォールバック: 最近接元頂点のピンウェイト
            d2 = (src_uvs[:, 0] - u) ** 2 + (src_uvs[:, 1] - v) ** 2
            nearest = int(np.argmin(d2))
            out[vi] = float(src_pins[nearest])

    return out


class _UnionFind:
    """縫合クラスタおよび UV シーム統合用の素集合データ構造"""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def merge_island_outputs(
    scene: PreparedQuadScene,
    island_outputs: list[dict],
    *,
    mode: str = "STATIC_MERGED",
) -> MergedQuadMeshResult:
    """各アイランドの Quad リメッシュ出力を統合し、モードに応じて縫合・UVシーム・属性を再構築する。

    Args:
        scene: 前処理済みシーンデータ
        island_outputs: 各アイランドに対する Rust 側出力辞書のリスト
        mode: `"STATIC_MERGED"`（縫合クラスタとUVシームを3D頂点統合）
              または `"KEEP_PATTERN"`（アイランド分離を維持し縫合Loose Edgeを再構築）
    """
    if len(scene.islands) != len(island_outputs):
        raise ValueError(
            f"アイランド数 ({len(scene.islands)}) とリメッシュ結果数 ({len(island_outputs)}) が一致しません"
        )
    if mode not in ("STATIC_MERGED", "KEEP_PATTERN"):
        raise ValueError(f"未知の結合モードが指定されました: {mode}")

    pos_chunks: list[np.ndarray] = []
    pin_chunks: list[np.ndarray] = []
    vert_island_list: list[set[int]] = []
    vert_piece_list: list[set[int]] = []

    all_quads: list[np.ndarray] = []
    all_tris: list[np.ndarray] = []
    quad_uv_chunks: list[np.ndarray] = []
    tri_uv_chunks: list[np.ndarray] = []

    quad_island_chunks: list[np.ndarray] = []
    tri_island_chunks: list[np.ndarray] = []
    quad_mat_chunks: list[np.ndarray] = []
    tri_mat_chunks: list[np.ndarray] = []
    quad_piece_chunks: list[np.ndarray] = []
    tri_piece_chunks: list[np.ndarray] = []
    quad_layer_chunks: list[np.ndarray] = []
    tri_layer_chunks: list[np.ndarray] = []

    # orig_vi -> list[(island_id, global_vert_idx)]
    orig_to_global_bnd: dict[int, list[tuple[int, int]]] = defaultdict(list)

    v_offset = 0
    for island, out in zip(scene.islands, island_outputs):
        pos = np.asarray(out["positions_3d"], dtype=np.float32)
        uvs = np.asarray(out["uvs_2d"], dtype=np.float32)
        quads = np.asarray(out["quads"], dtype=np.uint32)
        tris = np.asarray(out["triangles"], dtype=np.uint32)
        bnd_indices = np.asarray(out["boundary_vertex_indices"], dtype=np.int64)

        n_v = len(pos)
        pos_chunks.append(pos)

        if island.orig_pin_weights is not None:
            pins = _interpolate_pin_weights_from_uv_mesh(
                uvs,
                src_uvs=island.uvs_2d,
                src_tris=island.triangles,
                src_pins=island.orig_pin_weights,
            )
        else:
            pins = np.zeros(n_v, dtype=np.float32)
        pin_chunks.append(pins)

        for _ in range(n_v):
            vert_island_list.append({island.island_id})
            vert_piece_list.append({island.dominant_piece_index})

        # 外周境界点の元 3D 頂点対応を記録
        for k, orig_vi in enumerate(island.outer_orig_vi):
            if orig_vi >= 0 and k < len(bnd_indices):
                g_vi = v_offset + int(bnd_indices[k])
                orig_to_global_bnd[orig_vi].append((island.island_id, g_vi))

        # 穴境界点についても UV 一致により元 3D 頂点対応を記録
        if island.hole_orig_vis:
            for hole_uvs, hole_orig in zip(island.hole_loops_2d, island.hole_orig_vis):
                for h_uv, orig_vi in zip(hole_uvs, hole_orig):
                    if orig_vi < 0:
                        continue
                    d2 = (uvs[:, 0] - h_uv[0]) ** 2 + (uvs[:, 1] - h_uv[1]) ** 2
                    best_lv = int(np.argmin(d2))
                    if float(d2[best_lv]) < 1e-8:
                        orig_to_global_bnd[orig_vi].append((island.island_id, v_offset + best_lv))

        if len(quads) > 0:
            q_global = quads + np.uint32(v_offset)
            all_quads.append(q_global)
            quad_uv_chunks.append(uvs[quads])
            nq = len(quads)
            quad_island_chunks.append(np.full(nq, island.island_id, dtype=np.int32))
            quad_mat_chunks.append(np.full(nq, island.dominant_material_index, dtype=np.int32))
            quad_piece_chunks.append(np.full(nq, island.dominant_piece_index, dtype=np.int32))
            quad_layer_chunks.append(np.full(nq, island.dominant_cloth_layer, dtype=np.int32))

        if len(tris) > 0:
            t_global = tris + np.uint32(v_offset)
            all_tris.append(t_global)
            tri_uv_chunks.append(uvs[tris])
            nt = len(tris)
            tri_island_chunks.append(np.full(nt, island.island_id, dtype=np.int32))
            tri_mat_chunks.append(np.full(nt, island.dominant_material_index, dtype=np.int32))
            tri_piece_chunks.append(np.full(nt, island.dominant_piece_index, dtype=np.int32))
            tri_layer_chunks.append(np.full(nt, island.dominant_cloth_layer, dtype=np.int32))

        v_offset += n_v

    merged_pos = np.concatenate(pos_chunks, axis=0) if pos_chunks else np.zeros((0, 3), dtype=np.float32)
    merged_pins = np.concatenate(pin_chunks, axis=0) if pin_chunks else np.zeros(0, dtype=np.float32)
    merged_quads = np.concatenate(all_quads, axis=0) if all_quads else np.zeros((0, 4), dtype=np.uint32)
    merged_tris = np.concatenate(all_tris, axis=0) if all_tris else np.zeros((0, 3), dtype=np.uint32)
    quad_uvs = np.concatenate(quad_uv_chunks, axis=0) if quad_uv_chunks else np.zeros((0, 4, 2), dtype=np.float32)
    tri_uvs = np.concatenate(tri_uv_chunks, axis=0) if tri_uv_chunks else np.zeros((0, 3, 2), dtype=np.float32)

    q_islands = np.concatenate(quad_island_chunks) if quad_island_chunks else np.zeros(0, dtype=np.int32)
    t_islands = np.concatenate(tri_island_chunks) if tri_island_chunks else np.zeros(0, dtype=np.int32)
    q_mats = np.concatenate(quad_mat_chunks) if quad_mat_chunks else np.zeros(0, dtype=np.int32)
    t_mats = np.concatenate(tri_mat_chunks) if tri_mat_chunks else np.zeros(0, dtype=np.int32)
    q_pieces = np.concatenate(quad_piece_chunks) if quad_piece_chunks else np.zeros(0, dtype=np.int32)
    t_pieces = np.concatenate(tri_piece_chunks) if tri_piece_chunks else np.zeros(0, dtype=np.int32)
    q_layers = np.concatenate(quad_layer_chunks) if quad_layer_chunks else np.zeros(0, dtype=np.int32)
    t_layers = np.concatenate(tri_layer_chunks) if tri_layer_chunks else np.zeros(0, dtype=np.int32)

    n_total_verts = len(merged_pos)
    uf = _UnionFind(n_total_verts)

    # 1. 同一アイランド内で元 3D 頂点 orig_vi を共有する UV シーム境界を両モードで統合
    for orig_vi, entries in orig_to_global_bnd.items():
        by_island: dict[int, list[int]] = defaultdict(list)
        for isl_id, g_vi in entries:
            by_island[isl_id].append(g_vi)
        for g_vis in by_island.values():
            if len(g_vis) > 1:
                head = g_vis[0]
                for g in g_vis[1:]:
                    uf.union(head, g)

    loose_edges_out: list[tuple[int, int]] = []

    if mode == "STATIC_MERGED":
        # 1a. 別アイランド間の UV シーム（同一 orig_vi）も 3D 統合
        for orig_vi, entries in orig_to_global_bnd.items():
            if len(entries) > 1:
                head = entries[0][1]
                for _, g_vi in entries[1:]:
                    uf.union(head, g_vi)

        # 1b. extract_sewing_topology + build_seam_cluster_map を再利用し全縫合クラスタを統合
        partner_map = build_seam_cluster_map(scene.simple_sewing_edges, scene.waypoint_seams)
        for orig_a, partners in partner_map.items():
            entries_a = orig_to_global_bnd.get(orig_a)
            if not entries_a:
                continue
            head_a = entries_a[0][1]
            for orig_b in partners:
                entries_b = orig_to_global_bnd.get(orig_b)
                if not entries_b:
                    continue
                for _, g_b in entries_b:
                    uf.union(head_a, g_b)

    # Union-Find で統合された頂点クラスタの重心スナップ & 頂点配列コンパクション
    clusters: dict[int, list[int]] = defaultdict(list)
    for g_vi in range(n_total_verts):
        clusters[uf.find(g_vi)].append(g_vi)

    root_to_compact: dict[int, int] = {}
    compact_pos: list[np.ndarray] = []
    compact_pins: list[float] = []
    compact_islands: list[set[int]] = []
    compact_pieces: list[set[int]] = []

    for root in sorted(clusters.keys()):
        members = clusters[root]
        compact_id = len(compact_pos)
        root_to_compact[root] = compact_id

        if len(members) == 1:
            m0 = members[0]
            compact_pos.append(merged_pos[m0])
            compact_pins.append(float(merged_pins[m0]))
            compact_islands.append(set(vert_island_list[m0]))
            compact_pieces.append(set(vert_piece_list[m0]))
        else:
            idx_arr = np.array(members, dtype=np.int64)
            compact_pos.append(np.mean(merged_pos[idx_arr], axis=0).astype(np.float32))
            compact_pins.append(float(np.max(merged_pins[idx_arr])))
            isl_set: set[int] = set()
            piece_set: set[int] = set()
            for m_vi in members:
                isl_set.update(vert_island_list[m_vi])
                piece_set.update(vert_piece_list[m_vi])
            compact_islands.append(isl_set)
            compact_pieces.append(piece_set)

    remap = np.array([root_to_compact[uf.find(g)] for g in range(n_total_verts)], dtype=np.uint32)

    if len(merged_quads) > 0:
        merged_quads = remap[merged_quads]
        # 頂点統合で4点中の2点が一致してしまった極稀な退化Quadを安全にTri救済または除外
        clean_q = []
        clean_q_uv = []
        clean_q_isl = []
        clean_q_mat = []
        clean_q_piece = []
        clean_q_layer = []
        salvaged_t = []
        salvaged_t_uv = []
        salvaged_t_isl = []
        salvaged_t_mat = []
        salvaged_t_piece = []
        salvaged_t_layer = []

        for qi in range(len(merged_quads)):
            q = merged_quads[qi]
            uniq = []
            uniq_uv = []
            seen = set()
            for k in range(4):
                vid = int(q[k])
                if vid not in seen:
                    seen.add(vid)
                    uniq.append(vid)
                    uniq_uv.append(quad_uvs[qi, k])
            if len(uniq) == 4:
                clean_q.append(uniq)
                clean_q_uv.append(quad_uvs[qi])
                clean_q_isl.append(q_islands[qi])
                clean_q_mat.append(q_mats[qi])
                clean_q_piece.append(q_pieces[qi])
                clean_q_layer.append(q_layers[qi])
            elif len(uniq) == 3:
                salvaged_t.append(uniq)
                salvaged_t_uv.append(np.stack(uniq_uv, axis=0))
                salvaged_t_isl.append(q_islands[qi])
                salvaged_t_mat.append(q_mats[qi])
                salvaged_t_piece.append(q_pieces[qi])
                salvaged_t_layer.append(q_layers[qi])

        merged_quads = np.array(clean_q, dtype=np.uint32) if clean_q else np.zeros((0, 4), dtype=np.uint32)
        quad_uvs = np.array(clean_q_uv, dtype=np.float32) if clean_q_uv else np.zeros((0, 4, 2), dtype=np.float32)
        q_islands = np.array(clean_q_isl, dtype=np.int32)
        q_mats = np.array(clean_q_mat, dtype=np.int32)
        q_pieces = np.array(clean_q_piece, dtype=np.int32)
        q_layers = np.array(clean_q_layer, dtype=np.int32)

        if salvaged_t:
            st = np.array(salvaged_t, dtype=np.uint32)
            st_uv = np.array(salvaged_t_uv, dtype=np.float32)
            merged_tris = np.concatenate([merged_tris, st], axis=0) if len(merged_tris) > 0 else st
            tri_uvs = np.concatenate([tri_uvs, st_uv], axis=0) if len(tri_uvs) > 0 else st_uv
            t_islands = np.concatenate([t_islands, np.array(salvaged_t_isl, dtype=np.int32)])
            t_mats = np.concatenate([t_mats, np.array(salvaged_t_mat, dtype=np.int32)])
            t_pieces = np.concatenate([t_pieces, np.array(salvaged_t_piece, dtype=np.int32)])
            t_layers = np.concatenate([t_layers, np.array(salvaged_t_layer, dtype=np.int32)])

    if len(merged_tris) > 0:
        merged_tris = remap[merged_tris]
        keep_mask = (
            (merged_tris[:, 0] != merged_tris[:, 1])
            & (merged_tris[:, 1] != merged_tris[:, 2])
            & (merged_tris[:, 2] != merged_tris[:, 0])
        )
        merged_tris = merged_tris[keep_mask]
        tri_uvs = tri_uvs[keep_mask]
        t_islands = t_islands[keep_mask]
        t_mats = t_mats[keep_mask]
        t_pieces = t_pieces[keep_mask]
        t_layers = t_layers[keep_mask]

    if mode == "KEEP_PATTERN":
        # 2a. 単純縫合線の再構築 (同一頂点に統合されていなければ Loose Edge 生成)
        seen_pairs: set[tuple[int, int]] = set()
        for orig_a, orig_b in scene.simple_sewing_edges:
            ea = orig_to_global_bnd.get(orig_a)
            eb = orig_to_global_bnd.get(orig_b)
            if not ea or not eb:
                continue
            ga = int(remap[ea[0][1]])
            gb = int(remap[eb[0][1]])
            if ga == gb:
                continue
            key = (ga, gb) if ga < gb else (gb, ga)
            if key not in seen_pairs:
                seen_pairs.add(key)
                loose_edges_out.append((ga, gb))

        # 2b. UVシーム（元メッシュ上は同一 orig_vi で別アイランドだった点）も縫合エッジ化
        for orig_vi, entries in orig_to_global_bnd.items():
            distinct = []
            seen_c = set()
            for _, g_vi in entries:
                c_vi = int(remap[g_vi])
                if c_vi not in seen_c:
                    seen_c.add(c_vi)
                    distinct.append(c_vi)
            if len(distinct) > 1:
                head = distinct[0]
                for other in distinct[1:]:
                    key = (head, other) if head < other else (other, head)
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        loose_edges_out.append((head, other))

        # 2c. BONE_WRAP 等の Waypoint 縫合パス（中間ウェイポイント新規頂点＋連鎖エッジ）を再構築
        for ws in scene.waypoint_seams:
            ea = orig_to_global_bnd.get(int(ws.vert_a))
            eb = orig_to_global_bnd.get(int(ws.vert_b))
            if not ea or not eb:
                continue
            ga = int(remap[ea[0][1]])
            gb = int(remap[eb[0][1]])
            if len(ws.waypoint_coords) == 0:
                if ga != gb:
                    key = (ga, gb) if ga < gb else (gb, ga)
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        loose_edges_out.append((ga, gb))
                continue

            prev_v = ga
            for wp_pos in ws.waypoint_coords:
                wp_id = len(compact_pos)
                compact_pos.append(np.asarray(wp_pos, dtype=np.float32))
                compact_pins.append(0.0)
                compact_islands.append(set())
                compact_pieces.append(set())
                loose_edges_out.append((prev_v, wp_id))
                prev_v = wp_id
            loose_edges_out.append((prev_v, gb))

    final_pos = np.array(compact_pos, dtype=np.float32) if compact_pos else np.zeros((0, 3), dtype=np.float32)
    final_pins = np.array(compact_pins, dtype=np.float32) if compact_pins else np.zeros(0, dtype=np.float32)

    return MergedQuadMeshResult(
        positions_3d=final_pos,
        quads=merged_quads,
        triangles=merged_tris,
        quad_corner_uvs=quad_uvs,
        tri_corner_uvs=tri_uvs,
        quad_island_ids=q_islands,
        tri_island_ids=t_islands,
        quad_material_indices=q_mats,
        tri_material_indices=t_mats,
        quad_piece_indices=q_pieces,
        tri_piece_indices=t_pieces,
        quad_cloth_layers=q_layers,
        tri_cloth_layers=t_layers,
        loose_edges=loose_edges_out,
        vertex_pin_weights=final_pins,
        vertex_piece_indices=compact_pieces,
        vertex_island_ids=compact_islands,
    )
