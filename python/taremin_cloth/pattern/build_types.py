# -*- coding: utf-8 -*-
"""型紙エディタのメッシュビルドデータモデルおよびジオメトリ生成 (Blender非依存)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
import numpy as np

import taremin_cloth_core as core
from ..utils.pattern_ir import (
    BoundarySegmentIR,
    PatternContour2D,
    SeamPairIR,
    WaypointGuideIR,
    sample_pattern_boundary_loop,
)
from .document import Document, EdgeRange, PathRole, Seam
from .geometry import (
    build_arc_table,
    point_in_polygon,
    arc_length_at_t,
    sample_sub_segment_points,
    merge_breakpoints_by_arc_length,
)


@dataclass
class GarmentMeshGeometry:
    """統合衣服メッシュの幾何・トポロジー・属性データコンテナ (Blender非依存)."""
    # 頂点3D座標 (N, 3)
    vertices_3d: list[tuple[float, float, float]]
    # 面インデックス (F, 3)
    faces: list[tuple[int, int, int]]
    # 縫合エッジ (E, 2)
    sewing_edges: list[tuple[int, int]]
    # 各面のループ頂点UV座標 (F * 3, 2) [0.0, 1.0]
    face_uvs: list[tuple[float, float]]
    # 面ごとのピース番号 (F,)
    face_piece_indices: list[int]
    # 面ごとのレイヤー番号 (F,)
    face_layer_ids: list[int]
    # ピース名 -> 頂点インデックスリスト
    piece_vertex_map: dict[str, list[int]]
    # ピン留め対象頂点インデックスリスト
    pin_vertex_indices: list[int]


def _euler_to_rotation_matrix(euler: tuple[float, float, float]) -> np.ndarray:
    """Euler角 (XYZ) から 3x3 回転行列を算出する."""
    rx, ry, rz = euler
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)

    return (R_z @ R_y @ R_x).astype(np.float32)


def document_to_contours_and_seams(
    doc: Document,
) -> tuple[list[PatternContour2D], list[SeamPairIR]]:
    """Document から PatternContour2D リストおよび SeamPairIR リストへ変換する."""
    contours: list[PatternContour2D] = []

    for piece_id, piece in doc.pieces.items():
        outer_path = piece.paths.get(piece.outer_path_id)
        if not outer_path or len(outer_path.nodes) < 3:
            continue

        nodes = outer_path.nodes
        n_nodes = len(nodes)
        segments: list[BoundarySegmentIR] = []

        for i in range(n_nodes):
            curr_node = nodes[i]
            next_node = nodes[(i + 1) % n_nodes]

            ctrls = outer_path.get_segment_controls(i)
            table = build_arc_table(*ctrls)

            base_seg_id = f"{curr_node.id}_{next_node.id}"

            # セグメント上の合印ノッチ比率および部分縫合 (t0, t1) の分割点比率を取得
            raw_ratios: set[float] = {
                n.t for n in doc.notches.values()
                if n.piece_id == piece.id
                and n.path_id == piece.outer_path_id
                and n.segment_start_id == curr_node.id
                and 0.001 < n.t < 0.999
            }

            for seam in doc.seams.values():
                for er in seam.iter_unique_edge_ranges():
                    if er.piece_id == piece.id:
                        if er.start_node_id == curr_node.id and er.end_node_id == next_node.id:
                            if 0.001 < er.t0 < 0.999:
                                raw_ratios.add(er.t0)
                            if 0.001 < er.t1 < 0.999:
                                raw_ratios.add(er.t1)
                        elif er.start_node_id == next_node.id and er.end_node_id == curr_node.id:
                            if 0.001 < er.t0 < 0.999:
                                raw_ratios.add(1.0 - er.t0)
                            if 0.001 < er.t1 < 0.999:
                                raw_ratios.add(1.0 - er.t1)

            # 1mm (0.001m) 未満の近接点を統合
            merged_breakpoints = merge_breakpoints_by_arc_length(table, list(raw_ratios), min_arc_gap=0.001)

            if len(merged_breakpoints) == 2:
                poly_pts = np.array(table.points, dtype=np.float32)
                segments.append(
                    BoundarySegmentIR(
                        segment_id=base_seg_id,
                        start_uid=curr_node.id,
                        end_uid=next_node.id,
                        polyline_2d=poly_pts,
                        arc_length=table.total_length,
                        base_segment_id=base_seg_id,
                        t_start=0.0,
                        t_end=1.0,
                    )
                )
            else:
                num_subs = len(merged_breakpoints) - 1
                for j in range(num_subs):
                    t_a = merged_breakpoints[j]
                    t_b = merged_breakpoints[j + 1]
                    sub_seg_id = f"{base_seg_id}@sub{j}"
                    start_u = curr_node.id if j == 0 else f"{base_seg_id}_notch_{j}"
                    end_u = next_node.id if j == num_subs - 1 else f"{base_seg_id}_notch_{j + 1}"

                    sub_pts = sample_sub_segment_points(
                        ctrls[0], ctrls[1], ctrls[2], ctrls[3],
                        t0=t_a, t1=t_b,
                        steps=max(2, int(round(16 * (t_b - t_a)))),
                    )
                    sub_poly = np.array(sub_pts, dtype=np.float32)
                    sub_len = abs(arc_length_at_t(table, t_b) - arc_length_at_t(table, t_a))

                    segments.append(
                        BoundarySegmentIR(
                            segment_id=sub_seg_id,
                            start_uid=start_u,
                            end_uid=end_u,
                            polyline_2d=sub_poly,
                            arc_length=sub_len,
                            base_segment_id=base_seg_id,
                            t_start=t_a,
                            t_end=t_b,
                        )
                    )

        # 内部線および穴輪郭拘束の抽出
        internal_lines: list[np.ndarray] = []
        for path in piece.paths.values():
            if path.role in (PathRole.INTERNAL, PathRole.HOLE) and len(path.nodes) >= 2:
                int_nodes = path.nodes
                int_pts: list[tuple[float, float]] = []
                num_int_segs = len(int_nodes) if path.closed else len(int_nodes) - 1
                for k in range(num_int_segs):
                    ctrls = path.get_segment_controls(k)
                    table = build_arc_table(*ctrls)
                    if k == 0:
                        int_pts.extend(table.points)
                    else:
                        int_pts.extend(table.points[1:])
                if len(int_pts) >= 2:
                    internal_lines.append(np.array(int_pts, dtype=np.float32))

        contours.append(
            PatternContour2D(
                piece_name=piece.name,
                piece_id=piece.id,
                outer_segments=segments,
                internal_lines=internal_lines,
                target_edge_length=piece.target_edge_length,
                layer_id=piece.layer_id,
            )
        )

    # 縫合ペアの変換 (筒モード単体ピースの自動左右辺縫合を含む)
    effective_seams = _collect_effective_seams(doc, contours)
    seam_pairs: list[SeamPairIR] = []
    # ピースIDと名前の相互引当て
    piece_name_map = {pid: p.name for pid, p in doc.pieces.items()}

    for seam in effective_seams:
        sid = seam.id
        p_name_a = piece_name_map.get(seam.a.piece_id)
        p_name_b = piece_name_map.get(seam.b.piece_id)
        if not p_name_a or not p_name_b:
            continue

        seg_id_a = f"{seam.a.start_node_id}_{seam.a.end_node_id}"
        seg_id_b = f"{seam.b.start_node_id}_{seam.b.end_node_id}"

        # 中継点ガイド
        wps = [
            WaypointGuideIR(
                guide_id=wp.id,
                polyline_3d=np.array([wp.position_3d], dtype=np.float32),
            )
            for wp in seam.waypoints
        ]

        seam_pairs.append(
            SeamPairIR(
                pair_id=sid,
                piece_a=p_name_a,
                segment_a_id=seg_id_a,
                piece_b=p_name_b,
                segment_b_id=seg_id_b,
                reverse_b=seam.reverse,
                waypoint_guides=wps,
            )
        )

    return (contours, seam_pairs)


def _collect_effective_seams(doc: Document, contours: list[PatternContour2D]) -> list[Seam]:
    """明示的な Seam 定義に加え、筒モード (is_tubular) が有効で自己縫合が未定義のピースに対し左右側辺の自動縫合を合成して返す."""
    effective_seams: list[Seam] = list(doc.seams.values())
    has_self_seam: set[str] = {
        s.a.piece_id for s in doc.seams.values() if s.a.piece_id == s.b.piece_id
    }

    for contour in contours:
        piece = doc.pieces.get(contour.uid)
        if not piece or piece.id in has_self_seam:
            continue
        pl = doc.placements.get(piece.id)
        if not pl or not pl.is_tubular:
            continue
        path = piece.paths.get(piece.outer_path_id)
        if not path or len(path.nodes) < 2:
            continue
        nodes = path.nodes
        num_orig_segs = len(nodes)
        if num_orig_segs < 2:
            continue

        orig_seg_candidates: list[tuple[float, float, str, str, float]] = []
        for k in range(num_orig_segs):
            n0 = nodes[k]
            n1 = nodes[(k + 1) % num_orig_segs]
            base_id = f"{n0.id}_{n1.id}"
            subs = [s for s in contour.outer_segments if s.segment_id == base_id or s.segment_id.startswith(f"{base_id}@sub")]
            if not subs:
                continue
            all_pts_sub = np.concatenate([s.polyline_2d for s in subs], axis=0)
            mx = float(np.mean(all_pts_sub[:, 0]))
            dv = float(all_pts_sub[-1, 1] - all_pts_sub[0, 1])
            orig_seg_candidates.append((mx, abs(dv), n0.id, n1.id, dv))

        if len(orig_seg_candidates) < 2:
            continue

        sorted_cands = sorted(orig_seg_candidates, key=lambda item: item[0])
        cand_left = sorted_cands[0]
        cand_right = sorted_cands[-1]

        if cand_left[2] == cand_right[2] and cand_left[3] == cand_right[3]:
            continue

        dv_left = cand_left[4]
        dv_right = cand_right[4]
        # 外周ループは通常一方向（CCW）のため、左右の縦辺は上下逆向きに走る -> 平行に繋ぐため reverse=True
        reverse = (dv_left * dv_right < 0.0)

        effective_seams.append(
            Seam(
                id=f"auto_tubular_{piece.id}",
                a=EdgeRange(
                    piece_id=piece.id,
                    path_id=piece.outer_path_id,
                    start_node_id=cand_left[2],
                    end_node_id=cand_left[3],
                ),
                b=EdgeRange(
                    piece_id=piece.id,
                    path_id=piece.outer_path_id,
                    start_node_id=cand_right[2],
                    end_node_id=cand_right[3],
                ),
                reverse=reverse,
            )
        )

    return effective_seams


def _resolve_edge_range_sub_segments(
    er: EdgeRange,
    contour: Optional[PatternContour2D],
) -> tuple[list[BoundarySegmentIR], bool]:
    """EdgeRange に該当するサブセグメント列 (走行順) と外周CCWに対する逆走フラグ is_backward を返す."""
    if not contour:
        return ([], False)

    fwd_id = f"{er.start_node_id}_{er.end_node_id}"
    matching = [
        s for s in contour.outer_segments
        if (s.base_segment_id or s.segment_id.split("@sub")[0]) == fwd_id
    ]
    seg_reversed = False
    if not matching:
        rev_id = f"{er.end_node_id}_{er.start_node_id}"
        matching = [
            s for s in contour.outer_segments
            if (s.base_segment_id or s.segment_id.split("@sub")[0]) == rev_id
        ]
        if matching:
            seg_reversed = True

    if not matching:
        return ([], False)

    raw_t0 = max(0.0, min(1.0, float(er.t0)))
    raw_t1 = max(0.0, min(1.0, float(er.t1)))
    p_t0 = (1.0 - raw_t0) if seg_reversed else raw_t0
    p_t1 = (1.0 - raw_t1) if seg_reversed else raw_t1
    is_backward = p_t0 > p_t1

    if len(matching) == 1:
        return (matching, is_backward)

    t_min = min(p_t0, p_t1)
    t_max = max(p_t0, p_t1)
    selected = [
        s for s in matching
        if t_min - 1e-6 <= 0.5 * (s.t_start + s.t_end) <= t_max + 1e-6
    ]
    res = selected if selected else list(matching)
    if is_backward:
        res = list(reversed(res))
    return (res, is_backward)


def _synchronize_document_seams(
    doc: Document,
    contours: list[PatternContour2D],
    *,
    effective_seams: Optional[list[Seam]] = None,
) -> None:
    """Document の Seam 定義（部分縫合・鎖縫合・筒モード自動縫合含む）に基づき、各セグメントの目標分割数を完全同期する."""
    contour_map = {c.uid: c for c in contours}

    seams_to_sync = effective_seams if effective_seams is not None else _collect_effective_seams(doc, contours)
    for seam in seams_to_sync:
        edges_a = seam.chain_a if seam.chain_a else [seam.a]
        edges_b = seam.chain_b if seam.chain_b else [seam.b]

        sub_segs_a: list[BoundarySegmentIR] = []
        for er in edges_a:
            subs_a, _ = _resolve_edge_range_sub_segments(er, contour_map.get(er.piece_id))
            sub_segs_a.extend(subs_a)

        sub_segs_b: list[BoundarySegmentIR] = []
        for er in edges_b:
            subs_b, _ = _resolve_edge_range_sub_segments(er, contour_map.get(er.piece_id))
            sub_segs_b.extend(subs_b)

        if not sub_segs_a or not sub_segs_b:
            continue

        len_a = sum(s.arc_length for s in sub_segs_a)
        len_b = sum(s.arc_length for s in sub_segs_b)

        if len_a < 1e-6 or len_b < 1e-6:
            continue

        c_a = contour_map.get(edges_a[0].piece_id)
        c_b = contour_map.get(edges_b[0].piece_id)
        target_len = 0.5 * ((c_a.target_edge_length if c_a else 0.015) + (c_b.target_edge_length if c_b else 0.015))
        n_total = max(1, int(round((len_a + len_b) / (2.0 * target_len))))

        # 両側のサブセグメント数が等しく、複数ある場合 (ノッチ同士での合印合わせ)
        if len(sub_segs_a) == len(sub_segs_b) and len(sub_segs_a) > 1:
            sub_segs_b_eff = list(reversed(sub_segs_b)) if seam.reverse else list(sub_segs_b)
            for sa, sb in zip(sub_segs_a, sub_segs_b_eff):
                l_pair = sa.arc_length + sb.arc_length
                n_pair = max(1, int(round(l_pair / (2.0 * target_len))))
                sa.target_subdivisions = n_pair
                sb.target_subdivisions = n_pair
        else:
            # 一般の場合: それぞれの区間に n_total ステップを比例配分
            counts_a = []
            for s in sub_segs_a:
                c = max(1, int(round(n_total * s.arc_length / len_a)))
                counts_a.append(c)
            diff_a = sum(counts_a) - n_total
            if diff_a != 0:
                max_i = int(np.argmax([s.arc_length for s in sub_segs_a]))
                counts_a[max_i] = max(1, counts_a[max_i] - diff_a)
            for s, c in zip(sub_segs_a, counts_a):
                s.target_subdivisions = c

            counts_b = []
            for s in sub_segs_b:
                c = max(1, int(round(n_total * s.arc_length / len_b)))
                counts_b.append(c)
            diff_b = sum(counts_b) - n_total
            if diff_b != 0:
                max_i = int(np.argmax([s.arc_length for s in sub_segs_b]))
                counts_b[max_i] = max(1, counts_b[max_i] - diff_b)
            for s, c in zip(sub_segs_b, counts_b):
                s.target_subdivisions = c

    # 自由区間の目標分割数設定
    for c in contours:
        for seg in c.outer_segments:
            if seg.target_subdivisions is None:
                seg.target_subdivisions = max(1, int(round(seg.arc_length / max(1e-6, c.target_edge_length))))


def _extract_edge_range_verts(
    edge_range: EdgeRange,
    segment_global_map: dict[str, dict[str, list[int]]],
    *,
    piece_name_map: dict[str, str],
    contours: Optional[list[PatternContour2D]] = None,
) -> list[int]:
    """EdgeRange から頂点インデックス列を抽出する (サブセグメント完全対応)."""
    p_name = piece_name_map.get(edge_range.piece_id, edge_range.piece_id)
    seg_map = segment_global_map.get(edge_range.piece_id) or segment_global_map.get(p_name, {})

    contour = next((c for c in contours if c.uid == edge_range.piece_id or c.piece_name == p_name), None) if contours else None
    sub_segs, is_backward = _resolve_edge_range_sub_segments(edge_range, contour)

    all_verts: list[int] = []
    for s in sub_segs:
        raw_v_list = seg_map.get(s.segment_id)
        if not raw_v_list:
            continue
        v_list = list(reversed(raw_v_list)) if is_backward else list(raw_v_list)
        if all_verts and v_list and all_verts[-1] == v_list[0]:
            all_verts.extend(v_list[1:])
        else:
            all_verts.extend(v_list)

    if not all_verts:
        fwd_id = f"{edge_range.start_node_id}_{edge_range.end_node_id}"
        if fwd_id in seg_map:
            raw_list = seg_map[fwd_id]
            return list(reversed(raw_list)) if edge_range.t0 > edge_range.t1 else list(raw_list)
        rev_id = f"{edge_range.end_node_id}_{edge_range.start_node_id}"
        if rev_id in seg_map:
            raw_list = seg_map[rev_id]
            is_back = not (edge_range.t0 > edge_range.t1)
            return list(reversed(raw_list)) if is_back else list(raw_list)
        return []

    return all_verts


def _extract_chain_verts(
    primary_edge: EdgeRange,
    chain: list[EdgeRange],
    *,
    segment_global_map: dict[str, dict[str, list[int]]],
    piece_name_map: dict[str, str],
    contours: Optional[list[PatternContour2D]] = None,
) -> list[int]:
    """主EdgeRangeまたはchainリストから連続した頂点インデックス列を結合して抽出する."""
    edges = chain if chain else [primary_edge]
    all_verts: list[int] = []
    for edge in edges:
        v_list = _extract_edge_range_verts(
            edge,
            segment_global_map,
            piece_name_map=piece_name_map,
            contours=contours,
        )
        if not v_list:
            continue
        if all_verts and v_list and all_verts[-1] == v_list[0]:
            all_verts.extend(v_list[1:])
        else:
            all_verts.extend(v_list)
    return all_verts


def _build_document_sewing_loose_edges(
    base_positions: np.ndarray,
    segment_global_map: dict[str, dict[str, list[int]]],
    *,
    doc: Document,
    bones: Optional[dict[str, Any]] = None,
    contours: Optional[list[PatternContour2D]] = None,
    effective_seams: Optional[list[Seam]] = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """DocumentのSeam定義（部分縫合、鎖縫合、中継点、筒モード自動縫合含む）からLoose Edge列を生成する."""
    all_positions = [base_positions[i].copy() for i in range(len(base_positions))]
    loose_edges: list[tuple[int, int]] = []
    piece_name_map = {p.id: p.name for p in doc.pieces.values()}

    # Pass 1: 各縫合ペアの頂点対応を解決し、布頂点の縫合次数 (接続回数) を集計
    resolved_seams: list[tuple[Seam, list[int], list[int]]] = []
    cloth_degree: dict[int, int] = {}

    if effective_seams is not None:
        eff_seams = effective_seams
    elif contours is not None:
        eff_seams = _collect_effective_seams(doc, contours)
    else:
        eff_seams = list(doc.seams.values())

    for seam in eff_seams:
        verts_a = _extract_chain_verts(
            seam.a,
            seam.chain_a,
            segment_global_map=segment_global_map,
            piece_name_map=piece_name_map,
            contours=contours,
        )
        verts_b = _extract_chain_verts(
            seam.b,
            seam.chain_b,
            segment_global_map=segment_global_map,
            piece_name_map=piece_name_map,
            contours=contours,
        )

        if not verts_a or not verts_b:
            continue

        if seam.reverse:
            verts_b = list(reversed(verts_b))

        na = len(verts_a)
        nb = len(verts_b)
        num_steps = max(na, nb)

        step_pairs: list[tuple[int, int]] = []
        for step in range(num_steps):
            ia = int(round((step / max(1, num_steps - 1)) * (na - 1)))
            ib = int(round((step / max(1, num_steps - 1)) * (nb - 1)))
            va = verts_a[ia]
            vb = verts_b[ib]
            step_pairs.append((va, vb))
            cloth_degree[va] = cloth_degree.get(va, 0) + 1
            cloth_degree[vb] = cloth_degree.get(vb, 0) + 1

        resolved_seams.append((seam, [p[0] for p in step_pairs], [p[1] for p in step_pairs]))

    # Pass 2: Loose Edge 生成 (中継点付きで共有端点を持つ場合は直接縫合にフォールバック)
    seen_direct: set[tuple[int, int]] = set()

    for seam, list_a, list_b in resolved_seams:
        num_steps = len(list_a)

        piece_a = doc.pieces.get(seam.a.piece_id)
        piece_b = doc.pieces.get(seam.b.piece_id)
        pl_a = doc.placements.get(piece_a.id) if piece_a else None
        pl_b = doc.placements.get(piece_b.id) if piece_b else None

        # 筒状縫合 (is_tubular): 手動 waypoint が無く、どちらかのパーツが is_tubular かつボーン指定がある場合
        is_tubular = (not seam.waypoints) and (
            (pl_a and pl_a.is_tubular and pl_a.target_bone)
            or (pl_b and pl_b.is_tubular and pl_b.target_bone)
        )
        target_bone_name = (pl_a.target_bone if (pl_a and pl_a.is_tubular) else (pl_b.target_bone if pl_b else None))
        bone_info = bones.get(target_bone_name) if (bones and target_bone_name) else None

        if not seam.waypoints and not (is_tubular and bone_info is not None):
            for step in range(num_steps):
                va = list_a[step]
                vb = list_b[step]
                pair = (min(va, vb), max(va, vb))
                if pair not in seen_direct:
                    seen_direct.add(pair)
                    loose_edges.append((va, vb))
        elif is_tubular and bone_info is not None:
            # 筒モード: ボーン裏側を迂回する2つの中継点（Waypoints Loop）を自動生成
            pl_ref = pl_a if (pl_a and pl_a.is_tubular) else pl_b
            from .placement import compute_tubular_waypoints_for_pair

            for step in range(num_steps):
                va = list_a[step]
                vb = list_b[step]

                pair = (min(va, vb), max(va, vb))
                if pair in seen_direct:
                    continue
                seen_direct.add(pair)

                wp1, wp2 = compute_tubular_waypoints_for_pair(
                    all_positions[va],
                    all_positions[vb],
                    head_3d=bone_info.head_3d,
                    tail_3d=bone_info.tail_3d,
                    bone_t=pl_ref.bone_t,
                    bone_angle_deg=pl_ref.bone_angle_deg,
                    surface_offset=pl_ref.surface_offset,
                )

                idx_w1 = len(all_positions)
                all_positions.append(wp1)
                idx_w2 = len(all_positions)
                all_positions.append(wp2)

                # 3本のエッジで接続: va -> wp1 -> wp2 -> vb
                loose_edges.append((va, idx_w1))
                loose_edges.append((idx_w1, idx_w2))
                loose_edges.append((idx_w2, vb))
        else:
            wp_positions = [np.array(wp.position_3d, dtype=np.float32) for wp in seam.waypoints]

            for step in range(num_steps):
                va = list_a[step]
                vb = list_b[step]

                # 共有端点（他の縫合線と端点を共有）の場合は直接縫合エッジにフォールバックして分岐トポロジーを防止
                if cloth_degree.get(va, 1) > 1 or cloth_degree.get(vb, 1) > 1:
                    pair = (min(va, vb), max(va, vb))
                    if pair not in seen_direct:
                        seen_direct.add(pair)
                        loose_edges.append((va, vb))
                    continue

                pair = (min(va, vb), max(va, vb))
                if pair in seen_direct:
                    continue
                seen_direct.add(pair)

                chain_nodes = [va]
                for wp_pos in wp_positions:
                    node_idx = len(all_positions)
                    all_positions.append(wp_pos)
                    chain_nodes.append(node_idx)
                chain_nodes.append(vb)

                for k in range(len(chain_nodes) - 1):
                    loose_edges.append((chain_nodes[k], chain_nodes[k + 1]))

    all_pos_arr = np.array(all_positions, dtype=np.float32)
    return all_pos_arr, loose_edges


def build_garment_mesh_geometry(
    doc: Document,
    *,
    bones: Optional[dict[str, Any]] = None,
) -> GarmentMeshGeometry:
    """Document から完全な衣服メッシュ幾何データを構築して返す (Blender非依存)."""
    contours, seam_pairs = document_to_contours_and_seams(doc)
    if not contours:
        return GarmentMeshGeometry(
            vertices_3d=[],
            faces=[],
            sewing_edges=[],
            face_uvs=[],
            face_piece_indices=[],
            face_layer_ids=[],
            piece_vertex_map={},
            pin_vertex_indices=[],
        )

    # 1. 縫合ペアの分割数同期 (鎖縫合・部分縫合を含む完全同期)
    effective_seams = _collect_effective_seams(doc, contours)
    _synchronize_document_seams(doc, contours, effective_seams=effective_seams)

    all_3d_positions: list[np.ndarray] = []
    all_uvs_2d: list[np.ndarray] = []
    all_local_2d: list[np.ndarray] = []
    all_triangles: list[tuple[int, int, int]] = []
    piece_vert_indices: dict[str, list[int]] = {}
    piece_face_layers: list[int] = []
    piece_face_indices: list[int] = []
    segment_global_map: dict[str, dict[str, list[int]]] = {}

    global_vert_offset = 0

    for piece_idx, contour in enumerate(contours):
        piece = doc.pieces.get(contour.uid) or next((p for p in doc.pieces.values() if p.name == contour.piece_name), None)
        placement = doc.placements.get(piece.id) if piece else None

        bnd_loop, seg_map = sample_pattern_boundary_loop(contour)
        target_len = contour.target_edge_length

        # 内部拘束リストの変換
        constraints_list = [
            np.ascontiguousarray(line, dtype=np.float32)
            for line in contour.internal_lines
            if len(line) >= 2
        ]

        # Rust CDT 2D remesh
        verts_2d, tris = core.remesh_pattern_2d(
            np.ascontiguousarray(bnd_loop, dtype=np.float32),
            constraints_list if constraints_list else None,
            target_edge_length=target_len,
            iterations=6,
        )
        verts_2d = np.asarray(verts_2d, dtype=np.float32)
        tris = np.asarray(tris, dtype=np.uint32)

        # 3D 空間への平面配置（ボーン追従またはオイラー角指定）
        target_bone_name = placement.target_bone if placement else None
        bone_info = bones.get(target_bone_name) if (bones and target_bone_name) else None

        if bone_info is not None and placement is not None:
            from .placement import project_2d_points_to_bone_plane
            world_3d = project_2d_points_to_bone_plane(
                verts_2d,
                head_3d=bone_info.head_3d,
                tail_3d=bone_info.tail_3d,
                bone_t=placement.bone_t,
                bone_angle_deg=placement.bone_angle_deg,
                surface_offset=placement.surface_offset,
            )
        else:
            offset_3d = np.array(placement.offset_3d if placement else (0.0, 0.0, 0.0), dtype=np.float32)
            euler_3d = placement.euler_3d if placement else (0.0, 0.0, 0.0)
            R = _euler_to_rotation_matrix(euler_3d)

            local_3d = np.column_stack([verts_2d[:, 0], np.zeros(len(verts_2d), dtype=np.float32), verts_2d[:, 1]])
            world_3d = (local_3d @ R.T) + offset_3d

        n_verts = len(verts_2d)
        p_indices = list(range(global_vert_offset, global_vert_offset + n_verts))
        piece_vert_indices[contour.piece_name] = p_indices

        seg_global: dict[str, list[int]] = {}
        for seg_id, local_indices in seg_map.items():
            seg_global[seg_id] = [idx + global_vert_offset for idx in local_indices]
        segment_global_map[contour.piece_name] = seg_global
        if piece:
            segment_global_map[piece.id] = seg_global

        all_3d_positions.append(world_3d)
        all_local_2d.append(verts_2d)
        # UV展開用: ピースの2Dキャンバス配置変換を適用したワールド2D座標
        uv_2d = piece.transform.apply_array(verts_2d) if piece else verts_2d
        all_uvs_2d.append(uv_2d)

        # 閉じた穴パス (HOLE) のポリゴン境界を抽出
        hole_polygons: list[list[tuple[float, float]]] = []
        if piece:
            for path in piece.paths.values():
                if path.role == PathRole.HOLE and path.closed and len(path.nodes) >= 3:
                    h_pts: list[tuple[float, float]] = []
                    num_h_segs = len(path.nodes)
                    for k in range(num_h_segs):
                        ctrls = path.get_segment_controls(k)
                        table = build_arc_table(*ctrls)
                        if k == 0:
                            h_pts.extend(table.points)
                        else:
                            h_pts.extend(table.points[1:])
                    if len(h_pts) >= 3:
                        hole_polygons.append(h_pts)

        for tri in tris:
            ia, ib, ic = int(tri[0]), int(tri[1]), int(tri[2])
            va = verts_2d[ia]
            vb = verts_2d[ib]
            vc = verts_2d[ic]
            centroid = ((va[0] + vb[0] + vc[0]) / 3.0, (va[1] + vb[1] + vc[1]) / 3.0)

            # 穴内部の三角形を除去
            in_hole = False
            for h_poly in hole_polygons:
                if point_in_polygon(centroid, h_poly):
                    in_hole = True
                    break

            if in_hole:
                continue

            all_triangles.append((
                ia + global_vert_offset,
                ib + global_vert_offset,
                ic + global_vert_offset,
            ))
            piece_face_layers.append(contour.layer_id)
            piece_face_indices.append(piece_idx)

        global_vert_offset += n_verts

    base_positions_3d = np.concatenate(all_3d_positions, axis=0)
    all_uvs = np.concatenate(all_uvs_2d, axis=0)
    all_locals = np.concatenate(all_local_2d, axis=0)

    # 2. 縫合 Loose Edge の生成 (部分縫合・鎖縫合・中継点・筒状迂回対応)
    final_positions_3d, sewing_edges = _build_document_sewing_loose_edges(
        base_positions_3d,
        segment_global_map,
        doc=doc,
        bones=bones,
        contours=contours,
        effective_seams=effective_seams,
    )

    # 3. UV 全体正規化 (2% 余白 [0.02, 0.98] 適用)
    min_uv = np.min(all_uvs, axis=0)
    max_uv = np.max(all_uvs, axis=0)
    span = np.maximum(max_uv - min_uv, 1e-6)
    max_span = float(max(span[0], span[1]))
    normalized_uvs = 0.02 + 0.96 * ((all_uvs - min_uv) / max_span)

    # 各面のループ頂点 UV
    face_uvs: list[tuple[float, float]] = []
    for tri in all_triangles:
        for v_idx in tri:
            uv = normalized_uvs[v_idx]
            face_uvs.append((float(uv[0]), float(uv[1])))

    # 4. アンカーピン頂点の検索（ピースローカル座標での最近傍頂点）
    pin_indices: list[int] = []
    for anchor in doc.anchors.values():
        piece = doc.pieces.get(anchor.piece_id)
        if piece and piece.name in piece_vert_indices:
            v_idxs = piece_vert_indices[piece.name]
            p_locals = all_locals[v_idxs]
            target_pos = np.array([anchor.position[0], anchor.position[1]], dtype=np.float32)
            dists = np.linalg.norm(p_locals - target_pos, axis=1)
            closest_local_idx = int(np.argmin(dists))
            pin_indices.append(v_idxs[closest_local_idx])

    vertices_list = [tuple(p) for p in final_positions_3d]

    return GarmentMeshGeometry(
        vertices_3d=vertices_list,
        faces=all_triangles,
        sewing_edges=sewing_edges,
        face_uvs=face_uvs,
        face_piece_indices=piece_face_indices,
        face_layer_ids=piece_face_layers,
        piece_vertex_map=piece_vert_indices,
        pin_vertex_indices=pin_indices,
    )
