# -*- coding: utf-8 -*-
"""型紙エディタのメッシュビルドデータモデルおよびジオメトリ生成 (Blender非依存)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

import taremin_cloth_core as core
from ..utils.pattern_ir import (
    BoundarySegmentIR,
    PatternContour2D,
    SeamPairIR,
    WaypointGuideIR,
    build_sewing_loose_edges,
    sample_pattern_boundary_loop,
    synchronize_seam_subdivisions,
)
from .document import Document, NodeKind, PathRole
from .geometry import (
    Vec2,
    build_arc_table,
    compute_auto_handles,
    make_segment_controls,
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

        # AUTOノードのハンドルを計算
        eff_handles: list[tuple[Optional[Vec2], Optional[Vec2]]] = []
        for i, nd in enumerate(nodes):
            if nd.kind == NodeKind.AUTO:
                prev_nd = nodes[(i - 1) % n_nodes]
                next_nd = nodes[(i + 1) % n_nodes]
                in_h, out_h = compute_auto_handles(prev_nd.position, nd.position, next_nd.position)
                eff_handles.append((in_h, out_h))
            else:
                eff_handles.append((nd.in_handle, nd.out_handle))

        for i in range(n_nodes):
            curr_node = nodes[i]
            next_node = nodes[(i + 1) % n_nodes]

            out_h = eff_handles[i][1]
            in_h = eff_handles[(i + 1) % n_nodes][0]

            ctrls = make_segment_controls(curr_node.position, out_h, in_h, next_node.position)
            table = build_arc_table(*ctrls)

            poly_pts = np.array(table.points, dtype=np.float32)
            seg_id = f"{curr_node.id}_{next_node.id}"

            segments.append(
                BoundarySegmentIR(
                    segment_id=seg_id,
                    start_uid=curr_node.id,
                    end_uid=next_node.id,
                    polyline_2d=poly_pts,
                    arc_length=table.total_length,
                )
            )

        contours.append(
            PatternContour2D(
                piece_name=piece.name,
                outer_segments=segments,
                target_edge_length=piece.target_edge_length,
                layer_id=piece.layer_id,
            )
        )

    # 縫合ペアの変換
    seam_pairs: list[SeamPairIR] = []
    # ピースIDと名前の相互引当て
    piece_name_map = {pid: p.name for pid, p in doc.pieces.items()}

    for sid, seam in doc.seams.items():
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


def build_garment_mesh_geometry(doc: Document) -> GarmentMeshGeometry:
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

    # 1. 縫合ペアの分割数同期
    synchronize_seam_subdivisions(contours, seam_pairs)

    all_3d_positions: list[np.ndarray] = []
    all_uvs_2d: list[np.ndarray] = []
    all_triangles: list[tuple[int, int, int]] = []
    piece_vert_indices: dict[str, list[int]] = {}
    piece_face_layers: list[int] = []
    piece_face_indices: list[int] = []
    segment_global_map: dict[str, dict[str, list[int]]] = {}

    global_vert_offset = 0

    for piece_idx, contour in enumerate(contours):
        piece = next((p for p in doc.pieces.values() if p.name == contour.piece_name), None)
        placement = doc.placements.get(piece.id) if piece else None

        bnd_loop, seg_map = sample_pattern_boundary_loop(contour)
        target_len = contour.target_edge_length

        # Rust CDT 2D remesh
        verts_2d, tris = core.remesh_pattern_2d(
            np.ascontiguousarray(bnd_loop, dtype=np.float32),
            None,
            target_edge_length=target_len,
            iterations=6,
        )
        verts_2d = np.asarray(verts_2d, dtype=np.float32)
        tris = np.asarray(tris, dtype=np.uint32)

        # 3D 空間への初期配置 (u, 0, v)
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

        all_3d_positions.append(world_3d)
        all_uvs_2d.append(verts_2d)

        for tri in tris:
            all_triangles.append((
                int(tri[0]) + global_vert_offset,
                int(tri[1]) + global_vert_offset,
                int(tri[2]) + global_vert_offset,
            ))
            piece_face_layers.append(contour.layer_id)
            piece_face_indices.append(piece_idx)

        global_vert_offset += n_verts

    base_positions_3d = np.concatenate(all_3d_positions, axis=0)
    all_uvs = np.concatenate(all_uvs_2d, axis=0)

    # 2. 縫合 Loose Edge の生成
    final_positions_3d, sewing_edges = build_sewing_loose_edges(
        base_positions_3d,
        segment_global_map,
        seams=seam_pairs,
    )

    # 3. UV 全体正規化 [0, 1]
    min_uv = np.min(all_uvs, axis=0)
    max_uv = np.max(all_uvs, axis=0)
    span = np.maximum(max_uv - min_uv, 1e-6)
    max_span = float(max(span[0], span[1]))
    normalized_uvs = (all_uvs - min_uv) / max_span

    # 各面のループ頂点 UV
    face_uvs: list[tuple[float, float]] = []
    for tri in all_triangles:
        for v_idx in tri:
            uv = normalized_uvs[v_idx]
            face_uvs.append((float(uv[0]), float(uv[1])))

    # 4. アンカーピン頂点の検索（最近傍頂点）
    pin_indices: list[int] = []
    for anchor in doc.anchors.values():
        piece = doc.pieces.get(anchor.piece_id)
        if piece and piece.name in piece_vert_indices:
            v_idxs = piece_vert_indices[piece.name]
            p_uvs = all_uvs[v_idxs]
            target_pos = np.array([anchor.position[0], anchor.position[1]], dtype=np.float32)
            dists = np.linalg.norm(p_uvs - target_pos, axis=1)
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
