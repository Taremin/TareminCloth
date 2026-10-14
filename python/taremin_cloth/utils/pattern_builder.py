# -*- coding: utf-8 -*-
"""Milestone 3: Blender 2D型紙（Curve / Mesh）から単一3D衣装メッシュへの統合ビルド (pattern_builder.py)."""

import math
from typing import Optional
import bpy
import numpy as np

import taremin_cloth_core as core
from .pattern_ir import (
    BoundarySegmentIR,
    PatternContour2D,
    SeamPairIR,
    synchronize_seam_subdivisions,
    sample_pattern_boundary_loop,
    build_sewing_loose_edges,
)


def extract_contour_from_curve(obj: bpy.types.Object) -> PatternContour2D:
    """2D Curve オブジェクトから PatternContour2D を抽出する."""
    curve = obj.data
    pts: list[list[float]] = []

    for spline in curve.splines:
        if spline.type == 'POLY':
            for p in spline.points:
                pts.append([p.co[0], p.co[1]])
        elif spline.type == 'BEZIER':
            for bp in spline.bezier_points:
                pts.append([bp.co[0], bp.co[1]])

    if len(pts) < 3:
        raise ValueError(f"Curve object '{obj.name}' must have at least 3 points")

    pts_arr = np.array(pts, dtype=np.float32)

    # 符号付き面積が負（CW）なら反転して CCW に統一
    x = pts_arr[:, 0]
    y = pts_arr[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area < 0:
        pts_arr = pts_arr[::-1]

    # 各辺を BoundarySegmentIR として構築
    n_pts = len(pts_arr)
    segments: list[BoundarySegmentIR] = []
    seg_names = ["bottom", "right", "top", "left"] if n_pts == 4 else [str(i) for i in range(n_pts)]

    for i in range(n_pts):
        p0 = pts_arr[i]
        p1 = pts_arr[(i + 1) % n_pts]
        length = float(np.linalg.norm(p1 - p0))
        seg_id = f"{obj.name}_{seg_names[i]}"
        segments.append(
            BoundarySegmentIR(
                segment_id=seg_id,
                start_uid=f"{obj.name}_c{i}",
                end_uid=f"{obj.name}_c{(i + 1) % n_pts}",
                polyline_2d=np.stack([p0, p1], axis=0),
                arc_length=length,
            )
        )

    pat_settings = getattr(obj, "taremin_pattern", None)
    target_edge_len = getattr(pat_settings, "target_edge_length", 0.015) if pat_settings else 0.015
    layer_id = getattr(pat_settings, "layer_id", 0) if pat_settings else 0

    return PatternContour2D(
        piece_name=obj.name,
        outer_segments=segments,
        target_edge_length=target_edge_len,
        layer_id=layer_id,
    )


def extract_contour_from_mesh(obj: bpy.types.Object) -> PatternContour2D:
    """平面 Mesh オブジェクトから外周境界ループを取り出し PatternContour2D を抽出する."""
    mesh = obj.data
    coords = [v.co[:2] for v in mesh.vertices]
    if len(coords) < 3:
        raise ValueError(f"Mesh object '{obj.name}' must have at least 3 vertices")

    # 外周エッジ（1つの面にしか属さないエッジ）の抽出
    edge_face_count: dict[tuple[int, int], int] = {}
    for poly in mesh.polygons:
        v_list = list(poly.vertices)
        for i in range(len(v_list)):
            u = v_list[i]
            v = v_list[(i + 1) % len(v_list)]
            e = (min(u, v), max(u, v))
            edge_face_count[e] = edge_face_count.get(e, 0) + 1

    boundary_edges = [e for e, count in edge_face_count.items() if count == 1]
    if not boundary_edges:
        # 面がないワイヤーフレーム等の場合は全エッジを境界とみなす
        boundary_edges = [(min(e.vertices[0], e.vertices[1]), max(e.vertices[0], e.vertices[1])) for e in mesh.edges]

    # 外周ループの連結
    from collections import defaultdict
    adj = defaultdict(list)
    for u, v in boundary_edges:
        adj[u].append(v)
        adj[v].append(u)

    loop_verts: list[int] = []
    if adj:
        curr = next(iter(adj))
        prev = None
        loop_verts.append(curr)
        while len(loop_verts) < len(adj):
            neighbors = [n for n in adj[curr] if n != prev]
            if not neighbors:
                break
            nxt = neighbors[0]
            loop_verts.append(nxt)
            prev, curr = curr, nxt

    pts_arr = np.array([coords[v] for v in loop_verts], dtype=np.float32)

    # CCW 判定
    x = pts_arr[:, 0]
    y = pts_arr[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area < 0:
        pts_arr = pts_arr[::-1]

    n_pts = len(pts_arr)
    segments: list[BoundarySegmentIR] = []
    seg_names = ["bottom", "right", "top", "left"] if n_pts == 4 else [str(i) for i in range(n_pts)]

    for i in range(n_pts):
        p0 = pts_arr[i]
        p1 = pts_arr[(i + 1) % n_pts]
        length = float(np.linalg.norm(p1 - p0))
        seg_id = f"{obj.name}_{seg_names[i]}"
        segments.append(
            BoundarySegmentIR(
                segment_id=seg_id,
                start_uid=f"{obj.name}_c{i}",
                end_uid=f"{obj.name}_c{(i + 1) % n_pts}",
                polyline_2d=np.stack([p0, p1], axis=0),
                arc_length=length,
            )
        )

    pat_settings = getattr(obj, "taremin_pattern", None)
    target_edge_len = getattr(pat_settings, "target_edge_length", 0.015) if pat_settings else 0.015
    layer_id = getattr(pat_settings, "layer_id", 0) if pat_settings else 0

    return PatternContour2D(
        piece_name=obj.name,
        outer_segments=segments,
        target_edge_length=target_edge_len,
        layer_id=layer_id,
    )


def extract_pattern_contour(obj: bpy.types.Object) -> PatternContour2D:
    """オブジェクトのタイプに応じて Curve または Mesh から PatternContour2D を抽出する."""
    if obj.type == 'CURVE':
        return extract_contour_from_curve(obj)
    elif obj.type == 'MESH':
        return extract_contour_from_mesh(obj)
    else:
        raise TypeError(f"Unsupported pattern object type '{obj.type}'. Must be CURVE or MESH.")


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


def build_garment_from_patterns(
    scene: bpy.types.Scene,
    pattern_objects: list[bpy.types.Object],
    target_name: str = "Garment_Cloth",
    seams: Optional[list[SeamPairIR]] = None,
) -> bpy.types.Object:
    """2D型紙オブジェクト群から単一の3D衣服シミュレーションメッシュを統合構築する."""
    if not pattern_objects:
        raise ValueError("pattern_objects cannot be empty")

    seams = seams or []

    # 1. 各型紙から PatternContour2D を抽出
    contours: list[PatternContour2D] = []
    obj_map: dict[str, bpy.types.Object] = {}
    for obj in pattern_objects:
        contour = extract_pattern_contour(obj)
        contours.append(contour)
        obj_map[obj.name] = obj

    # 2. 縫合ペアの分割数を同期
    synchronize_seam_subdivisions(contours, seams)

    # 3. 各パーツを 2D CDT + 等方性リメッシュでメッシュ化
    all_3d_positions: list[np.ndarray] = []
    all_uvs_2d: list[np.ndarray] = []
    all_triangles: list[list[int]] = []
    piece_vert_indices: dict[str, list[int]] = {}
    piece_face_layers: list[int] = []
    segment_global_map: dict[str, dict[str, list[int]]] = {}

    global_vert_offset = 0

    for contour in contours:
        obj = obj_map[contour.piece_name]
        pat_settings = getattr(obj, "taremin_pattern", None)

        bnd_loop, seg_map = sample_pattern_boundary_loop(contour)
        target_len = contour.target_edge_length

        # Rust core 2D remesh
        verts_2d, tris = core.remesh_pattern_2d(
            np.ascontiguousarray(bnd_loop, dtype=np.float32),
            None,
            target_edge_length=target_len,
            iterations=6,
        )
        verts_2d = np.asarray(verts_2d, dtype=np.float32)
        tris = np.asarray(tris, dtype=np.uint32)

        # 3D 空間への初期配置: (u, 0, v) に回転・オフセットを適用
        offset_3d = np.array(getattr(pat_settings, "arrangement_offset", (0.0, 0.0, 0.0)), dtype=np.float32)
        euler_3d = getattr(pat_settings, "arrangement_euler", (0.0, 0.0, 0.0))
        R = _euler_to_rotation_matrix(euler_3d)

        # 2D 型紙の (u, v) を 3D 空間の (u, 0, v) へマッピング
        local_3d = np.column_stack([verts_2d[:, 0], np.zeros(len(verts_2d), dtype=np.float32), verts_2d[:, 1]])
        world_3d = (local_3d @ R.T) + offset_3d

        n_verts_piece = len(verts_2d)
        p_indices = list(range(global_vert_offset, global_vert_offset + n_verts_piece))
        piece_vert_indices[contour.piece_name] = p_indices

        # セグメント別グローバル頂点マップ
        seg_global: dict[str, list[int]] = {}
        for seg_id, local_indices in seg_map.items():
            seg_global[seg_id] = [idx + global_vert_offset for idx in local_indices]
        segment_global_map[contour.piece_name] = seg_global

        # 統合バッファへの追加
        all_3d_positions.append(world_3d)
        all_uvs_2d.append(verts_2d)

        for tri in tris:
            all_triangles.append([int(tri[0]) + global_vert_offset, int(tri[1]) + global_vert_offset, int(tri[2]) + global_vert_offset])
            piece_face_layers.append(contour.layer_id)

        global_vert_offset += n_verts_piece

    base_positions_arr = np.vstack(all_3d_positions).astype(np.float32)

    # 4. 縫合および中継ガイドの Loose Edge 生成
    all_positions_arr, loose_edges = build_sewing_loose_edges(
        base_positions=base_positions_arr,
        segment_vertex_map=segment_global_map,
        seams=seams,
    )

    # 5. 単一 Blender Mesh オブジェクトの作成
    # 既存の同名オブジェクトがあれば置換または新規作成
    old_obj = scene.objects.get(target_name)
    if old_obj:
        bpy.data.objects.remove(old_obj, do_unlink=True)
    old_mesh = bpy.data.meshes.get(target_name)
    if old_mesh:
        bpy.data.meshes.remove(old_mesh, do_unlink=True)

    garment_mesh = bpy.data.meshes.new(name=target_name)
    garment_obj = bpy.data.objects.new(name=target_name, object_data=garment_mesh)
    scene.collection.objects.link(garment_obj)

    # from_pydata で面と Loose Edge を一括構築
    py_verts = [tuple(pos) for pos in all_positions_arr]
    py_edges = loose_edges
    py_faces = all_triangles

    garment_mesh.from_pydata(py_verts, py_edges, py_faces)
    garment_mesh.update()

    # 6. UVレイヤー 'UVMap_Pattern' の設定
    uv_layer = garment_mesh.uv_layers.new(name="UVMap_Pattern")
    all_uvs_concat = np.vstack(all_uvs_2d).astype(np.float32)

    garment_mesh.calc_loop_triangles()
    for tri in garment_mesh.loop_triangles:
        for loop_idx, vert_idx in zip(tri.loops, tri.vertices):
            if vert_idx < len(all_uvs_concat):
                uv_layer.data[loop_idx].uv = all_uvs_concat[vert_idx]

    # 7. パーツ別頂点グループ 'Piece_<name>' の登録
    for piece_name, v_indices in piece_vert_indices.items():
        vg = garment_obj.vertex_groups.new(name=f"Piece_{piece_name}")
        vg.add(v_indices, 1.0, 'REPLACE')

    # 8. 面属性 'cloth_layer' の登録
    if all_triangles:
        try:
            layer_attr = garment_mesh.attributes.new(name="cloth_layer", type='INT', domain='FACE')
            for f_idx, layer_id in enumerate(piece_face_layers):
                layer_attr.data[f_idx].value = layer_id
        except Exception:
            pass

    # 9. 布シミュレーション設定の初期化
    if hasattr(garment_obj, "taremin_cloth"):
        garment_obj.taremin_cloth.is_cloth = True
        garment_obj.taremin_cloth.enable_sewing = True

    garment_mesh.update()
    return garment_obj
