"""
カーブ（Curve）オブジェクト高速抽出および紐（Drawstring）統合ユーティリティ (taremin_cloth.utils.curve_extract)

Blenderの bpy.types.Curve と物理シミュレーション頂点列・エッジ列の相互変換、
選択エッジからのパス探索（縫合ギャップ跨ぎ対応）、および紐パーツのランタイム構造を提供する。
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Set, Dict, List
import numpy as np


@dataclass
class CurvePartRuntimeData:
    """単一の紐（Curve）オブジェクトと布メッシュの統合ランタイム情報"""
    curve_obj_name: str
    vertex_offset: int                    # マージ後頂点配列における開始インデックス
    vertex_count: int                     # 紐の頂点数
    edge_offset: int                      # マージ後 normal_edges における開始インデックス
    left_tail_edge_indices: List[int]     # 左垂れ部分のエッジインデックス列
    channel_edge_indices: List[int]       # トンネル内区間のエッジインデックス列
    right_tail_edge_indices: List[int]    # 右垂れ部分のエッジインデックス列
    seam_bridge_edge_indices: List[int]   # 分割型紙の縫合ギャップ跨ぎエッジインデックス列
    initial_left_len: float               # 左垂れの初期自然長 (m)
    initial_channel_len: float            # トンネル区間の初期自然長 (m)
    initial_right_len: float              # 右垂れの初期自然長 (m)
    guide_cloth_verts: List[int]          # フード縁のガイド頂点列
    eyelet_pairs: List[Tuple[int, int]]   # 間引き結合 (cloth_vert_idx, curve_vert_idx) ペア


def extract_curve_polyline(
    curve_obj,
    *,
    apply_world_matrix: bool = True,
    default_radius: float = 0.003,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Blenderの Curve オブジェクトから制御点座標、エッジ列、半径配列を抽出する。
    
    Args:
        curve_obj: bpy.types.Object (type == 'CURVE')
        apply_world_matrix: ワールド座標へ変換するかどうか
        default_radius: bevel_depth が 0 の場合のデフォルト半径 (m)
        
    Returns:
        coords: shape [M, 3], float32 (頂点座標配列)
        edges: shape [M - 1, 2], uint32 (連続ラインエッジ)
        radii: shape [M], float32 (頂点ごとの厚み・半径)
    """
    curve_data = getattr(curve_obj, "data", None)
    if curve_data is None or not hasattr(curve_data, "splines") or len(curve_data.splines) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.uint32),
            np.empty(0, dtype=np.float32),
        )

    spline = curve_data.splines[0]
    bevel_depth = getattr(curve_data, "bevel_depth", 0.0)
    base_radius = bevel_depth if bevel_depth > 0.0 else default_radius

    if spline.type == 'BEZIER':
        bp_list = spline.bezier_points
        n_points = len(bp_list)
        if n_points == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 2), dtype=np.uint32),
                np.empty(0, dtype=np.float32),
            )
        raw_coords = np.empty(n_points * 3, dtype=np.float32)
        bp_list.foreach_get("co", raw_coords)
        coords = raw_coords.reshape((n_points, 3))

        raw_radii = np.empty(n_points, dtype=np.float32)
        bp_list.foreach_get("radius", raw_radii)
        radii = raw_radii * base_radius
    else:
        # POLY または NURBS
        p_list = spline.points
        n_points = len(p_list)
        if n_points == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 2), dtype=np.uint32),
                np.empty(0, dtype=np.float32),
            )
        raw_coords_4d = np.empty(n_points * 4, dtype=np.float32)
        p_list.foreach_get("co", raw_coords_4d)
        coords_4d = raw_coords_4d.reshape((n_points, 4))
        # 4D (x, y, z, w) -> 3D (x, y, z)
        w = coords_4d[:, 3:4]
        w_safe = np.where(np.abs(w) > 1e-6, w, 1.0)
        coords = (coords_4d[:, :3] / w_safe).astype(np.float32)

        raw_radii = np.empty(n_points, dtype=np.float32)
        p_list.foreach_get("radius", raw_radii)
        radii = raw_radii * base_radius

    if apply_world_matrix and hasattr(curve_obj, "matrix_world"):
        mat = np.array(curve_obj.matrix_world, dtype=np.float32).reshape((4, 4))
        # アフィン変換: [N, 3] -> [N, 4] -> mat @ v
        ones = np.ones((n_points, 1), dtype=np.float32)
        coords_h = np.hstack([coords, ones])
        coords = (coords_h @ mat.T)[:, :3].astype(np.float32)

    # 1Dストランドの連続ラインエッジ列 [0-1, 1-2, ..., (M-2)-(M-1)]
    if n_points > 1:
        v0 = np.arange(n_points - 1, dtype=np.uint32)
        v1 = np.arange(1, n_points, dtype=np.uint32)
        edges = np.column_stack([v0, v1])
    else:
        edges = np.empty((0, 2), dtype=np.uint32)

    return coords, edges, radii


def apply_coords_to_curve(
    curve_obj,
    coords: np.ndarray,
    *,
    update_handles: bool = True,
    is_world_coords: bool = True,
) -> None:
    """
    シミュレーション後のワールド座標配列（またはローカル座標配列）を
    Curve オブジェクトの制御点へ一括書き戻す。
    
    Args:
        curve_obj: bpy.types.Object (type == 'CURVE')
        coords: shape [M, 3], float32 (座標列)
        update_handles: Bezierカーブの場合にハンドル接線を自動整列するか
        is_world_coords: coords がワールド座標かどうか (False の場合はローカル座標としてそのまま適用)
    """
    curve_data = getattr(curve_obj, "data", None)
    if curve_data is None or not hasattr(curve_data, "splines") or len(curve_data.splines) == 0:
        return

    n_points = coords.shape[0]
    if n_points == 0:
        return

    # ワールド座標からローカル座標への逆変換
    if is_world_coords and hasattr(curve_obj, "matrix_world"):
        mat_world = curve_obj.matrix_world
        if hasattr(mat_world, "inverted"):
            inv_mat = np.array(mat_world.inverted(), dtype=np.float32).reshape((4, 4))
        else:
            inv_mat = np.linalg.inv(np.array(mat_world, dtype=np.float32).reshape((4, 4)))
        ones = np.ones((n_points, 1), dtype=np.float32)
        coords_h = np.hstack([coords, ones])
        local_coords = (coords_h @ inv_mat.T)[:, :3].astype(np.float32)
    else:
        local_coords = coords.astype(np.float32)

    spline = curve_data.splines[0]

    if spline.type == 'BEZIER':
        bp_list = spline.bezier_points
        if len(bp_list) != n_points:
            return
        bp_list.foreach_set("co", local_coords.ravel())

        if update_handles and n_points >= 2:
            # 連続する点列から自然な接線ベクトルを計算してハンドルを更新
            dirs = np.zeros_like(local_coords)
            dirs[1:-1] = (local_coords[2:] - local_coords[:-2]) * 0.25
            dirs[0] = (local_coords[1] - local_coords[0]) * 0.25
            dirs[-1] = (local_coords[-1] - local_coords[-2]) * 0.25

            left_handles = local_coords - dirs
            right_handles = local_coords + dirs

            bp_list.foreach_set("handle_left", left_handles.ravel())
            bp_list.foreach_set("handle_right", right_handles.ravel())
    else:
        p_list = spline.points
        if len(p_list) != n_points:
            return
        # 4D座標 (x, y, z, 1.0)
        coords_4d = np.ones((n_points, 4), dtype=np.float32)
        coords_4d[:, :3] = local_coords
        p_list.foreach_set("co", coords_4d.ravel())


def build_ordered_path_across_seams(
    edges: np.ndarray,
    selected_edge_indices: List[int],
    *,
    seam_cluster_map: Optional[Dict[int, List[int]]] = None,
) -> Tuple[List[int], Set[Tuple[int, int]]]:
    """
    選択されたエッジ列から順序付きの頂点列を構築する。
    左右2枚の分割型紙などで間に縫合エッジ（seam_cluster_map）が存在する場合、
    対応する縫合頂点へ自動ジャンプして単一の連続パスを復元する。
    
    Args:
        edges: 全エッジ配列 shape [E, 2], uint32
        selected_edge_indices: 選択されたエッジインデックス列
        seam_cluster_map: 頂点 -> 同一縫合クラスタの他頂点リストの辞書
        
    Returns:
        ordered_verts: 端から端までの順序付き頂点インデックス列
        seam_bridge_pairs: 縫合ギャップを跨いでジャンプした頂点ペアの集合 {(v_from, v_to), ...}
    """
    if len(selected_edge_indices) == 0:
        return [], set()

    # 1. 選択エッジから隣接リストを構築
    adj: Dict[int, List[int]] = {}
    for e_idx in selected_edge_indices:
        if e_idx >= len(edges):
            continue
        v0, v1 = int(edges[e_idx, 0]), int(edges[e_idx, 1])
        if v0 == v1:
            continue
        adj.setdefault(v0, []).append(v1)
        adj.setdefault(v1, []).append(v0)

    if not adj:
        return [], set()

    # 2. 縫合パートナー情報の整理
    seam_jump: Dict[int, int] = {}
    if seam_cluster_map:
        for v, partners in seam_cluster_map.items():
            for p in partners:
                if p in adj and v in adj:
                    seam_jump[v] = p
                    seam_jump[p] = v

    # 3. グラフ上の端点（次数1）を検出
    # 縫合跨ぎで接続される頂点は論理的に次数が+1されるため、真のエンドポイントは物理的に次数1かつ縫合ジャンプ先がないもの
    endpoints = []
    for v, neighbors in adj.items():
        has_seam = v in seam_jump
        effective_degree = len(neighbors) + (1 if has_seam else 0)
        if effective_degree == 1:
            endpoints.append(v)

    # 端点が見つからない場合は任意の次数1頂点、または最初の頂点を起点とする
    if endpoints:
        start_vert = endpoints[0]
    else:
        deg1_nodes = [v for v, neighbors in adj.items() if len(neighbors) == 1]
        start_vert = deg1_nodes[0] if deg1_nodes else next(iter(adj.keys()))

    # 4. パス追跡
    ordered_verts: List[int] = [start_vert]
    visited_edges: Set[Tuple[int, int]] = set()
    visited_verts: Set[int] = {start_vert}
    seam_bridge_pairs: Set[Tuple[int, int]] = set()

    curr = start_vert
    while True:
        # まず通常の選択エッジの隣接頂点を探す
        next_v = None
        for nbr in adj.get(curr, []):
            edge_key = (min(curr, nbr), max(curr, nbr))
            if edge_key not in visited_edges and nbr not in visited_verts:
                visited_edges.add(edge_key)
                next_v = nbr
                break

        if next_v is not None:
            ordered_verts.append(next_v)
            visited_verts.add(next_v)
            curr = next_v
            continue

        # 通常の隣接頂点がなく、縫合ジャンプ先がある場合
        jump_v = seam_jump.get(curr)
        if jump_v is not None and jump_v not in visited_verts:
            pair = (min(curr, jump_v), max(curr, jump_v))
            seam_bridge_pairs.add(pair)
            ordered_verts.append(jump_v)
            visited_verts.add(jump_v)
            curr = jump_v
            continue

        # これ以上進めない場合は探索終了
        break

    return ordered_verts, seam_bridge_pairs
