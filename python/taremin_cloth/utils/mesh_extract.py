"""
メッシュデータ高速抽出ユーティリティ (taremin_cloth.utils.mesh_extract)
Blenderの bpy.types.Mesh から NumPy 配列へのゼロコピー/一括データ転送および
布シミュレーション初期化用データの抽出・分類を提供する。
"""

from typing import NamedTuple, Optional, Tuple
import numpy as np

try:
    import taremin_cloth_core as _core

    _RUST_AREAL = hasattr(_core, "compute_areal_inv_masses")
except Exception:
    _core = None
    _RUST_AREAL = False


from dataclasses import dataclass
from .logger import logger
from .validation import detect_degenerate_faces, validate_sewing_topology


@dataclass
class WaypointSeam:
    """中継点（Waypoint）を持つ縫合パス（折れ線スプリング）"""
    vert_a: int                      # 布面頂点Aのインデックス
    vert_b: int                      # 布面頂点Bのインデックス
    waypoint_indices: list[int]      # 中継点頂点インデックス列 [W1, W2, ..., Wk] (順序付き)
    waypoint_coords: np.ndarray      # 中継点の初期3D座標列 [K, 3], float32


class ClothMeshData(NamedTuple):
    """布シミュレーション初期化に必要なメッシュデータコンテナ"""
    positions: np.ndarray          # shape [N, 3], float32
    normal_edges: np.ndarray       # shape [E, 2], uint32
    faces: Optional[np.ndarray]    # shape [F, 3], uint32 (存在しない場合は None)
    sewing_edges: Optional[np.ndarray]  # shape [S, 2], uint32 (存在しない場合は None)
    inv_masses: np.ndarray         # shape [N], float32
    layer_ids: Optional[np.ndarray] = None  # shape [N], uint32 (存在しない場合は None)
    waypoint_seams: Optional[list] = None   # list[WaypointSeam] (存在しない場合は None)
    seam_cluster_map: Optional[dict[int, list[int]]] = None  # 頂点インデックス -> 同一縫合クラスタの他頂点リスト (存在しない場合は None)
    thicknesses: Optional[np.ndarray] = None  # shape [N], float32 (存在しない場合は None)
    curve_parts: Optional[list] = None  # list[CurvePartRuntimeData] (存在しない場合は None)


def extract_mesh_vertices_and_triangles(mesh) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Blenderメッシュから頂点座標配列と三角形面インデックス配列を高速抽出する。
    
    Args:
        mesh: bpy.types.Mesh
    Returns:
        positions: shape [N, 3], float32
        faces: shape [M, 3], int32 (三角形が存在しない場合は None)
    """
    n_verts = len(mesh.vertices)
    if n_verts == 0:
        return np.empty((0, 3), dtype=np.float32), None

    raw_coords = np.empty(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", raw_coords)
    positions = raw_coords.reshape((n_verts, 3))

    mesh.calc_loop_triangles()
    n_tris = len(mesh.loop_triangles)
    if n_tris == 0:
        return positions, None

    tri_indices = np.empty(n_tris * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tri_indices)
    faces = tri_indices.reshape((n_tris, 3))

    return positions, faces


def get_pin_inv_masses(obj, *, settings=None, n_verts: Optional[int] = None) -> np.ndarray:
    """
    布オブジェクトのピン留め頂点グループからインバースマス配列を計算する。
    ウェイト 1.0 -> 質量無限（inv_mass = 0.0, 完全固定）
    ウェイト 0.0 -> 通常質量（inv_mass = 1.0）
    """
    if n_verts is None:
        n_verts = len(obj.data.vertices)

    inv_masses = np.ones(n_verts, dtype=np.float32)
    if not obj or not hasattr(obj, "vertex_groups"):
        return inv_masses

    vg_name = getattr(settings, "pin_vertex_group", "") if settings else ""
    pin_vg = None
    if vg_name:
        pin_vg = obj.vertex_groups.get(vg_name)
    if not pin_vg:
        pin_vg = obj.vertex_groups.get("Pin") or obj.vertex_groups.get("Cloth_Pin")

    if pin_vg:
        for i in range(n_verts):
            try:
                weight = pin_vg.weight(i)
                inv_masses[i] = max(0.0, 1.0 - weight)
            except RuntimeError:
                inv_masses[i] = 1.0

    return inv_masses


def compute_combined_inv_masses(
    positions: np.ndarray,
    faces: Optional[np.ndarray],
    *,
    strand_edges: Optional[np.ndarray] = None,
    pin_weights: Optional[np.ndarray] = None,
    areal_density: float = 0.15,
    linear_density: float = 0.0,
    min_mass: float = 1e-9,
) -> np.ndarray:
    """面密度および線密度に応じて分配した物理単位のインバースマス配列を計算する。

    各三角形の面積に面密度 (kg/m2) を掛けた質量を3頂点へ等分配し、
    三角形に属さない1Dエッジ長に線密度 (kg/m) を掛けた質量を2頂点へ等分配する。
    ピンウェイト (1.0=完全固定) との積で逆質量を求める。
    Rust核 (`taremin_cloth_core.compute_combined_inv_masses`) を正本とし、
    利用可能な場合はそちらを優先する。NumPy実装はフォールバック。
    """
    if _core is not None and hasattr(_core, "compute_combined_inv_masses"):
        try:
            pos_c = np.ascontiguousarray(positions, dtype=np.float32)
            faces_c = (
                np.ascontiguousarray(faces, dtype=np.uint32)
                if faces is not None and len(faces) > 0
                else None
            )
            strands_c = (
                np.ascontiguousarray(strand_edges, dtype=np.uint32)
                if strand_edges is not None and len(strand_edges) > 0
                else None
            )
            pin_c = (
                np.ascontiguousarray(pin_weights, dtype=np.float32)
                if pin_weights is not None and len(pin_weights) == len(positions)
                else None
            )
            return np.asarray(
                _core.compute_combined_inv_masses(
                    pos_c,
                    faces_c,
                    strands_c,
                    pin_c,
                    float(areal_density),
                    float(linear_density),
                ),
                dtype=np.float32,
            )
        except Exception:
            pass

    n_verts = len(positions)
    masses = np.zeros(n_verts, dtype=np.float64)
    # 面の質量加算
    if faces is not None and len(faces) > 0 and areal_density > 0.0:
        tris = np.asarray(faces, dtype=np.int64)
        valid = (
            (tris[:, 0] != tris[:, 1])
            & (tris[:, 1] != tris[:, 2])
            & (tris[:, 2] != tris[:, 0])
            & (tris.max() < n_verts)
            & (tris.min() >= 0)
        )
        tris = tris[valid]
        if len(tris) > 0:
            p0 = positions[tris[:, 0]].astype(np.float64)
            p1 = positions[tris[:, 1]].astype(np.float64)
            p2 = positions[tris[:, 2]].astype(np.float64)
            areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
            tri_mass = areas * float(areal_density) / 3.0
            np.add.at(masses, tris[:, 0], tri_mass)
            np.add.at(masses, tris[:, 1], tri_mass)
            np.add.at(masses, tris[:, 2], tri_mass)

    # 1Dストランド（紐）の質量加算
    if strand_edges is not None and len(strand_edges) > 0 and linear_density > 0.0:
        edges = np.asarray(strand_edges, dtype=np.int64)
        valid = (
            (edges[:, 0] != edges[:, 1])
            & (edges.max() < n_verts)
            & (edges.min() >= 0)
        )
        edges = edges[valid]
        if len(edges) > 0:
            pa = positions[edges[:, 0]].astype(np.float64)
            pb = positions[edges[:, 1]].astype(np.float64)
            lens = np.linalg.norm(pa - pb, axis=1)
            edge_mass = lens * float(linear_density) / 2.0
            np.add.at(masses, edges[:, 0], edge_mass)
            np.add.at(masses, edges[:, 1], edge_mass)

    masses = np.maximum(masses, min_mass)
    inv_masses = (1.0 / masses).astype(np.float32)
    if pin_weights is not None:
        w = np.clip(np.asarray(pin_weights, dtype=np.float32), 0.0, 1.0)
        if len(w) == n_verts:
            inv_masses = (inv_masses * (1.0 - w)).astype(np.float32)
    return inv_masses


def compute_areal_inv_masses(
    positions: np.ndarray,
    faces: Optional[np.ndarray],
    *,
    strand_edges: Optional[np.ndarray] = None,
    pin_weights: Optional[np.ndarray] = None,
    areal_density: float = 0.15,
    linear_density: float = 0.0,
    min_mass: float = 1e-9,
) -> np.ndarray:
    """面積に応じて分配した物理単位のインバースマス配列を計算する（後方互換ラッパー）。"""
    return compute_combined_inv_masses(
        positions,
        faces,
        strand_edges=strand_edges,
        pin_weights=pin_weights,
        areal_density=areal_density,
        linear_density=linear_density,
        min_mass=min_mass,
    )


def get_pin_weights(obj, *, settings=None, n_verts: Optional[int] = None) -> np.ndarray:
    """ピン頂点グループのウェイト配列 (0.0〜1.0) を取得する。"""
    if n_verts is None:
        n_verts = len(obj.data.vertices)
    weights = np.zeros(n_verts, dtype=np.float32)
    if not obj or not hasattr(obj, "vertex_groups"):
        return weights
    vg_name = getattr(settings, "pin_vertex_group", "") if settings else ""
    pin_vg = None
    if vg_name:
        pin_vg = obj.vertex_groups.get(vg_name)
    if not pin_vg:
        pin_vg = obj.vertex_groups.get("Pin") or obj.vertex_groups.get("Cloth_Pin")
    if pin_vg:
        for i in range(n_verts):
            try:
                weights[i] = float(pin_vg.weight(i))
            except RuntimeError:
                weights[i] = 0.0
    return weights


def extract_cloth_mesh_data(
    obj,
    *,
    settings=None,
    enable_sewing: Optional[bool] = None,
    apply_world_matrix: bool = True,
) -> ClothMeshData:
    """
    Blenderメッシュオブジェクトから布シミュレータ初期化に必要な全データを一括抽出・分類する。
    
    Args:
        obj: bpy.types.Object (type == 'MESH')
        settings: オブジェクトの taremin_cloth プロパティ（未指定時は obj.taremin_cloth を自動参照）
        enable_sewing: 縫合エッジを分離するか（未指定時は settings.enable_sewing を参照）
        apply_world_matrix: 頂点座標に obj.matrix_world を適用してワールド座標系に変換するか（デフォルト True）
    Returns:
        ClothMeshData: (positions, normal_edges, faces, sewing_edges, inv_masses)
    """
    if settings is None:
        settings = getattr(obj, "taremin_cloth", None)

    mesh = obj.data
    n_verts = len(mesh.vertices)
    coords = np.empty(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    pos_2d = coords.reshape((n_verts, 3))

    # ワールド座標系への変換 (コライダーや重力と空間を一致させる)
    if apply_world_matrix and hasattr(obj, "matrix_world"):
        try:
            mat = np.array(obj.matrix_world, dtype=np.float32)
            if mat.ndim == 2 and mat.shape == (4, 4):
                pos_2d = (pos_2d @ mat[:3, :3].T) + mat[:3, 3]
        except (ValueError, TypeError):
            pass

    mesh.calc_loop_triangles()
    tri_list = [tri.vertices for tri in mesh.loop_triangles]
    faces_2d = np.array(tri_list, dtype=np.uint32) if tri_list else None

    n_edges = len(mesh.edges)
    edge_indices = np.empty(n_edges * 2, dtype=np.uint32)
    mesh.edges.foreach_get("vertices", edge_indices)
    all_edges_2d = edge_indices.reshape((n_edges, 2))

    face_edge_set = set()
    if faces_2d is not None:
        for f in faces_2d:
            face_edge_set.add((min(f[0], f[1]), max(f[0], f[1])))
            face_edge_set.add((min(f[1], f[2]), max(f[1], f[2])))
            face_edge_set.add((min(f[2], f[0]), max(f[2], f[0])))

    # 面積0の縮退三角形の検出とシミュレーション用面からの除外
    if faces_2d is not None and len(faces_2d) > 0:
        degen_report = detect_degenerate_faces(pos_2d, faces_2d, mesh=mesh)
        if degen_report.degenerate_tri_count > 0:
            logger.warning(
                f"Mesh '{getattr(obj, 'name', 'Cloth')}' has {degen_report.degenerate_tri_count} "
                "degenerate triangle(s) with area ~ 0. "
                "Excluding them from simulation faces to prevent numerical instability."
            )
            keep_mask = np.ones(len(faces_2d), dtype=bool)
            keep_mask[degen_report.degenerate_tri_indices] = False
            faces_2d = faces_2d[keep_mask]
            if len(faces_2d) == 0:
                faces_2d = None

    normal_edges = []
    sewing_edges = []

    sewing_enabled = enable_sewing if enable_sewing is not None else getattr(settings, "enable_sewing", False)

    for e in all_edges_2d:
        pair = (min(e[0], e[1]), max(e[0], e[1]))
        if pair in face_edge_set:
            normal_edges.append(e)
        else:
            if sewing_enabled:
                sewing_edges.append(e)
            else:
                normal_edges.append(e)

    edges_2d = np.array(normal_edges, dtype=np.uint32) if normal_edges else np.empty((0, 2), dtype=np.uint32)

    # 縫合エッジ（Loose Edges）から直線エッジとウェイポイント縫合パスを分類
    simple_sewing, waypoint_seams = extract_sewing_topology(sewing_edges, faces_2d=faces_2d, positions=pos_2d)

    # 直線縫合と中継点パスの各セグメントを展開・統合
    all_sewing = list(simple_sewing)
    if waypoint_seams:
        for ws in waypoint_seams:
            all_sewing.append((ws.vert_a, ws.waypoint_indices[0]))
            for k in range(len(ws.waypoint_indices) - 1):
                all_sewing.append((ws.waypoint_indices[k], ws.waypoint_indices[k + 1]))
            all_sewing.append((ws.vert_b, ws.waypoint_indices[-1]))

    sew_2d = np.array(all_sewing, dtype=np.uint32) if all_sewing else None


    inv_masses = get_pin_inv_masses(obj, settings=settings, n_verts=n_verts)
    try:
        density = float(getattr(settings, "areal_density", 0.0) or 0.0)
    except (TypeError, ValueError):
        density = 0.0
    if density > 0.0 and faces_2d is not None and len(faces_2d) > 0:
        pin_weights = get_pin_weights(obj, settings=settings, n_verts=n_verts)
        inv_masses = compute_areal_inv_masses(pos_2d, faces_2d, pin_weights=pin_weights, areal_density=density)

    # 中継点頂点（面を持たないガイディング頂点）は静止ガイドとして inv_mass = 0.0 に固定
    if waypoint_seams:
        for ws in waypoint_seams:
            for wp_idx in ws.waypoint_indices:
                if wp_idx < len(inv_masses):
                    inv_masses[wp_idx] = 0.0

    layer_ids = get_cloth_layer_ids(obj, settings=settings, n_verts=n_verts, faces_2d=faces_2d)
    seam_cluster_map = build_seam_cluster_map(simple_sewing, waypoint_seams) if (simple_sewing or waypoint_seams) else None

    # 紐（Curve）パーツのマージ処理
    curve_parts_runtime = []
    string_parts = getattr(settings, "string_parts", None)
    if string_parts:
        from .curve_extract import extract_curve_polyline, CurvePartRuntimeData

        cloth_thickness = float(getattr(settings, "thickness", 0.005) or 0.005)
        all_thicknesses = [np.full(n_verts, cloth_thickness, dtype=np.float32)]
        all_positions = [pos_2d]
        all_normal_edges = [edges_2d]
        all_inv_masses = [inv_masses]
        all_layer_ids = [layer_ids] if layer_ids is not None else [np.zeros(n_verts, dtype=np.uint32)]

        curr_vert_offset = n_verts
        curr_edge_offset = len(edges_2d)
        additional_sewing_edges = []

        for sp in string_parts:
            if not getattr(sp, "enabled", True):
                continue
            curve_obj = getattr(sp, "curve_object", None)
            if not curve_obj:
                continue

            c_coords, c_edges, c_radii = extract_curve_polyline(
                curve_obj,
                apply_world_matrix=apply_world_matrix,
                default_radius=float(getattr(sp, "radius", 0.003) or 0.003),
            )
            n_curve_verts = len(c_coords)
            if n_curve_verts == 0:
                continue

            # ガイド頂点列とアイレット結合・縫合ギャップ跨ぎの解決
            guide_str = getattr(sp, "guide_vert_indices_str", "")
            guide_cloth_verts = [int(v.strip()) for v in guide_str.split(",") if v.strip().isdigit()]

            seam_bridge_str = getattr(sp, "seam_bridge_pairs_str", "")
            seam_bridges = set()
            if seam_bridge_str:
                for pair_item in seam_bridge_str.split(";"):
                    parts = [int(x.strip()) for x in pair_item.split(",") if x.strip().isdigit()]
                    if len(parts) == 2:
                        seam_bridges.add((min(parts[0], parts[1]), max(parts[0], parts[1])))

            tail_left = int(getattr(sp, "tail_vert_count_left", 8) or 0)
            tail_right = int(getattr(sp, "tail_vert_count_right", 8) or 0)
            stride = max(1, int(getattr(sp, "eyelet_stride", 3) or 3))

            channel_start_v = min(tail_left, n_curve_verts)
            channel_end_v = max(channel_start_v, n_curve_verts - tail_right)
            n_channel_verts = channel_end_v - channel_start_v

            # 分割型紙の縫合ギャップを跨ぐローカル紐エッジを特定し、通常エッジから除外して縫合エッジ化する
            bridge_local_edge_set = set()
            if seam_bridges and len(guide_cloth_verts) >= 2 and n_channel_verts >= 2:
                n_guides = len(guide_cloth_verts)
                for g_idx in range(n_guides - 1):
                    u_c = guide_cloth_verts[g_idx]
                    v_c = guide_cloth_verts[g_idx + 1]
                    pair_key = (min(u_c, v_c), max(u_c, v_c))
                    if pair_key in seam_bridges:
                        t0 = float(g_idx) / max(1, n_guides - 1)
                        t1 = float(g_idx + 1) / max(1, n_guides - 1)
                        v_rel0 = int(round(channel_start_v + t0 * (n_channel_verts - 1)))
                        v_rel1 = int(round(channel_start_v + t1 * (n_channel_verts - 1)))
                        for e_local in range(min(v_rel0, v_rel1), min(max(v_rel0, v_rel1), len(c_edges))):
                            bridge_local_edge_set.add(e_local)

            valid_local_edges_list = []
            for e_local_idx in range(len(c_edges)):
                v0 = int(c_edges[e_local_idx, 0])
                v1 = int(c_edges[e_local_idx, 1])
                if e_local_idx in bridge_local_edge_set:
                    additional_sewing_edges.append((curr_vert_offset + v0, curr_vert_offset + v1))
                else:
                    valid_local_edges_list.append([v0, v1])

            if valid_local_edges_list:
                c_edges_local = np.array(valid_local_edges_list, dtype=np.uint32)
            else:
                c_edges_local = np.empty((0, 2), dtype=np.uint32)
            c_edges_merged = c_edges_local + curr_vert_offset

            # 紐のインバースマス計算 (1D線密度: 縫合ギャップエッジを除いた実エッジ長のみから算出)
            lin_density = float(getattr(sp, "linear_density", 0.015) or 0.015)
            c_inv_masses = compute_combined_inv_masses(
                c_coords,
                faces=None,
                strand_edges=c_edges_local,
                linear_density=lin_density,
            )

            left_tail_e = []
            channel_e = []
            right_tail_e = []
            seam_bridge_e = []

            for i in range(len(c_edges_local)):
                abs_e_idx = curr_edge_offset + i
                v0 = int(c_edges_local[i, 0])
                v1 = int(c_edges_local[i, 1])
                if v1 <= channel_start_v:
                    left_tail_e.append(abs_e_idx)
                elif v0 >= channel_end_v - 1:
                    right_tail_e.append(abs_e_idx)
                else:
                    channel_e.append(abs_e_idx)

            def _calc_edge_length(e_indices: list[int]) -> float:
                l_sum = 0.0
                for e_idx in e_indices:
                    rel_idx = e_idx - curr_edge_offset
                    v0 = c_edges_local[rel_idx, 0]
                    v1 = c_edges_local[rel_idx, 1]
                    p0 = c_coords[v0]
                    p1 = c_coords[v1]
                    l_sum += float(np.linalg.norm(p1 - p0))
                return l_sum

            init_left_len = _calc_edge_length(left_tail_e)
            init_channel_len = _calc_edge_length(channel_e)
            init_right_len = _calc_edge_length(right_tail_e)

            eyelet_pairs = []
            if guide_cloth_verts and n_channel_verts > 0:
                n_guides = len(guide_cloth_verts)
                sampled_guide_indices = list(range(0, n_guides, stride))
                if (n_guides - 1) not in sampled_guide_indices:
                    sampled_guide_indices.append(n_guides - 1)
                # 縫合ギャップの両端もアイレット結合に含めてフード縁との一体性を確保
                for g_idx in range(n_guides - 1):
                    u_c = guide_cloth_verts[g_idx]
                    v_c = guide_cloth_verts[g_idx + 1]
                    if (min(u_c, v_c), max(u_c, v_c)) in seam_bridges:
                        if g_idx not in sampled_guide_indices:
                            sampled_guide_indices.append(g_idx)
                        if (g_idx + 1) not in sampled_guide_indices:
                            sampled_guide_indices.append(g_idx + 1)
                sampled_guide_indices.sort()

                for g_idx in sampled_guide_indices:
                    cloth_v = guide_cloth_verts[g_idx]
                    t = float(g_idx) / max(1, n_guides - 1)
                    curve_v_rel = int(round(channel_start_v + t * max(0, n_channel_verts - 1)))
                    curve_v_abs = curr_vert_offset + min(curve_v_rel, n_curve_verts - 1)
                    eyelet_pairs.append((cloth_v, curve_v_abs))
                    additional_sewing_edges.append((cloth_v, curve_v_abs))

            runtime_data = CurvePartRuntimeData(
                curve_obj_name=getattr(curve_obj, "name", "String"),
                vertex_offset=curr_vert_offset,
                vertex_count=n_curve_verts,
                edge_offset=curr_edge_offset,
                left_tail_edge_indices=left_tail_e,
                channel_edge_indices=channel_e,
                right_tail_edge_indices=right_tail_e,
                seam_bridge_edge_indices=seam_bridge_e,
                initial_left_len=init_left_len,
                initial_channel_len=init_channel_len,
                initial_right_len=init_right_len,
                guide_cloth_verts=guide_cloth_verts,
                eyelet_pairs=eyelet_pairs,
            )
            curve_parts_runtime.append(runtime_data)

            all_positions.append(c_coords)
            all_normal_edges.append(c_edges_merged)
            all_thicknesses.append(c_radii)
            all_inv_masses.append(c_inv_masses)
            c_layer_id = getattr(settings, "layer_id", 0) or 0
            all_layer_ids.append(np.full(n_curve_verts, c_layer_id, dtype=np.uint32))

            curr_vert_offset += n_curve_verts
            curr_edge_offset += len(c_edges_local)

        if curve_parts_runtime:
            pos_2d = np.vstack(all_positions)
            edges_2d = np.vstack(all_normal_edges) if all_normal_edges else np.empty((0, 2), dtype=np.uint32)
            inv_masses = np.concatenate(all_inv_masses)
            layer_ids = np.concatenate(all_layer_ids)
            thicknesses = np.concatenate(all_thicknesses)

            if additional_sewing_edges:
                if sew_2d is not None and len(sew_2d) > 0:
                    sew_2d = np.vstack([sew_2d, np.array(additional_sewing_edges, dtype=np.uint32)])
                else:
                    sew_2d = np.array(additional_sewing_edges, dtype=np.uint32)
        else:
            cloth_thickness = float(getattr(settings, "thickness", 0.005) or 0.005)
            thicknesses = np.full(n_verts, cloth_thickness, dtype=np.float32)
    else:
        cloth_thickness = float(getattr(settings, "thickness", 0.005) or 0.005) if settings else 0.005
        thicknesses = np.full(n_verts, cloth_thickness, dtype=np.float32)

    return ClothMeshData(
        positions=pos_2d,
        normal_edges=edges_2d,
        faces=faces_2d,
        sewing_edges=sew_2d,
        inv_masses=inv_masses,
        layer_ids=layer_ids,
        waypoint_seams=waypoint_seams if waypoint_seams else None,
        seam_cluster_map=seam_cluster_map,
        thicknesses=thicknesses,
        curve_parts=curve_parts_runtime if curve_parts_runtime else None,
    )


def build_seam_cluster_map(
    simple_sewing_edges: Optional[list[tuple[int, int]]],
    waypoint_seams: Optional[list[WaypointSeam]],
) -> dict[int, list[int]]:
    """縫合エッジおよび中継点縫合パスから、各頂点に対応する縫合パートナー頂点群のマップを構築する。

    戻り値:
        dict[int, list[int]]: 頂点インデックス -> その頂点と同一縫合線上にある他の頂点インデックスのリスト
    """
    clusters = []
    if simple_sewing_edges:
        for v0, v1 in simple_sewing_edges:
            clusters.append({int(v0), int(v1)})

    if waypoint_seams:
        for ws in waypoint_seams:
            cluster = {int(ws.vert_a), int(ws.vert_b)}
            for wp in ws.waypoint_indices:
                cluster.add(int(wp))
            clusters.append(cluster)

    partner_map: dict[int, list[int]] = {}
    for cluster in clusters:
        for v in cluster:
            partners = [p for p in cluster if p != v]
            if v in partner_map:
                existing = set(partner_map[v])
                for p in partners:
                    if p not in existing:
                        partner_map[v].append(p)
            else:
                partner_map[v] = partners

    return partner_map


def extract_sewing_topology(
    loose_edges: list,
    *,
    faces_2d: Optional[np.ndarray],
    positions: np.ndarray,
) -> tuple[list[tuple[int, int]], list[WaypointSeam]]:
    """面を持たないエッジ（Loose Edges）から直線縫合エッジとウェイポイント付き縫合パスを分類・抽出する。

    Args:
        loose_edges: 面を持たないエッジ [(u, v), ...] のリスト
        faces_2d: 三角形インデックス配列 [F, 3] または None
        positions: 頂点座標配列 [N, 3]

    Returns:
        simple_sewing_edges: 単純直線縫合エッジ [(v0, v1), ...]
        waypoint_seams: 中継点を持つ縫合パス [WaypointSeam, ...]
    """
    if not loose_edges:
        return [], []

    face_vertex_set = set()
    if faces_2d is not None and len(faces_2d) > 0:
        for f in faces_2d:
            face_vertex_set.add(int(f[0]))
            face_vertex_set.add(int(f[1]))
            face_vertex_set.add(int(f[2]))

    topo_report = validate_sewing_topology(loose_edges, face_vertex_set)
    if topo_report.has_issues:
        for issue in topo_report.branching_issues:
            logger.warning(
                f"Invalid sewing topology ({issue.issue_type}): {issue.message}. "
                "Excluding invalid seam component from simulation to guarantee single-path seams."
            )

    from collections import defaultdict
    adj = defaultdict(set)
    for e in loose_edges:
        u, v = int(e[0]), int(e[1])
        if u != v:
            adj[u].add(v)
            adj[v].add(u)

    visited = set()
    simple_edges = []
    waypoint_seams = []

    for v in list(adj.keys()):
        if v in visited:
            continue
        # 連結成分を探索 (BFS)
        comp = []
        queue = [v]
        visited.add(v)
        while queue:
            curr = queue.pop(0)
            comp.append(curr)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # 連結成分内の面頂点（布端点）と中継点を分類
        endpoints = [x for x in comp if x in face_vertex_set]
        internal = [x for x in comp if x not in face_vertex_set]

        # ケース 1: 単一エッジで両端が面頂点（通常の直接縫合）
        if len(comp) == 2 and len(endpoints) == 2 and len(internal) == 0:
            simple_edges.append((endpoints[0], endpoints[1]))
            continue

        # ケース 2: 両端が面頂点、中間が面を持たない頂点の単純パス（A - W1 - ... - Wk - B）
        if len(endpoints) == 2 and len(internal) >= 1:
            # 始点と終点の次数が1、中間の全頂点の次数が2であることを検証
            deg_ok = (len(adj[endpoints[0]]) == 1 and
                      len(adj[endpoints[1]]) == 1 and
                      all(len(adj[m]) == 2 for m in internal))
            if deg_ok:
                # endpoints[0] から順序付けられたパスを辿る
                start = endpoints[0]
                goal = endpoints[1]
                path = [start]
                prev = None
                curr = start
                while curr != goal:
                    nxt_candidates = [n for n in adj[curr] if n != prev]
                    if not nxt_candidates:
                        break
                    nxt = nxt_candidates[0]
                    path.append(nxt)
                    prev = curr
                    curr = nxt

                if path[-1] == goal and len(path) == len(comp) and len(path) >= 3:
                    wp_indices = path[1:-1]
                    wp_coords = positions[wp_indices].astype(np.float32)
                    waypoint_seams.append(WaypointSeam(
                        vert_a=start,
                        vert_b=goal,
                        waypoint_indices=wp_indices,
                        waypoint_coords=wp_coords,
                    ))
                    continue

        # ケース 3: 不正トポロジー（枝分かれ、閉路、宙ぶらりん、複数端点等）
        # 単一経路の保証を満たさないため、シミュレーション対象から安全に除外
        logger.warning(
            f"Excluding invalid seam component with vertices {comp} (endpoints: {endpoints}) "
            "from simulation because it does not form a valid single-path seam."
        )

    return simple_edges, waypoint_seams


def get_cloth_layer_ids(
    obj,
    *,
    settings=None,
    n_verts: Optional[int] = None,
    faces_2d: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """メッシュ属性 (cloth_layer) から頂点単位のレイヤーID配列 (uint32) を抽出する。

    Blender 3.0+ の Mesh Attribute ('cloth_layer') を参照し、
    Face / Point / Corner 各ドメインから頂点単位のレイヤーIDを算出する。
    複数の面で共有される頂点には、共有面の中で最大 (Max) のレイヤーIDが割り当てられる
    (例: レイヤードスカートの接合部が上位レイヤーと整合する)。

    Args:
        obj: bpy.types.Object
        settings: taremin_cloth プロパティ
        n_verts: 頂点数
        faces_2d: 三角形インデックス配列 [F, 3] (任意)

    Returns:
        np.ndarray [N] (uint32) または属性が存在しない場合は None
    """
    if not obj or not hasattr(obj, "data") or not hasattr(obj.data, "attributes"):
        return None

    mesh = obj.data
    attr = mesh.attributes.get("cloth_layer")
    if attr is None:
        return None

    if n_verts is None:
        n_verts = len(mesh.vertices)
    if n_verts == 0:
        return np.empty(0, dtype=np.uint32)

    default_layer = int(getattr(settings, "layer_id", 0)) if settings else 0
    default_layer = max(0, default_layer)

    domain = getattr(attr, "domain", "FACE")

    if domain == "POINT":
        raw = np.empty(n_verts, dtype=np.int32)
        try:
            attr.data.foreach_get("value", raw)
            return np.maximum(raw, 0).astype(np.uint32)
        except Exception:
            return None

    elif domain == "FACE":
        n_polys = len(mesh.polygons)
        if n_polys == 0:
            return None
        poly_layers = np.empty(n_polys, dtype=np.int32)
        try:
            attr.data.foreach_get("value", poly_layers)
        except Exception:
            return None
        poly_layers = np.maximum(poly_layers, 0).astype(np.uint32)

        layer_ids = np.full(n_verts, default_layer, dtype=np.uint32)

        try:
            if hasattr(mesh, "calc_loop_triangles") and hasattr(mesh, "loop_triangles"):
                mesh.calc_loop_triangles()
                n_tris = len(mesh.loop_triangles)
                if n_tris > 0:
                    tri_poly_indices = np.empty(n_tris, dtype=np.int32)
                    mesh.loop_triangles.foreach_get("polygon_index", tri_poly_indices)

                    if faces_2d is not None and len(faces_2d) == n_tris:
                        tris = faces_2d
                    else:
                        tri_verts = np.empty(n_tris * 3, dtype=np.int32)
                        mesh.loop_triangles.foreach_get("vertices", tri_verts)
                        tris = tri_verts.reshape((n_tris, 3))

                    valid_mask = (tri_poly_indices >= 0) & (tri_poly_indices < n_polys)
                    if np.all(valid_mask):
                        tri_layers = poly_layers[tri_poly_indices]
                        np.maximum.at(layer_ids, tris[:, 0], tri_layers)
                        np.maximum.at(layer_ids, tris[:, 1], tri_layers)
                        np.maximum.at(layer_ids, tris[:, 2], tri_layers)
                        return layer_ids
        except Exception:
            pass

        # ポリゴン直接走査のフォールバック
        for p_idx, poly in enumerate(mesh.polygons):
            l_val = poly_layers[p_idx]
            for v in poly.vertices:
                if 0 <= v < n_verts:
                    layer_ids[v] = max(layer_ids[v], l_val)

        return layer_ids

    elif domain == "CORNER":
        n_loops = len(mesh.loops)
        if n_loops == 0:
            return None
        loop_layers = np.empty(n_loops, dtype=np.int32)
        loop_verts = np.empty(n_loops, dtype=np.int32)
        try:
            attr.data.foreach_get("value", loop_layers)
            mesh.loops.foreach_get("vertex_index", loop_verts)
        except Exception:
            return None
        loop_layers = np.maximum(loop_layers, 0).astype(np.uint32)
        layer_ids = np.full(n_verts, default_layer, dtype=np.uint32)
        np.maximum.at(layer_ids, loop_verts, loop_layers)
        return layer_ids

    return None


def get_mesh_face_layers_summary(mesh) -> Optional[dict]:
    """メッシュに設定された cloth_layer 属性のサマリー情報を取得する。"""
    if not mesh or not hasattr(mesh, "attributes"):
        return None
    attr = mesh.attributes.get("cloth_layer")
    if attr is None:
        return None
    domain = getattr(attr, "domain", "FACE")
    count = len(attr.data)
    if count == 0:
        return None
    raw = np.empty(count, dtype=np.int32)
    try:
        attr.data.foreach_get("value", raw)
        unique_layers = sorted(list(set(raw.tolist())))
        return {
            "domain": domain,
            "unique_layers": unique_layers,
            "count": count,
        }
    except Exception:
        return None


def extract_mesh_data(obj):
    """
    互換用関数: extract_cloth_mesh_data のタプル形式 (positions, edges, faces, sewing_edges) を返却する
    """
    data = extract_cloth_mesh_data(obj)
    return data.positions, data.normal_edges, data.faces, data.sewing_edges
