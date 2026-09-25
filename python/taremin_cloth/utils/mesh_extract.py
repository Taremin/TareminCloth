"""
メッシュデータ高速抽出ユーティリティ (taremin_cloth.utils.mesh_extract)
Blenderの bpy.types.Mesh から NumPy 配列へのゼロコピー/一括データ転送および
布シミュレーション初期化用データの抽出・分類を提供する。
"""

from typing import NamedTuple, Optional, Tuple
import numpy as np


class ClothMeshData(NamedTuple):
    """布シミュレーション初期化に必要なメッシュデータコンテナ"""
    positions: np.ndarray          # shape [N, 3], float32
    normal_edges: np.ndarray       # shape [E, 2], uint32
    faces: Optional[np.ndarray]    # shape [F, 3], uint32 (存在しない場合は None)
    sewing_edges: Optional[np.ndarray]  # shape [S, 2], uint32 (存在しない場合は None)
    inv_masses: np.ndarray         # shape [N], float32


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


def get_pin_inv_masses(obj, settings=None, n_verts: Optional[int] = None) -> np.ndarray:
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


def compute_areal_inv_masses(
    positions: np.ndarray,
    faces: Optional[np.ndarray],
    pin_weights: Optional[np.ndarray] = None,
    areal_density: float = 0.15,
    min_mass: float = 1e-9,
) -> np.ndarray:
    """面積に応じて分配した物理単位のインバースマス配列を計算する。

    各三角形の面積に面密度 (kg/m2) を掛けた質量を3頂点へ等分配し、
    ピンウェイト (1.0=完全固定) との積で逆質量を求める。
    面を持たない孤立頂点は微小質量として扱い、ゼロ除算を防ぐ。
    """
    n_verts = len(positions)
    masses = np.zeros(n_verts, dtype=np.float64)
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
    masses = np.maximum(masses, min_mass)
    inv_masses = (1.0 / masses).astype(np.float32)
    if pin_weights is not None:
        w = np.clip(np.asarray(pin_weights, dtype=np.float32), 0.0, 1.0)
        if len(w) == n_verts:
            inv_masses = (inv_masses * (1.0 - w)).astype(np.float32)
    return inv_masses


def get_pin_weights(obj, settings=None, n_verts: Optional[int] = None) -> np.ndarray:
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


def extract_cloth_mesh_data(obj, settings=None, enable_sewing: Optional[bool] = None) -> ClothMeshData:
    """
    Blenderメッシュオブジェクトから布シミュレータ初期化に必要な全データを一括抽出・分類する。
    
    Args:
        obj: bpy.types.Object (type == 'MESH')
        settings: オブジェクトの taremin_cloth プロパティ（未指定時は obj.taremin_cloth を自動参照）
        enable_sewing: 縫合エッジを分離するか（未指定時は settings.enable_sewing を参照）
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
    sew_2d = np.array(sewing_edges, dtype=np.uint32) if sewing_edges else None

    inv_masses = get_pin_inv_masses(obj, settings, n_verts)
    try:
        density = float(getattr(settings, "areal_density", 0.0) or 0.0)
    except (TypeError, ValueError):
        density = 0.0
    if density > 0.0 and faces_2d is not None and len(faces_2d) > 0:
        pin_weights = get_pin_weights(obj, settings, n_verts)
        inv_masses = compute_areal_inv_masses(pos_2d, faces_2d, pin_weights, density)

    return ClothMeshData(
        positions=pos_2d,
        normal_edges=edges_2d,
        faces=faces_2d,
        sewing_edges=sew_2d,
        inv_masses=inv_masses,
    )


def extract_mesh_data(obj):
    """
    互換用関数: extract_cloth_mesh_data のタプル形式 (positions, edges, faces, sewing_edges) を返却する
    """
    data = extract_cloth_mesh_data(obj)
    return data.positions, data.normal_edges, data.faces, data.sewing_edges
