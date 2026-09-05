"""
taremin_cloth ボーン局所SDFベーカー (SDF Baker)
素体メッシュ（Skinned Mesh）およびアーマチュア情報から、
各ボーンのローカル符号付き距離場 (SDF) とボーンウェイト (Alpha) を
3D テクスチャアトラス (Rg16Float) として事前ベイクする。
"""

import os
import time
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional
from ..utils.logger import logger

try:
    import bpy
except ImportError:
    bpy = None

try:
    import mathutils
    from mathutils.bvhtree import BVHTree
    HAS_MATHUTILS_BVH = True
except ImportError:
    HAS_MATHUTILS_BVH = False


class BoneSdfBakeResult:
    """ボーンSDFベイク結果のコンテナ"""
    def __init__(
        self,
        texture_bytes: bytes,
        width: int,
        height: int,
        depth: int,
        bone_infos: np.ndarray,   # shape: [N, 20]
        bone_names: List[str],
        bind_matrices: np.ndarray,  # shape: [N, 4, 4]
        joint_face_indices: Optional[np.ndarray] = None,  # shape: [M] (ハイブリッド関節面インデックス)
        joint_faces_by_pair: Optional[Dict[Tuple[str, str], np.ndarray]] = None,  # {(parent, child): indices}
    ):
        self.texture_bytes = texture_bytes
        self.width = width
        self.height = height
        self.depth = depth
        self.bone_infos = bone_infos
        self.bone_names = bone_names
        self.bind_matrices = bind_matrices
        self.joint_face_indices = joint_face_indices if joint_face_indices is not None else np.empty(0, dtype=np.int32)
        self.joint_faces_by_pair = joint_faces_by_pair if joint_faces_by_pair is not None else {}


def extract_joint_mesh_from_sdf(
    mesh_verts: np.ndarray,
    mesh_tris: np.ndarray,
    atlas_3d: np.ndarray,
    bone_infos: np.ndarray,
    bind_matrices: np.ndarray,
    gap_threshold: float = 0.005,
    mode: str = "any1",
) -> np.ndarray:
    """
    素体メッシュ頂点位置でSDFテクスチャアトラスの実効距離を評価し、
    SDFがメッシュ表面を再現できていない領域（欠損部 + のりしろ）の三角形インデックスを抽出する。
    
    原理:
    - 剛体領域: SDF実効距離 eff_dist ≈ 0 (SDF単体で衝突判定可能)
    - 関節・境界領域: スキンウェイトのブレンドやカットオフにより eff_dist > gap_threshold
    - mode="any1" により、欠損頂点を含む三角形を抽出することで、完全剛体部との境界に
      自然で必要十分な1段のオーバーラップ（のりしろ）が自動形成される。
    
    戻り値: shape [M] (抽出された三角形の行インデックス配列, np.int32)
    """
    if len(mesh_verts) == 0 or len(mesh_tris) == 0 or len(bone_infos) == 0:
        return np.empty(0, dtype=np.int32)

    n_verts = len(mesh_verts)
    n_bones = len(bone_infos)
    D, H, W, _ = atlas_3d.shape

    eff_dist = np.full(n_verts, 10.0, dtype=np.float32)

    for b_idx in range(n_bones):
        b_info = bone_infos[b_idx]
        mat_inv = bind_matrices[b_idx]

        local_min = b_info[0:3]
        local_max = b_info[4:7]
        uvw_scale = b_info[8:11]
        uvw_offset = b_info[12:15]
        weight_th = max(b_info[19], 0.01)

        # 頂点をボーンローカル座標系へ射影
        ones = np.ones((n_verts, 1), dtype=np.float32)
        p_homo = np.hstack([mesh_verts, ones])
        p_local = (p_homo @ mat_inv.T)[:, :3]

        margin = 0.02
        in_aabb = np.all((p_local >= local_min - margin) & (p_local <= local_max + margin), axis=1)
        if not np.any(in_aabb):
            continue

        p_in = p_local[in_aabb]
        uvw = p_in * uvw_scale + uvw_offset

        uvw_min = local_min * uvw_scale + uvw_offset
        uvw_max = local_max * uvw_scale + uvw_offset

        in_tile = np.all((uvw >= uvw_min) & (uvw <= uvw_max), axis=1)
        if not np.any(in_tile):
            continue

        p_indices = np.where(in_aabb)[0][in_tile]
        uvw_valid = uvw[in_tile]

        eps_slice = (uvw_max - uvw_min) * 0.015
        uvw_safe = np.clip(uvw_valid, uvw_min + eps_slice, uvw_max - eps_slice)

        # ボクセルインデックスサンプリング
        gx = np.clip(np.round((uvw_safe[:, 0] * W) - 0.5).astype(int), 0, W - 1)
        gy = np.clip(np.round((uvw_safe[:, 1] * H) - 0.5).astype(int), 0, H - 1)
        gz = np.clip(np.round((uvw_safe[:, 2] * D) - 0.5).astype(int), 0, D - 1)

        samples = atlas_3d[gz, gy, gx]
        raw_dist = samples[:, 0].astype(np.float32)
        alpha = samples[:, 1].astype(np.float32)

        # WGSLシェーダー (collision.wgsl) と同一のマスク & 実効距離計算
        t = np.clip(alpha / weight_th, 0.0, 1.0)
        mask = t * t * (3.0 - 2.0 * t)
        cur_eff_dist = (1.0 - mask) * 10.0 + mask * raw_dist

        is_closer = cur_eff_dist < eff_dist[p_indices]
        update_idx = p_indices[is_closer]
        eff_dist[update_idx] = cur_eff_dist[is_closer]

    # 欠損頂点の検出 (SDF実効距離が表面から gap_threshold 以上逃げている領域)
    deficient_verts = eff_dist > gap_threshold

    # 三角形判定
    if mode == "any1":
        tri_mask = deficient_verts[mesh_tris[:, 0]] | deficient_verts[mesh_tris[:, 1]] | deficient_verts[mesh_tris[:, 2]]
    elif mode == "any2":
        cnt = (
            deficient_verts[mesh_tris[:, 0]].astype(int)
            + deficient_verts[mesh_tris[:, 1]].astype(int)
            + deficient_verts[mesh_tris[:, 2]].astype(int)
        )
        tri_mask = cnt >= 2
    else:  # all3
        tri_mask = deficient_verts[mesh_tris[:, 0]] & deficient_verts[mesh_tris[:, 1]] & deficient_verts[mesh_tris[:, 2]]

    return np.where(tri_mask)[0].astype(np.int32)


def extract_joint_mesh_indices(
    mesh_verts: np.ndarray,
    mesh_tris: np.ndarray,
    bone_weights: Dict[str, np.ndarray],
    threshold: float = 0.85,
    overlap_rings: int = 0,
    min_blend_weight: float = 0.02,
) -> np.ndarray:
    """
    複数ボーンブレンド領域（関節部）に含まれる三角形のインデックス配列をトポロジーウェイトから抽出する（フォールバック用）。
    ウェイト分布から第2ウェイト（min_blend_weight）以上のブレンド頂点を検出し、
    面判定(any1)により剛体側への1段のりしろを持った三角形群を抽出する。
    戻り値: shape [M] (元の mesh_tris の行インデックス配列)
    """
    if not bone_weights or len(mesh_tris) == 0:
        return np.empty(0, dtype=np.int32)

    n_verts = len(mesh_verts)

    # 全ボーンウェイト行列を構築 [V, B]
    weight_cols = [w_arr for w_arr in bone_weights.values() if np.max(w_arr) > min_blend_weight]
    if len(weight_cols) < 2:
        return np.empty(0, dtype=np.int32)

    w_matrix = np.column_stack(weight_cols)

    # 各頂点の上位2つのウェイトを取得 (partition により O(B) で高速)
    part = np.partition(w_matrix, -2, axis=1)[:, -2:]
    w_second = np.min(part, axis=1)  # 2番目に大きいウェイト

    # 1. 第2ウェイトが min_blend_weight を超える頂点をブレンド頂点として抽出
    blend_vert_indices = np.where(w_second > min_blend_weight)[0]
    if len(blend_vert_indices) == 0:
        return np.empty(0, dtype=np.int32)

    joint_verts = set(blend_vert_indices)

    # 2. 追加の剛体側へののりしろ（トポロジーリング拡張: 必要な場合のみ）
    if overlap_rings > 0:
        adj = [set() for _ in range(n_verts)]
        for t in mesh_tris:
            adj[t[0]].add(t[1]); adj[t[0]].add(t[2])
            adj[t[1]].add(t[0]); adj[t[1]].add(t[2])
            adj[t[2]].add(t[0]); adj[t[2]].add(t[1])

        current_ring = set(joint_verts)
        for _ in range(overlap_rings):
            next_ring = set()
            for v in current_ring:
                for neighbor in adj[v]:
                    if neighbor not in joint_verts:
                        next_ring.add(neighbor)
            joint_verts.update(next_ring)
            current_ring = next_ring

    # 3. 関節頂点を1つでも含む三角形を抽出 (境界に自然な1リングのりしろが付与される)
    vert_mask = np.zeros(n_verts, dtype=bool)
    vert_mask[list(joint_verts)] = True
    tri_has_joint = vert_mask[mesh_tris[:, 0]] | vert_mask[mesh_tris[:, 1]] | vert_mask[mesh_tris[:, 2]]
    joint_indices = np.where(tri_has_joint)[0].astype(np.int32)
    return joint_indices


def is_major_joint_pair(b1: str, b2: str) -> bool:
    """ボーン名から主要関節（股関節/鼠径部、膝、肘、肩、腰/背骨）のペアであるかを判定する"""
    b1_l, b2_l = b1.lower(), b2.lower()
    # 左右の一致チェック (左右が異なるボーン同士はペアとみなさない)
    if (".l" in b1_l or "_l" in b1_l) and (".r" in b2_l or "_r" in b2_l):
        return False
    if (".r" in b1_l or "_r" in b1_l) and (".l" in b2_l or "_l" in b2_l):
        return False
    # 股関節 / 鼠径部 (hips/pelvis と upper_leg/leg/thigh)
    if ("hip" in b1_l or "pelvis" in b1_l) and ("leg" in b2_l or "thigh" in b2_l):
        return True
    if ("hip" in b2_l or "pelvis" in b2_l) and ("leg" in b1_l or "thigh" in b1_l):
        return True
    # 膝 (upper_leg/thigh と lower_leg/calf/shin/knee)
    if ("upper_leg" in b1_l or "thigh" in b1_l) and ("lower_leg" in b2_l or "calf" in b2_l or "shin" in b2_l or "knee" in b2_l):
        return True
    if ("upper_leg" in b2_l or "thigh" in b2_l) and ("lower_leg" in b1_l or "calf" in b1_l or "shin" in b1_l or "knee" in b1_l):
        return True
    # 体幹・背骨の主要屈曲関節 (hips/pelvis と spine, spine と chest)
    if ("hip" in b1_l or "pelvis" in b1_l) and "spine" in b2_l:
        return True
    if ("hip" in b2_l or "pelvis" in b2_l) and "spine" in b1_l:
        return True
    if "spine" in b1_l and "chest" in b2_l:
        return True
    if "spine" in b2_l and "chest" in b1_l:
        return True
    return False

# 互換用エイリアス
is_limb_joint_pair = is_major_joint_pair


def extract_major_joint_mesh_indices(
    mesh_verts: np.ndarray,
    mesh_tris: np.ndarray,
    bone_weights: Dict[str, np.ndarray],
    min_blend_weight: float = 0.06,
) -> np.ndarray:
    """
    主要関節（股関節/鼠径部、膝、および腰/背骨境界）の回転ブレンド領域に含まれる三角形インデックスを抽出する。
    局所的な回転関節境界のみをメッシュ化することで、全身に広がる過剰な面数を抑え（約10〜12%）、
    衣服の巻き上がりを防ぎつつ、股関節の凹みおよび前屈時の背骨の出っ張りをメッシュコライダーで完全に解消する。
    """
    if not bone_weights or len(mesh_tris) == 0:
        return np.empty(0, dtype=np.int32)
    n_verts = len(mesh_verts)
    b_names = list(bone_weights.keys())
    pairs = []
    for i in range(len(b_names)):
        for j in range(i + 1, len(b_names)):
            if is_major_joint_pair(b_names[i], b_names[j]):
                pairs.append((b_names[i], b_names[j]))
    if not pairs:
        return extract_joint_mesh_indices(mesh_verts, mesh_tris, bone_weights, min_blend_weight=min_blend_weight)

    joint_verts = np.zeros(n_verts, dtype=bool)
    for b1, b2 in pairs:
        w1, w2 = bone_weights[b1], bone_weights[b2]
        joint_verts |= (w1 > min_blend_weight) & (w2 > min_blend_weight)

    tri_mask = joint_verts[mesh_tris[:, 0]] | joint_verts[mesh_tris[:, 1]] | joint_verts[mesh_tris[:, 2]]
    return np.where(tri_mask)[0].astype(np.int32)

# 互換用エイリアス
extract_limb_joint_mesh_indices = extract_major_joint_mesh_indices


def extract_hierarchy_joint_mesh_indices(
    mesh_verts: np.ndarray,
    mesh_tris: np.ndarray,
    bone_weights: Dict[str, np.ndarray],
    bone_parent_map: Dict[str, Optional[str]],
    min_blend_weight: float = 0.05,
    min_verts_per_joint: int = 4,
) -> Tuple[np.ndarray, Dict[Tuple[str, str], np.ndarray]]:
    """
    アーマチュアの親子階層（Parent-Child）に基づいて、関節部の三角形インデックスを自動抽出する。
    ボーン名（文字列）に一切依存せず、親子関係にあるボーンペア間でスキンウェイトがブレンドされている面を特定する。
    
    戻り値:
        (all_joint_face_indices, joint_faces_by_pair)
        - all_joint_face_indices: 全関節面のユニオン配列 (shape: [M], dtype: int32)
        - joint_faces_by_pair: 各ペア (parent, child) をキーとする面インデックス辞書
    """
    if not bone_weights or len(mesh_tris) == 0 or not bone_parent_map:
        return np.empty(0, dtype=np.int32), {}

    n_verts = len(mesh_verts)
    joint_faces_by_pair: Dict[Tuple[str, str], np.ndarray] = {}

    for child, parent in bone_parent_map.items():
        if not parent:
            continue
        if child not in bone_weights or parent not in bone_weights:
            continue

        w_child = bone_weights[child]
        w_parent = bone_weights[parent]

        # 両方のボーンから min_blend_weight 以上のウェイトを受けている頂点
        blend_mask = (w_child > min_blend_weight) & (w_parent > min_blend_weight)
        n_blend = int(np.count_nonzero(blend_mask))

        # 指先などの極小ブレンド領域はノイズとしてスキップ
        if n_blend < min_verts_per_joint:
            continue

        # any1: ブレンド頂点を1つでも含む三角形を抽出
        tri_mask = blend_mask[mesh_tris[:, 0]] | blend_mask[mesh_tris[:, 1]] | blend_mask[mesh_tris[:, 2]]
        pair_faces = np.where(tri_mask)[0].astype(np.int32)

        if len(pair_faces) > 0:
            joint_faces_by_pair[(parent, child)] = pair_faces

    if not joint_faces_by_pair:
        return np.empty(0, dtype=np.int32), {}

    all_faces = np.unique(np.concatenate(list(joint_faces_by_pair.values())))
    return all_faces, joint_faces_by_pair


def compute_3d_atlas_layout(n_bones: int, resolution: int, max_dim: int = 2048) -> Tuple[int, int, int, int, int, int, int]:
    """
    ボーン数と解像度から、3D Brick Atlas のタイル分割数とテクスチャ寸法を計算する。
    GPU上限(max_dim=2048)を超える場合は安全な最大解像度に自動クランプする。
    
    戻り値: (n_cols, n_rows, n_layers, safe_res, total_width, total_height, total_depth)
    """
    import math
    if n_bones <= 0:
        return 1, 1, 1, resolution, resolution, resolution, resolution
    
    n_cols = math.ceil(n_bones ** (1/3))
    n_rows = math.ceil(math.sqrt(math.ceil(n_bones / n_cols)))
    n_layers = math.ceil(n_bones / (n_cols * n_rows))
    
    max_tile = max(n_cols, n_rows, n_layers)
    safe_res = resolution
    if max_tile * resolution > max_dim:
        safe_res = max(16, max_dim // max_tile)
        logger.warning(
            f"[SDF Baker] 警告: ボーン数({n_bones})に対する解像度({resolution})がGPU上限({max_dim}px)を超過します。"
            f"安全のため解像度を {safe_res} に自動調整しました。(タイル: {n_cols}x{n_rows}x{n_layers})"
        )
    
    total_width = n_cols * safe_res
    total_height = n_rows * safe_res
    total_depth = n_layers * safe_res
    
    return n_cols, n_rows, n_layers, safe_res, total_width, total_height, total_depth


def compute_mesh_signature(
    verts: np.ndarray,
    tris: np.ndarray,
    bone_names: List[str],
    res: int,
    enable_joint_mesh: bool = False,
    joint_weight_threshold: float = 0.85,
) -> str:
    """メッシュと設定からユニークなキャッシュハッシュキーを生成する"""
    hasher = hashlib.sha256()
    prefix = f"v8_hierarchy_hybrid_{int(enable_joint_mesh)}_{joint_weight_threshold:.2f}"
    hasher.update(f"{prefix}_{verts.shape}_{tris.shape}_{res}_{len(bone_names)}".encode("utf-8"))
    # 先頭と末尾の数頂点をサンプリングしてハッシュ化
    sample_verts = verts[::max(1, len(verts) // 20)]
    hasher.update(sample_verts.tobytes())
    for name in sorted(bone_names):
        hasher.update(name.encode("utf-8"))
    return hasher.hexdigest()[:16]


def compute_vertex_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """メッシュ頂点の外向き法線を計算する"""
    normals = np.zeros_like(verts, dtype=np.float32)
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, tris[:, i], face_normals)
    lens = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.maximum(lens, 1e-8)


def bake_bone_sdf_from_data(
    mesh_verts: np.ndarray,               # shape: [V, 3] (メッシュ空間)
    mesh_tris: np.ndarray,                # shape: [F, 3] (三角形インデックス)
    bone_weights: Dict[str, np.ndarray],  # {bone_name: [V]}
    bone_bind_matrices: Dict[str, np.ndarray], # {bone_name: 4x4 matrix (ワールド/メッシュ空間 -> ボーンローカル空間)}
    resolution: int = 64,
    margin: float = 0.2,
    weight_threshold: float = 0.02,
    blend_k: float = 0.05,
    friction: float = 0.5,
    thickness: float = 0.005,
    restitution: float = 0.0,
    enable_joint_mesh: bool = False,
    joint_weight_threshold: float = 0.85,
    overlap_rings: int = 1,
    bone_parent_map: Optional[Dict[str, Optional[str]]] = None,
) -> BoneSdfBakeResult:
    """
    素体メッシュとボーン情報からローカルSDFテクスチャアトラスを生成する
    """
    # 1. 有効なボーンの選定
    active_bones = []
    for b_name, w_arr in bone_weights.items():
        if b_name in bone_bind_matrices and np.max(w_arr) > weight_threshold:
            active_bones.append(b_name)

    active_bones.sort()
    n_bones = len(active_bones)
    if n_bones == 0:
        logger.warning("[SDF Baker] 有効なボーンが見つかりませんでした")
        return BoneSdfBakeResult(b"", 0, 0, 0, np.zeros((0, 20), dtype=np.float32), [], np.zeros((0, 4, 4), dtype=np.float32))

    logger.info(f"[SDF Baker] ボーンSDFベイク開始: {n_bones} 本のボーン (解像度: {resolution}^3)")

    n_cols, n_rows, n_layers, res, total_width, total_height, total_depth = compute_3d_atlas_layout(n_bones, resolution)

    gpu_baked = False
    texture_bytes = b""
    bone_infos_arr = np.zeros((0, 20), dtype=np.float32)
    bind_matrices_arr = np.zeros((0, 4, 4), dtype=np.float32)

    # 2. GPUコンピュートシェーダーによる並列ベイクの優先試行
    try:
        import taremin_cloth_core
        if hasattr(taremin_cloth_core, "bake_bone_sdf_gpu") and taremin_cloth_core.is_gpu_available():
            logger.info(f"[SDF Baker] GPUコンピュートシェーダーによる高速並列ベイクを実行中...")
            t_start = time.time()
            b_weights_list = [np.ascontiguousarray(bone_weights[b], dtype=np.float32) for b in active_bones]
            b_mats_list = np.ascontiguousarray([bone_bind_matrices[b] for b in active_bones], dtype=np.float32)

            gpu_dict = taremin_cloth_core.bake_bone_sdf_gpu(
                mesh_verts=np.ascontiguousarray(mesh_verts, dtype=np.float32),
                mesh_tris=np.ascontiguousarray(mesh_tris, dtype=np.int32),
                bone_names=active_bones,
                bone_weights=b_weights_list,
                bone_bind_matrices=b_mats_list,
                resolution=int(resolution),
                margin=float(margin),
                weight_threshold=float(weight_threshold),
                blend_k=float(blend_k),
                friction=float(friction),
                thickness=float(thickness),
                restitution=float(restitution),
            )
            texture_bytes = gpu_dict["texture_bytes"]
            total_width = int(gpu_dict["width"])
            total_height = int(gpu_dict["height"])
            total_depth = int(gpu_dict["depth"])
            bone_infos_arr = np.ascontiguousarray(gpu_dict["bone_infos"], dtype=np.float32)
            bind_matrices_arr = np.ascontiguousarray(gpu_dict["bind_matrices"], dtype=np.float32)
            active_bones = list(gpu_dict["active_bones"])
            gpu_baked = True
            logger.info(f"[SDF Baker] GPUベイク完了 ({time.time() - t_start:.3f}秒, {len(active_bones)} 本)")
    except Exception as e:
        logger.warning(f"[SDF Baker] GPUベイク失敗、CPUフォールバックを実行します: {e}")
        gpu_baked = False

    # 3. CPUフォールバック (BVHTree または NumPy)
    if not gpu_baked:
        logger.info(f"[SDF Baker] CPUフォールバックベイクを開始します...")
        bvh = None
        if HAS_MATHUTILS_BVH:
            verts_math = [mathutils.Vector(v) for v in mesh_verts]
            polys = [tuple(t) for t in mesh_tris]
            bvh = BVHTree.FromPolygons(verts_math, polys, all_triangles=True)

        vram_bytes = total_width * total_height * total_depth * 4  # Rg16Float (4 bytes/voxel)
        vram_mb = vram_bytes / (1024 * 1024)
        if vram_mb > 1500.0:
            logger.warning(
                f"[SDF Baker] 警告: SDFテクスチャの推定VRAM使用量が大きいです: {vram_mb:.1f} MB ({total_width}x{total_height}x{total_depth})"
            )

        atlas_3d = np.zeros((total_depth, total_height, total_width, 2), dtype=np.float16)
        bone_infos_list = []
        bind_matrices_list = []

        for b_idx, b_name in enumerate(active_bones):
            w_arr = bone_weights[b_name]
            valid_indices = np.where(w_arr > weight_threshold)[0]
            if len(valid_indices) == 0:
                continue

            mat_inv = bone_bind_matrices[b_name]  # メッシュ空間 -> ボーン空間
            try:
                mat_fwd = np.linalg.inv(mat_inv)  # ボーン空間 -> メッシュ空間
            except np.linalg.LinAlgError:
                mat_fwd = np.eye(4, dtype=np.float32)

            verts_target = mesh_verts[valid_indices]
            ones = np.ones((len(verts_target), 1), dtype=np.float32)
            v_homo = np.hstack([verts_target, ones])
            v_local = (v_homo @ mat_inv.T)[:, :3]

            local_min = np.min(v_local, axis=0)
            local_max = np.max(v_local, axis=0)
            size = local_max - local_min
            size = np.maximum(size, 0.05)
            local_min -= size * margin
            local_max += size * margin
            size = local_max - local_min

            b_per_layer = n_cols * n_rows
            l = b_idx // b_per_layer
            rem = b_idx % b_per_layer
            r = rem // n_cols
            c = rem % n_cols

            uvw_scale_local = 1.0 / size
            uvw_offset_local = -local_min / size

            atlas_uvw_scale = np.array([
                uvw_scale_local[0] / float(n_cols),
                uvw_scale_local[1] / float(n_rows),
                uvw_scale_local[2] / float(n_layers),
                blend_k,
            ], dtype=np.float32)

            atlas_uvw_offset = np.array([
                (uvw_offset_local[0] + float(c)) / float(n_cols),
                (uvw_offset_local[1] + float(r)) / float(n_rows),
                (uvw_offset_local[2] + float(l)) / float(n_layers),
                0.0,
            ], dtype=np.float32)

            bone_params = np.array([friction, thickness, restitution, weight_threshold], dtype=np.float32)

            info_row = np.concatenate([
                np.append(local_min, float(b_idx)),
                np.append(local_max, float(res)),
                atlas_uvw_scale,
                atlas_uvw_offset,
                bone_params,
            ]).astype(np.float32)
            bone_infos_list.append(info_row)
            bind_matrices_list.append(mat_inv)

            grid_x = np.linspace(local_min[0], local_max[0], res, dtype=np.float32)
            grid_y = np.linspace(local_min[1], local_max[1], res, dtype=np.float32)
            grid_z = np.linspace(local_min[2], local_max[2], res, dtype=np.float32)
            gx, gy, gz = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')

            pts_local = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
            ones_pts = np.ones((len(pts_local), 1), dtype=np.float32)
            pts_mesh = (np.hstack([pts_local, ones_pts]) @ mat_fwd.T)[:, :3]

            dists = np.zeros(len(pts_mesh), dtype=np.float32)
            alphas = np.zeros(len(pts_mesh), dtype=np.float32)

            if bvh is not None:
                for p_idx, pt in enumerate(pts_mesh):
                    v_pt = mathutils.Vector((pt[0], pt[1], pt[2]))
                    loc, norm, poly_idx, d = bvh.find_nearest(v_pt)
                    if loc is not None:
                        diff = v_pt - loc
                        sign = 1.0 if diff.dot(norm) >= 0.0 else -1.0
                        dists[p_idx] = d * sign
                        tri = mesh_tris[poly_idx]
                        tri_weights = w_arr[tri]
                        alphas[p_idx] = np.mean(tri_weights)
                    else:
                        dists[p_idx] = 10.0
                        alphas[p_idx] = 0.0
            else:
                vert_normals = compute_vertex_normals(mesh_verts, mesh_tris)
                sub_verts = mesh_verts[valid_indices]
                sub_weights = w_arr[valid_indices]
                sub_normals = vert_normals[valid_indices]

                tri_v0 = mesh_verts[mesh_tris[:, 0]]
                tri_v1 = mesh_verts[mesh_tris[:, 1]]
                tri_v2 = mesh_verts[mesh_tris[:, 2]]
                tri_centers = (tri_v0 + tri_v1 + tri_v2) / 3.0
                tri_normals = np.cross(tri_v1 - tri_v0, tri_v2 - tri_v0)
                tri_lens = np.linalg.norm(tri_normals, axis=-1, keepdims=True)
                tri_normals = tri_normals / np.maximum(tri_lens, 1e-8)

                chunk_size = 2048
                for c_start in range(0, len(pts_mesh), chunk_size):
                    c_end = min(c_start + chunk_size, len(pts_mesh))
                    p_chunk = pts_mesh[c_start:c_end]

                    diffs_v = p_chunk[:, np.newaxis, :] - sub_verts[np.newaxis, :, :]
                    sq_dists_v = np.sum(diffs_v * diffs_v, axis=-1)
                    nearest_v_idx = np.argmin(sq_dists_v, axis=-1)
                    min_dv = np.sqrt(sq_dists_v[np.arange(len(p_chunk)), nearest_v_idx])

                    diffs_t = p_chunk[:, np.newaxis, :] - tri_centers[np.newaxis, :, :]
                    sq_dists_t = np.sum(diffs_t * diffs_t, axis=-1)
                    nearest_t_idx = np.argmin(sq_dists_t, axis=-1)

                    best_t = nearest_t_idx
                    t_norm = tri_normals[best_t]
                    t_center = tri_centers[best_t]
                    dot_t = np.sum((p_chunk - t_center) * t_norm, axis=-1)

                    plane_dist = np.abs(dot_t)
                    min_d = np.minimum(min_dv, plane_dist)

                    signs = np.where(dot_t >= 0.0, 1.0, -1.0)
                    dists[c_start:c_end] = min_d * signs
                    alphas[c_start:c_end] = sub_weights[nearest_v_idx]

            dists_3d = dists.reshape((res, res, res))
            alphas_3d = alphas.reshape((res, res, res))
            bone_voxels_xyz = np.stack([dists_3d, alphas_3d], axis=-1)
            bone_voxels_zyx = np.transpose(bone_voxels_xyz, (2, 1, 0, 3)).astype(np.float16)

            z0, z1 = l * res, (l + 1) * res
            y0, y1 = r * res, (r + 1) * res
            x0, x1 = c * res, (c + 1) * res
            atlas_3d[z0:z1, y0:y1, x0:x1, :] = bone_voxels_zyx

        texture_bytes = np.ascontiguousarray(atlas_3d).tobytes()
        bone_infos_arr = np.array(bone_infos_list, dtype=np.float32)
        bind_matrices_arr = np.array(bind_matrices_list, dtype=np.float32)

    # ハイブリッド用に関節部三角形インデックスを抽出 (親子階層自動抽出方式、または主要関節フォールバック)
    joint_face_indices = np.empty(0, dtype=np.int32)
    joint_faces_by_pair: Dict[Tuple[str, str], np.ndarray] = {}
    if enable_joint_mesh:
        if bone_parent_map:
            joint_face_indices, joint_faces_by_pair = extract_hierarchy_joint_mesh_indices(
                mesh_verts=mesh_verts,
                mesh_tris=mesh_tris,
                bone_weights=bone_weights,
                bone_parent_map=bone_parent_map,
                min_blend_weight=0.05,
            )
            logger.info(
                f"[SDF Baker] ハイブリッドモード有効 (親子階層方式): {len(joint_face_indices)} / {len(mesh_tris)} 面 "
                f"({len(joint_faces_by_pair)} 個の関節ペア) を抽出"
            )
        else:
            joint_face_indices = extract_major_joint_mesh_indices(
                mesh_verts=mesh_verts,
                mesh_tris=mesh_tris,
                bone_weights=bone_weights,
                min_blend_weight=0.06,
            )
            logger.info(
                f"[SDF Baker] ハイブリッドモード有効 (主要関節フォールバック): {len(joint_face_indices)} / {len(mesh_tris)} 面の関節三角形を抽出 (のりしろ: any1)"
            )

    logger.info(
        f"[SDF Baker] ベイク完了: テクスチャサイズ {total_width}x{total_height}x{total_depth} "
        f"({len(texture_bytes) // 1024} KB, タイル: {n_cols}x{n_rows}x{n_layers})"
    )

    return BoneSdfBakeResult(
        texture_bytes=texture_bytes,
        width=total_width,
        height=total_height,
        depth=total_depth,
        bone_infos=bone_infos_arr,
        bone_names=active_bones,
        bind_matrices=bind_matrices_arr,
        joint_face_indices=joint_face_indices,
        joint_faces_by_pair=joint_faces_by_pair,
    )


def get_cache_dir() -> str:
    """SDFキャッシュ保存先ディレクトリを取得・作成する"""
    import tempfile
    cache_dir = os.path.join(tempfile.gettempdir(), "taremin_cloth_sdf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def load_cached_sdf(cache_key: str) -> Optional[BoneSdfBakeResult]:
    """ディスクキャッシュからSDFベイク結果をロードする"""
    cache_path = os.path.join(get_cache_dir(), f"{cache_key}.npz")
    if not os.path.exists(cache_path):
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        joint_face_indices = data["joint_face_indices"] if "joint_face_indices" in data else np.empty(0, dtype=np.int32)
        joint_faces_by_pair: Dict[Tuple[str, str], np.ndarray] = {}
        if "pair_keys" in data and "pair_lens" in data and "pair_data" in data:
            p_keys = data["pair_keys"]
            p_lens = data["pair_lens"]
            p_data = data["pair_data"]
            offset = 0
            for k_str, l in zip(p_keys, p_lens):
                parts = str(k_str).split(":::")
                if len(parts) == 2:
                    joint_faces_by_pair[(parts[0], parts[1])] = p_data[offset : offset + l]
                offset += l

        return BoneSdfBakeResult(
            texture_bytes=data["texture_bytes"].tobytes(),
            width=int(data["width"]),
            height=int(data["height"]),
            depth=int(data["depth"]),
            bone_infos=data["bone_infos"],
            bone_names=list(data["bone_names"]),
            bind_matrices=data["bind_matrices"],
            joint_face_indices=joint_face_indices,
            joint_faces_by_pair=joint_faces_by_pair,
        )
    except Exception as e:
        logger.warning(f"[SDF Baker] キャッシュ読込失敗: {e}")
        return None


def save_cached_sdf(cache_key: str, result: BoneSdfBakeResult):
    """SDFベイク結果をディスクキャッシュに保存する"""
    cache_path = os.path.join(get_cache_dir(), f"{cache_key}.npz")
    try:
        pairs = list(result.joint_faces_by_pair.keys())
        pair_keys = np.array([f"{p}:::{c}" for p, c in pairs])
        pair_lens = np.array([len(result.joint_faces_by_pair[k]) for k in pairs], dtype=np.int32)
        pair_data = (
            np.concatenate([result.joint_faces_by_pair[k] for k in pairs])
            if pairs
            else np.empty(0, dtype=np.int32)
        )

        np.savez_compressed(
            cache_path,
            texture_bytes=np.frombuffer(result.texture_bytes, dtype=np.uint8),
            width=result.width,
            height=result.height,
            depth=result.depth,
            bone_infos=result.bone_infos,
            bone_names=np.array(result.bone_names),
            bind_matrices=result.bind_matrices,
            joint_face_indices=result.joint_face_indices,
            pair_keys=pair_keys,
            pair_lens=pair_lens,
            pair_data=pair_data,
        )
        logger.info(f"[SDF Baker] キャッシュ保存完了: {cache_path}")
    except Exception as e:
        logger.warning(f"[SDF Baker] キャッシュ保存失敗: {e}")


def clear_all_cached_sdf() -> int:
    """保存されているすべてのSDFキャッシュファイルを削除し、削除件数を返す"""
    cache_dir = get_cache_dir()
    count = 0
    if os.path.exists(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.endswith(".npz"):
                try:
                    os.remove(os.path.join(cache_dir, fname))
                    count += 1
                except Exception as e:
                    logger.warning(f"[SDF Baker] キャッシュ削除失敗: {fname}: {e}")
    logger.info(f"[SDF Baker] {count} 件のSDFキャッシュを削除しました")
    return count


def get_armature_modifier(obj):
    """オブジェクトのArmatureモディファイアを取得する"""
    if not obj or getattr(obj, "type", None) != 'MESH':
        return None
    for mod in getattr(obj, "modifiers", []):
        if getattr(mod, "type", None) == 'ARMATURE' and getattr(mod, "object", None):
            return mod
    return None


def get_or_bake_bone_sdf_for_object(obj, col_settings, force_rebake: bool = False) -> Optional[BoneSdfBakeResult]:
    """
    Blenderオブジェクトからアーマチュアとメッシュを解析し、SDFベイク結果を取得（またはキャッシュから読込）する
    """
    arm_mod = get_armature_modifier(obj)
    if not arm_mod:
        logger.warning(f"[SDF Baker] オブジェクト '{getattr(obj, 'name', '')}' に有効なArmatureモディファイアが見つかりません")
        return None

    arm_obj = arm_mod.object

    res_str = getattr(col_settings, "sdf_resolution", "64")
    if res_str == "CUSTOM":
        resolution = getattr(col_settings, "sdf_resolution_custom", 128)
    else:
        try:
            resolution = int(res_str)
        except ValueError:
            resolution = 64

    margin = getattr(col_settings, "sdf_margin", 0.2)
    weight_threshold = getattr(col_settings, "weight_threshold", 0.02)
    blend_k = getattr(col_settings, "blend_k", 0.05)
    cache_enabled = getattr(col_settings, "sdf_cache_enabled", True)

    # Armatureモディファイアより前の先行モディファイア（Mirror, Lattice等）を反映した静止メッシュを取得
    disabled_mods = []
    found_arm = False
    for mod in getattr(obj, "modifiers", []):
        if mod == arm_mod:
            found_arm = True
        if found_arm or getattr(mod, "type", None) == 'COLLISION':
            if getattr(mod, "show_viewport", False):
                mod.show_viewport = False
                disabled_mods.append(mod)

    mesh_is_eval = False
    eval_obj = None
    try:
        try:
            bpy.context.view_layer.update()
        except Exception:
            pass
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        mesh_is_eval = True
    except Exception as e:
        logger.warning(f"[SDF Baker] 評価メッシュ取得失敗、obj.dataにフォールバックします: {e}")
        mesh = obj.data
        mesh_is_eval = False
    # メッシュ頂点と三角形の抽出
    n_verts = len(mesh.vertices)
    raw_coords = np.empty(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", raw_coords)
    mesh_verts = raw_coords.reshape((n_verts, 3))

    mesh.calc_loop_triangles()
    n_tris = len(mesh.loop_triangles)
    tri_indices = np.empty(n_tris * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tri_indices)
    mesh_tris = tri_indices.reshape((n_tris, 3))

    # 頂点グループ名のマップ
    vgroup_indices = {vg.name: vg.index for vg in obj.vertex_groups}

    # 全頂点のグループウェイトを1パスで集約
    vg_weights_map = {}
    for v in mesh.vertices:
        v_idx = v.index
        for g in v.groups:
            g_idx = g.group
            if g_idx not in vg_weights_map:
                vg_weights_map[g_idx] = []
            vg_weights_map[g_idx].append((v_idx, g.weight))

    # データ抽出完了後に評価メッシュの解放とモディファイア復元
    if mesh_is_eval and eval_obj is not None:
        try:
            eval_obj.to_mesh_clear()
        except Exception:
            pass
    for mod in disabled_mods:
        mod.show_viewport = True
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass

    # 頂点グループからボーンウェイトを抽出
    bone_weights = {}
    bone_bind_matrices = {}
    arm_data = arm_obj.data

    # アーマチュア空間 -> メッシュ空間の相対変換
    mat_mesh_world = np.array(obj.matrix_world, dtype=np.float32)
    mat_arm_world = np.array(arm_obj.matrix_world, dtype=np.float32)
    try:
        mat_arm_inv = np.linalg.inv(mat_arm_world)
    except np.linalg.LinAlgError:
        mat_arm_inv = np.eye(4, dtype=np.float32)

    mat_mesh_to_arm = mat_arm_inv @ mat_mesh_world

    for bone in arm_data.bones:
        b_name = bone.name
        if b_name not in vgroup_indices:
            continue

        vg_idx = vgroup_indices[b_name]
        if vg_idx not in vg_weights_map:
            continue

        w_arr = np.zeros(n_verts, dtype=np.float32)
        for v_idx, w_val in vg_weights_map[vg_idx]:
            w_arr[v_idx] = w_val

        if np.max(w_arr) > weight_threshold:
            bone_weights[b_name] = w_arr

            # bone.matrix_local: アーマチュア空間におけるボーンの静止ポーズ変換行列
            b_mat_local = np.array(bone.matrix_local, dtype=np.float32)
            try:
                b_mat_inv = np.linalg.inv(b_mat_local)
            except np.linalg.LinAlgError:
                b_mat_inv = np.eye(4, dtype=np.float32)

            mat_mesh_to_bone = b_mat_inv @ mat_mesh_to_arm
            bone_bind_matrices[b_name] = mat_mesh_to_bone

    if not bone_weights:
        logger.warning(f"[SDF Baker] 有効なボーンウェイトが見つかりませんでした")
        return None

    bone_parent_map = {b.name: (b.parent.name if b.parent else None) for b in arm_data.bones}

    enable_joint_mesh = getattr(col_settings, "enable_joint_mesh", True)
    joint_weight_threshold = getattr(col_settings, "joint_weight_threshold", 0.85)

    # キャッシュチェック
    cache_key = compute_mesh_signature(
        mesh_verts,
        mesh_tris,
        list(bone_weights.keys()),
        resolution,
        enable_joint_mesh=enable_joint_mesh,
        joint_weight_threshold=joint_weight_threshold,
    )
    if cache_enabled and not force_rebake:
        cached = load_cached_sdf(cache_key)
        if cached is not None:
            # 3Dテクスチャはキャッシュを即座に復元しつつ、関節面は最新の親子階層ロジックを適用
            if enable_joint_mesh:
                cached.joint_face_indices, cached.joint_faces_by_pair = extract_hierarchy_joint_mesh_indices(
                    mesh_verts=mesh_verts,
                    mesh_tris=mesh_tris,
                    bone_weights=bone_weights,
                    bone_parent_map=bone_parent_map,
                    min_blend_weight=0.05,
                )
            logger.info(
                f"[SDF Baker] キャッシュからSDFを即座に復元しました: {cache_key} "
                f"(関節面数: {len(cached.joint_face_indices)}, ペア数: {len(cached.joint_faces_by_pair)})"
            )
            return cached

    # ベイク実行
    result = bake_bone_sdf_from_data(
        mesh_verts=mesh_verts,
        mesh_tris=mesh_tris,
        bone_weights=bone_weights,
        bone_bind_matrices=bone_bind_matrices,
        resolution=resolution,
        margin=margin,
        weight_threshold=weight_threshold,
        blend_k=blend_k,
        friction=float(col_settings.friction),
        thickness=float(col_settings.thickness),
        restitution=float(getattr(col_settings, "restitution", 0.0)),
        enable_joint_mesh=enable_joint_mesh,
        joint_weight_threshold=joint_weight_threshold,
        bone_parent_map=bone_parent_map,
    )

    if cache_enabled and result.depth > 0:
        save_cached_sdf(cache_key, result)

    return result
