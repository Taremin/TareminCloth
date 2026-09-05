"""
taremin_cloth ボーン局所SDFベーカー (SDF Baker)
素体メッシュ（Skinned Mesh）およびアーマチュア情報から、
各ボーンのローカル符号付き距離場 (SDF) とボーンウェイト (Alpha) を
3D テクスチャアトラス (Rg16Float) として事前ベイクする。
"""

import os
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
    ):
        self.texture_bytes = texture_bytes
        self.width = width
        self.height = height
        self.depth = depth
        self.bone_infos = bone_infos
        self.bone_names = bone_names
        self.bind_matrices = bind_matrices
        self.joint_face_indices = joint_face_indices if joint_face_indices is not None else np.empty(0, dtype=np.int32)


def extract_joint_mesh_indices(
    mesh_verts: np.ndarray,
    mesh_tris: np.ndarray,
    bone_weights: Dict[str, np.ndarray],
    threshold: float = 0.85,
) -> np.ndarray:
    """
    複数ボーンブレンド領域（関節部）に含まれる三角形のインデックス配列を抽出する。
    戻り値: shape [M] (元の mesh_tris の行インデックス配列)
    """
    if not bone_weights or len(mesh_tris) == 0:
        return np.empty(0, dtype=np.int32)

    n_verts = len(mesh_verts)
    max_weights = np.zeros(n_verts, dtype=np.float32)
    for w_arr in bone_weights.values():
        max_weights = np.maximum(max_weights, w_arr)

    # 関節頂点: 最大ウェイトが閾値未満（複数ボーンで分散）かつ微小ウェイト以上の頂点
    is_joint_vert = (max_weights < threshold) & (max_weights > 0.01)

    # 三角形のいずれかの頂点が関節頂点であれば関節三角形として抽出
    tri_has_joint = is_joint_vert[mesh_tris[:, 0]] | is_joint_vert[mesh_tris[:, 1]] | is_joint_vert[mesh_tris[:, 2]]
    joint_indices = np.where(tri_has_joint)[0].astype(np.int32)
    return joint_indices


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
    prefix = f"v3_hybrid_{int(enable_joint_mesh)}_{joint_weight_threshold:.2f}"
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

    # ハイブリッド用に関節部三角形インデックスを抽出
    joint_face_indices = np.empty(0, dtype=np.int32)
    if enable_joint_mesh:
        joint_face_indices = extract_joint_mesh_indices(
            mesh_verts, mesh_tris, bone_weights, threshold=joint_weight_threshold
        )
        logger.info(
            f"[SDF Baker] ハイブリッドモード有効: {len(joint_face_indices)} / {len(mesh_tris)} 面の関節三角形を抽出 (閾値: {joint_weight_threshold})"
        )

    logger.info(f"[SDF Baker] ボーンSDFベイク開始: {n_bones} 本のボーン (解像度: {resolution}^3)")

    # 2. BVHTree の構築 (BlenderのC実装BVH、またはNumPyフォールバック)
    bvh = None
    if HAS_MATHUTILS_BVH:
        verts_math = [mathutils.Vector(v) for v in mesh_verts]
        polys = [tuple(t) for t in mesh_tris]
        bvh = BVHTree.FromPolygons(verts_math, polys, all_triangles=True)

    n_cols, n_rows, n_layers, res, total_width, total_height, total_depth = compute_3d_atlas_layout(n_bones, resolution)

    vram_bytes = total_width * total_height * total_depth * 4  # Rg16Float (4 bytes/voxel)
    vram_mb = vram_bytes / (1024 * 1024)
    if vram_mb > 1500.0:
        logger.warning(
            f"[SDF Baker] 警告: SDFテクスチャの推定VRAM使用量が大きいです: {vram_mb:.1f} MB ({total_width}x{total_height}x{total_depth})"
        )

    # 全体アトラス配列 (Z, Y, X, C)
    atlas_3d = np.zeros((total_depth, total_height, total_width, 2), dtype=np.float16)
    bone_infos_list = []
    bind_matrices_list = []

    # 3. 各ボーンのAABB計算とSDFサンプリング
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

        # 該当ボーンの影響を受ける頂点のローカル座標を算出
        verts_target = mesh_verts[valid_indices]
        ones = np.ones((len(verts_target), 1), dtype=np.float32)
        v_homo = np.hstack([verts_target, ones])
        v_local = (v_homo @ mat_inv.T)[:, :3]

        # ローカルAABBとマージン
        local_min = np.min(v_local, axis=0)
        local_max = np.max(v_local, axis=0)
        size = local_max - local_min
        size = np.maximum(size, 0.05)  # 最小サイズ 5cm 保証
        local_min -= size * margin
        local_max += size * margin
        size = local_max - local_min

        # タイル座標 (c: col, r: row, l: layer)
        b_per_layer = n_cols * n_rows
        l = b_idx // b_per_layer
        rem = b_idx % b_per_layer
        r = rem // n_cols
        c = rem % n_cols

        # UVWスケールとオフセット (3Dタイルスロットへの射影用)
        # ローカル [local_min, local_max] -> タイル範囲 [c/n_cols, (c+1)/n_cols], etc.
        uvw_scale_local = 1.0 / size
        uvw_offset_local = -local_min / size

        atlas_uvw_scale = np.array([
            uvw_scale_local[0] / float(n_cols),
            uvw_scale_local[1] / float(n_rows),
            uvw_scale_local[2] / float(n_layers),
            blend_k,  # w成分に blend_k
        ], dtype=np.float32)

        atlas_uvw_offset = np.array([
            (uvw_offset_local[0] + float(c)) / float(n_cols),
            (uvw_offset_local[1] + float(r)) / float(n_rows),
            (uvw_offset_local[2] + float(l)) / float(n_layers),
            0.0,  # 将来の atlas_id 用スロット
        ], dtype=np.float32)

        bone_params = np.array([friction, thickness, restitution, weight_threshold], dtype=np.float32)

        info_row = np.concatenate([
            np.append(local_min, float(b_idx)),
            np.append(local_max, float(res)),
            atlas_uvw_scale,
            atlas_uvw_offset,
            bone_params,
        ]).astype(np.float32)  # 20 elements
        bone_infos_list.append(info_row)
        bind_matrices_list.append(mat_inv)

        # 4. ボクセル座標からメッシュへの距離サンプリング
        # linspace でボクセル中心をサンプリング
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
            # Blender C-BVH 高速クエリ
            for p_idx, pt in enumerate(pts_mesh):
                v_pt = mathutils.Vector((pt[0], pt[1], pt[2]))
                loc, norm, poly_idx, d = bvh.find_nearest(v_pt)
                if loc is not None:
                    # 面法線との内積で内外符号を判定 (外側: 正, 内側: 負)
                    diff = v_pt - loc
                    sign = 1.0 if diff.dot(norm) >= 0.0 else -1.0
                    dists[p_idx] = d * sign

                    # 最近傍ポリゴンの頂点ウェイトを取得
                    tri = mesh_tris[poly_idx]
                    tri_weights = w_arr[tri]
                    alphas[p_idx] = np.mean(tri_weights)
                else:
                    dists[p_idx] = 10.0
                    alphas[p_idx] = 0.0
        else:
            # フォールバック: 頂点法線と三角形中心を用いた符号付き距離算出 (Blender外・テスト用)
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

        # ハイブリッドモード時、関節部メッシュ補完領域における剛体角突出を抑制するため、
        # 閾値未満のブレンドウェイト領域のAlphaを滑らかにフェードアウト
        if enable_joint_mesh and joint_weight_threshold > 0.0:
            fade = np.clip(alphas / joint_weight_threshold, 0.0, 1.0) ** 2
            alphas = alphas * fade

        # ボクセルアレイに格納 (ローカル [X, Y, Z, C] -> テクスチャ順 [Z, Y, X, C])
        dists_3d = dists.reshape((res, res, res))
        alphas_3d = alphas.reshape((res, res, res))
        bone_voxels_xyz = np.stack([dists_3d, alphas_3d], axis=-1)
        bone_voxels_zyx = np.transpose(bone_voxels_xyz, (2, 1, 0, 3)).astype(np.float16)

        z0, z1 = l * res, (l + 1) * res
        y0, y1 = r * res, (r + 1) * res
        x0, x1 = c * res, (c + 1) * res
        atlas_3d[z0:z1, y0:y1, x0:x1, :] = bone_voxels_zyx

    # 3Dテクスチャ用連続バイト列に変換
    texture_bytes = np.ascontiguousarray(atlas_3d).tobytes()

    bone_infos_arr = np.array(bone_infos_list, dtype=np.float32)
    bind_matrices_arr = np.array(bind_matrices_list, dtype=np.float32)

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
        return BoneSdfBakeResult(
            texture_bytes=data["texture_bytes"].tobytes(),
            width=int(data["width"]),
            height=int(data["height"]),
            depth=int(data["depth"]),
            bone_infos=data["bone_infos"],
            bone_names=list(data["bone_names"]),
            bind_matrices=data["bind_matrices"],
            joint_face_indices=joint_face_indices,
        )
    except Exception as e:
        logger.warning(f"[SDF Baker] キャッシュ読込失敗: {e}")
        return None


def save_cached_sdf(cache_key: str, result: BoneSdfBakeResult):
    """SDFベイク結果をディスクキャッシュに保存する"""
    cache_path = os.path.join(get_cache_dir(), f"{cache_key}.npz")
    try:
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
            logger.info(f"[SDF Baker] キャッシュからSDFを即座に復元しました: {cache_key}")
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
    )

    if cache_enabled and result.depth > 0:
        save_cached_sdf(cache_key, result)

    return result
