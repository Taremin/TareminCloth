"""
taremin_cloth シミュレーション キャッシュ・状態管理モジュール
レストポーズ座標、トポロジーシグネチャ、シミュレータインスタンス等のグローバル状態を一元管理する。
"""

import bpy
import numpy as np
from ..utils import topology
from ..utils.logger import logger

# シミュレータインスタンスを保持するグローバルキャッシュ (obj_name -> (sim, coords))
_simulators = {}
# レストポーズ（初期頂点座標）を保持するグローバルキャッシュ
_rest_positions = {}
# 直前フレームの頂点座標キャッシュ（適応型サブステップ制御用）
_prev_coords_cache = {}
# メッシュ特性長 (最小エッジ長と厚みから算出) キャッシュ
_mesh_char_len_cache = {}
# 直前フレームの適応サブステップ数キャッシュ（平滑化・ジッター防止用）
_effective_substeps_cache = {}
# コライダーの直前ワールド位置キャッシュ（コライダー速度検知用）
_collider_prev_locs_cache = {}
# 伸縮グループの直前スケールキャッシュ（急激な変化の平滑化用）
_prev_elastic_scales = {}
# タイムライン再生用フレーム座標キャッシュ (obj_name -> { frame_num: np.ndarray })
_timeline_frame_cache = {}
# バッファリング開始フレーム番号 (obj_name -> start_frame)
_buffered_start_frames = {}
# 布パラメータシグネチャキャッシュ (obj_name -> params_tuple)
_cloth_param_signatures = {}
# 剛性パラメータの前回同期値キャッシュ (obj_name -> (tension, compression, shear, bending))
_prev_stiffness_cache = {}
# アタッチメントピンの頂点インデックス・ウェイト一覧キャッシュ (cache_key -> list[(vert_idx, weight)])
_attachment_pin_indices_cache = {}
# アタッチメントピンの前回ローカルターゲット位置キャッシュ (obj_name -> (local_target_tuple, target_mat_tuple))
_prev_attachment_pin_targets = {}


def _get_mesh_topology_signature(mesh):
    """メッシュのトポロジーシグネチャ (頂点数, 辺数, 面数) を取得する"""
    return (len(mesh.vertices), len(mesh.edges), len(mesh.polygons))


def cache_rest_positions(obj, force=False):
    """メッシュのレストポーズ（初期頂点座標）をキャッシュする
    - 初回
    - トポロジー（頂点数、辺数、面数）が変更された場合 (Dirty)
    - シミュレーションによる変形が起きていない初期状態（_taremin_cloth_is_deformed が False）の場合 (Dirty: ユーザーによる頂点移動編集)
    - force=True の場合
    にキャッシュを最新のメッシュ形状で更新する。
    """
    if not obj or obj.type != 'MESH':
        return

    mesh = obj.data
    sig = _get_mesh_topology_signature(mesh)
    n_verts = len(mesh.vertices)

    has_cache = "_taremin_cloth_rest_positions" in obj
    is_deformed = obj.get("_taremin_cloth_is_deformed", False)

    # シミュレーションによる変形が起きている最中は、force=Trueであっても変形座標でレストポーズを汚染しないよう上書きを拒否
    if is_deformed:
        logger.debug(f"[Cache] Refusing to cache rest positions for deformed object '{obj.name}' (is_deformed=True)")
        return

    cached_sig = None
    if "_taremin_cloth_rest_signature" in obj:
        try:
            cached_sig = tuple(obj["_taremin_cloth_rest_signature"])
        except Exception:
            cached_sig = None

    need_update = False
    if not has_cache or force:
        need_update = True
    elif cached_sig != sig:
        need_update = True
    elif not is_deformed:
        need_update = True

    logger.debug(
        f"[Cache] cache_rest_positions: obj='{obj.name}', force={force}, "
        f"has_cache={has_cache}, is_deformed={is_deformed}, sig={sig}, cached_sig={cached_sig}, need_update={need_update}"
    )

    if need_update:
        coords = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coords)
        if "_taremin_cloth_rest_positions" in obj:
            del obj["_taremin_cloth_rest_positions"]
        obj["_taremin_cloth_rest_positions"] = coords.tobytes()
        obj["_taremin_cloth_rest_signature"] = list(sig)
        obj["_taremin_cloth_is_deformed"] = False

        logger.info(f"[Cache] Cached rest positions for '{obj.name}' ({n_verts} verts, sig={sig})")
    else:
        logger.debug(f"[Cache] Rest positions update skipped for '{obj.name}'")


def restore_rest_positions(obj, clear=False, clear_timeline=True):
    """メッシュの頂点座標・トポロジーをレストポーズ（シミュレーション前の初期状態）に安全に復元する"""
    if not obj or obj.type != 'MESH':
        return

    logger.info(f"[Reset] restore_rest_positions called: obj='{obj.name}', clear={clear}, clear_timeline={clear_timeline}")
    clear_simulator_for_object(obj.name, clear_timeline=clear_timeline)

    # 1. 十字分割オブジェクトであり、メッシュが編集されていない場合はバックアップからQuad復元
    restore_fn = getattr(topology, "restore_pre_subdivision_mesh", None)
    if restore_fn and restore_fn(obj):
        logger.info(f"[Reset] Successfully restored topology & coordinates from pre-subdivision backup for '{obj.name}'")
        cache_rest_positions(obj, force=True)
        if clear:
            clear_backup_fn = getattr(topology, "clear_pre_subdivision_backup", None)
            if clear_backup_fn:
                clear_backup_fn(obj)
            if "_taremin_cloth_rest_positions" in obj:
                del obj["_taremin_cloth_rest_positions"]
            if "_taremin_cloth_rest_signature" in obj:
                del obj["_taremin_cloth_rest_signature"]
            if "_taremin_cloth_is_deformed" in obj:
                del obj["_taremin_cloth_is_deformed"]
        else:
            obj["_taremin_cloth_is_deformed"] = False
            obj.update_tag()
        return

    # 2. バックアップがない場合でも、十字分割の自己修復が可能な場合
    if getattr(topology, "is_cross_subdivided", None) and topology.is_cross_subdivided(obj):
        logger.info(f"[Reset] Self-healing: Dissolving center vertices to restore Quad mesh for '{obj.name}'")
        restore_quads_fn = getattr(topology, "restore_original_quads", None)
        if restore_quads_fn:
            restore_quads_fn(obj)

    # 3. レストポーズ座標の復元
    mesh = obj.data
    if "_taremin_cloth_rest_positions" in obj:
        initial_coords = np.frombuffer(obj["_taremin_cloth_rest_positions"], dtype=np.float32)
        mesh_verts = len(mesh.vertices)
        cache_verts = len(initial_coords) // 3
        logger.debug(f"[Reset] Coordinate restoration check for '{obj.name}': Mesh verts={mesh_verts}, Cache verts={cache_verts}")
        if mesh_verts == cache_verts:
            mesh.vertices.foreach_set("co", initial_coords)
            mesh.update()
            logger.info(f"[Reset] Restored {mesh_verts} vertex coordinates to rest positions for '{obj.name}'")
        else:
            logger.warning(
                f"[Reset] Vertex count mismatch for '{obj.name}': Mesh={mesh_verts}, Cache={cache_verts}. "
                f"Skipping coordinate restoration to preserve user mesh edits."
            )
            if not obj.get("_taremin_cloth_is_deformed", False):
                cache_rest_positions(obj, force=True)

    if clear:
        clear_backup_fn = getattr(topology, "clear_pre_subdivision_backup", None)
        if clear_backup_fn:
            clear_backup_fn(obj)
        if "_taremin_cloth_rest_positions" in obj:
            del obj["_taremin_cloth_rest_positions"]
        if "_taremin_cloth_rest_signature" in obj:
            del obj["_taremin_cloth_rest_signature"]
        if "_taremin_cloth_is_deformed" in obj:
            del obj["_taremin_cloth_is_deformed"]
        logger.info(f"[Reset] Cleared all rest caches for '{obj.name}'")
    else:
        obj["_taremin_cloth_is_deformed"] = False
        obj.update_tag()


def clear_simulator_for_object(obj_name, clear_timeline=True):
    """特定オブジェクトに対応するシミュレータおよび関連キャッシュを破棄する"""
    global _simulators, _prev_coords_cache, _mesh_char_len_cache
    global _effective_substeps_cache, _prev_elastic_scales
    had_sim = obj_name in _simulators
    _simulators.pop(obj_name, None)
    _prev_coords_cache.pop(obj_name, None)
    _mesh_char_len_cache.pop(obj_name, None)
    _effective_substeps_cache.pop(obj_name, None)
    _collider_prev_locs_cache.pop(obj_name, None)
    _prev_elastic_scales.pop(obj_name, None)
    _prev_stiffness_cache.pop(obj_name, None)
    _prev_attachment_pin_targets.pop(obj_name, None)
    for k in list(_attachment_pin_indices_cache.keys()):
        if isinstance(k, tuple) and k[0] == obj_name:
            _attachment_pin_indices_cache.pop(k, None)
    if clear_timeline:
        _timeline_frame_cache.pop(obj_name, None)
        _buffered_start_frames.pop(obj_name, None)
        _cloth_param_signatures.pop(obj_name, None)
    if had_sim:
        logger.debug(f"[Simulator] Cleared simulator and caches for '{obj_name}'")


def clear_simulators():
    """全シミュレータインスタンスおよび各種キャッシュを破棄する"""
    from .collider import clear_collider_cache
    from .runner import restore_fast_playback
    global _simulators, _prev_coords_cache
    global _mesh_char_len_cache, _effective_substeps_cache, _collider_prev_locs_cache
    global _prev_elastic_scales, _timeline_frame_cache, _buffered_start_frames, _cloth_param_signatures
    global _prev_stiffness_cache, _attachment_pin_indices_cache, _prev_attachment_pin_targets
    restore_fast_playback()
    clear_collider_cache()
    count = len(_simulators)
    _simulators.clear()
    _prev_coords_cache.clear()
    _mesh_char_len_cache.clear()
    _effective_substeps_cache.clear()
    _collider_prev_locs_cache.clear()
    _prev_elastic_scales.clear()
    _timeline_frame_cache.clear()
    _buffered_start_frames.clear()
    _cloth_param_signatures.clear()
    _prev_stiffness_cache.clear()
    _attachment_pin_indices_cache.clear()
    _prev_attachment_pin_targets.clear()
    logger.debug(f"[Simulator] Cleared all {count} simulators and caches")


def get_cloth_params_signature(obj):
    """布オブジェクトのシミュレーション物理パラメータシグネチャを生成する"""
    settings = getattr(obj, "taremin_cloth", None)
    if not settings or not getattr(settings, "is_cloth", False):
        return None

    elastic_tuples = ()
    if hasattr(settings, "elastic_groups"):
        elastic_tuples = tuple(
            (g.name, g.enabled, round(float(g.scale), 4), round(float(g.stiffness_multiplier), 4), str(g.edge_indices_str))
            for g in settings.elastic_groups
        )

    pin_obj_name = settings.pin_target_object.name if getattr(settings, "pin_target_object", None) else ""
    return (
        round(float(getattr(settings, "tension_stiffness", 1000.0)), 2),
        round(float(getattr(settings, "compression_stiffness", 1000.0)), 2),
        round(float(getattr(settings, "shear_stiffness", 1000.0)), 2),
        round(float(getattr(settings, "bending_stiffness", 10.0)), 2),
        round(float(getattr(settings, "air_damping", 0.05)), 4),
        round(float(getattr(settings, "tension_damping", 0.1)), 4),
        round(float(getattr(settings, "compression_damping", 0.1)), 4),
        round(float(getattr(settings, "shear_damping", 0.1)), 4),
        round(float(getattr(settings, "bending_damping", 0.1)), 4),
        round(float(getattr(settings, "gravity", 1.0)), 4),
        str(getattr(settings, "pin_vertex_group", "")),
        pin_obj_name,
        str(getattr(settings, "pin_target_bone", "")),
        bool(getattr(settings, "enable_sewing", False)),
        round(float(getattr(settings, "sewing_shrink_speed", 1.0)), 4),
        round(float(getattr(settings, "sewing_stiffness", 10000.0)), 2),
        bool(getattr(settings, "enable_sewing_lock", True)),
        round(float(getattr(settings, "thickness", 0.005)), 5),
        bool(getattr(settings, "enable_self_collision", False)),
        str(getattr(settings, "self_collision_purpose", "STANDARD")),
        int(getattr(settings, "substeps", 10)),
        int(getattr(settings, "solver_iterations", 2)),
        str(getattr(settings, "solver_mode", "COLORING")),
        elastic_tuples,
    )


def check_and_update_params_signature(obj):
    """パラメータが変更されていればキャッシュを自動破棄し、Trueを返す"""
    if not obj:
        return False
    new_sig = get_cloth_params_signature(obj)
    if new_sig is None:
        return False

    old_sig = _cloth_param_signatures.get(obj.name)
    if old_sig is not None and old_sig != new_sig:
        logger.info(f"[Cache] Cloth parameters changed for '{obj.name}'. Invaliding timeline cache & simulator.")
        _cloth_param_signatures[obj.name] = new_sig
        clear_simulator_for_object(obj.name)
        return True

    _cloth_param_signatures[obj.name] = new_sig
    return False


def get_timeline_cache_info(obj_name):
    """指定オブジェクトのタイムラインキャッシュ情報を取得する"""
    cache = _timeline_frame_cache.get(obj_name, {})
    count = len(cache)
    if count == 0:
        return {
            "count": 0,
            "range_str": "Empty",
            "size_kb": 0.0,
            "min_frame": 0,
            "max_frame": 0,
        }

    frames = sorted(cache.keys())
    min_f, max_f = frames[0], frames[-1]
    range_str = f"{min_f} - {max_f}" if count > 1 else str(min_f)
    sample = next(iter(cache.values()))
    size_kb = (len(sample) * 4 * count) / 1024.0
    return {
        "count": count,
        "range_str": range_str,
        "size_kb": size_kb,
        "min_frame": min_f,
        "max_frame": max_f,
    }


def clear_timeline_cache(obj_name=None):
    """タイムラインフレームキャッシュを破棄する"""
    if obj_name:
        _timeline_frame_cache.pop(obj_name, None)
    else:
        _timeline_frame_cache.clear()


def is_object_baked(obj):
    """指定Clothオブジェクトがベイク済み（2フレーム以上のキャッシュが存在する）かを返す"""
    if not obj:
        return False
    cache = _timeline_frame_cache.get(obj.name, {})
    return len(cache) >= 2


def is_scene_baked(scene):
    """シーン内のいずれかのClothオブジェクトがベイク済みかを返す"""
    if not scene:
        return False
    for obj in scene.objects:
        if getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth and getattr(obj.taremin_cloth, "enabled", True):
            if is_object_baked(obj):
                return True
    return False


def free_object_bake(obj):
    """指定Clothオブジェクトのベイクデータを破棄し、初期レストポーズに復元する"""
    if not obj or obj.type != 'MESH':
        return
    clear_timeline_cache(obj.name)
    clear_simulator_for_object(obj.name, clear_timeline=True)
    restore_rest_positions(obj, clear=False, clear_timeline=True)
    obj.update_tag()
    logger.info(f"[Bake] Freed bake cache and restored rest shape for '{obj.name}'")


def free_scene_bake(scene):
    """シーン内すべてのClothオブジェクトのベイクデータを破棄し、初期レストポーズに復元する"""
    if not scene:
        return
    for obj in scene.objects:
        if getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth:
            free_object_bake(obj)
    clear_simulators()
    logger.info(f"[Bake] Freed all bake caches for scene '{scene.name}'")

