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


def _get_mesh_topology_signature(mesh):
    """メッシュのトポロジーシグネチャ (頂点数, 辺数, 面数) を取得する"""
    return (len(mesh.vertices), len(mesh.edges), len(mesh.polygons))


def cache_rest_positions(obj, force=False):
    """メッシュのレストポーズ（初期頂点座標）をキャッシュする
    - 初回
    - トポロジー（頂点数、辺数、面数）が変更された場合 (Dirty)
    - シミュレーションによる変形が起きていない初期状態（_taremin_is_deformed が False）の場合 (Dirty: ユーザーによる頂点移動編集)
    - force=True の場合
    にキャッシュを最新のメッシュ形状で更新する。
    """
    if not obj or obj.type != 'MESH':
        return

    mesh = obj.data
    sig = _get_mesh_topology_signature(mesh)
    n_verts = len(mesh.vertices)

    has_cache = "_taremin_rest_positions" in obj
    is_deformed = obj.get("_taremin_is_deformed", False)

    # シミュレーションによる変形が起きている最中は、force=Trueであっても変形座標でレストポーズを汚染しないよう上書きを拒否
    if is_deformed:
        logger.debug(f"[Cache] Refusing to cache rest positions for deformed object '{obj.name}' (is_deformed=True)")
        return

    cached_sig = None
    if "_taremin_rest_signature" in obj:
        try:
            cached_sig = tuple(obj["_taremin_rest_signature"])
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
        if "_taremin_rest_positions" in obj:
            del obj["_taremin_rest_positions"]
        obj["_taremin_rest_positions"] = coords.tobytes()
        obj["_taremin_rest_signature"] = list(sig)
        obj["_taremin_is_deformed"] = False

        logger.info(f"[Cache] Cached rest positions for '{obj.name}' ({n_verts} verts, sig={sig})")
    else:
        logger.debug(f"[Cache] Rest positions update skipped for '{obj.name}'")


def restore_rest_positions(obj, clear=False):
    """メッシュの頂点座標・トポロジーをレストポーズ（シミュレーション前の初期状態）に安全に復元する"""
    if not obj or obj.type != 'MESH':
        return

    logger.info(f"[Reset] restore_rest_positions called: obj='{obj.name}', clear={clear}")
    clear_simulator_for_object(obj.name)

    # 1. 十字分割オブジェクトであり、メッシュが編集されていない場合はバックアップからQuad復元
    restore_fn = getattr(topology, "restore_pre_subdivision_mesh", None)
    if restore_fn and restore_fn(obj):
        logger.info(f"[Reset] Successfully restored topology & coordinates from pre-subdivision backup for '{obj.name}'")
        cache_rest_positions(obj, force=True)
        if clear:
            clear_backup_fn = getattr(topology, "clear_pre_subdivision_backup", None)
            if clear_backup_fn:
                clear_backup_fn(obj)
            if "_taremin_rest_positions" in obj:
                del obj["_taremin_rest_positions"]
            if "_taremin_rest_signature" in obj:
                del obj["_taremin_rest_signature"]
            if "_taremin_is_deformed" in obj:
                del obj["_taremin_is_deformed"]
        else:
            obj["_taremin_is_deformed"] = False
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
    if "_taremin_rest_positions" in obj:
        initial_coords = np.frombuffer(obj["_taremin_rest_positions"], dtype=np.float32)
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
            if not obj.get("_taremin_is_deformed", False):
                cache_rest_positions(obj, force=True)

    if clear:
        clear_backup_fn = getattr(topology, "clear_pre_subdivision_backup", None)
        if clear_backup_fn:
            clear_backup_fn(obj)
        if "_taremin_rest_positions" in obj:
            del obj["_taremin_rest_positions"]
        if "_taremin_rest_signature" in obj:
            del obj["_taremin_rest_signature"]
        if "_taremin_is_deformed" in obj:
            del obj["_taremin_is_deformed"]
        logger.info(f"[Reset] Cleared all rest caches for '{obj.name}'")
    else:
        obj["_taremin_is_deformed"] = False
        obj.update_tag()


def clear_simulator_for_object(obj_name):
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
    _timeline_frame_cache.pop(obj_name, None)
    _buffered_start_frames.pop(obj_name, None)
    if had_sim:
        logger.debug(f"[Simulator] Cleared simulator and caches for '{obj_name}'")


def clear_simulators():
    """全シミュレータインスタンスおよび各種キャッシュを破棄する"""
    from .collider import clear_collider_cache
    from .runner import restore_fast_playback
    global _simulators, _prev_coords_cache
    global _mesh_char_len_cache, _effective_substeps_cache, _collider_prev_locs_cache
    global _prev_elastic_scales, _timeline_frame_cache, _buffered_start_frames
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
    logger.debug(f"[Simulator] Cleared all {count} simulators and caches")
