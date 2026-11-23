"""
taremin_cloth シミュレーション実行・ステップ制御モジュール
ClothSimulatorのライフサイクル生成、CFL適応サブステップ計算、
タイムラインハンドラー、および高速再生Depsgraph最適化を管理する。
"""

import bpy
import numpy as np
from typing import Optional
from .cache import (
    _simulators,
    _simulator_topology_signatures,
    _timeline_frame_cache,
    _timeline_curve_cache,
    _buffered_start_frames,
    _waypoint_managers,
    set_seam_cluster_map,
    get_seam_cluster_map,
    cache_rest_positions,
    restore_rest_positions,
    clear_simulator_for_object,
    clear_simulators,
    clear_timeline_cache,
    check_and_update_params_signature,
)
from .collider import sync_colliders, clear_collider_cache
from .params import sync_cloth_parameters, sync_attachment_pins
from .simconfig import resolve_self_collision_algorithm
from .profiling import format_profile_table
from ..utils import anim_driver, allow_positional
from ..utils.logger import logger
from ..utils.mesh_extract import extract_cloth_mesh_data

_fast_playback_saved_mods = {}
_is_baking = False


def apply_fast_playback(scene, *, force: bool = False, include_sim_objs: bool = False):
    """シミュレーションに関係のないオブジェクトの重いモディファイアを一時無効化してDepsgraphを軽量化"""
    global _fast_playback_saved_mods
    if not force and not getattr(scene, "taremin_cloth_fast_playback", False):
        return

    sim_objs = set()
    collider_objs = set()
    for obj in scene.objects:
        if getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth:
            sim_objs.add(obj)
        if getattr(obj, "taremin_cloth_collider", None) and obj.taremin_cloth_collider.is_collider:
            sim_objs.add(obj)
            collider_objs.add(obj)

    # 形状デフォーマー（LATTICE等）は除外。純粋なトポロジー増殖・重い描画系のみを対象とする
    heavy_types = {'SUBSURF', 'SOLIDIFY', 'DATA_TRANSFER', 'WEIGHTED_NORMAL', 'NODES', 'WELD', 'BEVEL'}

    for obj in scene.objects:
        # コライダーは形状維持が必須のため常に保護（除外）する
        if obj in collider_objs:
            continue
        if obj.type == 'MESH' and (include_sim_objs or obj not in sim_objs) and obj.modifiers:
            for mod in obj.modifiers:
                if mod.type in heavy_types and mod.show_viewport:
                    key = (obj.name, mod.name)
                    if key not in _fast_playback_saved_mods:
                        _fast_playback_saved_mods[key] = mod.show_viewport
                        mod.show_viewport = False


def restore_fast_playback(scene=None):
    """高速プレビューで無効化したモディファイアを元の状態に復元"""
    global _fast_playback_saved_mods
    if not _fast_playback_saved_mods:
        return
    sc = scene or getattr(bpy.context, "scene", None)
    if not sc:
        return
    for (obj_name, mod_name), orig_val in list(_fast_playback_saved_mods.items()):
        obj = sc.objects.get(obj_name)
        if obj:
            mod = obj.modifiers.get(mod_name)
            if mod:
                mod.show_viewport = orig_val
    _fast_playback_saved_mods.clear()


def begin_simulator_init(obj):
    """シミュレータ初期化の前半: メッシュ抽出と生成パラメータの準備を行う (TIMER分割用)。

    Returns:
        dict: 初期化中間状態。既存インスタンスがある場合は {"cached": True} のみ。
    """
    if obj.name in _simulators:
        return {"cached": True}

    settings = obj.taremin_cloth
    cloth_data = extract_cloth_mesh_data(obj, settings=settings)
    pos_2d = cloth_data.positions
    coords = pos_2d.flatten().copy()
    n_verts = len(pos_2d)

    wg_size = int(getattr(settings, "workgroup_size", "32"))
    s_mode = 1 if getattr(settings, "solver_mode", "COLORING") == 'ATOMIC' else 0
    compact_rb = getattr(settings, "enable_compact_readback", True)

    is_deformed = obj.get("_taremin_cloth_is_deformed", False)
    rest_pos_2d = None
    n_mesh_verts = len(obj.data.vertices)
    # レスト頂点座標の取得 (未変形キャッシュがあれば優先して使用)
    if is_deformed and "_taremin_cloth_rest_positions" in obj:
        raw_rest = obj["_taremin_cloth_rest_positions"]
        if len(raw_rest) == n_mesh_verts * 3 * 4:
            cloth_rest_2d = np.frombuffer(raw_rest, dtype=np.float32).reshape((n_mesh_verts, 3)).copy()
            if hasattr(obj, "matrix_world"):
                try:
                    mat = np.array(obj.matrix_world, dtype=np.float32)
                    if mat.ndim == 2 and mat.shape == (4, 4):
                        cloth_rest_2d = (cloth_rest_2d @ mat[:3, :3].T) + mat[:3, 3]
                except (ValueError, TypeError, AttributeError):
                    pass

            if cloth_data.curve_parts:
                rest_blocks = [cloth_rest_2d]
                string_parts = getattr(settings, "string_parts", None) or []
                enabled_sp = [sp for sp in string_parts if getattr(sp, "enabled", True) and getattr(sp, "curve_object", None)]
                for idx, c_part in enumerate(cloth_data.curve_parts):
                    c_obj = enabled_sp[idx].curve_object if idx < len(enabled_sp) else None
                    c_rest_world = None
                    if c_obj is not None and "_taremin_cloth_rest_positions" in c_obj:
                        raw_c = c_obj["_taremin_cloth_rest_positions"]
                        if len(raw_c) == c_part.vertex_count * 3 * 4:
                            c_rest_local = np.frombuffer(raw_c, dtype=np.float32).reshape((c_part.vertex_count, 3)).copy()
                            if hasattr(c_obj, "matrix_world"):
                                try:
                                    c_mat = np.array(c_obj.matrix_world, dtype=np.float32).reshape((4, 4))
                                    c_rest_world = (c_rest_local @ c_mat[:3, :3].T) + c_mat[:3, 3]
                                except Exception:
                                    c_rest_world = c_rest_local
                            else:
                                c_rest_world = c_rest_local
                    if c_rest_world is None:
                        v_s = c_part.vertex_offset
                        v_e = v_s + c_part.vertex_count
                        c_rest_world = pos_2d[v_s:v_e].copy()
                    rest_blocks.append(c_rest_world.astype(np.float32))
                rest_pos_2d = np.vstack(rest_blocks)
            else:
                rest_pos_2d = cloth_rest_2d

    return {
        "cached": False,
        "pos_2d": pos_2d,
        "coords": coords,
        "edges_2d": cloth_data.normal_edges,
        "faces_2d": cloth_data.faces,
        "sew_2d": cloth_data.sewing_edges,
        "inv_masses": cloth_data.inv_masses,
        "layer_ids": cloth_data.layer_ids,
        "n_verts": n_verts,
        "wg_size": wg_size,
        "s_mode": s_mode,
        "compact_rb": compact_rb,
        "rest_pos_2d": rest_pos_2d,
        "sim_init_pos": rest_pos_2d if rest_pos_2d is not None else pos_2d,
        "waypoint_seams": cloth_data.waypoint_seams,
        "seam_cluster_map": cloth_data.seam_cluster_map,
        "thicknesses": cloth_data.thicknesses,
        "curve_parts": cloth_data.curve_parts,
    }


def create_simulator_from_state(obj, state):
    """シミュレータ初期化の中盤: GPU初期化 (ClothSimulator生成 + Cold Resume) を行う"""
    import taremin_cloth_core

    settings = obj.taremin_cloth
    sc_algo, enable_pair_cache = resolve_self_collision_algorithm(settings)
    sim = taremin_cloth_core.ClothSimulator(
        positions=state["sim_init_pos"],
        edges=state["edges_2d"],
        faces=state["faces_2d"],
        inv_masses=state["inv_masses"],
        sewing_springs=state["sew_2d"],
        layer_ids=state.get("layer_ids"),
        thicknesses=state.get("thicknesses"),
        layer_id=settings.layer_id,
        thickness=settings.thickness,
        stiffness=settings.tension_stiffness,
        compression_stiffness=settings.compression_stiffness,
        shear_stiffness=settings.shear_stiffness,
        bending_stiffness=settings.bending_stiffness,
        sewing_shrink_speed=settings.sewing_shrink_speed,
        sewing_stiffness=getattr(settings, "sewing_stiffness", 10000.0),
        enable_sewing_lock=getattr(settings, "enable_sewing_lock", True),
        sewing_lock_distance=getattr(settings, "sewing_lock_distance", 0.02),
        workgroup_size=state["wg_size"],
        solver_mode=state["s_mode"],
        enable_compact_readback=state["compact_rb"],
        enable_pair_cache=enable_pair_cache,
        areal_density=getattr(settings, "areal_density", 0.15),
        enable_coarse_constraints=bool(getattr(settings, "enable_coarse_constraints", False)),
        self_collision_algorithm=sc_algo,
    )

    # Cold Resume: レスト座標で自然長を初期化した後、現在の変形頂点座標をGPUにセットして停止位置から再開
    if state["rest_pos_2d"] is not None:
        sim.set_positions_and_velocities(state["pos_2d"])
        logger.info(f"[Simulator] Cold resume: Restored positions to previous paused state for '{obj.name}'")

    return sim


@allow_positional
def finalize_simulator_init(obj, sim, state, *, scene=None):
    """シミュレータ初期化の後半: 特性長・レスト・コライダー/SDF同期・パラメータ同期・登録を行う"""
    sc = scene if scene is not None else bpy.context.scene
    pos_2d = state["pos_2d"]
    edges_2d = state["edges_2d"]
    faces_2d = state["faces_2d"]
    coords = state["coords"]
    settings = obj.taremin_cloth

    thickness = getattr(settings, "thickness", 0.005)

    if state.get("seam_cluster_map"):
        set_seam_cluster_map(obj.name, state["seam_cluster_map"])

    if state.get("curve_parts"):
        from .cache import set_curve_parts_runtime
        set_curve_parts_runtime(obj.name, state["curve_parts"])

    cache_rest_positions(obj)
    sync_colliders(sim, sc, cloth_obj=obj)
    sync_cloth_parameters(sim, obj, scene=sc)

    if state.get("waypoint_seams"):
        from .waypoint import WaypointManager
        _waypoint_managers[obj.name] = WaypointManager(
            waypoint_seams=state["waypoint_seams"],
            initial_inv_masses=state["inv_masses"],
            merge_dist=max(0.01, float(thickness) * 2.0),
            pass_dist=max(0.015, float(thickness) * 3.0),
        )

    _simulators[obj.name] = (sim, coords)
    mesh = getattr(obj, "data", None)
    if mesh is not None and getattr(mesh, "vertices", None) is not None:
        edges_attr = getattr(mesh, "edges", ())
        polys_attr = getattr(mesh, "polygons", ())
        _simulator_topology_signatures[obj.name] = (
            len(mesh.vertices),
            len(edges_attr) if edges_attr is not None else 0,
            len(polys_attr) if polys_attr is not None else 0,
            int(state.get("n_verts", len(coords) // 3)),
        )
    n_faces = len(faces_2d) if faces_2d is not None else 0
    logger.debug(f"[Simulator] Initialized ClothSimulator for '{obj.name}' (verts={len(pos_2d)}, faces={n_faces})")
    return sim, coords


def get_or_create_simulator(obj):
    """オブジェクトに対応するClothSimulatorインスタンスを取得または生成する"""
    if obj.name in _simulators:
        expected_sig = _simulator_topology_signatures.get(obj.name)
        mesh = getattr(obj, "data", None)
        if expected_sig is not None and mesh is not None and getattr(mesh, "vertices", None) is not None:
            edges_attr = getattr(mesh, "edges", ())
            polys_attr = getattr(mesh, "polygons", ())
            cur_topo = (
                len(mesh.vertices),
                len(edges_attr) if edges_attr is not None else 0,
                len(polys_attr) if polys_attr is not None else 0,
            )
            if cur_topo != expected_sig[:3]:
                logger.info(
                    f"[Simulator] Mesh topology changed for '{obj.name}' "
                    f"({expected_sig[:3]} -> {cur_topo}). Recreating simulator."
                )
                clear_simulator_for_object(obj.name)
        if obj.name in _simulators:
            return _simulators[obj.name]

    state = begin_simulator_init(obj)
    sim = create_simulator_from_state(obj, state)
    return finalize_simulator_init(obj, sim, state)


def is_virtual_sampling_clamped(obj_name: str) -> bool:
    """指定オブジェクトの仮想頂点サンプリングが安全上限で制限されたかを返す"""
    if obj_name in _simulators:
        sim, _ = _simulators[obj_name]
        if hasattr(sim, "is_virtual_sampling_clamped"):
            return bool(sim.is_virtual_sampling_clamped())
    return False


def get_seam_partners(obj_name: str, vert_idx: int) -> list[int]:
    """オブジェクトの縫合クラスタマップから指定頂点のパートナー頂点一覧を取得する"""
    cluster_map = get_seam_cluster_map(obj_name)
    return cluster_map.get(int(vert_idx), [])


def get_seam_merge_threshold(obj) -> float:
    """オブジェクトの縫合密着距離および厚みに基づくシーム結合判定閾値を算出する。

    縫合前の型紙エッジ同士（10cm以上離れている）は未結合として単独操作を許可し、
    縫合後のシーム（数ミリ以下）は結合済みとして一体同期操作を許可する。
    """
    settings = getattr(obj, "taremin_cloth", None)
    if settings:
        try:
            lock_dist = float(getattr(settings, "sewing_lock_distance", 0.005))
            thickness = float(getattr(settings, "thickness", 0.005))
            return max(lock_dist * 2.0, thickness * 2.0, 0.02)
        except (TypeError, ValueError):
            return 0.02
    return 0.02


@allow_positional
def get_connected_seam_partners(
    obj_name: str,
    vert_idx: int,
    coords: np.ndarray,
    threshold: float = 0.02,
) -> list[int]:
    """指定頂点の縫合パートナーのうち、現在位置が密着閾値以内の結合済み頂点一覧を取得する。

    未結合で離れている型紙エッジを単独で操作（手動引き寄せ）できるようにし、
    結合済みのシームのみを一体として同期グラブ可能にする。
    """
    cluster_map = get_seam_cluster_map(obj_name)
    partners = cluster_map.get(int(vert_idx), [])
    if not partners or vert_idx >= len(coords):
        return []

    p_v = coords[vert_idx]
    connected = []
    for p in partners:
        if p < len(coords):
            d = float(np.linalg.norm(coords[p] - p_v))
            if d <= threshold:
                connected.append(p)
    return connected


def _gravity_active(settings, *, scene=None) -> bool:
    """重力荷重が有効か (適応サブステップ下限の判定用)。"""
    try:
        scale = abs(float(getattr(settings, "gravity", 1.0)))
    except (TypeError, ValueError):
        scale = 1.0
    if scale <= 1e-3:
        return False
    if scene is not None and not bool(getattr(scene, "use_gravity", True)):
        return False
    return True


@allow_positional
def get_effective_substeps(obj, coords=None, dt: float = 1.0 / 60.0, scene=None):
    """布およびコライダーの運動速度とCFL条件に基づき、最適なサブステップ数を動的に算出する。

    物理コア (cloth_core) への移管に伴い、コアから実効サブステップ数を取得するか、
    フォールバックとしての基本値を返却する。
    """
    settings = getattr(obj, "taremin_cloth", None)
    base_steps = getattr(settings, "substeps", 20) if settings else 20
    if not settings or not getattr(settings, "enable_adaptive_substep", False):
        return base_steps

    entry = _simulators.get(obj.name) if hasattr(obj, "name") else None
    sim = entry[0] if isinstance(entry, tuple) else entry
    if sim is not None and hasattr(sim, "get_effective_substeps"):
        eff = sim.get_effective_substeps()
        return eff if eff > 0 else base_steps

    return base_steps


_collider_armature_pose_cache = {}


def clear_collider_armature_cache(obj_name=None):
    """アーマチュアポーズ比較キャッシュを破棄する"""
    global _collider_armature_pose_cache
    if obj_name:
        _collider_armature_pose_cache.pop(obj_name, None)
    else:
        _collider_armature_pose_cache.clear()


def _has_collider_armature_moved(col_o) -> bool:
    """コライダーに紐づくアーマチュアのポーズ行列またはワールド行列が前フレームから変化したかを判定する"""
    global _collider_armature_pose_cache
    arm_o = col_o.find_armature() if hasattr(col_o, "find_armature") else None
    if arm_o is None:
        for mod in getattr(col_o, "modifiers", []):
            if getattr(mod, "type", None) == 'ARMATURE' and getattr(mod, "object", None):
                arm_o = mod.object
                break

    # アーマチュアがない場合はオブジェクトレベルのアニメーション有無をチェック
    if arm_o is None or not getattr(arm_o, "pose", None):
        return bool(col_o.animation_data and col_o.animation_data.action)

    # オブジェクト行列および全ボーン行列の全16成分をシグネチャ化（回転・移動・スケールを100%検知）
    sig_parts = []
    w_mat = getattr(col_o, "matrix_world", None)
    if w_mat is not None:
        for r in range(4):
            for c in range(4):
                sig_parts.append(round(float(w_mat[r][c]), 4))

    arm_mat = getattr(arm_o, "matrix_world", None)
    if arm_mat is not None:
        for r in range(4):
            for c in range(4):
                sig_parts.append(round(float(arm_mat[r][c]), 4))

    for pb in arm_o.pose.bones:
        m = pb.matrix
        for r in range(4):
            for c in range(4):
                sig_parts.append(round(float(m[r][c]), 4))

    sig = tuple(sig_parts)
    key = getattr(col_o, "name", "")
    prev_sig = _collider_armature_pose_cache.get(key)
    if prev_sig != sig:
        _collider_armature_pose_cache[key] = sig
        return prev_sig is not None
    return False


def _to_local_coords_2d(target_obj, world_coords_2d: np.ndarray) -> np.ndarray:
    """ワールド座標配列 [N, 3] を対象オブジェクトのローカル座標配列 [N, 3] へ変換する"""
    if target_obj is not None and hasattr(target_obj, "matrix_world"):
        try:
            mat_world = target_obj.matrix_world
            if hasattr(mat_world, "inverted"):
                inv_mat = np.array([list(row) for row in mat_world.inverted()], dtype=np.float32)
            else:
                inv_mat = np.linalg.inv(np.array(mat_world, dtype=np.float32).reshape((4, 4)))
            if inv_mat.ndim == 2 and inv_mat.shape == (4, 4):
                return ((world_coords_2d @ inv_mat[:3, :3].T) + inv_mat[:3, 3]).astype(np.float32)
        except (ValueError, TypeError, AttributeError, np.linalg.LinAlgError):
            pass
    return np.asarray(world_coords_2d, dtype=np.float32).copy()


def _apply_coords_to_mesh(obj, coords):
    """計算済み座標配列をBlenderメッシュ頂点および紐Curveに適用する（単位行列時はゼロコピー直書き）。"""
    n_mesh_verts = len(obj.data.vertices)
    if len(coords) < n_mesh_verts * 3:
        logger.warning(
            f"[Runner] Vertex count mismatch in _apply_coords_to_mesh for '{obj.name}': "
            f"coords_len={len(coords)}, expected>={n_mesh_verts * 3}"
        )
        return
    total_verts = len(coords) // 3
    mesh_coords = coords[:n_mesh_verts * 3]

    pos_local = None
    if hasattr(obj, "matrix_world"):
        try:
            mat = obj.matrix_world
            is_identity = True
            for r in range(4):
                for c in range(4):
                    expected = 1.0 if r == c else 0.0
                    if abs(mat[r][c] - expected) > 1e-5:
                        is_identity = False
                        break
                if not is_identity:
                    break

            if not is_identity:
                inv_mat = np.array([list(row) for row in mat.inverted()], dtype=np.float32)
                if inv_mat.ndim == 2 and inv_mat.shape == (4, 4):
                    pos_world = mesh_coords.reshape(-1, 3)
                    pos_local = (pos_world @ inv_mat[:3, :3].T) + inv_mat[:3, 3]
        except (ValueError, TypeError, AttributeError):
            pos_local = None
    if pos_local is not None:
        obj.data.vertices.foreach_set("co", pos_local.flatten())
    else:
        obj.data.vertices.foreach_set("co", mesh_coords)
    obj.data.update()
    obj["_taremin_cloth_is_deformed"] = True

    # 紐（Curve）オブジェクトへの書き戻し
    if total_verts > n_mesh_verts:
        from .cache import get_curve_parts_runtime
        from ..utils.curve_extract import apply_coords_to_curve

        curve_parts = get_curve_parts_runtime(obj.name)
        coords_2d = coords.reshape(-1, 3)
        for c_part in curve_parts:
            v_start = c_part.vertex_offset
            v_end = v_start + c_part.vertex_count
            if v_end <= total_verts:
                c_world_coords = coords_2d[v_start:v_end]
                curve_obj = bpy.data.objects.get(c_part.curve_obj_name)
                if curve_obj:
                    apply_coords_to_curve(curve_obj, c_world_coords)


def step_cloth_object(
    obj,
    scene,
    *,
    dt: float = 1.0 / 60.0,
    depsgraph=None,
    force_sync_colliders: bool = False,
    update_mesh: bool = True,
    substeps: Optional[int] = None,
    solver_iterations: Optional[int] = None,
    skip_sync: bool = False,
):
    """単一の布オブジェクトに対し、コライダー同期・パラメータ同期・物理ステップ・メッシュ更新を実行する。

    Args:
        obj: Blenderメッシュオブジェクト
        scene: 対象Blenderシーン
        dt: 時間刻み (秒)
        depsgraph: 評価済みDepsgraph (Noneの場合は自動取得)
        force_sync_colliders: コライダー変形時などに強制再同期するか
        update_mesh: 物理ステップ後にCPU頂点座標を取得してメッシュへ反映するか
        substeps: サブステップ数 (Noneの場合は get_effective_substeps で動的算出)
        solver_iterations: ソルバー反復回数 (Noneの場合は settings.solver_iterations)
        skip_sync: 同期処理をスキップして物理ステップのみ進めるか (同一フレーム内の複数ステップ実行用)

    Returns:
        tuple[taremin_cloth_core.ClothSimulator, np.ndarray]: (sim, coords)
    """
    sim, coords = get_or_create_simulator(obj)
    settings = getattr(obj, "taremin_cloth", None)

    from ..preferences import is_async_readback_experimental_enabled
    use_async = (
        not _is_baking
        and is_async_readback_experimental_enabled(scene)
        and settings is not None
        and bool(getattr(settings, "enable_async_readback", False))
        and hasattr(sim, "step_async")
        and hasattr(sim, "fetch_positions")
    )

    # 非同期モード: 前フレームのGPU計算結果を回収してメッシュに反映
    if use_async:
        if sim.fetch_positions(coords):
            if update_mesh:
                _apply_coords_to_mesh(obj, coords)
            wp_mgr = _waypoint_managers.get(obj.name)
            if wp_mgr and wp_mgr.has_active_seams:
                wp_mgr.update(coords, sim=sim)

    if not skip_sync:
        if depsgraph is None:
            try:
                depsgraph = bpy.context.evaluated_depsgraph_get()
            except Exception:
                depsgraph = None

        sync_colliders(sim, scene, depsgraph=depsgraph, force=force_sync_colliders, cloth_obj=obj)
        sync_cloth_parameters(sim, obj, scene=scene)
        sync_attachment_pins(sim, obj, scene=scene)

    # substeps未指定(None)時は、非適応ならsettings.substepsを使用し、適応サブステップ有効時はコア側に判定委譲(None)
    if substeps is not None:
        actual_substeps = substeps
    elif settings is not None:
        actual_substeps = None if bool(getattr(settings, "enable_adaptive_substep", False)) else getattr(settings, "substeps", None)
    else:
        actual_substeps = None
    actual_iters = solver_iterations if solver_iterations is not None else (getattr(settings, "solver_iterations", 1) if settings else 1)

    # 縫合優先モード: 直近座標で結合率を測定し、step 前にラッチ状態を更新する。
    # 追加のGPU読戻しなし (coords は前ステップ確定値のキャッシュ)。
    if settings is not None and bool(getattr(settings, "enable_sewing_priority", False)):
        updater = getattr(sim, "update_sewing_priority", None)
        if callable(updater):
            try:
                pos_2d = np.ascontiguousarray(coords.reshape(-1, 3), dtype=np.float32)
                updater(pos_2d)
            except Exception:
                pass

    if use_async:
        sim.step_async(dt=dt, substeps=actual_substeps, solver_iterations=actual_iters)
    else:
        sim.step(dt=dt, substeps=actual_substeps, solver_iterations=actual_iters)
        sim.get_positions(coords)
        if update_mesh:
            _apply_coords_to_mesh(obj, coords)

        # 中継点縫合の動的通過＆結合完了時ピン解放判定
        wp_mgr = _waypoint_managers.get(obj.name)
        if wp_mgr and wp_mgr.has_active_seams:
            wp_mgr.update(coords, sim=sim)

    return sim, coords


def step_cloth_scene(
    scene,
    *,
    dt: float = 1.0 / 60.0,
    anim_frame: Optional[int] = None,
    depsgraph=None,
    target_objs: Optional[list] = None,
    update_mesh: bool = True,
    steps_per_frame: int = 1,
) -> dict:
    """シーン内のコライダーアニメーションを更新し、対象布オブジェクト群のシミュレーションを1フレーム進める。

    Args:
        scene: 対象Blenderシーン
        dt: 1ステップの時間刻み (秒)
        anim_frame: コライダーアニメーションのフレーム番号 (Noneの場合は scene.frame_current)
        depsgraph: 評価済みDepsgraph (Noneの場合は自動評価)
        target_objs: 対象布オブジェクトのリスト (Noneの場合はシーン内の全有効布オブジェクト)
        update_mesh: 物理ステップ後にメッシュ頂点を更新するか
        steps_per_frame: 1フレーム内に進めるシミュレーションステップ数 (デフォルト1)

    Returns:
        dict[str, tuple]: {obj.name: (sim, coords)} の辞書
    """
    # Contextオブジェクトが渡された場合の防御的解決
    if hasattr(scene, "scene") and scene.scene is not None:
        scene = scene.scene

    # 1. コライダーのアニメーション駆動ステップ
    any_collider_deformed = False
    current_frame = anim_frame if anim_frame is not None else getattr(scene, "frame_current", 1)

    for col_o in scene.objects:
        c_set = getattr(col_o, "taremin_cloth_collider", None)
        if c_set and c_set.is_collider and getattr(c_set, "enabled", True):
            if getattr(c_set, "anim", None) and c_set.anim.enabled:
                _, deformed = anim_driver.step_collider_animation(col_o, current_frame)
                if deformed:
                    any_collider_deformed = True
            elif col_o.type == 'MESH' and (c_set.collider_type == 'MESH' or (c_set.collider_type == 'BONE_SDF' and getattr(c_set, "enable_joint_mesh", True))):
                if _has_collider_armature_moved(col_o):
                    any_collider_deformed = True

    # 2. 必要に応じてDepsgraphを再評価
    if depsgraph is None:
        try:
            if any_collider_deformed and hasattr(bpy.context, "view_layer") and bpy.context.view_layer:
                bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
        except Exception:
            depsgraph = None

    # 3. 対象布オブジェクトの決定
    if target_objs is not None:
        objs = [o for o in target_objs if o.type == 'MESH' and hasattr(o, "taremin_cloth") and o.taremin_cloth.is_cloth and getattr(o.taremin_cloth, "enabled", True)]
    else:
        objs = [o for o in scene.objects if o.type == 'MESH' and hasattr(o, "taremin_cloth") and o.taremin_cloth.is_cloth and getattr(o.taremin_cloth, "enabled", True)]

    results = {}
    for obj in objs:
        sim = None
        coords = None
        for s_idx in range(steps_per_frame):
            do_update = update_mesh and (s_idx == steps_per_frame - 1)
            skip_sync = (s_idx > 0)
            sim, coords = step_cloth_object(
                obj,
                scene,
                dt=dt,
                depsgraph=depsgraph,
                force_sync_colliders=any_collider_deformed,
                update_mesh=do_update,
                skip_sync=skip_sync,
            )
        if sim is not None and coords is not None:
            results[obj.name] = (sim, coords)

    return results


def set_profiling_enabled_all(*, enable: bool = True) -> dict:
    """登録済み全シミュレータのGPU計測を切替える。{obj名: 可否} を返す。"""
    out = {}
    for name, (sim, _coords) in list(_simulators.items()):
        try:
            out[name] = bool(sim.set_profiling_enabled(enable))
        except Exception:
            out[name] = False
    return out


def take_profiles() -> dict:
    """登録済み全シミュレータの直近プロファイルを回収する。{obj名: [(label, ms)]}。"""
    out = {}
    for name, (sim, _coords) in list(_simulators.items()):
        try:
            out[name] = list(sim.take_profile())
        except Exception:
            out[name] = []
    return out


def format_profiles(profiles: dict) -> str:
    """`take_profiles` 結果を可読表にする。"""
    lines = []
    for name, entries in profiles.items():
        lines.append(f"== {name} ==")
        lines.append(format_profile_table(entries))
    return "\n".join(lines) if lines else "profile: no simulators"


def save_profile_traces(directory: str = "scratch") -> dict:
    """登録済み全シミュレータのtraceを保存する。{obj名: パス} を返す。"""
    import os

    os.makedirs(directory, exist_ok=True)
    out = {}
    for name, (sim, _coords) in list(_simulators.items()):
        path = os.path.join(directory, f"profile_{name}.json")
        try:
            sim.save_profile_trace(path)
            out[name] = path
        except Exception:
            continue
    return out


def flush_async_simulation(obj, *, update_mesh: bool = True) -> bool:
    """指定オブジェクトの未回収非同期シミュレーション結果を待機・回収してメッシュに反映する。"""
    if obj.name not in _simulators:
        return False
    sim, coords = _simulators[obj.name]
    if hasattr(sim, "fetch_positions"):
        ok = sim.fetch_positions(coords)
        if ok and update_mesh:
            _apply_coords_to_mesh(obj, coords)
        return ok
    return False


def flush_all_async_simulations(*, update_mesh: bool = True) -> int:
    """全登録シミュレータの未回収非同期シミュレーション結果をフラッシュする。回収数を返す。"""
    count = 0
    for name, (sim, coords) in list(_simulators.items()):
        obj = bpy.data.objects.get(name) if hasattr(bpy, "data") else None
        if hasattr(sim, "fetch_positions"):
            ok = sim.fetch_positions(coords)
            if ok:
                count += 1
                if update_mesh and obj is not None:
                    _apply_coords_to_mesh(obj, coords)
    return count


def _persistent(func):
    """Blender環境では@persistentを適用し、スタンドアロンテスト環境では透過するデコレータ"""
    app = getattr(bpy, "app", None)
    if not isinstance(getattr(app, "version", None), tuple):
        return func
    handlers = getattr(app, "handlers", None)
    persistent_fn = getattr(handlers, "persistent", None)
    if callable(persistent_fn):
        return persistent_fn(func)
    return func


def _restore_cached_curves_for_frame(obj_name: str, frame: int) -> None:
    """タイムラインキャッシュに保存された紐カーブ制御点を復元する"""
    curve_frame_map = _timeline_curve_cache.get(obj_name)
    if not curve_frame_map or frame not in curve_frame_map:
        return
    curve_entries = curve_frame_map[frame]
    if not curve_entries:
        return
    from ..utils.curve_extract import apply_coords_to_curve
    data_objs = getattr(getattr(bpy, "data", None), "objects", None)
    for c_name, c_local in curve_entries:
        curve_obj = data_objs.get(c_name) if data_objs is not None else None
        if curve_obj is not None:
            apply_coords_to_curve(curve_obj, c_local, is_world_coords=False)


@_persistent
def cloth_frame_handler(scene):
    """Blenderタイムライン進行時のハンドラー（ベイクキャッシュ参照専用）"""
    global _is_baking
    if _is_baking:
        return

    current_frame = scene.frame_current

    # 各布オブジェクトのパラメータ変更チェック (ベイク済みの場合は変更があればキャッシュ無効化)
    for obj in scene.objects:
        if obj.type == 'MESH' and hasattr(obj, "taremin_cloth") and obj.taremin_cloth.is_cloth and getattr(obj.taremin_cloth, "enabled", True):
            check_and_update_params_signature(obj, scene=scene)

    # 各布オブジェクトのキャッシュ参照およびメッシュ適用
    for obj in scene.objects:
        if obj.type == 'MESH' and hasattr(obj, "taremin_cloth") and obj.taremin_cloth.is_cloth and getattr(obj.taremin_cloth, "enabled", True):
            obj_cache = _timeline_frame_cache.get(obj.name, {})
            n_v = len(obj.data.vertices)

            # 1. キャッシュヒット判定 (ベイク済みフレームが存在する場合: O(1)でメッシュ反映)
            if current_frame in obj_cache and len(obj_cache[current_frame]) == n_v * 3:
                cached_coords = obj_cache[current_frame]
                obj.data.vertices.foreach_set("co", cached_coords)
                obj.data.update()
                _restore_cached_curves_for_frame(obj.name, current_frame)
                obj["_taremin_cloth_is_deformed"] = (current_frame != scene.frame_start)
            else:
                # 2. 未ベイク時またはキャッシュ範囲外:
                # シミュレーション変形中であれば初期レスト形状へ安全に戻す
                if obj.get("_taremin_cloth_is_deformed", False):
                    restore_rest_positions(obj, clear_timeline=False)


@_persistent
def on_load_pre(_dummy=None):
    """シーン再読み込み・新規ファイル前にインタラクティブモード停止と全GPU/描画キャッシュを破棄する"""
    try:
        from ..ops.interactive import stop_interactive_if_running
        stop_interactive_if_running()
    except Exception:
        pass
    try:
        clear_simulators()
    except Exception as e:
        logger.warning(f"[Lifecycle] on_load_pre clear_simulators failed: {e}")
    try:
        from ..utils import drawing
        drawing.clear_overlay_caches()
    except Exception:
        pass


@_persistent
def on_undo_post(_dummy=None):
    """Undo/Redo 実行後に無効化されたオブジェクト参照やトポロジー不整合キャッシュを破棄する"""
    try:
        clear_collider_cache()
    except Exception:
        pass
    try:
        from ..utils import drawing
        drawing.clear_overlay_caches()
    except Exception:
        pass
    try:
        data_objs = getattr(getattr(bpy, "data", None), "objects", None)
        for obj_name in list(_simulators.keys()):
            obj = data_objs.get(obj_name) if data_objs is not None else None
            if obj is None:
                clear_simulator_for_object(obj_name)
                continue
            expected_sig = _simulator_topology_signatures.get(obj_name)
            mesh = getattr(obj, "data", None)
            if expected_sig is not None and mesh is not None and getattr(mesh, "vertices", None) is not None:
                edges_attr = getattr(mesh, "edges", ())
                polys_attr = getattr(mesh, "polygons", ())
                cur_topo = (
                    len(mesh.vertices),
                    len(edges_attr) if edges_attr is not None else 0,
                    len(polys_attr) if polys_attr is not None else 0,
                )
                if cur_topo != expected_sig[:3]:
                    clear_simulator_for_object(obj_name)
    except Exception as e:
        logger.warning(f"[Lifecycle] on_undo_post cleanup failed: {e}")


def bake_init_simulation(scene, *, start_frame=None, end_frame=None):
    """ベイクの準備処理（初期化、シミュレータクリア、F1キャッシュ格納）を行い、コンテキスト辞書を返す"""
    if not scene:
        return None

    start_f = int(start_frame if start_frame is not None else scene.frame_start)
    end_f = int(end_frame if end_frame is not None else scene.frame_end)
    if end_f < start_f:
        end_f = start_f

    cloth_objs = [
        o for o in scene.objects
        if o.type == 'MESH' and getattr(o, "taremin_cloth", None) and o.taremin_cloth.is_cloth and getattr(o.taremin_cloth, "enabled", True)
    ]
    if not cloth_objs:
        logger.warning("[Bake] No active cloth objects found in scene.")
        return None

    # 1. 準備: 全布の既存キャッシュ・シミュレータを破棄し、初期レストポーズを設定
    clear_simulators()
    for obj in cloth_objs:
        if obj.get("_taremin_cloth_is_deformed", False):
            restore_rest_positions(obj, clear=False, clear_timeline=True)
        else:
            cache_rest_positions(obj, force=True)
        clear_timeline_cache(obj.name)
        obj_cache = _timeline_frame_cache.setdefault(obj.name, {})
        n_v = len(obj.data.vertices)
        init_co = np.empty(n_v * 3, dtype=np.float32)
        obj.data.vertices.foreach_get("co", init_co)
        obj_cache[start_f] = init_co.copy()

        string_parts = getattr(getattr(obj, "taremin_cloth", None), "string_parts", None)
        if string_parts:
            from ..utils.curve_extract import extract_curve_polyline
            init_curves = []
            for sp in string_parts:
                if not getattr(sp, "enabled", True):
                    continue
                curve_obj = getattr(sp, "curve_object", None)
                if curve_obj is not None:
                    c_local, _, _ = extract_curve_polyline(curve_obj, apply_world_matrix=False)
                    if len(c_local) > 0:
                        init_curves.append((curve_obj.name, c_local.astype(np.float32)))
            if init_curves:
                _timeline_curve_cache.setdefault(obj.name, {})[start_f] = init_curves

        obj["_taremin_cloth_is_deformed"] = False

    fps = scene.render.fps
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    total_steps = end_f - start_f

    # タイムラインを開始フレームに移動
    scene.frame_set(start_f)

    logger.info(f"[Bake] Starting simulation bake for {len(cloth_objs)} cloth objects: frames {start_f} to {end_f} (total {total_steps + 1}f)")

    return {
        "scene": scene,
        "cloth_objs": cloth_objs,
        "start_frame": start_f,
        "end_frame": end_f,
        "total_steps": total_steps,
        "fps": fps,
        "dt": dt,
        "baked_count": 1,
    }


@allow_positional
def _store_baked_frame_coords(obj, frame: int, coords: np.ndarray) -> None:
    """1フレーム分の計算済みワールド座標をローカル座標へ変換してタイムラインキャッシュに格納する"""
    n_mesh_verts = len(obj.data.vertices)
    mesh_coords = coords[:n_mesh_verts * 3]
    pos_world_mesh = mesh_coords.reshape(-1, 3)
    local_coords = _to_local_coords_2d(obj, pos_world_mesh).flatten()
    _timeline_frame_cache.setdefault(obj.name, {})[frame] = local_coords
    obj["_taremin_cloth_is_deformed"] = True

    total_verts = len(coords) // 3
    if total_verts > n_mesh_verts:
        from .cache import get_curve_parts_runtime
        curve_parts = get_curve_parts_runtime(obj.name)
        if curve_parts:
            coords_2d = coords.reshape(-1, 3)
            frame_curves = []
            data_objs = getattr(getattr(bpy, "data", None), "objects", None)
            for c_part in curve_parts:
                v_start = c_part.vertex_offset
                v_end = v_start + c_part.vertex_count
                if v_end <= total_verts:
                    c_world = coords_2d[v_start:v_end]
                    c_obj = data_objs.get(c_part.curve_obj_name) if data_objs is not None else None
                    c_local = _to_local_coords_2d(c_obj, c_world) if c_obj is not None else c_world.copy()
                    frame_curves.append((c_part.curve_obj_name, c_local))
            if frame_curves:
                _timeline_curve_cache.setdefault(obj.name, {})[frame] = frame_curves


@allow_positional
def _flush_buffered_bake_for_object(bake_ctx, obj, sim, coords: np.ndarray) -> int:
    """GPUリングバッファに蓄積された複数フレーム座標を一括回収してタイムラインキャッシュへ格納する"""
    buffered_map = bake_ctx.get("buffered_frames") if bake_ctx else None
    if not buffered_map:
        return 0
    pending_frames = buffered_map.get(obj.name)
    if not pending_frames:
        return 0

    n_pending = len(pending_frames)
    total_floats = len(coords)
    if total_floats == 0 or not hasattr(sim, "fetch_buffered_positions"):
        pending_frames.clear()
        _buffered_start_frames.pop(obj.name, None)
        return 0

    batch_buf = np.empty(n_pending * total_floats, dtype=np.float32)
    fetched = int(sim.fetch_buffered_positions(batch_buf))
    actual_n = min(n_pending, fetched)
    for i in range(actual_n):
        f_num = pending_frames[i]
        f_coords = batch_buf[i * total_floats : (i + 1) * total_floats]
        _store_baked_frame_coords(obj, f_num, f_coords)

    if actual_n > 0:
        last_slice = batch_buf[(actual_n - 1) * total_floats : actual_n * total_floats]
        coords[:] = last_slice
        _apply_coords_to_mesh(obj, coords)

    pending_frames.clear()
    _buffered_start_frames.pop(obj.name, None)
    return actual_n


def bake_step_frame(bake_ctx, frame):
    """指定フレームのシミュレーション計算を1ステップ実行し、キャッシュに保存する"""
    scene = bake_ctx["scene"]
    cloth_objs = bake_ctx["cloth_objs"]
    start_f = bake_ctx["start_frame"]
    end_f = bake_ctx["end_frame"]
    dt = bake_ctx["dt"]

    scene.frame_set(frame)

    # コライダーのアニメーション駆動ステップ
    any_collider_deformed = False
    for col_o in scene.objects:
        c_set = getattr(col_o, "taremin_cloth_collider", None)
        if c_set and c_set.is_collider and getattr(c_set, "enabled", True):
            if getattr(c_set, "anim", None) and c_set.anim.enabled:
                anim_f = frame - start_f
                _, deformed = anim_driver.step_collider_animation(col_o, anim_f)
                if deformed:
                    any_collider_deformed = True
            elif col_o.type == 'MESH' and (c_set.collider_type == 'MESH' or (c_set.collider_type == 'BONE_SDF' and getattr(c_set, "enable_joint_mesh", True))):
                if col_o.find_armature() or (col_o.animation_data and col_o.animation_data.action):
                    any_collider_deformed = True

    try:
        if any_collider_deformed:
            bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
    except Exception:
        depsgraph = None

    buffered_map = bake_ctx.setdefault("buffered_frames", {})

    for obj in cloth_objs:
        sim, coords = get_or_create_simulator(obj)
        settings = obj.taremin_cloth

        enable_buf = (
            bool(getattr(settings, "enable_frame_buffering", False))
            and hasattr(sim, "step_buffered")
            and hasattr(sim, "fetch_buffered_positions")
        )
        try:
            buf_size = max(2, min(8, int(getattr(settings, "frame_buffer_size", 2) or 2)))
        except (TypeError, ValueError):
            buf_size = 2

        wp_mgr = _waypoint_managers.get(obj.name)
        has_active_wp = bool(wp_mgr and wp_mgr.has_active_seams)
        sew_prio_unlatched = bool(
            getattr(settings, "enable_sewing_priority", False)
            and hasattr(sim, "is_sewing_priority_active")
            and sim.is_sewing_priority_active()
        )
        needs_immediate_cpu_coords = has_active_wp or sew_prio_unlatched

        if enable_buf and not needs_immediate_cpu_coords:
            sync_colliders(sim, scene, depsgraph=depsgraph, force=any_collider_deformed, cloth_obj=obj)
            sync_cloth_parameters(sim, obj, scene=scene)
            sync_attachment_pins(sim, obj, scene=scene)

            actual_substeps = (
                None
                if bool(getattr(settings, "enable_adaptive_substep", False))
                else getattr(settings, "substeps", None)
            )
            actual_iters = getattr(settings, "solver_iterations", 1)

            pending_list = buffered_map.setdefault(obj.name, [])
            if not pending_list:
                _buffered_start_frames[obj.name] = frame
            sim.step_buffered(dt=dt, substeps=actual_substeps, solver_iterations=actual_iters)
            pending_list.append(frame)

            if len(pending_list) >= buf_size or frame >= end_f:
                _flush_buffered_bake_for_object(bake_ctx, obj, sim, coords)
        else:
            # バッファリング中の未回収フレームがあれば先にフラッシュしてから通常ステップへ移行
            if buffered_map.get(obj.name):
                _flush_buffered_bake_for_object(bake_ctx, obj, sim, coords)

            sim, coords = step_cloth_object(
                obj,
                scene,
                dt=dt,
                depsgraph=depsgraph,
                force_sync_colliders=any_collider_deformed,
                update_mesh=True,
                substeps=None,
                solver_iterations=settings.solver_iterations,
            )
            _store_baked_frame_coords(obj, frame, coords)

    bake_ctx["baked_count"] += 1
    return bake_ctx["baked_count"]


def bake_finalize_simulation(bake_ctx):
    """ベイクの完了・終了処理を行い、タイムラインを開始フレームに戻す"""
    flush_all_async_simulations(update_mesh=False)
    if not bake_ctx:
        return
    cloth_objs = bake_ctx["cloth_objs"]
    buffered_map = bake_ctx.get("buffered_frames") or {}
    for obj in cloth_objs:
        if buffered_map.get(obj.name) and obj.name in _simulators:
            sim, coords = _simulators[obj.name]
            _flush_buffered_bake_for_object(bake_ctx, obj, sim, coords)
    scene = bake_ctx["scene"]
    start_f = bake_ctx["start_frame"]

    scene.frame_set(start_f)
    for obj in cloth_objs:
        init_coords = _timeline_frame_cache.get(obj.name, {}).get(start_f)
        if init_coords is not None:
            obj.data.vertices.foreach_set("co", init_coords)
            obj.data.update()
            _restore_cached_curves_for_frame(obj.name, start_f)
            obj["_taremin_cloth_is_deformed"] = False

    logger.info(f"[Bake] Completed simulation bake: {bake_ctx['baked_count']} frames cached.")


def bake_cloth_simulation(scene, *, start_frame=None, end_frame=None, progress_callback=None, cancel_check=None):
    """シーン内の全Clothシミュレーションを指定フレーム範囲で一括計算してキャッシュ（ベイク）する（同期実行）"""
    global _is_baking
    bake_ctx = bake_init_simulation(scene, start_frame=start_frame, end_frame=end_frame)
    if not bake_ctx:
        return 0

    _is_baking = True
    try:
        start_f = bake_ctx["start_frame"]
        end_f = bake_ctx["end_frame"]
        total_steps = bake_ctx["total_steps"]

        if progress_callback:
            progress_callback(1, total_steps + 1)

        cancelled = False
        for frame in range(start_f + 1, end_f + 1):
            if cancel_check and cancel_check():
                logger.info(f"[Bake] Cancelled by user at frame {frame}")
                cancelled = True
                break
            bake_step_frame(bake_ctx, frame)
            if progress_callback:
                progress_callback(bake_ctx["baked_count"], total_steps + 1)

        if cancelled:
            flush_all_async_simulations(update_mesh=False)
            from .cache import free_scene_bake
            free_scene_bake(scene)
            try:
                scene.frame_set(start_f)
            except Exception:
                pass
            return 0

        bake_finalize_simulation(bake_ctx)
        return bake_ctx["baked_count"]
    finally:
        _is_baking = False

