"""
taremin_cloth シミュレーション実行・ステップ制御モジュール
ClothSimulatorのライフサイクル生成、CFL適応サブステップ計算、
タイムラインハンドラー、および高速再生Depsgraph最適化を管理する。
"""

import bpy
import numpy as np
from .cache import (
    _simulators,
    _prev_coords_cache,
    _mesh_char_len_cache,
    _effective_substeps_cache,
    _collider_prev_locs_cache,
    _timeline_frame_cache,
    _buffered_start_frames,
    cache_rest_positions,
    restore_rest_positions,
    clear_simulators,
    clear_timeline_cache,
    check_and_update_params_signature,
)
from .collider import sync_colliders
from .params import sync_cloth_parameters, sync_attachment_pins
from ..utils import anim_driver
from ..utils.logger import logger
from ..utils.mesh_extract import extract_cloth_mesh_data

_fast_playback_saved_mods = {}
_is_baking = False


def apply_fast_playback(scene, force=False, include_sim_objs=False):
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


def get_or_create_simulator(obj):
    """オブジェクトに対応するClothSimulatorインスタンスを取得または生成する"""
    import taremin_cloth_core

    if obj.name in _simulators:
        return _simulators[obj.name]

    settings = obj.taremin_cloth
    cloth_data = extract_cloth_mesh_data(obj, settings)
    pos_2d = cloth_data.positions
    coords = pos_2d.flatten().copy()
    edges_2d = cloth_data.normal_edges
    faces_2d = cloth_data.faces
    sew_2d = cloth_data.sewing_edges
    inv_masses = cloth_data.inv_masses
    n_verts = len(pos_2d)

    wg_size = int(getattr(settings, "workgroup_size", "32"))
    s_mode = 1 if getattr(settings, "solver_mode", "COLORING") == 'ATOMIC' else 0
    compact_rb = getattr(settings, "enable_compact_readback", True)

    is_deformed = obj.get("_taremin_cloth_is_deformed", False)
    rest_pos_2d = None
    # レスト頂点座標の取得 (未変形キャッシュがあれば優先して使用)
    if is_deformed and "_taremin_cloth_rest_positions" in obj:
        raw_rest = obj["_taremin_cloth_rest_positions"]
        if len(raw_rest) == n_verts * 3 * 4:
            rest_pos_2d = np.frombuffer(raw_rest, dtype=np.float32).reshape((n_verts, 3))

    sim_init_pos = rest_pos_2d if rest_pos_2d is not None else pos_2d

    sim = taremin_cloth_core.ClothSimulator(
        positions=sim_init_pos,
        edges=edges_2d,
        faces=faces_2d,
        inv_masses=inv_masses,
        sewing_springs=sew_2d,
        layer_id=settings.layer_id,
        thickness=settings.thickness,
        stiffness=settings.tension_stiffness,
        compression_stiffness=settings.compression_stiffness,
        shear_stiffness=settings.shear_stiffness,
        bending_stiffness=settings.bending_stiffness,
        sewing_shrink_speed=settings.sewing_shrink_speed,
        sewing_stiffness=getattr(settings, "sewing_stiffness", 10000.0),
        enable_sewing_lock=getattr(settings, "enable_sewing_lock", True),
        workgroup_size=wg_size,
        solver_mode=s_mode,
        enable_compact_readback=compact_rb,
    )

    # Cold Resume: レスト座標で自然長を初期化した後、現在の変形頂点座標をGPUにセットして停止位置から再開
    if rest_pos_2d is not None:
        sim.set_positions_and_velocities(pos_2d)
        logger.info(f"[Simulator] Cold resume: Restored positions to previous paused state for '{obj.name}'")

    # メッシュ特性長 (最小エッジ長と布厚み) を計算してキャッシュ
    if len(edges_2d) > 0:
        edge_diffs = pos_2d[edges_2d[:, 1]] - pos_2d[edges_2d[:, 0]]
        edge_lens_sq = np.sum(edge_diffs ** 2, axis=1)
        min_edge_len = float(np.sqrt(np.min(edge_lens_sq)))
        if min_edge_len < 1e-4:
            min_edge_len = 0.01
    else:
        min_edge_len = 0.02
    thickness = getattr(settings, "thickness", 0.005)
    char_len = max(0.001, min(min_edge_len, thickness * 2.0))
    _mesh_char_len_cache[obj.name] = char_len

    cache_rest_positions(obj)
    sync_colliders(sim, bpy.context.scene, cloth_obj=obj)
    sync_cloth_parameters(sim, obj, bpy.context.scene)

    _simulators[obj.name] = (sim, coords)
    n_faces = len(faces_2d) if faces_2d is not None else 0
    logger.debug(f"[Simulator] Initialized ClothSimulator for '{obj.name}' (verts={len(pos_2d)}, faces={n_faces})")
    return sim, coords


def get_effective_substeps(obj, coords, dt, scene=None):
    """布およびコライダーの運動速度とCFL条件に基づき、最適なサブステップ数を動的に算出する"""
    settings = getattr(obj, "taremin_cloth", None)
    if not settings or not getattr(settings, "enable_adaptive_substep", False):
        return getattr(settings, "substeps", 20)

    obj_name = obj.name
    prev_coords = _prev_coords_cache.get(obj_name)
    _prev_coords_cache[obj_name] = coords.copy()

    base_steps = getattr(settings, "substeps", 20)
    min_steps = getattr(settings, "min_substeps", 4)

    if prev_coords is None or prev_coords.shape != coords.shape or dt <= 0:
        _effective_substeps_cache[obj_name] = base_steps
        return base_steps

    # 1. 布頂点の最大変位量 (NumPy L_inf ノルムで高速算出)
    cloth_max_disp = float(np.max(np.abs(coords - prev_coords)))

    # 2. コライダーの最大変位量の算出
    collider_max_disp = 0.0
    sc = scene or getattr(bpy.context, "scene", None)
    if sc:
        for c_obj in sc.objects:
            if hasattr(c_obj, "taremin_cloth_collider") and c_obj.taremin_cloth_collider.is_collider and getattr(c_obj.taremin_cloth_collider, "enabled", True):
                curr_loc = np.array(c_obj.matrix_world.translation, dtype=np.float32)
                prev_loc = _collider_prev_locs_cache.get(c_obj.name)
                _collider_prev_locs_cache[c_obj.name] = curr_loc
                if prev_loc is not None:
                    disp = float(np.linalg.norm(curr_loc - prev_loc))
                    if disp > collider_max_disp:
                        collider_max_disp = disp

    max_disp = max(cloth_max_disp, collider_max_disp)

    # 3. メッシュ特性長および厚み (CFL条件) による目標ステップ数の決定
    char_len = _mesh_char_len_cache.get(obj_name, 0.01)
    thickness = getattr(settings, "thickness", 0.01)
    # 安全係数: 1サブステップあたりの移動量が、エッジ長だけでなく布の厚み（貫通閾値）に対しても過大にならないよう考慮
    # 厚みが極端に薄い場合でも、最低限の安全マージンを確保
    cfl_margin = min(0.5 * char_len, max(thickness * 1.5, 0.002))
    max_steps = getattr(settings, "max_substeps", max(base_steps, 64))

    if max_disp <= 1e-6:
        target_steps = min_steps
    else:
        computed_steps = int(np.ceil(max_disp / cfl_margin))
        target_steps = max(min_steps, min(max_steps, computed_steps))

    # 4. ヒステリシス制御（スムージング / ジッター防止）
    # 急激なステップ降下による剛性・ダンピングの揺らぎや布のピクつき、および急上昇による極端なFPSドロップを防止
    prev_steps = _effective_substeps_cache.get(obj_name, base_steps)
    if target_steps >= prev_steps:
        # 上昇時: 局所的な微小振動で一度に跳ね上がらないよう最大+4ステップずつ滑らかに追従
        current_steps = min(target_steps, prev_steps + 4)
    else:
        # 下降時: 減速・整定時は最大2ステップずつ緩やかに降下
        current_steps = max(target_steps, prev_steps - 2)

    current_steps = max(min_steps, min(max_steps, current_steps))
    _effective_substeps_cache[obj_name] = current_steps
    return current_steps


def step_cloth_object(
    obj,
    scene,
    dt: float = 1.0 / 60.0,
    depsgraph=None,
    force_sync_colliders: bool = False,
    update_mesh: bool = True,
    substeps: int = None,
    solver_iterations: int = None,
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

    if not skip_sync:
        if depsgraph is None:
            try:
                depsgraph = bpy.context.evaluated_depsgraph_get()
            except Exception:
                depsgraph = None

        sync_colliders(sim, scene, depsgraph=depsgraph, force=force_sync_colliders, cloth_obj=obj)
        sync_cloth_parameters(sim, obj, scene)
        sync_attachment_pins(sim, obj, scene)

    actual_substeps = substeps if substeps is not None else get_effective_substeps(obj, coords, dt, scene=scene)
    actual_iters = solver_iterations if solver_iterations is not None else (getattr(settings, "solver_iterations", 1) if settings else 1)

    sim.step(dt=dt, substeps=actual_substeps, solver_iterations=actual_iters)

    if update_mesh:
        sim.get_positions(coords)
        obj.data.vertices.foreach_set("co", coords)
        obj.data.update()
        obj["_taremin_cloth_is_deformed"] = True

    return sim, coords


def step_cloth_scene(
    scene,
    dt: float = 1.0 / 60.0,
    anim_frame: int = None,
    depsgraph=None,
    target_objs: list = None,
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
    # 1. コライダーのアニメーション駆動ステップ
    any_collider_deformed = False
    current_frame = anim_frame if anim_frame is not None else scene.frame_current

    for col_o in scene.objects:
        c_set = getattr(col_o, "taremin_cloth_collider", None)
        if c_set and c_set.is_collider and getattr(c_set, "enabled", True):
            if getattr(c_set, "anim", None) and c_set.anim.enabled:
                _, deformed = anim_driver.step_collider_animation(col_o, current_frame)
                if deformed:
                    any_collider_deformed = True
            elif col_o.type == 'MESH' and (c_set.collider_type == 'MESH' or (c_set.collider_type == 'BONE_SDF' and getattr(c_set, "enable_joint_mesh", True))):
                if col_o.find_armature() or (col_o.animation_data and col_o.animation_data.action):
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


def _persistent(func):
    """Blender環境では@persistentを適用し、スタンドアロンテスト環境では透過するデコレータ"""
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    persistent_fn = getattr(handlers, "persistent", None)
    if callable(persistent_fn):
        return persistent_fn(func)
    return func


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
            check_and_update_params_signature(obj)

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
                obj["_taremin_cloth_is_deformed"] = (current_frame != scene.frame_start)
            else:
                # 2. 未ベイク時またはキャッシュ範囲外:
                # シミュレーション変形中であれば初期レスト形状へ安全に戻す
                if obj.get("_taremin_cloth_is_deformed", False):
                    restore_rest_positions(obj, clear_timeline=False)


def bake_init_simulation(scene, start_frame=None, end_frame=None):
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


def bake_step_frame(bake_ctx, frame):
    """指定フレームのシミュレーション計算を1ステップ実行し、キャッシュに保存する"""
    scene = bake_ctx["scene"]
    cloth_objs = bake_ctx["cloth_objs"]
    start_f = bake_ctx["start_frame"]
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

    for obj in cloth_objs:
        sim, coords = get_or_create_simulator(obj)
        settings = obj.taremin_cloth
        actual_substeps = get_effective_substeps(obj, coords, dt, scene=scene)

        sim, coords = step_cloth_object(
            obj,
            scene,
            dt=dt,
            depsgraph=depsgraph,
            force_sync_colliders=any_collider_deformed,
            update_mesh=True,
            substeps=actual_substeps,
            solver_iterations=settings.solver_iterations,
        )
        _timeline_frame_cache[obj.name][frame] = coords.copy()
        obj["_taremin_cloth_is_deformed"] = True

    bake_ctx["baked_count"] += 1
    return bake_ctx["baked_count"]


def bake_finalize_simulation(bake_ctx):
    """ベイクの完了・終了処理を行い、タイムラインを開始フレームに戻す"""
    if not bake_ctx:
        return
    scene = bake_ctx["scene"]
    start_f = bake_ctx["start_frame"]
    cloth_objs = bake_ctx["cloth_objs"]

    scene.frame_set(start_f)
    for obj in cloth_objs:
        init_coords = _timeline_frame_cache.get(obj.name, {}).get(start_f)
        if init_coords is not None:
            obj.data.vertices.foreach_set("co", init_coords)
            obj.data.update()
            obj["_taremin_cloth_is_deformed"] = False

    logger.info(f"[Bake] Completed simulation bake: {bake_ctx['baked_count']} frames cached.")


def bake_cloth_simulation(scene, start_frame=None, end_frame=None, progress_callback=None, cancel_check=None):
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

        for frame in range(start_f + 1, end_f + 1):
            if cancel_check and cancel_check():
                logger.info(f"[Bake] Cancelled by user at frame {frame}")
                break
            bake_step_frame(bake_ctx, frame)
            if progress_callback:
                progress_callback(bake_ctx["baked_count"], total_steps + 1)

        bake_finalize_simulation(bake_ctx)
        return bake_ctx["baked_count"]
    finally:
        _is_baking = False

