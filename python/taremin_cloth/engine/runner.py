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
)
from .collider import sync_colliders
from .params import sync_cloth_parameters, sync_attachment_pins
from ..utils import anim_driver
from ..utils.logger import logger

_fast_playback_saved_mods = {}


def apply_fast_playback(scene):
    """シミュレーションに関係のないオブジェクトの重いモディファイアを一時無効化してDepsgraphを軽量化"""
    global _fast_playback_saved_mods
    if not getattr(scene, "taremin_cloth_fast_playback", False):
        return

    sim_objs = set()
    for obj in scene.objects:
        if getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth:
            sim_objs.add(obj)
        if getattr(obj, "taremin_collider", None) and obj.taremin_collider.is_collider:
            sim_objs.add(obj)

    heavy_types = {'SUBSURF', 'SOLIDIFY', 'LATTICE', 'DATA_TRANSFER', 'WEIGHTED_NORMAL', 'NODES', 'WELD', 'BEVEL'}

    for obj in scene.objects:
        if obj.type == 'MESH' and obj not in sim_objs and obj.modifiers:
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

    mesh = obj.data
    n_verts = len(mesh.vertices)
    coords = np.empty(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    pos_2d = coords.reshape((n_verts, 3))

    tri_list = []
    mesh.calc_loop_triangles()
    for tri in mesh.loop_triangles:
        tri_list.append(tri.vertices)
    faces_2d = np.array(tri_list, dtype=np.uint32) if tri_list else None

    settings = obj.taremin_cloth
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

    for e in all_edges_2d:
        pair = (min(e[0], e[1]), max(e[0], e[1]))
        if pair in face_edge_set:
            normal_edges.append(e)
        else:
            if settings.enable_sewing:
                sewing_edges.append(e)
            else:
                normal_edges.append(e)

    edges_2d = np.array(normal_edges, dtype=np.uint32) if normal_edges else np.empty((0, 2), dtype=np.uint32)
    sew_2d = np.array(sewing_edges, dtype=np.uint32) if sewing_edges else None

    inv_masses = np.ones(n_verts, dtype=np.float32)
    vg_name = settings.pin_vertex_group if settings and settings.pin_vertex_group else "Pin"
    pin_vg = obj.vertex_groups.get(vg_name) or obj.vertex_groups.get("Pin") or obj.vertex_groups.get("Cloth_Pin")
    if pin_vg:
        for i in range(n_verts):
            try:
                weight = pin_vg.weight(i)
                inv_masses[i] = max(0.0, 1.0 - weight)
            except RuntimeError:
                inv_masses[i] = 1.0

    wg_size = int(getattr(settings, "workgroup_size", "32"))
    s_mode = 1 if getattr(settings, "solver_mode", "COLORING") == 'ATOMIC' else 0
    compact_rb = getattr(settings, "enable_compact_readback", True)

    is_deformed = obj.get("_taremin_is_deformed", False)
    rest_pos_2d = None
    if is_deformed and "_taremin_rest_positions" in obj:
        raw_rest = obj["_taremin_rest_positions"]
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
            if hasattr(c_obj, "taremin_collider") and c_obj.taremin_collider.is_collider and getattr(c_obj.taremin_collider, "enabled", True):
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


def cloth_frame_handler(scene):
    """Blenderタイムライン進行時のハンドラー"""
    current_frame = scene.frame_current

    if current_frame == scene.frame_start:
        restore_fast_playback(scene)
        for obj in scene.objects:
            if hasattr(obj, "taremin_cloth") and obj.taremin_cloth.is_cloth:
                restore_rest_positions(obj)
        clear_simulators()
        return

    # 高速プレビューが有効ならDepsgraphバイパスを適用
    apply_fast_playback(scene)

    # コライダーのアニメーション駆動ステップ (タイムライン再生時もコライダーアニメーションを連動)
    any_collider_deformed = False
    for col_o in scene.objects:
        c_set = getattr(col_o, "taremin_collider", None)
        if c_set and c_set.is_collider and getattr(c_set, "enabled", True):
            if getattr(c_set, "anim", None) and c_set.anim.enabled:
                anim_frame = current_frame - scene.frame_start
                _, deformed = anim_driver.step_collider_animation(col_o, anim_frame)
                if deformed:
                    any_collider_deformed = True
            elif c_set.collider_type == 'MESH' and col_o.type == 'MESH':
                if col_o.find_armature() or (col_o.animation_data and col_o.animation_data.action):
                    any_collider_deformed = True

    try:
        if any_collider_deformed:
            bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
    except Exception:
        depsgraph = None

    for obj in scene.objects:
        if obj.type == 'MESH' and hasattr(obj, "taremin_cloth") and obj.taremin_cloth.is_cloth and getattr(obj.taremin_cloth, "enabled", True):
            sim, coords = get_or_create_simulator(obj)
            settings = obj.taremin_cloth
            dt = 1.0 / scene.render.fps
            actual_substeps = get_effective_substeps(obj, coords, dt, scene=scene)

            # 1. キャッシュヒット判定 (過去に計算済みのフレームならGPU計算不要で即座にメッシュ反映)
            obj_cache = _timeline_frame_cache.setdefault(obj.name, {})
            if current_frame in obj_cache:
                cached_coords = obj_cache[current_frame]
                coords[:] = cached_coords
                obj.data.vertices.foreach_set("co", cached_coords)
                obj.data.update()
                obj["_taremin_is_deformed"] = True
                continue

            # 2. フレームバッファリング判定
            buf_enabled = getattr(settings, "enable_frame_buffering", False) and hasattr(sim, "step_buffered")
            buf_size = getattr(settings, "frame_buffer_size", 2) if buf_enabled else 1
            is_last_frame = (current_frame == getattr(scene, "frame_end", 250))

            if buf_enabled:
                if obj.name not in _buffered_start_frames:
                    _buffered_start_frames[obj.name] = current_frame

                sync_colliders(sim, scene, depsgraph, force=any_collider_deformed, cloth_obj=obj)
                sync_cloth_parameters(sim, obj, scene)
                sync_attachment_pins(sim, obj, scene)

                # GPU内部リングバッファに保存（同期待機・転送ゼロで即座に復帰）
                count = sim.step_buffered(dt=dt, substeps=actual_substeps, solver_iterations=settings.solver_iterations)

                # バッファ満杯または最終フレーム時: まとめて一括取得
                if count >= buf_size or is_last_frame:
                    buf_coords = np.empty(count * len(coords), dtype=np.float32)
                    fetched = sim.fetch_buffered_positions(buf_coords)
                    start_f = _buffered_start_frames.pop(obj.name, current_frame - fetched + 1)

                    # 各フレームの座標をキャッシュに埋め込む
                    n_coords = len(coords)
                    for i in range(fetched):
                        f_num = start_f + i
                        frame_data = buf_coords[i * n_coords : (i + 1) * n_coords]
                        obj_cache[f_num] = frame_data.copy()

                    # 最新フレームの座標を現在のメッシュに反映して描画
                    latest_data = buf_coords[(fetched - 1) * n_coords : fetched * n_coords]
                    coords[:] = latest_data
                    obj.data.vertices.foreach_set("co", latest_data)
                    obj.data.update()
                    obj["_taremin_is_deformed"] = True
                else:
                    # バッファリング中: メッシュ更新・ビューポート描画をスキップ
                    pass

            elif getattr(settings, "enable_async_readback", False) and hasattr(sim, "step_async"):
                # 非同期リードバック
                if sim.fetch_positions(coords):
                    obj.data.vertices.foreach_set("co", coords)
                    obj.data.update()
                    obj["_taremin_is_deformed"] = True
                    if current_frame > 1:
                        obj_cache[current_frame - 1] = coords.copy()

                sync_colliders(sim, scene, depsgraph, force=any_collider_deformed, cloth_obj=obj)
                sync_cloth_parameters(sim, obj, scene)
                sync_attachment_pins(sim, obj, scene)
                sim.step_async(dt=dt, substeps=actual_substeps, solver_iterations=settings.solver_iterations)

            else:
                # 同期実行モード
                sync_colliders(sim, scene, depsgraph, force=any_collider_deformed, cloth_obj=obj)
                sync_cloth_parameters(sim, obj, scene)
                sync_attachment_pins(sim, obj, scene)
                sim.step(dt=dt, substeps=actual_substeps, solver_iterations=settings.solver_iterations)
                sim.get_positions(coords)
                obj_cache[current_frame] = coords.copy()
                obj.data.vertices.foreach_set("co", coords)
                obj.data.update()
                obj["_taremin_is_deformed"] = True
