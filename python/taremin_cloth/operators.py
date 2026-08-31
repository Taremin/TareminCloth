import os
import re
from datetime import datetime
import bpy
import mathutils
import numpy as np
from bpy_extras import view3d_utils
from .utils import drawing
from .utils import topology
from .utils import anim_driver
from .utils.logger import logger
from .preferences import get_preferences

# シミュレータインスタンスを保持するグローバルキャッシュ
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

    # シミュレーションによる変形が起きている最中は、変形座標でレストポーズを汚染しないよう上書きを拒否
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





_collider_cache = {}


def sync_colliders(sim, scene, depsgraph=None, force=False, cloth_obj=None):
    """シーン内のコライダーオブジェクトをシミュレータに同期する（差分キャッシュ・ブロードフェーズ対応）"""
    global _collider_cache

    if depsgraph is None:
        try:
            depsgraph = bpy.context.evaluated_depsgraph_get()
        except Exception:
            depsgraph = None

    # シーン内のコライダーの状態シグネチャをチェック
    collider_objs = []
    col_state = []
    has_active_anim = False

    for obj in scene.objects:
        col_settings = getattr(obj, "taremin_collider", None)
        if col_settings and col_settings.is_collider and getattr(col_settings, "enabled", True):
            collider_objs.append((obj, col_settings))
            anim_s = getattr(col_settings, "anim", None)
            if anim_s and anim_s.enabled:
                has_active_anim = True
            mat = obj.matrix_world
            mat_tuple = (
                round(mat[0][0], 4), round(mat[0][1], 4), round(mat[0][2], 4), round(mat[0][3], 4),
                round(mat[1][0], 4), round(mat[1][1], 4), round(mat[1][2], 4), round(mat[1][3], 4),
                round(mat[2][0], 4), round(mat[2][1], 4), round(mat[2][2], 4), round(mat[2][3], 4),
            )
            v_len = len(obj.data.vertices) if obj.type == 'MESH' else 0
            col_state.append((
                obj.name,
                col_settings.collider_type,
                mat_tuple,
                v_len,
                round(col_settings.friction, 3),
                round(col_settings.radius, 4),
                round(col_settings.thickness, 4),
                round(getattr(col_settings, "restitution", 0.0), 3),
                bool(getattr(col_settings, "single_sided", True)),
                bool(getattr(col_settings, "enabled", True)),
            ))

    sim_id = id(sim)
    state_key = tuple(col_state)

    if not force and not has_active_anim and _collider_cache.get(sim_id) == state_key:
        # コライダーの状態に変更がないため再アップロードをスキップ
        return

    _collider_cache[sim_id] = state_key
    sim.clear_colliders()
    all_mesh_triangles = []
    mesh_friction = 0.5
    mesh_thickness = 0.005
    mesh_restitution = 0.0
    mesh_single_sided = True

    for obj, col_settings in collider_objs:
        world_mat = obj.matrix_world
        loc = world_mat.translation
        restitution = getattr(col_settings, "restitution", 0.0)
        if col_settings.collider_type == 'SPHERE':
            sim.add_sphere_collider(
                [loc.x, loc.y, loc.z],
                col_settings.radius * max(obj.scale),
                col_settings.friction,
                restitution,
            )
        elif col_settings.collider_type == 'PLANE':
            normal = world_mat.to_3x3() @ bpy.types.mathutils.Vector((0.0, 0.0, 1.0)) if hasattr(bpy.types, "mathutils") else [0.0, 0.0, 1.0]
            norm_vec = [normal[0], normal[1], normal[2]]
            sim.add_plane_collider([loc.x, loc.y, loc.z], norm_vec, col_settings.friction, restitution)
        elif col_settings.collider_type == 'CAPSULE':
            pt_a = [loc.x, loc.y, loc.z - col_settings.radius]
            pt_b = [loc.x, loc.y, loc.z + col_settings.radius]
            sim.add_capsule_collider(pt_a, pt_b, col_settings.radius, col_settings.friction, restitution)
        elif col_settings.collider_type == 'MESH' and obj.type == 'MESH':
            mesh_friction = col_settings.friction
            mesh_thickness = col_settings.thickness
            mesh_restitution = restitution
            mesh_single_sided = getattr(col_settings, "single_sided", True)
            eval_obj = obj.evaluated_get(depsgraph) if depsgraph else obj
            mesh = eval_obj.to_mesh() if depsgraph else obj.data
            mesh.calc_loop_triangles()
            n_verts = len(mesh.vertices)
            n_tris = len(mesh.loop_triangles)

            if n_verts > 0 and n_tris > 0:
                raw_coords = np.empty(n_verts * 3, dtype=np.float32)
                mesh.vertices.foreach_get("co", raw_coords)
                v_local = raw_coords.reshape((n_verts, 3))

                # ワールド座標変換 (4x4行列積)
                mat_np = np.array(eval_obj.matrix_world, dtype=np.float32)
                ones = np.ones((n_verts, 1), dtype=np.float32)
                v_homo = np.hstack([v_local, ones])
                v_world = (v_homo @ mat_np.T)[:, :3]

                # NumPyファンシーインデックスによるベクトル化三角形抽出
                tri_indices = np.empty(n_tris * 3, dtype=np.int32)
                mesh.loop_triangles.foreach_get("vertices", tri_indices)
                tri_indices_2d = tri_indices.reshape((n_tris, 3))
                tri_coords = v_world[tri_indices_2d]  # shape: [n_tris, 3, 3]

                all_mesh_triangles.append(tri_coords)

            if depsgraph:
                eval_obj.to_mesh_clear()

    if all_mesh_triangles:
        tri_array = np.vstack(all_mesh_triangles) if len(all_mesh_triangles) > 1 else all_mesh_triangles[0]
        sim.set_mesh_collider_triangles(
            tri_array,
            friction=mesh_friction,
            thickness=mesh_thickness,
            restitution=mesh_restitution,
            single_sided=mesh_single_sided,
        )


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

    sim = taremin_cloth_core.ClothSimulator(
        positions=pos_2d,
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


def clear_simulators():
    """全シミュレータインスタンスおよび各種キャッシュを破棄する"""
    global _simulators, _collider_cache, _prev_coords_cache
    global _mesh_char_len_cache, _effective_substeps_cache, _collider_prev_locs_cache
    global _prev_elastic_scales
    restore_fast_playback()
    count = len(_simulators)
    _simulators.clear()
    _collider_cache.clear()
    _prev_coords_cache.clear()
    _mesh_char_len_cache.clear()
    _effective_substeps_cache.clear()
    _collider_prev_locs_cache.clear()
    _prev_elastic_scales.clear()
    _timeline_frame_cache.clear()
    _buffered_start_frames.clear()
    logger.debug(f"[Simulator] Cleared all {count} simulators and caches")


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

    # 3. メッシュ特性長 (CFL条件) による目標ステップ数の決定
    char_len = _mesh_char_len_cache.get(obj_name, 0.01)
    # 安全係数 0.5: 1サブステップあたりの移動量が特性長の半分以下になるように分割
    cfl_margin = 0.5 * char_len

    if max_disp <= 1e-6:
        target_steps = min_steps
    else:
        computed_steps = int(np.ceil(max_disp / cfl_margin))
        target_steps = max(min_steps, min(base_steps, computed_steps))

    # 4. ヒステリシス制御（スムージング / ジッター防止）
    # 急激なステップ降下による剛性・ダンピングの揺らぎや布のピクつきを防止
    prev_steps = _effective_substeps_cache.get(obj_name, base_steps)
    if target_steps >= prev_steps:
        # 上昇時（急な加速・衝突）は即時反映（安全優先）
        current_steps = target_steps
    else:
        # 下降時（減速・整定）は最大2ステップずつ緩やかに降下
        current_steps = max(target_steps, prev_steps - 2)

    current_steps = max(min_steps, min(base_steps, current_steps))
    _effective_substeps_cache[obj_name] = current_steps
    return current_steps


def sync_cloth_parameters(sim, obj, scene=None):
    """シミュレーション実行中にNパネルのプロパティ変更やシーン重力をGPUへ同期する"""
    settings = getattr(obj, "taremin_cloth", None)
    if not settings:
        return

    # 1. 剛性4種 (引張・圧縮・せん断・曲げ)
    if hasattr(sim, "set_stiffness_all"):
        sim.set_stiffness_all(
            settings.tension_stiffness,
            settings.compression_stiffness,
            settings.shear_stiffness,
            settings.bending_stiffness,
        )
    elif hasattr(sim, "set_stiffness"):
        sim.set_stiffness(settings.tension_stiffness, settings.bending_stiffness)

    # 2. 減衰 (空気抵抗・大域減衰 + 減衰4種)
    if hasattr(sim, "set_damping"):
        sim.set_damping(settings.air_damping)
    if hasattr(sim, "set_damping_all"):
        sim.set_damping_all(
            settings.tension_damping,
            settings.compression_damping,
            settings.shear_damping,
            settings.bending_damping,
        )

    # 3. 重力 (Blenderシーン重力 × オブジェクト重力倍率)
    if scene is not None and hasattr(sim, "set_gravity"):
        if getattr(scene, "use_gravity", True):
            sg = scene.gravity
            scale = settings.gravity
            sim.set_gravity(sg.x * scale, sg.y * scale, sg.z * scale)
        else:
            sim.set_gravity(0.0, 0.0, 0.0)

    # 4. ソルバー反復回数
    if hasattr(sim, "set_solver_iterations"):
        sim.set_solver_iterations(settings.solver_iterations)

    # 5. 自己衝突・レイヤー衝突
    if hasattr(sim, "set_enable_self_collision"):
        sim.set_enable_self_collision(getattr(settings, "enable_self_collision", False))
    if hasattr(sim, "set_self_collision_options"):
        sim.set_self_collision_options(
            relief_factor=getattr(settings, "self_collision_relief_factor", 0.2),
            max_displacement_ratio=getattr(settings, "self_collision_max_displacement_ratio", 0.2),
            exclude_neighbors=getattr(settings, "self_collision_exclude_neighbors", True),
            enable_normal_untangling=getattr(settings, "enable_normal_untangling", True),
        )

    # 5.5. エッジ詳細接触判定
    if hasattr(sim, "set_enable_edge_collision"):
        sim.set_enable_edge_collision(getattr(settings, "enable_edge_collision", False))
    if hasattr(sim, "set_edge_margin_scale"):
        sim.set_edge_margin_scale(getattr(settings, "edge_margin_scale", 1.0))
    if hasattr(sim, "set_edge_margin_offset"):
        sim.set_edge_margin_offset(getattr(settings, "edge_margin_offset", 0.0))

    # 5.6. チューニングパラメータ (ソルバー方式・ワークグループサイズ)
    if hasattr(sim, "set_solver_mode"):
        s_mode = 1 if getattr(settings, "solver_mode", "COLORING") == 'ATOMIC' else 0
        sim.set_solver_mode(s_mode)
    if hasattr(sim, "set_workgroup_size"):
        wg_size = int(getattr(settings, "workgroup_size", "32"))
        sim.set_workgroup_size(wg_size)
    if hasattr(sim, "set_enable_compact_readback"):
        sim.set_enable_compact_readback(getattr(settings, "enable_compact_readback", True))

    # 6. 伸縮グループ (Elastic Bands / Edge Scaling)
    sync_elastic_groups(sim, obj)


def sync_elastic_groups(sim, obj):
    """伸縮グループ（ゴム紐）の自然長スケールをGPUシミュレータに同期する（平滑化追従対応）"""
    global _prev_elastic_scales
    settings = getattr(obj, "taremin_cloth", None)
    if not settings or not hasattr(sim, "set_edge_rest_length_scales"):
        return

    groups = settings.elastic_groups
    if not groups:
        return

    # 各グループの有効なエッジとスケールを収集
    edge_scale_map = {}
    for g in groups:
        if not g.enabled:
            continue
        edges = g.get_edge_indices()
        scale = g.scale
        for e_idx in edges:
            edge_scale_map[e_idx] = scale

    if not edge_scale_map:
        # グループが無効または空の場合、スケールキャッシュがあればリセット
        if obj.name in _prev_elastic_scales:
            sim.reset_edge_rest_lengths()
            del _prev_elastic_scales[obj.name]
        return

    indices = list(edge_scale_map.keys())
    target_scales = [edge_scale_map[idx] for idx in indices]

    # スムージング追従（急激な変化による布の爆発防止）
    obj_name = obj.name
    prev_map = _prev_elastic_scales.get(obj_name, {})
    current_scales = []
    has_change = False

    for idx, target_s in zip(indices, target_scales):
        prev_s = prev_map.get(idx, 1.0)
        # 1フレームあたり 20% ずつ目標値へ近づける
        curr_s = prev_s + (target_s - prev_s) * 0.20
        if abs(curr_s - target_s) < 1e-4:
            curr_s = target_s
        if abs(curr_s - prev_s) > 1e-5 or idx not in prev_map:
            has_change = True
        current_scales.append(curr_s)
        prev_map[idx] = curr_s

    _prev_elastic_scales[obj_name] = prev_map

    if has_change or len(indices) > 0:
        sim.set_edge_rest_length_scales(
            np.array(indices, dtype=np.uint32),
            np.array(current_scales, dtype=np.float32),
        )


def sync_attachment_pins(sim, obj, scene):
    """外部オブジェクトやボーンに追従するアタッチメントピンを同期する"""
    settings = getattr(obj, "taremin_cloth", None)
    if not settings or not settings.pin_target_object:
        return

    target_obj = settings.pin_target_object
    vg_name = settings.pin_vertex_group or "Pin"
    vg = obj.vertex_groups.get(vg_name)
    if not vg:
        return

    target_mat = target_obj.matrix_world
    if settings.pin_target_bone and target_obj.type == 'ARMATURE' and target_obj.pose:
        bone = target_obj.pose.bones.get(settings.pin_target_bone)
        if bone:
            target_mat = target_mat @ bone.matrix

    inv_world = obj.matrix_world.inverted()
    mesh = obj.data
    for v in mesh.vertices:
        try:
            w = vg.weight(v.index)
        except RuntimeError:
            continue
        if w > 0.0:
            local_target = inv_world @ target_mat.translation
            sim.set_pin(v.index, [local_target.x, local_target.y, local_target.z], float(w))


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
        if c_set and c_set.is_collider and getattr(c_set, "enabled", True) and getattr(c_set, "anim", None) and c_set.anim.enabled:
            anim_frame = current_frame - scene.frame_start
            _, deformed = anim_driver.step_collider_animation(col_o, anim_frame)
            if deformed:
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


class TAREMIN_CLOTH_OT_toggle_cloth(bpy.types.Operator):
    """選択オブジェクトのClothシミュレーション有効/無効を切り替える"""
    bl_idname = "taremin_cloth.toggle_cloth"
    bl_label = "Toggle Cloth"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        settings = obj.taremin_cloth
        settings.is_cloth = not settings.is_cloth
        if not settings.is_cloth:
            restore_rest_positions(obj, clear=True)
        clear_simulators()
        self.report({'INFO'}, f"Cloth enabled: {settings.is_cloth}")
        return {'FINISHED'}


def clear_simulator_for_object(obj_name):
    """特定オブジェクトに対応するシミュレータおよび関連キャッシュを破棄する"""
    global _simulators, _prev_coords_cache, _mesh_char_len_cache
    global _effective_substeps_cache, _prev_elastic_scales
    had_sim = obj_name in _simulators
    _simulators.pop(obj_name, None)
    _prev_coords_cache.pop(obj_name, None)
    _mesh_char_len_cache.pop(obj_name, None)
    _effective_substeps_cache.pop(obj_name, None)
    _prev_elastic_scales.pop(obj_name, None)
    _timeline_frame_cache.pop(obj_name, None)
    _buffered_start_frames.pop(obj_name, None)
    if had_sim:
        logger.debug(f"[Simulator] Cleared simulator and caches for '{obj_name}'")


class TAREMIN_CLOTH_OT_reset_selected(bpy.types.Operator):
    """選択中のClothオブジェクトのみを初期レスト位置にリセットする"""
    bl_idname = "taremin_cloth.reset_selected"
    bl_label = "Reset Selected Cloth"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bool(obj and obj.type == 'MESH' and getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth)

    def execute(self, context):
        obj = context.active_object
        logger.info(f"[Reset] TAREMIN_CLOTH_OT_reset_selected executed for '{obj.name}'")
        restore_rest_positions(obj)
        clear_simulator_for_object(obj.name)
        obj.update_tag()
        if context.screen:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        self.report({'INFO'}, f"Reset Cloth: {obj.name}")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_reset_all(bpy.types.Operator):
    """シーン内のすべてのClothシミュレーションを初期状態にリセットする"""
    bl_idname = "taremin_cloth.reset_all"
    bl_label = "Reset All Simulation"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        if not scene:
            return False
        return any(getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth for obj in scene.objects)

    def execute(self, context):
        scene = context.scene
        if scene:
            for obj in scene.objects:
                if getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth:
                    restore_rest_positions(obj)
        clear_simulators()
        if context.screen:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        self.report({'INFO'}, "All Cloth Simulations Reset")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_reset_simulation(bpy.types.Operator):
    """シミュレーションを初期状態にリセットする（全体リセット互換）"""
    bl_idname = "taremin_cloth.reset_simulation"
    bl_label = "Reset Simulation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        return bpy.ops.taremin_cloth.reset_all()


_interactive_running = False
_interactive_operator_instance = None


def is_interactive_running():
    """インタラクティブモードが実行中かどうかを判定する"""
    global _interactive_running
    return _interactive_running


def stop_interactive_if_running():
    """インタラクティブモードが実行中であれば安全に停止要求を行う"""
    global _interactive_running, _interactive_operator_instance
    if _interactive_running and _interactive_operator_instance is not None:
        _interactive_operator_instance._stop_requested = True


def resolve_debug_filepath(prefs, obj_name: str, frame_count: int, ext: str = "jsonl.gz") -> str:
    """プレファレンスの設定値と現在のオブジェクト名・日時からデバッグ出力先ファイルパスを解決・生成する"""
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    datetime_str = now.strftime("%Y%m%d_%H%M%S")

    # オブジェクト名のサニタイズ（Windows/POSIXでファイル名に使えない文字を _ に置換）
    safe_obj_name = re.sub(r'[\\/*?:"<>|]', '_', obj_name)

    # テンプレートの取得
    template = getattr(prefs, "debug_filename_template", "cloth_debug_{datetime}_{object}.{ext}") if prefs else "cloth_debug_{datetime}_{object}.{ext}"
    if not template or not template.strip():
        template = "cloth_debug_{datetime}_{object}.{ext}"

    # 拡張子の置換・付与
    filename = template.format(
        date=date_str,
        time=time_str,
        datetime=datetime_str,
        object=safe_obj_name,
        frames=frame_count,
        ext=ext,
    )

    if not filename.endswith(f".{ext}") and not filename.endswith(".gz"):
        filename = f"{filename}.{ext}"

    # 出力ディレクトリの解決
    out_dir = getattr(prefs, "debug_output_dir", "").strip() if prefs else ""
    dir_path = None
    if out_dir:
        try:
            dir_path = bpy.path.abspath(out_dir)
        except Exception:
            pass
        if not dir_path:
            dir_path = os.path.abspath(out_dir)
    if not dir_path:
        import tempfile
        dir_path = getattr(getattr(bpy, "app", None), "tempdir", None) or tempfile.gettempdir()

    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, filename)


class TAREMIN_CLOTH_OT_interactive(bpy.types.Operator):
    """3Dビューポート上でリアルタイムに布を掴んで動かすモーダルオペレーター"""
    bl_idname = "taremin_cloth.interactive"
    bl_label = "Interactive Cloth Simulation"
    bl_options = {'REGISTER'}

    _timer = None
    _grabbed_vert = None
    _grab_initial_pos_local = None
    _grab_plane_point = None
    _grab_plane_normal = None
    _grab_initial_plane_hit = None
    _grab_current_target_local = None
    _stop_requested = False
    _pinned_verts = set()
    _anim_frame_counter = 0

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj
            and obj.type == 'MESH'
            and getattr(obj, "taremin_cloth", None)
            and obj.taremin_cloth.is_cloth
            and getattr(obj.taremin_cloth, "enabled", True)
        )

    def modal(self, context, event):
        global _interactive_running, _interactive_operator_instance
        if self._stop_requested:
            self.cancel(context)
            return {'FINISHED'}

        obj = context.active_object
        if not obj or not obj.taremin_cloth.is_cloth:
            self.cancel(context)
            return {'CANCELLED'}

        sim, coords = get_or_create_simulator(obj)

        if event.type == 'TIMER':
            # コライダーのアニメーション駆動ステップ
            any_collider_deformed = False
            for col_o in context.scene.objects:
                c_set = getattr(col_o, "taremin_collider", None)
                if c_set and c_set.is_collider and getattr(c_set, "enabled", True) and getattr(c_set, "anim", None) and c_set.anim.enabled:
                    _, deformed = anim_driver.step_collider_animation(col_o, self._anim_frame_counter)
                    if deformed:
                        any_collider_deformed = True
            self._anim_frame_counter += 1

            depsgraph = context.evaluated_depsgraph_get()
            if any_collider_deformed:
                context.view_layer.update()
                depsgraph = context.evaluated_depsgraph_get()

            sync_colliders(sim, context.scene, depsgraph=depsgraph, force=any_collider_deformed, cloth_obj=obj)
            sync_cloth_parameters(sim, obj, context.scene)
            actual_substeps = get_effective_substeps(obj, coords, 1.0 / 60.0, scene=context.scene)
            sim.step(dt=1.0 / 60.0, substeps=actual_substeps, solver_iterations=obj.taremin_cloth.solver_iterations)
            sim.get_positions(coords)
            obj.data.vertices.foreach_set("co", coords)
            obj.data.update()
            obj["_taremin_is_deformed"] = True

        elif event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                region = context.region
                rv3d = context.region_data
                if region and rv3d:
                    mouse_pos = (event.mouse_region_x, event.mouse_region_y)
                    pos_2d = coords.reshape((-1, 3))
                    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_pos)
                    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_pos)

                    best_idx = None

                    # 1. 視線レイキャストによる最前面ポリゴンの交差判定
                    if origin and direction:
                        matrix_world = obj.matrix_world
                        matrix_inv = matrix_world.inverted()
                        ray_origin_local = matrix_inv @ origin
                        ray_dir_local = (matrix_inv.to_3x3() @ direction).normalized()

                        hit, hit_loc, hit_normal, face_idx = obj.ray_cast(ray_origin_local, ray_dir_local)
                        if hit and face_idx is not None and 0 <= face_idx < len(obj.data.polygons):
                            polygon = obj.data.polygons[face_idx]
                            min_dist_sq = float('inf')
                            # ヒットしたポリゴンの構成頂点から交点に最も近い頂点を選択
                            for v_idx in polygon.vertices:
                                v_co = obj.data.vertices[v_idx].co
                                dist_sq = (v_co - hit_loc).length_squared
                                if dist_sq < min_dist_sq:
                                    min_dist_sq = dist_sq
                                    best_idx = v_idx

                    # 2. 面に直接ヒットしなかった場合のフォールバック（画面最近傍探索）
                    if best_idx is None:
                        best_dist = float('inf')
                        for i, p in enumerate(pos_2d):
                            world_p = obj.matrix_world @ mathutils.Vector(p)
                            screen_co = view3d_utils.location_3d_to_region_2d(region, rv3d, world_p)
                            if screen_co:
                                dist = (screen_co.x - mouse_pos[0]) ** 2 + (screen_co.y - mouse_pos[1]) ** 2
                                if dist < best_dist and dist < 2500:
                                    best_dist = dist
                                    best_idx = i

                    if best_idx is not None and origin and direction:
                        self._grabbed_vert = best_idx
                        p = pos_2d[best_idx]
                        v_initial_local = mathutils.Vector((p[0], p[1], p[2]))
                        v_initial_world = obj.matrix_world @ v_initial_local

                        # カメラの視線前方向ベクトルを取得（ビュー平面の法線）
                        view_inv = rv3d.view_matrix.inverted()
                        camera_forward = -view_inv.to_3x3().col[2].normalized()

                        # ビュー平面の通過点は「選択された頂点自身のワールド位置」
                        self._grab_plane_point = v_initial_world
                        self._grab_plane_normal = camera_forward
                        self._grab_initial_pos_local = v_initial_local
                        self._grab_current_target_local = v_initial_local

                        # クリック時のレイとビュー平面との初期交点を計算・記録
                        denom = direction.dot(camera_forward)
                        if abs(denom) > 1e-6:
                            t0 = (v_initial_world - origin).dot(camera_forward) / denom
                            self._grab_initial_plane_hit = origin + direction * t0
                        else:
                            self._grab_initial_plane_hit = v_initial_world

                        sim.set_pin(best_idx, [p[0], p[1], p[2]], 1.0)
                        drawing.set_active_grabbed_vertex(obj.name, best_idx, v_initial_world)
                        if context.area:
                            context.area.tag_redraw()
                        return {'RUNNING_MODAL'}

            elif event.value == 'RELEASE':
                if self._grabbed_vert is not None:
                    # ピン留めされていない頂点のみ物理解放
                    if self._grabbed_vert not in self._pinned_verts:
                        sim.release_pin(self._grabbed_vert)
                    drawing.clear_active_grabbed_vertex()
                    self._grabbed_vert = None
                    self._grab_initial_pos_local = None
                    self._grab_plane_point = None
                    self._grab_plane_normal = None
                    self._grab_initial_plane_hit = None
                    self._grab_current_target_local = None
                    if context.area:
                        context.area.tag_redraw()
                    return {'RUNNING_MODAL'}

        elif event.type == 'MOUSEMOVE' and self._grabbed_vert is not None:
            region = context.region
            rv3d = context.region_data
            if (region and rv3d and 
                self._grab_plane_point is not None and 
                self._grab_plane_normal is not None and 
                self._grab_initial_plane_hit is not None and 
                self._grab_initial_pos_local is not None):

                mouse_pos = (event.mouse_region_x, event.mouse_region_y)
                origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_pos)
                direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_pos)
                if origin and direction:
                    denom = direction.dot(self._grab_plane_normal)
                    if abs(denom) > 1e-6:
                        # 現在のマウスレイとビュー平面の交点を計算
                        t = (self._grab_plane_point - origin).dot(self._grab_plane_normal) / denom
                        current_plane_hit = origin + direction * t

                        # ビュー平面上のワールド移動差分（オフセット）
                        delta_world = current_plane_hit - self._grab_initial_plane_hit

                        # オブジェクトローカル空間の移動差分に変換
                        matrix_inv = obj.matrix_world.inverted()
                        delta_local = matrix_inv.to_3x3() @ delta_world

                        # 初期のローカル頂点位置にオフセットを加算
                        target_pos_local = self._grab_initial_pos_local + delta_local
                        self._grab_current_target_local = target_pos_local
                        sim.set_pin(self._grabbed_vert, [target_pos_local.x, target_pos_local.y, target_pos_local.z], 1.0)
                        target_world = obj.matrix_world @ target_pos_local
                        drawing.set_active_grabbed_vertex(obj.name, self._grabbed_vert, target_world)

        elif event.type == 'P' and event.value == 'PRESS':
            # Pキーで掴んでいる頂点（またはカーソル下の頂点）をピン留め／解除（トグル）
            target_v_idx = self._grabbed_vert
            if target_v_idx is None:
                # ドラッグしていない場合はマウスカーソル直下の頂点を探索
                region = context.region
                rv3d = context.region_data
                if region and rv3d:
                    mouse_pos = (event.mouse_region_x, event.mouse_region_y)
                    pos_2d = coords.reshape((-1, 3))
                    best_dist = float('inf')
                    for i, p in enumerate(pos_2d):
                        world_p = obj.matrix_world @ mathutils.Vector(p)
                        screen_co = view3d_utils.location_3d_to_region_2d(region, rv3d, world_p)
                        if screen_co:
                            dist = (screen_co.x - mouse_pos[0]) ** 2 + (screen_co.y - mouse_pos[1]) ** 2
                            if dist < best_dist and dist < 2500:
                                best_dist = dist
                                target_v_idx = i

            if target_v_idx is not None:
                settings = obj.taremin_cloth
                vg_name = settings.pin_vertex_group or "Pin"
                vg = obj.vertex_groups.get(vg_name)
                if not vg:
                    vg = obj.vertex_groups.new(name="Pin")
                    settings.pin_vertex_group = "Pin"

                # すでにピン留めされているか判定
                is_pinned = (target_v_idx in self._pinned_verts)
                if not is_pinned:
                    try:
                        w = vg.weight(target_v_idx)
                        if w > 0.0:
                            is_pinned = True
                    except RuntimeError:
                        is_pinned = False

                if is_pinned:
                    # ピン解除 (Unpin)
                    try:
                        vg.remove([target_v_idx])
                    except RuntimeError:
                        pass
                    self._pinned_verts.discard(target_v_idx)
                    # ドラッグ中でなければ物理ピンも解除
                    if self._grabbed_vert != target_v_idx:
                        sim.release_pin(target_v_idx)
                    self.report({'INFO'}, f"頂点 #{target_v_idx} のピン留めを解除しました")
                else:
                    # ピン留め (Pin)
                    vg.add([target_v_idx], 1.0, 'REPLACE')
                    self._pinned_verts.add(target_v_idx)
                    # 現在のローカル目標位置（または頂点現在位置）で固定
                    if self._grabbed_vert == target_v_idx and self._grab_current_target_local is not None:
                        t_pos = self._grab_current_target_local
                    else:
                        p = coords.reshape((-1, 3))[target_v_idx]
                        t_pos = mathutils.Vector((p[0], p[1], p[2]))
                    sim.set_pin(target_v_idx, [t_pos.x, t_pos.y, t_pos.z], 1.0)
                    self.report({'INFO'}, f"頂点 #{target_v_idx} をピン留めしました")

                if context.area:
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self.cancel(context)
            return {'FINISHED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        global _interactive_running, _interactive_operator_instance
        # トグル動作: すでに実行中の場合は停止を要求
        if _interactive_running:
            if _interactive_operator_instance is not None:
                _interactive_operator_instance._stop_requested = True
            return {'FINISHED'}

        obj = context.active_object
        self._stop_requested = False
        self._pinned_verts = set()

        # 十字分割の自動適用（未適用の四角面がある場合）
        if obj and obj.type == 'MESH':
            settings = getattr(obj, "taremin_cloth", None)
            if settings and settings.enable_cross_subdivision and not topology.is_cross_subdivided(obj):
                cache_rest_positions(obj, force=True)
                if topology.apply_cross_subdivision(obj):
                    clear_simulator_for_object(obj.name)
                    cache_rest_positions(obj, force=True)

        # 既存のピン留め頂点を事前ロード
        if obj and obj.type == 'MESH':
            settings = getattr(obj, "taremin_cloth", None)
            if settings:
                vg_name = settings.pin_vertex_group or "Pin"
                vg = obj.vertex_groups.get(vg_name)
                if vg:
                    for v in obj.data.vertices:
                        try:
                            if vg.weight(v.index) > 0.0:
                                self._pinned_verts.add(v.index)
                        except RuntimeError:
                            pass

        self._anim_frame_counter = 0

        # デバッグ状態記録の開始判定 (Console Log Level が DEBUG のとき、かつデバッグ記録設定有効時)
        prefs = get_preferences(context)
        should_debug_record = prefs and (prefs.log_level == 'DEBUG') and getattr(prefs, "enable_debug_recording", True)
        if should_debug_record and obj and obj.type == 'MESH':
            sim, _ = get_or_create_simulator(obj)
            if sim and hasattr(sim, "start_debug_recording"):
                max_frames = getattr(prefs, "debug_max_frames", 3600)
                sim.start_debug_recording(obj.name, max_frames=max_frames)
                logger.debug(f"[DebugRecorder] Started recording simulation states for '{obj.name}' (max_frames={max_frames})")

        wm = context.window_manager
        self._timer = wm.event_timer_add(1.0 / 60.0, window=context.window)
        wm.modal_handler_add(self)
        _interactive_running = True
        _interactive_operator_instance = self

        drawing.set_interactive_active(True)

        if hasattr(context.workspace, "status_text_set"):
            context.workspace.status_text_set(
                "Taremin Cloth: [左ドラッグ] 頂点移動 | [P] ピン留め/解除 | [右クリック / ESC] 終了"
            )

        self.report({'INFO'}, "Interactive Simulation Started (Press ESC / RightClick or Click Stop to exit)")
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        global _interactive_running, _interactive_operator_instance
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None

        drawing.clear_active_grabbed_vertex()
        drawing.set_interactive_active(False)

        if self._grabbed_vert is not None:
            if self._grabbed_vert not in self._pinned_verts:
                obj = context.active_object
                if obj and obj.name in _simulators:
                    sim, _ = _simulators[obj.name]
                    sim.release_pin(self._grabbed_vert)
            self._grabbed_vert = None

        self._grab_initial_pos_local = None
        self._grab_plane_point = None
        self._grab_plane_normal = None
        self._grab_initial_plane_hit = None
        self._grab_current_target_local = None
        self._pinned_verts.clear()
        _interactive_running = False
        _interactive_operator_instance = None

        if hasattr(context.workspace, "status_text_set"):
            context.workspace.status_text_set(None)

        if context.screen:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        # デバッグ状態記録のファイル保存とクリーンアップ
        obj = context.active_object
        if obj and obj.name in _simulators:
            sim, _ = _simulators[obj.name]
            if sim and hasattr(sim, "is_debug_recording") and sim.is_debug_recording():
                frame_count = sim.get_debug_frame_count()
                if frame_count > 0:
                    prefs = get_preferences(context)
                    filepath = resolve_debug_filepath(prefs, obj.name, frame_count, ext="jsonl.gz")
                    try:
                        saved_path = sim.save_debug_recording(filepath)
                        file_size = os.path.getsize(saved_path) if os.path.exists(saved_path) else 0
                        logger.info(f"[DebugRecorder] Successfully saved debug recording ({frame_count} frames, {file_size:,} bytes) to: {saved_path}")
                        self.report({'INFO'}, f"Debug recording saved: {os.path.basename(saved_path)} ({frame_count} frames)")
                    except Exception as e:
                        logger.error(f"[DebugRecorder] Failed to save debug recording: {e}")
                sim.stop_debug_recording()

        if obj and obj.type == 'MESH':
            settings = getattr(obj, "taremin_cloth", None)
            if settings and settings.enable_cross_subdivision and settings.auto_post_process and topology.is_cross_subdivided(obj):
                clear_simulator_for_object(obj.name)
                topology.apply_post_process(obj, mode=settings.post_process_mode, flatness_threshold=settings.adaptive_flatness_threshold)

        self.report({'INFO'}, "Interactive Simulation Stopped")


class TAREMIN_CLOTH_OT_create_seam(bpy.types.Operator):
    """選択された2頂点間に縫合エッジ（Loose Edge）を作成する"""
    bl_idname = "taremin_cloth.create_seam"
    bl_label = "Create Seam"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        import bmesh
        obj = context.active_object
        mesh = obj.data
        if obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(mesh)
            selected_verts = [v for v in bm.verts if v.select]
            if len(selected_verts) == 2:
                v0, v1 = selected_verts
                if not any(e for e in v0.link_edges if e.other_vert(v0) == v1):
                    bm.edges.new((v0, v1))
                    bmesh.update_edit_mesh(mesh)
                    self.report({'INFO'}, f"Created seam edge between vert {v0.index} and {v1.index}")
            else:
                self.report({'WARNING'}, "Please select exactly 2 vertices to connect as seam")
        else:
            selected_indices = [v.index for v in mesh.vertices if v.select]
            if len(selected_indices) == 2:
                v0, v1 = selected_indices
                bm = bmesh.new()
                bm.from_mesh(mesh)
                bm.verts.ensure_lookup_table()
                bm.edges.new((bm.verts[v0], bm.verts[v1]))
                bm.to_mesh(mesh)
                bm.free()
                mesh.update()
                self.report({'INFO'}, f"Created seam edge between vert {v0} and {v1}")
            else:
                self.report({'WARNING'}, "Please select exactly 2 vertices to connect as seam")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_add_elastic_group(bpy.types.Operator):
    """選択された辺から新規伸縮グループ（ゴム紐）を作成する"""
    bl_idname = "taremin_cloth.add_elastic_group"
    bl_label = "Add Elastic Group"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        import bmesh
        obj = context.active_object
        settings = obj.taremin_cloth

        # 選択中のエッジインデックスを抽出
        selected_edges = []
        if obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(obj.data)
            selected_edges = [e.index for e in bm.edges if e.select]
        else:
            selected_edges = [e.index for e in obj.data.edges if e.select]

        if not selected_edges:
            self.report({'WARNING'}, "辺（エッジ）を選択してください（Alt+クリックでループ選択等）")
            return {'CANCELLED'}

        group = settings.elastic_groups.add()
        group.name = f"Elastic {len(settings.elastic_groups)}"
        group.set_edge_indices(selected_edges)
        settings.active_elastic_group_index = len(settings.elastic_groups) - 1

        self.report({'INFO'}, f"新規伸縮グループ '{group.name}' を作成しました ({len(selected_edges)} 辺)")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_remove_elastic_group(bpy.types.Operator):
    """アクティブな伸縮グループを削除する"""
    bl_idname = "taremin_cloth.remove_elastic_group"
    bl_label = "Remove Elastic Group"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and len(obj.taremin_cloth.elastic_groups) > 0

    def execute(self, context):
        settings = context.active_object.taremin_cloth
        idx = settings.active_elastic_group_index
        if 0 <= idx < len(settings.elastic_groups):
            name = settings.elastic_groups[idx].name
            settings.elastic_groups.remove(idx)
            settings.active_elastic_group_index = max(0, idx - 1)
            self.report({'INFO'}, f"伸縮グループ '{name}' を削除しました")
            return {'FINISHED'}
        return {'CANCELLED'}


class TAREMIN_CLOTH_OT_assign_elastic_edges(bpy.types.Operator):
    """選択された辺をアクティブな伸縮グループに割り当てる（上書き）"""
    bl_idname = "taremin_cloth.assign_elastic_edges"
    bl_label = "Assign Selected Edges"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and len(obj.taremin_cloth.elastic_groups) > 0

    def execute(self, context):
        import bmesh
        obj = context.active_object
        settings = obj.taremin_cloth
        idx = settings.active_elastic_group_index
        if not (0 <= idx < len(settings.elastic_groups)):
            return {'CANCELLED'}

        group = settings.elastic_groups[idx]
        selected_edges = []
        if obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(obj.data)
            selected_edges = [e.index for e in bm.edges if e.select]
        else:
            selected_edges = [e.index for e in obj.data.edges if e.select]

        if not selected_edges:
            self.report({'WARNING'}, "辺を選択してください")
            return {'CANCELLED'}

        group.set_edge_indices(selected_edges)
        self.report({'INFO'}, f"グループ '{group.name}' に {len(selected_edges)} 辺を割り当てました")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_select_elastic_edges(bpy.types.Operator):
    """アクティブな伸縮グループに登録されている辺を3Dビューで選択する"""
    bl_idname = "taremin_cloth.select_elastic_edges"
    bl_label = "Select Group Edges"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and len(obj.taremin_cloth.elastic_groups) > 0

    def execute(self, context):
        import bmesh
        obj = context.active_object
        settings = obj.taremin_cloth
        idx = settings.active_elastic_group_index
        if not (0 <= idx < len(settings.elastic_groups)):
            return {'CANCELLED'}

        group = settings.elastic_groups[idx]
        target_indices = set(group.get_edge_indices())

        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')

        # 頂点/面選択から辺選択モードに切り替え
        context.tool_settings.mesh_select_mode = (False, True, False)
        bm = bmesh.from_edit_mesh(obj.data)
        for e in bm.edges:
            e.select = e.index in target_indices
        bmesh.update_edit_mesh(obj.data)

        self.report({'INFO'}, f"グループ '{group.name}' の {len(target_indices)} 辺を選択しました")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_apply_cross_subdivision(bpy.types.Operator):
    """選択中のメッシュの四角面を十字分割（Poke Faces）する"""
    bl_idname = "taremin_cloth.apply_cross_subdivision"
    bl_label = "Apply Cross Subdivision"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bool(obj and obj.type == 'MESH')

    def execute(self, context):
        obj = context.active_object
        if topology.is_cross_subdivided(obj):
            self.report({'WARNING'}, "すでに十字分割されています")
            return {'CANCELLED'}

        cache_rest_positions(obj, force=True)
        success = topology.apply_cross_subdivision(obj)
        if success:
            clear_simulator_for_object(obj.name)
            cache_rest_positions(obj, force=True)
            self.report({'INFO'}, "十字分割（Poke faces）を適用しました")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "分割対象の四角面（Quad）が見つかりませんでした")
            return {'CANCELLED'}


class TAREMIN_CLOTH_OT_apply_post_process(bpy.types.Operator):
    """シミュレーション結果に基づいてトポロジー後処理を実行する"""
    bl_idname = "taremin_cloth.apply_post_process"
    bl_label = "Apply Post-Process"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bool(obj and obj.type == 'MESH' and topology.is_cross_subdivided(obj))

    def execute(self, context):
        obj = context.active_object
        settings = getattr(obj, "taremin_cloth", None)
        mode = settings.post_process_mode if settings else 'OPTIMAL_TRI'
        flatness = settings.adaptive_flatness_threshold if settings else 5.0

        clear_simulator_for_object(obj.name)
        success = topology.apply_post_process(obj, mode=mode, flatness_threshold=flatness)
        if success:
            self.report({'INFO'}, f"後処理を適用しました: {mode}")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "後処理に失敗したか、対象データがありません")
            return {'CANCELLED'}


class TAREMIN_CLOTH_OT_restore_quad_topology(bpy.types.Operator):
    """十字分割の中心頂点を削除して四角面（Quad）に復元する"""
    bl_idname = "taremin_cloth.restore_quad_topology"
    bl_label = "Restore Quad Topology"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bool(obj and obj.type == 'MESH' and topology.is_cross_subdivided(obj))

    def execute(self, context):
        obj = context.active_object
        clear_simulator_for_object(obj.name)
        success = topology.restore_original_quads(obj)
        if success:
            self.report({'INFO'}, "四角面（Quad）に復元しました")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "四角面への復元に失敗しました")
            return {'CANCELLED'}


class TAREMIN_CLOTH_OT_clear_cache(bpy.types.Operator):
    """キャッシュとバックアップを破棄し、現在のメッシュ形状を初期レストポーズとして再設定する"""
    bl_idname = "taremin_cloth.clear_cache"
    bl_label = "Clear Cache"
    bl_description = "古いキャッシュやバックアップを破棄し、現在のメッシュ形状をシミュレーションの初期状態として再記憶します"
    bl_options = {'REGISTER', 'UNDO'}

    all_objects: bpy.props.BoolProperty(
        name="All Objects",
        description="シーン内のすべてのClothオブジェクトのキャッシュをクリアする",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        if not context.scene:
            return False
        obj = context.active_object
        has_sel = bool(obj and obj.type == 'MESH' and getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth)
        has_any = any(getattr(o, "taremin_cloth", None) and o.taremin_cloth.is_cloth for o in context.scene.objects)
        return has_sel or has_any

    def execute(self, context):
        if self.all_objects:
            targets = [o for o in context.scene.objects if getattr(o, "taremin_cloth", None) and o.taremin_cloth.is_cloth and o.type == 'MESH']
        else:
            obj = context.active_object
            targets = [obj] if (obj and getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth and obj.type == 'MESH') else []

        if not targets:
            self.report({'WARNING'}, "対象となるClothオブジェクトが見つかりません")
            return {'CANCELLED'}

        for o in targets:
            clear_simulator_for_object(o.name)
            topology.clear_pre_subdivision_backup(o)
            for prop in ("_taremin_rest_positions", "_taremin_rest_signature", "_taremin_is_deformed", "_taremin_cross_subdiv_map"):
                if prop in o:
                    try:
                        del o[prop]
                    except Exception:
                        pass
            cache_rest_positions(o, force=True)
            o.update_tag()
            logger.info(f"[Cache] Cleared and refreshed rest positions for '{o.name}' (verts={len(o.data.vertices)})")

        if self.all_objects:
            self.report({'INFO'}, f"全 {len(targets)} 個のClothキャッシュをクリアし、最新形状を記憶しました")
        else:
            self.report({'INFO'}, f"'{targets[0].name}' のキャッシュをクリアし、最新形状を記憶しました")

        return {'FINISHED'}


class TAREMIN_CLOTH_OT_apply_gpu_settings(bpy.types.Operator):
    """GPUバックエンドおよびデバイス設定を適用し、GPUコンテキストを再初期化する"""
    bl_idname = "taremin_cloth.apply_gpu_settings"
    bl_label = "Apply GPU Settings"
    bl_description = "選択したバックエンドおよびGPUデバイスを適用し、GPUコンテキストを再初期化します（シミュレーションはリセットされます）"
    bl_options = {'REGISTER'}

    def execute(self, context):
        from .preferences import get_preferences, apply_gpu_settings_from_prefs
        prefs = get_preferences(context)
        if not prefs:
            self.report({'ERROR'}, "アドオン設定の取得に失敗しました")
            return {'CANCELLED'}

        # 実行中のインタラクティブモードを停止
        stop_interactive_if_running()

        # 既存シミュレータキャッシュを破棄
        clear_simulators()

        try:
            apply_gpu_settings_from_prefs(prefs)
            msg = f"GPU設定を適用しました: {prefs.active_device_name} ({prefs.active_backend_name})"
            logger.info(f"[GPU] {msg}")
            self.report({'INFO'}, msg)
        except Exception as e:
            err_msg = f"GPUコンテキスト初期化エラー: {e}"
            logger.error(f"[GPU] {err_msg}")
            self.report({'ERROR'}, err_msg)
            return {'CANCELLED'}

        if context.screen:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        return {'FINISHED'}


class TAREMIN_CLOTH_OT_record_pose(bpy.types.Operator):
    """アーマチュアの現在のポーズ姿勢をスナップショットとして記録する"""
    bl_idname = "taremin_cloth.record_pose"
    bl_label = "Record Pose Snapshot"
    bl_options = {'REGISTER', 'UNDO'}

    slot: bpy.props.EnumProperty(
        name="Slot",
        items=[
            ('START', "Start (0.0)", "開始姿勢として記録"),
            ('TARGET', "Target (1.0)", "目標姿勢として記録"),
        ],
        default='START',
    )

    def execute(self, context):
        obj = context.active_object
        if not obj:
            self.report({'WARNING'}, "オブジェクトが選択されていません")
            return {'CANCELLED'}

        col_settings = getattr(obj, "taremin_collider", None)
        if not col_settings or not getattr(col_settings, "anim", None):
            self.report({'WARNING'}, "コライダー設定またはアニメーション設定が見つかりません")
            return {'CANCELLED'}

        anim_s = col_settings.anim
        armature = anim_s.armature_obj
        if not armature:
            if obj.type == 'ARMATURE':
                armature = obj
            elif obj.parent and obj.parent.type == 'ARMATURE':
                armature = obj.parent
            else:
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE' and mod.object:
                        armature = mod.object
                        break

        if not armature or armature.type != 'ARMATURE':
            self.report({'WARNING'}, "対象アーマチュアが見つかりません")
            return {'CANCELLED'}

        snapshot = anim_driver.capture_pose_snapshot(armature)
        if not snapshot:
            self.report({'WARNING'}, "ポーズデータのキャプチャに失敗しました")
            return {'CANCELLED'}

        if self.slot == 'START':
            anim_s.start_pose_data = snapshot
            self.report({'INFO'}, f"開始ポーズ (0.0) を記録しました: {armature.name}")
        else:
            anim_s.target_pose_data = snapshot
            self.report({'INFO'}, f"目標ポーズ (1.0) を記録しました: {armature.name}")

        return {'FINISHED'}


class TAREMIN_CLOTH_OT_apply_pose_preview(bpy.types.Operator):
    """記録されたポーズやレスト姿勢をアーマチュアに適用する"""
    bl_idname = "taremin_cloth.apply_pose_preview"
    bl_label = "Apply Pose Preview"
    bl_options = {'REGISTER', 'UNDO'}

    slot: bpy.props.EnumProperty(
        name="Slot",
        items=[
            ('START', "Start (0.0)", "開始姿勢を適用"),
            ('TARGET', "Target (1.0)", "目標姿勢を適用"),
            ('REST', "Rest", "ボーンTransformをクリアしてレスト姿勢に戻す"),
        ],
        default='START',
    )

    def execute(self, context):
        obj = context.active_object
        if not obj:
            return {'CANCELLED'}

        col_settings = getattr(obj, "taremin_collider", None)
        if not col_settings or not getattr(col_settings, "anim", None):
            return {'CANCELLED'}

        anim_s = col_settings.anim
        armature = anim_s.armature_obj
        if not armature:
            if obj.type == 'ARMATURE':
                armature = obj
            elif obj.parent and obj.parent.type == 'ARMATURE':
                armature = obj.parent
            else:
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE' and mod.object:
                        armature = mod.object
                        break

        if not armature or armature.type != 'ARMATURE' or not armature.pose:
            self.report({'WARNING'}, "対象アーマチュアが見つかりません")
            return {'CANCELLED'}

        if self.slot == 'REST':
            for pbone in armature.pose.bones:
                pbone.location = (0.0, 0.0, 0.0)
                pbone.scale = (1.0, 1.0, 1.0)
                if pbone.rotation_mode == 'QUATERNION':
                    pbone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
                elif pbone.rotation_mode == 'AXIS_ANGLE':
                    pbone.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
                else:
                    pbone.rotation_euler = (0.0, 0.0, 0.0)
            anim_s.progress = 0.0
            self.report({'INFO'}, "レストポーズに復帰しました")
        elif self.slot == 'START':
            if anim_s.start_pose_data:
                anim_driver.apply_pose_blend(armature, anim_s.start_pose_data, anim_s.start_pose_data, 0.0)
                anim_s.progress = 0.0
                self.report({'INFO'}, "開始ポーズ (0.0) を適用しました")
            else:
                self.report({'WARNING'}, "開始ポーズが記録されていません")
        elif self.slot == 'TARGET':
            if anim_s.target_pose_data:
                anim_driver.apply_pose_blend(armature, anim_s.target_pose_data, anim_s.target_pose_data, 1.0)
                anim_s.progress = 1.0
                self.report({'INFO'}, "目標ポーズ (1.0) を適用しました")
            else:
                self.report({'WARNING'}, "目標ポーズが記録されていません")

        context.view_layer.update()
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_select_object(bpy.types.Operator):
    """指定した布またはコライダーオブジェクトを選択してアクティブにする"""
    bl_idname = "taremin_cloth.select_object"
    bl_label = "Select Object"
    bl_description = "指定した布またはコライダーオブジェクトを選択してアクティブにします"
    bl_options = {'UNDO'}

    object_name: bpy.props.StringProperty(
        name="Object Name",
        description="選択するオブジェクトの名前",
        default="",
    )

    def execute(self, context):
        if not self.object_name or self.object_name not in context.scene.objects:
            self.report({'WARNING'}, f"オブジェクト '{self.object_name}' が見つかりません")
            return {'CANCELLED'}

        target_obj = context.scene.objects[self.object_name]

        # 編集モード等の場合はオブジェクトモードに安全に切り替える
        if context.mode != 'OBJECT' and bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='OBJECT')

        # 既存の選択をすべて解除
        for obj in context.selected_objects:
            obj.select_set(False)

        # ターゲットオブジェクトが非表示の場合は表示状態にする
        if target_obj.hide_get():
            target_obj.hide_set(False)

        # ターゲットオブジェクトを選択してアクティブ化
        target_obj.select_set(True)
        context.view_layer.objects.active = target_obj

        return {'FINISHED'}


classes = (
    TAREMIN_CLOTH_OT_toggle_cloth,
    TAREMIN_CLOTH_OT_reset_selected,
    TAREMIN_CLOTH_OT_reset_all,
    TAREMIN_CLOTH_OT_reset_simulation,
    TAREMIN_CLOTH_OT_clear_cache,
    TAREMIN_CLOTH_OT_apply_gpu_settings,
    TAREMIN_CLOTH_OT_interactive,
    TAREMIN_CLOTH_OT_create_seam,
    TAREMIN_CLOTH_OT_add_elastic_group,
    TAREMIN_CLOTH_OT_remove_elastic_group,
    TAREMIN_CLOTH_OT_assign_elastic_edges,
    TAREMIN_CLOTH_OT_select_elastic_edges,
    TAREMIN_CLOTH_OT_apply_cross_subdivision,
    TAREMIN_CLOTH_OT_apply_post_process,
    TAREMIN_CLOTH_OT_restore_quad_topology,
    TAREMIN_CLOTH_OT_record_pose,
    TAREMIN_CLOTH_OT_apply_pose_preview,
    TAREMIN_CLOTH_OT_select_object,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    if cloth_frame_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(cloth_frame_handler)


def unregister():
    if cloth_frame_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(cloth_frame_handler)
    clear_simulators()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
