"""
taremin_cloth コライダー同期モジュール
Blenderシーン内のコライダーオブジェクト（Sphere, Plane, Capsule, Mesh）を検出し、
ワールド座標変換を行ってGPUシミュレータに差分同期する。
"""

import math
import bpy
import numpy as np
from ..utils.logger import logger
from .sdf_baker import (
    get_or_bake_bone_sdf_for_object,
    get_or_bake_mesh_sdf_for_object,
    get_armature_modifier,
    BoneSdfBakeResult,
    MeshSdfBakeResult,
    extract_dynamic_sdf_setup_data,
)

_collider_cache = {}
_bone_sdf_cache = {}
_bone_sdf_signatures = {}

COLLIDER_IGNORED_MODIFIER_TYPES = {'SUBSURF', 'SOLIDIFY', 'MULTIRES', 'BEVEL'}


def get_collider_eval_mesh(obj, depsgraph):
    """
    コライダー用評価メッシュを取得する。
    LATTICEやARMATUREなどのデフォーマーは反映するが、
    頂点・面を増殖・二重化するSUBSURFやSOLIDIFY等は一時無効化して取得する。
    戻り値: (mesh, eval_obj, disabled_mods)
    """
    disabled_mods = []
    for mod in getattr(obj, "modifiers", []):
        if mod.type in COLLIDER_IGNORED_MODIFIER_TYPES and getattr(mod, "show_viewport", False):
            mod.show_viewport = False
            disabled_mods.append(mod)

    eval_obj = None
    try:
        if disabled_mods:
            if depsgraph is not None:
                depsgraph.update()
            else:
                try:
                    depsgraph = bpy.context.evaluated_depsgraph_get()
                except Exception:
                    depsgraph = None
        eval_obj = obj.evaluated_get(depsgraph) if depsgraph else obj
        mesh = eval_obj.to_mesh() if depsgraph else obj.data
        return mesh, eval_obj, disabled_mods
    except Exception as e:
        logger.warning(f"[Collider Sync] 評価メッシュ取得失敗、obj.dataにフォールバックします: {e}")
        return getattr(obj, "data", None), None, disabled_mods


def cleanup_collider_eval_mesh(eval_obj, disabled_mods):
    """get_collider_eval_mesh で取得した評価メッシュとモディファイア状態を解放・復元する"""
    if eval_obj is not None:
        try:
            eval_obj.to_mesh_clear()
        except Exception:
            pass
    for mod in disabled_mods:
        mod.show_viewport = True


def clear_collider_cache():
    """コライダーキャッシュをクリアする"""
    global _collider_cache, _bone_sdf_cache, _bone_sdf_signatures
    _collider_cache.clear()
    _bone_sdf_cache.clear()
    _bone_sdf_signatures.clear()


def sync_colliders(sim, scene, depsgraph=None, force=False, cloth_obj=None):
    """シーン内のコライダーオブジェクトをシミュレータに同期する（差分キャッシュ・ブロードフェーズ・ボーンSDF対応）"""
    global _collider_cache, _bone_sdf_cache, _bone_sdf_signatures

    if depsgraph is None:
        try:
            depsgraph = bpy.context.evaluated_depsgraph_get()
        except Exception:
            depsgraph = None

    # シーン内のコライダーの状態シグネチャをチェック
    collider_objs = []
    col_state = []
    has_active_anim = False
    has_bone_sdf = False

    for obj in scene.objects:
        col_settings = getattr(obj, "taremin_cloth_collider", None)
        if col_settings and col_settings.is_collider and getattr(col_settings, "enabled", True):
            collider_objs.append((obj, col_settings))
            anim_s = getattr(col_settings, "anim", None)
            if anim_s and anim_s.enabled:
                has_active_anim = True
            if col_settings.collider_type == 'BONE_SDF':
                has_bone_sdf = True
                if getattr(col_settings, "enable_joint_mesh", True):
                    if get_armature_modifier(obj) is not None:
                        has_active_anim = True
                    elif obj.animation_data and obj.animation_data.action:
                        has_active_anim = True
            elif col_settings.collider_type == 'MESH_SDF' and obj.type == 'MESH':
                has_bone_sdf = True
                if obj.animation_data and obj.animation_data.action:
                    has_active_anim = True
            elif col_settings.collider_type == 'MESH' and obj.type == 'MESH':
                # Armature変形、オブジェクトアニメーション、シェイプキーアニメーションがある場合は毎フレーム更新
                if get_armature_modifier(obj) is not None:
                    has_active_anim = True
                elif obj.animation_data and obj.animation_data.action:
                    has_active_anim = True
                elif getattr(obj.data, "shape_keys", None) and obj.data.shape_keys.animation_data:
                    has_active_anim = True
                else:
                    for mod in getattr(obj, "modifiers", []):
                        if mod.type == 'LATTICE' and getattr(mod, "object", None):
                            lat_o = mod.object
                            if (lat_o.animation_data and lat_o.animation_data.action) or (
                                getattr(lat_o.data, "animation_data", None) and lat_o.data.animation_data.action
                            ):
                                has_active_anim = True
                                break

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
                getattr(col_settings, "sdf_resolution", "64"),
                round(getattr(col_settings, "sdf_margin", 0.2), 3),
                round(getattr(col_settings, "weight_threshold", 0.02), 4),
                round(getattr(col_settings, "blend_k", 0.05), 4),
                bool(getattr(col_settings, "enable_joint_mesh", True)),
                round(getattr(col_settings, "joint_weight_threshold", 0.85), 2),
                round(getattr(col_settings, "mesh_sdf_voxel_size", 0.004), 4),
                round(getattr(col_settings, "mesh_sdf_margin", 0.02), 4),
                int(getattr(col_settings, "mesh_sdf_max_vram_mb", 256)),
                bool(getattr(col_settings, "mesh_sdf_auto_scale", True)),
            ))

    sim_id = id(sim)
    state_key = tuple(col_state)
    state_changed = force or has_active_anim or (_collider_cache.get(sim_id) != state_key)

    logger.debug(f"[Collider Sync] Frame {getattr(scene, 'frame_current', -1)}: state_changed={state_changed} (force={force}, has_active_anim={has_active_anim})")
    if state_changed:
        _collider_cache[sim_id] = state_key
        sim.clear_colliders()
        all_mesh_triangles = []
        all_mesh_attributes = []

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
                cur_friction = float(col_settings.friction)
                cur_thickness = float(col_settings.thickness)
                cur_restitution = float(restitution)
                cur_single_sided = 1.0 if getattr(col_settings, "single_sided", True) else 0.0

                mesh, eval_obj, disabled_mods = get_collider_eval_mesh(obj, depsgraph)
                try:
                    if mesh:
                        mesh.calc_loop_triangles()
                        n_verts = len(mesh.vertices)
                        n_tris = len(mesh.loop_triangles)

                        if n_verts > 0 and n_tris > 0:
                            raw_coords = np.empty(n_verts * 3, dtype=np.float32)
                            mesh.vertices.foreach_get("co", raw_coords)
                            v_local = raw_coords.reshape((n_verts, 3))

                            target_obj = eval_obj if eval_obj else obj
                            mat_np = np.array(target_obj.matrix_world, dtype=np.float32)
                            ones = np.ones((n_verts, 1), dtype=np.float32)
                            v_homo = np.hstack([v_local, ones])
                            v_world = (v_homo @ mat_np.T)[:, :3]

                            tri_indices = np.empty(n_tris * 3, dtype=np.int32)
                            mesh.loop_triangles.foreach_get("vertices", tri_indices)
                            tri_indices_2d = tri_indices.reshape((n_tris, 3))
                            tri_coords = v_world[tri_indices_2d]

                            all_mesh_triangles.append(tri_coords)
                            attr_row = np.array([cur_friction, cur_thickness, cur_restitution, cur_single_sided], dtype=np.float32)
                            attr_block = np.tile(attr_row, (n_tris, 1))
                            all_mesh_attributes.append(attr_block)
                finally:
                    cleanup_collider_eval_mesh(eval_obj, disabled_mods)

            elif col_settings.collider_type == 'BONE_SDF':
                arm_mod = get_armature_modifier(obj)
                if not arm_mod:
                    logger.warning(f"[Collider Sync] '{obj.name}' にArmatureモディファイアがありません。BONE_SDFをスキップします。")
                    continue

                update_mode = getattr(col_settings, "sdf_update_mode", "STATIC")
                current_bone_sig = (
                    obj.name,
                    update_mode,
                    getattr(col_settings, "sdf_resolution", "64"),
                    round(float(getattr(col_settings, "sdf_margin", 0.2)), 4),
                    round(float(getattr(col_settings, "thickness", 0.005)), 5),
                    round(float(getattr(col_settings, "friction", 0.5)), 3),
                    round(float(getattr(col_settings, "restitution", 0.0)), 3),
                    round(float(getattr(col_settings, "weight_threshold", 0.02)), 4),
                    round(float(getattr(col_settings, "blend_k", 0.05)), 4),
                    bool(getattr(col_settings, "enable_joint_mesh", True)),
                    round(float(getattr(col_settings, "joint_weight_threshold", 0.85)), 2),
                )

                needs_rebake = (sim_id not in _bone_sdf_cache) or (_bone_sdf_signatures.get(sim_id) != current_bone_sig)

                if needs_rebake:
                    bake_res = get_or_bake_bone_sdf_for_object(obj, col_settings)
                    if bake_res and bake_res.depth > 0:
                        if update_mode == 'DYNAMIC_GPU':
                            dyn_data = extract_dynamic_sdf_setup_data(obj, arm_mod.object, bake_res)
                            update_interval = int(getattr(col_settings, "sdf_dynamic_update_interval", 1))
                            sim.setup_dynamic_bone_sdf(
                                bake_res.width,
                                bake_res.height,
                                bake_res.depth,
                                bake_res.res,
                                bake_res.bone_infos,
                                dyn_data["rest_verts"],
                                dyn_data["bone_indices"],
                                dyn_data["bone_weights"],
                                dyn_data["tri_sources"],
                                dyn_data["tri_weights"],
                                dyn_data["bone_bind_inv_matrices"],
                                update_interval,
                            )
                            _bone_sdf_cache[sim_id] = (
                                arm_mod.object,
                                bake_res,
                                'DYNAMIC_GPU',
                                dyn_data.get("bone_dependency_map", []),
                                None,
                            )
                            _bone_sdf_signatures[sim_id] = current_bone_sig
                            logger.info(f"[Collider Sync] フルGPU動的SDFコライダーを初期化/更新しました (interval={update_interval})")
                        else:
                            sim.set_bone_sdf_colliders(
                                bake_res.width,
                                bake_res.height,
                                bake_res.depth,
                                bake_res.texture_bytes,
                                bake_res.bone_infos,
                            )
                            _bone_sdf_cache[sim_id] = (arm_mod.object, bake_res, 'STATIC')
                            _bone_sdf_signatures[sim_id] = current_bone_sig
                            logger.info(f"[Collider Sync] ボーンSDFコライダーを設定/更新しました: {obj.name}")
                    else:
                        bake_res = None
                        _bone_sdf_cache.pop(sim_id, None)
                        _bone_sdf_signatures.pop(sim_id, None)
                else:
                    cache_entry = _bone_sdf_cache[sim_id]
                    bake_res = cache_entry[1]

                # DYNAMIC_GPU の場合はGPU側でメッシュ変形に追従するため、関節メッシュハイブリッドの重いCPU to_mesh()評価はスキップ
                if update_mode == 'DYNAMIC_GPU':
                    continue

            elif col_settings.collider_type == 'MESH_SDF' and obj.type == 'MESH':
                current_mesh_sig = (
                    obj.name,
                    round(float(getattr(col_settings, "mesh_sdf_voxel_size", 0.004)), 4),
                    round(float(getattr(col_settings, "mesh_sdf_margin", 0.02)), 4),
                    round(float(getattr(col_settings, "thickness", 0.005)), 5),
                    round(float(getattr(col_settings, "friction", 0.5)), 3),
                    round(float(getattr(col_settings, "restitution", 0.0)), 3),
                    int(getattr(col_settings, "mesh_sdf_max_vram_mb", 256)),
                    bool(getattr(col_settings, "mesh_sdf_auto_scale", True)),
                )
                needs_rebake = (sim_id not in _bone_sdf_cache) or (_bone_sdf_signatures.get(sim_id) != current_mesh_sig)
                if needs_rebake:
                    bake_res = get_or_bake_mesh_sdf_for_object(obj, col_settings)
                    if bake_res and bake_res.depth > 0:
                        sim.set_bone_sdf_colliders(
                            bake_res.width,
                            bake_res.height,
                            bake_res.depth,
                            bake_res.texture_bytes,
                            bake_res.bone_infos,
                        )
                        _bone_sdf_cache[sim_id] = (obj, bake_res, 'MESH_SDF')
                        _bone_sdf_signatures[sim_id] = current_mesh_sig
                        logger.info(f"[Collider Sync] 単一メッシュ直方体SDFコライダーを設定/更新しました: {obj.name} ({bake_res.width}x{bake_res.height}x{bake_res.depth})")
                    else:
                        _bone_sdf_cache.pop(sim_id, None)
                        _bone_sdf_signatures.pop(sim_id, None)
                continue

                # ハイブリッドモード: 関節部メッシュ三角形を抽出してメッシュコライダーに追加（動的アクティブ化対応）
                if bake_res and getattr(col_settings, "enable_joint_mesh", True) and len(bake_res.joint_face_indices) > 0:
                    rot_threshold_deg = float(getattr(col_settings, "joint_rotation_threshold", 2.0))
                    rot_threshold_rad = math.radians(rot_threshold_deg)

                    # 動的アクティブ化判定
                    active_indices = None
                    if rot_threshold_deg <= 0.0 or not bake_res.joint_faces_by_pair:
                        # 0度以下設定またはペア情報がない場合は全関節面をアクティブ
                        active_indices = bake_res.joint_face_indices
                    else:
                        arm_obj = arm_mod.object
                        active_pair_faces = []
                        if arm_obj and arm_obj.pose:
                            for (p_name, c_name), p_faces in bake_res.joint_faces_by_pair.items():
                                pb_c = arm_obj.pose.bones.get(c_name)
                                if not pb_c:
                                    continue
                                delta_angle = 0.0
                                rot_mode = pb_c.rotation_mode
                                if rot_mode == 'QUATERNION':
                                    q = pb_c.rotation_quaternion
                                    delta_angle = 2.0 * math.acos(min(abs(float(q[0])), 1.0))
                                elif rot_mode == 'AXIS_ANGLE':
                                    delta_angle = abs(float(pb_c.rotation_axis_angle[0]))
                                else:
                                    e = pb_c.rotation_euler
                                    delta_angle = math.sqrt(float(e.x)**2 + float(e.y)**2 + float(e.z)**2)

                                if delta_angle >= rot_threshold_rad:
                                    active_pair_faces.append(p_faces)

                        if active_pair_faces:
                            active_indices = np.unique(np.concatenate(active_pair_faces))
                        else:
                            active_indices = np.empty(0, dtype=np.int32)

                    if active_indices is not None and len(active_indices) > 0:
                        eval_obj = obj.evaluated_get(depsgraph) if depsgraph else obj
                        mesh = eval_obj.to_mesh() if depsgraph else obj.data
                        mesh.calc_loop_triangles()
                        n_verts = len(mesh.vertices)
                        n_tris = len(mesh.loop_triangles)

                        if n_verts > 0 and n_tris > 0:
                            raw_coords = np.empty(n_verts * 3, dtype=np.float32)
                            mesh.vertices.foreach_get("co", raw_coords)
                            v_local = raw_coords.reshape((n_verts, 3))

                            mat_np = np.array(eval_obj.matrix_world, dtype=np.float32)
                            ones = np.ones((n_verts, 1), dtype=np.float32)
                            v_homo = np.hstack([v_local, ones])
                            v_world = (v_homo @ mat_np.T)[:, :3]

                            tri_indices = np.empty(n_tris * 3, dtype=np.int32)
                            mesh.loop_triangles.foreach_get("vertices", tri_indices)
                            tri_indices_2d = tri_indices.reshape((n_tris, 3))

                            valid_joint_idx = active_indices[active_indices < n_tris]
                            if len(valid_joint_idx) > 0:
                                joint_tri_coords = v_world[tri_indices_2d[valid_joint_idx]]
                                cur_friction = float(col_settings.friction)
                                cur_thickness = float(col_settings.thickness)
                                cur_restitution = float(getattr(col_settings, "restitution", 0.0))
                                cur_single_sided = 1.0 if getattr(col_settings, "single_sided", True) else 0.0

                                all_mesh_triangles.append(joint_tri_coords)
                                attr_row = np.array([cur_friction, cur_thickness, cur_restitution, cur_single_sided], dtype=np.float32)
                                attr_block = np.tile(attr_row, (len(valid_joint_idx), 1))
                                all_mesh_attributes.append(attr_block)

                        if depsgraph:
                            eval_obj.to_mesh_clear()

        if all_mesh_triangles:
            tri_array = np.vstack(all_mesh_triangles) if len(all_mesh_triangles) > 1 else all_mesh_triangles[0]
            attr_array = np.vstack(all_mesh_attributes) if len(all_mesh_attributes) > 1 else all_mesh_attributes[0]
            sim.set_mesh_collider_triangles(
                tri_array,
                attributes=attr_array,
            )
            logger.debug(f"[Collider Sync] sim.set_mesh_collider_triangles 実行: {len(tri_array)} 面")
        else:
            logger.debug("[Collider Sync] all_mesh_triangles は空です (0面)")

    # 毎フレームのボーン変換行列更新 (BONE_SDF / MESH_SDF)
    if sim_id in _bone_sdf_cache:
        cache_entry = _bone_sdf_cache[sim_id]
        target_obj = cache_entry[0]
        bake_res = cache_entry[1]
        mode = cache_entry[2] if len(cache_entry) >= 3 else 'STATIC'

        if mode == 'MESH_SDF' and target_obj:
            # 単一メッシュSDF: オブジェクトのワールド行列を1つの変換行列として同期
            eval_obj = target_obj.evaluated_get(depsgraph) if depsgraph else target_obj
            mat_world = np.array(eval_obj.matrix_world, dtype=np.float32)
            w_arr_row = mat_world.reshape(1, 4, 4)
            try:
                inv_arr_row = np.linalg.inv(w_arr_row)
            except np.linalg.LinAlgError:
                inv_arr_row = np.eye(4, dtype=np.float32).reshape(1, 4, 4)

            w_arr_col = np.ascontiguousarray(np.transpose(w_arr_row, (0, 2, 1)), dtype=np.float32)
            inv_arr_col = np.ascontiguousarray(np.transpose(inv_arr_row, (0, 2, 1)), dtype=np.float32)
            sim.update_bone_transforms(w_arr_col, inv_arr_col)

        elif target_obj and getattr(target_obj, "pose", None):
            eval_arm = target_obj.evaluated_get(depsgraph) if depsgraph else target_obj
            mat_arm_world = np.array(eval_arm.matrix_world, dtype=np.float32)
            world_mats = []
            for b_name in bake_res.bone_names:
                pbone = eval_arm.pose.bones.get(b_name)
                if pbone is not None:
                    p_mat = np.array(pbone.matrix, dtype=np.float32)
                    w_mat = mat_arm_world @ p_mat
                else:
                    w_mat = np.eye(4, dtype=np.float32)
                world_mats.append(w_mat)

            if world_mats:
                # [N, 4, 4] 配列として一括スタック
                w_arr_row = np.stack(world_mats)
                try:
                    # NumPy BLAS による一括バッチ逆行列計算 (個別ループより約10倍高速)
                    inv_arr_row = np.linalg.inv(w_arr_row)
                except np.linalg.LinAlgError:
                    inv_arr_row = np.tile(np.eye(4, dtype=np.float32), (len(world_mats), 1, 1))

                # WGSL mat4x4<f32> (Column-Major) に合わせて軸(1, 2)を一括転置
                w_arr_col = np.ascontiguousarray(np.transpose(w_arr_row, (0, 2, 1)), dtype=np.float32)
                inv_arr_col = np.ascontiguousarray(np.transpose(inv_arr_row, (0, 2, 1)), dtype=np.float32)

                dirty_bone_indices = None
                if len(cache_entry) >= 4 and cache_entry[2] == 'DYNAMIC_GPU':
                    dep_map = cache_entry[3]
                    prev_pose_mats = cache_entry[4] if len(cache_entry) >= 5 else None

                    # 各ボーンのポーズ行列（アーマチュアローカル空間）をスタック
                    # ※ワールド全体の平行移動・回転のみではローカルSDFは変化しないため、p_matの変化を検出
                    pose_mats_list = []
                    for b_name in bake_res.bone_names:
                        pbone = eval_arm.pose.bones.get(b_name)
                        if pbone is not None:
                            pose_mats_list.append(np.array(pbone.matrix, dtype=np.float32))
                        else:
                            pose_mats_list.append(np.eye(4, dtype=np.float32))
                    pose_mats_row = np.stack(pose_mats_list)

                    if prev_pose_mats is not None and prev_pose_mats.shape == pose_mats_row.shape:
                        diff = np.abs(pose_mats_row - prev_pose_mats)
                        bone_diff_max = np.max(diff, axis=(1, 2))
                        moved_mask = bone_diff_max > 1e-4
                        if not np.any(moved_mask):
                            # ポーズ変化なし -> Dirtyボーン 0本（SDFベイク・コピー完全スキップ: 0ms）
                            dirty_bone_indices = np.empty(0, dtype=np.uint32)
                        else:
                            moved_set = set(np.where(moved_mask)[0])
                            dirty_set = set()
                            for b_idx, deps in enumerate(dep_map):
                                for dep in deps:
                                    if dep in moved_set:
                                        dirty_set.add(b_idx)
                                        break
                            dirty_bone_indices = np.array(sorted(list(dirty_set)), dtype=np.uint32)
                    else:
                        # 初回フレーム: 全ボーン Dirty (None = 全ベイク)
                        dirty_bone_indices = None

                    # キャッシュを最新の pose_mats で更新
                    _bone_sdf_cache[sim_id] = (
                        cache_entry[0],
                        cache_entry[1],
                        cache_entry[2],
                        dep_map,
                        pose_mats_row,
                    )

                sim.update_bone_transforms(w_arr_col, inv_arr_col, dirty_bone_indices)

