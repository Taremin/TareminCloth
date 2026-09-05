"""
taremin_cloth コライダー同期モジュール
Blenderシーン内のコライダーオブジェクト（Sphere, Plane, Capsule, Mesh）を検出し、
ワールド座標変換を行ってGPUシミュレータに差分同期する。
"""

import bpy
import numpy as np
from ..utils.logger import logger
from .sdf_baker import get_or_bake_bone_sdf_for_object, get_armature_modifier, BoneSdfBakeResult

_collider_cache = {}
_bone_sdf_cache = {}


def clear_collider_cache():
    """コライダーキャッシュをクリアする"""
    global _collider_cache, _bone_sdf_cache
    _collider_cache.clear()
    _bone_sdf_cache.clear()


def sync_colliders(sim, scene, depsgraph=None, force=False, cloth_obj=None):
    """シーン内のコライダーオブジェクトをシミュレータに同期する（差分キャッシュ・ブロードフェーズ・ボーンSDF対応）"""
    global _collider_cache, _bone_sdf_cache

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
        col_settings = getattr(obj, "taremin_collider", None)
        if col_settings and col_settings.is_collider and getattr(col_settings, "enabled", True):
            collider_objs.append((obj, col_settings))
            anim_s = getattr(col_settings, "anim", None)
            if anim_s and anim_s.enabled:
                has_active_anim = True
            if col_settings.collider_type == 'BONE_SDF':
                has_bone_sdf = True
            elif col_settings.collider_type == 'MESH' and obj.type == 'MESH':
                # Armature変形、オブジェクトアニメーション、シェイプキーアニメーションがある場合は毎フレーム更新
                if get_armature_modifier(obj) is not None:
                    has_active_anim = True
                elif obj.animation_data and obj.animation_data.action:
                    has_active_anim = True
                elif getattr(obj.data, "shape_keys", None) and obj.data.shape_keys.animation_data:
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
                getattr(col_settings, "sdf_resolution", "64"),
                round(getattr(col_settings, "sdf_margin", 0.2), 3),
                round(getattr(col_settings, "weight_threshold", 0.02), 4),
                round(getattr(col_settings, "blend_k", 0.05), 4),
            ))

    sim_id = id(sim)
    state_key = tuple(col_state)
    state_changed = force or has_active_anim or (_collider_cache.get(sim_id) != state_key)

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
                    tri_coords = v_world[tri_indices_2d]

                    all_mesh_triangles.append(tri_coords)
                    attr_row = np.array([cur_friction, cur_thickness, cur_restitution, cur_single_sided], dtype=np.float32)
                    attr_block = np.tile(attr_row, (n_tris, 1))
                    all_mesh_attributes.append(attr_block)

                if depsgraph:
                    eval_obj.to_mesh_clear()

            elif col_settings.collider_type == 'BONE_SDF':
                arm_mod = get_armature_modifier(obj)
                if not arm_mod:
                    logger.warning(f"[Collider Sync] '{obj.name}' にArmatureモディファイアがありません。BONE_SDFをスキップします。")
                    continue

                bake_res = get_or_bake_bone_sdf_for_object(obj, col_settings)
                if bake_res and bake_res.depth > 0:
                    sim.set_bone_sdf_colliders(
                        bake_res.width,
                        bake_res.height,
                        bake_res.depth,
                        bake_res.texture_bytes,
                        bake_res.bone_infos,
                    )
                    _bone_sdf_cache[sim_id] = (arm_mod.object, bake_res)

        if all_mesh_triangles:
            tri_array = np.vstack(all_mesh_triangles) if len(all_mesh_triangles) > 1 else all_mesh_triangles[0]
            attr_array = np.vstack(all_mesh_attributes) if len(all_mesh_attributes) > 1 else all_mesh_attributes[0]
            sim.set_mesh_collider_triangles(
                tri_array,
                attributes=attr_array,
            )

    # 毎フレームのボーン変換行列更新 (BONE_SDF)
    if sim_id in _bone_sdf_cache:
        arm_obj, bake_res = _bone_sdf_cache[sim_id]
        if arm_obj and getattr(arm_obj, "pose", None):
            eval_arm = arm_obj.evaluated_get(depsgraph) if depsgraph else arm_obj
            mat_arm_world = np.array(eval_arm.matrix_world, dtype=np.float32)

            world_mats = []
            inv_mats = []
            for b_name in bake_res.bone_names:
                pbone = eval_arm.pose.bones.get(b_name)
                if pbone is not None:
                    p_mat = np.array(pbone.matrix, dtype=np.float32)
                    w_mat = mat_arm_world @ p_mat
                else:
                    w_mat = np.eye(4, dtype=np.float32)

                try:
                    inv_mat = np.linalg.inv(w_mat)
                except np.linalg.LinAlgError:
                    inv_mat = np.eye(4, dtype=np.float32)

                # WGSL mat4x4<f32> (Column-Major) に合わせて転置して格納
                world_mats.append(np.ascontiguousarray(w_mat.T, dtype=np.float32))
                inv_mats.append(np.ascontiguousarray(inv_mat.T, dtype=np.float32))

            if world_mats:
                w_arr = np.stack(world_mats)
                inv_arr = np.stack(inv_mats)
                sim.update_bone_transforms(w_arr, inv_arr)

