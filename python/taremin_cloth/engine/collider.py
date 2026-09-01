"""
taremin_cloth コライダー同期モジュール
Blenderシーン内のコライダーオブジェクト（Sphere, Plane, Capsule, Mesh）を検出し、
ワールド座標変換を行ってGPUシミュレータに差分同期する。
"""

import bpy
import numpy as np
from ..utils.logger import logger

_collider_cache = {}


def clear_collider_cache():
    """コライダーキャッシュをクリアする"""
    global _collider_cache
    _collider_cache.clear()


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
