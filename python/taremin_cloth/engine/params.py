"""
taremin_cloth 物理・拘束パラメータ同期モジュール
剛性・減衰・重力・自己衝突・伸縮グループ・外部追従ピンの各プロパティをGPUシミュレータに同期する。
"""

import bpy
import numpy as np
from .cache import (
    _prev_elastic_scales,
    _prev_stiffness_cache,
    _attachment_pin_indices_cache,
    _prev_attachment_pin_targets,
)
from ..utils.logger import logger


def sync_cloth_parameters(sim, obj, scene=None):
    """シミュレーション実行中にNパネルのプロパティ変更やシーン重力をGPUへ同期する"""
    settings = getattr(obj, "taremin_cloth", None)
    if not settings:
        return

    # 1. 剛性4種 (引張・圧縮・せん断・曲げ)
    curr_stiffness = (
        float(settings.tension_stiffness),
        float(settings.compression_stiffness),
        float(settings.shear_stiffness),
        float(settings.bending_stiffness),
    )
    if _prev_stiffness_cache.get(obj.name) != curr_stiffness:
        if hasattr(sim, "set_stiffness_all"):
            sim.set_stiffness_all(
                curr_stiffness[0],
                curr_stiffness[1],
                curr_stiffness[2],
                curr_stiffness[3],
            )
        elif hasattr(sim, "set_stiffness"):
            sim.set_stiffness(curr_stiffness[0], curr_stiffness[3])
        _prev_stiffness_cache[obj.name] = curr_stiffness

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
        try:
            max_iters = int(getattr(settings, "self_collision_max_iterations", 256))
        except (ValueError, TypeError):
            max_iters = 256
        sim.set_self_collision_options(
            relief_factor=getattr(settings, "self_collision_relief_factor", 0.2),
            max_displacement_ratio=getattr(settings, "self_collision_max_displacement_ratio", 0.2),
            exclude_neighbors=getattr(settings, "self_collision_exclude_neighbors", True),
            enable_normal_untangling=getattr(settings, "enable_normal_untangling", True),
            max_iterations=max_iters,
        )
    if hasattr(sim, "set_coupled_self_collision_options"):
        mode_str = getattr(settings, "coupled_self_collision_mode", "RELAXATION")
        if mode_str == "OFF":
            mode_int = 0
            relax_iters = 0
        elif mode_str == "FULL_COUPLED":
            mode_int = 3
            relax_iters = getattr(settings, "post_collision_relaxation_iters", 1)
            if relax_iters == 0:
                relax_iters = 1
        else:  # "RELAXATION"
            mode_int = 1
            relax_iters = getattr(settings, "post_collision_relaxation_iters", 2)
            if relax_iters == 0:
                relax_iters = 2
        sim.set_coupled_self_collision_options(mode_int, relax_iters)
    if hasattr(sim, "set_self_collision_substep_interval"):
        interval = int(getattr(settings, "self_collision_substep_interval", 1))
        sim.set_self_collision_substep_interval(interval)

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

    # 5.7. 縫合パラメータ (剛性・密着ロック)
    if hasattr(sim, "set_sewing_stiffness"):
        sim.set_sewing_stiffness(getattr(settings, "sewing_stiffness", 10000.0))
    if hasattr(sim, "set_enable_sewing_lock"):
        sim.set_enable_sewing_lock(getattr(settings, "enable_sewing_lock", True))

    # 6. 伸縮グループ (Elastic Bands / Edge Scaling)
    sync_elastic_groups(sim, obj)


def sync_elastic_groups(sim, obj):
    """伸縮グループ（ゴム紐）の自然長スケールをGPUシミュレータに同期する（平滑化追従対応）"""
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

    is_first_sync = not prev_map.get("_is_synced", False)
    _prev_elastic_scales[obj_name] = prev_map
    if has_change or is_first_sync:
        sim.set_edge_rest_length_scales(
            np.array(indices, dtype=np.uint32),
            np.array(current_scales, dtype=np.float32),
        )
        prev_map["_is_synced"] = True


def sync_attachment_pins(sim, obj, scene):
    settings = getattr(obj, "taremin_cloth", None)
    if not settings or not settings.pin_target_object:
        return

    target_obj = settings.pin_target_object
    vg_name = settings.pin_vertex_group or "Pin"
    vg = obj.vertex_groups.get(vg_name)
    if not vg:
        return

    target_mat = target_obj.matrix_world
    if settings.pin_target_bone and target_obj.type == "ARMATURE" and target_obj.pose:
        bone = target_obj.pose.bones.get(settings.pin_target_bone)
        if bone:
            target_mat = target_mat @ bone.matrix

    inv_world = obj.matrix_world.inverted()
    local_target = inv_world @ target_mat.translation
    loc_tuple = (round(local_target.x, 5), round(local_target.y, 5), round(local_target.z, 5))

    prev_loc = _prev_attachment_pin_targets.get(obj.name)
    if prev_loc == loc_tuple:
        return
    _prev_attachment_pin_targets[obj.name] = loc_tuple

    mesh = obj.data
    cache_key = (obj.name, vg_name, len(mesh.vertices))
    pinned_entries = _attachment_pin_indices_cache.get(cache_key)
    if pinned_entries is None:
        pinned_entries = []
        vg_idx = vg.index
        for v in mesh.vertices:
            for g in v.groups:
                if g.group == vg_idx and g.weight > 0.0:
                    pinned_entries.append((v.index, float(g.weight)))
                    break
        _attachment_pin_indices_cache[cache_key] = pinned_entries

    if not pinned_entries:
        return

    target_xyz = [local_target.x, local_target.y, local_target.z]
    for v_idx, weight in pinned_entries:
        sim.set_pin(v_idx, target_xyz, weight)
