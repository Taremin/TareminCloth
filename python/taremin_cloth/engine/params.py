"""
taremin_cloth 物理・拘束パラメータ同期モジュール
剛性・減衰・重力・自己衝突・伸縮グループ・外部追従ピンの各プロパティをGPUシミュレータに同期する。
設定値の収集は engine.simconfig.collect_sim_config に一本化している。
"""

import bpy
import numpy as np
from .cache import (
    _prev_elastic_scales,
    _attachment_pin_indices_cache,
    _prev_attachment_pin_targets,
)
from .simconfig import (
    collect_sim_config,
    sim_config_signature,
    apply_sim_config,
    is_config_current,
    mark_config_applied,
)
from ..utils.logger import logger


def sync_cloth_parameters(sim, obj, scene=None):
    """シミュレーション実行中にNパネルのプロパティ変更やシーン重力をGPUへ同期する"""
    settings = getattr(obj, "taremin_cloth", None)
    if not settings:
        return

    # 設定収集 (単一経路) と差分スキップ
    cfg = collect_sim_config(settings, scene)
    if scene is None:
        # 従来動作の維持: sceneなしでは重力に触れない
        cfg.pop("gravity", None)
    sig = sim_config_signature(cfg)
    if not is_config_current(obj.name, sig):
        apply_sim_config(sim, cfg)
        mark_config_applied(obj.name, sig)

    # 転送最適化フラグ (物理外のためSimConfig対象外、従来通り毎フレーム同期)
    if hasattr(sim, "set_enable_compact_readback"):
        sim.set_enable_compact_readback(getattr(settings, "enable_compact_readback", True))

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
