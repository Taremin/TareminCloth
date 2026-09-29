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

    # 7. シワフィールド (Wrinkle Field 2D-SDF Texture)
    sync_wrinkle_field(sim, obj)


_prev_wrinkle_signatures = {}


def sync_wrinkle_field(sim, obj):
    """シワフィールド設定および2D-SDFテクスチャをGPUシミュレータに同期する"""
    settings = getattr(obj, "taremin_cloth", None)
    if not hasattr(sim, "set_wrinkle_field_texture_2d") or not hasattr(sim, "set_enable_wrinkle_field"):
        return

    if not settings or not getattr(settings, "use_wrinkle_field", False):
        if obj.name in _prev_wrinkle_signatures:
            sim.set_enable_wrinkle_field(False)
            del _prev_wrinkle_signatures[obj.name]
        return

    # コレクション名とアーマチュアの特定
    col_name = settings.wrinkle_collection.name if settings.wrinkle_collection else "TareminCloth_Wrinkles"
    global_str = float(getattr(settings, "wrinkle_global_strength", 1.0))
    arm_obj = getattr(settings, "wrinkle_armature", None)
    bone_name = getattr(settings, "wrinkle_bone_name", "")

    if not arm_obj:
        for o in bpy.data.objects:
            if o.type == 'ARMATURE':
                arm_obj = o
                break

    if not arm_obj or not arm_obj.pose or not arm_obj.pose.bones:
        return

    if not bone_name or bone_name not in arm_obj.pose.bones:
        bone_name = arm_obj.pose.bones[0].name

    pb = arm_obj.pose.bones[bone_name]
    w_mat = arm_obj.matrix_world
    bone_head = np.array((w_mat @ pb.head)[:3], dtype=np.float32)
    bone_tail = np.array((w_mat @ pb.tail)[:3], dtype=np.float32)
    from .wrinkle_field import resolve_bone_radius
    explicit_rad = float(getattr(settings, "wrinkle_bone_radius", 0.0) or 0.0)
    bone_rad = resolve_bone_radius(
        explicit_rad,
        float(getattr(pb.bone, "head_radius", 0.0) or 0.0),
        float(getattr(pb.bone, "tail_radius", 0.0) or 0.0),
        float(getattr(pb, "length", 0.0) or 0.0),
    )

    # コレクションからシワカーブを抽出
    from .wrinkle_field import (
        extract_curves_from_blender_collection,
        bake_wrinkle_2d_sdf_texture,
        build_orthonormal_basis,
    )
    crest_curves, root_curves = extract_curves_from_blender_collection(col_name)
    if not crest_curves and not root_curves:
        if obj.name in _prev_wrinkle_signatures:
            sim.set_enable_wrinkle_field(False)
            del _prev_wrinkle_signatures[obj.name]
        return

    # ボーン直交基底の算出 (GPU WrinkleFieldParams と 1:1 対応)
    axis_raw = bone_tail - bone_head
    if float(np.linalg.norm(axis_raw)) < 1e-6:
        # 縮退ボーンでは円柱座標系が定義できないため無効化する
        if obj.name in _prev_wrinkle_signatures:
            sim.set_enable_wrinkle_field(False)
            del _prev_wrinkle_signatures[obj.name]
        return
    _axis_n, _normal, _binormal = build_orthonormal_basis(axis_raw)

    # 差分キャッシュのチェック (点数や座標、強度に変更がなければベイクをスキップ)
    # 署名は軽量に保つ: 全点 tobytes() の毎フレームコピーを避け、shape + 先頭/末尾サンプルで検出する。
    # 中盤のみの微編集を見逃す可能性は残るが、Blender側のカーブ編集は通常 全点再生成を伴うため実用上十分。
    sig_parts = [
        col_name, global_str,
        bone_head.tobytes(), bone_tail.tobytes(), bone_rad, explicit_rad,
        float(_normal[0]), float(_normal[1]), float(_normal[2]),
        128, 128,
    ]
    for c in crest_curves:
        pts = np.asarray(c.points, dtype=np.float32)
        sig_parts.extend([
            c.points.shape, c.strength, c.influence_radius,
            pts.tobytes()[:64], pts.tobytes()[-64:] if pts.size else b"",
        ])
    for r in root_curves:
        pts = np.asarray(r.points, dtype=np.float32)
        sig_parts.extend([
            r.points.shape, r.strength, r.influence_radius,
            pts.tobytes()[:64], pts.tobytes()[-64:] if pts.size else b"",
        ])
    current_sig = hash(tuple(sig_parts))

    if _prev_wrinkle_signatures.get(obj.name) == current_sig:
        return

    # 布の推定半径
    cloth_rad = bone_rad + 0.02
    tex_2d = bake_wrinkle_2d_sdf_texture(
        crest_curves=crest_curves,
        root_curves=root_curves,
        origin=bone_head,
        axis=_axis_n,
        normal=_normal,
        width=128,
        height=128,
        influence_radius=0.03,
        bone_radius=float(bone_rad),
        cloth_radius=float(cloth_rad),
    )

    # GPU Uniform へボーン基底・剛性を同期する (縮退射影の防止)。
    # set_wrinkle_field_texture_2d は z_range/r_range のみ上書きするため、
    # bone_origin/axis/normal/binormal は本呼び出しで必ず設定すること。
    sim.set_wrinkle_field_params(
        origin=[float(bone_head[0]), float(bone_head[1]), float(bone_head[2])],
        axis=[float(_axis_n[0]), float(_axis_n[1]), float(_axis_n[2])],
        normal=[float(_normal[0]), float(_normal[1]), float(_normal[2])],
        binormal=[float(_binormal[0]), float(_binormal[1]), float(_binormal[2])],
        influence_radius=0.03,
        bone_radius=float(bone_rad),
        stiffness=30.0 * global_str,
        blend_weight=1.0,
    )
    sim.set_wrinkle_field_texture_2d(
        tex_2d.width,
        tex_2d.height,
        tex_2d.texture_bytes,
        tex_2d.z_min,
        tex_2d.z_max,
        tex_2d.r_min,
        tex_2d.r_max,
    )
    sim.set_enable_wrinkle_field(True)
    _prev_wrinkle_signatures[obj.name] = current_sig


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
