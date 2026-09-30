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
# _prev_wrinkle_signatures[obj_name] は下記のキャッシュ辞書である:
# {
#   "desc": [WrinkleCurveDesc, ...] (骨ローカル・照合用),
#   "tex": WrinkleTexture2D (再送不要時のrange参照用),
#   "width"/"height": ベイク寸法 (同一寸法判定用),
#   "z_min/z_max/r_min/r_max": float (スライド適用後の現在range),
#   "bone_radius"/"cloth_radius": float,
# }
# 等価判定は wrinkles_descs_match（許容値付き、量子化境界の非決定性なし）。


def _wrinkle_pose_params(bone_head, axis_n, normal, binormal, bone_rad, stiffness,
                         valley_window=0.015, crest_window=0.020):
    return dict(
        origin=[float(bone_head[0]), float(bone_head[1]), float(bone_head[2])],
        axis=[float(axis_n[0]), float(axis_n[1]), float(axis_n[2])],
        normal=[float(normal[0]), float(normal[1]), float(normal[2])],
        binormal=[float(binormal[0]), float(binormal[1]), float(binormal[2])],
        influence_radius=0.03,
        bone_radius=float(bone_rad),
        stiffness=float(stiffness),
        blend_weight=1.0,
        valley_window=float(valley_window),
        crest_window=float(crest_window),
    )


def _wrinkle_depths(settings):
    """設定から谷/山グラデーション窓 (m) を読む。旧ファイルの欠損時は既定値。"""
    try:
        v = float(getattr(settings, "wrinkle_valley_depth", 0.015) or 0.015)
    except Exception:
        v = 0.015
    try:
        c = float(getattr(settings, "wrinkle_crest_depth", 0.020) or 0.020)
    except Exception:
        c = 0.020
    return max(v, 0.001), max(c, 0.001)


def sync_wrinkle_field(sim, obj):
    """シワフィールド設定および2D-SDFテクスチャをGPUシミュレータに同期する。

    ベイク回避設計:
    - 照合は骨ローカル記述子のみで行うため、骨とカーブの剛体追従
      （アニメ・スライド）ではベイクせず uniform 更新のみとなる。
    - 純粋スライド (dz, r_scale) および一様強度変化は z_range/r_range/stiffness
      の uniform 更新のみで厳密等価のため再送しない。
    - texture upload は形状実変時のみ。これにより MOUSEMOVE 毎の数百ms
      ブロックが消える。
    """
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
    valley_depth, crest_depth = _wrinkle_depths(settings)
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
        describe_wrinkle_curves,
        wrinkles_descs_match,
        match_slide_transform,
    )
    crest_curves, root_curves = extract_curves_from_blender_collection(col_name)
    if not crest_curves and not root_curves:
        if obj.name in _prev_wrinkle_signatures:
            sim.set_enable_wrinkle_field(False)
            del _prev_wrinkle_signatures[obj.name]
        return

    # カーブ帰属ボーンの全会一致による上書き:
    # 全カーブが同一ボーンを指す場合は設定値よりそちらをベイク座標系に使う。
    # 設定値の陳腐化（別ボーン残置）による別位置フィールド・別位置プロキシを防ぐ。
    try:
        _curve_bones = {str(getattr(c, "target_bone", "") or "")
                        for c in list(crest_curves) + list(root_curves)}
        _curve_bones.discard("")
    except Exception:
        _curve_bones = set()
    if len(_curve_bones) == 1:
        _unanimous = next(iter(_curve_bones))
        try:
            _in_pose = _unanimous in arm_obj.pose.bones
        except Exception:
            _in_pose = False
        if _in_pose and _unanimous != bone_name:
            bone_name = _unanimous
            pb = arm_obj.pose.bones[bone_name]
            bone_head = np.array((w_mat @ pb.head)[:3], dtype=np.float32)
            bone_tail = np.array((w_mat @ pb.tail)[:3], dtype=np.float32)
            explicit_rad = float(getattr(settings, "wrinkle_bone_radius", 0.0) or 0.0)
            bone_rad = resolve_bone_radius(
                explicit_rad,
                float(getattr(pb.bone, "head_radius", 0.0) or 0.0),
                float(getattr(pb.bone, "tail_radius", 0.0) or 0.0),
                float(getattr(pb, "length", 0.0) or 0.0),
            )

    # ボーン直交基底の算出 (GPU WrinkleFieldParams と 1:1 対応)
    axis_raw = bone_tail - bone_head
    if float(np.linalg.norm(axis_raw)) < 1e-6:
        # 縮退ボーンでは円柱座標系が定義できないため無効化する
        if obj.name in _prev_wrinkle_signatures:
            sim.set_enable_wrinkle_field(False)
            del _prev_wrinkle_signatures[obj.name]
        return
    _axis_n, _normal, _binormal = build_orthonormal_basis(axis_raw)

    # 骨ローカル形状記述子（姿勢不変・テクスチャ内容を決定する最小集合）
    tex_w, tex_h = 128, 128
    cloth_rad = bone_rad + 0.02
    desc = describe_wrinkle_curves(
        crest_curves, root_curves, bone_head, _axis_n, _normal,
        default_influence_radius=0.03,
    )
    if not desc:
        if obj.name in _prev_wrinkle_signatures:
            sim.set_enable_wrinkle_field(False)
            del _prev_wrinkle_signatures[obj.name]
        return
    cached = _prev_wrinkle_signatures.get(obj.name)

    if (cached is not None
            and cached.get("width") == tex_w
            and cached.get("height") == tex_h
            and cached.get("bone_radius") == float(bone_rad)
            and cached.get("cloth_radius") == float(cloth_rad)
            and wrinkles_descs_match(cached.get("desc", []), desc)):
        # 剛体追従: テクスチャ再送なし・uniform のみ更新
        sim.set_wrinkle_field_params(
            **_wrinkle_pose_params(bone_head, _axis_n, _normal, _binormal, bone_rad, 30.0 * global_str,
                                  valley_depth, crest_depth),
            z_min=float(cached["z_min"]),
            z_max=float(cached["z_max"]),
            r_min=float(cached["r_min"]),
            r_max=float(cached["r_max"]),
        )
        sim.set_enable_wrinkle_field(True)
        return

    if cached is not None and cached.get("desc"):
        m = match_slide_transform(cached["desc"], desc)
        if m is not None:
            # スライド等価変換: range/stiffness の uniform 更新のみで厳密等価
            # target' = a*target + b より range' = a*range + b（B/Aは不変）
            dz = float(m["dz"])
            k = float(m["r_scale"])
            b = float(m["r_offset"])
            s = float(m["strength_scale"])
            z_min = float(cached["z_min"]) + dz
            z_max = float(cached["z_max"]) + dz
            r_min = float(k * cached["r_min"] + b)
            r_max = float(k * cached["r_max"] + b)
            sim.set_wrinkle_field_params(
                **_wrinkle_pose_params(bone_head, _axis_n, _normal, _binormal, bone_rad, 30.0 * global_str * s,
                                      valley_depth, crest_depth),
                z_min=z_min,
                z_max=z_max,
                r_min=r_min,
                r_max=r_max,
            )
            sim.set_enable_wrinkle_field(True)
            cached["desc"] = desc
            cached["z_min"] = z_min
            cached["z_max"] = z_max
            cached["r_min"] = r_min
            cached["r_max"] = r_max
            return

    # 形状実変: フルベイク（splat高速化済み）
    tex_2d = bake_wrinkle_2d_sdf_texture(
        crest_curves=crest_curves,
        root_curves=root_curves,
        origin=bone_head,
        axis=_axis_n,
        normal=_normal,
        width=tex_w,
        height=tex_h,
        influence_radius=0.03,
        bone_radius=float(bone_rad),
        cloth_radius=float(cloth_rad),
    )

    # GPU Uniform へボーン基底・剛性を同期する (縮退射影の防止)。
    # set_wrinkle_field_texture_2d は z_range/r_range のみ上書きするため、
    # bone_origin/axis/normal/binormal は本呼び出しで必ず設定すること。
    sim.set_wrinkle_field_params(
        **_wrinkle_pose_params(bone_head, _axis_n, _normal, _binormal, bone_rad, 30.0 * global_str,
                              valley_depth, crest_depth),
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
    _prev_wrinkle_signatures[obj.name] = {
        "desc": desc,
        "tex": tex_2d,
        "width": tex_w,
        "height": tex_h,
        "z_min": float(tex_2d.z_min),
        "z_max": float(tex_2d.z_max),
        "r_min": float(tex_2d.r_min),
        "r_max": float(tex_2d.r_max),
        "bone_radius": float(bone_rad),
        "cloth_radius": float(cloth_rad),
    }


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
