"""
taremin_cloth 物理・拘束パラメータ同期モジュール
剛性・減衰・重力・自己衝突・伸縮グループ・外部追従ピンの各プロパティをGPUシミュレータに同期する。
設定値の収集は engine.simconfig.collect_sim_config に一本化している。
"""

import bpy
import numpy as np
from .cache import (
    _prev_elastic_scales,
    _prev_elastic_multipliers,
    _prev_string_scales,
    _attachment_pin_indices_cache,
    _prev_attachment_pin_targets,
    _prev_attachment_pin_indices,
    clear_attachment_pin_cache,
)
from .simconfig import (
    collect_sim_config,
    sim_config_signature,
    apply_sim_config,
    is_config_current,
    mark_config_applied,
)
from ..utils.args_guard import allow_positional
from ..utils.logger import logger


def sync_cloth_parameters(sim, obj, *, scene=None):
    """シミュレーション実行中にNパネルのプロパティ変更やシーン重力をGPUへ同期する"""
    settings = getattr(obj, "taremin_cloth", None)
    if not settings:
        return

    # 設定収集 (単一経路) と差分スキップ
    cfg = collect_sim_config(settings, scene=scene)
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

    # 6.5. 紐パーツ (Drawstring / Strap Edge Scaling & Cinch Transfer)
    sync_string_parts(sim, obj)

    # 7. ドレープガイド (Wrinkle Field / Drape Guide 2D-SDF Texture)
    sync_wrinkle_field(sim, obj)


_prev_wrinkle_signatures = {}
_multi_bone_warned_objs = set()


def clear_wrinkle_signature_cache(obj_name=None):
    """シミュレータ破棄時にドレープガイドのベイク署名キャッシュを消去する"""
    if obj_name is None:
        _prev_wrinkle_signatures.clear()
        _multi_bone_warned_objs.clear()
    else:
        _prev_wrinkle_signatures.pop(obj_name, None)
        _multi_bone_warned_objs.discard(obj_name)

# _prev_wrinkle_signatures[obj_name] は下記のキャッシュ辞書である:
# {
#   "desc": [WrinkleCurveDesc, ...] (骨ローカル・照合用),
#   "tex": WrinkleTexture2D (再送不要時のrange参照用),
#   "width"/"height": ベイク寸法 (同一寸法判定用),
#   "z_min/z_max/r_min/r_max": float (スライド適用後の現在range),
#   "bone_radius"/"cloth_radius": float,
# }
# 等価判定は wrinkles_descs_match（許容値付き、量子化境界の非決定性なし）。


@allow_positional
def _wrinkle_pose_params(bone_head, axis_n, normal, binormal, bone_rad, stiffness,
                         valley_window=0.015, crest_window=0.020, influence_radius=0.03):
    return dict(
        origin=[float(bone_head[0]), float(bone_head[1]), float(bone_head[2])],
        axis=[float(axis_n[0]), float(axis_n[1]), float(axis_n[2])],
        normal=[float(normal[0]), float(normal[1]), float(normal[2])],
        binormal=[float(binormal[0]), float(binormal[1]), float(binormal[2])],
        influence_radius=float(influence_radius),
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


def _is_wrinkle_field_experimental_enabled() -> bool:
    """ドレープガイド実験フラグの有効判定（preferencesへ委譲）"""
    try:
        from ..preferences import is_wrinkle_field_enabled
        return bool(is_wrinkle_field_enabled())
    except Exception:
        return True


def sync_wrinkle_field(sim, obj):
    """ドレープガイド設定および2D-SDFテクスチャをGPUシミュレータに同期する。

    ベイク回避設計:
    - 照合は骨ローカル記述子のみで行うため、骨とカーブの剛体追従
      （ポーズ追従・スライド）ではベイクせず uniform 更新のみとなる。
    - 純粋スライド (dz, r_scale) および一様強度変化は z_range/r_range/stiffness
      の uniform 更新のみで厳密等価のため再送しない。
    - texture upload は形状実変時のみ。これにより MOUSEMOVE 毎の数百ms
      ブロックが消える。
    """
    settings = getattr(obj, "taremin_cloth", None)
    if not hasattr(sim, "set_wrinkle_field_texture_2d") or not hasattr(sim, "set_enable_wrinkle_field"):
        return

    # 実験的機能ゲート: 無効時は旧ファイルの use_wrinkle_field=True でも強制無効化
    if not _is_wrinkle_field_experimental_enabled():
        if obj.name in _prev_wrinkle_signatures:
            try:
                sim.set_enable_wrinkle_field(False)
            except Exception:
                pass
            del _prev_wrinkle_signatures[obj.name]
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
        resolve_pose_bone_normal_hint,
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

    # カーブ帰属ボーンの確認:
    # 1) 全カーブが同一ボーンを指す場合は設定値よりそちらをベイク座標系に使う。
    # 2) 複数ボーンが混在している場合は、設定ボーンに属するカーブのみに絞り込み、
    #    設定ボーンに一致するカーブが無い（帰属不一致）ときは設定ボーン座標系へフォールバックする。
    try:
        _curve_bones = {str(getattr(c, "target_bone", "") or "")
                        for c in list(crest_curves) + list(root_curves)}
        _curve_bones.discard("")
    except Exception:
        _curve_bones = set()
    _target_override_bone = None
    if len(_curve_bones) == 1:
        _target_override_bone = next(iter(_curve_bones))
    elif len(_curve_bones) > 1:
        if obj.name not in _multi_bone_warned_objs:
            _multi_bone_warned_objs.add(obj.name)
            logger.warning(
                f"[Wrinkle] Collection '{col_name}' contains curves for multiple bones "
                f"{sorted(_curve_bones)}; resolving against setting bone '{bone_name}'."
            )
        if bone_name in _curve_bones:
            crest_curves = [c for c in crest_curves if (getattr(c, "target_bone", "") or "") in ("", bone_name)]
            root_curves = [c for c in root_curves if (getattr(c, "target_bone", "") or "") in ("", bone_name)]
            if not crest_curves and not root_curves:
                if obj.name in _prev_wrinkle_signatures:
                    sim.set_enable_wrinkle_field(False)
                    del _prev_wrinkle_signatures[obj.name]
                return
        # else: 設定ボーンがカーブ群に含まれない帰属不一致時は _target_override_bone を設定せず、
        #       設定ボーン bone_name へフォールバックする。

    if _target_override_bone:
        try:
            _in_pose = _target_override_bone in arm_obj.pose.bones
        except Exception:
            _in_pose = False
        if _in_pose and _target_override_bone != bone_name:
            bone_name = _target_override_bone
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

    # ボーン直交基底の算出 (GPU WrinkleFieldParams と 1:1 対応。ボーンRoll追従・90度フリップ防止)
    axis_raw = bone_tail - bone_head
    if float(np.linalg.norm(axis_raw)) < 1e-6:
        # 縮退ボーンでは円柱座標系が定義できないため無効化する
        if obj.name in _prev_wrinkle_signatures:
            sim.set_enable_wrinkle_field(False)
            del _prev_wrinkle_signatures[obj.name]
        return
    _normal_hint = resolve_pose_bone_normal_hint(arm_obj, pb)
    if _normal_hint is not None:
        _axis_n, _normal, _binormal = build_orthonormal_basis(axis_raw, normal_hint=_normal_hint)
    else:
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
    max_inf_r = max((float(getattr(d, "influence_radius", 0.03) or 0.03) for d in desc), default=0.03)
    cached = _prev_wrinkle_signatures.get(obj.name)

    if (cached is not None
            and cached.get("width") == tex_w
            and cached.get("height") == tex_h
            and cached.get("bone_radius") == float(bone_rad)
            and cached.get("cloth_radius") == float(cloth_rad)
            and wrinkles_descs_match(cached.get("desc", []), desc)):
        # 剛体追従: テクスチャ再送なし・uniform のみ更新（累積スライド強度を保持）
        strength_scale = float(cached.get("strength_scale", 1.0))
        sim.set_wrinkle_field_params(
            **_wrinkle_pose_params(bone_head, _axis_n, _normal, _binormal, bone_rad,
                                  30.0 * global_str * strength_scale,
                                  valley_depth, crest_depth, max_inf_r),
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
            cum_scale = float(cached.get("strength_scale", 1.0)) * s
            z_min = float(cached["z_min"]) + dz
            z_max = float(cached["z_max"]) + dz
            r_min = float(k * cached["r_min"] + b)
            r_max = float(k * cached["r_max"] + b)
            sim.set_wrinkle_field_params(
                **_wrinkle_pose_params(bone_head, _axis_n, _normal, _binormal, bone_rad,
                                      30.0 * global_str * cum_scale,
                                      valley_depth, crest_depth, max_inf_r),
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
            cached["strength_scale"] = cum_scale
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
        influence_radius=max_inf_r,
        bone_radius=float(bone_rad),
        cloth_radius=float(cloth_rad),
    )

    # GPU Uniform へボーン基底・剛性を同期する (縮退射影の防止)。
    # set_wrinkle_field_texture_2d は z_range/r_range のみ上書きするため、
    # bone_origin/axis/normal/binormal は本呼び出しで必ず設定すること。
    sim.set_wrinkle_field_params(
        **_wrinkle_pose_params(bone_head, _axis_n, _normal, _binormal, bone_rad, 30.0 * global_str,
                              valley_depth, crest_depth, max_inf_r),
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
        "strength_scale": 1.0,
    }


def sync_elastic_groups(sim, obj):
    """伸縮グループ（ゴム紐）の自然長スケールと剛性倍率をGPUシミュレータに同期する（平滑化追従対応）"""
    settings = getattr(obj, "taremin_cloth", None)
    if not settings or not hasattr(sim, "set_edge_rest_length_scales"):
        return

    obj_name = obj.name
    groups = getattr(settings, "elastic_groups", None)
    has_mult_api = hasattr(sim, "set_edge_stiffness_multipliers")

    # 各グループの有効なエッジ・スケール・剛性倍率を収集
    edge_scale_map = {}
    edge_mult_map = {}
    if groups:
        for g in groups:
            if not getattr(g, "enabled", True):
                continue
            edges = g.get_edge_indices()
            scale = float(getattr(g, "scale", 1.0))
            mult = max(0.001, float(getattr(g, "stiffness_multiplier", 1.0)))
            for e_idx in edges:
                edge_scale_map[e_idx] = scale
                edge_mult_map[e_idx] = mult

    prev_map = _prev_elastic_scales.get(obj_name, {})
    prev_mult_map = _prev_elastic_multipliers.get(obj_name, {})
    prev_edge_keys = [k for k in prev_map.keys() if isinstance(k, int)]
    removed_edges = [k for k in prev_edge_keys if k not in edge_scale_map]

    # 削除または無効化されたグループのエッジだけを個別に scale=1.0, multiplier=1.0 へ戻す
    # (sim.reset_edge_rest_lengths() を呼ぶと紐パーツ string_parts の収縮まで巻き添えにするため)
    if removed_edges:
        sim.set_edge_rest_length_scales(
            np.array(removed_edges, dtype=np.uint32),
            np.ones(len(removed_edges), dtype=np.float32),
        )
        if has_mult_api:
            sim.set_edge_stiffness_multipliers(
                np.array(removed_edges, dtype=np.uint32),
                np.ones(len(removed_edges), dtype=np.float32),
            )
        for k in removed_edges:
            prev_map.pop(k, None)
            prev_mult_map.pop(k, None)

    if not edge_scale_map:
        _prev_elastic_scales.pop(obj_name, None)
        _prev_elastic_multipliers.pop(obj_name, None)
        return

    indices = list(edge_scale_map.keys())
    target_scales = [edge_scale_map[idx] for idx in indices]
    target_mults = [edge_mult_map[idx] for idx in indices]

    # スムージング追従（急激な変化による布の爆発防止）
    current_scales = []
    has_scale_change = False
    has_mult_change = False

    for idx, target_s, target_m in zip(indices, target_scales, target_mults):
        prev_s = prev_map.get(idx, 1.0)
        # 1フレームあたり 20% ずつ目標値へ近づける
        curr_s = prev_s + (target_s - prev_s) * 0.20
        if abs(curr_s - target_s) < 1e-4:
            curr_s = target_s
        if abs(curr_s - prev_s) > 1e-5 or idx not in prev_map:
            has_scale_change = True
        current_scales.append(curr_s)
        prev_map[idx] = curr_s

        prev_m = prev_mult_map.get(idx)
        if prev_m is None or abs(prev_m - target_m) > 1e-5:
            has_mult_change = True
        prev_mult_map[idx] = target_m

    is_first_sync = not prev_map.get("_is_synced", False)
    _prev_elastic_scales[obj_name] = prev_map
    _prev_elastic_multipliers[obj_name] = prev_mult_map
    idx_arr = np.array(indices, dtype=np.uint32)
    if has_scale_change or is_first_sync:
        sim.set_edge_rest_length_scales(
            idx_arr,
            np.array(current_scales, dtype=np.float32),
        )
    if has_mult_api and (has_mult_change or is_first_sync):
        sim.set_edge_stiffness_multipliers(
            idx_arr,
            np.array(target_mults, dtype=np.float32),
        )
    prev_map["_is_synced"] = True


def sync_string_parts(sim, obj):
    """紐（Curve）パーツの引き絞りスライダーと弧長保存自然長トランスファーを同期する"""
    settings = getattr(obj, "taremin_cloth", None)
    if not settings or not hasattr(sim, "set_edge_rest_length_scales"):
        return

    from .cache import get_curve_parts_runtime
    curve_parts = get_curve_parts_runtime(obj.name)
    if not curve_parts:
        return

    string_parts = getattr(settings, "string_parts", None)
    if not string_parts:
        return

    edge_indices = []
    target_scales = []

    # 縫合優先モードがアクティブかどうかの確認
    is_sewing_priority = False
    if hasattr(sim, "is_sewing_priority_active"):
        try:
            is_sewing_priority = sim.is_sewing_priority_active()
        except Exception:
            is_sewing_priority = False

    enabled_parts = [sp for sp in string_parts if getattr(sp, "enabled", True) and getattr(sp, "curve_object", None)]
    for idx, sp in enumerate(enabled_parts):
        if idx >= len(curve_parts):
            break
        c_part = curve_parts[idx]

        pull = float(getattr(sp, "pull_length", 0.0) or 0.0)
        off_l = float(getattr(sp, "pull_offset_left", 0.0) or 0.0)
        off_r = float(getattr(sp, "pull_offset_right", 0.0) or 0.0)

        init_c_len = max(1e-4, c_part.initial_channel_len)
        init_l_len = max(1e-4, c_part.initial_left_len)
        init_r_len = max(1e-4, c_part.initial_right_len)

        # 1. 左右の引き込み量とトンネル区間の縮小（最大90%＝片側45%まで短縮可能、弧長を厳密に保存）
        if is_sewing_priority:
            pull_l = 0.0
            pull_r = 0.0
        else:
            max_half_pull = init_c_len * 0.45
            pull_l = float(np.clip(pull * 0.5 + off_l, 0.0, max_half_pull))
            pull_r = float(np.clip(pull * 0.5 + off_r, 0.0, max_half_pull))

        cinch_len = pull_l + pull_r
        channel_scale = max(0.1, (init_c_len - cinch_len) / init_c_len)

        # 2. 縮んだ長さを左右のテール（垂れ下がり）へ転送（総弧長 L_total = L_c + L_l + L_r を保存）
        target_l_len = max(1e-4, init_l_len + pull_l)
        target_r_len = max(1e-4, init_r_len + pull_r)

        left_scale = target_l_len / init_l_len
        right_scale = target_r_len / init_r_len

        for e_idx in c_part.channel_edge_indices:
            edge_indices.append(e_idx)
            target_scales.append(channel_scale)

        for e_idx in c_part.left_tail_edge_indices:
            edge_indices.append(e_idx)
            target_scales.append(left_scale)

        for e_idx in c_part.right_tail_edge_indices:
            edge_indices.append(e_idx)
            target_scales.append(right_scale)

        # 3. 縫合ギャップ跨ぎエッジが通常エッジとして残っている場合の互換処理
        for e_idx in c_part.seam_bridge_edge_indices:
            edge_indices.append(e_idx)
            target_scales.append(0.001)

    if not edge_indices:
        return

    obj_name = obj.name
    prev_map = _prev_string_scales.get(obj_name, {})
    current_scales = []
    has_change = False

    for idx, target_s in zip(edge_indices, target_scales):
        prev_s = prev_map.get(idx, 1.0)
        # スライダー操作に滑らかに追従
        curr_s = prev_s + (target_s - prev_s) * 0.25
        if abs(curr_s - target_s) < 1e-4:
            curr_s = target_s
        if abs(curr_s - prev_s) > 1e-5 or idx not in prev_map:
            has_change = True
        current_scales.append(curr_s)
        prev_map[idx] = curr_s

    is_first_sync = not prev_map.get("_is_synced", False)
    _prev_string_scales[obj_name] = prev_map
    if has_change or is_first_sync:
        sim.set_edge_rest_length_scales(
            np.array(edge_indices, dtype=np.uint32),
            np.array(current_scales, dtype=np.float32),
        )
        prev_map["_is_synced"] = True


@allow_positional
def _extract_vertex_pin_weight(v, vg, vg_idx) -> float:
    """MeshVertex から指定頂点グループのウェイトを取得する (v.groups / vg.weight 両対応)"""
    if vg_idx is not None and hasattr(v, "groups") and v.groups is not None:
        for g in v.groups:
            if getattr(g, "group", None) == vg_idx and g.weight > 0.0:
                return float(g.weight)
        return 0.0
    if hasattr(vg, "weight"):
        try:
            w = float(vg.weight(v.index))
            return w if w > 0.0 else 0.0
        except (RuntimeError, TypeError, ValueError):
            return 0.0
    return 0.0


def compute_attachment_pin_targets(obj):
    """アタッチメントピンのワールド目標座標配列を算出する共通関数 (In-process / GUI IPC 共通正本)。

    Returns:
        tuple[list[int], list[list[float]], list[float], tuple] | None:
            (vertex_indices, world_positions, weights, target_matrix_signature)
            設定無効または対象頂点なしの場合は None。
    """
    settings = getattr(obj, "taremin_cloth", None)
    if not settings or not getattr(settings, "pin_target_object", None):
        return None

    target_obj = settings.pin_target_object
    vg_name = getattr(settings, "pin_vertex_group", "") or "Pin"
    vg = obj.vertex_groups.get(vg_name)
    if not vg:
        return None

    target_mat = target_obj.matrix_world
    pin_bone = str(getattr(settings, "pin_target_bone", "") or "")
    if pin_bone and getattr(target_obj, "type", "") == "ARMATURE" and getattr(target_obj, "pose", None):
        bone = target_obj.pose.bones.get(pin_bone)
        if bone:
            target_mat = target_mat @ bone.matrix

    try:
        mat_np = np.array([list(row) for row in target_mat], dtype=np.float32)
    except (TypeError, ValueError):
        mat_np = np.asarray(target_mat, dtype=np.float32)
    target_name = str(getattr(target_obj, "name", "") or "")
    mat_sig = (target_name, pin_bone, vg_name, tuple(np.round(mat_np.flatten(), 5).tolist()))

    mesh = obj.data
    cache_key = (obj.name, vg_name, len(mesh.vertices), target_name, pin_bone)
    pinned_entries = _attachment_pin_indices_cache.get(cache_key)
    if pinned_entries is None:
        pinned_entries = []
        vg_idx = getattr(vg, "index", None)
        obj_mat = getattr(obj, "matrix_world", None)
        try:
            inv_init_target = target_mat.inverted()
        except Exception:
            inv_init_target = None
        for v in mesh.vertices:
            w = _extract_vertex_pin_weight(v, vg, vg_idx)
            if w > 0.0:
                v_world = (obj_mat @ v.co) if obj_mat is not None else v.co
                rel_co = (inv_init_target @ v_world) if inv_init_target is not None else v_world
                pinned_entries.append((int(v.index), float(w), rel_co))
        _attachment_pin_indices_cache[cache_key] = pinned_entries

    if not pinned_entries:
        return None

    v_indices = []
    positions = []
    weights = []
    if len(pinned_entries) == 1:
        # 単一頂点ピンはターゲットのワールド位置へ直接追従
        t_world = target_mat.translation
        v_idx, weight, _ = pinned_entries[0]
        v_indices.append(v_idx)
        positions.append([float(t_world.x), float(t_world.y), float(t_world.z)])
        weights.append(weight)
    else:
        # 複数頂点ピンはターゲット初期姿勢に対する各頂点の相対オフセットを維持してワールド座標を算出
        for v_idx, weight, rel_co in pinned_entries:
            p_world = target_mat @ rel_co
            v_indices.append(v_idx)
            positions.append([float(p_world.x), float(p_world.y), float(p_world.z)])
            weights.append(weight)

    return v_indices, positions, weights, mat_sig


def sync_attachment_pins(sim, obj, *, scene=None):
    """In-process シミュレータへアタッチメントピン座標を同期する（差分スキップ・解除対応）"""
    res = compute_attachment_pin_targets(obj)
    if res is None:
        prev_indices = _prev_attachment_pin_indices.pop(obj.name, None)
        if prev_indices and hasattr(sim, "release_pin"):
            for v_idx in prev_indices:
                sim.release_pin(v_idx)
        _prev_attachment_pin_targets.pop(obj.name, None)
        return

    v_indices, positions, weights, mat_sig = res
    new_set = set(v_indices)
    prev_set = _prev_attachment_pin_indices.get(obj.name, set())
    removed = prev_set - new_set
    if removed and hasattr(sim, "release_pin"):
        for v_idx in removed:
            sim.release_pin(v_idx)

    if _prev_attachment_pin_targets.get(obj.name) == mat_sig and not removed and prev_set == new_set:
        return
    _prev_attachment_pin_targets[obj.name] = mat_sig
    _prev_attachment_pin_indices[obj.name] = new_set

    for v_idx, target_xyz, weight in zip(v_indices, positions, weights):
        sim.set_pin(v_idx, target_xyz, weight)

