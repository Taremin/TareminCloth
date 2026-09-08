"""
コライダーアニメーション駆動モジュール (anim_driver)
シェイプキー、2ポーズスナップショット補間、アクションキーフレーム評価、
およびサイクル再生制御（once, repeat, pingpong, イージング）を提供します。
"""

import json
import math
import bpy
import mathutils
from .logger import logger



def apply_easing(t: float, easing_type: str = 'SMOOTH') -> float:
    """進行度 t (0.0 - 1.0) に対するイージング補間を計算する"""
    clamped = max(0.0, min(1.0, t))
    if easing_type == 'SMOOTH':
        # Smoothstep (3t^2 - 2t^3): 始点と終点で速度ゼロ
        return clamped * clamped * (3.0 - 2.0 * clamped)
    # デフォルト: LINEAR
    return clamped


def compute_cycle_progress(
    frame_index: int,
    cycle_frames: int,
    play_mode: str = 'ONCE',
    loop_count: int = 1,
    infinite_loop: bool = False,
) -> tuple[float, bool]:
    """
    フレームインデックスから現在の正規化進行度 (0.0 - 1.0) と完了フラグを計算する。

    Returns:
        (raw_progress, is_finished)
    """
    if cycle_frames <= 0:
        return 1.0, True

    if play_mode == 'ONCE':
        if frame_index >= cycle_frames:
            return 1.0, True
        return max(0.0, min(1.0, frame_index / cycle_frames)), False

    elif play_mode == 'REPEAT':
        cycle_idx = frame_index // cycle_frames
        if not infinite_loop and cycle_idx >= loop_count:
            return 1.0, True
        cycle_sub = frame_index % cycle_frames
        return cycle_sub / cycle_frames, False

    elif play_mode == 'PINGPONG':
        # 片道 (0 -> 1 または 1 -> 0) を1 cycle_frames と定義
        half_cycle_idx = frame_index // cycle_frames
        round_trip_idx = half_cycle_idx // 2
        if not infinite_loop and round_trip_idx >= loop_count:
            return 0.0, True

        sub_t = (frame_index % cycle_frames) / cycle_frames
        if half_cycle_idx % 2 == 0:
            # 順再生 (0.0 -> 1.0)
            return sub_t, False
        else:
            # 逆再生 (1.0 -> 0.0)
            return 1.0 - sub_t, False

    return 1.0, True


def capture_pose_snapshot(armature_obj: bpy.types.Object) -> str:
    """
    指定されたアーマチュアオブジェクトの現在のポーズ状態をJSON文字列としてキャプチャする。
    """
    if not armature_obj or armature_obj.type != 'ARMATURE' or not armature_obj.pose:
        return ""

    data = {}
    for pbone in armature_obj.pose.bones:
        b_name = pbone.name
        rot_mode = pbone.rotation_mode
        b_data = {
            "rot_mode": rot_mode,
            "loc": [pbone.location.x, pbone.location.y, pbone.location.z],
            "scale": [pbone.scale.x, pbone.scale.y, pbone.scale.z],
        }
        if rot_mode == 'QUATERNION':
            q = pbone.rotation_quaternion
            b_data["rot_quat"] = [q.w, q.x, q.y, q.z]
        elif rot_mode == 'AXIS_ANGLE':
            aa = pbone.rotation_axis_angle
            b_data["rot_axis_angle"] = [aa[0], aa[1], aa[2], aa[3]]
        else:
            e = pbone.rotation_euler
            b_data["rot_euler"] = [e.x, e.y, e.z]

        data[b_name] = b_data

    return json.dumps(data)


def apply_pose_blend(
    armature_obj: bpy.types.Object,
    snapshot_a_json: str,
    snapshot_b_json: str,
    t: float,
) -> bool:
    """
    2つのポーズスナップショット（A: 0.0, B: 1.0）間を進行度 t (0.0 - 1.0) でブレンドする。
    """
    if not armature_obj or armature_obj.type != 'ARMATURE' or not armature_obj.pose:
        return False
    if not snapshot_a_json or not snapshot_b_json:
        return False

    try:
        data_a = json.loads(snapshot_a_json)
        data_b = json.loads(snapshot_b_json)
    except Exception as ex:
        logger.warning(f"Failed to parse pose snapshots: {ex}")
        return False

    clamped_t = max(0.0, min(1.0, t))

    for pbone in armature_obj.pose.bones:
        b_name = pbone.name
        if b_name not in data_a or b_name not in data_b:
            continue

        item_a = data_a[b_name]
        item_b = data_b[b_name]

        # Location Lerp
        loc_a = mathutils.Vector(item_a["loc"])
        loc_b = mathutils.Vector(item_b["loc"])
        pbone.location = loc_a.lerp(loc_b, clamped_t)

        # Scale Lerp
        scale_a = mathutils.Vector(item_a["scale"])
        scale_b = mathutils.Vector(item_b["scale"])
        pbone.scale = scale_a.lerp(scale_b, clamped_t)

        # Rotation Slerp / Lerp
        rot_mode = pbone.rotation_mode
        if rot_mode == 'QUATERNION':
            q_a_raw = item_a.get("rot_quat", [1.0, 0.0, 0.0, 0.0])
            q_b_raw = item_b.get("rot_quat", [1.0, 0.0, 0.0, 0.0])
            qa = mathutils.Quaternion(q_a_raw)
            qb = mathutils.Quaternion(q_b_raw)
            if qa.dot(qb) < 0.0:
                qb.negate()
            pbone.rotation_quaternion = qa.slerp(qb, clamped_t)

        elif rot_mode == 'AXIS_ANGLE':
            aa_a_raw = item_a.get("rot_axis_angle", [0.0, 0.0, 1.0, 0.0])
            aa_b_raw = item_b.get("rot_axis_angle", [0.0, 0.0, 1.0, 0.0])
            qa = mathutils.Quaternion(mathutils.Vector((aa_a_raw[1], aa_a_raw[2], aa_a_raw[3])), aa_a_raw[0])
            qb = mathutils.Quaternion(mathutils.Vector((aa_b_raw[1], aa_b_raw[2], aa_b_raw[3])), aa_b_raw[0])
            if qa.dot(qb) < 0.0:
                qb.negate()
            q_blended = qa.slerp(qb, clamped_t)
            axis, angle = q_blended.to_axis_angle()
            pbone.rotation_axis_angle = [angle, axis.x, axis.y, axis.z]

        else:
            # EULER
            e_a_raw = item_a.get("rot_euler", [0.0, 0.0, 0.0])
            e_b_raw = item_b.get("rot_euler", [0.0, 0.0, 0.0])
            ea = mathutils.Euler(e_a_raw, rot_mode)
            eb = mathutils.Euler(e_b_raw, rot_mode)
            # クォータニオン経由でSlerpしてジンバルロックを防止
            qa = ea.to_quaternion()
            qb = eb.to_quaternion()
            if qa.dot(qb) < 0.0:
                qb.negate()
            pbone.rotation_euler = qa.slerp(qb, clamped_t).to_euler(rot_mode)

    return True


def get_action_fcurves(action: bpy.types.Action):
    """
    Blender 3.x/4.x/5.x (Legacy Action & Layered/Slotted Action) の両方に対応して
    Action内の全 FCurve を取得する。
    """
    if not action:
        return []
    # Legacy Action (Blender 4.3 以前、またはレガシーAction)
    if hasattr(action, "fcurves") and action.fcurves is not None:
        return list(action.fcurves)
    # Layered / Slotted Action (Blender 4.4 / 5.x+)
    fcurves = []
    if hasattr(action, "layers"):
        for layer in action.layers:
            for strip in layer.strips:
                if hasattr(strip, "channelbags"):
                    for bag in strip.channelbags:
                        if hasattr(bag, "fcurves"):
                            fcurves.extend(bag.fcurves)
    return fcurves


def apply_action_frame(
    armature_obj: bpy.types.Object,
    action: bpy.types.Action,
    eval_frame: float,
) -> bool:
    """
    指定されたアクションの FCurve を評価し、タイムライン全体を進めずに
    対象アーマチュアのポーズにのみキーフレーム値を直接適用する。
    """
    if not armature_obj or armature_obj.type != 'ARMATURE' or not armature_obj.pose:
        return False
    if not action:
        return False

    fcurves = get_action_fcurves(action)
    if not fcurves:
        return False

    for fcurve in fcurves:
        data_path = fcurve.data_path
        array_index = fcurve.array_index
        val = fcurve.evaluate(eval_frame)

        try:
            target = armature_obj.path_resolve(data_path)
            if hasattr(target, "__getitem__"):
                target[array_index] = val
            else:
                # 配列型でないプロパティ
                parts = data_path.rsplit(".", 1)
                if len(parts) == 2:
                    parent_obj = armature_obj.path_resolve(parts[0])
                    setattr(parent_obj, parts[1], val)
                else:
                    setattr(armature_obj, data_path, val)
        except Exception:
            # 対象ボーンが存在しない等の場合はスキップ
            continue

    return True


def step_collider_animation(
    col_obj: bpy.types.Object,
    frame_index: int,
) -> tuple[float, bool]:
    """
    コライダーオブジェクトのアニメーション設定に基づいて1ステップ分のアニメーションを適用する。

    Returns:
        (applied_progress, is_deformed)
    """
    col_settings = getattr(col_obj, "taremin_cloth_collider", None)
    if not col_settings or not col_settings.is_collider:
        return 0.0, False

    anim_settings = getattr(col_settings, "anim", None)
    if not anim_settings or not anim_settings.enabled:
        return 0.0, False

    raw_t, is_finished = compute_cycle_progress(
        frame_index=frame_index,
        cycle_frames=anim_settings.cycle_frames,
        play_mode=anim_settings.play_mode,
        loop_count=anim_settings.loop_count,
        infinite_loop=anim_settings.infinite_loop,
    )
    eased_t = apply_easing(raw_t, anim_settings.easing)
    anim_settings.progress = eased_t

    target_type = anim_settings.target_type

    if target_type == 'SHAPE_KEY':
        if col_obj.type == 'MESH' and col_obj.data and col_obj.data.shape_keys:
            key_block = col_obj.data.shape_keys.key_blocks.get(anim_settings.shape_key_name)
            if key_block:
                val = anim_settings.start_value + eased_t * (anim_settings.end_value - anim_settings.start_value)
                key_block.value = val
                return eased_t, True

    elif target_type == 'POSE_BLEND':
        armature = anim_settings.armature_obj
        if not armature:
            # アーマチュアが指定されていない場合、親やArmatureモディファイアを自動探索
            if col_obj.type == 'ARMATURE':
                armature = col_obj
            elif col_obj.parent and col_obj.parent.type == 'ARMATURE':
                armature = col_obj.parent
            else:
                for mod in col_obj.modifiers:
                    if mod.type == 'ARMATURE' and mod.object:
                        armature = mod.object
                        break

        if armature and anim_settings.start_pose_data and anim_settings.target_pose_data:
            success = apply_pose_blend(
                armature,
                anim_settings.start_pose_data,
                anim_settings.target_pose_data,
                eased_t,
            )
            return eased_t, success

    elif target_type == 'ACTION':
        armature = anim_settings.armature_obj
        if not armature:
            if col_obj.type == 'ARMATURE':
                armature = col_obj
            elif col_obj.parent and col_obj.parent.type == 'ARMATURE':
                armature = col_obj.parent
            else:
                for mod in col_obj.modifiers:
                    if mod.type == 'ARMATURE' and mod.object:
                        armature = mod.object
                        break

        if armature and anim_settings.action:
            f_start = anim_settings.frame_start
            f_end = anim_settings.frame_end
            eval_frame = f_start + eased_t * (f_end - f_start)
            success = apply_action_frame(armature, anim_settings.action, eval_frame)
            return eased_t, success

    return eased_t, False
