"""
taremin_cloth ポーズ管理オペレーター
アーマチュアのポーズスナップショット記録（開始/目標姿勢）およびポーズプレビュー適用オペレーターを提供する。
"""

import bpy
from ..utils import anim_driver


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

        col_settings = getattr(obj, "taremin_cloth_collider", None)
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

        col_settings = getattr(obj, "taremin_cloth_collider", None)
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
