"""
taremin_cloth 基本ライフサイクル・設定オペレーター
シミュレーションの有効化/無効化、リセット、レスト形状適用、キャッシュクリア、
GPU設定適用、オブジェクト選択オペレーターを提供する。
"""

import bpy
from ..engine.cache import (
    cache_rest_positions,
    restore_rest_positions,
    clear_simulator_for_object,
    clear_simulators,
)
from ..utils import topology
from ..utils.logger import logger


class TAREMIN_CLOTH_OT_toggle_cloth(bpy.types.Operator):
    """選択オブジェクトのClothシミュレーション有効/無効を切り替える"""
    bl_idname = "taremin_cloth.toggle_cloth"
    bl_label = "Toggle Cloth"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        settings = obj.taremin_cloth
        settings.is_cloth = not settings.is_cloth
        if not settings.is_cloth:
            restore_rest_positions(obj, clear=True)
        clear_simulators()
        self.report({'INFO'}, f"Cloth enabled: {settings.is_cloth}")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_reset_selected(bpy.types.Operator):
    """選択中のClothオブジェクトのみを初期レスト位置にリセットする"""
    bl_idname = "taremin_cloth.reset_selected"
    bl_label = "Reset Selected Cloth"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bool(obj and obj.type == 'MESH' and getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth)

    def execute(self, context):
        obj = context.active_object
        logger.info(f"[Reset] TAREMIN_CLOTH_OT_reset_selected executed for '{obj.name}'")
        restore_rest_positions(obj)
        clear_simulator_for_object(obj.name)
        obj.update_tag()
        if context.screen:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        self.report({'INFO'}, f"Reset Cloth: {obj.name}")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_reset_all(bpy.types.Operator):
    """シーン内のすべてのClothシミュレーションを初期状態にリセットする"""
    bl_idname = "taremin_cloth.reset_all"
    bl_label = "Reset All Simulation"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        if not scene:
            return False
        return any(getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth for obj in scene.objects)

    def execute(self, context):
        scene = context.scene
        if scene:
            for obj in scene.objects:
                if getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth:
                    restore_rest_positions(obj)
        clear_simulators()
        if context.screen:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        self.report({'INFO'}, "All Cloth Simulations Reset")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_reset_simulation(bpy.types.Operator):
    """シミュレーションを初期状態にリセットする（全体リセット互換）"""
    bl_idname = "taremin_cloth.reset_simulation"
    bl_label = "Reset Simulation"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        return bpy.ops.taremin_cloth.reset_all()


class TAREMIN_CLOTH_OT_apply_rest_shape(bpy.types.Operator):
    """現在の変形メッシュ形状をシミュレーションの新しい初期レスト形状（自然長）として確定する"""
    bl_idname = "taremin_cloth.apply_rest_shape"
    bl_label = "Apply Rest Shape"
    bl_description = "現在の変形メッシュ形状をシミュレーションの新しい初期状態（レストポーズ）として確定します"
    bl_options = {'REGISTER', 'UNDO'}

    all_objects: bpy.props.BoolProperty(
        name="All Objects",
        description="シーン内のすべてのClothオブジェクトに適用する",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        if not context.scene:
            return False
        obj = context.active_object
        has_sel = bool(obj and obj.type == 'MESH' and getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth)
        has_any = any(getattr(o, "taremin_cloth", None) and o.taremin_cloth.is_cloth for o in context.scene.objects)
        return has_sel or has_any

    def execute(self, context):
        if self.all_objects:
            targets = [o for o in context.scene.objects if getattr(o, "taremin_cloth", None) and o.taremin_cloth.is_cloth and o.type == 'MESH']
        else:
            obj = context.active_object
            targets = [obj] if (obj and getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth and obj.type == 'MESH') else []

        if not targets:
            self.report({'WARNING'}, "対象となるClothオブジェクトが見つかりません")
            return {'CANCELLED'}

        for o in targets:
            clear_simulator_for_object(o.name)
            topology.clear_pre_subdivision_backup(o)
            for prop in ("_taremin_rest_positions", "_taremin_rest_signature", "_taremin_is_deformed", "_taremin_cross_subdiv_map"):
                if prop in o:
                    try:
                        del o[prop]
                    except Exception:
                        pass
            cache_rest_positions(o, force=True)
            o.update_tag()
            logger.info(f"[Cache] Applied current shape as rest positions for '{o.name}' (verts={len(o.data.vertices)})")

        if self.all_objects:
            self.report({'INFO'}, f"全 {len(targets)} 個のClothの現在形状を新しいレスト形状として確定しました")
        else:
            self.report({'INFO'}, f"'{targets[0].name}' の現在形状を新しいレスト形状として確定しました")

        return {'FINISHED'}


class TAREMIN_CLOTH_OT_clear_cache(bpy.types.Operator):
    """【互換用】古いキャッシュやバックアップを破棄し、現在のメッシュ形状を初期状態として再記憶します"""
    bl_idname = "taremin_cloth.clear_cache"
    bl_label = "Clear Cache"
    bl_description = "古いキャッシュやバックアップを破棄し、現在のメッシュ形状を初期状態として再記憶します（Apply Rest Shapeと同等）"
    bl_options = {'REGISTER', 'UNDO'}

    all_objects: bpy.props.BoolProperty(
        name="All Objects",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        return TAREMIN_CLOTH_OT_apply_rest_shape.poll(context)

    def execute(self, context):
        return bpy.ops.taremin_cloth.apply_rest_shape(all_objects=self.all_objects)


class TAREMIN_CLOTH_OT_apply_gpu_settings(bpy.types.Operator):
    """GPUバックエンドおよびデバイス設定を適用し、GPUコンテキストを再初期化する"""
    bl_idname = "taremin_cloth.apply_gpu_settings"
    bl_label = "Apply GPU Settings"
    bl_description = "選択したバックエンドおよびGPUデバイスを適用し、GPUコンテキストを再初期化します（シミュレーションはリセットされます）"
    bl_options = {'REGISTER'}

    def execute(self, context):
        from ..preferences import get_preferences, apply_gpu_settings_from_prefs
        from .interactive import stop_interactive_if_running

        prefs = get_preferences(context)
        if not prefs:
            self.report({'ERROR'}, "アドオン設定の取得に失敗しました")
            return {'CANCELLED'}

        # 実行中のインタラクティブモードを停止
        stop_interactive_if_running()

        # 既存シミュレータキャッシュを破棄
        clear_simulators()

        try:
            apply_gpu_settings_from_prefs(prefs)
            msg = f"GPU設定を適用しました: {prefs.active_device_name} ({prefs.active_backend_name})"
            logger.info(f"[GPU] {msg}")
            self.report({'INFO'}, msg)
        except Exception as e:
            err_msg = f"GPUコンテキスト初期化エラー: {e}"
            logger.error(f"[GPU] {err_msg}")
            self.report({'ERROR'}, err_msg)
            return {'CANCELLED'}

        if context.screen:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        return {'FINISHED'}


class TAREMIN_CLOTH_OT_select_object(bpy.types.Operator):
    """指定した布またはコライダーオブジェクトを選択してアクティブにする"""
    bl_idname = "taremin_cloth.select_object"
    bl_label = "Select Object"
    bl_description = "指定した布またはコライダーオブジェクトを選択してアクティブにします"
    bl_options = {'UNDO'}

    object_name: bpy.props.StringProperty(
        name="Object Name",
        description="選択するオブジェクトの名前",
        default="",
    )

    def execute(self, context):
        if not self.object_name or self.object_name not in context.scene.objects:
            self.report({'WARNING'}, f"オブジェクト '{self.object_name}' が見つかりません")
            return {'CANCELLED'}

        target_obj = context.scene.objects[self.object_name]

        # 編集モード等の場合はオブジェクトモードに安全に切り替える
        if context.mode != 'OBJECT' and bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='OBJECT')

        # 既存の選択をすべて解除
        for obj in context.selected_objects:
            obj.select_set(False)

        # ターゲットオブジェクトが非表示の場合は表示状態にする
        if target_obj.hide_get():
            target_obj.hide_set(False)

        # ターゲットオブジェクトを選択してアクティブ化
        target_obj.select_set(True)
        context.view_layer.objects.active = target_obj

        return {'FINISHED'}


class TAREMIN_CLOTH_OT_clear_bone_sdf_cache(bpy.types.Operator):
    """保存されているすべてのボーンSDFキャッシュを消去します"""
    bl_idname = "taremin_cloth.clear_bone_sdf_cache"
    bl_label = "Clear Bone SDF Cache"
    bl_description = "ディスクに保存されているすべてのボーンSDFキャッシュファイルを削除します"
    bl_options = {'REGISTER'}

    def execute(self, context):
        from ..engine.sdf_baker import clear_all_cached_sdf
        count = clear_all_cached_sdf()
        self.report({'INFO'}, f"ボーンSDFキャッシュを削除しました ({count}件)")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_rebake_bone_sdf(bpy.types.Operator):
    """アクティブな素体オブジェクトのボーンSDFを強制的に再ベイクします"""
    bl_idname = "taremin_cloth.rebake_bone_sdf"
    bl_label = "Rebake Bone SDF"
    bl_description = "キャッシュを使用せず、現在のメッシュとボーン構造からボーン局所SDFを再計算します"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and getattr(obj, "taremin_collider", None) and obj.taremin_collider.is_collider and obj.taremin_collider.collider_type == 'BONE_SDF'

    def execute(self, context):
        obj = context.active_object
        col_settings = obj.taremin_collider
        from ..engine.sdf_baker import get_or_bake_bone_sdf_for_object
        result = get_or_bake_bone_sdf_for_object(obj, col_settings, force_rebake=True)
        if result is not None:
            self.report({'INFO'}, f"'{obj.name}' のボーンSDFを再ベイクしました: {len(result.bone_names)} ボーン ({result.width}x{result.height}x{result.depth})")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, f"'{obj.name}' のボーンSDFベイクに失敗しました。Armatureモディファイアとウェイトを確認してください")
            return {'CANCELLED'}
