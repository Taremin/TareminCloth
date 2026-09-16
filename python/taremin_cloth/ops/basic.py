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
from .. import i18n


class TAREMIN_CLOTH_OT_toggle_cloth(bpy.types.Operator):
    """選択オブジェクトのClothシミュレーション有効/無効を切り替える"""
    bl_idname = "taremin_cloth.toggle_cloth"
    bl_label = "Toggle Cloth"
    bl_translation_context = i18n.CONTEXT
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
        else:
            cache_rest_positions(obj, force=True)
        clear_simulators()
        self.report({'INFO'}, f"Cloth enabled: {settings.is_cloth}")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_reset_selected(bpy.types.Operator):
    """選択中のClothオブジェクトのみを初期レスト位置にリセットする"""
    bl_idname = "taremin_cloth.reset_selected"
    bl_label = "Reset Selected Cloth"
    bl_translation_context = i18n.CONTEXT
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
    bl_translation_context = i18n.CONTEXT
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
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        return bpy.ops.taremin_cloth.reset_all()


class TAREMIN_CLOTH_OT_apply_rest_shape(bpy.types.Operator):
    """Bake current deformed mesh shape as the new initial state (rest pose)"""
    bl_idname = "taremin_cloth.apply_rest_shape"
    bl_label = "Apply Rest Shape"
    bl_description = "Bake the current deformed mesh shape as the new initial state (rest pose) for simulation"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    all_objects: bpy.props.BoolProperty(
        name="All Objects",
        description="Apply to all cloth objects in scene",
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
            for prop in ("_taremin_cloth_rest_positions", "_taremin_cloth_rest_signature", "_taremin_cloth_is_deformed", "_taremin_cloth_cross_subdiv_map"):
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
    """[Compatibility] Discard old cache and backups, and re-capture current mesh shape as initial state"""
    bl_idname = "taremin_cloth.clear_cache"
    bl_label = "Clear Cache"
    bl_description = "Discard old cache and backups, and re-capture the current mesh shape as initial state (same as Apply Rest Shape)"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    all_objects: bpy.props.BoolProperty(
        name="All Objects",
        description="Apply to all cloth objects in scene",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        return TAREMIN_CLOTH_OT_apply_rest_shape.poll(context)

    def execute(self, context):
        return bpy.ops.taremin_cloth.apply_rest_shape(all_objects=self.all_objects)


class TAREMIN_CLOTH_OT_apply_gpu_settings(bpy.types.Operator):
    """Apply GPU backend and device settings, and reinitialize GPU context"""
    bl_idname = "taremin_cloth.apply_gpu_settings"
    bl_label = "Apply GPU Settings"
    bl_description = "Apply selected backend and GPU device, and reinitialize GPU context (resets simulation)"
    bl_translation_context = i18n.CONTEXT
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
    """Select and activate specified cloth or collider object"""
    bl_idname = "taremin_cloth.select_object"
    bl_label = "Select Object"
    bl_description = "Select and activate the specified cloth or collider object"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'UNDO'}

    object_name: bpy.props.StringProperty(
        name="Object Name",
        description="Name of the object to select",
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
    """Clear all stored bone SDF cache files on disk"""
    bl_idname = "taremin_cloth.clear_bone_sdf_cache"
    bl_label = "Clear Bone SDF Cache"
    bl_description = "Delete all bone SDF cache files stored on disk"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER'}

    def execute(self, context):
        from ..engine.sdf_baker import clear_all_cached_sdf
        count = clear_all_cached_sdf()
        self.report({'INFO'}, f"ボーンSDFキャッシュを削除しました ({count}件)")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_rebake_bone_sdf(bpy.types.Operator):
    """Force rebake bone SDF for active body object"""
    bl_idname = "taremin_cloth.rebake_bone_sdf"
    bl_label = "Rebake Bone SDF"
    bl_description = "Bypass cache and recompute bone-local SDF from the current mesh and bone hierarchy"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj
            and getattr(obj, "taremin_cloth_collider", None)
            and obj.taremin_cloth_collider.is_collider
            and obj.taremin_cloth_collider.collider_type in {'BONE_SDF', 'MESH_SDF'}
        )

    def execute(self, context):
        obj = context.active_object
        col_settings = obj.taremin_cloth_collider
        if col_settings.collider_type == 'MESH_SDF':
            from ..engine.sdf_baker import get_or_bake_mesh_sdf_for_object
            result = get_or_bake_mesh_sdf_for_object(obj, col_settings, force_rebake=True)
            if result is not None:
                self.report({'INFO'}, f"'{obj.name}' のメッシュSDFを再ベイクしました: ({result.width}x{result.height}x{result.depth})")
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, f"'{obj.name}' のメッシュSDFベイクに失敗しました")
                return {'CANCELLED'}
        else:
            from ..engine.sdf_baker import get_or_bake_bone_sdf_for_object
            result = get_or_bake_bone_sdf_for_object(obj, col_settings, force_rebake=True)
            if result is not None:
                self.report({'INFO'}, f"'{obj.name}' のボーンSDFを再ベイクしました: {len(result.bone_names)} ボーン ({result.width}x{result.height}x{result.depth})")
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, f"'{obj.name}' のボーンSDFベイクに失敗しました。Armatureモディファイアとウェイトを確認してください")
                return {'CANCELLED'}


class TAREMIN_CLOTH_OT_auto_detect_collider(bpy.types.Operator):
    """Automatically detect and set optimal collider shape based on object structure"""
    bl_idname = "taremin_cloth.auto_detect_collider"
    bl_label = "Auto Detect Collider Type"
    bl_description = "Analyze object structure (modifiers, vertex weights, poly count, etc.) to automatically configure the optimal collider type"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and getattr(obj, "taremin_cloth_collider", None) and obj.taremin_cloth_collider.is_collider

    def execute(self, context):
        from ..utils.collider_detect import detect_collider_type
        obj = context.active_object
        col_settings = obj.taremin_cloth_collider
        detected = detect_collider_type(obj)
        col_settings.collider_type = detected
        col_settings.collider_purpose = 'AUTO'
        type_names = {
            'BONE_SDF': "Bone SDF (素体・キャラクタ)",
            'MESH_SDF': "Mesh SDF (マネキン・剛体)",
            'PLANE': "Plane (床・地面)",
            'SPHERE': "Sphere (球体)",
            'MESH': "Mesh (単純メッシュ)",
        }
        self.report({'INFO'}, f"コライダー形状を自動判定しました: {type_names.get(detected, detected)}")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_auto_fit_self_collision(bpy.types.Operator):
    """Automatically configure optimal self-collision parameters based on edge length and purpose"""
    bl_idname = "taremin_cloth.auto_fit_self_collision"
    bl_label = "Auto Fit Self Collision"
    bl_description = "Automatically calculate and apply self-collision parameters based on edge length and preset (disabled for Custom)"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj
            and obj.type == 'MESH'
            and getattr(obj, "taremin_cloth", None)
            and obj.taremin_cloth.is_cloth
            and getattr(obj.taremin_cloth, "self_collision_purpose", 'STANDARD') != 'CUSTOM'
        )

    def execute(self, context):
        from ..utils.self_collision_fit import fit_self_collision_for_object
        obj = context.active_object
        settings = obj.taremin_cloth
        params = fit_self_collision_for_object(obj, settings)
        if params:
            thick_mm = params['thickness'] * 1000.0
            purpose_items = dict(settings.rna_type.properties['self_collision_purpose'].enum_items)
            purpose_label = purpose_items[settings.self_collision_purpose].name if settings.self_collision_purpose in purpose_items else settings.self_collision_purpose
            self.report({'INFO'}, f"自己衝突パラメータを最適化しました: 厚み {thick_mm:.1f} mm ({purpose_label})")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "メッシュにエッジが存在しないため、自己衝突パラメータを自動設定できませんでした")
            return {'CANCELLED'}


class TAREMIN_CLOTH_OT_save_as_shape_key(bpy.types.Operator):
    """Save current simulation deformed shape as a shape key in target shapes object"""
    bl_idname = "taremin_cloth.save_as_shape_key"
    bl_label = "Save as Shape Key"
    bl_description = "Save current simulation deformed shape as a shape key in target shapes object"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    shape_key_name: bpy.props.StringProperty(
        name="Shape Key Name",
        description="Name of the new shape key (default: Cloth_Shape)",
        default="",
    )
    target_mode: bpy.props.EnumProperty(
        name="Target Mode",
        description="Target object destination mode",
        items=[
            ('AUTO_TARGET', "Auto Target (_Shapes)", "Add to <Object>_Shapes object or create if not exists"),
            ('NEW_OBJECT', "New Object", "Always create a new snapshot object"),
        ],
        default='AUTO_TARGET',
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bool(
            obj
            and obj.type == 'MESH'
            and getattr(obj, "taremin_cloth", None)
            and obj.taremin_cloth.is_cloth
        )

    def execute(self, context):
        cloth_obj = context.active_object
        if not cloth_obj or cloth_obj.type != 'MESH':
            self.report({'WARNING'}, i18n.trans("Please select a mesh object"))
            return {'CANCELLED'}

        # 1. 十字分割チェック（トポロジー保護）
        if topology.is_cross_subdivided(cloth_obj):
            self.report(
                {'WARNING'},
                i18n.trans("Cannot save shape key while mesh is cross-subdivided. Please restore quad topology first.")
            )
            return {'CANCELLED'}

        mesh = cloth_obj.data
        n_verts = len(mesh.vertices)
        if n_verts == 0:
            self.report({'WARNING'}, i18n.trans("Mesh has no vertices"))
            return {'CANCELLED'}

        # 2. 現在の変形座標を取得
        import numpy as np
        curr_coords = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", curr_coords)

        # 3. レストポーズ初期座標を取得（変形前形状）
        if "_taremin_cloth_rest_positions" in cloth_obj:
            initial_coords = np.frombuffer(cloth_obj["_taremin_cloth_rest_positions"], dtype=np.float32)
            if len(initial_coords) != n_verts * 3:
                initial_coords = curr_coords.copy()
        else:
            initial_coords = curr_coords.copy()

        # 4. 出力先オブジェクト（受け皿）の解決
        target_name = f"{cloth_obj.name}_Shapes"
        target_obj = None

        if self.target_mode == 'AUTO_TARGET' and context.scene:
            candidate = context.scene.objects.get(target_name)
            if candidate and candidate.type == 'MESH' and len(candidate.data.vertices) == n_verts:
                target_obj = candidate

        is_new_target = (target_obj is None)
        if is_new_target:
            # ターゲットオブジェクトを新規作成
            new_mesh = mesh.copy()
            new_name = target_name if self.target_mode == 'AUTO_TARGET' else f"{cloth_obj.name}_Snapshot"
            new_mesh.name = new_name
            target_obj = bpy.data.objects.new(new_name, new_mesh)
            target_obj.matrix_world = cloth_obj.matrix_world.copy()

            # 既存のシェイプキーをクリーンアップ
            if target_obj.data.shape_keys:
                if hasattr(target_obj, "shape_key_clear"):
                    target_obj.shape_key_clear()
                else:
                    while target_obj.data.shape_keys.key_blocks:
                        target_obj.shape_key_remove(target_obj.data.shape_keys.key_blocks[0])

            # 布設定は無効にして純粋な展示・レンダリングメッシュとする
            if hasattr(target_obj, "taremin_cloth"):
                target_obj.taremin_cloth.is_cloth = False

            # コレクションにリンク
            parent_col = cloth_obj.users_collection[0] if getattr(cloth_obj, "users_collection", None) else (context.collection or context.scene.collection)
            parent_col.objects.link(target_obj)

            # メッシュ本体の頂点座標を初期レスト座標（変形前）にリセット
            target_obj.data.vertices.foreach_set("co", initial_coords)
            target_obj.data.update()

            # Basis キーを作成し、初期レスト座標をセット
            basis = target_obj.shape_key_add(name="Basis", from_mix=False)
            basis.data.foreach_set("co", initial_coords)
            target_obj.data.update()

        # 既存ターゲットだが Basis キーがない場合のフォールバック
        if not target_obj.data.shape_keys:
            target_obj.data.vertices.foreach_set("co", initial_coords)
            target_obj.data.update()
            basis = target_obj.shape_key_add(name="Basis", from_mix=False)
            basis.data.foreach_set("co", initial_coords)
            target_obj.data.update()

        # 5. 新規シェイプキーを追加
        # 既存シェイプキーの変形が加算合成されてメッシュが崩れるのを防ぐため、
        # 既存キーのウェイトを 0.0 にリセットし、今回追加する新規キーのみを 1.0 にする
        if target_obj.data.shape_keys:
            for kb in target_obj.data.shape_keys.key_blocks:
                kb.value = 0.0

        key_name = self.shape_key_name.strip() or "Cloth_Shape"
        key = target_obj.shape_key_add(name=key_name, from_mix=False)
        key.data.foreach_set("co", curr_coords)
        key.value = 1.0

        # 新規キーをアクティブシェイプキーとして選択
        if target_obj.data.shape_keys:
            target_obj.active_shape_key_index = len(target_obj.data.shape_keys.key_blocks) - 1

        target_obj.data.update()

        # 6. 元の布オブジェクトのアクティブ・選択状態を維持
        if hasattr(context, "view_layer") and context.view_layer:
            context.view_layer.objects.active = cloth_obj
            cloth_obj.select_set(True)

        logger.info(f"[ShapeKey] Saved shape key '{key.name}' to '{target_obj.name}' (verts={n_verts})")
        self.report(
            {'INFO'},
            i18n.trans("Saved shape key '%s' to '%s'") % (key.name, target_obj.name)
        )
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_create_pin_group(bpy.types.Operator):
    """Create a new vertex group for pinning and switch to weight paint mode"""
    bl_idname = "taremin_cloth.create_pin_group"
    bl_label = "Create Pin Group"
    bl_description = "Create a new vertex group for pinning and switch to weight paint mode"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bool(
            obj
            and obj.type == 'MESH'
            and getattr(obj, "taremin_cloth", None)
            and obj.taremin_cloth.is_cloth
        )

    def execute(self, context):
        obj = context.active_object
        settings = obj.taremin_cloth

        # 1. 新規頂点グループを作成（同名重複時はBlenderが自動で Cloth_Pin.001 等を採番）
        new_vg = obj.vertex_groups.new(name="Cloth_Pin")
        settings.pin_vertex_group = new_vg.name
        obj.vertex_groups.active_index = new_vg.index

        # 2. ウェイトペイントモードへ移行
        if context.mode != 'PAINT_WEIGHT' and bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='WEIGHT_PAINT')

        msg = i18n.trans("Created pin vertex group '%s' and switched to Weight Paint mode") % new_vg.name
        logger.info(f"[Pin] {msg}")
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_toggle_weight_paint(bpy.types.Operator):
    """Toggle between Object Mode and Weight Paint Mode for active pin vertex group"""
    bl_idname = "taremin_cloth.toggle_weight_paint"
    bl_label = "Toggle Weight Paint"
    bl_description = "Toggle between Object Mode and Weight Paint Mode for active pin vertex group"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bool(
            obj
            and obj.type == 'MESH'
            and getattr(obj, "taremin_cloth", None)
            and obj.taremin_cloth.is_cloth
        )

    def execute(self, context):
        obj = context.active_object
        settings = obj.taremin_cloth

        if context.mode == 'PAINT_WEIGHT':
            if bpy.ops.object.mode_set.poll():
                bpy.ops.object.mode_set(mode='OBJECT')
            self.report({'INFO'}, i18n.trans("Switched to Object Mode"))
            return {'FINISHED'}

        # オブジェクトモード等からウェイトペイントモードへ移行
        vg_name = (settings.pin_vertex_group or "").strip() or "Cloth_Pin"
        vg = obj.vertex_groups.get(vg_name)
        if not vg:
            vg = obj.vertex_groups.new(name=vg_name)
            settings.pin_vertex_group = vg.name

        obj.vertex_groups.active_index = vg.index

        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='WEIGHT_PAINT')

        msg = i18n.trans("Switched to Weight Paint Mode (%s)") % vg.name
        logger.info(f"[Pin] {msg}")
        self.report({'INFO'}, msg)
        return {'FINISHED'}



