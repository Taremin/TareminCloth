import bpy
from . import i18n
from .operators import is_interactive_running
from .utils import topology


class TAREMIN_CLOTH_UL_elastic_groups(bpy.types.UIList):
    """伸縮グループ一覧のUIList"""
    bl_translation_context = i18n.CONTEXT
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="")
            row.prop(item, "name", text="", emboss=False, icon='EDGESEL')
            row.prop(item, "scale", text=i18n.trans("Scale"), slider=True)
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon='EDGESEL')


def _draw_diagnostics_box(layout, context):
    """GPU設定およびログレベル等の診断情報ボックスを描画する"""
    from .preferences import get_preferences
    prefs = get_preferences(context)
    if not prefs:
        return

    box_diag = layout.box()
    box_diag.label(text=i18n.trans("GPU & Diagnostics"), icon='PREFERENCES')

    col = box_diag.column(align=True)
    col.prop(prefs, "gpu_backend", text=i18n.trans("Backend"))
    col.prop(prefs, "gpu_device", text=i18n.trans("Device"))

    row_status = box_diag.row(align=True)
    active_name = prefs.active_device_name
    active_be = prefs.active_backend_name
    if active_name and active_name != "Unknown":
        row_status.label(text=f"{i18n.trans('Active:')} {active_name} ({active_be})", icon='CHECKMARK')
    else:
        row_status.label(text=i18n.trans("Active: Uninitialized (Auto)"), icon='INFO')

    box_diag.operator("taremin_cloth.apply_gpu_settings", text=i18n.trans("Apply GPU Settings"), icon='FILE_REFRESH')

    row_log = box_diag.row(align=True)
    row_log.label(text=i18n.trans("Log Level:"), icon='CONSOLE')
    row_log.prop(prefs, "log_level", text="")


class TAREMIN_CLOTH_PT_objects_panel(bpy.types.Panel):
    """GPU Cloth / GPU Collider が設定されたオブジェクト一覧パネル"""
    bl_label = "Cloth & Collider Objects"
    bl_idname = "TAREMIN_CLOTH_PT_objects_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Taremin Cloth"
    bl_order = 0
    bl_translation_context = i18n.CONTEXT

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        if not scene:
            return

        active_obj = context.active_object

        # シーン内の全オブジェクトからClothとColliderを抽出
        cloth_objs = [
            obj for obj in scene.objects
            if getattr(obj, "taremin_cloth", None) and obj.taremin_cloth.is_cloth
        ]
        collider_objs = [
            obj for obj in scene.objects
            if getattr(obj, "taremin_cloth_collider", None) and obj.taremin_cloth_collider.is_collider
        ]

        # --- UIモード切り替え (Simple / Advanced) ---
        row_mode = layout.row(align=True)
        row_mode.prop(scene, "taremin_cloth_ui_mode", expand=True)

        # --- シミュレーション構成プリセット & 一括操作 ---
        box_config = layout.box()
        row_cfg = box_config.row(align=True)
        preset_title = scene.taremin_cloth_active_config_preset or i18n.trans("Config Preset")
        row_cfg.menu("TAREMIN_CLOTH_MT_config_presets", text=preset_title, icon='SETTINGS')
        row_cfg.operator("taremin_cloth.save_config_preset", text="", icon='ADD')
        if scene.taremin_cloth_active_config_preset:
            row_cfg.operator("taremin_cloth.delete_config_preset", text="", icon='REMOVE')

        row_batch = box_config.row(align=True)
        op_on = row_batch.operator("taremin_cloth.batch_simulation_state", text=i18n.trans("All ON"))
        op_on.action = 'ALL_ON'
        op_off = row_batch.operator("taremin_cloth.batch_simulation_state", text=i18n.trans("All OFF"))
        op_off.action = 'ALL_OFF'
        op_solo = row_batch.operator("taremin_cloth.batch_simulation_state", text=i18n.trans("Solo"))
        op_solo.action = 'SOLO'

        row_glob = box_config.row(align=True)
        row_glob.operator("taremin_cloth.reset_all", text=i18n.trans("Reset All"), icon='RECOVER_LAST')
        op_clr_all = row_glob.operator("taremin_cloth.apply_rest_shape", text=i18n.trans("Apply All"), icon='CHECKMARK')
        if op_clr_all:
            op_clr_all.all_objects = True
        row_glob.operator("taremin_cloth.clear_cache", text=i18n.trans("Clear Cache"), icon='TRASH')

        # --- Cloth Objects セクション ---
        box_cloth = layout.box()
        row_c_hdr = box_cloth.row(align=True)
        row_c_hdr.label(text=f"{i18n.trans('Cloth Objects')} ({len(cloth_objs)})", icon='MOD_CLOTH')

        if cloth_objs:
            col_c = box_cloth.column(align=True)
            for obj in cloth_objs:
                row = col_c.row(align=True)
                is_active = (obj == active_obj)
                is_sim_enabled = obj.taremin_cloth.enabled

                # 選択オペレーターボタン
                op = row.operator(
                    "taremin_cloth.select_object",
                    text=obj.name,
                    icon='OBJECT_DATAMODE',
                    depress=is_active,
                )
                op.object_name = obj.name

                # レイヤー情報バッジ（L0, L1など）
                row.label(text=f"L{obj.taremin_cloth.layer_id}")

                # シミュレーション有効/無効トグル (PHYSICS アイコン)
                row.prop(
                    obj.taremin_cloth,
                    "enabled",
                    text="",
                    icon='PHYSICS' if is_sim_enabled else 'CHECKBOX_DEHLT',
                )

                # ビューポート可視性トグル（目のアイコン）
                row.prop(obj, "hide_viewport", text="", emboss=False)
        else:
            box_cloth.label(text=i18n.trans("No cloth objects configured"), icon='INFO')

        # --- Collider Objects セクション ---
        box_col = layout.box()
        row_col_hdr = box_col.row(align=True)
        row_col_hdr.label(text=f"{i18n.trans('Collider Objects')} ({len(collider_objs)})", icon='PHYSICS')

        if collider_objs:
            col_col = box_col.column(align=True)
            for obj in collider_objs:
                row = col_col.row(align=True)
                is_active = (obj == active_obj)
                is_col_enabled = obj.taremin_cloth_collider.enabled
                col_type = obj.taremin_cloth_collider.collider_type

                # 形状に応じたアイコン決定
                if col_type == 'SPHERE':
                    col_icon = 'MESH_UVSPHERE'
                elif col_type == 'CAPSULE':
                    col_icon = 'MESH_CAPSULE'
                elif col_type == 'PLANE':
                    col_icon = 'MESH_PLANE'
                elif col_type == 'BONE_SDF':
                    col_icon = 'ARMATURE_DATA'
                else:
                    col_icon = 'MESH_DATA'

                # 選択オペレーターボタン
                op = row.operator(
                    "taremin_cloth.select_object",
                    text=obj.name,
                    icon=col_icon,
                    depress=is_active,
                )
                op.object_name = obj.name

                # 形状ラベル
                type_labels = {
                    'SPHERE': "Sphere",
                    'CAPSULE': "Capsule",
                    'PLANE': "Plane",
                    'MESH': "Mesh",
                    'BONE_SDF': "Bone SDF",
                    'MESH_SDF': "Mesh SDF",
                }
                row.label(text=i18n.trans(type_labels.get(col_type, col_type)))

                # コライダー有効/無効トグル (PHYSICS アイコン)
                row.prop(
                    obj.taremin_cloth_collider,
                    "enabled",
                    text="",
                    icon='PHYSICS' if is_col_enabled else 'CHECKBOX_DEHLT',
                )

                # ビューポート可視性トグル
                row.prop(obj, "hide_viewport", text="", emboss=False)
        else:
            box_col.label(text=i18n.trans("No collider objects configured"), icon='INFO')


def _is_cloth_active(context):
    """選択中のオブジェクトが布メッシュであるかを判定する共通ヘルパー"""
    obj = getattr(context, "active_object", None)
    return (
        obj is not None
        and obj.type == 'MESH'
        and getattr(obj, "taremin_cloth", None) is not None
        and obj.taremin_cloth.is_cloth
    )


def _is_cloth_active_advanced(context):
    """ADVANCEDモードかつ布オブジェクトがアクティブであるかを判定する共通ヘルパー"""
    if not _is_cloth_active(context):
        return False
    scene = getattr(context, "scene", None)
    return scene is not None and getattr(scene, "taremin_cloth_ui_mode", "SIMPLE") == 'ADVANCED'


class TAREMIN_CLOTH_PT_main_panel(bpy.types.Panel):
    """3Dビューポートのサイドバー（Nパネル）に表示されるGPU Cloth親パネル"""
    bl_label = "GPU Cloth"
    bl_idname = "TAREMIN_CLOTH_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Taremin Cloth"
    bl_order = 1
    bl_translation_context = i18n.CONTEXT

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        if scene:
            row_mode = layout.row(align=True)
            row_mode.prop(scene, "taremin_cloth_ui_mode", expand=True)

        obj = context.active_object

        if not obj or obj.type != 'MESH':
            layout.label(text=i18n.trans("Please select a mesh object"), icon='INFO')
            return

        # スケール未適用警告 (Scale != 1.0)
        scale = obj.scale
        if abs(scale.x - 1.0) > 1e-3 or abs(scale.y - 1.0) > 1e-3 or abs(scale.z - 1.0) > 1e-3:
            box_warn = layout.box()
            box_warn.alert = True
            box_warn.label(text=f"{i18n.trans('Unapplied Scale:')} ({scale.x:.2f}, {scale.y:.2f}, {scale.z:.2f})", icon='ERROR')
            box_warn.label(text=i18n.trans("Please apply scale to avoid simulation instability"))
            op_scale = box_warn.operator("object.transform_apply", text=i18n.trans("Apply Scale (Ctrl+A)"), icon='CHECKMARK')
            op_scale.location = False
            op_scale.rotation = False
            op_scale.scale = True

        settings = getattr(obj, "taremin_cloth", None)
        if not settings:
            return

        col = layout.column(align=True)
        if not settings.is_cloth:
            col.operator("taremin_cloth.toggle_cloth", text=i18n.trans("Enable Cloth"), icon='MOD_CLOTH')
            return

        col.operator("taremin_cloth.toggle_cloth", text=i18n.trans("Disable Cloth"), icon='CANCEL')

        ui_mode = getattr(scene, "taremin_cloth_ui_mode", "SIMPLE") if scene else "SIMPLE"

        if ui_mode == 'SIMPLE':
            # --- 簡単モード (Simple Mode) ---
            # 個別操作 (Quick Controls)
            box_selected = layout.box()
            box_selected.label(text=f"{i18n.trans('Selected:')} {obj.name}", icon='OBJECT_DATA')
            col_sel = box_selected.column(align=True)
            col_sel.prop(settings, "enabled", text=i18n.trans("Simulation Active"), icon='PHYSICS')
            if is_interactive_running():
                col_sel.operator("taremin_cloth.interactive", text=i18n.trans("Stop Interactive Mode"), icon='CANCEL', depress=True)
            else:
                col_sel.operator("taremin_cloth.interactive", text=i18n.trans("Interactive Mode (Grab/Drag)"), icon='HAND', depress=False)
            row_sel = col_sel.row(align=True)
            row_sel.operator("taremin_cloth.reset_selected", text=i18n.trans("Reset"), icon='FILE_REFRESH')
            row_sel.operator("taremin_cloth.clear_cache", text=i18n.trans("Clear Cache"), icon='TRASH')

            # 1. 固定 (Attachment & Pinning)
            box_pin = layout.box()
            box_pin.label(text=i18n.trans("Attachment & Pinning"), icon='PINNED')
            col_pin = box_pin.column(align=True)
            col_pin.prop_search(settings, "pin_vertex_group", obj, "vertex_groups", text=i18n.trans("Pin Group"))
            col_pin.prop(settings, "pin_target_object", text=i18n.trans("Target Object"))
            if settings.pin_target_object and settings.pin_target_object.type == 'ARMATURE':
                col_pin.prop_search(settings, "pin_target_bone", settings.pin_target_object.data, "bones", text=i18n.trans("Bone"))

            # 2. 縫合 (Sewing)
            box_sew = layout.box()
            box_sew.label(text=i18n.trans("Sewing"), icon='MOD_CLOTH')
            col_sew = box_sew.column(align=True)
            col_sew.prop(settings, "enable_sewing", text=i18n.trans("Enable Sewing"))
            if settings.enable_sewing:
                col_sew.prop(settings, "sewing_shrink_speed", text=i18n.trans("Shrink Speed"))
                col_sew.operator("taremin_cloth.create_seam", text=i18n.trans("Create Seam (Select 2 Verts)"), icon='EDGESEL')

            # 3. 素材プリセット (Fabric Material)
            box_mat = layout.box()
            box_mat.label(text=i18n.trans("Fabric Material"), icon='MATERIAL')
            row_preset = box_mat.row(align=True)
            preset_title = f"{i18n.trans('Material:')} {settings.last_fabric_preset}" if settings.last_fabric_preset else i18n.trans("Material Preset")
            row_preset.menu("TAREMIN_CLOTH_MT_fabric_presets", text=preset_title, icon='MATERIAL')
            row_thick = box_mat.row(align=True)
            row_thick.prop(settings, "thickness", text=i18n.trans("Thickness"))
            row_thick.operator("taremin_cloth.auto_fit_thickness", text=i18n.trans("Auto Fit"), icon='FIXED_SIZE')

            # 4. 自己衝突 (Self Collision)
            box_sc = layout.box()
            box_sc.label(text=i18n.trans("Self Collision"), icon='PHYSICS')
            col_sc = box_sc.column(align=True)
            col_sc.prop(settings, "enable_self_collision", text=i18n.trans("Enable Self Collision"))
            if settings.enable_self_collision:
                row_sc = col_sc.row(align=True)
                row_sc.prop(settings, "self_collision_purpose", text=i18n.trans("Purpose"))
                row_sc.operator("taremin_cloth.auto_fit_self_collision", text=i18n.trans("Auto Fit"), icon='FIXED_SIZE')
                col_sc.prop(settings, "thickness", text=i18n.trans("Thickness"))

            # 4. 重力 (Forces & Gravity)
            box_grav = layout.box()
            box_grav.label(text=i18n.trans("Forces & Gravity"), icon='FORCE_VORTEX')
            col_grav = box_grav.column(align=True)
            col_grav.prop(settings, "gravity", slider=True)

            # 5. シミュレーション品質プリセット (Simulation Quality)
            box_qual = layout.box()
            box_qual.label(text=i18n.trans("Simulation Quality"), icon='PREFERENCES')
            row_q = box_qual.row(align=True)
            sim_preset_title = f"{i18n.trans('Quality:')} {settings.last_simulation_preset}" if settings.last_simulation_preset else i18n.trans("Quality Preset")
            row_q.menu("TAREMIN_CLOTH_MT_simulation_presets", text=sim_preset_title, icon='SETTINGS')
        else:
            # --- 詳細モード (Advanced Mode) ---
            box_selected = layout.box()
            box_selected.label(text=f"{i18n.trans('Selected:')} {obj.name}", icon='OBJECT_DATA')
            col_sel = box_selected.column(align=True)
            col_sel.prop(settings, "enabled", text=i18n.trans("Simulation Active"), icon='PHYSICS')
            if is_interactive_running():
                col_sel.operator("taremin_cloth.interactive", text=i18n.trans("Stop Interactive Mode"), icon='CANCEL', depress=True)
            else:
                col_sel.operator("taremin_cloth.interactive", text=i18n.trans("Interactive Mode (Grab/Drag)"), icon='HAND', depress=False)
            row_sel = col_sel.row(align=True)
            row_sel.operator("taremin_cloth.reset_selected", text=i18n.trans("Reset"), icon='FILE_REFRESH')
            row_sel.operator("taremin_cloth.clear_cache", text=i18n.trans("Clear Cache"), icon='TRASH')
            op_clr = row_sel.operator("taremin_cloth.apply_rest_shape", text=i18n.trans("Apply Rest"), icon='CHECKMARK')
            if op_clr:
                op_clr.all_objects = False


class TAREMIN_CLOTH_PT_pinning(bpy.types.Panel):
    """固定ピンおよびボーン追従設定サブパネル"""
    bl_label = "Attachment & Pinning"
    bl_idname = "TAREMIN_CLOTH_PT_pinning"
    bl_parent_id = "TAREMIN_CLOTH_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Taremin Cloth"
    bl_translation_context = i18n.CONTEXT

    @classmethod
    def poll(cls, context):
        return _is_cloth_active_advanced(context)

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        settings = obj.taremin_cloth

        col = layout.column(align=True)
        col.prop_search(settings, "pin_vertex_group", obj, "vertex_groups", text=i18n.trans("Pin Group"))
        col.prop(settings, "pin_color", text=i18n.trans("Pin Color"))

        row_pin_opts = col.row(align=True)
        row_pin_opts.prop(settings, "pin_overlay_interactive_only", text=i18n.trans("Interactive Only"))
        row_pin_opts.prop(settings, "overlay_depth_test", text=i18n.trans("Depth Test (Z)"))

        col.separator()
        col.prop(settings, "pin_target_object", text=i18n.trans("Target Object"))
        if settings.pin_target_object and settings.pin_target_object.type == 'ARMATURE':
            col.prop_search(settings, "pin_target_bone", settings.pin_target_object.data, "bones", text=i18n.trans("Bone"))


class TAREMIN_CLOTH_PT_fabric(bpy.types.Panel):
    """布素材・物性設定サブパネル (プリセット・剛性・減衰・厚み)"""
    bl_label = "Fabric & Material"
    bl_idname = "TAREMIN_CLOTH_PT_fabric"
    bl_parent_id = "TAREMIN_CLOTH_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Taremin Cloth"
    bl_translation_context = i18n.CONTEXT

    @classmethod
    def poll(cls, context):
        return _is_cloth_active_advanced(context)

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        settings = obj.taremin_cloth

        # プリセットセレクター
        row_preset = layout.row(align=True)
        preset_title = f"{i18n.trans('Material:')} {settings.last_fabric_preset}" if settings.last_fabric_preset else i18n.trans("Material Preset")
        row_preset.menu("TAREMIN_CLOTH_MT_fabric_presets", text=preset_title, icon='MATERIAL')
        op_add = row_preset.operator("taremin_cloth.save_preset", text="", icon='ADD')
        if op_add:
            op_add.category = 'fabric'
        op_del = row_preset.operator("taremin_cloth.delete_preset", text="", icon='REMOVE')
        if op_del:
            op_del.category = 'fabric'

        # 剛性 (Stiffness)
        box_stiff = layout.box()
        box_stiff.label(text=i18n.trans("Stiffness"), icon='PHYSICS')
        s_col = box_stiff.column(align=True)
        s_col.prop(settings, "tension_stiffness", slider=True)
        s_col.prop(settings, "compression_stiffness", slider=True)
        s_col.prop(settings, "shear_stiffness", slider=True)
        s_col.prop(settings, "bending_stiffness", slider=True)

        # 減衰 (Damping)
        box_damp = layout.box()
        box_damp.label(text=i18n.trans("Damping"), icon='FORCE_DRAG')
        d_col = box_damp.column(align=True)
        d_col.prop(settings, "air_damping", slider=True)
        d_col.prop(settings, "tension_damping", slider=True)
        d_col.prop(settings, "compression_damping", slider=True)
        d_col.prop(settings, "shear_damping", slider=True)
        d_col.prop(settings, "bending_damping", slider=True)

        # 物性厚み (Thickness)
        row_thick = layout.row(align=True)
        row_thick.prop(settings, "thickness", text=i18n.trans("Thickness"))
        row_thick.operator("taremin_cloth.auto_fit_thickness", text=i18n.trans("Auto Fit"), icon='FIXED_SIZE')


class TAREMIN_CLOTH_PT_forces(bpy.types.Panel):
    """重力・外力設定サブパネル"""
    bl_label = "Forces"
    bl_idname = "TAREMIN_CLOTH_PT_forces"
    bl_parent_id = "TAREMIN_CLOTH_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Taremin Cloth"
    bl_translation_context = i18n.CONTEXT

    @classmethod
    def poll(cls, context):
        return _is_cloth_active_advanced(context)

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        settings = obj.taremin_cloth
        scene = context.scene

        col = layout.column(align=True)
        col.prop(settings, "gravity", slider=True)

        # シーン重力状態インジケーター
        box_info = layout.box()
        if scene and getattr(scene, "use_gravity", True):
            sg = scene.gravity
            box_info.label(text=f"{i18n.trans('Scene Gravity:')} ({sg.x:.1f}, {sg.y:.1f}, {sg.z:.1f}) m/s²", icon='PHYSICS')
        else:
            box_info.label(text=i18n.trans("Scene Gravity: Disabled (0 m/s²)"), icon='INFO')


class TAREMIN_CLOTH_PT_collisions(bpy.types.Panel):
    """コライダー接触および自己衝突・レイヤー設定サブパネル"""
    bl_label = "Collisions & Layers"
    bl_idname = "TAREMIN_CLOTH_PT_collisions"
    bl_parent_id = "TAREMIN_CLOTH_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Taremin Cloth"
    bl_options = {'DEFAULT_CLOSED'}
    bl_translation_context = i18n.CONTEXT

    @classmethod
    def poll(cls, context):
        return _is_cloth_active_advanced(context)

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        settings = obj.taremin_cloth

        # コライダー接触
        box_col = layout.box()
        box_col.label(text=i18n.trans("Collider Interaction"), icon='PHYSICS')
        c_col = box_col.column(align=True)
        c_col.prop(settings, "enable_edge_collision")
        if settings.enable_edge_collision:
            edge_sub = c_col.column(align=True)
            edge_sub.prop(settings, "edge_margin_scale", text=i18n.trans("Margin Scale"))
            edge_sub.prop(settings, "edge_margin_offset", text=i18n.trans("Margin Offset"))

        # 自己・レイヤー衝突
        box_layer = layout.box()
        box_layer.label(text=i18n.trans("Self & Layer Collision"), icon='RENDERLAYERS')
        l_col = box_layer.column(align=True)
        l_col.prop(settings, "enable_self_collision")
        if settings.enable_self_collision:
            row_p = l_col.row(align=True)
            row_p.prop(settings, "self_collision_purpose", text=i18n.trans("Purpose"))
            row_p.operator("taremin_cloth.auto_fit_self_collision", text=i18n.trans("Auto Fit"), icon='FIXED_SIZE')

            self_sub = l_col.column(align=True)
            self_sub.prop(settings, "self_collision_relief_factor", text=i18n.trans("Relief Factor"))
            self_sub.prop(settings, "self_collision_max_displacement_ratio", text=i18n.trans("Max Step Ratio"))
            self_sub.prop(settings, "self_collision_max_iterations", text=i18n.trans("Search Limit"))
            self_sub.prop(settings, "enable_normal_untangling", text=i18n.trans("Normal Untangling"))
            self_sub.prop(settings, "coupled_self_collision_mode", text=i18n.trans("Coupled Mode"))
            if settings.coupled_self_collision_mode != 'OFF':
                self_sub.prop(settings, "post_collision_relaxation_iters", text=i18n.trans("Relax Steps"))
        l_col.separator()
        l_col.prop(settings, "layer_id")
        row_thick = l_col.row(align=True)
        row_thick.prop(settings, "thickness")
        row_thick.operator("taremin_cloth.auto_fit_thickness", text=i18n.trans("Auto Fit"), icon='FIXED_SIZE')


class TAREMIN_CLOTH_PT_pattern(bpy.types.Panel):
    """縫合および伸縮ゴム設定サブパネル"""
    bl_label = "Pattern & Tailoring"
    bl_idname = "TAREMIN_CLOTH_PT_pattern"
    bl_parent_id = "TAREMIN_CLOTH_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Taremin Cloth"
    bl_options = {'DEFAULT_CLOSED'}
    bl_translation_context = i18n.CONTEXT

    @classmethod
    def poll(cls, context):
        return _is_cloth_active_advanced(context)

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        settings = obj.taremin_cloth

        # 縫合 (Sewing)
        box_sew = layout.box()
        box_sew.label(text=i18n.trans("Sewing (Pattern Seaming)"), icon='MOD_CLOTH')
        s_col = box_sew.column(align=True)
        s_col.prop(settings, "enable_sewing")
        if settings.enable_sewing:
            s_col.prop(settings, "sewing_shrink_speed")
            s_col.operator("taremin_cloth.create_seam", text=i18n.trans("Create Seam Between 2 Verts"), icon='EDGESEL')

        # 伸縮グループ (Elastic Bands)
        box_elastic = layout.box()
        box_elastic.label(text=i18n.trans("Elastic Bands / Edge Scaling"), icon='MOD_SHRINKWRAP')
        row = box_elastic.row()
        row.template_list(
            "TAREMIN_CLOTH_UL_elastic_groups",
            "",
            settings,
            "elastic_groups",
            settings,
            "active_elastic_group_index",
            rows=2,
        )
        col_ops = row.column(align=True)
        col_ops.operator("taremin_cloth.add_elastic_group", text="", icon='ADD')
        col_ops.operator("taremin_cloth.remove_elastic_group", text="", icon='REMOVE')
        col_ops.separator()
        col_ops.operator("taremin_cloth.assign_elastic_edges", text="", icon='FILE_REFRESH')
        col_ops.operator("taremin_cloth.select_elastic_edges", text="", icon='RESTRICT_SELECT_OFF')

        if 0 <= settings.active_elastic_group_index < len(settings.elastic_groups):
            active_grp = settings.elastic_groups[settings.active_elastic_group_index]
            col_details = box_elastic.column(align=True)
            col_details.prop(active_grp, "name", text=i18n.trans("Name"))
            col_details.prop(active_grp, "scale", text=i18n.trans("Scale (Rest Length)"), slider=True)
            col_details.prop(active_grp, "color", text=i18n.trans("Line Color"))
            n_edges = len(active_grp.get_edge_indices())
            col_details.label(text=f"{i18n.trans('Registered Edges:')} {n_edges}", icon='INFO')

        row_elastic_disp = box_elastic.row(align=True)
        row_elastic_disp.prop(settings, "show_elastic_overlay", text=i18n.trans("Show Overlay"))
        if settings.show_elastic_overlay:
            row_elastic_disp.prop(settings, "elastic_overlay_interactive_only", text=i18n.trans("Interactive Only"))
            row_elastic_disp.prop(settings, "overlay_depth_test", text=i18n.trans("Depth Test (Z)"))


class TAREMIN_CLOTH_PT_quality(bpy.types.Panel):
    """シミュレーション品質およびソルバー設定サブパネル"""
    bl_label = "Quality & Solver"
    bl_idname = "TAREMIN_CLOTH_PT_quality"
    bl_parent_id = "TAREMIN_CLOTH_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Taremin Cloth"
    bl_options = {'DEFAULT_CLOSED'}
    bl_translation_context = i18n.CONTEXT

    @classmethod
    def poll(cls, context):
        return _is_cloth_active_advanced(context)

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        settings = obj.taremin_cloth

        # 品質プリセット
        box_sim = layout.box()
        row = box_sim.row(align=True)
        sim_preset_title = f"{i18n.trans('Quality:')} {settings.last_simulation_preset}" if settings.last_simulation_preset else i18n.trans("Quality Preset")
        row.menu("TAREMIN_CLOTH_MT_simulation_presets", text=sim_preset_title, icon='SETTINGS')
        op_add = row.operator("taremin_cloth.save_preset", text="", icon='ADD')
        if op_add:
            op_add.category = 'simulation'
        op_del = row.operator("taremin_cloth.delete_preset", text="", icon='REMOVE')
        if op_del:
            op_del.category = 'simulation'

        sim_col = box_sim.column(align=True)
        sim_col.prop(settings, "substeps")
        sim_col.prop(settings, "solver_iterations")
        sim_col.prop(settings, "enable_adaptive_substep")
        if settings.enable_adaptive_substep:
            row_steps = sim_col.row(align=True)
            row_steps.prop(settings, "min_substeps", text=i18n.trans("Min"))
            row_steps.prop(settings, "max_substeps", text=i18n.trans("Max"))

        # ソルバー & パフォーマンス
        box_perf = layout.box()
        box_perf.label(text=i18n.trans("Performance Tuning"), icon='PREFERENCES')
        p_col = box_perf.column(align=True)
        p_col.prop(settings, "solver_mode")
        row_buf = p_col.row(align=True)
        row_buf.prop(settings, "enable_frame_buffering")
        if settings.enable_frame_buffering:
            row_buf.prop(settings, "frame_buffer_size")
        if hasattr(context.scene, "taremin_cloth_fast_playback"):
            p_col.prop(context.scene, "taremin_cloth_fast_playback")


class TAREMIN_CLOTH_PT_topology(bpy.types.Panel):
    """メッシュトポロジー・動的分割設定サブパネル"""
    bl_label = "Topology & Triangulation"
    bl_idname = "TAREMIN_CLOTH_PT_topology"
    bl_parent_id = "TAREMIN_CLOTH_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Taremin Cloth"
    bl_options = {'DEFAULT_CLOSED'}
    bl_translation_context = i18n.CONTEXT

    @classmethod
    def poll(cls, context):
        return _is_cloth_active_advanced(context)

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        settings = obj.taremin_cloth

        t_col = layout.column(align=True)
        t_col.prop(settings, "triangulation_mode", text=i18n.trans("Mode"))

        if settings.triangulation_mode == 'DYNAMIC_DIAGONAL':
            t_col.prop(settings, "dynamic_preserve_flat")
            if settings.dynamic_preserve_flat:
                t_col.prop(settings, "dynamic_flatness_threshold")
            t_col.prop(settings, "auto_triangulate_on_stop")

            row_topo_ops = t_col.row(align=True)
            row_topo_ops.operator("taremin_cloth.apply_dynamic_diagonal", text=i18n.trans("Split by Strain"), icon='MOD_TRIANGULATE')
            row_topo_ops.operator("taremin_cloth.restore_quad_topology", text=i18n.trans("Restore Quad"), icon='RECOVER_LAST')

        elif settings.triangulation_mode == 'CROSS_SUBDIV':
            t_col.prop(settings, "post_process_mode")
            if settings.post_process_mode == 'ADAPTIVE':
                t_col.prop(settings, "adaptive_flatness_threshold")
            t_col.prop(settings, "auto_post_process")

            row_topo_ops = t_col.row(align=True)
            is_subdivided = topology.is_cross_subdivided(obj)
            if not is_subdivided:
                row_topo_ops.operator("taremin_cloth.apply_cross_subdivision", text=i18n.trans("Subdivide Quads (Poke)"), icon='MOD_TRIANGULATE')
            else:
                row_topo_ops.operator("taremin_cloth.apply_post_process", text=i18n.trans("Apply Post-Process"), icon='CHECKMARK')
                row_topo_ops.operator("taremin_cloth.restore_quad_topology", text=i18n.trans("Restore Quad"), icon='RECOVER_LAST')


class TAREMIN_CLOTH_PT_interactive_opts(bpy.types.Panel):
    """インタラクティブシミュレーション設定サブパネル"""
    bl_label = "Interactive Options"
    bl_idname = "TAREMIN_CLOTH_PT_interactive_opts"
    bl_parent_id = "TAREMIN_CLOTH_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Taremin Cloth"
    bl_options = {'DEFAULT_CLOSED'}
    bl_translation_context = i18n.CONTEXT

    @classmethod
    def poll(cls, context):
        return _is_cloth_active_advanced(context)

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        settings = obj.taremin_cloth

        col = layout.column(align=True)
        col.prop(settings, "interactive_realtime_sync", text=i18n.trans("Real-time Sync"))
        if settings.interactive_realtime_sync:
            col.prop(settings, "interactive_max_steps", text=i18n.trans("Max Steps / Frame"))
        col.prop(settings, "isolate_viewport_view", text=i18n.trans("Isolate View (Local)"))
        row_fps = col.row(align=True)
        row_fps.prop(settings, "show_fps_overlay", text=i18n.trans("Show FPS"))
        if settings.show_fps_overlay:
            row_fps.prop(settings, "fps_overlay_position", text="")
        col.separator()
        col.operator("taremin_cloth.benchmark_fps", text=i18n.trans("Benchmark FPS (2 sec)"), icon='TIME')


class TAREMIN_CLOTH_PT_gui_experimental(bpy.types.Panel):
    """独立GUIクライアント（実験的機能）サブパネル"""
    bl_label = "Standalone GUI (Experimental)"
    bl_idname = "TAREMIN_CLOTH_PT_gui_experimental"
    bl_parent_id = "TAREMIN_CLOTH_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Taremin Cloth"
    bl_options = {'DEFAULT_CLOSED'}
    bl_translation_context = i18n.CONTEXT

    @classmethod
    def poll(cls, context):
        from .preferences import get_preferences
        prefs = get_preferences(context)
        if not prefs or not getattr(prefs, "enable_standalone_gui", False):
            return False
        return _is_cloth_active_advanced(context)

    def draw(self, context):
        layout = self.layout
        from .engine.gui_client import get_gui_client
        from .ops.gui import is_gui_preview_running, get_gui_fps_stats

        client = get_gui_client()
        col_gui = layout.column(align=True)
        if client.is_connected:
            b_fps, g_fps = get_gui_fps_stats()
            conn_txt = i18n.trans("Connected")
            stat_text = f"● {conn_txt} | GUI: {g_fps:.0f} FPS" if g_fps > 0 else f"● {conn_txt}"
            col_gui.label(text=stat_text, icon='CHECKMARK')
            row_ctrl = col_gui.row(align=True)
            if is_gui_preview_running():
                row_ctrl.operator("taremin_cloth.stop_gui_preview", text=i18n.trans("Stop Preview"), icon='CANCEL')
            else:
                row_ctrl.operator("taremin_cloth.gui_preview", text=i18n.trans("Live Preview"), icon='PLAY')
            row_ctrl.operator("taremin_cloth.apply_gui_pose", text=i18n.trans("Apply to Mesh"), icon='CHECKMARK')
            col_gui.operator("taremin_cloth.sync_gui_colliders", text=i18n.trans("Sync Colliders"), icon='FILE_REFRESH')
        else:
            col_gui.operator("taremin_cloth.launch_gui", text=i18n.trans("Launch Taremin Cloth GUI"), icon='WINDOW')
            col_gui.operator("taremin_cloth.connect_gui", text=i18n.trans("Connect to Existing GUI"), icon='LINKED')


class TAREMIN_CLOTH_PT_diagnostics(bpy.types.Panel):
    """GPU設定およびログ診断サブパネル"""
    bl_label = "GPU & Diagnostics"
    bl_idname = "TAREMIN_CLOTH_PT_diagnostics"
    bl_parent_id = "TAREMIN_CLOTH_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Taremin Cloth"
    bl_options = {'DEFAULT_CLOSED'}
    bl_translation_context = i18n.CONTEXT

    @classmethod
    def poll(cls, context):
        return _is_cloth_active_advanced(context)

    def draw(self, context):
        _draw_diagnostics_box(self.layout, context)


class TAREMIN_CLOTH_PT_collider_panel(bpy.types.Panel):
    """コライダー設定パネル"""
    bl_label = "GPU Collider"
    bl_idname = "TAREMIN_CLOTH_PT_collider_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Taremin Cloth"
    bl_order = 2
    bl_translation_context = i18n.CONTEXT

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        if scene:
            row_mode = layout.row(align=True)
            row_mode.prop(scene, "taremin_cloth_ui_mode", expand=True)

        obj = context.active_object

        if not obj:
            layout.label(text=i18n.trans("Please select an object"), icon='INFO')
            return

        # スケール未適用警告 (Scale != 1.0)
        scale = obj.scale
        if abs(scale.x - 1.0) > 1e-3 or abs(scale.y - 1.0) > 1e-3 or abs(scale.z - 1.0) > 1e-3:
            box_warn = layout.box()
            box_warn.alert = True
            box_warn.label(text=f"{i18n.trans('Unapplied Scale:')} ({scale.x:.2f}, {scale.y:.2f}, {scale.z:.2f})", icon='ERROR')
            box_warn.label(text=i18n.trans("Please apply scale to avoid contact detection errors"))
            op_scale = box_warn.operator("object.transform_apply", text=i18n.trans("Apply Scale (Ctrl+A)"), icon='CHECKMARK')
            op_scale.location = False
            op_scale.rotation = False
            op_scale.scale = True

        col_settings = getattr(obj, "taremin_cloth_collider", None)
        if not col_settings:
            return

        col = layout.column(align=True)
        col.prop(col_settings, "is_collider", text=i18n.trans("Enable Collider"), icon='PHYSICS')
        if col_settings.is_collider:
            col.prop(col_settings, "enabled", text=i18n.trans("Collider Active"), icon='PHYSICS')

        if not col_settings.is_collider:
            return

        ui_mode = getattr(scene, "taremin_cloth_ui_mode", "SIMPLE") if scene else "SIMPLE"

        if ui_mode == 'SIMPLE':
            # --- 簡単モード (Simple Mode) ---
            box = layout.box()
            box.label(text=i18n.trans("Collider Setup (Simple)"), icon='PHYSICS')
            b_col = box.column(align=True)

            # 目的別選択と自動判別
            row_p = b_col.row(align=True)
            row_p.prop(col_settings, "collider_purpose", text=i18n.trans("Purpose"))
            row_p.operator("taremin_cloth.auto_detect_collider", text="", icon='FILE_REFRESH')

            # 判定された形状の表示
            type_names = {
                'BONE_SDF': ("Bone SDF (Character Body)", 'ARMATURE_DATA'),
                'MESH_SDF': ("Mesh SDF (Mannequin/Rigid)", 'MESH_DATA'),
                'PLANE': ("Plane (Floor/Ground)", 'MESH_PLANE'),
                'SPHERE': ("Sphere", 'MESH_UVSPHERE'),
                'CAPSULE': ("Capsule", 'MESH_CAPSULE'),
                'MESH': ("Mesh (Simple)", 'MESH_DATA'),
            }
            name_icon = type_names.get(col_settings.collider_type, (col_settings.collider_type, 'PHYSICS'))
            type_label = i18n.trans(name_icon[0])
            b_col.label(text=f"{i18n.trans('Type:')} {type_label}", icon=name_icon[1])

            # 形状に応じた主要パラメータ
            if col_settings.collider_type in {'SPHERE', 'CAPSULE'}:
                b_col.prop(col_settings, "radius", text=i18n.trans("Radius"))
            else:
                b_col.prop(col_settings, "thickness", text=i18n.trans("Thickness"))
            b_col.prop(col_settings, "friction", slider=True)

            # SDF系の場合のワンクリックベイクボタン
            if col_settings.collider_type in {'BONE_SDF', 'MESH_SDF'}:
                b_col.separator()
                b_col.operator("taremin_cloth.rebake_bone_sdf", text=i18n.trans("Bake / Update SDF"), icon='FILE_REFRESH')
        else:
            # --- 詳細モード (Advanced Mode) ---
            box = layout.box()
            row = box.row(align=True)
            col_preset_title = f"{i18n.trans('Collider:')} {col_settings.last_collider_preset}" if col_settings.last_collider_preset else i18n.trans("Collider Preset")
            row.menu("TAREMIN_CLOTH_MT_collider_presets", text=col_preset_title, icon='PHYSICS')
            op_add = row.operator("taremin_cloth.save_preset", text="", icon='ADD')
            if op_add:
                op_add.category = 'collider'
            op_del = row.operator("taremin_cloth.delete_preset", text="", icon='REMOVE')
            if op_del:
                op_del.category = 'collider'

            b_col = box.column(align=True)

            # クイック目的設定・自動判別
            row_p = b_col.row(align=True)
            row_p.prop(col_settings, "collider_purpose", text=i18n.trans("Purpose"))
            row_p.operator("taremin_cloth.auto_detect_collider", text="", icon='FILE_REFRESH')

            b_col.prop(col_settings, "collider_type")
            if col_settings.collider_type in {'SPHERE', 'CAPSULE'}:
                b_col.prop(col_settings, "radius")
            elif col_settings.collider_type == 'MESH':
                b_col.prop(col_settings, "thickness")
                b_col.prop(col_settings, "single_sided")
                if col_settings.single_sided:
                    sub_rec = b_col.column(align=True)
                    sub_rec.prop(col_settings, "enable_single_sided_recovery", text=f"  {i18n.trans('Single-Sided Recovery')}")

                box_opt = b_col.box()
                box_opt.label(text=i18n.trans("Optimization"), icon='PREFERENCES')
                box_opt.prop(col_settings, "enable_cluster_culling")
                if col_settings.enable_cluster_culling:
                    box_opt.prop(col_settings, "sweep_margin_offset")
            elif col_settings.collider_type == 'BONE_SDF':
                arm_mod = None
                if obj and getattr(obj, "type", None) == 'MESH':
                    for mod in getattr(obj, "modifiers", []):
                        if getattr(mod, "type", None) == 'ARMATURE' and getattr(mod, "object", None):
                            arm_mod = mod
                            break
                if not arm_mod:
                    box_warn = b_col.box()
                    box_warn.alert = True
                    box_warn.label(text=i18n.trans("Armature modifier required"), icon='ERROR')
                    box_warn.label(text=i18n.trans("Configure on a skinned character mesh"))
                else:
                    b_col.label(text=f"{i18n.trans('Armature:')} {arm_mod.object.name}", icon='ARMATURE_DATA')

                b_col.prop(col_settings, "sdf_resolution")
                if col_settings.sdf_resolution == 'CUSTOM':
                    b_col.prop(col_settings, "sdf_resolution_custom")

                # 事前サイズ超過警告とテクスチャ情報表示
                if arm_mod and arm_mod.object and getattr(arm_mod.object, "data", None):
                    arm_obj = arm_mod.object
                    n_bones_est = len(arm_obj.data.bones)
                    req_res = int(col_settings.sdf_resolution) if col_settings.sdf_resolution != 'CUSTOM' else col_settings.sdf_resolution_custom
                    from .engine.sdf_baker import compute_3d_atlas_layout
                    n_cols, n_rows, n_layers, safe_res, total_w, total_h, total_d = compute_3d_atlas_layout(n_bones_est, req_res)
                    vram_mb = (total_w * total_h * total_d * 4) / (1024 * 1024)

                    if safe_res < req_res:
                        box_ovf = b_col.box()
                        box_ovf.alert = True
                        box_ovf.label(text=i18n.trans("Warning: Resolution exceeds GPU limit (2048px)"), icon='ERROR')
                        box_ovf.label(text=f"{i18n.trans('Safe resolution for bone count')}({n_bones_est}): <={safe_res}")
                    else:
                        b_col.label(text=f"{i18n.trans('SDF Texture:')} {total_w}x{total_h}x{total_d} ({i18n.trans('Approx.')}{vram_mb:.0f}MB)", icon='INFO')

                b_col.prop(col_settings, "sdf_margin")
                b_col.prop(col_settings, "weight_threshold")
                b_col.prop(col_settings, "blend_k")
                b_col.prop(col_settings, "sdf_update_mode")
                if col_settings.sdf_update_mode == 'DYNAMIC_GPU':
                    b_col.prop(col_settings, "sdf_dynamic_update_interval")
                b_col.prop(col_settings, "thickness")

                # ハイブリッドコライダー（関節部メッシュ補完）
                box_hybrid = b_col.box()
                box_hybrid.prop(col_settings, "enable_joint_mesh", text=i18n.trans("Joint Mesh Hybrid"), icon='MOD_MESHDEFORM')
                if col_settings.enable_joint_mesh:
                    col_h = box_hybrid.column(align=True)
                    col_h.prop(col_settings, "joint_weight_threshold")
                    col_h.prop(col_settings, "joint_rotation_threshold", text=i18n.trans("Activation Angle (°)"))
                    col_h.label(text=i18n.trans("Dynamically mesh only bent joints for speedup"), icon='INFO')

                b_col.prop(col_settings, "sdf_cache_enabled")

                row_cache = b_col.row(align=True)
                row_cache.operator("taremin_cloth.clear_bone_sdf_cache", text=i18n.trans("Clear Cache"), icon='TRASH')
                row_cache.operator("taremin_cloth.rebake_bone_sdf", text=i18n.trans("Rebake SDF"), icon='FILE_REFRESH')
            elif col_settings.collider_type == 'MESH_SDF':
                b_col.prop(col_settings, "mesh_sdf_voxel_size")
                b_col.prop(col_settings, "mesh_sdf_margin")
                b_col.prop(col_settings, "thickness")

                # VRAM上限予算と自動調整設定
                box_mem = b_col.box()
                box_mem.prop(col_settings, "mesh_sdf_max_vram_mb")
                box_mem.prop(col_settings, "mesh_sdf_auto_scale")

                # 推定解像度とVRAM容量の表示
                from .engine.sdf_baker import estimate_mesh_sdf_info
                w_est, h_est, d_est, vram_est, eff_v, is_clamped, raw_vram = estimate_mesh_sdf_info(
                    obj,
                    col_settings.mesh_sdf_voxel_size,
                    col_settings.mesh_sdf_margin,
                    col_settings.thickness,
                    col_settings.mesh_sdf_max_vram_mb,
                    col_settings.mesh_sdf_auto_scale,
                )
                if w_est > 0:
                    if is_clamped:
                        box_mem.label(
                            text=f"{i18n.trans('Limit exceeded')}({col_settings.mesh_sdf_max_vram_mb}MB): {i18n.trans('Estimated:')} {raw_vram:.1f}MB",
                            icon='ERROR',
                        )
                        box_mem.label(
                            text=f"{i18n.trans('Auto-optimized:')} {eff_v*1000:.1f}mm ({w_est}x{h_est}x{d_est}, {i18n.trans('Approx.')}{vram_est:.1f}MB)",
                            icon='CHECKMARK',
                        )
                    else:
                        box_mem.label(text=f"{i18n.trans('Resolution:')} {w_est}x{h_est}x{d_est} ({i18n.trans('Approx.')}{vram_est:.1f}MB)", icon='INFO')

                b_col.prop(col_settings, "mesh_sdf_cache_enabled")

                row_cache = b_col.row(align=True)
                row_cache.operator("taremin_cloth.clear_bone_sdf_cache", text=i18n.trans("Clear Cache"), icon='TRASH')
                row_cache.operator("taremin_cloth.rebake_bone_sdf", text=i18n.trans("Rebake SDF"), icon='FILE_REFRESH')
            b_col.prop(col_settings, "friction")
            b_col.prop(col_settings, "restitution")

            # Collider Animation (変形駆動)
            anim = getattr(col_settings, "anim", None)
            if anim:
                box_anim = layout.box()
                box_anim.label(text=i18n.trans("Collider Animation"), icon='ARMATURE_DATA')
                col_anim = box_anim.column(align=True)
                col_anim.prop(anim, "enabled", text=i18n.trans("Enable Animation"), icon='PLAY')

                if anim.enabled:
                    col_anim.prop(anim, "target_type", text=i18n.trans("Type"))

                    if anim.target_type == 'SHAPE_KEY':
                        if obj.type == 'MESH' and obj.data and obj.data.shape_keys:
                            col_anim.prop_search(anim, "shape_key_name", obj.data.shape_keys, "key_blocks", text=i18n.trans("Shape Key"))
                        else:
                            col_anim.prop(anim, "shape_key_name", text=i18n.trans("Shape Key"))
                        row_vals = col_anim.row(align=True)
                        row_vals.prop(anim, "start_value", text=i18n.trans("Start"))
                        row_vals.prop(anim, "end_value", text=i18n.trans("End"))

                    elif anim.target_type == 'POSE_BLEND':
                        col_anim.prop(anim, "armature_obj", text=i18n.trans("Armature"))

                        # ポーズ記録ボタン
                        row_rec = col_anim.row(align=True)
                        op_rec_start = row_rec.operator("taremin_cloth.record_pose", text=i18n.trans("Rec Start (0.0)"), icon='KEY_HLT')
                        op_rec_start.slot = 'START'
                        op_rec_target = row_rec.operator("taremin_cloth.record_pose", text=i18n.trans("Rec Target (1.0)"), icon='KEY_HLT')
                        op_rec_target.slot = 'TARGET'

                        # プレビュー・レストボタン
                        row_prev = col_anim.row(align=True)
                        op_prev_start = row_prev.operator("taremin_cloth.apply_pose_preview", text=i18n.trans("0.0"), icon='REW')
                        op_prev_start.slot = 'START'
                        op_prev_target = row_prev.operator("taremin_cloth.apply_pose_preview", text=i18n.trans("1.0"), icon='FF')
                        op_prev_target.slot = 'TARGET'
                        op_prev_rest = row_prev.operator("taremin_cloth.apply_pose_preview", text=i18n.trans("Rest"), icon='FILE_REFRESH')
                        op_prev_rest.slot = 'REST'

                    elif anim.target_type == 'ACTION':
                        col_anim.prop(anim, "armature_obj", text=i18n.trans("Armature"))
                        col_anim.prop(anim, "action", text=i18n.trans("Action"))
                        row_f = col_anim.row(align=True)
                        row_f.prop(anim, "frame_start", text=i18n.trans("Start F"))
                        row_f.prop(anim, "frame_end", text=i18n.trans("End F"))

                    # 再生サイクル設定
                    box_cycle = col_anim.box()
                    box_cycle.label(text=i18n.trans("Playback Settings"), icon='TIME')
                    c_col = box_cycle.column(align=True)
                    c_col.prop(anim, "play_mode", text=i18n.trans("Mode"))
                    c_col.prop(anim, "cycle_frames", text=i18n.trans("Cycle Frames"))

                    if anim.play_mode in {'REPEAT', 'PINGPONG'}:
                        row_loop = c_col.row(align=True)
                        row_loop.prop(anim, "infinite_loop", text=i18n.trans("Infinite"))
                        if not anim.infinite_loop:
                            row_loop.prop(anim, "loop_count", text=i18n.trans("Loops"))

                    c_col.prop(anim, "easing", text=i18n.trans("Easing"))
                    c_col.prop(anim, "progress", text=i18n.trans("Progress"), slider=True)


classes = (
    TAREMIN_CLOTH_UL_elastic_groups,
    TAREMIN_CLOTH_PT_objects_panel,
    TAREMIN_CLOTH_PT_main_panel,
    TAREMIN_CLOTH_PT_pinning,
    TAREMIN_CLOTH_PT_fabric,
    TAREMIN_CLOTH_PT_forces,
    TAREMIN_CLOTH_PT_collisions,
    TAREMIN_CLOTH_PT_pattern,
    TAREMIN_CLOTH_PT_quality,
    TAREMIN_CLOTH_PT_topology,
    TAREMIN_CLOTH_PT_interactive_opts,
    TAREMIN_CLOTH_PT_gui_experimental,
    TAREMIN_CLOTH_PT_diagnostics,
    TAREMIN_CLOTH_PT_collider_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
