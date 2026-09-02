import bpy
from .operators import is_interactive_running
from .utils import topology


class TAREMIN_CLOTH_UL_elastic_groups(bpy.types.UIList):
    """伸縮グループ一覧のUIList"""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.prop(item, "enabled", text="")
            row.prop(item, "name", text="", emboss=False, icon='EDGESEL')
            row.prop(item, "scale", text="Scale", slider=True)
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
    box_diag.label(text="GPU & Diagnostics", icon='PREFERENCES')

    col = box_diag.column(align=True)
    col.prop(prefs, "gpu_backend", text="Backend")
    col.prop(prefs, "gpu_device", text="Device")

    row_status = box_diag.row(align=True)
    active_name = prefs.active_device_name
    active_be = prefs.active_backend_name
    if active_name and active_name != "Unknown":
        row_status.label(text=f"Active: {active_name} ({active_be})", icon='CHECKMARK')
    else:
        row_status.label(text="Active: 未初期化 (Auto)", icon='INFO')

    box_diag.operator("taremin_cloth.apply_gpu_settings", text="Apply GPU Settings", icon='FILE_REFRESH')

    row_log = box_diag.row(align=True)
    row_log.label(text="Log Level:", icon='CONSOLE')
    row_log.prop(prefs, "log_level", text="")


class TAREMIN_CLOTH_PT_objects_panel(bpy.types.Panel):
    """GPU Cloth / GPU Collider が設定されたオブジェクト一覧パネル"""
    bl_label = "Cloth & Collider Objects"
    bl_idname = "TAREMIN_CLOTH_PT_objects_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Taremin Cloth"
    bl_order = 0

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
            if getattr(obj, "taremin_collider", None) and obj.taremin_collider.is_collider
        ]

        # --- シミュレーション構成プリセット & 一括操作 ---
        box_config = layout.box()
        row_cfg = box_config.row(align=True)
        preset_title = scene.taremin_cloth_active_config_preset or "Config Preset"
        row_cfg.menu("TAREMIN_CLOTH_MT_config_presets", text=preset_title, icon='SETTINGS')
        row_cfg.operator("taremin_cloth.save_config_preset", text="", icon='ADD')
        if scene.taremin_cloth_active_config_preset:
            row_cfg.operator("taremin_cloth.delete_config_preset", text="", icon='REMOVE')

        row_batch = box_config.row(align=True)
        op_on = row_batch.operator("taremin_cloth.batch_simulation_state", text="All ON")
        op_on.action = 'ALL_ON'
        op_off = row_batch.operator("taremin_cloth.batch_simulation_state", text="All OFF")
        op_off.action = 'ALL_OFF'
        op_solo = row_batch.operator("taremin_cloth.batch_simulation_state", text="Solo")
        op_solo.action = 'SOLO'

        # --- Cloth Objects セクション ---
        box_cloth = layout.box()
        row_c_hdr = box_cloth.row(align=True)
        row_c_hdr.label(text=f"Cloth Objects ({len(cloth_objs)})", icon='MOD_CLOTH')

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
            box_cloth.label(text="Clothが設定されたオブジェクトはありません", icon='INFO')

        # --- Collider Objects セクション ---
        box_col = layout.box()
        row_col_hdr = box_col.row(align=True)
        row_col_hdr.label(text=f"Collider Objects ({len(collider_objs)})", icon='PHYSICS')

        if collider_objs:
            col_col = box_col.column(align=True)
            for obj in collider_objs:
                row = col_col.row(align=True)
                is_active = (obj == active_obj)
                is_col_enabled = obj.taremin_collider.enabled
                col_type = obj.taremin_collider.collider_type

                # 形状に応じたアイコン決定
                if col_type == 'SPHERE':
                    col_icon = 'MESH_UVSPHERE'
                elif col_type == 'CAPSULE':
                    col_icon = 'MESH_CAPSULE'
                elif col_type == 'PLANE':
                    col_icon = 'MESH_PLANE'
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
                }
                row.label(text=type_labels.get(col_type, col_type))

                # コライダー有効/無効トグル (PHYSICS アイコン)
                row.prop(
                    obj.taremin_collider,
                    "enabled",
                    text="",
                    icon='PHYSICS' if is_col_enabled else 'CHECKBOX_DEHLT',
                )

                # ビューポート可視性トグル
                row.prop(obj, "hide_viewport", text="", emboss=False)
        else:
            box_col.label(text="Colliderが設定されたオブジェクトはありません", icon='INFO')


class TAREMIN_CLOTH_PT_main_panel(bpy.types.Panel):
    """3Dビューポートのサイドバー（Nパネル）に表示されるメインパネル"""
    bl_label = "GPU Cloth"
    bl_idname = "TAREMIN_CLOTH_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Taremin Cloth"
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        scene = context.scene
        has_any_cloth = any(getattr(o, "taremin_cloth", None) and o.taremin_cloth.is_cloth for o in scene.objects) if scene else False

        if not obj or obj.type != 'MESH':
            layout.label(text="メッシュオブジェクトを選択してください", icon='INFO')
            if has_any_cloth:
                box_global = layout.box()
                box_global.label(text="Scene Simulation", icon='PHYSICS')
                row_glob = box_global.row(align=True)
                row_glob.operator("taremin_cloth.reset_all", text="Reset All", icon='RECOVER_LAST')
                op_clr_all = row_glob.operator("taremin_cloth.apply_rest_shape", text="Apply All Shapes", icon='CHECKMARK')
                if op_clr_all:
                    op_clr_all.all_objects = True
            _draw_diagnostics_box(layout, context)
            return

        settings = obj.taremin_cloth
        col = layout.column(align=True)

        if not settings.is_cloth:
            col.operator("taremin_cloth.toggle_cloth", text="Enable Cloth", icon='MOD_CLOTH')
            if has_any_cloth:
                box_global = layout.box()
                box_global.label(text="Scene Simulation", icon='PHYSICS')
                row_glob = box_global.row(align=True)
                row_glob.operator("taremin_cloth.reset_all", text="Reset All", icon='RECOVER_LAST')
                op_clr_all = row_glob.operator("taremin_cloth.apply_rest_shape", text="Apply All Shapes", icon='CHECKMARK')
                if op_clr_all:
                    op_clr_all.all_objects = True
            _draw_diagnostics_box(layout, context)
        else:
            col.operator("taremin_cloth.toggle_cloth", text="Disable Cloth", icon='CANCEL')

            # 個別操作 (Selected Cloth)
            box_selected = layout.box()
            box_selected.label(text=f"Selected: {obj.name}", icon='OBJECT_DATA')
            col_sel = box_selected.column(align=True)
            col_sel.prop(settings, "enabled", text="Simulation Active", icon='PHYSICS')
            if is_interactive_running():
                col_sel.operator("taremin_cloth.interactive", text="Stop Interactive Mode", icon='CANCEL', depress=True)
            else:
                col_sel.operator("taremin_cloth.interactive", text="Interactive Mode (Grab/Drag)", icon='HAND', depress=False)
            row_sel = col_sel.row(align=True)
            row_sel.operator("taremin_cloth.reset_selected", text="Reset Cloth", icon='FILE_REFRESH')
            op_clr = row_sel.operator("taremin_cloth.apply_rest_shape", text="Apply Rest Shape", icon='CHECKMARK')
            if op_clr:
                op_clr.all_objects = False

            # 全体操作 (Scene Simulation)
            box_global = layout.box()
            box_global.label(text="Scene Simulation", icon='PHYSICS')
            row_glob = box_global.row(align=True)
            row_glob.operator("taremin_cloth.reset_all", text="Reset All", icon='RECOVER_LAST')
            op_clr_all = row_glob.operator("taremin_cloth.apply_rest_shape", text="Apply All Shapes", icon='CHECKMARK')
            if op_clr_all:
                op_clr_all.all_objects = True

            # 布素材プリセット (Fabric Presets)
            box_preset = layout.box()
            row = box_preset.row(align=True)
            preset_title = f"Material: {settings.last_fabric_preset}" if settings.last_fabric_preset else "Material Preset"
            row.menu("TAREMIN_CLOTH_MT_fabric_presets", text=preset_title, icon='MATERIAL')
            op_add = row.operator("taremin_cloth.save_preset", text="", icon='ADD')
            if op_add:
                op_add.category = 'fabric'
            op_del = row.operator("taremin_cloth.delete_preset", text="", icon='REMOVE')
            if op_del:
                op_del.category = 'fabric'

            # 剛性 (Stiffness)
            box_stiff = layout.box()
            box_stiff.label(text="Stiffness", icon='PHYSICS')
            s_col = box_stiff.column(align=True)
            s_col.prop(settings, "tension_stiffness")
            s_col.prop(settings, "compression_stiffness")
            s_col.prop(settings, "shear_stiffness")
            s_col.prop(settings, "bending_stiffness")

            # 減衰 (Damping)
            box_damp = layout.box()
            box_damp.label(text="Damping", icon='FORCE_DRAG')
            d_col = box_damp.column(align=True)
            d_col.prop(settings, "air_damping")
            d_col.prop(settings, "tension_damping")
            d_col.prop(settings, "compression_damping")
            d_col.prop(settings, "shear_damping")
            d_col.prop(settings, "bending_damping")

            # シミュレーション設定 (Simulation Settings)
            box_sim = layout.box()
            row = box_sim.row(align=True)
            sim_preset_title = f"Quality: {settings.last_simulation_preset}" if settings.last_simulation_preset else "Quality Preset"
            row.menu("TAREMIN_CLOTH_MT_simulation_presets", text=sim_preset_title, icon='SETTINGS')
            op_add = row.operator("taremin_cloth.save_preset", text="", icon='ADD')
            if op_add:
                op_add.category = 'simulation'
            op_del = row.operator("taremin_cloth.delete_preset", text="", icon='REMOVE')
            if op_del:
                op_del.category = 'simulation'

            sim_col = box_sim.column(align=True)
            sim_col.prop(settings, "gravity")
            sim_col.prop(settings, "substeps")
            sim_col.prop(settings, "solver_iterations")
            sim_col.prop(settings, "enable_adaptive_substep")
            if settings.enable_adaptive_substep:
                row_steps = sim_col.row(align=True)
                row_steps.prop(settings, "min_substeps", text="Min")
                row_steps.prop(settings, "max_substeps", text="Max")

            # パフォーマンス & チューニング設定 (Performance Tuning)
            box_perf = layout.box()
            box_perf.label(text="Performance Tuning", icon='PREFERENCES')
            p_col = box_perf.column(align=True)
            p_col.prop(settings, "solver_mode")
            row_buf = p_col.row(align=True)
            row_buf.prop(settings, "enable_frame_buffering")
            if settings.enable_frame_buffering:
                row_buf.prop(settings, "frame_buffer_size")
            if hasattr(context.scene, "taremin_cloth_fast_playback"):
                p_col.prop(context.scene, "taremin_cloth_fast_playback")

            # コライダー接触設定 (Collider Interaction)
            box_col = layout.box()
            box_col.label(text="Collider Interaction", icon='PHYSICS')
            c_col = box_col.column(align=True)
            c_col.prop(settings, "enable_edge_collision")
            if settings.enable_edge_collision:
                edge_sub = c_col.column(align=True)
                edge_sub.prop(settings, "edge_margin_scale", text="  Margin Scale")
                edge_sub.prop(settings, "edge_margin_offset", text="  Margin Offset")

            # 自己・レイヤー衝突設定 (Self & Layer Collision)
            box_layer = layout.box()
            box_layer.label(text="Self & Layer Collision", icon='RENDERLAYERS')
            l_col = box_layer.column(align=True)
            l_col.prop(settings, "enable_self_collision")
            if settings.enable_self_collision:
                self_sub = l_col.column(align=True)
                self_sub.prop(settings, "self_collision_relief_factor", text="  Relief Factor")
                self_sub.prop(settings, "self_collision_max_displacement_ratio", text="  Max Step Ratio")
                self_sub.prop(settings, "self_collision_max_iterations", text="  Search Limit")
                self_sub.prop(settings, "enable_normal_untangling", text="  Normal Untangling")
            l_col.separator()
            l_col.prop(settings, "layer_id")
            row_thick = l_col.row(align=True)
            row_thick.prop(settings, "thickness")
            row_thick.operator("taremin_cloth.auto_fit_thickness", text="Auto Fit", icon='FIXED_SIZE')

            box_attach = layout.box()
            box_attach.label(text="Attachment & Pinning", icon='PINNED')
            a_col = box_attach.column(align=True)
            a_col.prop_search(settings, "pin_vertex_group", obj, "vertex_groups")
            a_col.prop(settings, "pin_color", text="Pin Color")
            row_pin_opts = a_col.row(align=True)
            row_pin_opts.prop(settings, "pin_overlay_interactive_only", text="Interactive Only")
            row_pin_opts.prop(settings, "overlay_depth_test", text="Depth Test (Z)")
            a_col.separator()
            a_col.prop(settings, "pin_target_object")
            if settings.pin_target_object and settings.pin_target_object.type == 'ARMATURE':
                a_col.prop_search(settings, "pin_target_bone", settings.pin_target_object.data, "bones")

            # インタラクティブシミュレーション設定 (Interactive Simulation)
            box_inter = layout.box()
            box_inter.label(text="Interactive Simulation", icon='PLAY')
            col_inter = box_inter.column(align=True)
            col_inter.prop(settings, "interactive_realtime_sync", text="Real-time Sync")
            if settings.interactive_realtime_sync:
                col_inter.prop(settings, "interactive_max_steps", text="Max Steps / Frame")
            row_fps = col_inter.row(align=True)
            row_fps.prop(settings, "show_fps_overlay", text="Show FPS")
            if settings.show_fps_overlay:
                row_fps.prop(settings, "fps_overlay_position", text="")

            # 伸縮グループ (Elastic Bands / Edge Scaling)
            box_elastic = layout.box()
            box_elastic.label(text="Elastic Bands / Edge Scaling", icon='MOD_SHRINKWRAP')
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
                col_details.prop(active_grp, "name", text="Name")
                col_details.prop(active_grp, "scale", text="Scale (Rest Length)", slider=True)
                col_details.prop(active_grp, "color", text="Line Color")
                n_edges = len(active_grp.get_edge_indices())
                col_details.label(text=f"Registered Edges: {n_edges}", icon='INFO')

            row_elastic_disp = box_elastic.row(align=True)
            row_elastic_disp.prop(settings, "show_elastic_overlay", text="Show Overlay")
            if settings.show_elastic_overlay:
                row_elastic_disp.prop(settings, "elastic_overlay_interactive_only", text="Interactive Only")
                row_elastic_disp.prop(settings, "overlay_depth_test", text="Depth Test (Z)")

            box_sew = layout.box()
            box_sew.label(text="Sewing (Pattern Seaming)", icon='MOD_CLOTH')
            s_col = box_sew.column(align=True)
            s_col.prop(settings, "enable_sewing")
            if settings.enable_sewing:
                s_col.prop(settings, "sewing_shrink_speed")
                s_col.operator("taremin_cloth.create_seam", text="Create Seam Between 2 Verts", icon='EDGESEL')

            # メッシュトポロジー・分割 (Mesh & Topology / Triangulation)
            box_topo = layout.box()
            box_topo.label(text="Topology & Triangulation", icon='MOD_TRIANGULATE')
            t_col = box_topo.column(align=True)
            t_col.prop(settings, "triangulation_mode", text="Mode")

            if settings.triangulation_mode == 'DYNAMIC_DIAGONAL':
                t_col.prop(settings, "dynamic_preserve_flat")
                if settings.dynamic_preserve_flat:
                    t_col.prop(settings, "dynamic_flatness_threshold")
                t_col.prop(settings, "auto_triangulate_on_stop")

                row_topo_ops = t_col.row(align=True)
                row_topo_ops.operator("taremin_cloth.apply_dynamic_diagonal", text="Split by Strain", icon='MOD_TRIANGULATE')
                row_topo_ops.operator("taremin_cloth.restore_quad_topology", text="Restore Quad", icon='RECOVER_LAST')

            elif settings.triangulation_mode == 'CROSS_SUBDIV':
                t_col.prop(settings, "post_process_mode")
                if settings.post_process_mode == 'ADAPTIVE':
                    t_col.prop(settings, "adaptive_flatness_threshold")
                t_col.prop(settings, "auto_post_process")

                row_topo_ops = t_col.row(align=True)
                is_subdivided = topology.is_cross_subdivided(obj)
                if not is_subdivided:
                    row_topo_ops.operator("taremin_cloth.apply_cross_subdivision", text="Subdivide Quads (Poke)", icon='MOD_TRIANGULATE')
                else:
                    row_topo_ops.operator("taremin_cloth.apply_post_process", text="Apply Post-Process", icon='CHECKMARK')
                    row_topo_ops.operator("taremin_cloth.restore_quad_topology", text="Restore Quad", icon='RECOVER_LAST')

            # GPU設定およびログレベル（デバッグ・診断）
            _draw_diagnostics_box(layout, context)


class TAREMIN_CLOTH_PT_collider_panel(bpy.types.Panel):
    """コライダー設定パネル"""
    bl_label = "GPU Collider"
    bl_idname = "TAREMIN_CLOTH_PT_collider_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Taremin Cloth"
    bl_order = 2

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        if not obj:
            layout.label(text="オブジェクトを選択してください", icon='INFO')
            return

        col_settings = getattr(obj, "taremin_collider", None)
        if not col_settings:
            return

        col = layout.column(align=True)
        col.prop(col_settings, "is_collider", text="Enable Collider", icon='PHYSICS')
        if col_settings.is_collider:
            col.prop(col_settings, "enabled", text="Collider Active", icon='PHYSICS')

        if col_settings.is_collider:
            box = layout.box()
            row = box.row(align=True)
            col_preset_title = f"Collider: {col_settings.last_collider_preset}" if col_settings.last_collider_preset else "Collider Preset"
            row.menu("TAREMIN_CLOTH_MT_collider_presets", text=col_preset_title, icon='PHYSICS')
            op_add = row.operator("taremin_cloth.save_preset", text="", icon='ADD')
            if op_add:
                op_add.category = 'collider'
            op_del = row.operator("taremin_cloth.delete_preset", text="", icon='REMOVE')
            if op_del:
                op_del.category = 'collider'

            b_col = box.column(align=True)
            b_col.prop(col_settings, "collider_type")
            if col_settings.collider_type in {'SPHERE', 'CAPSULE'}:
                b_col.prop(col_settings, "radius")
            elif col_settings.collider_type == 'MESH':
                b_col.prop(col_settings, "thickness")
                b_col.prop(col_settings, "single_sided")
            b_col.prop(col_settings, "friction")
            b_col.prop(col_settings, "restitution")

            # Collider Animation (変形駆動)
            anim = getattr(col_settings, "anim", None)
            if anim:
                box_anim = layout.box()
                box_anim.label(text="Collider Animation", icon='ARMATURE_DATA')
                col_anim = box_anim.column(align=True)
                col_anim.prop(anim, "enabled", text="Enable Animation", icon='PLAY')

                if anim.enabled:
                    col_anim.prop(anim, "target_type", text="Type")

                    if anim.target_type == 'SHAPE_KEY':
                        if obj.type == 'MESH' and obj.data and obj.data.shape_keys:
                            col_anim.prop_search(anim, "shape_key_name", obj.data.shape_keys, "key_blocks", text="Shape Key")
                        else:
                            col_anim.prop(anim, "shape_key_name", text="Shape Key")
                        row_vals = col_anim.row(align=True)
                        row_vals.prop(anim, "start_value", text="Start")
                        row_vals.prop(anim, "end_value", text="End")

                    elif anim.target_type == 'POSE_BLEND':
                        col_anim.prop(anim, "armature_obj", text="Armature")

                        # ポーズ記録ボタン
                        row_rec = col_anim.row(align=True)
                        op_rec_start = row_rec.operator("taremin_cloth.record_pose", text="Rec Start (0.0)", icon='KEY_HLT')
                        op_rec_start.slot = 'START'
                        op_rec_target = row_rec.operator("taremin_cloth.record_pose", text="Rec Target (1.0)", icon='KEY_HLT')
                        op_rec_target.slot = 'TARGET'

                        # プレビュー・レストボタン
                        row_prev = col_anim.row(align=True)
                        op_prev_start = row_prev.operator("taremin_cloth.apply_pose_preview", text="0.0", icon='REW')
                        op_prev_start.slot = 'START'
                        op_prev_target = row_prev.operator("taremin_cloth.apply_pose_preview", text="1.0", icon='FF')
                        op_prev_target.slot = 'TARGET'
                        op_prev_rest = row_prev.operator("taremin_cloth.apply_pose_preview", text="Rest", icon='FILE_REFRESH')
                        op_prev_rest.slot = 'REST'

                    elif anim.target_type == 'ACTION':
                        col_anim.prop(anim, "armature_obj", text="Armature")
                        col_anim.prop(anim, "action", text="Action")
                        row_f = col_anim.row(align=True)
                        row_f.prop(anim, "frame_start", text="Start F")
                        row_f.prop(anim, "frame_end", text="End F")

                    # 再生サイクル設定
                    box_cycle = col_anim.box()
                    box_cycle.label(text="Playback Settings", icon='TIME')
                    c_col = box_cycle.column(align=True)
                    c_col.prop(anim, "play_mode", text="Mode")
                    c_col.prop(anim, "cycle_frames", text="Cycle Frames")

                    if anim.play_mode in {'REPEAT', 'PINGPONG'}:
                        row_loop = c_col.row(align=True)
                        row_loop.prop(anim, "infinite_loop", text="Infinite")
                        if not anim.infinite_loop:
                            row_loop.prop(anim, "loop_count", text="Loops")

                    c_col.prop(anim, "easing", text="Easing")
                    c_col.prop(anim, "progress", text="Progress", slider=True)


classes = (
    TAREMIN_CLOTH_UL_elastic_groups,
    TAREMIN_CLOTH_PT_objects_panel,
    TAREMIN_CLOTH_PT_main_panel,
    TAREMIN_CLOTH_PT_collider_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
