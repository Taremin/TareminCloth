"""
taremin_cloth メッシュ編集・ツールオペレーター
縫合線（Seam）作成、伸縮グループ（Elastic Band）操作、十字分割（Cross Subdivision）、
歪み対角線分割、Quadトポロジー復元、および自動厚み調整オペレーターを提供する。
"""

import bpy
import numpy as np
from ..engine.cache import (
    cache_rest_positions,
    clear_simulator_for_object,
)
from ..properties import IntProperty, FloatProperty
from ..utils import topology
from .. import i18n


class TAREMIN_CLOTH_OT_create_seam(bpy.types.Operator):
    """選択された2頂点間に縫合エッジ（Loose Edge）を作成する"""
    bl_idname = "taremin_cloth.create_seam"
    bl_label = "Create Seam"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        import bmesh
        obj = context.active_object
        mesh = obj.data
        if obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(mesh)
            selected_verts = [v for v in bm.verts if v.select]
            if len(selected_verts) == 2:
                v0, v1 = selected_verts
                if not any(e for e in v0.link_edges if e.other_vert(v0) == v1):
                    bm.edges.new((v0, v1))
                    bmesh.update_edit_mesh(mesh)
                    self.report({'INFO'}, f"Created seam edge between vert {v0.index} and {v1.index}")
            else:
                self.report({'WARNING'}, "Please select exactly 2 vertices to connect as seam")
        else:
            selected_indices = [v.index for v in mesh.vertices if v.select]
            if len(selected_indices) == 2:
                v0, v1 = selected_indices
                bm = bmesh.new()
                bm.from_mesh(mesh)
                bm.verts.ensure_lookup_table()
                bm.edges.new((bm.verts[v0], bm.verts[v1]))
                bm.to_mesh(mesh)
                bm.free()
                mesh.update()
                self.report({'INFO'}, f"Created seam edge between vert {v0} and {v1}")
            else:
                self.report({'WARNING'}, "Please select exactly 2 vertices to connect as seam")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_add_elastic_group(bpy.types.Operator):
    """選択された辺から新規伸縮グループ（ゴム紐）を作成する"""
    bl_idname = "taremin_cloth.add_elastic_group"
    bl_label = "Add Elastic Group"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        import bmesh
        obj = context.active_object
        settings = obj.taremin_cloth

        # 選択中のエッジインデックスを抽出
        selected_edges = []
        if obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(obj.data)
            selected_edges = [e.index for e in bm.edges if e.select]
        else:
            selected_edges = [e.index for e in obj.data.edges if e.select]

        if not selected_edges:
            self.report({'WARNING'}, "辺（エッジ）を選択してください（Alt+クリックでループ選択等）")
            return {'CANCELLED'}

        group = settings.elastic_groups.add()
        group.name = f"Elastic {len(settings.elastic_groups)}"
        group.set_edge_indices(selected_edges)
        settings.active_elastic_group_index = len(settings.elastic_groups) - 1

        self.report({'INFO'}, f"新規伸縮グループ '{group.name}' を作成しました ({len(selected_edges)} 辺)")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_remove_elastic_group(bpy.types.Operator):
    """アクティブな伸縮グループを削除する"""
    bl_idname = "taremin_cloth.remove_elastic_group"
    bl_label = "Remove Elastic Group"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and len(obj.taremin_cloth.elastic_groups) > 0

    def execute(self, context):
        settings = context.active_object.taremin_cloth
        idx = settings.active_elastic_group_index
        if 0 <= idx < len(settings.elastic_groups):
            name = settings.elastic_groups[idx].name
            settings.elastic_groups.remove(idx)
            settings.active_elastic_group_index = max(0, idx - 1)
            self.report({'INFO'}, f"伸縮グループ '{name}' を削除しました")
            return {'FINISHED'}
        return {'CANCELLED'}


class TAREMIN_CLOTH_OT_assign_elastic_edges(bpy.types.Operator):
    """選択された辺をアクティブな伸縮グループに割り当てる（上書き）"""
    bl_idname = "taremin_cloth.assign_elastic_edges"
    bl_label = "Assign Selected Edges"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and len(obj.taremin_cloth.elastic_groups) > 0

    def execute(self, context):
        import bmesh
        obj = context.active_object
        settings = obj.taremin_cloth
        idx = settings.active_elastic_group_index
        if not (0 <= idx < len(settings.elastic_groups)):
            return {'CANCELLED'}

        group = settings.elastic_groups[idx]
        selected_edges = []
        if obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(obj.data)
            selected_edges = [e.index for e in bm.edges if e.select]
        else:
            selected_edges = [e.index for e in obj.data.edges if e.select]

        if not selected_edges:
            self.report({'WARNING'}, "辺を選択してください")
            return {'CANCELLED'}

        group.set_edge_indices(selected_edges)
        self.report({'INFO'}, f"グループ '{group.name}' に {len(selected_edges)} 辺を割り当てました")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_select_elastic_edges(bpy.types.Operator):
    """アクティブな伸縮グループに登録されている辺を3Dビューで選択する"""
    bl_idname = "taremin_cloth.select_elastic_edges"
    bl_label = "Select Group Edges"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and len(obj.taremin_cloth.elastic_groups) > 0

    def execute(self, context):
        import bmesh
        obj = context.active_object
        settings = obj.taremin_cloth
        idx = settings.active_elastic_group_index
        if not (0 <= idx < len(settings.elastic_groups)):
            return {'CANCELLED'}

        group = settings.elastic_groups[idx]
        target_indices = set(group.get_edge_indices())

        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')

        # 頂点/面選択から辺選択モードに切り替え
        context.tool_settings.mesh_select_mode = (False, True, False)
        bm = bmesh.from_edit_mesh(obj.data)
        for e in bm.edges:
            e.select = e.index in target_indices
        bmesh.update_edit_mesh(obj.data)

        self.report({'INFO'}, f"グループ '{group.name}' の {len(target_indices)} 辺を選択しました")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_apply_cross_subdivision(bpy.types.Operator):
    """選択中のメッシュの四角面を十字分割（Poke Faces）する"""
    bl_idname = "taremin_cloth.apply_cross_subdivision"
    bl_label = "Apply Cross Subdivision"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(context.active_object and context.active_object.type == 'MESH')

    def execute(self, context):
        obj = context.active_object
        if topology.is_cross_subdivided(obj):
            self.report({'WARNING'}, "すでに十字分割されています")
            return {'CANCELLED'}

        cache_rest_positions(obj, force=True)
        success = topology.apply_cross_subdivision(obj)
        if success:
            clear_simulator_for_object(obj.name)
            cache_rest_positions(obj, force=True)
            self.report({'INFO'}, "十字分割（Poke faces）を適用しました")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "分割対象の四角面（Quad）が見つかりませんでした")
            return {'CANCELLED'}


class TAREMIN_CLOTH_OT_apply_post_process(bpy.types.Operator):
    """シミュレーション結果に基づいてトポロジー後処理を実行する"""
    bl_idname = "taremin_cloth.apply_post_process"
    bl_label = "Apply Post-Process"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bool(obj and obj.type == 'MESH' and topology.is_cross_subdivided(obj))

    def execute(self, context):
        obj = context.active_object
        settings = getattr(obj, "taremin_cloth", None)
        mode = settings.post_process_mode if settings else 'OPTIMAL_TRI'
        flatness = settings.adaptive_flatness_threshold if settings else 5.0

        clear_simulator_for_object(obj.name)
        success = topology.apply_post_process(obj, mode=mode, flatness_threshold=flatness)
        if success:
            self.report({'INFO'}, f"後処理を適用しました: {mode}")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "後処理に失敗したか、対象データがありません")
            return {'CANCELLED'}


class TAREMIN_CLOTH_OT_apply_dynamic_diagonal(bpy.types.Operator):
    """歪み（Strain）に基づいて四角面をシワの稜線に沿った最適な2つの三角形に分割する"""
    bl_idname = "taremin_cloth.apply_dynamic_diagonal"
    bl_label = "Split by Strain"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(context.active_object and context.active_object.type == 'MESH')

    def execute(self, context):
        obj = context.active_object
        settings = getattr(obj, "taremin_cloth", None)
        preserve_flat = settings.dynamic_preserve_flat if settings else False
        flatness = settings.dynamic_flatness_threshold if settings else 5.0

        clear_simulator_for_object(obj.name)
        success = topology.apply_dynamic_diagonal_triangulation(
            obj,
            preserve_flat=preserve_flat,
            flatness_threshold=flatness
        )
        if success:
            self.report({'INFO'}, "歪み（Strain）に基づいて最適な2三角面に分割しました")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "分割対象の四角面が見つからないか、処理に失敗しました")
            return {'CANCELLED'}


class TAREMIN_CLOTH_OT_restore_quad_topology(bpy.types.Operator):
    """十字分割または動的分割されたメッシュを元の四角面（Quad）に復元する"""
    bl_idname = "taremin_cloth.restore_quad_topology"
    bl_label = "Restore Quad Topology"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bool(obj and obj.type == 'MESH' and (topology.is_cross_subdivided(obj) or "_taremin_cloth_backup_mesh" in obj))

    def execute(self, context):
        obj = context.active_object
        clear_simulator_for_object(obj.name)
        success = topology.restore_original_quads(obj)
        if success:
            self.report({'INFO'}, "四角面（Quad）に復元しました")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "四角面への復元に失敗しました")
            return {'CANCELLED'}


class TAREMIN_CLOTH_OT_auto_fit_thickness(bpy.types.Operator):
    """Automatically calculate and apply recommended cloth thickness based on average edge length"""
    bl_idname = "taremin_cloth.auto_fit_thickness"
    bl_label = "Auto Fit Thickness"
    bl_description = "Measure average edge length and automatically calculate and apply recommended thickness to prevent penetration"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH' or not hasattr(obj, "taremin_cloth"):
            self.report({'WARNING'}, "有効な布メッシュオブジェクトを選択してください")
            return {'CANCELLED'}

        mesh = obj.data
        if len(mesh.edges) == 0:
            self.report({'WARNING'}, "メッシュにエッジが存在しません")
            return {'CANCELLED'}

        # ワールドスケールを考慮した平均エッジ長を計算
        mat = obj.matrix_world
        edge_lengths = []
        for e in mesh.edges:
            v0 = mat @ mesh.vertices[e.vertices[0]].co
            v1 = mat @ mesh.vertices[e.vertices[1]].co
            edge_lengths.append((v1 - v0).length)

        avg_len = float(np.mean(edge_lengths))
        # 推奨厚み: 平均エッジ長の 4%〜6%（最小 1mm、最大 5cm）
        recommended_thick = max(0.001, min(0.05, avg_len * 0.05))

        obj.taremin_cloth.thickness = recommended_thick
        self.report({'INFO'}, f"推奨厚み {recommended_thick * 1000.0:.1f} mm を設定しました (平均エッジ長: {avg_len * 1000.0:.1f} mm)")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_assign_face_layer(bpy.types.Operator):
    """選択された面に指定レイヤーIDを割り当てる (cloth_layer 属性を作成・更新)"""
    bl_idname = "taremin_cloth.assign_face_layer"
    bl_label = "Assign Face Layer"
    bl_description = "Assign target layer ID to selected faces"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and hasattr(obj, "taremin_cloth")

    def execute(self, context):
        import bmesh
        obj = context.active_object
        settings = obj.taremin_cloth
        mesh = obj.data
        target_layer = int(settings.target_face_layer_id)

        if obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(mesh)
            layer = bm.faces.layers.int.get("cloth_layer")
            if layer is None:
                layer = bm.faces.layers.int.new("cloth_layer")

            count = 0
            for f in bm.faces:
                if f.select:
                    f[layer] = target_layer
                    count += 1

            if count == 0:
                self.report({'WARNING'}, "面が選択されていません。面を選択して実行してください")
                return {'CANCELLED'}

            bmesh.update_edit_mesh(mesh)
            self.report({'INFO'}, f"{count} 面にレイヤー {target_layer} を割り当てました")
        else:
            attr = mesh.attributes.get("cloth_layer")
            if attr is None:
                attr = mesh.attributes.new(name="cloth_layer", type='INT', domain='FACE')

            selected_polys = [p for p in mesh.polygons if p.select]
            if not selected_polys:
                self.report({'WARNING'}, "編集モードで面を選択するか、ポリゴンを選択して実行してください")
                return {'CANCELLED'}

            for p in selected_polys:
                attr.data[p.index].value = target_layer

            mesh.update()
            self.report({'INFO'}, f"{len(selected_polys)} 面にレイヤー {target_layer} を割り当てました")

        return {'FINISHED'}


class TAREMIN_CLOTH_OT_select_face_layer(bpy.types.Operator):
    """指定レイヤーIDを持つ面を選択する"""
    bl_idname = "taremin_cloth.select_face_layer"
    bl_label = "Select Face Layer"
    bl_description = "Select faces that match target layer ID"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and hasattr(obj, "taremin_cloth")

    def execute(self, context):
        import bmesh
        obj = context.active_object
        settings = obj.taremin_cloth
        mesh = obj.data
        target_layer = int(settings.target_face_layer_id)

        if obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(mesh)
            layer = bm.faces.layers.int.get("cloth_layer")
            if layer is None:
                self.report({'WARNING'}, "メッシュに cloth_layer 属性が設定されていません")
                return {'CANCELLED'}

            count = 0
            for f in bm.faces:
                is_target = (f[layer] == target_layer)
                f.select = is_target
                if is_target:
                    count += 1

            bmesh.update_edit_mesh(mesh)
            self.report({'INFO'}, f"レイヤー {target_layer} の面を {count} 個選択しました")
        else:
            attr = mesh.attributes.get("cloth_layer")
            if attr is None:
                self.report({'WARNING'}, "メッシュに cloth_layer 属性が設定されていません")
                return {'CANCELLED'}

            count = 0
            for p in mesh.polygons:
                is_target = (attr.data[p.index].value == target_layer)
                p.select = is_target
                if is_target:
                    count += 1

            mesh.update()
            self.report({'INFO'}, f"レイヤー {target_layer} のポリゴンを {count} 個選択しました")

        return {'FINISHED'}


class TAREMIN_CLOTH_OT_clear_face_layers(bpy.types.Operator):
    """メッシュから cloth_layer 属性を削除する (オブジェクト共通レイヤーに復元)"""
    bl_idname = "taremin_cloth.clear_face_layers"
    bl_label = "Clear Face Layers"
    bl_description = "Clear cloth_layer attribute to use default layer ID"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and hasattr(obj, "taremin_cloth")

    def execute(self, context):
        import bmesh
        obj = context.active_object
        mesh = obj.data

        if "cloth_layer" not in mesh.attributes:
            self.report({'INFO'}, "cloth_layer 属性は設定されていません")
            return {'FINISHED'}

        if obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(mesh)
            layer = bm.faces.layers.int.get("cloth_layer")
            if layer is not None:
                bm.faces.layers.int.remove(layer)
            bmesh.update_edit_mesh(mesh)
        else:
            mesh.attributes.remove(mesh.attributes["cloth_layer"])
            mesh.update()

        self.report({'INFO'}, "Face レイヤー指定をクリアしました (オブジェクト共通 Layer ID を使用します)")
        return {'FINISHED'}



class TAREMIN_CLOTH_OT_add_string_part(bpy.types.Operator):
    """選択された辺から紐（Curve）オブジェクトを自動生成し、布の紐パーツとして登録する"""
    bl_idname = "taremin_cloth.add_string_part"
    bl_label = "Generate Drawstring from Edges"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    tail_vert_count: IntProperty(
        name="Tail Vertices",
        description="Number of vertices to extend outward on both ends for hanging tails",
        default=8,
        min=0,
        max=64,
    )

    string_radius: FloatProperty(
        name="String Radius",
        description="Collision radius and bevel depth of the curve (m)",
        default=0.003,
        min=0.0005,
        max=0.05,
        unit='LENGTH',
    )

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        import bmesh
        from ..utils.curve_extract import build_ordered_path_across_seams
        from ..utils.mesh_extract import extract_cloth_mesh_data
        from ..engine.cache import get_seam_cluster_map

        obj = context.active_object
        settings = getattr(obj, "taremin_cloth", None)
        if not settings:
            return {'CANCELLED'}

        # 1. 選択エッジインデックスの収集
        if obj.mode == 'EDIT':
            obj.update_from_editmode()
            bm = bmesh.from_edit_mesh(obj.data)
            selected_edges = [e.index for e in bm.edges if e.select]
        else:
            selected_edges = [e.index for e in obj.data.edges if e.select]

        all_edges = np.array([[e.vertices[0], e.vertices[1]] for e in obj.data.edges], dtype=np.uint32)

        if not selected_edges:
            self.report({'WARNING'}, "フード等の縁エッジを選択してください（Alt+クリックでループ選択等）")
            return {'CANCELLED'}

        # 2. 共通ユーティリティでメッシュデータおよび縫合クラスタマップを取得（シミュレーション実行前でも有効）
        cloth_data = extract_cloth_mesh_data(obj, settings=settings, apply_world_matrix=False)
        seam_cluster_map = cloth_data.seam_cluster_map or get_seam_cluster_map(obj.name)
        ordered_verts, seam_bridges = build_ordered_path_across_seams(
            all_edges,
            selected_edges,
            seam_cluster_map=seam_cluster_map,
        )

        if len(ordered_verts) < 2:
            self.report({'WARNING'}, "有効な連続境界パスが見つかりませんでした")
            return {'CANCELLED'}

        # 3. 頂点座標の取得 (ローカル座標: extract_cloth_mesh_data の先頭 n_verts 分)
        n_verts = len(obj.data.vertices)
        coords_2d = cloth_data.positions[:n_verts]

        channel_coords = [coords_2d[v].copy() for v in ordered_verts]

        # 4. 両端に垂れ紐（テール）を延ばす
        tail_count = self.tail_vert_count
        left_tail_pts = []
        right_tail_pts = []

        if tail_count > 0:
            # 左端の接線方向 (外向き)
            dir_left = channel_coords[0] - channel_coords[1]
            len_left = float(np.linalg.norm(dir_left))
            if len_left > 1e-5:
                dir_left /= len_left
            else:
                dir_left = np.array([0.0, 0.0, -1.0], dtype=np.float32)

            step_left = max(0.01, len_left)
            for k in range(tail_count, 0, -1):
                left_tail_pts.append(channel_coords[0] + dir_left * (k * step_left))

            # 右端の接線方向 (外向き)
            dir_right = channel_coords[-1] - channel_coords[-2]
            len_right = float(np.linalg.norm(dir_right))
            if len_right > 1e-5:
                dir_right /= len_right
            else:
                dir_right = np.array([0.0, 0.0, -1.0], dtype=np.float32)

            step_right = max(0.01, len_right)
            for k in range(1, tail_count + 1):
                right_tail_pts.append(channel_coords[-1] + dir_right * (k * step_right))

        all_curve_points = left_tail_pts + channel_coords + right_tail_pts
        total_curve_pts = len(all_curve_points)

        # 5. Blender Curve オブジェクトの新規生成
        curve_data = bpy.data.curves.new(name=f"{obj.name}_Drawstring", type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.bevel_depth = self.string_radius
        curve_data.bevel_resolution = 2

        spline = curve_data.splines.new('POLY')
        spline.points.add(total_curve_pts - 1)

        raw_points_4d = np.ones(total_curve_pts * 4, dtype=np.float32)
        for i, pt in enumerate(all_curve_points):
            raw_points_4d[i * 4:(i * 4) + 3] = pt
        spline.points.foreach_set("co", raw_points_4d)

        curve_obj = bpy.data.objects.new(name=f"{obj.name}_Drawstring", object_data=curve_data)
        # 布オブジェクトと同一のワールド行列を設定
        if hasattr(obj, "matrix_world"):
            curve_obj.matrix_world = obj.matrix_world.copy()

        # シーンコレクションへリンク
        col = obj.users_collection[0] if obj.users_collection else context.scene.collection
        col.objects.link(curve_obj)

        # 6. 布オブジェクトの設定へ登録
        part = settings.string_parts.add()
        part.name = f"String {len(settings.string_parts)}"
        part.curve_object = curve_obj
        part.set_guide_vert_indices(ordered_verts)
        part.set_seam_bridge_pairs(seam_bridges)
        part.tail_vert_count_left = tail_count
        part.tail_vert_count_right = tail_count
        part.radius = self.string_radius
        settings.active_string_part_index = len(settings.string_parts) - 1

        # 分割型紙を跨ぐ場合または縫合が有効な場合は、縫合優先モードを自動でONにする
        if seam_bridges or getattr(settings, "enable_sewing", False):
            settings.enable_sewing_priority = True

        self.report({'INFO'}, f"紐パーツ '{part.name}' を作成しました (頂点数: {total_curve_pts}, ガイド頂点: {len(ordered_verts)})")
        return {'FINISHED'}


class TAREMIN_CLOTH_OT_remove_string_part(bpy.types.Operator):
    """アクティブな紐パーツを削除する"""
    bl_idname = "taremin_cloth.remove_string_part"
    bl_label = "Remove String Part"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and len(getattr(obj.taremin_cloth, "string_parts", [])) > 0

    def execute(self, context):
        settings = context.active_object.taremin_cloth
        idx = settings.active_string_part_index
        if 0 <= idx < len(settings.string_parts):
            name = settings.string_parts[idx].name
            settings.string_parts.remove(idx)
            settings.active_string_part_index = max(0, idx - 1)
            self.report({'INFO'}, f"紐パーツ '{name}' を削除しました")
            return {'FINISHED'}
        return {'CANCELLED'}


class TAREMIN_CLOTH_OT_select_string_guide_verts(bpy.types.Operator):
    """登録されたガイド頂点列を選択する"""
    bl_idname = "taremin_cloth.select_string_guide_verts"
    bl_label = "Select Guide Vertices"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and len(getattr(obj.taremin_cloth, "string_parts", [])) > 0

    def execute(self, context):
        import bmesh
        obj = context.active_object
        settings = obj.taremin_cloth
        idx = settings.active_string_part_index
        if idx < 0 or idx >= len(settings.string_parts):
            return {'CANCELLED'}

        part = settings.string_parts[idx]
        guide_verts = set(part.get_guide_vert_indices())
        if not guide_verts:
            self.report({'WARNING'}, "ガイド頂点が登録されていません")
            return {'CANCELLED'}

        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)
        for v in bm.verts:
            if v.index in guide_verts:
                v.select = True
        bmesh.update_edit_mesh(obj.data)

        self.report({'INFO'}, f"{len(guide_verts)} 個のガイド頂点を選択しました")
        return {'FINISHED'}
