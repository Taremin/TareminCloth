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
from ..utils import topology


class TAREMIN_CLOTH_OT_create_seam(bpy.types.Operator):
    """選択された2頂点間に縫合エッジ（Loose Edge）を作成する"""
    bl_idname = "taremin_cloth.create_seam"
    bl_label = "Create Seam"
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
    """メッシュのエッジ長スケールに基づき、貫通・破綻を起こさない適正な布の厚みを自動設定する"""
    bl_idname = "taremin_cloth.auto_fit_thickness"
    bl_label = "Auto Fit Thickness"
    bl_description = "メッシュの平均エッジ長を測定し、貫通を防ぐ推奨の厚みを自動算出・適用します"
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
