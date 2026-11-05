# -*- coding: utf-8 -*-
"""Milestone 4: 2D型紙オペレーター (pattern.py)."""

import bpy
from bpy.types import Operator

from .. import i18n
from ..pattern.bridge import load_document_from_scene, save_document_to_scene
from ..pattern.build import build_garment_mesh_in_place
from ..pattern.canvas import PatternCanvasRenderer
from ..pattern.editor import EditorController, EditorEvent
from ..properties import FloatProperty, StringProperty, EnumProperty
from ..utils.drawing import is_interactive_active
from ..utils.pattern_builder import build_garment_from_patterns
from ..utils.pattern_proxy import (
    QUICK_PLACEMENT_PRESETS,
    get_or_create_arrangement_proxy,
    get_quick_placement_transform,
    sync_pattern_to_proxy,
    sync_proxy_to_pattern,
)
from ..utils.pattern_view import (
    is_pattern_view_active,
    restore_3d_viewport,
    setup_2d_pattern_viewport,
)
from ..utils.view3d import tag_redraw_view3d


class TAREMIN_OT_create_pattern_piece(Operator):
    """2D型紙パーツを新規作成するオペレーター"""
    bl_idname = "taremin_cloth.create_pattern_piece"
    bl_label = "New Pattern Piece"
    bl_description = "Create a new 2D pattern piece in current view"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    name: StringProperty(
        name="Piece Name",
        description="Name of the new pattern piece",
        default="Pattern_Piece",
    )
    shape_type: EnumProperty(
        name="Shape Type",
        description="Type of base 2D object for pattern drawing",
        items=[
            ('CURVE', "2D Curve", "Bezier / Poly curve for smooth CAD drafting", 'CURVE_DATA', 0),
            ('MESH', "Plane Mesh", "Flat polygon mesh for box modeling", 'MESH_PLANE', 1),
        ],
        default='CURVE',
    )
    width: FloatProperty(
        name="Width",
        description="Initial width in meters",
        default=0.20,
        min=0.01,
        max=5.0,
        unit='LENGTH',
    )
    height: FloatProperty(
        name="Height",
        description="Initial height in meters",
        default=0.30,
        min=0.01,
        max=5.0,
        unit='LENGTH',
    )

    def execute(self, context):
        w = self.width
        h = self.height

        if self.shape_type == 'CURVE':
            curve_data = bpy.data.curves.new(name=f"{self.name}_Curve", type='CURVE')
            curve_data.dimensions = '2D'
            spline = curve_data.splines.new(type='POLY')
            spline.points.add(3)
            coords = [(0.0, 0.0, 0.0, 1.0), (w, 0.0, 0.0, 1.0), (w, h, 0.0, 1.0), (0.0, h, 0.0, 1.0)]
            for i, co in enumerate(coords):
                spline.points[i].co = co
            spline.use_cyclic_u = True

            obj = bpy.data.objects.new(self.name, curve_data)
            context.collection.objects.link(obj)
        else:
            mesh = bpy.data.meshes.new(name=f"{self.name}_Mesh")
            verts = [(0.0, 0.0, 0.0), (w, 0.0, 0.0), (w, h, 0.0), (0.0, h, 0.0)]
            faces = [(0, 1, 2, 3)]
            mesh.from_pydata(verts, [], faces)
            mesh.update()

            obj = bpy.data.objects.new(self.name, mesh)
            context.collection.objects.link(obj)

        if hasattr(obj, "taremin_pattern"):
            obj.taremin_pattern.is_pattern_piece = True
            obj.taremin_pattern.target_edge_length = 0.020

        # 作成したオブジェクトを選択・アクティブ化
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj

        # 自動的に新規オブジェクト単体のみの2D製図ビュー（ローカルビュー・Top直交）に切り替える
        setup_2d_pattern_viewport(context, target_objects=[obj])

        # 作成した型紙にカメラをフォーカス
        try:
            bpy.ops.view3d.view_selected()
        except Exception:
            pass

        self.report({'INFO'}, i18n.trans("Created pattern piece: %s") % obj.name)
        return {'FINISHED'}


class TAREMIN_OT_toggle_pattern_2d_view(Operator):
    """3Dビューポートの2D型紙視点切替オペレーター"""
    bl_idname = "taremin_cloth.toggle_pattern_2d_view"
    bl_label = "Toggle 2D Pattern View"
    bl_description = "Switch 3D viewport between 2D CAD drafting view and 3D view"
    bl_translation_context = i18n.CONTEXT

    def execute(self, context):
        if is_pattern_view_active(context):
            restore_3d_viewport(context)
            self.report({'INFO'}, i18n.trans("Switched to standard 3D view"))
        else:
            setup_2d_pattern_viewport(context)
            self.report({'INFO'}, i18n.trans("Switched to 2D pattern view (Rotation locked)"))
        return {'FINISHED'}


class TAREMIN_OT_apply_quick_arrangement(Operator):
    """型紙パーツに素体周辺のクイック配置プリセットを適用するオペレーター"""
    bl_idname = "taremin_cloth.apply_quick_arrangement"
    bl_label = "Apply Quick Placement"
    bl_description = "Quickly position and orient pattern piece around body"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    preset: EnumProperty(
        name="Placement Position",
        description="Standard body region for pattern arrangement",
        items=QUICK_PLACEMENT_PRESETS,
        default='CHEST_FRONT',
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bool(obj and hasattr(obj, "taremin_pattern") and obj.taremin_pattern.is_pattern_piece)

    def execute(self, context):
        obj = context.active_object
        if not obj or not hasattr(obj, "taremin_pattern"):
            return {'CANCELLED'}

        offset, euler = get_quick_placement_transform(self.preset, scene=context.scene)
        pat = obj.taremin_pattern
        pat.arrangement_offset = offset
        pat.arrangement_euler = euler

        # 3Dプロキシが存在すればプロキシ側も同期
        sync_pattern_to_proxy(obj)

        self.report({'INFO'}, i18n.trans("Applied placement preset: %s") % self.preset)
        return {'FINISHED'}


class TAREMIN_OT_edit_arrangement_3d(Operator):
    """3D空間で型紙の配置プロキシ（ゴースト）を直感移動・回転するオペレーター"""
    bl_idname = "taremin_cloth.edit_arrangement_3d"
    bl_label = "Edit 3D Arrangement"
    bl_description = "Show interactive 3D proxy object and switch to 3D view for easy G/R placement"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return bool(obj and hasattr(obj, "taremin_pattern") and obj.taremin_pattern.is_pattern_piece)

    def execute(self, context):
        obj = context.active_object
        if not obj or not hasattr(obj, "taremin_pattern"):
            return {'CANCELLED'}

        # 2Dビュー中であれば標準3Dビューに復帰
        if is_pattern_view_active(context):
            restore_3d_viewport(context)

        # 3Dプロキシを取得または生成
        proxy_obj = get_or_create_arrangement_proxy(context, obj)

        # プロキシを選択・アクティブ化
        for o in context.selected_objects:
            o.select_set(False)
        proxy_obj.select_set(True)
        context.view_layer.objects.active = proxy_obj

        # プロキシにフォーカス
        try:
            bpy.ops.view3d.view_selected()
        except Exception:
            pass

        self.report({'INFO'}, i18n.trans("Move or rotate 3D proxy to arrange pattern: %s") % proxy_obj.name)
        return {'FINISHED'}


class TAREMIN_OT_sync_arrangement_proxy(Operator):
    """3D配置プロキシの位置・回転を型紙パラメータに同期するオペレーター"""
    bl_idname = "taremin_cloth.sync_arrangement_proxy"
    bl_label = "Sync Proxy Transforms"
    bl_description = "Sync 3D proxy transforms back to pattern arrangement properties"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj:
            return False
        return "taremin_pattern_source" in obj or (hasattr(obj, "taremin_pattern") and obj.taremin_pattern.is_pattern_piece)

    def execute(self, context):
        obj = context.active_object
        if not obj:
            return {'CANCELLED'}

        if "taremin_pattern_source" in obj:
            sync_proxy_to_pattern(obj, scene=context.scene)
            self.report({'INFO'}, i18n.trans("Synced 3D proxy transform to pattern properties"))
        else:
            sync_pattern_to_proxy(obj)
            self.report({'INFO'}, i18n.trans("Synced pattern properties to 3D proxy"))
        return {'FINISHED'}


class TAREMIN_OT_pattern_editor(Operator):
    """2D型紙・縫合エディタを開くモーダルオペレーター"""
    bl_idname = "taremin_cloth.pattern_editor"
    bl_label = "Open Pattern Editor"
    bl_description = "Open the 2D garment pattern drafting and seam editing canvas"
    bl_translation_context = i18n.CONTEXT

    def invoke(self, context, event):
        if getattr(context.scene, "taremin_pattern_editor_active", False):
            self.report({'WARNING'}, i18n.trans("Pattern editor is already active"))
            return {'CANCELLED'}

        doc = load_document_from_scene(context.scene)
        self.controller = EditorController(doc=doc)
        self.canvas = PatternCanvasRenderer(self.controller)
        self.canvas.register_handler()

        self._pan_dragging = False
        self._last_mouse = (float(event.mouse_region_x), float(event.mouse_region_y))

        context.scene.taremin_pattern_editor_active = True
        context.window_manager.modal_handler_add(self)
        tag_redraw_view3d(context)
        self.report({'INFO'}, i18n.trans("Opened Pattern Editor (Press Esc to close)"))
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        mx = float(event.mouse_region_x)
        my = float(event.mouse_region_y)
        dx = mx - self._last_mouse[0]
        dy = my - self._last_mouse[1]
        self._last_mouse = (mx, my)

        # 中ボタンドラッグによる Pan 移動
        if event.type == 'MIDDLEMOUSE':
            if event.value == 'PRESS':
                self._pan_dragging = True
                return {'RUNNING_MODAL'}
            elif event.value == 'RELEASE':
                self._pan_dragging = False
                return {'RUNNING_MODAL'}

        if self._pan_dragging and event.type == 'MOUSEMOVE':
            self.controller.view.pan_x += dx
            self.controller.view.pan_y += dy
            tag_redraw_view3d(context)
            return {'RUNNING_MODAL'}

        # マウスホイールによるズーム
        if event.type == 'WHEELINMOUSE':
            self.controller.view.zoom_at((mx, my), 1.15)
            tag_redraw_view3d(context)
            return {'RUNNING_MODAL'}
        elif event.type == 'WHEELOUTMOUSE':
            self.controller.view.zoom_at((mx, my), 0.85)
            tag_redraw_view3d(context)
            return {'RUNNING_MODAL'}

        # ツール切替キー
        if event.value == 'PRESS' and not event.ctrl and not event.alt:
            if event.type in ('ONE', 'Q'):
                self.controller.set_active_tool("SELECT")
                tag_redraw_view3d(context)
                return {'RUNNING_MODAL'}
            elif event.type in ('TWO', 'R'):
                self.controller.set_active_tool("RECTANGLE")
                tag_redraw_view3d(context)
                return {'RUNNING_MODAL'}
            elif event.type in ('THREE', 'P'):
                self.controller.set_active_tool("POLYGON")
                tag_redraw_view3d(context)
                return {'RUNNING_MODAL'}
            elif event.type in ('FOUR', 'S'):
                self.controller.set_active_tool("SEW")
                tag_redraw_view3d(context)
                return {'RUNNING_MODAL'}

        # 終了キー (ESC)
        if event.type == 'ESC' and event.value == 'PRESS':
            if self.controller.preview_points or self.controller.preview_seam_candidates:
                tool = self.controller.active_tool
                if tool:
                    tool.cancel(self.controller)
                tag_redraw_view3d(context)
                return {'RUNNING_MODAL'}
            return self._finish(context)

        # イベント配送
        is_press = (event.value == 'PRESS')
        is_release = (event.value == 'RELEASE')
        is_move = (event.type == 'MOUSEMOVE')
        button = "LEFT" if event.type in ('LEFTMOUSE', 'SELECTMOUSE') else "RIGHT" if event.type in ('RIGHTMOUSE', 'ACTIONMOUSE') else ""

        ed_event = EditorEvent(
            screen_pos=(mx, my),
            is_press=is_press,
            is_release=is_release,
            is_move=is_move,
            button=button,
            shift=event.shift,
            ctrl=event.ctrl,
            alt=event.alt,
            key=event.type if event.value == 'PRESS' else "",
        )

        consumed = self.controller.handle_event(ed_event)
        if self.controller.is_dirty:
            tag_redraw_view3d(context)
            self.controller.is_dirty = False

        if consumed:
            return {'RUNNING_MODAL'}

        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            return self._finish(context)

        return {'PASS_THROUGH'}

    def _finish(self, context):
        self.canvas.unregister_handler()
        context.scene.taremin_pattern_editor_active = False
        save_document_to_scene(self.controller.doc, scene=context.scene)
        try:
            build_garment_mesh_in_place(self.controller.doc, scene=context.scene)
        except Exception as e:
            print(f"[Taremin Cloth] Auto build on editor close error: {e}")
        tag_redraw_view3d(context)
        self.report({'INFO'}, i18n.trans("Closed Pattern Editor"))
        return {'FINISHED'}


class TAREMIN_OT_build_garment(Operator):
    """型紙パーツ群から単一3D衣装メッシュを統合生成するオペレーター"""
    bl_idname = "taremin_cloth.build_garment"
    bl_label = "Build 3D Garment"
    bl_description = "Assemble all active 2D pattern pieces into a single 3D cloth simulation mesh"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    target_name: StringProperty(
        name="Target Mesh Name",
        description="Name of the assembled garment mesh object",
        default="Garment_Cloth",
    )

    @classmethod
    def poll(cls, context):
        # シミュレーション実行中はトポロジー変更をロック
        if is_interactive_active():
            return False
        # Document JSON が存在するか、従来の型紙パーツが存在するか
        raw_doc = getattr(context.scene, "taremin_pattern_document", "")
        if raw_doc and raw_doc.strip():
            return True
        for obj in context.scene.objects:
            if hasattr(obj, "taremin_pattern") and obj.taremin_pattern.is_pattern_piece:
                return True
        return False

    def execute(self, context):
        # 1. Document ベースのビルド
        raw_doc = getattr(context.scene, "taremin_pattern_document", "")
        if raw_doc and raw_doc.strip():
            doc = load_document_from_scene(context.scene)
            if doc.pieces:
                try:
                    obj = build_garment_mesh_in_place(
                        doc,
                        target_object_name=self.target_name,
                        scene=context.scene,
                    )
                    if obj:
                        for o in context.selected_objects:
                            o.select_set(False)
                        obj.select_set(True)
                        context.view_layer.objects.active = obj
                        tag_redraw_view3d(context)
                        self.report({'INFO'}, i18n.trans("Successfully built garment mesh: %s") % obj.name)
                        return {'FINISHED'}
                except Exception as e:
                    self.report({'ERROR'}, i18n.trans("Failed to build garment: %s") % str(e))
                    return {'CANCELLED'}

        # 2. 従来の型紙パーツからのフォールバックビルド
        pattern_objects = [
            obj for obj in context.scene.objects
            if hasattr(obj, "taremin_pattern") and obj.taremin_pattern.is_pattern_piece
        ]
        if not pattern_objects:
            self.report({'WARNING'}, i18n.trans("No active pattern pieces found in scene"))
            return {'CANCELLED'}

        # プロキシが存在していれば最新のトランスフォームを同期
        for p in pattern_objects:
            proxy_name = f"{p.name}_Proxy3D"
            proxy_obj = bpy.data.objects.get(proxy_name)
            if proxy_obj:
                sync_proxy_to_pattern(proxy_obj, scene=context.scene)

        try:
            garment_obj = build_garment_from_patterns(
                context.scene,
                pattern_objects=pattern_objects,
                target_name=self.target_name,
            )

            if is_pattern_view_active(context):
                restore_3d_viewport(context)

            for o in context.selected_objects:
                o.select_set(False)
            garment_obj.select_set(True)
            context.view_layer.objects.active = garment_obj

            try:
                bpy.ops.view3d.view_selected()
            except Exception:
                pass

            self.report({'INFO'}, i18n.trans("Successfully built garment mesh: %s") % garment_obj.name)
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, i18n.trans("Failed to build garment: %s") % str(e))
            return {'CANCELLED'}


class TAREMIN_OT_remesh_to_quad(Operator):
    """シミュレーション済み布メッシュをプロダクションQuadトポロジーに変換するオペレーター"""
    bl_idname = "taremin_cloth.remesh_to_quad"
    bl_label = "Remesh to Quad"
    bl_description = "Convert simulated cloth mesh to production-quality quad topology via barycentric inverse mapping"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    target_quad_size: FloatProperty(
        name="Target Quad Size",
        description="Target grid size for production quad elements",
        default=0.010,
        min=0.002,
        max=0.050,
        unit='LENGTH',
    )

    @classmethod
    def poll(cls, context):
        if is_interactive_active():
            return False
        obj = context.active_object
        return bool(obj and obj.type == 'MESH' and hasattr(obj, "taremin_cloth") and obj.taremin_cloth.is_cloth)

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            return {'CANCELLED'}

        try:
            import taremin_cloth_core as core
        except ImportError:
            self.report({'ERROR'}, i18n.trans("Core engine taremin_cloth_core is not available"))
            return {'CANCELLED'}

        if not hasattr(core, "remesh_cloth_to_quad_3d"):
            self.report({'ERROR'}, i18n.trans("Quad remeshing is not supported in current core module"))
            return {'CANCELLED'}

        from ..utils.mesh_extract import extract_mesh_vertices_and_triangles
        import numpy as np

        mesh = obj.data
        positions_3d, tris = extract_mesh_vertices_and_triangles(mesh)
        if tris is None or len(tris) == 0:
            self.report({'WARNING'}, i18n.trans("Mesh contains no triangles to remesh"))
            return {'CANCELLED'}

        # UVレイヤーの取得
        uv_layer = mesh.uv_layers.get("UVMap_Pattern") or mesh.uv_layers.active
        if not uv_layer:
            self.report({'ERROR'}, i18n.trans("No UV map found for 2D pattern inverse mapping"))
            return {'CANCELLED'}

        uvs_2d = np.zeros((len(positions_3d), 2), dtype=np.float32)
        uv_counts = np.zeros(len(positions_3d), dtype=np.int32)
        mesh.calc_loop_triangles()
        for lt in mesh.loop_triangles:
            for vi, li in zip(lt.vertices, lt.loops):
                uv = uv_layer.data[li].uv
                uvs_2d[vi] += np.array([uv[0], uv[1]], dtype=np.float32)
                uv_counts[vi] += 1
        mask = uv_counts > 0
        uvs_2d[mask] /= uv_counts[mask, None]

        # 境界エッジの抽出
        edge_face_counts = {}
        for t in tris:
            for k in range(3):
                e = (min(t[k], t[(k + 1) % 3]), max(t[k], t[(k + 1) % 3]))
                edge_face_counts[e] = edge_face_counts.get(e, 0) + 1
        boundary_edges = [e for e, count in edge_face_counts.items() if count == 1]

        if not boundary_edges:
            self.report({'ERROR'}, i18n.trans("Failed to detect boundary loop"))
            return {'CANCELLED'}

        # 境界ループのチェーン化
        adj = {}
        for u, v in boundary_edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

        start_v = boundary_edges[0][0]
        loop_verts = [start_v]
        curr_v = adj[start_v][0]
        prev_v = start_v
        while curr_v != start_v and len(loop_verts) < len(boundary_edges):
            loop_verts.append(curr_v)
            neighbors = adj[curr_v]
            next_v = neighbors[1] if neighbors[0] == prev_v else neighbors[0]
            prev_v = curr_v
            curr_v = next_v

        boundary_loop_2d = uvs_2d[loop_verts]

        try:
            out = core.remesh_cloth_to_quad_3d(
                sim_positions_3d=positions_3d,
                sim_uvs_2d=uvs_2d,
                sim_triangles=tris.astype(np.uint32),
                boundary_loop_2d=boundary_loop_2d,
                target_quad_size=self.target_quad_size,
                boundary_belt_loops=1,
                smoothing_iterations=10,
            )
        except Exception as e:
            self.report({'ERROR'}, i18n.trans("Quad remesh failed: %s") % str(e))
            return {'CANCELLED'}

        quad_pos = out["positions_3d"]
        quads = out["quads"]
        quad_uvs = out["uvs_2d"]

        # 新規メッシュ・オブジェクト生成
        quad_mesh = bpy.data.meshes.new(f"{obj.name}_Quad_Mesh")
        polys = [list(q) for q in quads]
        quad_mesh.from_pydata(quad_pos, [], polys)
        quad_mesh.update()

        # UVレイヤー設定
        new_uv_layer = quad_mesh.uv_layers.new(name="UVMap_Pattern")
        for poly in quad_mesh.polygons:
            for li, vi in zip(poly.loop_indices, poly.vertices):
                new_uv_layer.data[li].uv = (quad_uvs[vi][0], quad_uvs[vi][1])

        quad_obj = bpy.data.objects.new(f"{obj.name}_Quad", quad_mesh)
        context.collection.objects.link(quad_obj)
        quad_obj.matrix_world = obj.matrix_world.copy()

        for o in context.selected_objects:
            o.select_set(False)
        quad_obj.select_set(True)
        context.view_layer.objects.active = quad_obj

        self.report({'INFO'}, i18n.trans("Created quad mesh: %s") % quad_obj.name)
        return {'FINISHED'}


classes = (
    TAREMIN_OT_pattern_editor,
    TAREMIN_OT_create_pattern_piece,
    TAREMIN_OT_toggle_pattern_2d_view,
    TAREMIN_OT_apply_quick_arrangement,
    TAREMIN_OT_edit_arrangement_3d,
    TAREMIN_OT_sync_arrangement_proxy,
    TAREMIN_OT_build_garment,
    TAREMIN_OT_remesh_to_quad,
)


def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            pass


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
