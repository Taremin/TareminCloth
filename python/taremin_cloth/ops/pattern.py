# -*- coding: utf-8 -*-
"""Milestone 4: 2D型紙オペレーター (pattern.py)."""

import math
import bpy
from bpy.types import Operator
from mathutils import Euler, Quaternion
try:
    from bpy_extras.io_utils import ExportHelper, ImportHelper
except Exception:
    class ExportHelper:
        filepath: str = ""
    class ImportHelper:
        filepath: str = ""

from .. import i18n
from .wrinkle import _find_target_armature, find_nearest_bone_to_mouse
from ..pattern.bridge import load_document_from_scene, save_document_to_scene
from ..pattern.build import build_garment_mesh_in_place
from ..pattern.canvas import PatternCanvasRenderer
from ..pattern.document import Document
from ..pattern.editor import EditorController, EditorEvent
from ..pattern.keymap import resolve_tool_hotkey
from ..properties import FloatProperty, StringProperty, EnumProperty
from ..utils.drawing import is_interactive_active
from ..utils.logger import logger
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
from ..utils.view3d import (
    extract_armature_bones_for_canvas,
    get_viewport_overlay_offsets,
    pan_region_3d_by_pixels,
    stop_animation,
    sync_view2d_from_region_3d,
    tag_redraw_view3d,
    zoom_region_3d,
)


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

        if is_interactive_active():
            self.report({'WARNING'}, i18n.trans("Cannot open pattern editor during interactive simulation"))
            return {'CANCELLED'}

        # タイムラインアニメーション再生中の場合は停止
        stop_animation(context)

        # 既存のSpaceView3D設定をバックアップし、被り防止のためTパネル（Toolbar）およびツールヘッダーを非表示化
        space = getattr(context, "space_data", None)
        if space and getattr(space, "type", "") == 'VIEW_3D':
            self._orig_show_toolbar = getattr(space, "show_region_toolbar", False)
            space.show_region_toolbar = False
            self._orig_show_tool_header = getattr(space, "show_region_tool_header", False)
            space.show_region_tool_header = False

        doc = load_document_from_scene(context.scene)
        self.controller = EditorController(doc=doc)
        region = getattr(context, "region", None)
        rv3d = getattr(context, "region_data", None)
        if region:
            self.controller.screen_size = (region.width, region.height)

        # 3Dビューポートカメラのバックアップと正面正投影 (Front Ortho) 設定
        if rv3d:
            try:
                self._orig_view_perspective = rv3d.view_perspective
                self._orig_view_rotation = rv3d.view_rotation.copy()
                self._orig_view_location = rv3d.view_location.copy()
                self._orig_view_distance = rv3d.view_distance
                rv3d.view_perspective = 'ORTHO'
                # 正面 (-Y から +Y を見る)
                rv3d.view_rotation = Quaternion((0.7071068, 0.7071068, 0.0, 0.0))
            except Exception:
                pass

        # シーン内アーマチュアからボーン情報を単一ソース (BoneInfo3D) として抽出
        self.armature_obj = _find_target_armature(context)
        self.controller.bones = extract_armature_bones_for_canvas(context.scene)
        self.all_bones_data = [b.to_overlay_tuple() for b in self.controller.bones.values()]

        # 3Dビューポートの投影スケールを2Dキャンバスへ1:1実寸同期
        if region and rv3d:
            sync_view2d_from_region_3d(region, rv3d, view_2d=self.controller.view)

        # 初期オフセットの算出
        lo, ro, to, bo = get_viewport_overlay_offsets(area=getattr(context, "area", None))
        self.controller.left_offset = lo
        self.controller.right_offset = ro
        self.controller.top_offset = to
        self.controller.bottom_offset = bo

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
        region = getattr(context, "region", None)
        rv3d = getattr(context, "region_data", None)
        if region:
            self.controller.screen_size = (region.width, region.height)

        # 各パネル（ヘッダー・ツールヘッダー・T/Nパネル）のオーバーレイ幅・高さを同期してHUDオフセット設定
        lo, ro, to, bo = get_viewport_overlay_offsets(area=getattr(context, "area", None))
        self.controller.left_offset = lo
        self.controller.right_offset = ro
        self.controller.top_offset = to
        self.controller.bottom_offset = bo

        mx = float(event.mouse_region_x)
        my = float(event.mouse_region_y)
        dx = mx - self._last_mouse[0]
        dy = my - self._last_mouse[1]
        self._last_mouse = (mx, my)

        # 中ボタンドラッグによる Pan 移動 (3Dビューポートと1:1連動)
        if event.type == 'MIDDLEMOUSE':
            if event.value == 'PRESS':
                self._pan_dragging = True
                return {'RUNNING_MODAL'}
            elif event.value == 'RELEASE':
                self._pan_dragging = False
                return {'RUNNING_MODAL'}

        if self._pan_dragging and event.type == 'MOUSEMOVE':
            if rv3d and region:
                pan_region_3d_by_pixels(rv3d, dx_px=dx, dy_px=dy, zoom_px_per_m=self.controller.view.zoom)
                sync_view2d_from_region_3d(region, rv3d, view_2d=self.controller.view)
            else:
                self.controller.view.pan_x += dx
                self.controller.view.pan_y += dy
            tag_redraw_view3d(context)
            return {'RUNNING_MODAL'}

        # マウスホイールによるズーム (3Dビューポートと1:1連動)
        if event.type in ('WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'WHEELINMOUSE', 'WHEELOUTMOUSE'):
            factor = 1.15 if event.type in ('WHEELUPMOUSE', 'WHEELINMOUSE') else 0.85
            if rv3d and region:
                zoom_region_3d(rv3d, factor=factor)
                if not sync_view2d_from_region_3d(region, rv3d, view_2d=self.controller.view):
                    self.controller.view.zoom_at((mx, my), factor)
            else:
                self.controller.view.zoom_at((mx, my), factor)
            tag_redraw_view3d(context)
            return {'RUNNING_MODAL'}

        # Pick Bone モード中のマウス移動によるホバーボーン検出（確定・解除・キャンセルは controller.handle_event へ委譲）
        if self.controller.picking_bone and event.type == 'MOUSEMOVE':
            b_name = find_nearest_bone_to_mouse(
                context,
                self.armature_obj,
                (float(event.mouse_x), float(event.mouse_y)),
                all_bones_data=self.all_bones_data,
            )
            if (b_name or "") != self.controller.hovered_bone:
                self.controller.hovered_bone = b_name or ""
                tag_redraw_view3d(context)

        # テンキー視点切り替え (正面: NUMPAD_1, 側面: NUMPAD_3, 上面: NUMPAD_7)
        if self.controller.editing_field is None and event.value == 'PRESS' and rv3d:
            if event.type == 'NUMPAD_1':
                e = Euler((math.radians(90.0), 0.0, math.radians(180.0) if event.ctrl else 0.0))
                rv3d.view_rotation = e.to_quaternion()
                if not event.ctrl and region:
                    sync_view2d_from_region_3d(region, rv3d, view_2d=self.controller.view)
                tag_redraw_view3d(context)
                return {'RUNNING_MODAL'}
            elif event.type == 'NUMPAD_3':
                e = Euler((math.radians(90.0), 0.0, math.radians(-90.0) if event.ctrl else math.radians(90.0)))
                rv3d.view_rotation = e.to_quaternion()
                tag_redraw_view3d(context)
                return {'RUNNING_MODAL'}
            elif event.type == 'NUMPAD_7':
                e = Euler((math.radians(180.0) if event.ctrl else 0.0, 0.0, 0.0))
                rv3d.view_rotation = e.to_quaternion()
                tag_redraw_view3d(context)
                return {'RUNNING_MODAL'}

        # ツール切替キーおよび寸法線トグル (keymap.resolve_tool_hotkey 委譲)
        if event.value == 'PRESS' and not event.ctrl and not event.alt and not event.shift:
            if self.controller.editing_field is None and not self.controller.picking_bone:
                # 縫合反転ショートカット (R / F): 縫合選択中またはSEWツール中はツール切替(RECTANGLE)より反転を優先
                is_flip_key = event.type in ('R', 'F')
                has_seam_context = bool(self.controller.selection.seam_ids) or (self.controller.active_tool_name == "SEW")
                if is_flip_key and has_seam_context:
                    pass  # ツール切替を行わず、後続の controller.handle_event に流す
                else:
                    tool_name = resolve_tool_hotkey(event.type)
                    if tool_name and tool_name in self.controller.tools:
                        self.controller.set_active_tool(tool_name)
                        tag_redraw_view3d(context)
                        return {'RUNNING_MODAL'}
                    if event.type == 'L':
                        self.controller.show_lengths = not self.controller.show_lengths
                        tag_redraw_view3d(context)
                        return {'RUNNING_MODAL'}

        # 終了キー (ESC)
        if event.type == 'ESC' and event.value == 'PRESS' and not self.controller.picking_bone:
            if self.controller.editing_field is not None:
                self.controller.editing_field = None
                self.controller.editing_buffer = ""
                tag_redraw_view3d(context)
                return {'RUNNING_MODAL'}
            if self.controller.preview_points or self.controller.preview_seam_candidates:
                tool = self.controller.active_tool
                if tool:
                    tool.cancel(self.controller)
                tag_redraw_view3d(context)
                return {'RUNNING_MODAL'}
            return self._finish(context)

        # イベント配送
        is_mouse_button = event.type in ('LEFTMOUSE', 'SELECTMOUSE', 'RIGHTMOUSE', 'ACTIONMOUSE', 'MIDDLEMOUSE')
        is_double_click = (event.value == 'DOUBLE_CLICK' and is_mouse_button)
        is_press = (event.value in ('PRESS', 'DOUBLE_CLICK') and is_mouse_button)
        is_release = (event.value == 'RELEASE' and is_mouse_button)
        is_move = (event.type == 'MOUSEMOVE')
        button = "LEFT" if event.type in ('LEFTMOUSE', 'SELECTMOUSE') else "RIGHT" if event.type in ('RIGHTMOUSE', 'ACTIONMOUSE') else "MIDDLE" if event.type == 'MIDDLEMOUSE' else ""
        key = event.type if (event.value == 'PRESS' and not is_mouse_button) else ""
        unicode_char = getattr(event, "unicode", "") or ""

        ed_event = EditorEvent(
            screen_pos=(mx, my),
            is_press=is_press,
            is_release=is_release,
            is_move=is_move,
            is_double_click=is_double_click,
            button=button,
            shift=event.shift,
            ctrl=event.ctrl,
            alt=event.alt,
            key=key,
            unicode=unicode_char,
        )

        if key:
            logger.debug(f"[Pattern] Modal Key: key='{key}', ctrl={event.ctrl}, shift={event.shift}, alt={event.alt}")

        consumed = self.controller.handle_event(ed_event)
        if key:
            logger.debug(f"[Pattern] Modal handle_event returned: consumed={consumed}")

        # Pick Bone モード開始直後の初期ホバーボーン設定
        if self.controller.picking_bone and not self.controller.hovered_bone:
            b_name = find_nearest_bone_to_mouse(
                context,
                self.armature_obj,
                (float(event.mouse_x), float(event.mouse_y)),
                all_bones_data=self.all_bones_data,
            )
            self.controller.hovered_bone = b_name or ""

        # HUDからのビルド要求または終了要求の即時処理
        if self.controller.request_build:
            self.controller.request_build = False
            save_document_to_scene(self.controller.doc, scene=context.scene)
            try:
                build_garment_mesh_in_place(self.controller.doc, scene=context.scene)
                self.report({'INFO'}, i18n.trans("Built 3D garment mesh"))
            except Exception as e:
                self.report({'ERROR'}, f"Build failed: {e}")
            tag_redraw_view3d(context)
            return {'RUNNING_MODAL'}

        if self.controller.request_close:
            self.controller.request_close = False
            return self._finish(context)

        if self.controller.is_dirty:
            tag_redraw_view3d(context)
            self.controller.is_dirty = False

        if consumed:
            return {'RUNNING_MODAL'}

        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            return self._finish(context)

        # 型紙エディタ起動中は、Blender 本体のショートカット（Cのサークル選択、Iのキーフレーム挿入等）の
        # 誤爆を防ぐため、単独キー押下イベントはすべて捕捉してパススルーしない
        if event.value == 'PRESS' and not event.ctrl and not event.alt:
            return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def _finish(self, context):
        # ツールバー（Tパネル）およびツールヘッダーの表示状態を元に戻す
        space = getattr(context, "space_data", None)
        if space and getattr(space, "type", "") == 'VIEW_3D':
            if hasattr(self, "_orig_show_toolbar"):
                space.show_region_toolbar = self._orig_show_toolbar
            if hasattr(self, "_orig_show_tool_header"):
                space.show_region_tool_header = self._orig_show_tool_header

        # 3Dビューポートカメラの復元
        rv3d = getattr(context, "region_data", None)
        if rv3d and hasattr(self, "_orig_view_perspective"):
            try:
                rv3d.view_perspective = self._orig_view_perspective
                rv3d.view_rotation = self._orig_view_rotation
                rv3d.view_location = self._orig_view_location
                rv3d.view_distance = self._orig_view_distance
            except Exception:
                pass

        self.canvas.unregister_handler()
        context.scene.taremin_pattern_editor_active = False
        save_document_to_scene(self.controller.doc, scene=context.scene)
        try:
            build_garment_mesh_in_place(self.controller.doc, scene=context.scene)
        except Exception as e:
            logger.warning(f"[Pattern] Auto build on editor close error: {e}")
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


class TAREMIN_OT_export_pattern_json(Operator, ExportHelper):
    """型紙データを外部JSONファイルにエクスポートする"""
    bl_idname = "taremin_cloth.export_pattern_json"
    bl_label = "Export Pattern JSON"
    bl_description = "Save current 2D pattern drafting document to an external JSON file"
    bl_translation_context = i18n.CONTEXT

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    def execute(self, context):
        doc = load_document_from_scene(context.scene)
        try:
            doc.save_to_file(self.filepath)
            self.report({'INFO'}, i18n.trans("Exported pattern to: %s") % self.filepath)
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, i18n.trans("Failed to export pattern: %s") % str(e))
            return {'CANCELLED'}


class TAREMIN_OT_import_pattern_json(Operator, ImportHelper):
    """外部JSONファイルから型紙データをインポートする"""
    bl_idname = "taremin_cloth.import_pattern_json"
    bl_label = "Import Pattern JSON"
    bl_description = "Load a 2D pattern drafting document from an external JSON file"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    def execute(self, context):
        try:
            from ..pattern.document import Document
            doc = Document.load_from_file(self.filepath)
            save_document_to_scene(doc, scene=context.scene)
            build_garment_mesh_in_place(doc, scene=context.scene)
            tag_redraw_view3d(context)
            self.report({'INFO'}, i18n.trans("Imported pattern from: %s") % self.filepath)
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, i18n.trans("Failed to import pattern: %s") % str(e))
            return {'CANCELLED'}


class TAREMIN_OT_set_reference_image(Operator, ImportHelper):
    """キャンバス背景に下絵画像（参考画像）を設定する"""
    bl_idname = "taremin_cloth.set_reference_image"
    bl_label = "Set Reference Image"
    bl_description = "Load a background reference image for 2D drafting canvas"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".png"
    filter_glob: StringProperty(default="*.png;*.jpg;*.jpeg;*.bmp", options={'HIDDEN'})

    def execute(self, context):
        doc = load_document_from_scene(context.scene)
        import bpy
        aspect = 1.0
        try:
            img = bpy.data.images.load(self.filepath, check_existing=True)
            if img.size[0] > 0 and img.size[1] > 0:
                aspect = float(img.size[1]) / float(img.size[0])
        except Exception:
            pass
        from ..pattern.document import ReferenceImage
        doc.reference_image = ReferenceImage(
            image_path=self.filepath,
            width=1.0,
            opacity=0.5,
            visible=True,
            aspect_ratio=aspect,
        )
        save_document_to_scene(doc, scene=context.scene)
        tag_redraw_view3d(context)
        self.report({'INFO'}, i18n.trans("Loaded reference image: %s") % self.filepath)
        return {'FINISHED'}


class TAREMIN_OT_migrate_curves_to_pattern(Operator):
    """シーン内の既存 2D Curve 型紙オブジェクトを Document に変換統合する"""
    bl_idname = "taremin_cloth.migrate_curves_to_pattern"
    bl_label = "Migrate 2D Curves to Pattern Document"
    bl_description = "Convert legacy 2D Curve pattern objects in the scene to the unified Pattern Document"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from ..pattern.migration import migrate_curve_objects_to_document
        curve_objs = [
            obj for obj in context.scene.objects
            if obj.type == 'CURVE' and hasattr(obj, "taremin_pattern") and obj.taremin_pattern.is_pattern_piece
        ]
        if not curve_objs:
            curve_objs = [obj for obj in context.selected_objects if obj.type == 'CURVE']

        if not curve_objs:
            self.report({'WARNING'}, i18n.trans("No 2D Curve pattern objects found to migrate"))
            return {'CANCELLED'}

        doc = load_document_from_scene(context.scene)
        doc = migrate_curve_objects_to_document(curve_objs, target_doc=doc)
        save_document_to_scene(doc, scene=context.scene)
        build_garment_mesh_in_place(doc, scene=context.scene)
        tag_redraw_view3d(context)
        self.report({'INFO'}, i18n.trans("Migrated %d curve object(s) to pattern document") % len(curve_objs))
        return {'FINISHED'}


classes = (
    TAREMIN_OT_pattern_editor,
    TAREMIN_OT_export_pattern_json,
    TAREMIN_OT_import_pattern_json,
    TAREMIN_OT_set_reference_image,
    TAREMIN_OT_migrate_curves_to_pattern,
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
