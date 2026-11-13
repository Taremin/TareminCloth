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
from ..properties import FloatProperty, StringProperty, EnumProperty, BoolProperty
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
    is_mouse_over_viewport_chrome,
    pan_region_3d_by_pixels,
    stop_animation,
    sync_view2d_from_region_3d,
    tag_redraw_view3d,
    zoom_region_3d,
)

# 現在起動中の型紙エディタモーダルセッション参照（パネル等からの即時同期およびライフサイクル管理用）
_active_pattern_editor_op = None
_active_pattern_editor_canvas = None


def set_active_pattern_editor_op(op) -> None:
    """現在起動中の型紙エディタモーダルオペレーター（またはテスト用モック）を登録する."""
    global _active_pattern_editor_op, _active_pattern_editor_canvas
    _active_pattern_editor_op = op
    try:
        _active_pattern_editor_canvas = getattr(op, "canvas", None) if op is not None else None
    except ReferenceError:
        _active_pattern_editor_canvas = None


def clear_active_pattern_editor_op() -> None:
    """起動中の型紙エディタ参照をクリアする."""
    global _active_pattern_editor_op, _active_pattern_editor_canvas
    _active_pattern_editor_op = None
    _active_pattern_editor_canvas = None


def _is_pattern_editor_op_alive(op) -> bool:
    """渡された型紙エディタオペレータが生存中の有効なセッションかを返す."""
    if op is None:
        return False
    try:
        # MagicMock 等では _session_closed が Truthy なモックになるため厳密に `is True` で判定する
        if getattr(op, "_session_closed", False) is True:
            return False
        # Blender側でモーダルが破棄済みの場合は属性アクセス時に ReferenceError となる
        getattr(op, "controller", None)
        getattr(op, "canvas", None)
        return True
    except ReferenceError:
        return False
    except Exception:
        return False


def is_pattern_editor_active(*, scene=None) -> bool:
    """現在有効な2D型紙エディタセッションが稼働中かどうかを返す."""
    del scene  # シーン引数付き呼び出しの将来互換性を保持
    return _is_pattern_editor_op_alive(_active_pattern_editor_op)


def _sync_pattern_editor_active_flag(*, scene=None, active: bool) -> None:
    """シーン上の taremin_pattern_editor_active フラグをセッション状態と同期する."""
    target_state = bool(active)
    if scene is not None:
        try:
            if hasattr(scene, "taremin_pattern_editor_active"):
                scene.taremin_pattern_editor_active = target_state
        except (ReferenceError, AttributeError):
            pass
    if not target_state:
        scenes = getattr(getattr(bpy, "data", None), "scenes", None)
        if scenes is not None:
            try:
                for sc in scenes:
                    try:
                        if getattr(sc, "taremin_pattern_editor_active", False) is True:
                            sc.taremin_pattern_editor_active = False
                    except (ReferenceError, AttributeError):
                        pass
            except Exception:
                pass


def _resolve_editor_space_and_rv3d(op, *, context=None):
    """復元対象となる SpaceView3D と RegionView3D を安全に解決する."""
    area = None
    if op is not None:
        try:
            saved_area = getattr(op, "_editor_area", None)
            if saved_area is not None and getattr(saved_area, "type", "") == 'VIEW_3D':
                area = saved_area
        except (ReferenceError, AttributeError):
            area = None

    if area is None and context is not None:
        try:
            ctx_area = getattr(context, "area", None)
            if ctx_area is not None and getattr(ctx_area, "type", "") == 'VIEW_3D':
                area = ctx_area
        except (ReferenceError, AttributeError):
            area = None

    space = None
    rv3d = None
    if context is not None:
        try:
            ctx_space = getattr(context, "space_data", None)
            if ctx_space is not None and getattr(ctx_space, "type", "") == 'VIEW_3D':
                if area is None or getattr(area, "spaces", None) is None:
                    space = ctx_space
                else:
                    active_space = getattr(area.spaces, "active", None)
                    space = active_space or ctx_space
        except (ReferenceError, AttributeError):
            pass

    if space is None and area is not None:
        try:
            space = getattr(getattr(area, "spaces", None), "active", None)
        except (ReferenceError, AttributeError):
            space = None

    if space is not None:
        try:
            rv3d = getattr(space, "region_3d", None)
        except (ReferenceError, AttributeError):
            rv3d = None

    if rv3d is None and context is not None:
        try:
            rv3d = getattr(context, "region_data", None)
        except (ReferenceError, AttributeError):
            rv3d = None

    return space, rv3d


def _cleanup_pattern_editor_session(
    op,
    *,
    context=None,
    save_and_build: bool = False,
    reason: str = "",
) -> None:
    """型紙エディタセッションの後始末を一元的にかつ冪等に実行する."""
    del reason  # 将来の診断ログ拡張用に理由識別子を受け取る
    global _active_pattern_editor_op, _active_pattern_editor_canvas

    already_closed = False
    ctrl = None
    canvas = None
    op_scene = None

    if op is not None:
        try:
            already_closed = getattr(op, "_session_closed", False) is True
            op._session_closed = True
        except (ReferenceError, AttributeError):
            already_closed = False

        if not already_closed:
            try:
                ctrl = getattr(op, "controller", None)
            except (ReferenceError, AttributeError):
                ctrl = None
            try:
                canvas = getattr(op, "canvas", None)
            except (ReferenceError, AttributeError):
                canvas = None
            try:
                op_scene = getattr(op, "_session_scene", None)
            except (ReferenceError, AttributeError):
                op_scene = None

            # 1. ツールバー（Tパネル）およびツールヘッダー表示状態の復元
            try:
                space, rv3d = _resolve_editor_space_and_rv3d(op, context=context)
                if space is not None and getattr(space, "type", "") == 'VIEW_3D':
                    if hasattr(op, "_orig_show_toolbar"):
                        space.show_region_toolbar = op._orig_show_toolbar
                    if hasattr(op, "_orig_show_tool_header"):
                        space.show_region_tool_header = op._orig_show_tool_header

                # 2. 3Dビューポートカメラの復元
                if rv3d is not None and hasattr(op, "_orig_view_perspective"):
                    rv3d.view_perspective = op._orig_view_perspective
                    rv3d.view_rotation = op._orig_view_rotation
                    rv3d.view_location = op._orig_view_location
                    rv3d.view_distance = op._orig_view_distance
            except Exception:
                pass

    # 3. 描画ハンドラの解除（op 側が ReferenceError でも退避参照から重複なく確実に解除する）
    fallback_canvas = _active_pattern_editor_canvas
    targets = (canvas,) if fallback_canvas is canvas else (canvas, fallback_canvas)
    for target_canvas in targets:
        if target_canvas is not None:
            try:
                target_canvas.unregister_handler()
            except Exception:
                pass

    # 4. グローバル参照の解放
    if op is None or _active_pattern_editor_op is op or not _is_pattern_editor_op_alive(_active_pattern_editor_op):
        clear_active_pattern_editor_op()
    else:
        _active_pattern_editor_canvas = None

    # 5. シーンフラグの完全リセット
    ctx_scene = getattr(context, "scene", None) if context is not None else None
    _sync_pattern_editor_active_flag(scene=op_scene, active=False)
    _sync_pattern_editor_active_flag(scene=ctx_scene, active=False)

    # 6. 通常終了時のみドキュメント保存および 3D 衣装メッシュの自動ビルドを実行
    if save_and_build and not already_closed and ctrl is not None:
        target_scene = ctx_scene or op_scene
        if target_scene is not None:
            try:
                save_document_to_scene(ctrl.doc, scene=target_scene)
            except Exception as exc:
                logger.warning(f"[Pattern] Auto save on editor close error: {exc}")
            try:
                build_garment_mesh_in_place(ctrl.doc, scene=target_scene)
            except Exception as exc:
                logger.warning(f"[Pattern] Auto build on editor close error: {exc}")

    # 7. 再描画要求
    if context is not None:
        try:
            tag_redraw_view3d(context)
        except Exception:
            pass


def get_active_pattern_editor_controller():
    """現在起動中の型紙エディタの EditorController を返す. 未起動・無効時は None."""
    op = _active_pattern_editor_op
    if op is None:
        return None
    try:
        if getattr(op, "_session_closed", False) is True:
            clear_active_pattern_editor_op()
            return None
        return getattr(op, "controller", None)
    except ReferenceError:
        clear_active_pattern_editor_op()
        return None


def get_active_pattern_canvas():
    """現在起動中の型紙エディタの PatternCanvasRenderer を返す. 未起動・無効時は None."""
    op = _active_pattern_editor_op
    if op is None:
        return None
    try:
        if getattr(op, "_session_closed", False) is True:
            clear_active_pattern_editor_op()
            return None
        return getattr(op, "canvas", None)
    except ReferenceError:
        clear_active_pattern_editor_op()
        return None


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


def _format_pattern_editor_active_diagnostics(context) -> str:
    """パターンエディタ既起動時および不整合検知時の診断文字列を組み立てる（副作用なし）."""
    scene = getattr(context, "scene", None)
    scene_name = getattr(scene, "name", None) if scene is not None else None
    active_flag = getattr(scene, "taremin_pattern_editor_active", None) if scene is not None else None
    area = getattr(context, "area", None)
    area_type = getattr(area, "type", None) if area is not None else None

    try:
        interactive_active = is_interactive_active()
    except Exception as exc:
        interactive_active = f"error({type(exc).__name__})"

    # 診断読み取りで _active_pattern_editor_op をクリアしないよう直接参照する
    op = _active_pattern_editor_op
    if op is None:
        op_state = "op_ref=None"
    else:
        try:
            # Blender側でモーダルが破棄済みの場合は属性アクセス時に ReferenceError となる
            ctrl = getattr(op, "controller", None)
            canvas = getattr(op, "canvas", None)
            closed = getattr(op, "_session_closed", False) is True
            if ctrl is not None:
                ctrl_state = (
                    f"controller(tool={getattr(ctrl, 'active_tool_name', None)!r}, "
                    f"modal_input={getattr(ctrl, 'is_modal_input_active', None)}, "
                    f"editing_field={getattr(ctrl, 'editing_field', None)!r}, "
                    f"picking_bone={getattr(ctrl, 'picking_bone', None)}, "
                    f"sliding_field={getattr(ctrl, 'sliding_field', None)!r}, "
                    f"has_transform={getattr(ctrl, 'transform_session', None) is not None}, "
                    f"request_close={getattr(ctrl, 'request_close', None)})"
                )
            else:
                ctrl_state = "controller=None"

            if canvas is not None:
                canvas_state = (
                    f"canvas(handler_2d={getattr(canvas, '_handler', None) is not None}, "
                    f"handler_3d={getattr(canvas, '_handler_3d', None) is not None})"
                )
            else:
                canvas_state = "canvas=None"

            op_state = f"op_ref=present, op_rna_alive=True, closed={closed}, {ctrl_state}, {canvas_state}"
        except ReferenceError:
            op_state = "op_ref=present, op_rna_alive=False(ReferenceError)"
        except Exception as exc:
            op_state = f"op_ref=present, op_inspect_error={type(exc).__name__}({exc})"

    return (
        f"scene={scene_name!r}, flag={active_flag!r}, area={area_type!r}, "
        f"interactive={interactive_active!r}, {op_state}"
    )


class TAREMIN_OT_pattern_editor(Operator):
    """2D型紙・縫合エディタを開くモーダルオペレーター"""
    bl_idname = "taremin_cloth.pattern_editor"
    bl_label = "Open Pattern Editor"
    bl_description = "Open the 2D garment pattern drafting and seam editing canvas"
    bl_translation_context = i18n.CONTEXT

    def invoke(self, context, event):
        scene = getattr(context, "scene", None)
        scene_flag = bool(getattr(scene, "taremin_pattern_editor_active", False))
        op_ref = _active_pattern_editor_op
        op_alive = _is_pattern_editor_op_alive(op_ref)

        if op_alive:
            try:
                diag = _format_pattern_editor_active_diagnostics(context)
            except Exception as exc:
                diag = f"diagnostics_error={type(exc).__name__}({exc})"
            if not scene_flag:
                # セッション実体が生存しているのにシーンフラグだけ False になった不整合を検出・修復
                logger.warning(
                    f"[Pattern] Detected active pattern editor session with False scene flag (syncing): {diag}"
                )
                _sync_pattern_editor_active_flag(scene=scene, active=True)
            else:
                logger.warning(f"[Pattern] Cannot open pattern editor (already active): {diag}")
            self.report({'WARNING'}, i18n.trans("Pattern editor is already active"))
            return {'CANCELLED'}

        # 生存セッション不在にもかかわらずフラグや参照が残っている不正状態を検出し、記録した上で復旧
        if scene_flag or op_ref is not None or _active_pattern_editor_canvas is not None:
            try:
                diag = _format_pattern_editor_active_diagnostics(context)
            except Exception as exc:
                diag = f"diagnostics_error={type(exc).__name__}({exc})"
            logger.warning(f"[Pattern] Detected stale pattern editor state (cleaning up): {diag}")
            _cleanup_pattern_editor_session(
                op_ref,
                context=context,
                save_and_build=False,
                reason="stale_state_recovery",
            )

        if is_interactive_active():
            self.report({'WARNING'}, i18n.trans("Cannot open pattern editor during interactive simulation"))
            return {'CANCELLED'}

        self._session_closed = False
        self._session_scene = scene
        self._editor_area = getattr(context, "area", None)
        self.controller = None
        self.canvas = None

        try:
            # タイムラインアニメーション再生中の場合は停止
            stop_animation(context)

            # 既存のSpaceView3D設定をバックアップし、被り防止のためTパネル（Toolbar）およびツールヘッダーを非表示化
            space = getattr(context, "space_data", None)
            if space and getattr(space, "type", "") == 'VIEW_3D':
                self._orig_show_toolbar = getattr(space, "show_region_toolbar", False)
                space.show_region_toolbar = False
                self._orig_show_tool_header = getattr(space, "show_region_tool_header", False)
                space.show_region_tool_header = False

            doc = load_document_from_scene(scene)
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
            self.controller.bones = extract_armature_bones_for_canvas(scene)
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

            context.window_manager.modal_handler_add(self)
            set_active_pattern_editor_op(self)
            _sync_pattern_editor_active_flag(scene=scene, active=True)
            tag_redraw_view3d(context)
            self.report({'INFO'}, i18n.trans("Opened Pattern Editor (Press Esc to close)"))
            return {'RUNNING_MODAL'}
        except Exception as exc:
            logger.error(f"[Pattern] Failed to open pattern editor: {exc}", exc_info=True)
            _cleanup_pattern_editor_session(
                self,
                context=context,
                save_and_build=False,
                reason="invoke_error",
            )
            self.report({'ERROR'}, f"Failed to open pattern editor: {exc}")
            return {'CANCELLED'}

    def modal(self, context, event):
        if getattr(self, "_session_closed", False) is True:
            return {'FINISHED'}
        try:
            return self._modal_impl(context, event)
        except Exception as exc:
            logger.error(f"[Pattern] Unhandled exception in pattern editor modal: {exc}", exc_info=True)
            _cleanup_pattern_editor_session(
                self,
                context=context,
                save_and_build=False,
                reason="modal_error",
            )
            self.report({'ERROR'}, f"Pattern editor stopped due to error: {exc}")
            return {'CANCELLED'}

    def _modal_impl(self, context, event):
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

        # Nパネル・ヘッダー等のUIクローム上の操作はエディタのモーダル入力中でない限りBlender側へパススルー
        if (
            not self.controller.is_modal_input_active
            and not self._pan_dragging
            and is_mouse_over_viewport_chrome(
                getattr(context, "area", None),
                mouse_x=float(event.mouse_x),
                mouse_y=float(event.mouse_y),
            )
            and event.type in (
                'LEFTMOUSE',
                'SELECTMOUSE',
                'RIGHTMOUSE',
                'ACTIONMOUSE',
                'MIDDLEMOUSE',
                'WHEELUPMOUSE',
                'WHEELDOWNMOUSE',
                'WHEELINMOUSE',
                'WHEELOUTMOUSE',
                'MOUSEMOVE',
                'INBETWEEN_MOUSEMOVE',
            )
        ):
            return {'PASS_THROUGH'}

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

        # ツール切替キーおよび寸法線トグル (keymap.resolve_tool_hotkey 委譲、数字キーのみ)
        if event.value == 'PRESS' and not event.ctrl and not event.alt and not event.shift:
            if not self.controller.is_modal_input_active:
                tool_name = resolve_tool_hotkey(event.type)
                if tool_name and tool_name in self.controller.tools:
                    self.controller.set_active_tool(tool_name)
                    tag_redraw_view3d(context)
                    return {'RUNNING_MODAL'}
                if event.type == 'L':
                    self.controller.show_lengths = not self.controller.show_lengths
                    tag_redraw_view3d(context)
                    return {'RUNNING_MODAL'}

        # 終了キー (ESC: モーダル操作・ドラッグ・変形中は中断せず controller 側でキャンセルさせる)
        if event.type == 'ESC' and event.value == 'PRESS' and not self.controller.is_modal_input_active:
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
        if event.value == 'PRESS' and not is_mouse_button:
            key = "ESCAPE" if event.type == 'ESC' else event.type
            unicode_char = getattr(event, "unicode", "") or ""
        else:
            key = ""
            unicode_char = ""

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
        _cleanup_pattern_editor_session(
            self,
            context=context,
            save_and_build=True,
            reason="finished",
        )
        self.report({'INFO'}, i18n.trans("Closed Pattern Editor"))
        return {'FINISHED'}

    def cancel(self, context):
        logger.info("[Pattern] Pattern editor modal cancelled by Blender")
        _cleanup_pattern_editor_session(
            self,
            context=context,
            save_and_build=False,
            reason="cancelled",
        )


class TAREMIN_OT_close_pattern_editor(Operator):
    """起動中の2D型紙・縫合エディタを閉じるオペレーター"""
    bl_idname = "taremin_cloth.close_pattern_editor"
    bl_label = "Close Pattern Editor"
    bl_description = "Close the active 2D garment pattern drafting editor"
    bl_translation_context = i18n.CONTEXT

    @classmethod
    def poll(cls, context):
        scene = getattr(context, "scene", None)
        return is_pattern_editor_active(scene=scene) or bool(
            getattr(scene, "taremin_pattern_editor_active", False)
        )

    def execute(self, context):
        op_ref = _active_pattern_editor_op
        if _is_pattern_editor_op_alive(op_ref):
            _cleanup_pattern_editor_session(
                op_ref,
                context=context,
                save_and_build=True,
                reason="close_operator",
            )
            self.report({'INFO'}, i18n.trans("Closed Pattern Editor"))
            return {'FINISHED'}

        try:
            diag = _format_pattern_editor_active_diagnostics(context)
        except Exception as exc:
            diag = f"diagnostics_error={type(exc).__name__}({exc})"
        logger.warning(f"[Pattern] Close requested without active session (cleaning up): {diag}")
        _cleanup_pattern_editor_session(
            op_ref,
            context=context,
            save_and_build=False,
            reason="close_stale",
        )
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


def _extract_quad_remesh_source_data(obj, mesh, *, uv_layer):
    """Blender メッシュから Quad リメッシュ用の三角形・コーナー UV・縫合線・属性を抽出する"""
    import numpy as np
    from ..utils.mesh_extract import (
        extract_mesh_vertices_and_triangles,
        get_cloth_layer_ids,
        get_pin_weights,
    )

    positions_3d, tris = extract_mesh_vertices_and_triangles(mesh)
    if tris is None or len(tris) == 0:
        return None

    mesh.calc_loop_triangles()
    lt_list = mesh.loop_triangles
    n_tris = len(lt_list)

    tri_uvs = np.zeros((n_tris, 3, 2), dtype=np.float32)
    tri_mats = np.zeros(n_tris, dtype=np.int32)
    uv_data = uv_layer.data
    polys = mesh.polygons

    attr_piece = mesh.attributes.get("pattern_piece_index")
    has_face_piece = bool(attr_piece and attr_piece.domain == 'FACE')
    tri_pieces = np.full(n_tris, -1, dtype=np.int32)

    for fi, lt in enumerate(lt_list):
        li0, li1, li2 = lt.loops[0], lt.loops[1], lt.loops[2]
        uv0 = uv_data[li0].uv
        uv1 = uv_data[li1].uv
        uv2 = uv_data[li2].uv
        tri_uvs[fi, 0, 0] = uv0[0]
        tri_uvs[fi, 0, 1] = uv0[1]
        tri_uvs[fi, 1, 0] = uv1[0]
        tri_uvs[fi, 1, 1] = uv1[1]
        tri_uvs[fi, 2, 0] = uv2[0]
        tri_uvs[fi, 2, 1] = uv2[1]

        poly_idx = lt.polygon_index
        tri_mats[fi] = polys[poly_idx].material_index
        if has_face_piece:
            tri_pieces[fi] = int(attr_piece.data[poly_idx].value)

    # 面に含まれない孤立エッジ（縫合線・BONE_WRAP 中継パス）を抽出
    face_edge_keys = set()
    for poly in polys:
        pverts = poly.vertices
        m = len(pverts)
        for k in range(m):
            a = pverts[k]
            b = pverts[(k + 1) % m]
            face_edge_keys.add((a, b) if a < b else (b, a))

    loose_edges = []
    for edge in mesh.edges:
        a, b = edge.vertices[0], edge.vertices[1]
        key = (a, b) if a < b else (b, a)
        if key not in face_edge_keys:
            loose_edges.append((int(a), int(b)))

    n_verts = len(positions_3d)
    settings = getattr(obj, "taremin_cloth", None)
    vertex_layers = get_cloth_layer_ids(obj, settings=settings, n_verts=n_verts, faces_2d=tris)
    if vertex_layers is None:
        vertex_layers = np.zeros(n_verts, dtype=np.int32)
    else:
        vertex_layers = np.asarray(vertex_layers, dtype=np.int32)

    vertex_pins = get_pin_weights(obj, settings=settings, n_verts=n_verts)
    if vertex_pins is None:
        vertex_pins = np.zeros(n_verts, dtype=np.float32)

    # 元 Piece_* 頂点グループ名を引き継ぐための頂点別マップ
    piece_vg_by_vert: dict[int, list[str]] = {}
    piece_vgs = [vg for vg in obj.vertex_groups if vg.name.startswith("Piece_")]
    if piece_vgs:
        vg_idx_to_name = {vg.index: vg.name for vg in piece_vgs}
        for v in mesh.vertices:
            names = [vg_idx_to_name[g.group] for g in v.groups if g.group in vg_idx_to_name and g.weight > 0.0]
            if names:
                piece_vg_by_vert[v.index] = names

    return {
        "positions_3d": positions_3d,
        "tris": tris,
        "tri_uvs": tri_uvs,
        "tri_mats": tri_mats,
        "tri_pieces": tri_pieces if has_face_piece else None,
        "loose_edges": loose_edges,
        "vertex_layers": vertex_layers,
        "vertex_pins": vertex_pins,
        "piece_vg_by_vert": piece_vg_by_vert,
    }


def _apply_merged_quad_mesh_attributes(
    src_obj,
    dst_obj,
    *,
    src_mesh,
    merged,
    scene,
    uv_layer_name: str,
    piece_vg_by_vert: dict[int, list[str]],
) -> None:
    """統合済み Quad メッシュにループ UV・マテリアル・属性・頂点グループ・設定を反映する"""
    import numpy as np

    dst_mesh = dst_obj.data

    # 1. マテリアルスロットとポリゴン別 material_index
    if src_mesh is not None:
        dst_mesh.materials.clear()
        for mat in src_mesh.materials:
            dst_mesh.materials.append(mat)

    n_quads = len(merged.quads)
    n_tris = len(merged.triangles)
    n_faces = len(dst_mesh.polygons)

    if n_faces > 0 and n_faces == (n_quads + n_tris):
        face_mats = np.concatenate([merged.quad_material_indices, merged.tri_material_indices]).astype(np.int32)
        if len(dst_mesh.materials) > 0:
            face_mats = np.clip(face_mats, 0, len(dst_mesh.materials) - 1)
            dst_mesh.polygons.foreach_set("material_index", face_mats)

        # 2. ループ単位 UV 書き込み（UVシーム結合時もコーナーUVを完全維持）
        new_uv_layer = dst_mesh.uv_layers.new(name=uv_layer_name)
        flat_q_uv = merged.quad_corner_uvs.reshape(-1, 2) if n_quads > 0 else np.zeros((0, 2), dtype=np.float32)
        flat_t_uv = merged.tri_corner_uvs.reshape(-1, 2) if n_tris > 0 else np.zeros((0, 2), dtype=np.float32)
        flat_uvs = np.concatenate([flat_q_uv, flat_t_uv], axis=0).astype(np.float32).ravel()
        if len(flat_uvs) == len(new_uv_layer.data) * 2:
            new_uv_layer.data.foreach_set("uv", flat_uvs)
        else:
            all_uv_faces = list(merged.quad_corner_uvs) + list(merged.tri_corner_uvs)
            for poly, f_uvs in zip(dst_mesh.polygons, all_uv_faces):
                for li, uv in zip(poly.loop_indices, f_uvs):
                    new_uv_layer.data[li].uv = (float(uv[0]), float(uv[1]))

        # 3. pattern_piece_index / cloth_layer 面属性
        face_pieces = np.concatenate([merged.quad_piece_indices, merged.tri_piece_indices]).astype(np.int32)
        attr_piece = dst_mesh.attributes.get("pattern_piece_index") or dst_mesh.attributes.new(
            name="pattern_piece_index", type='INT', domain='FACE'
        )
        attr_piece.data.foreach_set("value", face_pieces)

        face_layers = np.concatenate([merged.quad_cloth_layers, merged.tri_cloth_layers]).astype(np.int32)
        attr_layer = dst_mesh.attributes.get("cloth_layer") or dst_mesh.attributes.new(
            name="cloth_layer", type='INT', domain='FACE'
        )
        attr_layer.data.foreach_set("value", face_layers)

    # 4. 既存の Piece_* / Pin 頂点グループを再構築
    for vg in list(dst_obj.vertex_groups):
        if vg.name.startswith("Piece_") or vg.name in ("Cloth_Pin", "Pin"):
            dst_obj.vertex_groups.remove(vg)

    # アイランドごとの代表 Piece_* 名を決定
    island_to_piece_name: dict[int, str] = {}
    for isl in scene.islands:
        name_counts: dict[str, int] = {}
        for orig_vi in isl.local_to_orig_vi:
            for nm in piece_vg_by_vert.get(int(orig_vi), ()):
                name_counts[nm] = name_counts.get(nm, 0) + 1
        if name_counts:
            island_to_piece_name[isl.island_id] = max(name_counts, key=name_counts.get)
        elif isl.dominant_piece_index >= 0:
            island_to_piece_name[isl.island_id] = f"Piece_{isl.dominant_piece_index}"

    piece_name_to_verts: dict[str, list[int]] = {}
    for vi, isl_set in enumerate(merged.vertex_island_ids):
        for isl_id in isl_set:
            p_name = island_to_piece_name.get(isl_id)
            if p_name:
                piece_name_to_verts.setdefault(p_name, []).append(vi)

    for p_name, verts in piece_name_to_verts.items():
        if verts:
            vg = dst_obj.vertex_groups.get(p_name) or dst_obj.vertex_groups.new(name=p_name)
            vg.add(verts, 1.0, 'REPLACE')

    # Pin 頂点グループ引き継ぎ
    pin_weights = merged.vertex_pin_weights
    if len(pin_weights) == len(dst_mesh.vertices) and float(np.max(pin_weights)) > 1e-6:
        pin_vg_name = ""
        if hasattr(src_obj, "taremin_cloth"):
            pin_vg_name = getattr(src_obj.taremin_cloth, "pin_vertex_group", "") or ""
        if not pin_vg_name:
            pin_vg_name = "Cloth_Pin"
        vg_pin = dst_obj.vertex_groups.get(pin_vg_name) or dst_obj.vertex_groups.new(name=pin_vg_name)
        for vi, w in enumerate(pin_weights):
            fw = float(w)
            if fw > 1e-6:
                vg_pin.add([vi], fw, 'REPLACE')


def _copy_taremin_cloth_settings(src_obj, dst_obj) -> None:
    """taremin_cloth プロパティグループの設定値をコピーする"""
    if not hasattr(src_obj, "taremin_cloth") or not hasattr(dst_obj, "taremin_cloth"):
        return
    src_s = src_obj.taremin_cloth
    dst_s = dst_obj.taremin_cloth
    for prop in src_s.bl_rna.properties:
        pid = prop.identifier
        if pid == "rna_type" or prop.is_readonly:
            continue
        try:
            setattr(dst_s, pid, getattr(src_s, pid))
        except Exception:
            pass


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
        max=0.100,
        unit='LENGTH',
    )
    remesh_mode: EnumProperty(
        name="Merge Mode",
        description="How seams and UV islands are handled in the remeshed garment",
        items=[
            (
                'STATIC_MERGED',
                "Static Merged",
                "Merge seam clusters and UV seams into a single welded 3D mesh without sewing edges",
            ),
            (
                'KEEP_PATTERN',
                "Keep Pattern",
                "Keep UV islands separated and rebuild sewing loose edges",
            ),
        ],
        default='STATIC_MERGED',
    )
    keep_structured_grid: BoolProperty(
        name="Structured Rectangular Grid",
        description="Use pure structured quad grid on convex rectangular pattern islands when aspect fits",
        default=True,
    )
    replace_original: BoolProperty(
        name="Replace Original",
        description="Replace the active cloth object mesh in-place instead of creating a new object",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        if is_interactive_active():
            return False
        from ..preferences import is_quad_remesh_enabled
        if not is_quad_remesh_enabled(context):
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

        if not hasattr(core, "remesh_cloth_to_quad_3d_ext"):
            self.report({'ERROR'}, i18n.trans("Quad remeshing is not supported in current core module"))
            return {'CANCELLED'}

        import numpy as np
        from ..utils.quad_remesh_prep import merge_island_outputs, prepare_quad_remesh_islands

        mesh = obj.data
        uv_layer = (
            mesh.uv_layers.get("UVMap")
            or mesh.uv_layers.get("UVMap_Pattern")
            or mesh.uv_layers.active
        )
        if not uv_layer:
            self.report({'ERROR'}, i18n.trans("No UV map found for 2D pattern inverse mapping"))
            return {'CANCELLED'}

        src = _extract_quad_remesh_source_data(obj, mesh, uv_layer=uv_layer)
        if src is None:
            self.report({'WARNING'}, i18n.trans("Mesh contains no triangles to remesh"))
            return {'CANCELLED'}

        try:
            scene = prepare_quad_remesh_islands(
                src["positions_3d"],
                src["tris"],
                triangle_uvs=src["tri_uvs"],
                loose_edges=src["loose_edges"],
                triangle_piece_ids=src["tri_pieces"],
                triangle_material_indices=src["tri_mats"],
                vertex_cloth_layers=src["vertex_layers"],
                vertex_pin_weights=src["vertex_pins"],
            )
            island_outputs = []
            for island in scene.islands:
                out = core.remesh_cloth_to_quad_3d_ext(
                    sim_positions_3d=island.positions_3d,
                    sim_uvs_2d=island.uvs_2d,
                    sim_triangles=island.triangles.astype(np.uint32),
                    boundary_loop_2d=island.boundary_loop_2d,
                    hole_loops_2d=island.hole_loops_2d,
                    target_quad_size=self.target_quad_size,
                    boundary_belt_loops=1,
                    smoothing_iterations=10,
                    keep_structured_when_possible=self.keep_structured_grid,
                    target_size_in_3d=True,
                )
                island_outputs.append(out)

            merged = merge_island_outputs(scene, island_outputs, mode=self.remesh_mode)
        except Exception as e:
            self.report({'ERROR'}, i18n.trans("Quad remesh failed: %s") % str(e))
            return {'CANCELLED'}

        # 新規メッシュ生成（Quads + 残余 Triangles + KeepPattern 用 Loose Edges）
        quad_mesh = bpy.data.meshes.new(f"{obj.name}_Quad_Mesh")
        polys = [list(q) for q in merged.quads.tolist()] + [list(t) for t in merged.triangles.tolist()]
        quad_mesh.from_pydata(merged.positions_3d.tolist(), merged.loose_edges, polys)
        quad_mesh.update()

        if self.replace_original:
            old_mesh = obj.data
            obj.data = quad_mesh
            _apply_merged_quad_mesh_attributes(
                obj,
                obj,
                src_mesh=old_mesh,
                merged=merged,
                scene=scene,
                uv_layer_name=uv_layer.name,
                piece_vg_by_vert=src["piece_vg_by_vert"],
            )
            if hasattr(obj, "taremin_cloth"):
                obj.taremin_cloth.enable_sewing = bool(
                    self.remesh_mode == 'KEEP_PATTERN' and len(merged.loose_edges) > 0
                )
            if old_mesh is not None and old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)
            self.report({'INFO'}, i18n.trans("Created quad mesh: %s") % obj.name)
            return {'FINISHED'}

        quad_obj = bpy.data.objects.new(f"{obj.name}_Quad", quad_mesh)
        if obj.users_collection:
            for col in obj.users_collection:
                col.objects.link(quad_obj)
        else:
            context.collection.objects.link(quad_obj)

        quad_obj.matrix_world = obj.matrix_world.copy()
        _copy_taremin_cloth_settings(obj, quad_obj)
        _apply_merged_quad_mesh_attributes(
            obj,
            quad_obj,
            src_mesh=mesh,
            merged=merged,
            scene=scene,
            uv_layer_name=uv_layer.name,
            piece_vg_by_vert=src["piece_vg_by_vert"],
        )
        if hasattr(quad_obj, "taremin_cloth"):
            quad_obj.taremin_cloth.enable_sewing = bool(
                self.remesh_mode == 'KEEP_PATTERN' and len(merged.loose_edges) > 0
            )

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
        active_ctrl = get_active_pattern_editor_controller()
        doc = active_ctrl.doc if active_ctrl is not None else load_document_from_scene(context.scene)
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
            active_ctrl = get_active_pattern_editor_controller()
            if active_ctrl is not None:
                active_ctrl.replace_document_in_place(doc, description="Import Pattern JSON")
                active_canvas = get_active_pattern_canvas()
                if active_canvas is not None:
                    active_canvas.clear_image_cache()
                save_document_to_scene(active_ctrl.doc, scene=context.scene, push_undo=False)
            else:
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
        import bpy
        aspect = 1.0
        try:
            img = bpy.data.images.load(self.filepath, check_existing=True)
            if hasattr(img, "reload"):
                try:
                    img.reload()
                except Exception:
                    pass
            if img.size[0] > 0 and img.size[1] > 0:
                aspect = float(img.size[1]) / float(img.size[0])
        except Exception:
            pass
        from ..pattern.document import ReferenceImage
        ref = ReferenceImage(
            image_path=self.filepath,
            width=1.0,
            opacity=0.5,
            visible=True,
            aspect_ratio=aspect,
        )
        active_ctrl = get_active_pattern_editor_controller()
        if active_ctrl is not None:
            active_ctrl.set_reference_image(ref, description="Set Reference Image")
            active_canvas = get_active_pattern_canvas()
            if active_canvas is not None:
                active_canvas.clear_image_cache(image_path=self.filepath)
            save_document_to_scene(active_ctrl.doc, scene=context.scene, push_undo=False)
        else:
            doc = load_document_from_scene(context.scene)
            doc.reference_image = ref
            save_document_to_scene(doc, scene=context.scene)
        tag_redraw_view3d(context)
        self.report({'INFO'}, i18n.trans("Loaded reference image: %s") % self.filepath)
        return {'FINISHED'}


class TAREMIN_OT_toggle_reference_image(Operator):
    """下絵画像の表示・非表示を切り替える"""
    bl_idname = "taremin_cloth.toggle_reference_image"
    bl_label = "Toggle Reference Image"
    bl_description = "Show or hide the background reference image on the 2D drafting canvas"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        active_ctrl = get_active_pattern_editor_controller()
        if active_ctrl is not None:
            if not active_ctrl.toggle_reference_image_visible():
                return {'CANCELLED'}
            save_document_to_scene(active_ctrl.doc, scene=context.scene, push_undo=False)
        else:
            doc = load_document_from_scene(context.scene)
            if doc.reference_image is None:
                return {'CANCELLED'}
            doc.reference_image.visible = not doc.reference_image.visible
            save_document_to_scene(doc, scene=context.scene)
        tag_redraw_view3d(context)
        return {'FINISHED'}


class TAREMIN_OT_clear_reference_image(Operator):
    """設定されている下絵画像を削除する"""
    bl_idname = "taremin_cloth.clear_reference_image"
    bl_label = "Clear Reference Image"
    bl_description = "Remove the background reference image from the 2D drafting canvas"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        active_ctrl = get_active_pattern_editor_controller()
        if active_ctrl is not None:
            if not active_ctrl.clear_reference_image():
                return {'CANCELLED'}
            active_canvas = get_active_pattern_canvas()
            if active_canvas is not None:
                active_canvas.clear_image_cache()
            save_document_to_scene(active_ctrl.doc, scene=context.scene, push_undo=False)
        else:
            doc = load_document_from_scene(context.scene)
            if doc.reference_image is None:
                return {'CANCELLED'}
            doc.reference_image = None
            save_document_to_scene(doc, scene=context.scene)
        tag_redraw_view3d(context)
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


def _persistent(func):
    """Blender環境では@persistentを適用し、モックテスト環境では元の関数をそのまま返す."""
    app = getattr(bpy, "app", None)
    handlers = getattr(app, "handlers", None)
    persistent_fn = getattr(handlers, "persistent", None)
    if (
        callable(persistent_fn)
        and "Mock" not in type(persistent_fn).__name__
        and not hasattr(persistent_fn, "_mock_name")
    ):
        return persistent_fn(func)
    return func


@_persistent
def _on_pattern_editor_load_pre(_dummy=None) -> None:
    """ファイル読み込み前に稼働中パターンエディタセッションを安全に解放する."""
    op = _active_pattern_editor_op
    if op is not None or _active_pattern_editor_canvas is not None:
        logger.info("[Pattern] Cleaning up pattern editor session before file load")
        _cleanup_pattern_editor_session(
            op,
            context=getattr(bpy, "context", None),
            save_and_build=False,
            reason="load_pre",
        )


@_persistent
def _on_pattern_editor_load_post(_dummy=None) -> None:
    """ファイル読み込み後に過去の .blend に残った状態フラグや参照を初期化する."""
    stale_scenes = []
    scenes = getattr(getattr(bpy, "data", None), "scenes", None)
    if scenes is not None:
        try:
            for sc in scenes:
                if getattr(sc, "taremin_pattern_editor_active", False) is True:
                    stale_scenes.append(getattr(sc, "name", "?"))
        except Exception:
            pass
    if stale_scenes or _active_pattern_editor_op is not None or _active_pattern_editor_canvas is not None:
        logger.warning(
            f"[Pattern] Resetting stale pattern editor state after file load (scenes={stale_scenes!r})"
        )
    _cleanup_pattern_editor_session(
        _active_pattern_editor_op,
        context=getattr(bpy, "context", None),
        save_and_build=False,
        reason="load_post",
    )


def register_pattern_editor_handlers() -> None:
    """パターンエディタのファイル読込前後ハンドラを登録する."""
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    if handlers:
        for attr_name, fn in (
            ("load_pre", _on_pattern_editor_load_pre),
            ("load_post", _on_pattern_editor_load_post),
        ):
            handler_list = getattr(handlers, attr_name, None)
            if handler_list is not None and hasattr(handler_list, "append"):
                try:
                    if fn not in handler_list:
                        handler_list.append(fn)
                except Exception:
                    pass
    _sync_pattern_editor_active_flag(
        scene=getattr(getattr(bpy, "context", None), "scene", None),
        active=False,
    )


def unregister_pattern_editor_handlers() -> None:
    """パターンエディタのアクティブセッションを終了し、読込前後ハンドラを解除する."""
    _cleanup_pattern_editor_session(
        _active_pattern_editor_op,
        context=getattr(bpy, "context", None),
        save_and_build=False,
        reason="unregister",
    )
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    if handlers:
        for attr_name, fn in (
            ("load_pre", _on_pattern_editor_load_pre),
            ("load_post", _on_pattern_editor_load_post),
        ):
            handler_list = getattr(handlers, attr_name, None)
            if handler_list is not None and hasattr(handler_list, "remove"):
                try:
                    if fn in handler_list:
                        handler_list.remove(fn)
                except Exception:
                    pass


classes = (
    TAREMIN_OT_pattern_editor,
    TAREMIN_OT_close_pattern_editor,
    TAREMIN_OT_export_pattern_json,
    TAREMIN_OT_import_pattern_json,
    TAREMIN_OT_set_reference_image,
    TAREMIN_OT_toggle_reference_image,
    TAREMIN_OT_clear_reference_image,
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
    register_pattern_editor_handlers()


def unregister():
    unregister_pattern_editor_handlers()
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
