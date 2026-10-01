"""
ドレープガイド（シワ造形）操作・アセット管理・3Dビュースライドオペレーター (ops/wrinkle.py)

1. プリセットからシワカーブをボーンに配置 (TAREMIN_CLOTH_OT_add_wrinkle_preset)
2. 3Dビューポート上でボーンに沿ってシワを直感スライド (TAREMIN_CLOTH_OT_slide_wrinkle_curves)
3. 選択カーブを基準ボーン正規化でプリセットJSON保存 (TAREMIN_CLOTH_OT_save_wrinkle_preset)
"""

import functools
import math
from typing import Any, List, Optional, Tuple
import bpy
from bpy.props import (
    BoolProperty as _BoolProperty,
    EnumProperty as _EnumProperty,
    FloatProperty as _FloatProperty,
    StringProperty as _StringProperty,
)
from bpy.types import Operator
import numpy as np

from .. import i18n

def _wrap_prop(prop_func):
    @functools.wraps(prop_func)
    def wrapper(*args, **kwargs):
        if "translation_context" not in kwargs:
            kwargs["translation_context"] = i18n.CONTEXT
        return prop_func(*args, **kwargs)
    return wrapper

FloatProperty = _wrap_prop(_FloatProperty)
BoolProperty = _wrap_prop(_BoolProperty)
EnumProperty = _wrap_prop(_EnumProperty)
StringProperty = _wrap_prop(_StringProperty)
from ..engine.wrinkle_field import WrinkleCurveItem
from ..engine.wrinkle_preset import (
    WrinklePreset,
    get_builtin_presets,
    get_user_preset_files,
    instantiate_preset_on_bone,
    instantiate_preset_on_chain,
    load_preset_from_json,
    normalize_curves_to_preset,
    save_preset_to_json,
)
from ..engine.wrinkle_slide import BoneChain, slide_curves_along_chain
from ..utils.drawing import (
    draw_wrinkle_slide_hud,
    draw_wrinkle_bone_overlay,
    draw_wrinkle_armature_overlay,
    draw_wrinkle_influence_meshes,
    build_wrinkle_influence_meshes,
)
from ..utils.view3d import tag_redraw_view3d
from ..utils.modal_event import PressDragTracker, is_left_release

COLLECTION_NAME = "TareminCloth_Wrinkles"


def _is_wrinkle_field_experimental_enabled(context=None) -> bool:
    """ドレープガイド実験フラグの有効判定（preferencesへ委譲）"""
    try:
        from ..preferences import is_wrinkle_field_enabled
        return bool(is_wrinkle_field_enabled(context))
    except Exception:
        return True


def _require_experimental_enabled(operator) -> bool:
    """実験フラグOFF時に警告を出してTrue（ブロック要）を返す。"""
    try:
        ctx = getattr(operator, "_exec_context", None)
    except Exception:
        ctx = None
    if _is_wrinkle_field_experimental_enabled(ctx):
        return False
    try:
        operator.report({'WARNING'}, i18n.trans("Wrinkle Field is experimental. Enable it in Preferences > Experimental Features."))
    except Exception:
        pass
    return True


def _get_preset_items(self, context):
    """組み込みおよびユーザー保存プリセットのEnumPropertyアイテムリストを返す"""
    items = []
    # 1. 組み込みプリセット
    builtins = get_builtin_presets()
    for key, p in builtins.items():
        items.append((key, f"{p.name} (Built-in)", p.description, 'CURVE_DATA', len(items)))

    # 2. ユーザー保存プリセット
    user_files = get_user_preset_files()
    for p_path in user_files:
        p_data = load_preset_from_json(p_path)
        if p_data:
            key = f"user_{p_data.name}"
            items.append((key, f"{p_data.name} (User)", p_data.description, 'FILE_FOLDER', len(items)))

    if not items:
        items.append(('NONE', "No Presets Available", "No wrinkle presets found", 'ERROR', 0))
    return items


def _find_or_create_collection(context, name: str = COLLECTION_NAME) -> bpy.types.Collection:
    """シワカーブ格納用コレクションを取得または新規作成"""
    col = bpy.data.collections.get(name)
    if not col:
        col = bpy.data.collections.new(name)
        context.scene.collection.children.link(col)
    return col


def _extract_bone_chain_data(armature_obj: bpy.types.Object, start_bone_name: str) -> List[Tuple[str, List[float], List[float], Tuple[float, float]]]:
    """
    アーマチュアから指定ボーンを含むボーンチェーンのワールド幾何データを抽出。
    親ボーンから子ボーンへと辿り、一連のボーンリストを返す。
    """
    if armature_obj.type != 'ARMATURE' or not armature_obj.pose:
        return []

    pose_bones = armature_obj.pose.bones
    target_pb = pose_bones.get(start_bone_name)
    if not target_pb:
        if pose_bones:
            target_pb = pose_bones[0]
        else:
            return []

    # チェーンの先祖（ルート方向）を1〜2段辿る
    chain_pbs = []
    curr = target_pb
    ancestors = []
    while curr.parent and len(ancestors) < 2:
        ancestors.append(curr.parent)
        curr = curr.parent
    chain_pbs.extend(reversed(ancestors))
    chain_pbs.append(target_pb)

    # チェーンの子孫を1〜2段辿る
    curr = target_pb
    while curr.children and len(chain_pbs) < 5:
        # 最初の子を採用
        curr = curr.children[0]
        chain_pbs.append(curr)

    w_mat = armature_obj.matrix_world
    bones_data = []
    for pb in chain_pbs:
        h_w = list((w_mat @ pb.head)[:3])
        t_w = list((w_mat @ pb.tail)[:3])
        # 半径（エンベロープ半径または長さに応じた妥当な半径）
        b = pb.bone
        r_h = float(getattr(b, "head_radius", 0.05))
        r_t = float(getattr(b, "tail_radius", 0.05))
        if r_h <= 1e-4:
            r_h = float(pb.length * 0.15)
        if r_t <= 1e-4:
            r_t = float(pb.length * 0.12)
        bones_data.append((pb.name, h_w, t_w, (r_h, r_t)))

    return bones_data


def _find_target_armature(context) -> Optional[bpy.types.Object]:
    """コンテキストからシワ配置対象のアーマチュアを検索"""
    active_obj = getattr(context, "active_object", None)
    if active_obj and getattr(active_obj, "type", None) == 'ARMATURE':
        return active_obj

    sel = getattr(context, "selected_objects", []) or []
    for obj in sel:
        if getattr(obj, "type", None) == 'ARMATURE':
            return obj
        tc = getattr(obj, "taremin_cloth", None)
        if tc and getattr(tc, "wrinkle_armature", None):
            return tc.wrinkle_armature

    if hasattr(context, "scene") and hasattr(context.scene, "objects"):
        scene_objs = context.scene.objects
        obj_list = scene_objs.values() if hasattr(scene_objs, "values") else scene_objs
        for obj in obj_list:
            if getattr(obj, "type", None) == 'ARMATURE':
                return obj
    return None


def _extract_all_armature_bones(armature_obj: Optional[bpy.types.Object]) -> List[Tuple[str, List[float], List[float], float]]:
    """
    アーマチュアオブジェクトから全ボーンのワールド座標 (name, head_world, tail_world, radius) を抽出する。
    """
    if not armature_obj or armature_obj.type != 'ARMATURE' or not armature_obj.pose or not armature_obj.pose.bones:
        return []

    w_mat = armature_obj.matrix_world
    bones_data = []
    pose_bones = armature_obj.pose.bones
    pb_list = pose_bones.values() if hasattr(pose_bones, "values") else pose_bones
    for pb in pb_list:
        h_w = list((w_mat @ pb.head)[:3])
        t_w = list((w_mat @ pb.tail)[:3])
        b = pb.bone
        r_h = float(getattr(b, "head_radius", 0.05))
        r_t = float(getattr(b, "tail_radius", 0.05))
        r = max(r_h, r_t)
        if r <= 1e-4:
            r = float(max(0.015, pb.length * 0.08))
        bones_data.append((pb.name, h_w, t_w, r))

    return bones_data


def _get_3d_view_window_region(context) -> Tuple[Any, Any]:
    """3DビューポートのWINDOWリージョンとregion_3dを確実に取得する"""
    area = getattr(context, "area", None)
    if area and getattr(area, "type", None) == 'VIEW_3D':
        for r in getattr(area, "regions", []):
            if getattr(r, "type", None) == 'WINDOW':
                r3d = getattr(context, "region_data", None)
                if r3d is None and hasattr(area, "spaces") and area.spaces:
                    r3d = getattr(area.spaces.active, "region_3d", None)
                return r, r3d

    # スクリーン全体から VIEW_3D を検索
    screen = getattr(context, "screen", None)
    if screen:
        for a in getattr(screen, "areas", []):
            if getattr(a, "type", None) == 'VIEW_3D':
                for r in getattr(a, "regions", []):
                    if getattr(r, "type", None) == 'WINDOW':
                        r3d = getattr(a.spaces.active, "region_3d", None) if hasattr(a, "spaces") and a.spaces else None
                        return r, r3d

    return getattr(context, "region", None), getattr(context, "region_data", None)


def find_nearest_bone_to_mouse(
    context,
    armature: Optional[bpy.types.Object],
    mouse_pos_2d: Tuple[float, float],
    all_bones_data: Optional[List[Tuple[str, List[float], List[float], float]]] = None,
) -> Optional[str]:
    """
    3Dビューポート上のマウス座標 (x, y) に最も近いアーマチュアのボーン名を特定する。
    - 各ボーンの (head, tail) を 2D スクリーン座標に投影
    - マウス点と線分 (head_2d, tail_2d) のピクセル最短距離を計算
    """
    region, region_3d = _get_3d_view_window_region(context)

    if not region or not region_3d:
        if all_bones_data:
            return all_bones_data[0][0]
        if armature and armature.pose and armature.pose.bones:
            return armature.pose.bones[0].name
        return None

    try:
        from bpy_extras import view3d_utils
    except ImportError:
        if all_bones_data:
            return all_bones_data[0][0]
        if armature and armature.pose and armature.pose.bones:
            return armature.pose.bones[0].name
        return None

    px, py = float(mouse_pos_2d[0]), float(mouse_pos_2d[1])
    # ウィンドウ絶対座標が渡された場合はリージョン相対座標にオフセット変換
    rx = getattr(region, "x", None)
    ry = getattr(region, "y", None)
    rw = getattr(region, "width", None)
    rh = getattr(region, "height", None)
    if isinstance(rx, (int, float)) and isinstance(ry, (int, float)) and isinstance(rw, (int, float)) and isinstance(rh, (int, float)):
        if px >= rx and px <= rx + rw and py >= ry and py <= ry + rh:
            px -= float(rx)
            py -= float(ry)

    p = np.array([px, py], dtype=np.float32)

    best_bone_name = None
    min_dist_sq = float('inf')

    # all_bones_data が渡されている場合はキャッシュを高速走査
    if all_bones_data:
        for b_name, head_world, tail_world, _ in all_bones_data:
            head_2d = view3d_utils.location_3d_to_region_2d(region, region_3d, head_world)
            tail_2d = view3d_utils.location_3d_to_region_2d(region, region_3d, tail_world)
            if head_2d is None or tail_2d is None:
                continue

            a = np.array([head_2d[0], head_2d[1]], dtype=np.float32)
            b = np.array([tail_2d[0], tail_2d[1]], dtype=np.float32)
            ab = b - a
            ab_len_sq = float(np.dot(ab, ab))

            if ab_len_sq < 1e-4:
                dist_sq = float(np.sum((p - a) ** 2))
            else:
                ap = p - a
                t = float(np.clip(np.dot(ap, ab) / ab_len_sq, 0.0, 1.0))
                proj = a + t * ab
                dist_sq = float(np.sum((p - proj) ** 2))

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_bone_name = b_name

        if best_bone_name:
            return best_bone_name
        return all_bones_data[0][0] if all_bones_data else None

    # フォールバック: armature から走査
    if not armature or armature.type != 'ARMATURE' or not armature.pose or not armature.pose.bones:
        return None

    world_mat = armature.matrix_world
    pose_bones = armature.pose.bones
    pb_list = pose_bones.values() if hasattr(pose_bones, "values") else pose_bones
    for pb in pb_list:
        head_world = world_mat @ pb.head
        tail_world = world_mat @ pb.tail
        head_2d = view3d_utils.location_3d_to_region_2d(region, region_3d, head_world)
        tail_2d = view3d_utils.location_3d_to_region_2d(region, region_3d, tail_world)
        if head_2d is None or tail_2d is None:
            continue

        a = np.array([head_2d[0], head_2d[1]], dtype=np.float32)
        b = np.array([tail_2d[0], tail_2d[1]], dtype=np.float32)
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq < 1e-4:
            dist_sq = float(np.sum((p - a) ** 2))
        else:
            ap = p - a
            t = float(np.clip(np.dot(ap, ab) / ab_len_sq, 0.0, 1.0))
            proj = a + t * ab
            dist_sq = float(np.sum((p - proj) ** 2))

        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_bone_name = pb.name

    return best_bone_name or armature.pose.bones[0].name


def _auto_setup_cloth_wrinkle_settings(context, col: bpy.types.Collection, armature: Optional[bpy.types.Object], bone_name: str):
    """
    対象の布オブジェクトにシワコレクションとドレープガイド設定を自動割り当てする。
    実験フラグOFF時は誤有効化を防ぐため何もしない。
    """
    if not _is_wrinkle_field_experimental_enabled(context):
        return
    target_cloths = []

    act = getattr(context, "active_object", None)
    if act and getattr(act, "type", None) == 'MESH' and hasattr(act, "taremin_cloth"):
        target_cloths.append(act)

    sel = getattr(context, "selected_objects", []) or []
    for obj in sel:
        if getattr(obj, "type", None) == 'MESH' and hasattr(obj, "taremin_cloth") and obj not in target_cloths:
            target_cloths.append(obj)

    if not target_cloths and hasattr(context, "scene") and hasattr(context.scene, "objects"):
        scene_objs = context.scene.objects
        obj_list = scene_objs.values() if hasattr(scene_objs, "values") else scene_objs
        for obj in obj_list:
            if getattr(obj, "type", None) == 'MESH' and hasattr(obj, "taremin_cloth"):
                tc = getattr(obj, "taremin_cloth", None)
                if tc and getattr(tc, "is_cloth", False):
                    target_cloths.append(obj)

    for cloth in target_cloths:
        tc = cloth.taremin_cloth
        if getattr(tc, "wrinkle_collection", None) is None:
            tc.wrinkle_collection = col
        if not getattr(tc, "use_wrinkle_field", False):
            tc.use_wrinkle_field = True
        if getattr(tc, "wrinkle_armature", None) is None and armature:
            tc.wrinkle_armature = armature
        if bone_name and not getattr(tc, "wrinkle_bone_name", ""):
            tc.wrinkle_bone_name = bone_name


class TAREMIN_CLOTH_OT_add_wrinkle_preset(Operator):
    """選択ボーンに対してシワプリセットカーブを生成・配置するオペレーター"""
    bl_idname = "taremin_cloth.add_wrinkle_preset"
    bl_label = "Add Wrinkle Preset"
    bl_description = "Add stylized wrinkle guide curves to the selected bone from preset"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    preset_name: EnumProperty(
        name="Preset",
        description="Select wrinkle preset to instantiate",
        items=_get_preset_items,
    )
    t_position: FloatProperty(
        name="Chain Position (t)",
        description="Normalized position along the bone chain (0.0=Head, 1.0=Tail)",
        default=0.5,
        min=0.0,
        max=1.0,
    )
    scale_radius: FloatProperty(
        name="Radius Scale",
        description="Radius scaling factor for the wrinkle curves",
        default=1.0,
        min=0.1,
        max=5.0,
    )
    start_modal_slide: BoolProperty(
        name="Interactive Slide",
        description="Start 3D viewport modal slide immediately after adding",
        default=True,
    )

    _initial_picked_bone: str = ""

    def invoke(self, context, event):
        self._exec_context = context
        if _require_experimental_enabled(self):
            return {'CANCELLED'}
        self._initial_picked_bone = ""
        armature = _find_target_armature(context)
        if armature:
            all_bones = _extract_all_armature_bones(armature)
            nearest = find_nearest_bone_to_mouse(
                context,
                armature,
                (event.mouse_x, event.mouse_y),
                all_bones_data=all_bones,
            )
            if nearest:
                self._initial_picked_bone = nearest
        return self.execute(context)

    def execute(self, context):
        self._exec_context = context
        if _require_experimental_enabled(self):
            return {'CANCELLED'}
        # 1. アクティブなアーマチュアとボーンを特定
        armature = _find_target_armature(context)
        bone_name = getattr(self, "_initial_picked_bone", "")

        if not bone_name:
            active_obj = context.active_object
            if active_obj and active_obj.type == 'ARMATURE':
                if context.active_pose_bone:
                    bone_name = context.active_pose_bone.name
                elif active_obj.data.bones.active:
                    bone_name = active_obj.data.bones.active.name
            else:
                for obj in context.selected_objects:
                    if hasattr(obj, "taremin_cloth") and obj.taremin_cloth.wrinkle_bone_name:
                        bone_name = obj.taremin_cloth.wrinkle_bone_name
                        break

        if not armature:
            armature = _find_target_armature(context)

        if not armature or not armature.pose or not armature.pose.bones:
            self.report({'ERROR'}, i18n.trans("No armature found in scene. Please select an armature bone."))
            return {'CANCELLED'}

        if not bone_name or bone_name not in armature.pose.bones:
            bone_name = armature.pose.bones[0].name

        # 2. プリセットデータの取得
        preset = None
        if self.preset_name.startswith("user_"):
            p_name = self.preset_name[5:]
            for p_path in get_user_preset_files():
                p = load_preset_from_json(p_path)
                if p and p.name == p_name:
                    preset = p
                    break
        else:
            builtins = get_builtin_presets()
            preset = builtins.get(self.preset_name)

        if not preset:
            self.report({'ERROR'}, f"Preset '{self.preset_name}' not found.")
            return {'CANCELLED'}

        # 3. ボーンチェーンの構築
        bones_data = _extract_bone_chain_data(armature, bone_name)
        if not bones_data:
            self.report({'ERROR'}, f"Failed to extract bone chain for '{bone_name}'.")
            return {'CANCELLED'}

        chain = BoneChain.from_bones(bones_data)

        # 4. 指定位置 t_position における実体化 (チェーン評価中心)
        crest_items, root_items = instantiate_preset_on_chain(
            preset=preset,
            chain=chain,
            t_param=self.t_position,
            scale_radius=self.scale_radius,
        )

        # 5. コレクションへの Curve オブジェクト作成
        col = _find_or_create_collection(context)
        created_objs = []

        all_curve_defs = []
        for item in crest_items:
            all_curve_defs.append(("crest", item))
        for item in root_items:
            all_curve_defs.append(("root", item))

        for idx, (c_type, c_item) in enumerate(all_curve_defs):
            curve_data = bpy.data.curves.new(name=f"{c_type.capitalize()}_{preset.name}_{idx}", type='CURVE')
            curve_data.dimensions = '3D'
            spline = curve_data.splines.new(type='POLY')
            spline.use_cyclic_u = True

            pts = c_item.points
            spline.points.add(len(pts) - 1)
            for p_idx, p in enumerate(pts):
                spline.points[p_idx].co = (p[0], p[1], p[2], 1.0)

            curve_obj = bpy.data.objects.new(name=curve_data.name, object_data=curve_data)
            col.objects.link(curve_obj)

            # カスタムプロパティの設定
            curve_obj["wrinkle_type"] = c_type
            curve_obj["wrinkle_curve_index"] = idx
            curve_obj["wrinkle_strength"] = float(c_item.strength)
            curve_obj["wrinkle_influence_radius"] = float(c_item.influence_radius)
            if c_item.target_radius is not None:
                curve_obj["wrinkle_target_radius"] = float(c_item.target_radius)
            curve_obj["wrinkle_target_bone"] = bone_name
            curve_obj["wrinkle_armature"] = armature.name
            curve_obj["wrinkle_t_param"] = float(self.t_position)
            curve_obj["wrinkle_scale_radius"] = float(self.scale_radius)
            curve_obj["wrinkle_preset"] = preset.name

            # RNAプロパティへの同期
            tw = getattr(curve_obj, "taremin_wrinkle", None)
            if tw:
                try:
                    tw.curve_type = c_type
                    tw.strength = float(c_item.strength)
                    tw.influence_radius = float(c_item.influence_radius)
                    if c_item.target_radius is not None:
                        tw.target_radius = float(c_item.target_radius)
                except Exception:
                    pass

            created_objs.append(curve_obj)

        # 6. オブジェクトの選択状態更新
        bpy.ops.object.select_all(action='DESELECT')
        for o in created_objs:
            o.select_set(True)
        if created_objs:
            context.view_layer.objects.active = created_objs[0]

        # 7. 布オブジェクトへのシワコレクション・ドレープガイド自動割り当て
        _auto_setup_cloth_wrinkle_settings(context, col, armature, bone_name)

        tag_redraw_view3d(context)
        self.report({'INFO'}, i18n.trans(f"Added {len(created_objs)} wrinkle curves for bone '{bone_name}'."))

        # 8. スライドモーダルの自動起動 (ボーン選択フェイズから対話的に開始)
        if self.start_modal_slide and created_objs:
            bpy.ops.taremin_cloth.slide_wrinkle_curves('INVOKE_DEFAULT', initial_phase='PICK_BONE', is_new_addition=True)

        return {'FINISHED'}


class TAREMIN_CLOTH_OT_slide_wrinkle_curves(Operator):
    """3Dビューポート上でボーンチェーンに沿ってシワカーブを直感スライドするモーダルオペレーター"""
    bl_idname = "taremin_cloth.slide_wrinkle_curves"
    bl_label = "Slide Wrinkles Along Bone"
    bl_description = "Interactively slide wrinkle curves along bone chain with mouse drag (Wheel: Radius, Shift+Wheel: Influence, Ctrl+Wheel: Strength, B: Retarget to nearest bone)"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}

    initial_phase: EnumProperty(
        items=[
            ('PICK_BONE', "Pick Bone", "Interactively pick target bone with cursor before sliding"),
            ('SLIDE', "Slide", "Slide along current bone chain"),
        ],
        default='SLIDE',
    )
    is_new_addition: BoolProperty(
        name="Is New Addition",
        description="Whether this modal was invoked right after adding a preset",
        default=False,
    )

    _phase: str = 'SLIDE'
    _is_new_addition: bool = False
    _draw_handler_2d = None
    _draw_handler_3d = None
    _orig_show_in_front: bool = False
    _target_bone_name: str = ""
    _chain_bones_coords: list = []
    _all_bones_data: list = []
    _press_tracker: Any = None
    _target_curves: List[bpy.types.Object] = []
    _armature_obj: Optional[bpy.types.Object] = None
    _chain: Optional[BoneChain] = None
    _init_mouse_x: int = 0
    _base_t: float = 0.5
    _current_t: float = 0.5
    _scale_radius: float = 1.0
    _base_scale_radius: float = 1.0
    _influence_radius: float = 0.03
    _strength: float = 1.0
    _initial_points: List[np.ndarray] = []
    _true_initial_points: List[np.ndarray] = []
    _true_initial_bone_name: str = ""
    _true_initial_base_t: float = 0.5
    _true_initial_scale_radius: float = 1.0
    _true_initial_influence_radius: float = 0.03
    _true_initial_strength: float = 1.0

    def _init_state(self):
        """モーダル実行前の内部状態を初期化"""
        self._phase = 'SLIDE'
        self._is_new_addition = False
        self._draw_handler_2d = None
        self._draw_handler_3d = None
        self._orig_show_in_front = False
        self._target_bone_name = ""
        self._chain_bones_coords = []
        self._all_bones_data = []
        self._press_tracker = PressDragTracker()
        self._target_curves = []
        self._armature_obj = None
        self._chain = None
        self._init_mouse_x = 0
        self._base_t = 0.5
        self._current_t = 0.5
        self._scale_radius = 1.0
        self._base_scale_radius = 1.0
        self._influence_radius = 0.03
        self._strength = 1.0
        self._initial_points = []
        self._true_initial_points = []
        self._true_initial_bone_name = ""
        self._true_initial_base_t = 0.5
        self._true_initial_scale_radius = 1.0
        self._true_initial_influence_radius = 0.03
        self._true_initial_strength = 1.0

    def _delete_target_curves(self, context):
        """新規追加時にキャンセルされた場合、生成されたカーブオブジェクトを削除"""
        for obj in self._target_curves:
            try:
                curve_data = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if curve_data and curve_data.users == 0:
                    bpy.data.curves.remove(curve_data)
            except Exception:
                pass
        self._target_curves = []

    def _collect_curves(self, context) -> bool:
        """選択オブジェクトまたはコレクションからシワカーブを収集"""
        self._target_curves = [o for o in context.selected_objects if o.type == 'CURVE' and "wrinkle_type" in o]
        if not self._target_curves and context.active_object and context.active_object.type == 'CURVE':
            self._target_curves = [context.active_object]

        if not self._target_curves:
            # コレクションから探索
            col = bpy.data.collections.get(COLLECTION_NAME)
            if col:
                self._target_curves = [o for o in col.objects if o.type == 'CURVE']

        if not self._target_curves:
            return False

        # アーマチュアとボーンを特定
        first_obj = self._target_curves[0]
        arm_name = first_obj.get("wrinkle_armature", "")
        bone_name = first_obj.get("wrinkle_target_bone", "")

        arm_obj = bpy.data.objects.get(arm_name)
        if not arm_obj or arm_obj.type != 'ARMATURE':
            for o in context.scene.objects:
                if o.type == 'ARMATURE':
                    arm_obj = o
                    break

        if not arm_obj:
            return False

        self._armature_obj = arm_obj
        self._all_bones_data = _extract_all_armature_bones(arm_obj)

        bones_data = _extract_bone_chain_data(arm_obj, bone_name)
        if not bones_data:
            return False

        self._chain = BoneChain.from_bones(bones_data)
        self._target_bone_name = bone_name
        self._chain_bones_coords = [(head, tail, b_name) for b_name, head, tail, rads in bones_data]

        # 初期パラメータの取得
        self._base_t = float(first_obj.get("wrinkle_t_param", 0.5))
        self._current_t = self._base_t
        self._scale_radius = float(first_obj.get("wrinkle_scale_radius", 1.0))
        # _initial_points に焼き込み済みのscale。以降の更新はこの比で相対拡縮し、
        # 初動での二重拡縮を防ぐ（wrinkle_slide.slide_curves_along_chain 参照）。
        self._base_scale_radius = float(self._scale_radius)
        self._influence_radius = float(first_obj.get("wrinkle_influence_radius", 0.03))
        self._strength = float(first_obj.get("wrinkle_strength", 1.0))

        # 初期点列の保存
        self._initial_points = []
        for obj in self._target_curves:
            pts = []
            for sp in obj.data.splines:
                for p in sp.points:
                    pts.append(list(p.co[:3]))
            self._initial_points.append(np.array(pts, dtype=np.float32))

        # モーダル起動前の真の初期状態（キャンセル時の完全復元用）を退避
        self._true_initial_points = [pts.copy() for pts in self._initial_points]
        self._true_initial_bone_name = bone_name
        self._true_initial_base_t = self._base_t
        self._true_initial_scale_radius = self._scale_radius
        self._true_initial_influence_radius = self._influence_radius
        self._true_initial_strength = self._strength

        return True

    def _get_preset_for_curves(self) -> Optional[WrinklePreset]:
        """対象カーブに設定されたプリセット定義データを取得・復元"""
        if not self._target_curves:
            return None
        preset_name = self._target_curves[0].get("wrinkle_preset", "")
        if not preset_name:
            return None
        builtins = get_builtin_presets()
        if preset_name in builtins:
            return builtins[preset_name]
        for p_path in get_user_preset_files():
            p = load_preset_from_json(p_path)
            if p and p.name == preset_name:
                return p
        return None

    def _retarget_to_bone(self, context, new_bone_name: str):
        """現在のシワカーブを指定ボーンへ瞬時にリターゲット（再吸着）する"""
        if not self._armature_obj or new_bone_name == self._target_bone_name:
            return

        bones_data = _extract_bone_chain_data(self._armature_obj, new_bone_name)
        if not bones_data:
            return

        old_chain = self._chain
        new_chain = BoneChain.from_bones(bones_data)
        self._chain = new_chain
        self._target_bone_name = new_bone_name
        self._chain_bones_coords = [(head, tail, b_name) for b_name, head, tail, rads in bones_data]

        for obj in self._target_curves:
            obj["wrinkle_target_bone"] = new_bone_name

        self._current_t = 0.5
        self._base_t = 0.5

        preset = self._get_preset_for_curves()
        if preset:
            # プリセット定義から新ボーンへ直接実体化
            crest_items, root_items = instantiate_preset_on_chain(
                preset=preset,
                chain=new_chain,
                t_param=0.5,
                scale_radius=self._scale_radius,
            )
            all_defs = []
            for item in crest_items:
                all_defs.append(("crest", item))
            for item in root_items:
                all_defs.append(("root", item))

            self._initial_points = []
            for obj_idx, obj in enumerate(self._target_curves):
                c_idx = obj.get("wrinkle_curve_index", obj_idx)
                if 0 <= c_idx < len(all_defs):
                    c_item = all_defs[c_idx][1]
                else:
                    c_type = obj.get("wrinkle_type", "crest")
                    c_item = None
                    for it_type, it in all_defs:
                        if it_type == c_type:
                            c_item = it
                            break
                    if c_item is None and all_defs:
                        c_item = all_defs[0][1]

                if c_item is not None:
                    pts = c_item.points
                    self._initial_points.append(pts.copy())
                    p_idx = 0
                    for sp in obj.data.splines:
                        for p in sp.points:
                            if p_idx < len(pts):
                                p.co = (pts[p_idx, 0], pts[p_idx, 1], pts[p_idx, 2], 1.0)
                                p_idx += 1
                else:
                    pts = []
                    for sp in obj.data.splines:
                        for p in sp.points:
                            pts.append(list(p.co[:3]))
                    self._initial_points.append(np.array(pts, dtype=np.float32))

                obj["wrinkle_t_param"] = 0.5
                obj["wrinkle_scale_radius"] = float(self._scale_radius)
            # 再取得点列に焼き込み済みのscaleを同期（不変条件の回復）
            self._base_scale_radius = float(self._scale_radius)
        else:
            # プリセット定義がない場合: 旧チェーン断面から新チェーン断面へ正規化円柱座標転送
            if old_chain and self._initial_points:
                old_src = old_chain.evaluate(self._base_t)
                new_tgt = new_chain.evaluate(0.5)
                # _initial_points は取得時scale焼き込み済みのため相対比で転送
                user_scale = float(self._scale_radius) / max(float(self._base_scale_radius), 1e-4)
                rad_ratio = (new_tgt.radius / max(old_src.radius, 1e-4)) * user_scale

                new_initial_pts = []
                for obj, pts in zip(self._target_curves, self._initial_points):
                    if len(pts) == 0:
                        new_initial_pts.append(pts.copy())
                        continue
                    new_pts = []
                    for p in pts:
                        diff = p - old_src.position
                        dz = float(np.dot(diff, old_src.axis))
                        r_vec = diff - old_src.axis * dz
                        x_comp = float(np.dot(r_vec, old_src.normal))
                        y_comp = float(np.dot(r_vec, old_src.binormal))

                        new_r = (new_tgt.normal * x_comp + new_tgt.binormal * y_comp) * rad_ratio
                        new_p = new_tgt.position + new_tgt.axis * dz + new_r
                        new_pts.append(new_p)

                    arr = np.array(new_pts, dtype=np.float32)
                    new_initial_pts.append(arr)

                    p_idx = 0
                    for sp in obj.data.splines:
                        for p in sp.points:
                            if p_idx < len(arr):
                                p.co = (arr[p_idx, 0], arr[p_idx, 1], arr[p_idx, 2], 1.0)
                                p_idx += 1

                    obj["wrinkle_t_param"] = 0.5
                    obj["wrinkle_scale_radius"] = float(self._scale_radius)

                self._initial_points = new_initial_pts
                # 転送後点列の焼き込みscaleに同期（不変条件の回復）
                self._base_scale_radius = float(self._scale_radius) * (
                    new_tgt.radius / max(old_src.radius, 1e-4))
            else:
                self._update_curves_geometry()

        try:
            pb = self._armature_obj.pose.bones.get(new_bone_name)
            if pb and hasattr(self._armature_obj.data, "bones"):
                self._armature_obj.data.bones.active = pb.bone
        except Exception:
            pass

        tag_redraw_view3d(context)


    def _update_curves_geometry(self):
        """現在のパラメータ (current_t, scale_radius) に基づき全カーブの頂点座標を更新"""
        if not self._chain or not self._target_curves:
            return

        slid = slide_curves_along_chain(
            curve_points_list=self._initial_points,
            chain=self._chain,
            source_t=self._base_t,
            target_t=self._current_t,
            scale_radius=self._scale_radius,
            base_scale_radius=self._base_scale_radius,
        )

        for obj, pts in zip(self._target_curves, slid):
            obj["wrinkle_t_param"] = float(self._current_t)
            obj["wrinkle_scale_radius"] = float(self._scale_radius)
            obj["wrinkle_influence_radius"] = float(self._influence_radius)
            obj["wrinkle_strength"] = float(self._strength)

            tw = getattr(obj, "taremin_wrinkle", None)
            if tw:
                try:
                    tw.influence_radius = float(self._influence_radius)
                    tw.strength = float(self._strength)
                except Exception:
                    pass

            p_idx = 0
            for sp in obj.data.splines:
                for p in sp.points:
                    if p_idx < len(pts):
                        p.co = (pts[p_idx, 0], pts[p_idx, 1], pts[p_idx, 2], 1.0)
                        p_idx += 1

    def _restore_initial_geometry(self):
        """キャンセル時にモーダル起動前の真の初期座標・初期ボーン・初期パラメータを完全復元"""
        pts_list = self._true_initial_points if (hasattr(self, "_true_initial_points") and self._true_initial_points) else self._initial_points
        orig_bone = getattr(self, "_true_initial_bone_name", "") or self._target_bone_name
        orig_t = getattr(self, "_true_initial_base_t", self._base_t)
        orig_scale_r = getattr(self, "_true_initial_scale_radius", self._scale_radius)
        orig_inf_r = getattr(self, "_true_initial_influence_radius", self._influence_radius)
        orig_str = getattr(self, "_true_initial_strength", self._strength)

        for obj, pts in zip(self._target_curves, pts_list):
            if orig_bone:
                obj["wrinkle_target_bone"] = orig_bone
            obj["wrinkle_t_param"] = float(orig_t)
            obj["wrinkle_scale_radius"] = float(orig_scale_r)
            obj["wrinkle_influence_radius"] = float(orig_inf_r)
            obj["wrinkle_strength"] = float(orig_str)

            tw = getattr(obj, "taremin_wrinkle", None)
            if tw:
                try:
                    tw.influence_radius = float(orig_inf_r)
                    tw.strength = float(orig_str)
                except Exception:
                    pass

            p_idx = 0
            for sp in obj.data.splines:
                for p in sp.points:
                    if p_idx < len(pts):
                        p.co = (pts[p_idx, 0], pts[p_idx, 1], pts[p_idx, 2], 1.0)
                        p_idx += 1

    def _draw_bone_3d(self):
        """3D空間上で対象ボーンおよびアーマチュア全体を最前面で八面体強調表示し、シワ影響範囲を描画"""
        if self._all_bones_data:
            chain_names = [item[2] for item in self._chain_bones_coords] if self._chain_bones_coords else []
            draw_wrinkle_armature_overlay(
                all_bones_data=self._all_bones_data,
                active_bone_name=self._target_bone_name,
                chain_bone_names=chain_names,
            )

        # 操作中カーブの影響範囲（角度・長さ帯と目標前後の厚み）を半透明描画
        if self._target_curves:
            modal_curves_data = []
            for obj in self._target_curves:
                pts = []
                w_mat = getattr(obj, "matrix_world", None)
                for sp in obj.data.splines:
                    for p in sp.points:
                        p_xyz = p.co[:3]
                        if w_mat is not None and hasattr(w_mat, "__matmul__"):
                            try:
                                import mathutils
                                world_pt = w_mat @ mathutils.Vector(p_xyz)
                                pts.append(list(world_pt[:3]))
                            except Exception:
                                pts.append(list(p_xyz))
                        else:
                            pts.append(list(p_xyz))
                if len(pts) >= 2:
                    tw = getattr(obj, "taremin_wrinkle", None)
                    c_type = tw.curve_type if tw else obj.get("wrinkle_type", "crest")
                    tgt_r = 0.0
                    if tw:
                        try:
                            tgt_r = float(getattr(tw, "target_radius", 0.0) or 0.0)
                        except Exception:
                            tgt_r = 0.0
                    else:
                        try:
                            tgt_r = float(obj.get("wrinkle_target_radius", 0.0) or 0.0)
                        except Exception:
                            tgt_r = 0.0
                    modal_curves_data.append((c_type, pts, float(self._influence_radius), tgt_r))
            if modal_curves_data:
                try:
                    import numpy as _np
                    from ..engine.wrinkle_field import build_orthonormal_basis
                    # カーブはチェーン上の current_t に存在するため、描画座標系も
                    # 同一チェーン評価位置にする (固定ボーン座標系では屈曲時に垂直化する)。
                    chain_ev = None
                    try:
                        if self._chain is not None:
                            chain_ev = self._chain.evaluate(float(self._current_t))
                    except Exception:
                        chain_ev = None
                    if chain_ev is not None:
                        meshes = build_wrinkle_influence_meshes(
                            modal_curves_data,
                            _np.array(chain_ev.position[:3], dtype=_np.float64),
                            _np.array(chain_ev.axis[:3], dtype=_np.float64),
                            _np.array(chain_ev.normal[:3], dtype=_np.float64),
                            float(chain_ev.radius),
                        )
                        if meshes:
                            draw_wrinkle_influence_meshes(meshes)
                    else:
                        b_head = b_tail = None
                        b_rad = 0.0
                        for item in (self._all_bones_data or []):
                            if item[0] == self._target_bone_name:
                                b_head = _np.array(item[1][:3], dtype=_np.float64)
                                b_tail = _np.array(item[2][:3], dtype=_np.float64)
                                try:
                                    b_rad = float(item[3]) if len(item) > 3 else 0.0
                                except Exception:
                                    b_rad = 0.0
                                if isinstance(b_rad, tuple):
                                    b_rad = float((b_rad[0] + b_rad[1]) / 2.0)
                                break
                        if b_head is not None:
                            axis_raw = b_tail - b_head
                            if float(_np.linalg.norm(axis_raw)) >= 1e-6:
                                axis_n, n_vec, _bn = build_orthonormal_basis(axis_raw)
                                meshes = build_wrinkle_influence_meshes(
                                    modal_curves_data, b_head, axis_n, n_vec, b_rad,
                                )
                                if meshes:
                                    draw_wrinkle_influence_meshes(meshes)
                except Exception:
                    pass

    def _draw_hud(self):
        """3DビューポートのPOST_PIXEL描画コールバック"""
        context = bpy.context
        region = getattr(context, "region", None)
        if not region:
            return

        if self._phase == 'PICK_BONE':
            line1 = i18n.trans("[LMB] Select Bone  |  [RMB/Esc] Cancel")
            line2 = i18n.trans(f"Phase 1: Hover over bone to pick -> Target: [{self._target_bone_name}]")
        else:
            line1 = i18n.trans("[LMB/Enter] Confirm  [RMB/Esc] Cancel  [Wheel] Radius  [Shift+Wheel] Influence  [B] Change Bone")
            line2 = f"Phase 2: Bone [{self._target_bone_name}] | Slide t={self._current_t:.3f} | Radius={self._scale_radius:.2f}x | Inf={self._influence_radius*1000.0:.1f}mm | Str={self._strength:.2f}"

        draw_wrinkle_slide_hud(region, line1, line2)

    def _cleanup(self, context):
        """描画ハンドラーの解除とアーマチュア表示状態の復元"""
        if self._draw_handler_2d:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler_2d, 'WINDOW')
            except Exception:
                pass
            self._draw_handler_2d = None
        if self._draw_handler_3d:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler_3d, 'WINDOW')
            except Exception:
                pass
            self._draw_handler_3d = None
        if self._armature_obj:
            try:
                self._armature_obj.show_in_front = self._orig_show_in_front
            except Exception:
                pass
        tag_redraw_view3d(context)

    def invoke(self, context, event):
        self._exec_context = context
        if _require_experimental_enabled(self):
            return {'CANCELLED'}
        self._init_state()
        self._phase = getattr(self, "initial_phase", 'SLIDE')
        self._is_new_addition = bool(getattr(self, "is_new_addition", False))

        if not self._collect_curves(context):
            self.report({'WARNING'}, i18n.trans("No wrinkle curves or armature found to slide."))
            return {'CANCELLED'}

        self._init_mouse_x = event.mouse_x

        # アーマチュアの最前面表示を一時的に有効化
        if self._armature_obj:
            try:
                self._orig_show_in_front = bool(getattr(self._armature_obj, "show_in_front", False))
                self._armature_obj.show_in_front = True
                pb = self._armature_obj.pose.bones.get(self._target_bone_name)
                if pb and hasattr(self._armature_obj.data, "bones"):
                    self._armature_obj.data.bones.active = pb.bone
            except Exception:
                pass

        # 3Dオーバーレイ描画（最前面ボーン強調表示）の登録
        self._draw_handler_3d = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_bone_3d, (), 'WINDOW', 'POST_VIEW'
        )

        # 2D HUD描画の登録 (docs/hud.md に準拠)
        self._draw_handler_2d = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_hud, (), 'WINDOW', 'POST_PIXEL'
        )

        context.window_manager.modal_handler_add(self)
        tag_redraw_view3d(context)
        return {'RUNNING_MODAL'}

    def _report(self, type_set, msg: str):
        """オペレーターのreportを安全に呼び出すヘルパー"""
        if hasattr(self, "report"):
            try:
                self.report(type_set, msg)
            except Exception:
                pass

    def modal(self, context, event):
        # 左マウスボタンの解放で押下追跡を解消し、クリック/ドラッグを確定する
        if is_left_release(event):
            self._press_tracker.on_release_event(event)

        # ---------------------------------------------------------
        # フェイズ1: 初期対象ボーン決定フェイズ (PICK_BONE)
        # ---------------------------------------------------------
        if self._phase == 'PICK_BONE':
            if event.type == 'MOUSEMOVE':
                if self._armature_obj or self._all_bones_data:
                    nearest = find_nearest_bone_to_mouse(
                        context,
                        self._armature_obj,
                        (event.mouse_x, event.mouse_y),
                        all_bones_data=self._all_bones_data,
                    )
                    if nearest and nearest != self._target_bone_name:
                        self._retarget_to_bone(context, nearest)
                tag_redraw_view3d(context)
                return {'RUNNING_MODAL'}

            elif event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
                # ボーン決定 -> スライドフェイズへ移行！
                self._phase = 'SLIDE'
                self._init_mouse_x = event.mouse_x
                self._base_t = 0.5
                self._current_t = 0.5

                # スライドキャンセル（ESC）時のロールバック先を、決定されたボーン・配置に更新
                self._true_initial_bone_name = self._target_bone_name
                self._true_initial_base_t = 0.5
                self._true_initial_scale_radius = self._scale_radius
                self._true_initial_influence_radius = self._influence_radius
                self._true_initial_strength = self._strength
                self._true_initial_points = [pts.copy() for pts in self._initial_points]

                if event.type == 'LEFTMOUSE':
                    self._press_tracker.on_press_event(event)
                tag_redraw_view3d(context)
                self._report({'INFO'}, i18n.trans(f"Bone '{self._target_bone_name}' selected. Drag mouse to slide."))
                return {'RUNNING_MODAL'}

            elif event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
                # ボーン選択中にキャンセル (初期位置を復元して維持)
                self._cleanup(context)
                self._restore_initial_geometry()
                tag_redraw_view3d(context)
                self._report({'INFO'}, i18n.trans("Cancelled wrinkle bone picking."))
                return {'CANCELLED'}

            return {'RUNNING_MODAL'}

        # ---------------------------------------------------------
        # フェイズ2: スライドフェイズ (SLIDE)
        # ---------------------------------------------------------
        # 1. マウス移動によるスライド (t パラメータ)
        if event.type == 'MOUSEMOVE':
            self._press_tracker.on_move_event(event)
            dx = event.mouse_x - self._init_mouse_x
            speed = 0.0005 if event.shift else 0.002
            self._current_t = float(np.clip(self._base_t + dx * speed, 0.0, 1.0))
            self._update_curves_geometry()
            tag_redraw_view3d(context)
            return {'RUNNING_MODAL'}

        # 2. マウスホイールによるパラメータ調整
        elif event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            delta = 1.0 if event.type == 'WHEELUPMOUSE' else -1.0
            if event.shift:
                # 影響半径 (Influence Radius) の調整 (+- 2mm)
                self._influence_radius = max(0.002, self._influence_radius + delta * 0.002)
            elif event.ctrl:
                # 強度 (Strength) の調整 (+- 0.1)
                self._strength = max(0.0, self._strength + delta * 0.1)
            else:
                # 半径スケール (Radius Scale) の調整 (+- 5%)
                self._scale_radius = max(0.1, self._scale_radius + delta * 0.05)

            self._update_curves_geometry()
            tag_redraw_view3d(context)
            return {'RUNNING_MODAL'}

        # 3. Bキー: 再びボーン選択フェイズに戻る
        elif event.type == 'B' and event.value == 'PRESS':
            self._phase = 'PICK_BONE'
            self._press_tracker.reset()
            tag_redraw_view3d(context)
            self._report({'INFO'}, i18n.trans("Switched to bone picking phase. Hover over a bone and click LMB."))
            return {'RUNNING_MODAL'}

        # 4. 確定操作 (LMB または Enter)
        elif event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            if event.type == 'LEFTMOUSE' and self._press_tracker.pressed:
                # フェイズ1でのクリック（押し込み）がまだ完了していないため確定を無視
                return {'RUNNING_MODAL'}
            self._cleanup(context)
            col = bpy.data.collections.get(COLLECTION_NAME)
            if col:
                _auto_setup_cloth_wrinkle_settings(context, col, self._armature_obj, self._target_bone_name)
            self._report({'INFO'}, i18n.trans(f"Confirmed wrinkle slide at t={self._current_t:.3f} on bone '{self._target_bone_name}'."))
            return {'FINISHED'}

        # 5. キャンセル操作 (RMB または Esc: スライド前の位置を復元して維持)
        elif event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            self._cleanup(context)
            self._restore_initial_geometry()
            tag_redraw_view3d(context)
            self._report({'INFO'}, i18n.trans("Cancelled wrinkle slide."))
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}


class TAREMIN_CLOTH_OT_save_wrinkle_preset(Operator):
    """選択されたシワカーブを基準ボーン正規化でプリセットJSONとして保存するオペレーター"""
    bl_idname = "taremin_cloth.save_wrinkle_preset"
    bl_label = "Save Wrinkle Preset"
    bl_description = "Save selected wrinkle curves normalized against reference bone as a preset JSON"
    bl_translation_context = i18n.CONTEXT
    bl_options = {'REGISTER', 'UNDO'}

    preset_name: StringProperty(
        name="Preset Name",
        description="Name of the new wrinkle preset",
        default="custom_wrinkle",
    )
    category: StringProperty(
        name="Category",
        description="Preset category",
        default="user",
    )
    description: StringProperty(
        name="Description",
        description="Description of the wrinkle preset",
        default="User defined wrinkle preset",
    )
    bone_name: StringProperty(
        name="Reference Bone",
        description="Reference bone used to normalize curve coordinates",
        default="",
    )

    def invoke(self, context, event):
        self._exec_context = context
        if _require_experimental_enabled(self):
            return {'CANCELLED'}
        # 選択カーブから基準ボーン名を自動検知
        for o in context.selected_objects:
            if o.type == 'CURVE' and "wrinkle_target_bone" in o:
                self.bone_name = o["wrinkle_target_bone"]
                break
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        self._exec_context = context
        if _require_experimental_enabled(self):
            return {'CANCELLED'}
        curves_to_save: List[WrinkleCurveItem] = []
        arm_name = ""

        # 選択中のカーブオブジェクトから点列を収集
        for obj in context.selected_objects:
            if obj.type != 'CURVE':
                continue
            w_type = obj.get("wrinkle_type", "crest")
            w_str = float(obj.get("wrinkle_strength", 1.0))
            w_inf = float(obj.get("wrinkle_influence_radius", 0.03))
            w_target_r = obj.get("wrinkle_target_radius", None)
            if w_target_r is not None:
                w_target_r = float(w_target_r)
            if not arm_name:
                arm_name = obj.get("wrinkle_armature", "")

            world_matrix = obj.matrix_world
            for sp in obj.data.splines:
                pts = [list((world_matrix @ p.co)[:3]) for p in sp.points]
                if len(pts) >= 2:
                    curves_to_save.append(WrinkleCurveItem(
                        points=np.array(pts, dtype=np.float32),
                        strength=w_str,
                        influence_radius=w_inf,
                        target_radius=w_target_r,
                    ))

        if not curves_to_save:
            self.report({'ERROR'}, i18n.trans("No valid curve points selected to save."))
            return {'CANCELLED'}

        # 基準ボーンの幾何を取得
        arm_obj = bpy.data.objects.get(arm_name)
        if not arm_obj:
            for o in context.scene.objects:
                if o.type == 'ARMATURE':
                    arm_obj = o
                    break

        if not arm_obj or not arm_obj.pose or not self.bone_name or self.bone_name not in arm_obj.pose.bones:
            self.report({'ERROR'}, f"Reference bone '{self.bone_name}' not found.")
            return {'CANCELLED'}

        pb = arm_obj.pose.bones[self.bone_name]
        w_mat = arm_obj.matrix_world
        h_w = np.array((w_mat @ pb.head)[:3], dtype=np.float32)
        t_w = np.array((w_mat @ pb.tail)[:3], dtype=np.float32)
        from ..engine.wrinkle_field import resolve_bone_radius
        r_w = resolve_bone_radius(
            0.0,
            float(getattr(pb.bone, "head_radius", 0.0) or 0.0),
            float(getattr(pb.bone, "tail_radius", 0.0) or 0.0),
            float(getattr(pb, "length", 0.0) or 0.0),
        )

        # 正規化プリセットの生成
        preset = normalize_curves_to_preset(
            curves=curves_to_save,
            bone_head=h_w,
            bone_tail=t_w,
            bone_radius=r_w,
            preset_name=self.preset_name,
            category=self.category,
            description=self.description,
        )

        saved_path = save_preset_to_json(preset)
        self.report({'INFO'}, i18n.trans(f"Saved wrinkle preset '{self.preset_name}' to {saved_path}."))
        return {'FINISHED'}


OPERATOR_CLASSES = (
    TAREMIN_CLOTH_OT_add_wrinkle_preset,
    TAREMIN_CLOTH_OT_slide_wrinkle_curves,
    TAREMIN_CLOTH_OT_save_wrinkle_preset,
)
