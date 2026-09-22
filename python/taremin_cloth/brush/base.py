"""
ブラシ基盤 (taremin_cloth.brush.base)
全ブラシ共通のサービスとツールインターフェース。
個別ブラシ (single_grab / range_grab) は BaseBrushTool を継承する。
新規ブラシの追加は: 新規モジュール + brush/__init__.py の registry 登録のみ。
"""

import mathutils
import numpy as np
from bpy_extras import view3d_utils

from ..utils import drawing
from .math import (
    BRUSH_RADIUS_MIN,
    BRUSH_RADIUS_MAX,
    BRUSH_STRENGTH_MIN,
    BRUSH_STRENGTH_MAX,
    radial_adjust,
)


class BrushContext:
    """1イベント分のブラシ実行コンテキスト (確保・解放なしの軽量束ね)"""

    def __init__(self, op, context, obj, sim, coords, event, mouse_pos):
        self.op = op
        self.context = context
        self.obj = obj
        self.sim = sim
        self.coords = coords
        self.event = event
        self.mouse_pos = mouse_pos
        settings = getattr(obj, "taremin_cloth", None)
        self.brush = getattr(settings, "brush", None)


def brush_opt(brush, name, default):
    """ブラシ設定値の防御的取得 (未登録ブレンド対策)"""
    if brush is None:
        return default
    return getattr(brush, name, default)


class BaseBrushTool:
    """個別ブラシのインターフェース。戻り値 True = イベント消費"""

    name = 'BASE'

    def on_press(self, ctx):
        return False

    def on_move(self, ctx):
        return False

    def on_release(self, ctx, pinned_verts):
        return False

    def on_hover(self, ctx):
        return False

    def on_activate(self, ctx):
        pass

    def on_deactivate(self, ctx, pinned_verts=None):
        pass

    def abort(self, ctx):
        self.reset()

    def reset(self):
        pass


def resolve_brush_center(context, obj, coords, mouse_pos):
    """ブラシ中心 (ローカル) とレイ情報の解決。レイ外れ時は最近傍頂点にフォールバックする"""
    region = context.region
    rv3d = context.region_data
    if not (region and rv3d):
        return None
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_pos)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_pos)
    if not (origin and direction):
        return None
    center_local = None
    hit_t = None
    try:
        matrix_inv = obj.matrix_world.inverted()
        ray_o = matrix_inv @ origin
        ray_d = (matrix_inv.to_3x3() @ direction).normalized()
        hit, hit_loc, _, _ = obj.ray_cast(ray_o, ray_d)
        if hit:
            center_local = np.array(hit_loc, dtype=np.float32)
            hit_world = obj.matrix_world @ mathutils.Vector(hit_loc)
            hit_t = float((hit_world - origin).dot(direction))
    except Exception:
        center_local = None
    pos_2d = np.asarray(coords, dtype=np.float32).reshape((-1, 3))
    if center_local is None:
        try:
            best_idx, best_d = None, float('inf')
            for i, p in enumerate(pos_2d):
                world_p = obj.matrix_world @ mathutils.Vector(p)
                screen_co = view3d_utils.location_3d_to_region_2d(region, rv3d, world_p)
                if screen_co:
                    d = (screen_co.x - mouse_pos[0]) ** 2 + (screen_co.y - mouse_pos[1]) ** 2
                    if d < best_d:
                        best_d, best_idx = d, i
            if best_idx is None:
                return None
            center_local = pos_2d[best_idx]
        except Exception:
            return None
    return {
        "center": center_local,
        "origin": origin,
        "direction": direction,
        "hit_t": hit_t,
        "pos_2d": pos_2d,
    }


def update_brush_circle(context, obj, center_local, radius, mouse_pos):
    """ブラシカーソル円を更新する"""
    region = context.region
    rv3d = context.region_data
    try:
        _center_world = obj.matrix_world @ mathutils.Vector(np.asarray(center_local).tolist())
        _c2d = view3d_utils.location_3d_to_region_2d(region, rv3d, _center_world)
        _radius_px = 40.0
        if _c2d is not None:
            _view_inv = rv3d.view_matrix.inverted()
            _right = _view_inv.to_3x3().col[0].normalized()
            _edge_world = _center_world + _right * radius
            _e2d = view3d_utils.location_3d_to_region_2d(region, rv3d, _edge_world)
            if _e2d is not None:
                _radius_px = max(2.0, float((_e2d - _c2d).length))
        drawing.set_brush_info(mouse_pos[0], mouse_pos[1], _radius_px)
    except Exception:
        pass


def init_brush_circle_at(ctx, radius):
    """指定マウス位置にブラシ円を初期化する (モード突入・ラジアル開始用)"""
    try:
        res = resolve_brush_center(ctx.context, ctx.obj, ctx.coords, ctx.mouse_pos)
        if res is not None:
            update_brush_circle(ctx.context, ctx.obj, res["center"], radius, ctx.mouse_pos)
            return True
    except Exception:
        pass
    return False


def nudge_brush_radius(obj, factor, mouse_pos):
    """ブラシ半径を段階変更し、表示円も同率で即時更新する"""
    settings = getattr(obj, "taremin_cloth", None)
    brush = getattr(settings, "brush", None) if settings else None
    if brush is None:
        return False
    prev_r = float(getattr(brush, "radius", 0.05))
    brush.radius = max(BRUSH_RADIUS_MIN, min(BRUSH_RADIUS_MAX, prev_r * factor))
    try:
        prev = drawing.get_brush_info()
        if prev is not None and prev_r > 1e-9:
            drawing.set_brush_info(
                mouse_pos[0], mouse_pos[1],
                float(prev.get("radius_px", 40.0)) * (float(brush.radius) / prev_r))
    except Exception:
        pass
    return True


class RadialController:
    """F / Shift+F ラジアル調整の状態機械 (全ブラシ共通)"""

    MODES = ('RADIUS', 'STRENGTH')

    def __init__(self):
        self.mode = None
        self.start_x = 0.0
        self.start_value = 0.0
        self.attr = 'radius'

    @property
    def active(self):
        return self.mode is not None

    def enter(self, mode, start_x, brush):
        if mode == 'STRENGTH':
            self.attr = 'strength'
            default, lo, hi = 0.8, BRUSH_STRENGTH_MIN, BRUSH_STRENGTH_MAX
        else:
            mode = 'RADIUS'
            self.attr = 'radius'
            default, lo, hi = 0.05, BRUSH_RADIUS_MIN, BRUSH_RADIUS_MAX
        self.mode = mode
        self.start_x = float(start_x)
        self.start_value = float(getattr(brush, self.attr, default)) if brush else default
        self._lo, self._hi = lo, hi

    def update(self, ctx):
        brush = ctx.brush
        if brush is None or not self.active:
            return False
        dx = float(ctx.mouse_pos[0]) - float(self.start_x)
        new_val = radial_adjust(self.start_value, dx, self._lo, self._hi)
        if self.attr == 'strength':
            brush.strength = new_val
        else:
            prev_r = float(getattr(brush, "radius", new_val))
            brush.radius = new_val
            try:
                prev = drawing.get_brush_info()
                if prev is not None and prev_r > 1e-9:
                    drawing.set_brush_info(
                        ctx.mouse_pos[0], ctx.mouse_pos[1],
                        float(prev.get("radius_px", 40.0)) * (new_val / prev_r))
            except Exception:
                pass
        return True

    def confirm(self):
        self.mode = None

    def cancel(self, ctx):
        brush = ctx.brush
        if brush is not None and self.active:
            try:
                if self.attr == 'strength':
                    brush.strength = float(self.start_value)
                else:
                    brush.radius = float(self.start_value)
            except Exception:
                pass
        self.mode = None
