"""
ブラシ基盤 (taremin_cloth.brush.base)
全ブラシ共通のサービスとツールインターフェース。
個別ブラシ (single_grab / range_grab) は BaseBrushTool を継承する。
新規ブラシの追加は: 新規モジュール + brush/__init__.py の registry 登録のみ。
"""

import mathutils
import numpy as np
from bpy_extras import view3d_utils

from ..utils.args_guard import allow_positional
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


@allow_positional
def brush_opt(brush, name, default):
    """ブラシ設定値の防御的取得 (未登録ブレンド対策)"""
    if brush is None:
        return default
    return getattr(brush, name, default)


class ActiveRegionManager:
    """Active-Region Sculpt Mode: ブラシ影響半径外の遠隔頂点を一時ピン固定して高速化するマネージャー"""

    def __init__(self):
        self._temp_frozen_indices = []

    @property
    def is_active(self):
        return len(self._temp_frozen_indices) > 0

    @allow_positional
    def freeze_inactive_region(self, sim, coords, center, radius, *, active_scale=2.5, exclude_pins=None):
        """centerからの距離が radius * active_scale を超える頂点を現在位置でピン固定する"""
        if sim is None or coords is None or len(coords) == 0:
            return
        if self._temp_frozen_indices:
            self.release_temp_pins(sim)

        coords_2d = coords.reshape((-1, 3))
        center_vec = np.asarray(center, dtype=np.float32).reshape((1, 3))
        diff = coords_2d - center_vec
        d2 = np.sum(diff * diff, axis=1)
        thresh2 = (float(radius) * float(active_scale)) ** 2

        # 閾値より外側の頂点
        outer_indices = np.where(d2 > thresh2)[0]
        if exclude_pins:
            exclude_set = set(exclude_pins)
            outer_indices = [int(idx) for idx in outer_indices if int(idx) not in exclude_set]
        else:
            outer_indices = [int(idx) for idx in outer_indices]

        if outer_indices:
            targets = coords_2d[outer_indices]
            weights = [1.0] * len(outer_indices)
            if hasattr(sim, "set_pins_batch"):
                sim.set_pins_batch(outer_indices, targets, weights)
            else:
                for idx, pos in zip(outer_indices, targets):
                    sim.set_pin(idx, [float(pos[0]), float(pos[1]), float(pos[2])], 1.0)
            self._temp_frozen_indices = outer_indices
        else:
            self._temp_frozen_indices = []

    def release_temp_pins(self, sim):
        """一時固定された全ピンを解放する"""
        if sim is None or not self._temp_frozen_indices:
            self._temp_frozen_indices = []
            return
        if hasattr(sim, "release_pins_batch"):
            try:
                sim.release_pins_batch(self._temp_frozen_indices)
            except Exception:
                pass
        else:
            for idx in self._temp_frozen_indices:
                try:
                    sim.release_pin(idx)
                except Exception:
                    pass
        self._temp_frozen_indices = []


class BaseBrushTool:
    """個別ブラシのインターフェース。戻り値 True = イベント消費"""

    name = 'BASE'
    # HUD表示用ラベルキー（i18n.trans適用は呼出側で行う）
    label_key = 'Grab'

    def __init__(self):
        self.active_region = ActiveRegionManager()
        self.reset()

    def hud_info(self, brush, *, radial=None):
        """HUD Guidance用の構造化情報を返す（docs/hud.md:5）。

        戻り値: {"tool", "label_key", "radius", "strength",
                 "falloff", "show_falloff", "adjusting"}
        新規ブラシは本メソッドのオーバーライド＋TOOLS登録のみで表示拡張できる。
        """
        adjusting = None
        try:
            if radial is not None and bool(getattr(radial, "active", False)):
                adjusting = getattr(radial, "mode", None)
        except Exception:
            adjusting = None
        return {
            "tool": self.name,
            "label_key": self.label_key,
            "radius": None,
            "strength": None,
            "falloff": None,
            "show_falloff": False,
            "adjusting": adjusting,
        }

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

    def on_deactivate(self, ctx, pinned_verts=None):  # audit: allow-positional
        pass

    def abort(self, ctx):
        self.reset()

    def reset(self):
        pass


def _project_points_to_screen_2d(pos_2d, *, region, rv3d):
    """rv3d.perspective_matrix を使って全頂点を2Dスクリーン座標へ一括投影する。

    利用不可 (モック・領域サイズ未定義等) の場合は None を返す。
    戻り値: (screen_x, screen_y, valid_indices)
    """
    persp_raw = getattr(rv3d, "perspective_matrix", None)
    rw = getattr(region, "width", None)
    rh = getattr(region, "height", None)
    if (
        persp_raw is None
        or not isinstance(rw, (int, float))
        or not isinstance(rh, (int, float))
        or rw <= 0
        or rh <= 0
        or len(pos_2d) == 0
    ):
        return None
    try:
        persp = np.asarray(persp_raw, dtype=np.float64)
        if persp.shape != (4, 4):
            return None
        ones = np.ones((len(pos_2d), 1), dtype=np.float64)
        pts_h = np.hstack([pos_2d.astype(np.float64), ones])
        clip = pts_h @ persp.T
        w = clip[:, 3]
        valid_mask = (w > 1e-6) & np.all(np.isfinite(clip), axis=1)
        if not np.any(valid_mask):
            return (
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64),
            )
        valid_idx = np.flatnonzero(valid_mask)
        w_v = w[valid_idx]
        ndc_x = clip[valid_idx, 0] / w_v
        ndc_y = clip[valid_idx, 1] / w_v
        sx = (ndc_x * 0.5 + 0.5) * float(rw)
        sy = (ndc_y * 0.5 + 0.5) * float(rh)
        return sx, sy, valid_idx
    except Exception:
        return None


def _pick_nearest_screen_vertex(pos_2d, *, region, rv3d, mouse_pos, max_dist_px=50.0, view3d_mod=None, vector_cls=None):
    """2Dスクリーン距離 (max_dist_px 未満) に基づき最寄り頂点インデックスを返す。"""
    max_dist_sq = float(max_dist_px) ** 2
    proj = _project_points_to_screen_2d(pos_2d, region=region, rv3d=rv3d)
    if proj is not None:
        sx, sy, valid_idx = proj
        if len(valid_idx) == 0:
            return None
        dist_sq = (sx - float(mouse_pos[0])) ** 2 + (sy - float(mouse_pos[1])) ** 2
        min_pos = int(np.argmin(dist_sq))
        if float(dist_sq[min_pos]) < max_dist_sq:
            return int(valid_idx[min_pos])
        return None

    v3d = view3d_mod if view3d_mod is not None else view3d_utils
    vec_fn = vector_cls if vector_cls is not None else mathutils.Vector
    best_idx = None
    best_dist = max_dist_sq
    for i, p in enumerate(pos_2d):
        world_p = vec_fn((float(p[0]), float(p[1]), float(p[2])))
        screen_co = v3d.location_3d_to_region_2d(region, rv3d, world_p)
        if screen_co:
            dist = (screen_co.x - mouse_pos[0]) ** 2 + (screen_co.y - mouse_pos[1]) ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = i
    return best_idx


def pick_target_vertex_index(context, obj, *, coords, mouse_pos, view3d_mod=None, vector_cls=None):
    """視線レイキャスト優先・紐優先・2D近傍フォールバックでカーソル下の頂点インデックスを返す。"""
    region = getattr(context, "region", None)
    rv3d = getattr(context, "region_data", None)
    if not (region and rv3d):
        return None, None, None
    v3d = view3d_mod if view3d_mod is not None else view3d_utils
    vec_fn = vector_cls if vector_cls is not None else mathutils.Vector
    origin = v3d.region_2d_to_origin_3d(region, rv3d, mouse_pos)
    direction = v3d.region_2d_to_vector_3d(region, rv3d, mouse_pos)
    if not (origin and direction):
        return None, origin, direction

    pos_2d = np.asarray(coords, dtype=np.float32).reshape((-1, 3))
    best_idx = None
    hit_world_dist = float('inf')

    def _vec_len(v) -> float:
        if hasattr(v, "length"):
            return float(v.length)
        return float((v.x ** 2 + v.y ** 2 + v.z ** 2) ** 0.5)

    # 1. 視線レイキャストによる最前面ポリゴンの交差判定
    try:
        matrix_world = obj.matrix_world
        matrix_inv = matrix_world.inverted()
        ray_origin_local = matrix_inv @ origin
        ray_dir_local = (matrix_inv.to_3x3() @ direction).normalized()

        hit, hit_loc, _hit_normal, face_idx = obj.ray_cast(ray_origin_local, ray_dir_local)
        if hit and face_idx is not None and 0 <= face_idx < len(obj.data.polygons):
            hit_world = matrix_world @ hit_loc
            hit_world_dist = _vec_len(hit_world - origin)
            polygon = obj.data.polygons[face_idx]
            min_dist_sq = float('inf')
            for v_idx in polygon.vertices:
                v_co = obj.data.vertices[v_idx].co
                dist_sq = (v_co - hit_loc).length_squared
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_idx = int(v_idx)
    except Exception:
        best_idx = None

    # 1b. 紐（Curve）頂点が手前またはマウス直下 (30px以内) にある場合は紐頂点を優先選択
    try:
        n_mesh_verts = len(obj.data.vertices)
    except Exception:
        n_mesh_verts = 0
    if n_mesh_verts > 0 and len(pos_2d) > n_mesh_verts:
        best_curve_dist_sq = 900.0
        best_curve_idx = None
        for i in range(n_mesh_verts, len(pos_2d)):
            world_p = vec_fn((float(pos_2d[i][0]), float(pos_2d[i][1]), float(pos_2d[i][2])))
            screen_co = v3d.location_3d_to_region_2d(region, rv3d, world_p)
            if screen_co:
                dist_sq = (screen_co.x - mouse_pos[0]) ** 2 + (screen_co.y - mouse_pos[1]) ** 2
                if dist_sq < best_curve_dist_sq:
                    cam_dist = _vec_len(world_p - origin)
                    if cam_dist <= hit_world_dist + 0.03:
                        best_curve_dist_sq = dist_sq
                        best_curve_idx = i
        if best_curve_idx is not None:
            best_idx = best_curve_idx

    # 2. 面に直接ヒットしなかった場合のフォールバック（画面50px以内最近傍探索）
    if best_idx is None:
        best_idx = _pick_nearest_screen_vertex(
            pos_2d,
            region=region,
            rv3d=rv3d,
            mouse_pos=mouse_pos,
            max_dist_px=50.0,
            view3d_mod=v3d,
            vector_cls=vec_fn,
        )

    return best_idx, origin, direction


@allow_positional
def resolve_brush_center(context, obj, coords, mouse_pos):
    """ブラシ中心 (ワールド) とレイ情報の解決。レイ外れ時は最近傍頂点 (50px以内) にフォールバックする"""
    region = context.region
    rv3d = context.region_data
    if not (region and rv3d):
        return None
    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_pos)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_pos)
    if not (origin and direction):
        return None
    center_world = None
    hit_t = None
    try:
        matrix_inv = obj.matrix_world.inverted()
        ray_o = matrix_inv @ origin
        ray_d = (matrix_inv.to_3x3() @ direction).normalized()
        hit, hit_loc, _, _ = obj.ray_cast(ray_o, ray_d)
        if hit:
            hit_world = obj.matrix_world @ mathutils.Vector(hit_loc)
            center_world = np.array([hit_world.x, hit_world.y, hit_world.z], dtype=np.float32)
            hit_t = float((hit_world - origin).dot(direction))
    except Exception:
        center_world = None
    pos_2d = np.asarray(coords, dtype=np.float32).reshape((-1, 3))
    if center_world is None:
        try:
            best_idx = _pick_nearest_screen_vertex(
                pos_2d,
                region=region,
                rv3d=rv3d,
                mouse_pos=mouse_pos,
                max_dist_px=50.0,
            )
            if best_idx is None:
                return None
            center_world = pos_2d[best_idx]
            try:
                best_vec = mathutils.Vector((float(center_world[0]), float(center_world[1]), float(center_world[2])))
                cand_t = (best_vec - origin).dot(direction)
                if isinstance(cand_t, (int, float)):
                    if float(cand_t) <= 0.0:
                        return None
                    hit_t = float(cand_t)
            except Exception:
                hit_t = None
        except Exception:
            return None
    return {
        "center": center_world,
        "origin": origin,
        "direction": direction,
        "hit_t": hit_t,
        "pos_2d": pos_2d,
    }


@allow_positional
def update_brush_circle(context, obj, center, radius, mouse_pos):
    """ブラシカーソル円を更新する"""
    region = context.region
    rv3d = context.region_data
    try:
        _center_world = mathutils.Vector(np.asarray(center).tolist())
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


def nudge_brush_radius(obj, factor, *, mouse_pos):
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

    def enter(self, mode, start_x, *, brush):
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
