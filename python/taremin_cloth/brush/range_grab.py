"""
範囲グラブブラシ (taremin_cloth.brush.range_grab)
ブラシ球内の頂点を減衰付きで掴み、ビュー平面上でドラッグする。
"""

import mathutils
import numpy as np

from .base import BaseBrushTool, resolve_brush_center, update_brush_circle, brush_opt
from .math import verts_in_brush, depth_keep_mask, select_grab_pins


class RangeGrabTool(BaseBrushTool):
    """ブラシ半径内の頂点群を減衰重みでドラッグする"""

    name = 'RANGE_GRAB'

    def __init__(self):
        self.reset()

    def reset(self):
        self.dragging = False
        self.grab = None
        self.plane_point = None
        self.plane_normal = None
        self.initial_plane_hit = None

    def on_press(self, ctx):
        self.dragging = True
        self._start(ctx)
        return True

    def on_move(self, ctx):
        if not self.dragging:
            return False
        self._move(ctx)
        try:
            radius = max(float(brush_opt(ctx.brush, "radius", 0.05)), 1e-4)
            res = resolve_brush_center(ctx.context, ctx.obj, ctx.coords, ctx.mouse_pos)
            if res is not None:
                update_brush_circle(ctx.context, ctx.obj, res["center"], radius, ctx.mouse_pos)
        except Exception:
            pass
        return True

    def on_release(self, ctx, pinned_verts):
        if not self.dragging:
            return False
        self.dragging = False
        self._release(ctx, pinned_verts)
        return True

    def on_hover(self, ctx):
        brush = ctx.brush
        if brush is None:
            return False
        from ..utils import drawing

        prev = drawing.get_brush_info()
        if prev is not None:
            drawing.set_brush_info(ctx.mouse_pos[0], ctx.mouse_pos[1],
                                   prev.get("radius_px", 40.0))
            return True
        # 円未表示 (モード切替直後等) はこの場で初期化する
        try:
            radius = max(float(getattr(brush, "radius", 0.05)), 1e-4)
            res = resolve_brush_center(ctx.context, ctx.obj, ctx.coords, ctx.mouse_pos)
            if res is not None:
                update_brush_circle(ctx.context, ctx.obj, res["center"], radius, ctx.mouse_pos)
                return True
        except Exception:
            pass
        return False

    def on_activate(self, ctx):
        brush = ctx.brush
        if brush is None:
            return
        try:
            radius = max(float(getattr(brush, "radius", 0.05)), 1e-4)
            res = resolve_brush_center(ctx.context, ctx.obj, ctx.coords, ctx.mouse_pos)
            if res is not None:
                update_brush_circle(ctx.context, ctx.obj, res["center"], radius, ctx.mouse_pos)
        except Exception:
            pass

    def on_deactivate(self, ctx, pinned_verts):
        self.dragging = False
        self._release(ctx, pinned_verts)
        from ..utils import drawing

        drawing.clear_brush_info()

    def abort(self, sim, pinned_verts):
        self.dragging = False
        if self.grab:
            for v_idx in list(self.grab.keys()):
                if v_idx not in pinned_verts:
                    try:
                        sim.release_pin(v_idx)
                    except Exception:
                        pass
        self.grab = None
        self.plane_point = None
        self.plane_normal = None
        self.initial_plane_hit = None

    def _start(self, ctx):
        obj, sim = ctx.obj, ctx.sim
        radius = max(float(brush_opt(ctx.brush, "radius", 0.05)), 1e-4)
        shape = brush_opt(ctx.brush, "falloff_shape", 'SMOOTH')
        res = resolve_brush_center(ctx.context, obj, ctx.coords, ctx.mouse_pos)
        if res is None:
            self.grab = {}
            return False
        pos_2d = res["pos_2d"]
        found, weights = verts_in_brush(pos_2d, res["center"], radius, shape)
        if len(found) > 0 and res["hit_t"] is not None:
            # デプスフィルタ: 命中面より奥 (裏面) の頂点を除外する
            try:
                wpos = np.array(
                    [(obj.matrix_world @ mathutils.Vector(pos_2d[int(i)])).to_tuple()
                     for i in np.asarray(found).tolist()],
                    dtype=np.float32,
                )
                keep = depth_keep_mask(wpos, res["origin"], res["direction"], res["hit_t"], radius * 0.5)
                found = np.asarray(found)[keep]
                weights = np.asarray(weights)[keep]
            except Exception:
                pass
        grab = {}
        pin_idx, pin_w = select_grab_pins(found, weights)
        for v_idx, w in zip(pin_idx, pin_w):
            p = pos_2d[v_idx]
            init = mathutils.Vector((p[0], p[1], p[2]))
            grab[v_idx] = (init, float(w))
            # ウェイトは減衰値そのまま: 中心は完全固定、裾野はソフト追従。
            # 全頂点を 1.0 固定すると裾野が運動学的壁となり周囲へ伝播しない。
            sim.set_pin(v_idx, [init.x, init.y, init.z], float(w))
        self.grab = grab
        # 単一グラブと同一のビュー平面方式でドラッグする
        try:
            region = ctx.context.region
            rv3d = ctx.context.region_data
            view_inv = rv3d.view_matrix.inverted()
            camera_forward = -view_inv.to_3x3().col[2].normalized()
            center_world = obj.matrix_world @ mathutils.Vector(res["center"].tolist())
            self.plane_point = center_world
            self.plane_normal = camera_forward
            origin, direction = res["origin"], res["direction"]
            denom = direction.dot(camera_forward)
            if abs(denom) > 1e-6:
                t0 = (center_world - origin).dot(camera_forward) / denom
                self.initial_plane_hit = origin + direction * t0
            else:
                self.initial_plane_hit = center_world
        except Exception:
            self.plane_point = None
            self.plane_normal = None
            self.initial_plane_hit = None
        update_brush_circle(ctx.context, obj, res["center"], radius, ctx.mouse_pos)
        return True

    def _move(self, ctx):
        from bpy_extras import view3d_utils

        region = ctx.context.region
        rv3d = ctx.context.region_data
        if not (region and rv3d):
            return False
        if not self.grab:
            return True
        if (self.plane_point is None or self.plane_normal is None
                or self.initial_plane_hit is None):
            return False
        obj, sim = ctx.obj, ctx.sim
        strength = float(brush_opt(ctx.brush, "strength", 0.8))
        mouse_pos = ctx.mouse_pos
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_pos)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_pos)
        if not (origin and direction):
            return False
        denom = direction.dot(self.plane_normal)
        if abs(denom) <= 1e-6:
            return False
        t = (self.plane_point - origin).dot(self.plane_normal) / denom
        current_hit = origin + direction * t
        delta_world = current_hit - self.initial_plane_hit
        try:
            matrix_inv = obj.matrix_world.inverted()
            delta_local = matrix_inv.to_3x3() @ delta_world
        except Exception:
            return False
        for v_idx, (init, w) in self.grab.items():
            k = w * strength
            tx = init.x + delta_local.x * k
            ty = init.y + delta_local.y * k
            tz = init.z + delta_local.z * k
            sim.set_pin(v_idx, [tx, ty, tz], w)
        return True

    def _release(self, ctx, pinned_verts):
        """ピン留めされていない頂点を物理へ戻す"""
        if self.grab:
            for v_idx in list(self.grab.keys()):
                if v_idx not in pinned_verts:
                    try:
                        ctx.sim.release_pin(v_idx)
                    except Exception:
                        pass
        self.grab = None
        self.plane_point = None
        self.plane_normal = None
        self.initial_plane_hit = None
