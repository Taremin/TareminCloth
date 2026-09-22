"""
平滑化ブラシ (taremin_cloth.brush.smooth)
ドラッグ中だけブラシ内の頂点を1-ring平均へソフトピンで寄せ、
離したら物理に戻す一時シワ伸ばしブラシ。RangeGrabと同一ライフサイクル。
"""

import mathutils
import numpy as np

from .base import BaseBrushTool, resolve_brush_center, update_brush_circle, brush_opt
from .math import (
    TENSION_EXPAND,
    verts_in_brush,
    depth_keep_mask,
    select_grab_pins,
    build_adjacency_csr,
    laplacian_smooth_targets,
    radial_expand_targets,
)

try:
    import taremin_cloth_core as _core

    _RUST_BRUSH = all(
        hasattr(_core, n)
        for n in (
            "brush_build_adjacency_csr",
            "brush_smooth_targets",
            "brush_radial_expand_targets",
        )
    )
except Exception:
    _core = None
    _RUST_BRUSH = False


def _pinned_set_of(ctx, fallback=None):
    """コンテキストからピン留め集合を防御的に取得する"""
    if fallback is not None:
        try:
            return set(fallback)
        except TypeError:
            return set()
    try:
        op = getattr(ctx, "op", None)
        pinned = getattr(op, "_pinned_verts", None)
        if pinned is None:
            return set()
        return set(pinned)
    except Exception:
        return set()


class SmoothTool(BaseBrushTool):
    """ブラシ円内を1反復ラプラシアン平滑化ターゲットへ一時ピン留めする"""

    name = 'SMOOTH'

    def __init__(self):
        self.reset()

    def reset(self):
        self.dragging = False
        self.owned = {}
        self.adj_offsets = None
        self.adj_indices = None
        self.adj_num_verts = 0

    def on_press(self, ctx):
        self.dragging = True
        self._ensure_adjacency(ctx)
        self._dab(ctx, _pinned_set_of(ctx))
        return True

    def on_move(self, ctx):
        if not self.dragging:
            return False
        self._dab(ctx, _pinned_set_of(ctx))
        try:
            radius = max(float(brush_opt(ctx.brush, "radius", 0.05)), 1e-4)
            res = resolve_brush_center(ctx.context, ctx.obj, ctx.coords, ctx.mouse_pos)
            if res is not None:
                update_brush_circle(ctx.context, ctx.obj, res["center"], radius, ctx.mouse_pos)
        except Exception:
            pass
        return True

    def on_release(self, ctx, pinned_verts):
        if not self.dragging and not self.owned:
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
        if self.owned:
            for v_idx in list(self.owned.keys()):
                if v_idx not in (pinned_verts or set()):
                    try:
                        sim.release_pin(v_idx)
                    except Exception:
                        pass
        self.owned = {}

    def _ensure_adjacency(self, ctx):
        try:
            n_verts = int(np.asarray(ctx.coords).size // 3)
        except Exception:
            return False
        if (self.adj_offsets is not None and self.adj_indices is not None
                and self.adj_num_verts == n_verts):
            return True
        try:
            from ..utils.mesh_extract import extract_cloth_mesh_data

            data = extract_cloth_mesh_data(ctx.obj)
            edges = np.ascontiguousarray(data.normal_edges, dtype=np.uint32).reshape(-1, 2)
            if len(edges) == 0 or n_verts <= 0:
                return False
            # sim頂点順とメッシュ頂点順の一致を前提とする (不一致時は再抽出不能なので無効化)
            if int(np.max(edges)) >= n_verts:
                return False
            if _RUST_BRUSH:
                try:
                    offsets, indices = _core.brush_build_adjacency_csr(n_verts, edges)
                    offsets = np.ascontiguousarray(offsets, dtype=np.uint32)
                    indices = np.ascontiguousarray(indices, dtype=np.uint32)
                except Exception:
                    offsets, indices = build_adjacency_csr(n_verts, edges)
            else:
                offsets, indices = build_adjacency_csr(n_verts, edges)
            self.adj_offsets = offsets
            self.adj_indices = indices
            self.adj_num_verts = n_verts
            return True
        except Exception:
            self.adj_offsets = None
            self.adj_indices = None
            self.adj_num_verts = 0
            return False

    def _dab(self, ctx, pinned_verts):
        obj, sim = ctx.obj, ctx.sim
        if obj is None or sim is None:
            return False
        if self.adj_offsets is None or self.adj_indices is None:
            return False
        radius = max(float(brush_opt(ctx.brush, "radius", 0.05)), 1e-4)
        shape = brush_opt(ctx.brush, "falloff_shape", 'SMOOTH')
        strength = float(brush_opt(ctx.brush, "strength", 0.8))
        res = resolve_brush_center(ctx.context, obj, ctx.coords, ctx.mouse_pos)
        if res is None:
            return False
        pos_2d = res["pos_2d"]
        found, weights = verts_in_brush(pos_2d, res["center"], radius, shape)
        if len(found) > 0 and res["hit_t"] is not None:
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
        pin_idx, pin_w = select_grab_pins(found, weights)
        if not pin_idx:
            update_brush_circle(ctx.context, obj, res["center"], radius, ctx.mouse_pos)
            return True
        pinned = set(pinned_verts) if pinned_verts else set()
        live_idx, live_w = [], []
        for v_idx, w in zip(pin_idx, pin_w):
            if int(v_idx) not in pinned:
                live_idx.append(int(v_idx))
                live_w.append(float(w))
        if not live_idx:
            update_brush_circle(ctx.context, obj, res["center"], radius, ctx.mouse_pos)
            return True
        try:
            pos_c = np.ascontiguousarray(pos_2d, dtype=np.float32)
            idx_arr = np.ascontiguousarray(live_idx, dtype=np.uint32)
            w_arr = np.ascontiguousarray(live_w, dtype=np.float32)
            off_c = np.ascontiguousarray(self.adj_offsets, dtype=np.uint32)
            nbr_c = np.ascontiguousarray(self.adj_indices, dtype=np.uint32)
            center = np.ascontiguousarray(res["center"], dtype=np.float32).reshape(3)
            if _RUST_BRUSH:
                try:
                    targets = np.asarray(
                        _core.brush_smooth_targets(
                            pos_c, off_c, nbr_c, idx_arr, w_arr, strength, list(pinned)),
                        dtype=np.float32)
                except Exception:
                    targets = laplacian_smooth_targets(
                        pos_c, off_c, nbr_c, idx_arr, w_arr, strength, exclude=pinned)
            else:
                targets = laplacian_smooth_targets(
                    pos_c, off_c, nbr_c, idx_arr, w_arr, strength, exclude=pinned)
            # 平滑化だけでは余剰長が残りリリース後に再座屈するため、
            # 放射方向への弱い拡張で弛みをパッチ外へ逃がす (張力アシスト)。
            smoothed = pos_c.copy()
            smoothed[idx_arr.astype(np.int64)] = targets
            if _RUST_BRUSH:
                try:
                    targets = np.asarray(
                        _core.brush_radial_expand_targets(
                            smoothed, center, idx_arr, w_arr, strength, TENSION_EXPAND),
                        dtype=np.float32)
                except Exception:
                    targets = radial_expand_targets(
                        smoothed, center, idx_arr, w_arr, strength)
            else:
                targets = radial_expand_targets(
                    smoothed, center, idx_arr, w_arr, strength)
        except Exception:
            return False
        if hasattr(sim, "set_pins_batch"):
            try:
                sim.set_pins_batch(
                    [int(v) for v in live_idx],
                    np.ascontiguousarray(targets, dtype=np.float32),
                    [float(w) for w in live_w])
                for v_idx, w in zip(live_idx, live_w):
                    self.owned[int(v_idx)] = float(w)
            except Exception:
                return False
        else:
            for v_idx, w, t in zip(live_idx, live_w, targets.tolist()):
                try:
                    sim.set_pin(int(v_idx), [float(t[0]), float(t[1]), float(t[2])], float(w))
                    self.owned[int(v_idx)] = float(w)
                except Exception:
                    pass
        update_brush_circle(ctx.context, obj, res["center"], radius, ctx.mouse_pos)
        return True

    def _release(self, ctx, pinned_verts):
        if self.owned:
            pinned = set(pinned_verts) if pinned_verts else set()
            for v_idx in list(self.owned.keys()):
                if v_idx not in pinned:
                    try:
                        ctx.sim.release_pin(v_idx)
                    except Exception:
                        pass
        self.owned = {}
