"""
単一グラブブラシ (taremin_cloth.brush.single_grab)
従来の1頂点掴み・ドラッグ操作。BaseBrushTool 実装の参照例でもある。
"""

import mathutils
from bpy_extras import view3d_utils

from ..utils import drawing
from .base import BaseBrushTool
from .math import ray_plane_hit


class SingleGrabTool(BaseBrushTool):
    """1頂点を掴んでビュー平面上でドラッグする"""

    name = 'GRAB'
    label_key = 'Grab'

    def __init__(self):
        self.reset()

    def reset(self):
        self.grabbed_vert = None
        self.grabbed_cluster = []
        self.grab_initial_positions = {}
        self.grab_initial_pos_world = None
        self.grab_plane_point = None
        self.grab_plane_normal = None
        self.grab_initial_plane_hit = None
        self.grab_current_target_world = None

    @property
    def dragging(self):
        return self.grabbed_vert is not None

    def on_press(self, ctx):
        region = ctx.context.region
        rv3d = ctx.context.region_data
        if not (region and rv3d):
            return False
        obj, sim = ctx.obj, ctx.sim
        mouse_pos = ctx.mouse_pos
        pos_2d = ctx.coords.reshape((-1, 3))
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_pos)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_pos)

        best_idx = None

        # 1. 視線レイキャストによる最前面ポリゴンの交差判定
        if origin and direction:
            matrix_world = obj.matrix_world
            matrix_inv = matrix_world.inverted()
            ray_origin_local = matrix_inv @ origin
            ray_dir_local = (matrix_inv.to_3x3() @ direction).normalized()

            hit, hit_loc, hit_normal, face_idx = obj.ray_cast(ray_origin_local, ray_dir_local)
            if hit and face_idx is not None and 0 <= face_idx < len(obj.data.polygons):
                polygon = obj.data.polygons[face_idx]
                min_dist_sq = float('inf')
                # ヒットしたポリゴンの構成頂点から交点に最も近い頂点を選択
                for v_idx in polygon.vertices:
                    v_co = obj.data.vertices[v_idx].co
                    dist_sq = (v_co - hit_loc).length_squared
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
                        best_idx = v_idx

        # 2. 面に直接ヒットしなかった場合のフォールバック（画面最近傍探索）
        if best_idx is None:
            best_dist = float('inf')
            for i, p in enumerate(pos_2d):
                world_p = mathutils.Vector(p)
                screen_co = view3d_utils.location_3d_to_region_2d(region, rv3d, world_p)
                if screen_co:
                    dist = (screen_co.x - mouse_pos[0]) ** 2 + (screen_co.y - mouse_pos[1]) ** 2
                    if dist < best_dist and dist < 2500:
                        best_dist = dist
                        best_idx = i

        if best_idx is None or not (origin and direction):
            return False

        self.grabbed_vert = best_idx
        p = pos_2d[best_idx]
        v_initial_world = mathutils.Vector((p[0], p[1], p[2]))

        # カメラの視線前方向ベクトルを取得（ビュー平面の法線）
        view_inv = rv3d.view_matrix.inverted()
        camera_forward = -view_inv.to_3x3().col[2].normalized()

        # ビュー平面の通過点は「選択された頂点自身のワールド位置」
        self.grab_plane_point = v_initial_world
        self.grab_plane_normal = camera_forward
        self.grab_initial_pos_world = v_initial_world
        self.grab_current_target_world = v_initial_world

        # クリック時のレイとビュー平面との初期交点を計算・記録
        denom = direction.dot(camera_forward)
        if abs(denom) > 1e-6:
            t0 = (v_initial_world - origin).dot(camera_forward) / denom
            self.grab_initial_plane_hit = origin + direction * t0
        else:
            self.grab_initial_plane_hit = v_initial_world

        # 縫合クラスタ（対向端点や中継点）の同期ピン留め:
        # 未結合で離れているエッジは単独操作（手動引き寄せ）を可能にし、結合済みシームのみ一体同期する
        from ..engine.runner import get_connected_seam_partners, get_seam_merge_threshold
        threshold = get_seam_merge_threshold(obj)
        partners = get_connected_seam_partners(obj.name, best_idx, pos_2d, threshold)
        all_grabbed = [best_idx] + [p for p in partners if p < len(pos_2d) and p != best_idx]
        self.grabbed_cluster = all_grabbed

        self.grab_initial_positions = {}
        for v in all_grabbed:
            pv = pos_2d[v]
            init_v = mathutils.Vector((pv[0], pv[1], pv[2]))
            self.grab_initial_positions[v] = init_v
            sim.set_pin(v, [pv[0], pv[1], pv[2]], 1.0)

        drawing.set_active_grabbed_vertex(obj.name, best_idx, v_initial_world)
        return True

    def on_move(self, ctx):
        if self.grabbed_vert is None:
            return False
        region = ctx.context.region
        rv3d = ctx.context.region_data
        if not (region and rv3d):
            return False
        if (self.grab_plane_point is None or self.grab_plane_normal is None
                or self.grab_initial_plane_hit is None or getattr(self, "grab_initial_pos_world", None) is None):
            return False
        obj, sim = ctx.obj, ctx.sim
        mouse_pos = ctx.mouse_pos
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_pos)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_pos)
        if not (origin and direction):
            return False
        # 現在のマウスレイとビュー平面の交点を計算 (実演算はRust核)
        try:
            hit = ray_plane_hit(
                (origin.x, origin.y, origin.z),
                (direction.x, direction.y, direction.z),
                (self.grab_plane_point.x, self.grab_plane_point.y, self.grab_plane_point.z),
                (self.grab_plane_normal.x, self.grab_plane_normal.y, self.grab_plane_normal.z))
        except Exception:
            return False
        if hit is None:
            return False
        current_plane_hit = mathutils.Vector((float(hit[0]), float(hit[1]), float(hit[2])))

        # ビュー平面上のワールド移動差分（オフセット）
        delta_world = current_plane_hit - self.grab_initial_plane_hit

        # 目標のワールド頂点位置にオフセットを加算
        target_world = self.grab_initial_pos_world + delta_world
        self.grab_current_target_world = target_world

        # クラスタ内の全頂点に同一オフセットを適用してピン位置更新
        for v in getattr(self, "grabbed_cluster", [self.grabbed_vert]):
            init_v = self.grab_initial_positions.get(v, self.grab_initial_pos_world)
            t_v = init_v + delta_world
            sim.set_pin(v, [t_v.x, t_v.y, t_v.z], 1.0)

        drawing.set_active_grabbed_vertex(obj.name, self.grabbed_vert, target_world)
        return True

    def on_release(self, ctx, pinned_verts):
        if self.grabbed_vert is None:
            return False
        # ピン留めされていない頂点のみ物理解放
        for v in getattr(self, "grabbed_cluster", [self.grabbed_vert]):
            if v not in pinned_verts:
                try:
                    ctx.sim.release_pin(v)
                except Exception:
                    pass
        drawing.clear_active_grabbed_vertex()
        self.reset()
        return True

    def abort(self, sim, pinned_verts):
        for v in getattr(self, "grabbed_cluster", []):
            if v not in pinned_verts:
                try:
                    sim.release_pin(v)
                except Exception:
                    pass
        self.reset()
