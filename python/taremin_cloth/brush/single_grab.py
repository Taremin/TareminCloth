"""
単一グラブブラシ (taremin_cloth.brush.single_grab)
従来の1頂点掴み・ドラッグ操作。BaseBrushTool 実装の参照例でもある。
"""

import mathutils
from bpy_extras import view3d_utils

from ..utils import drawing
from .base import BaseBrushTool


class SingleGrabTool(BaseBrushTool):
    """1頂点を掴んでビュー平面上でドラッグする"""

    name = 'GRAB'

    def __init__(self):
        self.reset()

    def reset(self):
        self.grabbed_vert = None
        self.grab_initial_pos_local = None
        self.grab_plane_point = None
        self.grab_plane_normal = None
        self.grab_initial_plane_hit = None
        self.grab_current_target_local = None

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
                world_p = obj.matrix_world @ mathutils.Vector(p)
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
        v_initial_local = mathutils.Vector((p[0], p[1], p[2]))
        v_initial_world = obj.matrix_world @ v_initial_local

        # カメラの視線前方向ベクトルを取得（ビュー平面の法線）
        view_inv = rv3d.view_matrix.inverted()
        camera_forward = -view_inv.to_3x3().col[2].normalized()

        # ビュー平面の通過点は「選択された頂点自身のワールド位置」
        self.grab_plane_point = v_initial_world
        self.grab_plane_normal = camera_forward
        self.grab_initial_pos_local = v_initial_local
        self.grab_current_target_local = v_initial_local

        # クリック時のレイとビュー平面との初期交点を計算・記録
        denom = direction.dot(camera_forward)
        if abs(denom) > 1e-6:
            t0 = (v_initial_world - origin).dot(camera_forward) / denom
            self.grab_initial_plane_hit = origin + direction * t0
        else:
            self.grab_initial_plane_hit = v_initial_world

        sim.set_pin(best_idx, [p[0], p[1], p[2]], 1.0)
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
                or self.grab_initial_plane_hit is None or self.grab_initial_pos_local is None):
            return False
        obj, sim = ctx.obj, ctx.sim
        mouse_pos = ctx.mouse_pos
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, mouse_pos)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, mouse_pos)
        if not (origin and direction):
            return False
        denom = direction.dot(self.grab_plane_normal)
        if abs(denom) <= 1e-6:
            return False
        # 現在のマウスレイとビュー平面の交点を計算
        t = (self.grab_plane_point - origin).dot(self.grab_plane_normal) / denom
        current_plane_hit = origin + direction * t

        # ビュー平面上のワールド移動差分（オフセット）
        delta_world = current_plane_hit - self.grab_initial_plane_hit

        # オブジェクトローカル空間の移動差分に変換
        matrix_inv = obj.matrix_world.inverted()
        delta_local = matrix_inv.to_3x3() @ delta_world

        # 初期のローカル頂点位置にオフセットを加算
        target_pos_local = self.grab_initial_pos_local + delta_local
        self.grab_current_target_local = target_pos_local
        sim.set_pin(self.grabbed_vert, [target_pos_local.x, target_pos_local.y, target_pos_local.z], 1.0)
        target_world = obj.matrix_world @ target_pos_local
        drawing.set_active_grabbed_vertex(obj.name, self.grabbed_vert, target_world)
        return True

    def on_release(self, ctx, pinned_verts):
        if self.grabbed_vert is None:
            return False
        # ピン留めされていない頂点のみ物理解放
        if self.grabbed_vert not in pinned_verts:
            try:
                ctx.sim.release_pin(self.grabbed_vert)
            except Exception:
                pass
        drawing.clear_active_grabbed_vertex()
        self.reset()
        return True

    def abort(self, sim, pinned_verts):
        if self.grabbed_vert is not None and self.grabbed_vert not in pinned_verts:
            try:
                sim.release_pin(self.grabbed_vert)
            except Exception:
                pass
        self.reset()
