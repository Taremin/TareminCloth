# -*- coding: utf-8 -*-
"""型紙エディタのBlender GPU/BLF描画実行器 (canvas.py)."""

from __future__ import annotations

import math
from typing import Sequence

try:
    import blf
    import gpu
    from gpu_extras.batch import batch_for_shader
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False
    gpu = None
    blf = None
    batch_for_shader = None

from .editor import EditorController
from .geometry import triangulate_polygon_2d
from .placement import (
    compute_bone_aligned_frame,
    compute_tubular_waypoints_for_pair,
    project_2d_points_to_bone_plane,
)
from .renderer import DrawCommandKind, build_render_commands
from ..utils.view3d import get_viewport_overlay_offsets


class PatternCanvasRenderer:
    """Blender 3DビューポートのPOST_PIXELおよびPOST_VIEWコールバックで描画する実行器."""

    def __init__(self, controller: EditorController) -> None:
        self.controller: EditorController = controller
        self._handler = None
        self._handler_3d = None
        self._shader_2d = None
        self.all_bones_data: list = []

    def _get_overlay_bones(self) -> list:
        """コントローラーのボーン辞書または直接指定リストから3D描画用ボーンタプル列を返す."""
        if self.controller.bones:
            return [b.to_overlay_tuple() for b in self.controller.bones.values()]
        return self.all_bones_data

    def _get_shader(self):
        if self._shader_2d is None and _HAS_GPU:
            # Blender 4.0+ / 5.x 共通組み込みシェーダー
            self._shader_2d = gpu.shader.from_builtin('UNIFORM_COLOR')
        return self._shader_2d

    def register_handler(self) -> None:
        """3DビューポートのPOST_PIXELおよびPOST_VIEW描画ハンドラーを登録する."""
        if not _HAS_GPU:
            return
        import bpy
        if self._handler is None:
            self._handler = bpy.types.SpaceView3D.draw_handler_add(
                self.draw_callback,
                (bpy.context,),
                'WINDOW',
                'POST_PIXEL',
            )
        if self._handler_3d is None:
            self._handler_3d = bpy.types.SpaceView3D.draw_handler_add(
                self.draw_callback_3d,
                (bpy.context,),
                'WINDOW',
                'POST_VIEW',
            )

    def unregister_handler(self) -> None:
        """3Dビューポートの描画ハンドラーを解除する."""
        if not _HAS_GPU:
            return
        import bpy
        if self._handler is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handler, 'WINDOW')
            except Exception:
                pass
            self._handler = None
        if self._handler_3d is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handler_3d, 'WINDOW')
            except Exception:
                pass
            self._handler_3d = None

    def draw_callback(self, context) -> None:
        """ビューポート再描画時に呼び出されるメイン描画処理."""
        if not _HAS_GPU:
            return

        region = getattr(context, "region", None)
        if not region:
            return

        # 初回または未設定時のみビューポート内パネルのオーバーレイ幅・高さを検出
        if self.controller.top_offset == 0.0 and self.controller.bottom_offset == 0.0:
            area = getattr(context, "area", None)
            lo, ro, to, bo = get_viewport_overlay_offsets(area=area)
            self.controller.left_offset = lo
            self.controller.right_offset = ro
            self.controller.top_offset = to
            self.controller.bottom_offset = bo

        screen_size = (region.width, region.height)
        commands = build_render_commands(
            self.controller.doc,
            self.controller,
            screen_size=screen_size,
        )

        shader = self._get_shader()
        if shader is None:
            return

        gpu.state.blend_set('ALPHA')

        for cmd in commands:
            if cmd.kind == DrawCommandKind.LINE:
                self._draw_lines(shader, cmd.points, color=cmd.color, width=cmd.width)
            elif cmd.kind == DrawCommandKind.POLYLINE:
                self._draw_polyline(shader, cmd.points, color=cmd.color, width=cmd.width)
            elif cmd.kind == DrawCommandKind.POLYGON:
                self._draw_polygon(shader, cmd.points, color=cmd.color)
            elif cmd.kind == DrawCommandKind.RECT:
                self._draw_rect(shader, cmd.points, color=cmd.color, filled=cmd.filled)
            elif cmd.kind == DrawCommandKind.CIRCLE:
                self._draw_circle(shader, cmd.points[0], radius=cmd.radius, color=cmd.color, filled=cmd.filled)
            elif cmd.kind == DrawCommandKind.TEXT:
                self._draw_text(cmd.points[0], cmd.text, color=cmd.color, font_size=cmd.font_size)

        gpu.state.blend_set('NONE')

    def _draw_lines(self, shader, points: Sequence[tuple[float, float]], *, color, width: float) -> None:
        if len(points) < 2:
            return
        gpu.state.line_width_set(width)
        shader.bind()
        shader.uniform_float("color", color)
        batch = batch_for_shader(shader, 'LINES', {"pos": points})
        batch.draw(shader)
        gpu.state.line_width_set(1.0)

    def _draw_polyline(self, shader, points: Sequence[tuple[float, float]], *, color, width: float) -> None:
        if len(points) < 2:
            return
        gpu.state.line_width_set(width)
        shader.bind()
        shader.uniform_float("color", color)
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": points})
        batch.draw(shader)
        gpu.state.line_width_set(1.0)

    def _draw_polygon(self, shader, points: Sequence[tuple[float, float]], *, color) -> None:
        if len(points) < 3:
            return
        # 単純なファン分割で三角形化
        tri_pts = []
        p0 = points[0]
        for i in range(1, len(points) - 1):
            tri_pts.extend([p0, points[i], points[i + 1]])

        shader.bind()
        shader.uniform_float("color", color)
        batch = batch_for_shader(shader, 'TRIS', {"pos": tri_pts})
        batch.draw(shader)

    def _draw_rect(self, shader, points: Sequence[tuple[float, float]], *, color, filled: bool) -> None:
        if len(points) < 2:
            return
        x0, y0 = points[0]
        x1, y1 = points[1]
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        if filled:
            self._draw_polygon(shader, pts, color=color)
        else:
            self._draw_polyline(shader, pts + [(x0, y0)], color=color, width=1.0)

    def _draw_circle(self, shader, center: tuple[float, float], *, radius: float, color, filled: bool) -> None:
        cx, cy = center
        n_segments = 24
        pts = []
        for i in range(n_segments):
            theta = 2.0 * math.pi * i / n_segments
            pts.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))

        if filled:
            self._draw_polygon(shader, pts, color=color)
        else:
            self._draw_polyline(shader, pts + [pts[0]], color=color, width=1.0)

    def _draw_text(self, pos: tuple[float, float], text: str, *, color, font_size: int) -> None:
        if not blf:
            return
        font_id = 0
        blf.size(font_id, font_size)
        blf.color(font_id, *color)
        blf.position(font_id, pos[0], pos[1], 0.0)
        blf.draw(font_id, text)

    def draw_callback_3d(self, context) -> None:
        """3D空間上のリアルタイム配置プレビューおよびPick Boneボーンオーバーレイを描画する."""
        if not _HAS_GPU:
            return

        overlay_bones = self._get_overlay_bones()
        if not overlay_bones:
            return

        from ..utils.drawing import (
            draw_wrinkle_armature_overlay,
            get_3d_uniform_color_shader,
            get_line_shader,
        )

        # 1. Pick Bone モード中のボーン八面体オーバーレイ (X-Ray)
        if self.controller.picking_bone:
            draw_wrinkle_armature_overlay(
                overlay_bones,
                active_bone_name=self.controller.hovered_bone,
            )

        # 2. 選択中ピースの3D配置リアルタイムプレビュー
        if len(self.controller.selection.piece_ids) == 1:
            pid = next(iter(self.controller.selection.piece_ids))
            piece = self.controller.doc.pieces.get(pid)
            pl = self.controller.doc.placements.get(pid)
            if piece and pl and pl.target_bone:
                target_item = next((b for b in overlay_bones if b[0] == pl.target_bone), None)
                if target_item:
                    # Pick Bone モード中でない場合でも、追従対象のボーン八面体をX-Ray表示
                    if not self.controller.picking_bone:
                        draw_wrinkle_armature_overlay(
                            [target_item],
                            active_bone_name=pl.target_bone,
                        )

                    head_3d = (float(target_item[1][0]), float(target_item[1][1]), float(target_item[1][2]))
                    tail_3d = (float(target_item[2][0]), float(target_item[2][1]), float(target_item[2][2]))

                    plane_center, _, _, _, axis_origin = compute_bone_aligned_frame(
                        head_3d,
                        tail_3d,
                        bone_t=pl.bone_t,
                        bone_angle_deg=pl.bone_angle_deg,
                        surface_offset=pl.surface_offset,
                    )

                    shader = get_3d_uniform_color_shader() or get_line_shader()
                    if shader:
                        orig_blend = gpu.state.blend_get()
                        orig_depth_test = gpu.state.depth_test_get()
                        gpu.state.blend_set('ALPHA')
                        gpu.state.depth_test_set('NONE')

                        try:
                            # 2.1 ボーン主軸線 (head -> tail)
                            gpu.state.line_width_set(2.5)
                            shader.bind()
                            shader.uniform_float("color", (0.2, 0.85, 1.0, 0.9))
                            batch_axis = batch_for_shader(
                                shader, 'LINES',
                                {"pos": [head_3d, tail_3d]}
                            )
                            batch_axis.draw(shader)

                            # 2.2 ボーン位置マーカー (ボーン軸上の基準点 axis_origin から plane_center への法線オフセット線)
                            pt_axis = (float(axis_origin[0]), float(axis_origin[1]), float(axis_origin[2]))
                            pt_center = (float(plane_center[0]), float(plane_center[1]), float(plane_center[2]))
                            shader.uniform_float("color", (1.0, 0.9, 0.2, 0.95))
                            gpu.state.line_width_set(2.5)
                            batch_off = batch_for_shader(
                                shader, 'LINES',
                                {"pos": [pt_axis, pt_center]}
                            )
                            batch_off.draw(shader)

                            # 2.3 型紙ピースの 3D 平面メッシュプレビュー
                            outer = piece.paths.get(piece.outer_path_id)
                            poly_2d = (
                                [piece.transform.apply(pt) for pt in outer.sample_outline_points(steps_per_segment=16)]
                                if outer else []
                            )
                            if len(poly_2d) >= 3:
                                world_3d = project_2d_points_to_bone_plane(
                                    poly_2d,
                                    head_3d=head_3d,
                                    tail_3d=tail_3d,
                                    bone_t=pl.bone_t,
                                    bone_angle_deg=pl.bone_angle_deg,
                                    surface_offset=pl.surface_offset,
                                )
                                poly_3d = [(float(p[0]), float(p[1]), float(p[2])) for p in world_3d]

                                # 耳刈り法三角形化による半透明面描画
                                tri_indices = triangulate_polygon_2d(poly_2d)
                                tri_pts_3d = []
                                for i0, i1, i2 in tri_indices:
                                    tri_pts_3d.extend([poly_3d[i0], poly_3d[i1], poly_3d[i2]])

                                if tri_pts_3d:
                                    shader.uniform_float("color", (0.2, 0.75, 1.0, 0.35))
                                    batch_poly = batch_for_shader(
                                        shader, 'TRIS',
                                        {"pos": tri_pts_3d}
                                    )
                                    batch_poly.draw(shader)

                                # 外枠線描画
                                gpu.state.line_width_set(2.2)
                                shader.uniform_float("color", (0.3, 0.9, 1.0, 0.95))
                                line_loop = poly_3d + [poly_3d[0]]
                                batch_outline = batch_for_shader(
                                    shader, 'LINE_STRIP',
                                    {"pos": line_loop}
                                )
                                batch_outline.draw(shader)

                                # 2.4 筒モード（is_tubular）の裏側迂回ルートプレビュー
                                if pl.is_tubular:
                                    min_x_idx = min(range(len(poly_2d)), key=lambda i: poly_2d[i][0])
                                    max_x_idx = max(range(len(poly_2d)), key=lambda i: poly_2d[i][0])
                                    v_min_3d = poly_3d[min_x_idx]
                                    v_max_3d = poly_3d[max_x_idx]

                                    w1, w2 = compute_tubular_waypoints_for_pair(
                                        v_min_3d,
                                        v_max_3d,
                                        head_3d=head_3d,
                                        tail_3d=tail_3d,
                                        bone_t=pl.bone_t,
                                        bone_angle_deg=pl.bone_angle_deg,
                                        surface_offset=pl.surface_offset,
                                    )

                                    gpu.state.line_width_set(2.5)
                                    shader.uniform_float("color", (1.0, 0.6, 0.15, 0.95))
                                    loop_pts = [
                                        v_min_3d,
                                        (float(w1[0]), float(w1[1]), float(w1[2])),
                                        (float(w2[0]), float(w2[1]), float(w2[2])),
                                        v_max_3d,
                                    ]
                                    batch_tube = batch_for_shader(
                                        shader, 'LINE_STRIP',
                                        {"pos": loop_pts}
                                    )
                                    batch_tube.draw(shader)

                            gpu.state.line_width_set(1.0)
                        finally:
                            gpu.state.blend_set(orig_blend)
                            gpu.state.depth_test_set(orig_depth_test)
