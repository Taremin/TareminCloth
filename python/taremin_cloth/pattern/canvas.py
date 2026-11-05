# -*- coding: utf-8 -*-
"""型紙エディタのBlender GPU/BLF描画実行器 (canvas.py)."""

from __future__ import annotations

import math
from typing import Optional, Sequence

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
from .renderer import DrawCommand, DrawCommandKind, build_render_commands


class PatternCanvasRenderer:
    """Blender 3DビューポートのPOST_PIXELコールバックでDrawCommand列を描画する実行器."""

    def __init__(self, controller: EditorController) -> None:
        self.controller: EditorController = controller
        self._handler = None
        self._shader_2d = None

    def _get_shader(self):
        if self._shader_2d is None and _HAS_GPU:
            # Blender 4.0+ / 5.x 共通組み込みシェーダー
            self._shader_2d = gpu.shader.from_builtin('UNIFORM_COLOR')
        return self._shader_2d

    def register_handler(self) -> None:
        """3DビューポートのPOST_PIXEL描画ハンドラーを登録する."""
        if not _HAS_GPU:
            return
        if self._handler is None:
            import bpy
            self._handler = bpy.types.SpaceView3D.draw_handler_add(
                self.draw_callback,
                (bpy.context,),
                'WINDOW',
                'POST_PIXEL',
            )

    def unregister_handler(self) -> None:
        """3Dビューポートの描画ハンドラーを解除する."""
        if self._handler is not None and _HAS_GPU:
            import bpy
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handler, 'WINDOW')
            except Exception:
                pass
            self._handler = None

    def draw_callback(self, context) -> None:
        """ビューポート再描画時に呼び出されるメイン描画処理."""
        if not _HAS_GPU:
            return

        region = getattr(context, "region", None)
        if not region:
            return

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
