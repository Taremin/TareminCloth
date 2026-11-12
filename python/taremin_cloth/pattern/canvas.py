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

from .build_types import _euler_to_rotation_matrix
from .document import SeamRouting
from .editor import EditorController
from .geometry import triangulate_polygon_2d
from .placement import (
    compute_bone_aligned_frame,
    compute_tubular_waypoints_for_pair,
    project_2d_points_to_bone_plane,
    resolve_seam_bone_wrap_target,
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

    def _build_preview_bone_map(self, overlay_bones: list) -> dict:
        """3Dプレビュー用のボーン名 -> (head_3d, tail_3d を持つオブジェクト) マップを構築する."""
        if self.controller.bones:
            return dict(self.controller.bones)
        from types import SimpleNamespace
        return {
            b[0]: SimpleNamespace(
                name=b[0],
                head_3d=(float(b[1][0]), float(b[1][1]), float(b[1][2])),
                tail_3d=(float(b[2][0]), float(b[2][1]), float(b[2][2])),
            )
            for b in overlay_bones
        }

    def _project_world2d_point_to_3d(
        self,
        piece_id: str,
        world_pt_2d: tuple[float, float],
        *,
        bone_map: dict,
        centroid_cache: dict[str, tuple[float, float]],
    ) -> tuple[float, float, float]:
        """2Dワールド座標上のピース点を実際の3D配置空間座標へ投影する."""
        import numpy as np

        doc = self.controller.doc
        piece = doc.pieces.get(piece_id)
        if not piece:
            return (0.0, 0.0, 0.0)
        local_pt = piece.transform.inverse_apply(world_pt_2d)
        pl = doc.placements.get(piece_id)

        bone_info = bone_map.get(pl.target_bone) if (pl and pl.target_bone) else None
        if bone_info is not None and pl is not None:
            if piece_id not in centroid_cache:
                outer = piece.paths.get(piece.outer_path_id)
                pts = outer.sample_outline_points(steps_per_segment=16) if outer else []
                if pts:
                    cx = sum(p[0] for p in pts) / len(pts)
                    cy = sum(p[1] for p in pts) / len(pts)
                    centroid_cache[piece_id] = (cx, cy)
                else:
                    centroid_cache[piece_id] = local_pt
            projected = project_2d_points_to_bone_plane(
                [local_pt],
                head_3d=bone_info.head_3d,
                tail_3d=bone_info.tail_3d,
                bone_t=pl.bone_t,
                bone_angle_deg=pl.bone_angle_deg,
                surface_offset=pl.surface_offset,
                pattern_up_angle_deg=pl.pattern_up_angle_deg,
                centroid_2d=centroid_cache[piece_id],
            )
            p0 = projected[0]
            return (float(p0[0]), float(p0[1]), float(p0[2]))

        offset_3d = np.array(pl.offset_3d if pl else (0.0, 0.0, 0.0), dtype=np.float32)
        euler_3d = pl.euler_3d if pl else (0.0, 0.0, 0.0)
        R = _euler_to_rotation_matrix(euler_3d)
        local_3d = np.array([local_pt[0], 0.0, local_pt[1]], dtype=np.float32)
        world = (R @ local_3d) + offset_3d
        return (float(world[0]), float(world[1]), float(world[2]))

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

        doc = self.controller.doc
        bone_map = self._build_preview_bone_map(overlay_bones)
        shader = get_3d_uniform_color_shader() or get_line_shader()
        if not shader:
            return

        orig_blend = gpu.state.blend_get()
        orig_depth_test = gpu.state.depth_test_get()
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')

        try:
            # 2. 選択中ピースの3D配置リアルタイムプレビュー
            selected_pid: str | None = None
            if len(self.controller.selection.piece_ids) == 1:
                pid = next(iter(self.controller.selection.piece_ids))
                selected_pid = pid
                piece = doc.pieces.get(pid)
                pl = doc.placements.get(pid)
                if piece and pl and pl.target_bone:
                    target_item = next((b for b in overlay_bones if b[0] == pl.target_bone), None)
                    if target_item:
                        if not self.controller.picking_bone:
                            draw_wrinkle_armature_overlay(
                                [target_item],
                                active_bone_name=pl.target_bone,
                            )

                        head_3d = (float(target_item[1][0]), float(target_item[1][1]), float(target_item[1][2]))
                        tail_3d = (float(target_item[2][0]), float(target_item[2][1]), float(target_item[2][2]))

                        plane_center, _, _, ez, axis_origin = compute_bone_aligned_frame(
                            head_3d,
                            tail_3d,
                            bone_t=pl.bone_t,
                            bone_angle_deg=pl.bone_angle_deg,
                            surface_offset=pl.surface_offset,
                        )

                        # 2.1 ボーン主軸線 (head -> tail)
                        gpu.state.line_width_set(2.5)
                        shader.bind()
                        shader.uniform_float("color", (0.2, 0.85, 1.0, 0.9))
                        batch_axis = batch_for_shader(
                            shader, 'LINES',
                            {"pos": [head_3d, tail_3d]}
                        )
                        batch_axis.draw(shader)

                        # 2.2 ボーン位置マーカー (axis_origin -> plane_center) & ボーン上向き軸ガイド (+ez)
                        pt_axis = (float(axis_origin[0]), float(axis_origin[1]), float(axis_origin[2]))
                        pt_center = (float(plane_center[0]), float(plane_center[1]), float(plane_center[2]))
                        shader.uniform_float("color", (1.0, 0.9, 0.2, 0.95))
                        gpu.state.line_width_set(2.5)
                        batch_off = batch_for_shader(
                            shader, 'LINES',
                            {"pos": [pt_axis, pt_center]}
                        )
                        batch_off.draw(shader)

                        pt_up = (
                            pt_center[0] + float(ez[0]) * 0.08,
                            pt_center[1] + float(ez[1]) * 0.08,
                            pt_center[2] + float(ez[2]) * 0.08,
                        )
                        shader.uniform_float("color", (0.3, 1.0, 0.5, 0.95))
                        gpu.state.line_width_set(2.5)
                        batch_up = batch_for_shader(
                            shader, 'LINES',
                            {"pos": [pt_center, pt_up]}
                        )
                        batch_up.draw(shader)

                        # 2.3 型紙ピースの 3D 平面メッシュプレビュー (ローカル座標基準でビルドと一致させる)
                        outer = piece.paths.get(piece.outer_path_id)
                        poly_2d = outer.sample_outline_points(steps_per_segment=16) if outer else []
                        if len(poly_2d) >= 3:
                            world_3d = project_2d_points_to_bone_plane(
                                poly_2d,
                                head_3d=head_3d,
                                tail_3d=tail_3d,
                                bone_t=pl.bone_t,
                                bone_angle_deg=pl.bone_angle_deg,
                                surface_offset=pl.surface_offset,
                                pattern_up_angle_deg=pl.pattern_up_angle_deg,
                            )
                            poly_3d = [(float(p[0]), float(p[1]), float(p[2])) for p in world_3d]

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

                            gpu.state.line_width_set(2.2)
                            shader.uniform_float("color", (0.3, 0.9, 1.0, 0.95))
                            line_loop = poly_3d + [poly_3d[0]]
                            batch_outline = batch_for_shader(
                                shader, 'LINE_STRIP',
                                {"pos": line_loop}
                            )
                            batch_outline.draw(shader)

            # 3. ボーン巻付け (BONE_WRAP) 縫合の裏側迂回ルートプレビュー
            preview_seams = []
            seen_seam_ids: set[str] = set()
            for sid in self.controller.selection.seam_ids:
                seam = doc.seams.get(sid)
                if seam and seam.routing == SeamRouting.BONE_WRAP and not seam.waypoints and sid not in seen_seam_ids:
                    preview_seams.append(seam)
                    seen_seam_ids.add(sid)
            if selected_pid is not None:
                for sid, seam in doc.seams.items():
                    if sid in seen_seam_ids or seam.routing != SeamRouting.BONE_WRAP or seam.waypoints:
                        continue
                    if seam.a.piece_id == selected_pid or seam.b.piece_id == selected_pid:
                        preview_seams.append(seam)
                        seen_seam_ids.add(sid)

            if preview_seams:
                centroid_cache: dict[str, tuple[float, float]] = {}
                for seam in preview_seams:
                    bone_info, pl_ref = resolve_seam_bone_wrap_target(doc, seam, bones=bone_map)
                    if bone_info is None or pl_ref is None:
                        continue

                    if not selected_pid and not self.controller.picking_bone:
                        target_item = next((b for b in overlay_bones if b[0] == pl_ref.target_bone), None)
                        if target_item:
                            draw_wrinkle_armature_overlay(
                                [target_item],
                                active_bone_name=pl_ref.target_bone,
                            )

                    w_a0, w_a_mid, w_a1, w_b0, w_b_mid, w_b1 = doc.get_seam_geometry(seam)
                    shader.bind()
                    gpu.state.line_width_set(2.5)
                    shader.uniform_float("color", (1.0, 0.6, 0.15, 0.95))

                    for pt_a_2d, pt_b_2d in ((w_a0, w_b0), (w_a_mid, w_b_mid), (w_a1, w_b1)):
                        v_a_3d = self._project_world2d_point_to_3d(
                            seam.a.piece_id,
                            pt_a_2d,
                            bone_map=bone_map,
                            centroid_cache=centroid_cache,
                        )
                        v_b_3d = self._project_world2d_point_to_3d(
                            seam.b.piece_id,
                            pt_b_2d,
                            bone_map=bone_map,
                            centroid_cache=centroid_cache,
                        )
                        w1, w2 = compute_tubular_waypoints_for_pair(
                            v_a_3d,
                            v_b_3d,
                            head_3d=bone_info.head_3d,
                            tail_3d=bone_info.tail_3d,
                            bone_t=pl_ref.bone_t,
                            bone_angle_deg=pl_ref.bone_angle_deg,
                            surface_offset=pl_ref.surface_offset,
                        )
                        loop_pts = [
                            v_a_3d,
                            (float(w1[0]), float(w1[1]), float(w1[2])),
                            (float(w2[0]), float(w2[1]), float(w2[2])),
                            v_b_3d,
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
