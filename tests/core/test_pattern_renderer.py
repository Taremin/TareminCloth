# -*- coding: utf-8 -*-
"""型紙エディタ 描画コマンド生成モジュール (renderer.py) の単体テスト."""

import unittest

from taremin_cloth.pattern.document import Document
from taremin_cloth.pattern.editor import EditorController
from taremin_cloth.pattern.layout import compute_hud_layout
from taremin_cloth.pattern.renderer import (
    DrawCommandKind,
    build_render_commands,
)
from taremin_cloth.pattern.session import ViewTransform2D
from taremin_cloth.pattern.theme import DEFAULT_THEME


class TestPatternRenderer(unittest.TestCase):
    """描画コマンド生成の完全性テスト."""

    def setUp(self):
        self.doc = Document.new(name="Render Test Doc")
        view = ViewTransform2D(pan_x=400.0, pan_y=300.0, zoom=1000.0)
        self.ctrl = EditorController(doc=self.doc, view=view)
        self.screen_size = (800, 600)

    def test_empty_scene_commands(self):
        cmds = build_render_commands(self.doc, self.ctrl, screen_size=self.screen_size)
        self.assertGreater(len(cmds), 0)

        # グリッド線の存在
        grid_lines = [c for c in cmds if c.kind == DrawCommandKind.LINE]
        self.assertGreater(len(grid_lines), 5)

        # HUD要素の存在
        text_cmds = [c for c in cmds if c.kind == DrawCommandKind.TEXT]
        texts = [c.text for c in text_cmds]
        self.assertTrue(any("Render Test Doc" in t for t in texts))
        self.assertTrue(any("未選択" in t for t in texts))

    def test_piece_render_commands(self):
        piece = self.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.3), name="PieceA")
        cmds = build_render_commands(self.doc, self.ctrl, screen_size=self.screen_size)

        # 面ポリゴン (POLYGON)
        polys = [c for c in cmds if c.kind == DrawCommandKind.POLYGON]
        self.assertEqual(len(polys), 1)
        self.assertEqual(len(polys[0].points), 4)

        # 外周線 (POLYLINE) - 4辺分
        polylines = [c for c in cmds if c.kind == DrawCommandKind.POLYLINE]
        self.assertEqual(len(polylines), 4)

        # ノード (CIRCLE) - 4ノード分
        circles = [c for c in cmds if c.kind == DrawCommandKind.CIRCLE]
        self.assertEqual(len(circles), 4)

        # ピースを選択状態にする -> 選択色に変化すること
        self.ctrl.selection.select_piece(piece.id)
        cmds_sel = build_render_commands(self.doc, self.ctrl, screen_size=self.screen_size)
        polys_sel = [c for c in cmds_sel if c.kind == DrawCommandKind.POLYGON]
        self.assertEqual(polys_sel[0].color, DEFAULT_THEME.piece_fill_selected)

    def test_seam_render_commands(self):
        p1 = self.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.3), name="A")
        p2 = self.doc.add_rectangle_piece((0.4, 0.0), (0.6, 0.3), name="B")
        segs_a = self.doc.evaluate_path_segments(p1.id, p1.outer_path_id)
        segs_b = self.doc.evaluate_path_segments(p2.id, p2.outer_path_id)

        # 縫合を追加
        from taremin_cloth.pattern.document import EdgeRange, Seam
        seam = Seam(
            id="s1",
            a=EdgeRange(piece_id=p1.id, path_id=p1.outer_path_id, start_node_id=segs_a[1][0], end_node_id=segs_a[1][1]),
            b=EdgeRange(piece_id=p2.id, path_id=p2.outer_path_id, start_node_id=segs_b[3][0], end_node_id=segs_b[3][1]),
        )
        self.doc.seams[seam.id] = seam

        cmds = build_render_commands(self.doc, self.ctrl, screen_size=self.screen_size)
        seam_lines = [c for c in cmds if c.kind == DrawCommandKind.LINE and c.color in (DEFAULT_THEME.seam_line, DEFAULT_THEME.seam_mismatch)]
        self.assertEqual(len(seam_lines), 1)

    def test_preview_render_commands(self):
        self.ctrl.active_tool_name = "RECTANGLE"
        self.ctrl.preview_points = [(0.0, 0.0), (0.2, 0.3)]
        cmds = build_render_commands(self.doc, self.ctrl, screen_size=self.screen_size)

        preview_polys = [c for c in cmds if c.kind == DrawCommandKind.POLYLINE and c.color == DEFAULT_THEME.preview_line]
        self.assertEqual(len(preview_polys), 1)
        self.assertEqual(len(preview_polys[0].points), 5)  # 閉じた矩形 (5頂点)

    def test_hud_layout(self):
        layout = compute_hud_layout(1000, 800)
        self.assertAlmostEqual(layout.header_bar.w, 1000 - 24.0)
        self.assertAlmostEqual(layout.footer_bar.w, 1000 - 24.0)
        self.assertTrue(layout.header_bar.contains(500.0, 780.0))
        self.assertTrue(layout.footer_bar.contains(500.0, 20.0))


if __name__ == "__main__":
    unittest.main()
