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
        self.assertTrue(any("No Selection" in t or "未選択" in t for t in texts))

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
        # 両端を結ぶ2本の主縫合線が生成されること
        seam_lines = [c for c in cmds if c.kind == DrawCommandKind.LINE and c.color in (DEFAULT_THEME.seam_line, DEFAULT_THEME.seam_mismatch)]
        self.assertEqual(len(seam_lines), 2)

    def test_bone_wrap_seam_and_up_arrow_render_commands(self):
        """BONE_WRAP 縫合線の識別描画・ボーン付きピースの上方向矢印・カード文字列の検証."""
        from taremin_cloth.pattern.document import EdgeRange, Placement, Seam, SeamRouting

        p1 = self.doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="Sleeve")
        self.doc.placements[p1.id] = Placement(
            piece_id=p1.id,
            target_bone="UpperArm.L",
            pattern_up_angle_deg=90.0,
        )
        nodes = p1.get_outer_path().nodes
        seam = Seam(
            id="s_wrap",
            a=EdgeRange(piece_id=p1.id, path_id=p1.outer_path_id, start_node_id=nodes[0].id, end_node_id=nodes[1].id),
            b=EdgeRange(piece_id=p1.id, path_id=p1.outer_path_id, start_node_id=nodes[3].id, end_node_id=nodes[2].id),
            routing=SeamRouting.BONE_WRAP,
        )
        self.doc.seams[seam.id] = seam
        self.ctrl.selection.select_piece(p1.id)

        cmds = build_render_commands(self.doc, self.ctrl, screen_size=self.screen_size)
        # 2D上方向矢印 (緑色 LINE 3本) と BONE_WRAP 縫合線・補助円が出力されること
        up_arrows = [c for c in cmds if c.kind == DrawCommandKind.LINE and c.color == (0.3, 1.0, 0.5, 0.9)]
        self.assertEqual(len(up_arrows), 3)
        wrap_lines = [c for c in cmds if c.kind == DrawCommandKind.LINE and c.color == DEFAULT_THEME.seam_wrap]
        self.assertEqual(len(wrap_lines), 2)
        wrap_circles = [
            c for c in cmds
            if c.kind == DrawCommandKind.CIRCLE and c.color == DEFAULT_THEME.seam_wrap and not c.filled
        ]
        self.assertEqual(len(wrap_circles), 1)
        texts = [c.text for c in cmds if c.kind == DrawCommandKind.TEXT]
        self.assertTrue(any("90°" in t for t in texts))

        # 縫合選択時は Routing 行が表示されること
        self.ctrl.selection.select_seam(seam.id)
        cmds_seam = build_render_commands(self.doc, self.ctrl, screen_size=self.screen_size)
        seam_texts = [c.text for c in cmds_seam if c.kind == DrawCommandKind.TEXT]
        self.assertTrue(any("Routing" in t or "経路" in t for t in seam_texts))

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

    def test_notch_preview_render_commands_and_show_lengths(self):
        """ノッチプレビュー時の幾何・分割長表示・通常寸法テキストの重なり回避を検証."""
        from taremin_cloth.pattern.editor import EditorEvent

        p = self.doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="Panel")
        self.ctrl.set_active_tool("NOTCH")
        # x=0.10 (下辺 0.40m の 10cm / 30cm 位置) にホバー
        w_pos = (0.10, 0.0)
        self.ctrl.handle_event(
            EditorEvent(
                screen_pos=self.ctrl.view.world_to_screen(w_pos),
                world_pos=w_pos,
                is_move=True,
            )
        )
        self.assertIsNotNone(self.ctrl.notch_preview)

        cmds = build_render_commands(self.doc, self.ctrl, screen_size=self.screen_size)
        # プレビューポリラインが出力されること
        pv_polys = [c for c in cmds if c.kind == DrawCommandKind.POLYLINE and c.color == DEFAULT_THEME.notch_preview]
        self.assertEqual(len(pv_polys), 1)
        # 分割長 10.0 cm と 30.0 cm が表示され、プレビュー中セグメント（下辺）の元の 40.0 cm dimension_text は非表示となり上辺の1つのみ残ること
        texts = [(c.text, c.color) for c in cmds if c.kind == DrawCommandKind.TEXT]
        self.assertTrue(any(t == "10.0 cm" and col == DEFAULT_THEME.notch_preview_text for t, col in texts))
        self.assertTrue(any(t == "30.0 cm" and col == DEFAULT_THEME.notch_preview_text for t, col in texts))
        dim_40_count = sum(1 for t, col in texts if t == "40.0 cm" and col == DEFAULT_THEME.dimension_text)
        self.assertEqual(dim_40_count, 1)

        # show_lengths=False 時は通常 dimension_text は 0 件のままだが、プレビュー分割長は表示され続けること
        self.ctrl.show_lengths = False
        cmds_no_dim = build_render_commands(self.doc, self.ctrl, screen_size=self.screen_size)
        dim_texts = [c for c in cmds_no_dim if c.kind == DrawCommandKind.TEXT and c.color == DEFAULT_THEME.dimension_text]
        self.assertEqual(len(dim_texts), 0)
        pv_texts = [c.text for c in cmds_no_dim if c.kind == DrawCommandKind.TEXT and c.color == DEFAULT_THEME.notch_preview_text]
        self.assertIn("10.0 cm", pv_texts)
        self.assertIn("30.0 cm", pv_texts)

        # 中点 (x=0.20) にホバー時は中点スナップ色およびバッジが出力されること
        self.ctrl.handle_event(
            EditorEvent(
                screen_pos=self.ctrl.view.world_to_screen((0.198, 0.0)),
                world_pos=(0.198, 0.0),
                is_move=True,
            )
        )
        cmds_mid = build_render_commands(self.doc, self.ctrl, screen_size=self.screen_size)
        mid_polys = [c for c in cmds_mid if c.kind == DrawCommandKind.POLYLINE and c.color == DEFAULT_THEME.notch_snap_midpoint]
        self.assertEqual(len(mid_polys), 1)
        mid_texts = [c.text for c in cmds_mid if c.kind == DrawCommandKind.TEXT and c.color == DEFAULT_THEME.notch_snap_midpoint]
        self.assertTrue(any(t in ("Mid", "中点") for t in mid_texts))

        # ノッチを確定した後、同じ場所へホバーすると既存強調リングが出力されること
        self.ctrl.handle_event(
            EditorEvent(
                screen_pos=self.ctrl.view.world_to_screen((0.20, 0.0)),
                world_pos=(0.20, 0.0),
                is_press=True,
                button="LEFT",
            )
        )
        self.ctrl.handle_event(
            EditorEvent(
                screen_pos=self.ctrl.view.world_to_screen((0.2002, 0.0)),
                world_pos=(0.2002, 0.0),
                is_move=True,
            )
        )
        cmds_ex = build_render_commands(self.doc, self.ctrl, screen_size=self.screen_size)
        ex_rings = [
            c for c in cmds_ex
            if c.kind == DrawCommandKind.CIRCLE and c.color == DEFAULT_THEME.notch_selected and not c.filled
        ]
        self.assertEqual(len(ex_rings), 1)

    def test_existing_notch_preview_suppresses_both_split_dimensions(self) -> None:
        """EXISTING プレビュー時、左右両区間の通常 dimension_text が抑止され左右プレビュー長が対称描画されること."""
        from taremin_cloth.pattern.editor import EditorEvent

        self.doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="Panel")
        self.ctrl.set_active_tool("NOTCH")
        # x=0.16 (0.40m 辺上の t=0.40 -> 16.0 cm / 24.0 cm) にノッチ作成
        self.ctrl.handle_event(
            EditorEvent(
                screen_pos=self.ctrl.view.world_to_screen((0.16, 0.0)),
                world_pos=(0.16, 0.0),
                is_press=True,
                button="LEFT",
            )
        )
        # 作成済みノッチ上へホバーして EXISTING 状態にする
        self.ctrl.handle_event(
            EditorEvent(
                screen_pos=self.ctrl.view.world_to_screen((0.1602, 0.0)),
                world_pos=(0.1602, 0.0),
                is_move=True,
            )
        )
        self.assertIsNotNone(self.ctrl.notch_preview)
        self.assertIsNotNone(self.ctrl.notch_preview.existing_notch_id)

        self.ctrl.show_lengths = True
        cmds = build_render_commands(self.doc, self.ctrl, screen_size=self.screen_size)
        dim_texts = [c.text for c in cmds if c.kind == DrawCommandKind.TEXT and c.color == DEFAULT_THEME.dimension_text]
        pv_texts = [c.text for c in cmds if c.kind == DrawCommandKind.TEXT and c.color == DEFAULT_THEME.notch_preview_text]

        self.assertNotIn("16.0 cm", dim_texts)
        self.assertNotIn("24.0 cm", dim_texts)
        self.assertEqual(pv_texts.count("16.0 cm"), 1)
        self.assertEqual(pv_texts.count("24.0 cm"), 1)


if __name__ == "__main__":
    unittest.main()
