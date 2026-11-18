# -*- coding: utf-8 -*-
"""型紙エディタ 辺分割ツール（SplitEdgeTool）・共通スナップ・プレビュー描画の単体テスト (Blender非依存)."""

import unittest

from taremin_cloth.pattern.document import (
    Document,
    EdgeRange,
    Node,
    NodeKind,
    Notch,
    NotchSnapKind,
    Path,
    PathRole,
    Seam,
    SeamMode,
)
from taremin_cloth.pattern.editor import EditorController, EditorEvent
from taremin_cloth.pattern.keymap import get_tool_hotkey_label, resolve_tool_hotkey
from taremin_cloth.pattern.layout import TOOL_ORDER, compute_hud_layout
from taremin_cloth.pattern.renderer import DrawCommandKind, build_render_commands
from taremin_cloth.pattern.session import ViewTransform2D
from taremin_cloth.pattern.theme import DEFAULT_THEME


class TestPatternSplitEdgeTool(unittest.TestCase):
    """辺分割ツール、共通スナップ解決、SelectTool連携、および描画コマンドを検証する."""

    def setUp(self) -> None:
        # 1m = 1000px, 原点 (0, 0) -> screen (0, 0)
        self.view = ViewTransform2D(pan_x=0.0, pan_y=0.0, zoom=1000.0)
        self.doc = Document.new()
        self.ctrl = EditorController(doc=self.doc, view=self.view)
        self.ctrl.screen_size = (1280, 720)

    def test_keymap_and_layout_registration(self) -> None:
        """ZEROキーとカスタムホットキー辞書、および左ツールレール配置を検証."""
        self.assertEqual(resolve_tool_hotkey("ZERO"), "SPLIT_EDGE")
        self.assertEqual(resolve_tool_hotkey("zero"), "SPLIT_EDGE")
        self.assertEqual(get_tool_hotkey_label("SPLIT_EDGE"), "0")

        custom_hotkeys = {"ONE": "SPLIT_EDGE", "TWO": "SELECT"}
        self.assertEqual(resolve_tool_hotkey("ONE", hotkeys=custom_hotkeys), "SPLIT_EDGE")
        self.assertEqual(get_tool_hotkey_label("SPLIT_EDGE", hotkeys=custom_hotkeys), "1")

        self.assertIn("SPLIT_EDGE", TOOL_ORDER)
        self.assertIn("SPLIT_EDGE", self.ctrl.tools)

        layout_720 = compute_hud_layout(1280, 720)
        self.assertIn("SPLIT_EDGE", layout_720.tool_buttons)
        self.assertGreaterEqual(layout_720.left_rail.y, layout_720.footer_bar.top + 8.0)

        layout_512 = compute_hud_layout(1280, 512)
        self.assertGreaterEqual(layout_512.left_rail.y, layout_512.footer_bar.top + 8.0)

        # レールボタンクリックで SPLIT_EDGE に切り替えられること
        split_btn = layout_720.tool_buttons["SPLIT_EDGE"]
        self.ctrl.handle_event(
            EditorEvent(
                screen_pos=(split_btn.x + split_btn.w * 0.5, split_btn.y + split_btn.h * 0.5),
                is_press=True,
                button="LEFT",
            )
        )
        self.assertEqual(self.ctrl.active_tool_name, "SPLIT_EDGE")

    def test_insert_node_arc_length_clamp_and_short_edge_guard(self) -> None:
        """長辺の 1cm 位置挿入が 0.01 比率クランプで潰れず、2mm 以下短辺は None を返すこと."""
        # 1. 長辺 2.0m の下辺端から 1cm (t=0.005)
        long_p = self.doc.add_rectangle_piece((0.0, 0.0), (2.0, 0.2), name="LongPanel")
        outer = long_p.get_outer_path()
        assert outer is not None
        inserted = self.doc.insert_node_on_segment(long_p.id, outer.id, seg_idx=0, t=0.005)
        self.assertIsNotNone(inserted)
        assert inserted is not None
        self.assertAlmostEqual(inserted.position[0], 0.01, places=5)
        self.assertAlmostEqual(inserted.position[1], 0.0, places=5)

        # 2. 極短辺 (1.5mm <= 2mm) では安全に分割拒否されること
        short_p = self.doc.add_rectangle_piece((0.0, 1.0), (0.0015, 1.2), name="ShortPanel")
        short_outer = short_p.get_outer_path()
        assert short_outer is not None
        orig_len = len(short_outer.nodes)
        rejected = self.doc.insert_node_on_segment(short_p.id, short_outer.id, seg_idx=0, t=0.5)
        self.assertIsNone(rejected)
        self.assertEqual(len(short_outer.nodes), orig_len)

    def test_resolve_edge_point_snap_split_mode(self) -> None:
        """SPLIT モードのスナップ種別（中点・1cm・クリップボード・縫合相手・既存ノッチ・フリー）を検証."""
        p1 = self.doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="PanelA")
        p2 = self.doc.add_rectangle_piece((0.0, 0.4), (0.4, 0.6), name="PanelB")
        out1 = p1.get_outer_path()
        out2 = p2.get_outer_path()
        assert out1 is not None and out2 is not None
        n0 = out1.nodes[0]

        # 1cm 刻みスナップ (10.2cm -> 10.0cm)
        res_1cm = self.doc.resolve_edge_point_snap(
            p1.id, n0.id, raw_t=0.102 / 0.40, mode="SPLIT", path_id=out1.id
        )
        self.assertEqual(res_1cm.kind, NotchSnapKind.STEP_1CM)
        self.assertAlmostEqual(res_1cm.split_left_len, 0.10, places=4)
        self.assertAlmostEqual(res_1cm.split_right_len, 0.30, places=4)

        # 中点スナップ (19.7cm -> 20.0cm)
        res_mid = self.doc.resolve_edge_point_snap(
            p1.id, n0.id, raw_t=0.197 / 0.40, mode="SPLIT", path_id=out1.id
        )
        self.assertEqual(res_mid.kind, NotchSnapKind.MIDPOINT)
        self.assertAlmostEqual(res_mid.t, 0.5, places=5)

        # フリー配置 (Alt = snap_enabled=False)
        res_free = self.doc.resolve_edge_point_snap(
            p1.id, n0.id, raw_t=0.123 / 0.40, mode="SPLIT", path_id=out1.id, snap_enabled=False
        )
        self.assertEqual(res_free.kind, NotchSnapKind.FREE)
        self.assertAlmostEqual(res_free.arc_s, 0.123, places=4)

        # クリップボード長 (15.5cm)
        res_clip = self.doc.resolve_edge_point_snap(
            p1.id,
            n0.id,
            raw_t=0.158 / 0.40,
            mode="SPLIT",
            path_id=out1.id,
            clipboard_length=0.155,
        )
        self.assertEqual(res_clip.kind, NotchSnapKind.CLIPBOARD)
        self.assertAlmostEqual(res_clip.arc_s, 0.155, places=4)

        # 既存ノッチ位置 (t=0.3375 -> 13.5cm) に 8mm 以内で吸着し、左右長が辺全体基準で返ること
        self.doc.notches["n_ex"] = Notch(
            id="n_ex",
            piece_id=p1.id,
            path_id=out1.id,
            segment_start_id=n0.id,
            t=0.3375,
        )
        res_ex = self.doc.resolve_edge_point_snap(
            p1.id, n0.id, raw_t=0.138 / 0.40, mode="SPLIT", path_id=out1.id
        )
        self.assertEqual(res_ex.kind, NotchSnapKind.EXISTING)
        self.assertEqual(res_ex.existing_notch_id, "n_ex")
        self.assertAlmostEqual(res_ex.t, 0.3375, places=5)
        self.assertAlmostEqual(res_ex.split_left_len, 0.135, places=4)
        self.assertAlmostEqual(res_ex.split_right_len, 0.265, places=4)

        # 別の 22.3cm 付近をスナップした場合も左右分割長がノッチ基準ではなく辺全体 (22.0cm / 18.0cm) になること
        res_other = self.doc.resolve_edge_point_snap(
            p1.id, n0.id, raw_t=0.223 / 0.40, mode="SPLIT", path_id=out1.id
        )
        self.assertEqual(res_other.kind, NotchSnapKind.STEP_1CM)
        self.assertAlmostEqual(res_other.split_left_len, 0.22, places=4)
        self.assertAlmostEqual(res_other.split_right_len, 0.18, places=4)

        # Seam 相手ノッチ位置 (PanelB 下辺 7.3cm) へスナップすること
        er_a = EdgeRange(piece_id=p1.id, path_id=out1.id, start_node_id=out1.nodes[0].id, end_node_id=out1.nodes[1].id)
        er_b = EdgeRange(piece_id=p2.id, path_id=out2.id, start_node_id=out2.nodes[0].id, end_node_id=out2.nodes[1].id)
        self.doc.seams["s_ab"] = Seam(id="s_ab", a=er_a, b=er_b, reverse=False, mode=SeamMode.SEGMENT)
        self.doc.notches["n_partner"] = Notch(
            id="n_partner",
            piece_id=p2.id,
            path_id=out2.id,
            segment_start_id=out2.nodes[0].id,
            t=0.073 / 0.40,
        )
        res_seam = self.doc.resolve_edge_point_snap(
            p1.id, n0.id, raw_t=0.075 / 0.40, mode="SPLIT", path_id=out1.id
        )
        self.assertEqual(res_seam.kind, NotchSnapKind.SEAM_PARTNER)
        self.assertAlmostEqual(res_seam.arc_s, 0.073, places=4)

    def test_split_tool_hover_alt_recompute_click_seam_notch_and_undo(self) -> None:
        """SPLIT_EDGE でのホバー、Alt即時切替、クリック分割、縫合CHAIN分割、既存ノッチ置換、Undo/Redoを検証."""
        p1 = self.doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="Front")
        p2 = self.doc.add_rectangle_piece((0.0, 0.4), (0.4, 0.6), name="Back")
        out1 = p1.get_outer_path()
        out2 = p2.get_outer_path()
        assert out1 is not None and out2 is not None

        er1 = EdgeRange(piece_id=p1.id, path_id=out1.id, start_node_id=out1.nodes[0].id, end_node_id=out1.nodes[1].id)
        er2 = EdgeRange(piece_id=p2.id, path_id=out2.id, start_node_id=out2.nodes[0].id, end_node_id=out2.nodes[1].id)
        seam = Seam(id="s1", a=er1, b=er2, reverse=False, mode=SeamMode.SEGMENT)
        self.doc.seams[seam.id] = seam

        # 既存ノッチを 10.0cm 位置 (t=0.25) に配置
        self.doc.notches["n_at_10"] = Notch(
            id="n_at_10",
            piece_id=p1.id,
            path_id=out1.id,
            segment_start_id=out1.nodes[0].id,
            t=0.25,
        )

        self.ctrl.set_active_tool("SPLIT_EDGE")

        # 1. 0.143m 付近へホバー -> 14.0cm スナップ
        pt = (0.143, 0.0)
        self.ctrl.handle_event(EditorEvent(screen_pos=self.view.world_to_screen(pt), is_move=True))
        self.assertIsNotNone(self.ctrl.edge_preview)
        assert self.ctrl.edge_preview is not None
        self.assertEqual(self.ctrl.edge_preview.target_kind, "SPLIT")
        self.assertEqual(self.ctrl.edge_preview.snap_kind, NotchSnapKind.STEP_1CM)
        self.assertAlmostEqual(self.ctrl.edge_preview.split_left_len, 0.14, places=4)

        # 2. Alt 単体押下イベントで移動せず即時 FREE へ再評価されること
        hover_screen = self.view.world_to_screen(pt)
        self.ctrl.handle_event(EditorEvent(screen_pos=hover_screen, key="LEFT_ALT", button="", alt=True))
        assert self.ctrl.edge_preview is not None
        self.assertEqual(self.ctrl.edge_preview.snap_kind, NotchSnapKind.FREE)
        self.assertAlmostEqual(self.ctrl.edge_preview.split_left_len, 0.143, places=4)

        # 3. Alt 解放イベント (ops/pattern.py 同等の button='', key='') で STEP_1CM へ戻ること
        self.ctrl.handle_event(EditorEvent(screen_pos=hover_screen, key="", button="", alt=False))
        assert self.ctrl.edge_preview is not None
        self.assertEqual(self.ctrl.edge_preview.snap_kind, NotchSnapKind.STEP_1CM)

        # 4. 既存ノッチ位置 (x=0.102) へ移動しクリック分割 -> 既存ノッチが置換され、縫合が CHAIN 分割されること
        pt_notch = (0.102, 0.0)
        self.ctrl.handle_event(EditorEvent(screen_pos=self.view.world_to_screen(pt_notch), is_move=True))
        assert self.ctrl.edge_preview is not None
        self.assertEqual(self.ctrl.edge_preview.snap_kind, NotchSnapKind.EXISTING)

        consumed = self.ctrl.handle_event(
            EditorEvent(screen_pos=self.view.world_to_screen(pt_notch), is_press=True, button="LEFT")
        )
        self.assertTrue(consumed)
        self.assertEqual(len(out1.nodes), 5)
        self.assertEqual(len(self.doc.notches), 0)
        self.assertEqual(len(self.ctrl.selection.node_ids), 1)
        self.assertIsNone(self.ctrl.edge_preview)
        self.assertEqual(len(seam.chain_a), 2)
        inserted_node = next(n for n in out1.nodes if n.id in self.ctrl.selection.node_ids)
        self.assertAlmostEqual(inserted_node.position[0], 0.10, places=4)

        # 5. Undo で元通り（4頂点・ノッチ復活・縫合1区間）へ戻り、Redo で再分割されること
        self.assertTrue(self.ctrl.undo())
        out1_undo = self.ctrl.doc.pieces[p1.id].get_outer_path()
        assert out1_undo is not None
        self.assertEqual(len(out1_undo.nodes), 4)
        self.assertEqual(len(self.ctrl.doc.notches), 1)
        self.assertEqual(len(self.ctrl.doc.seams[seam.id].effective_a()), 1)
        self.assertIsNone(self.ctrl.edge_preview)

        self.assertTrue(self.ctrl.redo())
        out1_redo = self.ctrl.doc.pieces[p1.id].get_outer_path()
        assert out1_redo is not None
        self.assertEqual(len(out1_redo.nodes), 5)
        self.assertEqual(len(self.ctrl.doc.notches), 0)
        self.assertEqual(len(self.ctrl.doc.seams[seam.id].chain_a), 2)
        self.assertIsNone(self.ctrl.edge_preview)

    def test_split_internal_line_and_render_suppression(self) -> None:
        """内部線（INTERNAL_PATH）の分割、分割マーカー描画、内部線元寸法テキストの重複抑止を検証."""
        p = self.doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="Panel")
        int_path = Path(
            id="int_1",
            role=PathRole.INTERNAL,
            nodes=[
                Node(id="in0", position=(0.05, 0.10), kind=NodeKind.CORNER),
                Node(id="in1", position=(0.35, 0.10), kind=NodeKind.CORNER),
            ],
            closed=False,
        )
        p.paths[int_path.id] = int_path
        self.ctrl.set_active_tool("SPLIT_EDGE")

        # 内部線 30cm の x=0.148 (左 10.0cm / 右 20.0cm) 付近へホバー
        hover_pt = (0.148, 0.10)
        self.ctrl.handle_event(EditorEvent(screen_pos=self.view.world_to_screen(hover_pt), is_move=True))
        self.assertIsNotNone(self.ctrl.edge_preview)
        assert self.ctrl.edge_preview is not None
        self.assertEqual(self.ctrl.edge_preview.path_id, int_path.id)
        self.assertEqual(self.ctrl.edge_preview.target_kind, "SPLIT")
        self.assertAlmostEqual(self.ctrl.edge_preview.split_left_len, 0.10, places=4)
        self.assertAlmostEqual(self.ctrl.edge_preview.split_right_len, 0.20, places=4)

        # 描画コマンド: ノッチ用 POLYLINE ではなく分割ガイド LINE + CIRCLE 2個が出力され、元の 30.0 cm 寸法は抑止
        cmds = build_render_commands(self.doc, self.ctrl, screen_size=self.ctrl.screen_size)
        split_lines = [
            c for c in cmds
            if c.kind == DrawCommandKind.LINE and c.color == DEFAULT_THEME.split_preview
        ]
        split_circles = [
            c for c in cmds
            if c.kind == DrawCommandKind.CIRCLE and c.color == DEFAULT_THEME.split_preview
        ]
        notch_pv_polys = [
            c for c in cmds
            if c.kind == DrawCommandKind.POLYLINE and c.color == DEFAULT_THEME.notch_preview
        ]
        self.assertEqual(len(split_lines), 1)
        self.assertEqual(len(split_circles), 2)
        self.assertEqual(len(notch_pv_polys), 0)

        texts = [(c.text, c.color) for c in cmds if c.kind == DrawCommandKind.TEXT]
        self.assertTrue(any(t == "10.0 cm" and col == DEFAULT_THEME.notch_preview_text for t, col in texts))
        self.assertTrue(any(t == "20.0 cm" and col == DEFAULT_THEME.notch_preview_text for t, col in texts))
        self.assertFalse(any(t == "30.0 cm" and col == DEFAULT_THEME.dimension_text for t, col in texts))
        self.assertTrue(any(t == "Split" for t, _ in texts))

        # クリックして内部線を実際に分割
        self.ctrl.handle_event(
            EditorEvent(screen_pos=self.view.world_to_screen(hover_pt), is_press=True, button="LEFT")
        )
        self.assertEqual(len(int_path.nodes), 3)
        self.assertAlmostEqual(int_path.nodes[1].position[0], 0.15, places=4)

    def test_split_render_existing_notch_badge_and_suppresses_notch_subdims(self) -> None:
        """既存ノッチスナップ時に Notch/合印 バッジが出力され、ノッチ既存区間寸法が抑止されること."""
        p = self.doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="Panel")
        outer = p.get_outer_path()
        assert outer is not None
        self.doc.notches["n1"] = Notch(
            id="n1",
            piece_id=p.id,
            path_id=outer.id,
            segment_start_id=outer.nodes[0].id,
            t=0.25,
        )
        self.ctrl.set_active_tool("SPLIT_EDGE")
        self.ctrl.handle_event(
            EditorEvent(screen_pos=self.view.world_to_screen((0.102, 0.0)), is_move=True)
        )
        cmds = build_render_commands(self.doc, self.ctrl, screen_size=self.ctrl.screen_size)
        dim_texts = [c.text for c in cmds if c.kind == DrawCommandKind.TEXT and c.color == DEFAULT_THEME.dimension_text]
        notch_badges = [
            c.text for c in cmds
            if c.kind == DrawCommandKind.TEXT and c.color == DEFAULT_THEME.notch
        ]
        pv_texts = [
            c.text for c in cmds
            if c.kind == DrawCommandKind.TEXT and c.color == DEFAULT_THEME.notch_preview_text
        ]
        self.assertTrue(any(t in ("Notch", "合印") for t in notch_badges))
        self.assertNotIn("10.0 cm", dim_texts)
        self.assertNotIn("30.0 cm", dim_texts)
        self.assertIn("10.0 cm", pv_texts)
        self.assertIn("30.0 cm", pv_texts)

    def test_select_tool_ctrl_hover_preview_and_snapped_split(self) -> None:
        """SelectTool での Ctrl ホバー分割プレビュー表示・解除、および Ctrl+クリック/ダブルクリックのスナップ分割を検証."""
        p = self.doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="Panel")
        outer = p.get_outer_path()
        assert outer is not None
        self.ctrl.set_active_tool("SELECT")

        # 1. Ctrl+ホバーで SPLIT プレビューが表示され、1cm 刻み (10.2cm -> 10.0cm) に吸着すること
        hover_pt = (0.102, 0.0)
        hover_screen = self.view.world_to_screen(hover_pt)
        self.ctrl.handle_event(
            EditorEvent(screen_pos=hover_screen, is_move=True, ctrl=True)
        )
        self.assertIsNotNone(self.ctrl.edge_preview)
        assert self.ctrl.edge_preview is not None
        self.assertEqual(self.ctrl.edge_preview.target_kind, "SPLIT")
        self.assertEqual(self.ctrl.edge_preview.snap_kind, NotchSnapKind.STEP_1CM)

        # 2. Ctrl キーを離した時点でカーソルを動かさずとも即座にプレビューが消えること
        self.ctrl.handle_event(EditorEvent(screen_pos=hover_screen, key="", button="", ctrl=False))
        self.assertIsNone(self.ctrl.edge_preview)

        # Ctrl 単体押下でプレビューが再表示され、key="LEFT_CTRL", is_release=True でも消えること
        self.ctrl.handle_event(EditorEvent(screen_pos=hover_screen, key="LEFT_CTRL", button="", ctrl=True))
        self.assertIsNotNone(self.ctrl.edge_preview)
        self.ctrl.handle_event(
            EditorEvent(screen_pos=hover_screen, key="LEFT_CTRL", button="", is_release=True, ctrl=False)
        )
        self.assertIsNone(self.ctrl.edge_preview)

        # 3. Ctrl+クリックでスナップ位置 (x=0.10) に分割ノードが挿入されること
        self.ctrl.handle_event(
            EditorEvent(screen_pos=self.view.world_to_screen(hover_pt), is_press=True, button="LEFT", ctrl=True)
        )
        self.assertEqual(len(outer.nodes), 5)
        self.assertAlmostEqual(outer.nodes[1].position[0], 0.10, places=4)

        # 4. 上辺の 30.2cm 付近を Alt=False ダブルクリックするとスナップ (x=0.30) で分割されること
        # 上辺は nodes[3] (0, 0.2) -> nodes[2] (0.4, 0.2)
        dbl_pt = (0.302, 0.2)
        self.ctrl.handle_event(
            EditorEvent(
                screen_pos=self.view.world_to_screen(dbl_pt),
                is_press=True,
                is_double_click=True,
                button="LEFT",
            )
        )
        self.assertEqual(len(outer.nodes), 6)
        inserted_top = next(n for n in outer.nodes if n.id in self.ctrl.selection.node_ids)
        self.assertAlmostEqual(inserted_top.position[0], 0.30, places=4)


if __name__ == "__main__":
    unittest.main()
