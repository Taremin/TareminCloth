# -*- coding: utf-8 -*-
"""型紙パターンエディタ Phase 1-3（辺移動・複数選択・モーダル変形・整列・回転/拡縮・一括編集）の検証テスト."""

import os
import sys
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PY_DIR = os.path.join(_ROOT, "python")
if _PY_DIR not in sys.path:
    sys.path.insert(0, _PY_DIR)

from taremin_cloth.pattern.build_types import build_garment_mesh_geometry
from taremin_cloth.pattern.cards import format_selection_card_lines
from taremin_cloth.pattern.document import (
    Document,
    EdgeRange,
    Node,
    NodeKind,
    Path,
    PathRole,
    Placement,
    Seam,
)
from taremin_cloth.pattern.editor import EditorController, EditorEvent
from taremin_cloth.pattern.geometry import signed_area
from taremin_cloth.pattern.ids import new_id
from taremin_cloth.pattern.keymap import resolve_tool_hotkey
from taremin_cloth.pattern.layout import compute_hud_layout
from taremin_cloth.pattern.renderer import DrawCommandKind, build_render_commands
from taremin_cloth.pattern.session import ViewTransform2D


class TestPatternMultiTransform(unittest.TestCase):
    """Phase 1〜3 の複数要素操作・辺移動・モーダル変形テストスイート."""

    def setUp(self) -> None:
        self.doc = Document.new(name="MultiTest")
        # view: pan=(0,0), zoom=1000 -> world (x, y) が screen (1000*x, 1000*y)
        self.view = ViewTransform2D(pan_x=0.0, pan_y=0.0, zoom=1000.0)
        self.ctrl = EditorController(doc=self.doc, view=self.view)
        self.ctrl.screen_size = (1280, 720)
        self.ctrl.snap_settings.enabled = False

    def _add_rect(self, x0: float, y0: float, x1: float, y1: float, name: str = "Rect"):
        return self.ctrl.doc.add_rectangle_piece((x0, y0), (x1, y1), name=name)

    # ------------------------------------------------------------------
    # Phase 1: 辺移動 / 誤削除ガード / Pending Click & Toggle / Single Snap
    # ------------------------------------------------------------------

    def test_edge_and_internal_path_selection_and_delete_guard(self) -> None:
        """外周辺選択ではピースごと誤削除されず、内部線選択時は内部線のみ削除できること."""
        rect = self._add_rect(0.0, 0.0, 0.2, 0.2, name="Body")
        outer = rect.get_outer_path()
        assert outer is not None

        # 内部線を追加: (0.05, 0.1) -> (0.15, 0.1)
        int_n0 = Node(id=new_id(), position=(0.05, 0.1), kind=NodeKind.CORNER)
        int_n1 = Node(id=new_id(), position=(0.15, 0.1), kind=NodeKind.CORNER)
        int_path = Path(id=new_id(), role=PathRole.INTERNAL, nodes=[int_n0, int_n1], closed=False)
        rect.paths[int_path.id] = int_path

        # 1. 外周上辺をクリック -> edge_ranges に入り piece_ids は空
        self.ctrl.handle_event(EditorEvent(screen_pos=(100.0, 200.0), is_press=True, button="LEFT"))
        self.ctrl.handle_event(EditorEvent(screen_pos=(100.0, 200.0), is_release=True, button="LEFT"))
        self.assertEqual(len(self.ctrl.selection.edge_ranges), 1)
        self.assertEqual(len(self.ctrl.selection.piece_ids), 0)

        # 外周辺選択中に X キー -> 削除は拒否されピースも無傷
        deleted_outer = self.ctrl.handle_event(EditorEvent(key="X"))
        self.assertIn(rect.id, self.ctrl.doc.pieces)
        self.assertTrue(bool(self.ctrl.status_message) or not deleted_outer)

        # 2. 内部線中央 (100, 100) をクリック -> 内部線が edge_ranges に選ばれ piece_ids は空
        self.ctrl.handle_event(EditorEvent(screen_pos=(100.0, 100.0), is_press=True, button="LEFT"))
        self.ctrl.handle_event(EditorEvent(screen_pos=(100.0, 100.0), is_release=True, button="LEFT"))
        self.assertEqual(len(self.ctrl.selection.edge_ranges), 1)
        self.assertEqual(self.ctrl.selection.edge_ranges[0].path_id, int_path.id)
        self.assertEqual(len(self.ctrl.selection.piece_ids), 0)

        # 内部線選択中に X キー -> 内部線のみ削除されピースは維持
        self.assertTrue(self.ctrl.handle_event(EditorEvent(key="X")))
        self.assertIn(rect.id, self.ctrl.doc.pieces)
        self.assertNotIn(int_path.id, rect.paths)

        # Undo で内部線復活
        self.ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        self.assertIn(int_path.id, self.ctrl.doc.pieces[rect.id].paths)

    def test_edge_drag_moves_shared_vertices_once_without_moving_piece_transform(self) -> None:
        """隣接する複数辺を選択してドラッグ移動しても共有頂点が二重移動せず piece.transform も変わらないこと."""
        rect = self._add_rect(0.0, 0.0, 0.2, 0.2, name="Rect")
        outer = rect.get_outer_path()
        assert outer is not None

        # 下辺 (nodes[0]->nodes[1]) と右辺 (nodes[1]->nodes[2]) を Shift 複数選択
        er_bottom = EdgeRange(
            piece_id=rect.id,
            path_id=outer.id,
            start_node_id=outer.nodes[0].id,
            end_node_id=outer.nodes[1].id,
        )
        er_right = EdgeRange(
            piece_id=rect.id,
            path_id=outer.id,
            start_node_id=outer.nodes[1].id,
            end_node_id=outer.nodes[2].id,
        )
        self.ctrl.selection.select_edge(er_bottom)
        self.ctrl.selection.select_edge(er_right, add=True)

        # 右辺中点 (0.2, 0.1) -> screen (200, 100) を押して (+0.05, +0.02) 移動
        self.ctrl.handle_event(EditorEvent(screen_pos=(200.0, 100.0), is_press=True, button="LEFT"))
        self.ctrl.handle_event(EditorEvent(screen_pos=(250.0, 120.0), is_move=True))
        self.ctrl.handle_event(EditorEvent(screen_pos=(250.0, 120.0), is_release=True, button="LEFT"))

        self.assertAlmostEqual(rect.transform.tx, 0.0, places=5)
        self.assertAlmostEqual(rect.transform.ty, 0.0, places=5)
        # nodes[0], nodes[1](共有頂点!), nodes[2] がぴったり +0.05, +0.02 移動し、nodes[3] は不変
        self.assertAlmostEqual(outer.nodes[0].position[0], 0.05, places=5)
        self.assertAlmostEqual(outer.nodes[0].position[1], 0.02, places=5)
        self.assertAlmostEqual(outer.nodes[1].position[0], 0.25, places=5)
        self.assertAlmostEqual(outer.nodes[1].position[1], 0.02, places=5)
        self.assertAlmostEqual(outer.nodes[2].position[0], 0.25, places=5)
        self.assertAlmostEqual(outer.nodes[2].position[1], 0.22, places=5)
        self.assertAlmostEqual(outer.nodes[3].position[0], 0.0, places=5)
        self.assertAlmostEqual(outer.nodes[3].position[1], 0.2, places=5)

    def test_multi_node_pending_click_shift_toggle_and_single_reference_snap(self) -> None:
        """複数頂点選択時のクリック遅延確定・Shiftトグル解除・単一基準点スナップによる形状維持."""
        rect = self._add_rect(0.0, 0.0, 0.2, 0.2, name="Rect")
        outer = rect.get_outer_path()
        assert outer is not None
        n2, n3 = outer.nodes[2], outer.nodes[3]  # (0.2, 0.2) と (0.0, 0.2)

        self.ctrl.snap_settings.enabled = True
        self.ctrl.snap_settings.snap_grid = True
        self.ctrl.snap_settings.grid_step = 0.05
        self.ctrl.snap_settings.snap_nodes = False

        self.ctrl.selection.select_node(rect.id, n2.id)
        self.ctrl.selection.select_node(rect.id, n3.id, add=True)
        self.assertEqual(self.ctrl.selection.node_ids, {n2.id, n3.id})

        # 1. n2 (200, 200) を押しただけでは即リセットされず、ドラッグ (247, 252 -> snap(0.25, 0.25)) で両頂点が等量移動
        self.ctrl.handle_event(EditorEvent(screen_pos=(200.0, 200.0), is_press=True, button="LEFT"))
        self.assertEqual(self.ctrl.selection.node_ids, {n2.id, n3.id})
        self.ctrl.handle_event(EditorEvent(screen_pos=(247.0, 252.0), is_move=True))
        self.ctrl.handle_event(EditorEvent(screen_pos=(247.0, 252.0), is_release=True, button="LEFT"))

        self.assertEqual(self.ctrl.selection.node_ids, {n2.id, n3.id})
        self.assertAlmostEqual(n2.position[0] - n3.position[0], 0.20, places=5)
        self.assertAlmostEqual(n2.position[1], n3.position[1], places=5)
        self.assertAlmostEqual(n2.position[1], 0.25, places=5)

        # 2. Shift+クリック（閾値 3px 未満）で離上するとその1頂点だけがトグル解除されること
        s_n2 = self.view.world_to_screen(n2.position)
        self.ctrl.handle_event(EditorEvent(screen_pos=s_n2, is_press=True, button="LEFT", shift=True))
        self.ctrl.handle_event(EditorEvent(screen_pos=s_n2, is_release=True, button="LEFT", shift=True))
        self.assertEqual(self.ctrl.selection.node_ids, {n3.id})

        # 3. 再度2頂点選び直して、Shiftなしの単発クリック離上でその1点のみへ絞られること
        self.ctrl.selection.select_node(rect.id, n2.id, add=True)
        self.ctrl.handle_event(EditorEvent(screen_pos=s_n2, is_press=True, button="LEFT"))
        self.ctrl.handle_event(EditorEvent(screen_pos=(s_n2[0] + 1.0, s_n2[1]), is_release=True, button="LEFT"))
        self.assertEqual(self.ctrl.selection.node_ids, {n2.id})

    def test_info_card_displays_for_nodes_and_edges_without_piece_ids(self) -> None:
        """piece_ids が空の頂点/辺選択状態でも No Selection にならないこと."""
        rect = self._add_rect(0.0, 0.0, 0.2, 0.2, name="Panel")
        outer = rect.get_outer_path()
        assert outer is not None

        self.ctrl.selection.select_node(rect.id, outer.nodes[0].id)
        lines_one = format_selection_card_lines(self.ctrl.doc, self.ctrl.selection)
        self.assertFalse(any("No Selection" in ln or "選択なし" in ln for ln in lines_one))

        self.ctrl.selection.select_node(rect.id, outer.nodes[1].id, add=True)
        lines_multi = format_selection_card_lines(self.ctrl.doc, self.ctrl.selection)
        self.assertTrue(len(lines_multi) >= 7)
        self.assertTrue(any("Align X" in ln or "X整列" in ln for ln in lines_multi))

        self.ctrl.selection.select_edge(
            EdgeRange(piece_id=rect.id, path_id=outer.id, start_node_id=outer.nodes[0].id, end_node_id=outer.nodes[1].id)
        )
        self.ctrl.selection.select_edge(
            EdgeRange(piece_id=rect.id, path_id=outer.id, start_node_id=outer.nodes[1].id, end_node_id=outer.nodes[2].id),
            add=True,
        )
        lines_edges = format_selection_card_lines(self.ctrl.doc, self.ctrl.selection)
        self.assertTrue(any("2" in ln for ln in lines_edges))

    # ------------------------------------------------------------------
    # Phase 2: キーマップ整理・モーダル移動 (G/X/Y/N/数値/Shift)・全選択・ボックス交差
    # ------------------------------------------------------------------

    def test_keymap_restricts_to_digits_only_and_f_flips_seam(self) -> None:
        """ツール切替ホットキーは数字キー 1-9 のみ有効で、縫合反転は F のみで動作すること."""
        self.assertEqual(resolve_tool_hotkey("ONE"), "SELECT")
        self.assertEqual(resolve_tool_hotkey("TWO"), "RECTANGLE")
        self.assertEqual(resolve_tool_hotkey("NINE"), "DART")
        for letter in ("Q", "V", "R", "P", "E", "I", "N", "S", "J", "K", "D", "F"):
            self.assertIsNone(resolve_tool_hotkey(letter))

        # F のみで縫合方向が反転し、R では反転しない
        r1 = self._add_rect(0.0, 0.0, 0.2, 0.2, name="A")
        r2 = self._add_rect(0.3, 0.0, 0.5, 0.2, name="B")
        o1, o2 = r1.get_outer_path(), r2.get_outer_path()
        assert o1 is not None and o2 is not None
        seam = Seam(
            id=new_id(),
            a=EdgeRange(piece_id=r1.id, path_id=o1.id, start_node_id=o1.nodes[1].id, end_node_id=o1.nodes[2].id),
            b=EdgeRange(piece_id=r2.id, path_id=o2.id, start_node_id=o2.nodes[3].id, end_node_id=o2.nodes[0].id),
            reverse=False,
        )
        self.ctrl.doc.seams[seam.id] = seam
        self.ctrl.selection.select_seam(seam.id)
        self.ctrl.handle_event(EditorEvent(key="R"))
        self.assertFalse(seam.reverse)
        self.ctrl.handle_event(EditorEvent(key="F"))
        self.assertTrue(seam.reverse)

    def test_modal_move_axes_numeric_units_ortho_and_cancel(self) -> None:
        """G モーダル移動の X/Y ロック、単位数値入力 (cm/mm/m)、数字誤爆防止、ESCキャンセルとUndo."""
        rect = self._add_rect(0.0, 0.0, 0.2, 0.2, name="Rect")
        pid = rect.id
        self.ctrl.selection.select_piece(pid)

        # 1. G -> X -> "5" (5cm) -> Enter で確定
        self.assertTrue(self.ctrl.handle_event(EditorEvent(key="G")))
        self.assertTrue(self.ctrl.is_modal_input_active)
        self.assertTrue(self.ctrl.handle_event(EditorEvent(key="X")))
        self.assertTrue(self.ctrl.handle_event(EditorEvent(key="FIVE")))
        self.assertTrue(self.ctrl.handle_event(EditorEvent(key="ENTER")))
        self.assertFalse(self.ctrl.is_modal_input_active)
        self.assertAlmostEqual(self.ctrl.doc.pieces[pid].transform.tx, 0.05, places=5)
        self.assertAlmostEqual(self.ctrl.doc.pieces[pid].transform.ty, 0.0, places=5)

        # Undo 検証 (undo() は self.ctrl.doc を差し替えるため doc から再取得する)
        self.ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        self.assertAlmostEqual(self.ctrl.doc.pieces[pid].transform.tx, 0.0, places=5)

        # 2. G -> Y -> "15mm" -> 確定 (+0.015m)
        self.ctrl.selection.select_piece(pid)
        self.ctrl.handle_event(EditorEvent(key="G"))
        self.ctrl.handle_event(EditorEvent(key="Y"))
        for ch in ("1", "5", "m", "m"):
            self.ctrl.handle_event(EditorEvent(key=ch.upper(), unicode=ch))
        self.ctrl.handle_event(EditorEvent(key="RETURN"))
        self.assertAlmostEqual(self.ctrl.doc.pieces[pid].transform.ty, 0.015, places=5)

        # 3. G -> ドラッグ + ESC でキャンセル復元
        self.ctrl.handle_event(EditorEvent(key="G"))
        self.ctrl.handle_event(EditorEvent(world_pos=(0.3, 0.4), is_move=True))
        self.assertNotAlmostEqual(self.ctrl.doc.pieces[pid].transform.ty, 0.015, places=2)
        self.ctrl.handle_event(EditorEvent(key="ESCAPE"))
        self.assertAlmostEqual(self.ctrl.doc.pieces[pid].transform.ty, 0.015, places=5)
        self.assertIsNone(self.ctrl.transform_session)

        # 4. Shift 押下ドラッグの一時直交クランプ (X方向優位なら dy 不変)
        self.ctrl.handle_event(EditorEvent(world_pos=(0.0, 0.0), is_move=True))
        self.ctrl.handle_event(EditorEvent(key="G"))
        self.ctrl.handle_event(EditorEvent(world_pos=(0.08, 0.02), is_move=True, shift=True))
        self.ctrl.handle_event(EditorEvent(is_press=True, button="LEFT"))
        self.assertAlmostEqual(self.ctrl.doc.pieces[pid].transform.tx, 0.08, places=5)
        self.assertAlmostEqual(self.ctrl.doc.pieces[pid].transform.ty, 0.015, places=5)

    def test_modal_move_normal_single_and_miter_multi_edges(self) -> None:
        """G + N の単一辺法線および隣接2辺マイター法線・アクティブ法線トグル."""
        rect = self._add_rect(0.0, 0.0, 0.2, 0.2, name="Rect")
        pid = rect.id
        outer = rect.get_outer_path()
        assert outer is not None

        er0 = EdgeRange(piece_id=pid, path_id=outer.id, start_node_id=outer.nodes[0].id, end_node_id=outer.nodes[1].id)
        er1 = EdgeRange(piece_id=pid, path_id=outer.id, start_node_id=outer.nodes[1].id, end_node_id=outer.nodes[2].id)
        self.ctrl.selection.select_edge(er0)
        self.ctrl.selection.select_edge(er1, add=True)

        # 1. N 1回目 (NORMAL: マイターオフセット) + "2" (2cm)
        self.ctrl.handle_event(EditorEvent(key="G"))
        self.ctrl.handle_event(EditorEvent(key="N"))
        assert self.ctrl.transform_session is not None
        self.assertEqual(self.ctrl.transform_session.axis_lock, "NORMAL")

        cmds = build_render_commands(self.ctrl.doc, self.ctrl, screen_size=(1280, 720))
        self.assertTrue(any(c.kind == DrawCommandKind.LINE and c.color == (0.25, 0.88, 0.95, 0.85) for c in cmds))

        self.ctrl.handle_event(EditorEvent(key="TWO", unicode="2"))
        self.ctrl.handle_event(EditorEvent(key="ENTER"))

        outer_cur = self.ctrl.doc.pieces[pid].get_outer_path()
        assert outer_cur is not None
        self.assertAlmostEqual(outer_cur.nodes[0].position[1], -0.02, places=5)
        self.assertAlmostEqual(outer_cur.nodes[1].position[0], 0.22, places=5)
        self.assertAlmostEqual(outer_cur.nodes[1].position[1], -0.02, places=5)
        self.assertAlmostEqual(outer_cur.nodes[2].position[0], 0.22, places=5)

        # Undo して 2回目 NORMAL_ACTIVE を検証
        self.ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        outer_u = self.ctrl.doc.pieces[pid].get_outer_path()
        assert outer_u is not None
        er0_u = EdgeRange(piece_id=pid, path_id=outer_u.id, start_node_id=outer_u.nodes[0].id, end_node_id=outer_u.nodes[1].id)
        er1_u = EdgeRange(piece_id=pid, path_id=outer_u.id, start_node_id=outer_u.nodes[1].id, end_node_id=outer_u.nodes[2].id)
        self.ctrl.selection.select_edge(er0_u)
        self.ctrl.selection.select_edge(er1_u, add=True)

        self.ctrl.handle_event(EditorEvent(key="G"))
        self.ctrl.handle_event(EditorEvent(key="N"))  # NORMAL
        self.ctrl.handle_event(EditorEvent(key="N"))  # NORMAL_ACTIVE (active=er1=(1,0))
        assert self.ctrl.transform_session is not None
        self.assertEqual(self.ctrl.transform_session.axis_lock, "NORMAL_ACTIVE")
        self.ctrl.handle_event(EditorEvent(key="TWO", unicode="2"))
        self.ctrl.handle_event(EditorEvent(key="ENTER"))

        outer_after = self.ctrl.doc.pieces[pid].get_outer_path()
        assert outer_after is not None
        self.assertAlmostEqual(outer_after.nodes[0].position[0], 0.02, places=5)
        self.assertAlmostEqual(outer_after.nodes[0].position[1], 0.0, places=5)
        self.assertAlmostEqual(outer_after.nodes[1].position[0], 0.22, places=5)
        self.assertAlmostEqual(outer_after.nodes[1].position[1], 0.0, places=5)

    def test_select_all_deselect_all_and_box_edge_intersection(self) -> None:
        """A 全選択 / Alt+A 全解除、および頂点を含まない空白ボックスの辺交差選択."""
        r1 = self._add_rect(0.0, 0.0, 0.2, 0.2, name="P1")
        r2 = self._add_rect(0.4, 0.0, 0.6, 0.2, name="P2")
        o1 = r1.get_outer_path()
        assert o1 is not None

        # A: 全ピース選択
        self.ctrl.handle_event(EditorEvent(key="A"))
        self.assertEqual(self.ctrl.selection.piece_ids, {r1.id, r2.id})

        # Alt+A: 全解除
        self.ctrl.handle_event(EditorEvent(key="A", alt=True))
        self.assertFalse(self.ctrl.selection.has_any_selection())

        # 頂点選択中の A: 対象ピースの全頂点を選択
        self.ctrl.selection.select_node(r1.id, o1.nodes[0].id)
        self.ctrl.handle_event(EditorEvent(key="A"))
        self.assertEqual(len(self.ctrl.selection.node_ids), 4)
        self.assertEqual(len(self.ctrl.selection.piece_ids), 0)

        # Alt+A 後に、P1 の上辺 (y=0.2, x in [0.05..0.15]) だけを横切る矩形を空白からドラッグ
        self.ctrl.handle_event(EditorEvent(key="A", alt=True))
        self.ctrl.handle_event(EditorEvent(screen_pos=(80.0, 220.0), is_press=True, button="LEFT"))
        self.ctrl.handle_event(EditorEvent(screen_pos=(120.0, 180.0), is_move=True))
        self.ctrl.handle_event(EditorEvent(screen_pos=(120.0, 180.0), is_release=True, button="LEFT"))

        self.assertEqual(len(self.ctrl.selection.node_ids), 0)
        self.assertEqual(len(self.ctrl.selection.edge_ranges), 1)
        self.assertEqual(self.ctrl.selection.edge_ranges[0].piece_id, r1.id)
        self.assertEqual(
            {self.ctrl.selection.edge_ranges[0].start_node_id, self.ctrl.selection.edge_ranges[0].end_node_id},
            {o1.nodes[2].id, o1.nodes[3].id},
        )

    # ------------------------------------------------------------------
    # Phase 3: 整列・一括分割・回転/拡縮・複数ピース一括編集
    # ------------------------------------------------------------------

    def test_multi_node_alignment_card_clicks_and_renderer_buttons(self) -> None:
        """複数頂点 X/Y 整列 (Min / Center / Avg / Max) のカードボタン描画とクリック動作・Undo."""
        rect = self._add_rect(0.0, 0.0, 0.2, 0.2, name="Rect")
        pid = rect.id
        outer = rect.get_outer_path()
        assert outer is not None
        outer.nodes[0].position = (0.0, 0.0)
        outer.nodes[1].position = (0.1, 0.0)
        outer.nodes[2].position = (0.4, 0.2)
        outer.nodes[3].position = (0.0, 0.3)

        # 3点 (x = 0.0, 0.1, 0.4) を選択
        self.ctrl.selection.select_node(pid, outer.nodes[0].id)
        self.ctrl.selection.select_node(pid, outer.nodes[1].id, add=True)
        self.ctrl.selection.select_node(pid, outer.nodes[2].id, add=True)

        # 整列ボタン描画 (Min/Ctr/Avg/Max の8個) 確認
        cmds = build_render_commands(self.ctrl.doc, self.ctrl, screen_size=(1280, 720))
        btn_texts = [c.text for c in cmds if c.kind == DrawCommandKind.TEXT and c.text in ("Min", "Ctr", "Avg", "Max")]
        self.assertEqual(len(btn_texts), 8)

        # Center カードボタンクリック -> x = (0.0 + 0.4)/2 = 0.2
        layout = compute_hud_layout(*self.ctrl.screen_size)
        ctr_rect = layout.card_fields["align_x_center"]
        self.ctrl.handle_event(EditorEvent(screen_pos=(ctr_rect.x + 2.0, ctr_rect.y + 2.0), is_press=True, button="LEFT"))
        outer_c = self.ctrl.doc.pieces[pid].get_outer_path()
        assert outer_c is not None
        self.assertAlmostEqual(outer_c.nodes[0].position[0], 0.2, places=5)
        self.assertAlmostEqual(outer_c.nodes[1].position[0], 0.2, places=5)
        self.assertAlmostEqual(outer_c.nodes[2].position[0], 0.2, places=5)

        # Undo してノード再選択 -> Avg -> (0.0 + 0.1 + 0.4) / 3 = 0.5 / 3
        self.ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        outer_u1 = self.ctrl.doc.pieces[pid].get_outer_path()
        assert outer_u1 is not None
        self.ctrl.selection.select_node(pid, outer_u1.nodes[0].id)
        self.ctrl.selection.select_node(pid, outer_u1.nodes[1].id, add=True)
        self.ctrl.selection.select_node(pid, outer_u1.nodes[2].id, add=True)
        avg_rect = layout.card_fields["align_x_avg"]
        self.ctrl.handle_event(EditorEvent(screen_pos=(avg_rect.x + 2.0, avg_rect.y + 2.0), is_press=True, button="LEFT"))
        expected_avg = 0.5 / 3.0
        self.assertAlmostEqual(outer_u1.nodes[0].position[0], expected_avg, places=5)
        self.assertAlmostEqual(outer_u1.nodes[2].position[0], expected_avg, places=5)

        # Y Max カードボタンクリック -> y = 0.2
        self.ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        outer_u2 = self.ctrl.doc.pieces[pid].get_outer_path()
        assert outer_u2 is not None
        self.ctrl.selection.select_node(pid, outer_u2.nodes[0].id)
        self.ctrl.selection.select_node(pid, outer_u2.nodes[1].id, add=True)
        self.ctrl.selection.select_node(pid, outer_u2.nodes[2].id, add=True)
        ymax_rect = layout.card_fields["align_y_max"]
        self.ctrl.handle_event(EditorEvent(screen_pos=(ymax_rect.x + 2.0, ymax_rect.y + 2.0), is_press=True, button="LEFT"))
        self.assertAlmostEqual(outer_u2.nodes[0].position[1], 0.2, places=5)
        self.assertAlmostEqual(outer_u2.nodes[1].position[1], 0.2, places=5)

    def test_multi_edge_subdivide_w_key_and_undo(self) -> None:
        """複数の選択辺を W キーで1トランザクション一括分割でき、Ctrl+Z で一括 Undo できること."""
        rect = self._add_rect(0.0, 0.0, 0.2, 0.2, name="Rect")
        pid = rect.id
        outer = rect.get_outer_path()
        assert outer is not None

        er0 = EdgeRange(piece_id=pid, path_id=outer.id, start_node_id=outer.nodes[0].id, end_node_id=outer.nodes[1].id)
        er1 = EdgeRange(piece_id=pid, path_id=outer.id, start_node_id=outer.nodes[1].id, end_node_id=outer.nodes[2].id)
        self.ctrl.selection.select_edge(er0)
        self.ctrl.selection.select_edge(er1, add=True)

        self.assertTrue(self.ctrl.handle_event(EditorEvent(key="W")))
        self.assertEqual(len(outer.nodes), 6)
        self.assertEqual(len(self.ctrl.selection.node_ids), 2)
        self.assertEqual(len(self.ctrl.selection.edge_ranges), 0)

        # 1回の Ctrl+Z で 4頂点に戻る
        self.ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        outer_after_undo = self.ctrl.doc.pieces[pid].get_outer_path()
        assert outer_after_undo is not None
        self.assertEqual(len(outer_after_undo.nodes), 4)

    def test_modal_rotate_and_scale_local_coords_ccw_zero_guard_and_3d_build(self) -> None:
        """2D回転/拡縮でローカル座標・ベジェハンドルが更新され3Dメッシュ構築に実反映、負スケールCCW維持・ゼロガード."""
        rect = self._add_rect(0.0, 0.0, 0.2, 0.1, name="Sleeve")
        outer = rect.get_outer_path()
        assert outer is not None
        # 単一頂点のみで R / S は起動拒否されること
        self.ctrl.selection.select_node(rect.id, outer.nodes[0].id)
        self.ctrl.handle_event(EditorEvent(key="R"))
        self.assertIsNone(self.ctrl.transform_session)
        self.assertTrue(bool(self.ctrl.status_message))

        mesh_orig = build_garment_mesh_geometry(self.ctrl.doc)
        orig_x_span = max(v[0] for v in mesh_orig.vertices_3d) - min(v[0] for v in mesh_orig.vertices_3d)
        orig_y_span = max(v[1] for v in mesh_orig.vertices_3d) - min(v[1] for v in mesh_orig.vertices_3d)

        # S -> "2" -> ENTER (2倍拡大: 案Aによりローカル座標と3Dサイズが2倍)
        self.ctrl.selection.select_piece(rect.id)
        self.assertTrue(self.ctrl.handle_event(EditorEvent(key="S")))
        cmds = build_render_commands(self.ctrl.doc, self.ctrl, screen_size=(1280, 720))
        self.assertTrue(any(c.kind == DrawCommandKind.CIRCLE and c.radius == 5.0 for c in cmds))

        self.ctrl.handle_event(EditorEvent(key="TWO", unicode="2"))
        self.ctrl.handle_event(EditorEvent(key="ENTER"))
        self.assertAlmostEqual(rect.transform.sx, 1.0, places=5)
        xs = [n.position[0] for n in outer.nodes]
        ys = [n.position[1] for n in outer.nodes]
        self.assertAlmostEqual(max(xs) - min(xs), 0.4, places=4)
        self.assertAlmostEqual(max(ys) - min(ys), 0.2, places=4)

        mesh_scaled = build_garment_mesh_geometry(self.ctrl.doc)
        new_x_span = max(v[0] for v in mesh_scaled.vertices_3d) - min(v[0] for v in mesh_scaled.vertices_3d)
        new_y_span = max(v[1] for v in mesh_scaled.vertices_3d) - min(v[1] for v in mesh_scaled.vertices_3d)
        self.assertAlmostEqual(new_x_span, orig_x_span * 2.0, places=3)
        self.assertAlmostEqual(new_y_span, orig_y_span * 2.0, places=3)

        # R -> "90" -> ENTER で 90°回転 + ハンドル回転
        outer.nodes[0].out_handle = (0.04, 0.0)
        self.ctrl.handle_event(EditorEvent(key="R"))
        self.ctrl.handle_event(EditorEvent(key="NINE", unicode="9"))
        self.ctrl.handle_event(EditorEvent(key="ZERO", unicode="0"))
        self.ctrl.handle_event(EditorEvent(key="ENTER"))
        xs_r = [n.position[0] for n in outer.nodes]
        ys_r = [n.position[1] for n in outer.nodes]
        self.assertAlmostEqual(max(xs_r) - min(xs_r), 0.2, places=4)
        self.assertAlmostEqual(max(ys_r) - min(ys_r), 0.4, places=4)
        self.assertIsNotNone(outer.nodes[0].out_handle)
        self.assertAlmostEqual(outer.nodes[0].out_handle[0], 0.0, places=4)
        self.assertAlmostEqual(outer.nodes[0].out_handle[1], 0.04, places=4)

        # S -> X -> "-1" -> ENTER (負スケール反転後も CCW 維持 & 3D ビルド成功)
        self.ctrl.handle_event(EditorEvent(key="S"))
        self.ctrl.handle_event(EditorEvent(key="X"))
        self.ctrl.handle_event(EditorEvent(key="MINUS", unicode="-"))
        self.ctrl.handle_event(EditorEvent(key="ONE", unicode="1"))
        self.ctrl.handle_event(EditorEvent(key="ENTER"))
        self.assertGreater(signed_area(outer.sample_outline_points(steps_per_segment=8)), 0.0)
        mesh_mirrored = build_garment_mesh_geometry(self.ctrl.doc)
        self.assertGreater(len(mesh_mirrored.vertices_3d), 0)

        # S -> "0" -> ENTER ゼロスケール確定抑止ガード
        self.ctrl.handle_event(EditorEvent(key="S"))
        self.ctrl.handle_event(EditorEvent(key="ZERO", unicode="0"))
        self.ctrl.handle_event(EditorEvent(key="ENTER"))
        self.assertIsNone(self.ctrl.transform_session)
        self.assertGreater(signed_area(outer.sample_outline_points(steps_per_segment=8)), 0.01)

    def test_multi_piece_mixed_card_and_bulk_edit(self) -> None:
        """複数ピース選択時の Mixed 表示、インライン編集・スライダー・コンパス・ボーンの一括適用とUndo."""
        p1 = self._add_rect(0.0, 0.0, 0.2, 0.2, name="Front")
        p2 = self._add_rect(0.3, 0.0, 0.5, 0.2, name="Back")
        p1.layer_id = 1
        p2.layer_id = 2
        p1.target_edge_length = 0.02
        p2.target_edge_length = 0.02
        self.doc.placements[p1.id] = Placement(piece_id=p1.id, target_bone="Spine", bone_t=0.2)
        self.doc.placements[p2.id] = Placement(piece_id=p2.id, target_bone="Spine", bone_t=0.8)

        self.ctrl.selection.select_piece(p1.id)
        self.ctrl.selection.select_piece(p2.id, add=True)

        lines = format_selection_card_lines(self.doc, self.ctrl.selection)
        self.assertTrue(any("2.0" in ln for ln in lines))
        self.assertTrue(any("Spine" in ln for ln in lines))
        self.assertTrue(any("Mixed" in ln or "混在" in ln for ln in lines))

        layout = compute_hud_layout(*self.ctrl.screen_size)

        # 1. Layer カードフィールドをクリック -> BackSpace で既存バッファクリア -> "3" 入力
        lr = layout.card_fields["layer_id"]
        self.ctrl.handle_event(EditorEvent(screen_pos=(lr.x + 5.0, lr.y + 5.0), is_press=True, button="LEFT"))
        self.assertEqual(self.ctrl.editing_field, "layer_id")
        self.ctrl.handle_event(EditorEvent(key="BACK_SPACE"))
        self.ctrl.handle_event(EditorEvent(key="THREE", unicode="3"))
        self.ctrl.handle_event(EditorEvent(key="ENTER"))
        self.assertEqual(p1.layer_id, 3)
        self.assertEqual(p2.layer_id, 3)

        # 2. スライダードラッグ (bone_t) でアクティブ(p2: 0.8)基準に左へ -40px (-0.2) スライドし両ピース 0.6 に揃うこと
        bt_rect = layout.card_fields["bone_t"]
        self.ctrl.handle_event(EditorEvent(screen_pos=(bt_rect.x + 50.0, bt_rect.y + 5.0), is_press=True, button="LEFT"))
        self.ctrl.handle_event(EditorEvent(screen_pos=(bt_rect.x + 10.0, bt_rect.y + 5.0), is_move=True))
        self.ctrl.handle_event(EditorEvent(screen_pos=(bt_rect.x + 10.0, bt_rect.y + 5.0), is_release=True, button="LEFT"))
        self.assertAlmostEqual(self.doc.placements[p1.id].bone_t, self.doc.placements[p2.id].bone_t, places=5)
        self.assertAlmostEqual(self.doc.placements[p1.id].bone_t, 0.6, places=5)

        # 3. コンパスドラッグで両ピースの bone_angle_deg が一括更新されること
        comp = layout.compass_rect
        self.ctrl.handle_event(
            EditorEvent(screen_pos=(comp.x + comp.w * 0.5 + 20.0, comp.y + comp.h * 0.5), is_press=True, button="LEFT")
        )
        self.ctrl.handle_event(
            EditorEvent(screen_pos=(comp.x + comp.w * 0.5 + 20.0, comp.y + comp.h * 0.5), is_release=True, button="LEFT")
        )
        self.assertAlmostEqual(self.doc.placements[p1.id].bone_angle_deg, 90.0, places=3)
        self.assertAlmostEqual(self.doc.placements[p2.id].bone_angle_deg, 90.0, places=3)

        # 4. Pick Bone モードで両ピースの target_bone が一括更新され、1回 Undo で戻ること
        tb_rect = layout.card_fields["target_bone"]
        self.ctrl.handle_event(EditorEvent(screen_pos=(tb_rect.x + 5.0, tb_rect.y + 5.0), is_press=True, button="LEFT"))
        self.assertTrue(self.ctrl.picking_bone)
        self.ctrl.hovered_bone = "Chest"
        self.ctrl.handle_event(EditorEvent(screen_pos=(300.0, 300.0), is_press=True, button="LEFT"))
        self.assertEqual(self.ctrl.doc.placements[p1.id].target_bone, "Chest")
        self.assertEqual(self.ctrl.doc.placements[p2.id].target_bone, "Chest")

        self.ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        self.assertEqual(self.ctrl.doc.placements[p1.id].target_bone, "Spine")
        self.assertEqual(self.ctrl.doc.placements[p2.id].target_bone, "Spine")


if __name__ == "__main__":
    unittest.main()
