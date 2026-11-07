# -*- coding: utf-8 -*-
"""Phase 2 Step 2.2: 内部線、合印ノッチ、寸法線描画の単体テスト."""

import unittest

from taremin_cloth.pattern.document import Document, Node, NodeKind, NotchKind, Path, PathRole
from taremin_cloth.pattern.editor import EditorController, EditorEvent
from taremin_cloth.pattern.ids import new_id
from taremin_cloth.pattern.renderer import DrawCommandKind, build_render_commands
from taremin_cloth.pattern.selection import HitTargetKind, hit_test
from taremin_cloth.pattern.session import ViewTransform2D


class TestPatternInternalNotch(unittest.TestCase):
    """内部線ツール、ノッチツール、寸法線表示のテスト."""

    def setUp(self) -> None:
        self.doc = Document.new()
        # 0.3m x 0.4m の矩形ピースを作成
        self.piece = self.doc.add_rectangle_piece((0.0, 0.0), (0.3, 0.4), name="Front")
        view = ViewTransform2D(pan_x=0.0, pan_y=0.0, zoom=1000.0)
        self.ctrl = EditorController(doc=self.doc, view=view)

    def _event(self, world_pos: tuple[float, float], **kwargs) -> EditorEvent:
        screen_pos = self.ctrl.view.world_to_screen(world_pos)
        return EditorEvent(screen_pos=screen_pos, world_pos=world_pos, **kwargs)

    def test_internal_line_creation(self) -> None:
        """内部線ツールでピース内に折れ線を作成し、Undo/Redo できることを検証."""
        self.ctrl.set_active_tool("INTERNAL_LINE")

        # ピース内部の2点をクリック
        p1 = (0.05, 0.1)
        p2 = (0.25, 0.1)
        # 1点目
        self.ctrl.handle_event(self._event(p1, is_press=True, button="LEFT"))
        # 2点目
        self.ctrl.handle_event(self._event(p2, is_press=True, button="LEFT"))
        # Enterで確定
        self.ctrl.handle_event(EditorEvent(key="ENTER"))

        # 内部パスが追加されたか確認
        internal_paths = [
            p for p in self.piece.paths.values()
            if p.role == PathRole.INTERNAL
        ]
        self.assertEqual(len(internal_paths), 1)
        int_path = internal_paths[0]
        self.assertEqual(len(int_path.nodes), 2)
        self.assertFalse(int_path.closed)
        self.assertAlmostEqual(int_path.nodes[0].position[0], 0.05)
        self.assertAlmostEqual(int_path.nodes[1].position[0], 0.25)

        # Undo で削除されることを確認
        self.ctrl.undo()
        internal_paths_after_undo = [
            p for p in self.ctrl.doc.pieces[self.piece.id].paths.values()
            if p.role == PathRole.INTERNAL
        ]
        self.assertEqual(len(internal_paths_after_undo), 0)

        # Redo で復活することを確認
        self.ctrl.redo()
        internal_paths_after_redo = [
            p for p in self.ctrl.doc.pieces[self.piece.id].paths.values()
            if p.role == PathRole.INTERNAL
        ]
        self.assertEqual(len(internal_paths_after_redo), 1)

    def test_notch_creation(self) -> None:
        """合印ノッチツールで外周の辺上にノッチを追加できることを検証."""
        self.ctrl.set_active_tool("NOTCH")

        # 矩形ピースの下辺 y=0.0, x: 0.0 -> 0.3 の中央 (0.15, 0.0) 付近をクリック
        click_pos = (0.15, 0.001)
        self.ctrl.handle_event(self._event(click_pos, is_press=True, button="LEFT"))

        # ノッチが作成されたか確認
        self.assertEqual(len(self.ctrl.doc.notches), 1)
        notch = list(self.ctrl.doc.notches.values())[0]
        self.assertEqual(notch.piece_id, self.piece.id)
        self.assertEqual(notch.kind, NotchKind.V)
        self.assertAlmostEqual(notch.t, 0.5, delta=0.05)

        # ノッチが選択状態になっていること
        self.assertTrue(self.ctrl.selection.is_notch_selected(notch.id))

    def test_notch_hit_and_delete(self) -> None:
        """ノッチをヒットテストで検出し、選択削除できることを検証."""
        self.ctrl.set_active_tool("NOTCH")
        # 上辺 (0.15, 0.4) 付近をクリックしてノッチ追加
        self.ctrl.handle_event(self._event((0.15, 0.399), is_press=True, button="LEFT"))
        self.assertEqual(len(self.ctrl.doc.notches), 1)
        notch_id = list(self.ctrl.doc.notches.keys())[0]

        # ツールを SELECT に切り替え、ノッチ位置をクリック
        self.ctrl.set_active_tool("SELECT")
        self.ctrl.selection.clear()

        # ヒットテスト確認
        hit = hit_test(self.ctrl.doc, (0.15, 0.4))
        self.assertIsNotNone(hit)
        self.assertEqual(hit.kind, HitTargetKind.NOTCH)
        self.assertEqual(hit.notch_id, notch_id)

        # クリックイベント
        self.ctrl.handle_event(self._event((0.15, 0.4), is_press=True, button="LEFT"))
        self.assertTrue(self.ctrl.selection.is_notch_selected(notch_id))

        # Xキーで削除
        self.ctrl.handle_event(EditorEvent(key="X"))
        self.assertEqual(len(self.ctrl.doc.notches), 0)

    def test_dimension_toggle_and_rendering(self) -> None:
        """寸法線のトグルおよびノッチ・内部線の描画コマンド生成を検証."""
        # 1. 内部線とノッチを追加
        self.ctrl.set_active_tool("INTERNAL_LINE")
        self.ctrl.handle_event(self._event((0.1, 0.1), is_press=True, button="LEFT"))
        self.ctrl.handle_event(self._event((0.2, 0.1), is_press=True, button="LEFT"))
        self.ctrl.handle_event(EditorEvent(key="ENTER"))

        self.ctrl.set_active_tool("NOTCH")
        self.ctrl.handle_event(self._event((0.15, 0.0), is_press=True, button="LEFT"))

        # 描画コマンド生成（show_lengths = True）
        self.ctrl.show_lengths = True
        cmds = build_render_commands(self.ctrl.doc, self.ctrl, screen_size=(800, 600))

        # 寸法線テキスト（TEXTコマンドかつdimension_text色）が存在すること
        from taremin_cloth.pattern.theme import DEFAULT_THEME
        dim_text_cmds = [
            c for c in cmds
            if c.kind == DrawCommandKind.TEXT and c.color == DEFAULT_THEME.dimension_text
        ]
        self.assertTrue(len(dim_text_cmds) >= 4)  # 4辺 + 内部線

        # ノッチ描画（POLYLINEコマンド）が存在すること
        poly_cmds = [c for c in cmds if c.kind == DrawCommandKind.POLYLINE]
        self.assertTrue(len(poly_cmds) >= 6)  # 4辺 + 内部線 + ノッチ

        # Lキーで寸法線トグル
        self.ctrl.handle_event(EditorEvent(key="L"))
        self.assertFalse(self.ctrl.show_lengths)

        # 描画コマンド再生成（show_lengths = False）
        cmds_no_dim = build_render_commands(self.ctrl.doc, self.ctrl, screen_size=(800, 600))
        dim_text_cmds_no_dim = [
            c for c in cmds_no_dim
            if c.kind == DrawCommandKind.TEXT and c.color == DEFAULT_THEME.dimension_text
        ]
        self.assertEqual(len(dim_text_cmds_no_dim), 0)

    def test_notch_deletion_does_not_delete_piece(self) -> None:
        """ノッチを削除した際に親ピースが削除されず保持されることを検証 (バグ防止)."""
        self.ctrl.set_active_tool("NOTCH")
        self.ctrl.handle_event(self._event((0.15, 0.0), is_press=True, button="LEFT"))
        self.assertEqual(len(self.ctrl.doc.notches), 1)
        self.assertEqual(len(self.ctrl.doc.pieces), 1)

        # Xキーでノッチを削除
        self.ctrl.handle_event(EditorEvent(key="X"))
        # ノッチは削除されている
        self.assertEqual(len(self.ctrl.doc.notches), 0)
        # しかしピースは確実に残っていること！
        self.assertEqual(len(self.ctrl.doc.pieces), 1)
        self.assertIn(self.piece.id, self.ctrl.doc.pieces)

    def test_c_key_with_piece_selected(self) -> None:
        """ピース選択時にCキーを押すと外周ノードのみが一括巡回し、内部線ノードは巻き込まれないことを検証."""
        # 内部線を追加
        int_nodes = [
            Node(id=new_id(), position=(0.1, 0.1), kind=NodeKind.CORNER),
            Node(id=new_id(), position=(0.2, 0.1), kind=NodeKind.CORNER),
        ]
        int_id = new_id()
        self.piece.paths[int_id] = Path(id=int_id, role=PathRole.INTERNAL, nodes=int_nodes, closed=False)

        self.ctrl.set_active_tool("SELECT")
        self.ctrl.selection.clear()
        self.ctrl.selection.select_piece(self.piece.id)

        # Cキー押下
        self.ctrl.handle_event(EditorEvent(key="C"))

        # 外周ノードは SMOOTH に切り替わっていること
        for nd in self.ctrl.doc.pieces[self.piece.id].paths[self.piece.outer_path_id].nodes:
            self.assertEqual(nd.kind, NodeKind.SMOOTH)

        # 内部線ノードは CORNER のまま維持されていること（巻き込み防止）
        for nd in self.ctrl.doc.pieces[self.piece.id].paths[int_id].nodes:
            self.assertEqual(nd.kind, NodeKind.CORNER)

    def test_internal_line_finish_with_ret_key(self) -> None:
        """Blenderの標準Enterキー名 'RET' で内部線が確定できることを検証 (実機 event.value=='PRESS' 互換)."""
        self.ctrl.set_active_tool("INTERNAL_LINE")
        self.ctrl.handle_event(self._event((0.05, 0.2), is_press=True, button="LEFT"))
        self.ctrl.handle_event(self._event((0.25, 0.2), is_press=True, button="LEFT"))
        # Blender実機ではキーボード押下時に is_press=True と key="RET" の両方が渡される
        res = self.ctrl.handle_event(EditorEvent(is_press=True, key="RET"))
        self.assertTrue(res)

        int_paths = [
            p for p in self.piece.paths.values()
            if p.role == PathRole.INTERNAL
        ]
        self.assertEqual(len(int_paths), 1)

    def test_c_key_with_is_press_flag(self) -> None:
        """Blender実機で is_press=True と key='C' が同時に渡された場合でも確実にノード種別が切り替わることを検証."""
        self.ctrl.set_active_tool("SELECT")
        self.ctrl.selection.clear()
        node_id = self.piece.paths[self.piece.outer_path_id].nodes[0].id
        self.ctrl.selection.select_node(self.piece.id, node_id)

        # 実機と同等のキー押下イベント
        res = self.ctrl.handle_event(EditorEvent(is_press=True, key="C"))
        self.assertTrue(res)

        from taremin_cloth.pattern.document import NodeKind
        first_node = self.ctrl.doc.pieces[self.piece.id].paths[self.piece.outer_path_id].nodes[0]
        self.assertEqual(first_node.kind, NodeKind.SMOOTH)


if __name__ == "__main__":
    unittest.main()
