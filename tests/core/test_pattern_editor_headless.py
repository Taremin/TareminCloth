# -*- coding: utf-8 -*-
"""型紙エディタのヘッドレスイベント列テスト (Blender非依存)."""

import unittest

from taremin_cloth.pattern.document import Document
from taremin_cloth.pattern.editor import EditorController, EditorEvent
from taremin_cloth.pattern.session import ViewTransform2D


class TestPatternEditorHeadless(unittest.TestCase):
    """EditorController に対するマウス・キーボードイベント列のヘッドレステスト."""

    def setUp(self):
        # 1m = 1000px, pan=(0, 0) -> screen(100, 200) = world(0.1, 0.2)
        view = ViewTransform2D(pan_x=0.0, pan_y=0.0, zoom=1000.0)
        self.controller = EditorController(doc=Document.new(), view=view)

    def test_draw_rectangle_tool_headless(self):
        ctrl = self.controller
        ctrl.set_active_tool("RECTANGLE")

        # 1. マウスダウン (0, 0) -> world (0.0, 0.0)
        ctrl.handle_event(EditorEvent(screen_pos=(0.0, 0.0), is_press=True))
        self.assertEqual(len(ctrl.preview_points), 2)

        # 2. マウス移動 (200, 300) -> world (0.2, 0.3)
        ctrl.handle_event(EditorEvent(screen_pos=(200.0, 300.0), is_move=True))
        self.assertEqual(ctrl.preview_points[1], (0.2, 0.3))

        # 3. マウスアップ
        ctrl.handle_event(EditorEvent(screen_pos=(200.0, 300.0), is_release=True))
        self.assertEqual(len(ctrl.doc.pieces), 1)
        self.assertEqual(len(ctrl.preview_points), 0)
        self.assertTrue(ctrl.history.can_undo)

        # 4. Undo (Ctrl+Z)
        ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        self.assertEqual(len(ctrl.doc.pieces), 0)

        # 5. Redo (Ctrl+Shift+Z)
        ctrl.handle_event(EditorEvent(ctrl=True, shift=True, key="Z"))
        self.assertEqual(len(ctrl.doc.pieces), 1)

    def test_draw_polygon_tool_headless(self):
        ctrl = self.controller
        ctrl.set_active_tool("POLYGON")

        # 3点をクリック
        ctrl.handle_event(EditorEvent(screen_pos=(0.0, 0.0), is_press=True))
        ctrl.handle_event(EditorEvent(screen_pos=(200.0, 0.0), is_press=True))
        ctrl.handle_event(EditorEvent(screen_pos=(100.0, 200.0), is_press=True))

        # Enter で閉じる
        ctrl.handle_event(EditorEvent(key="ENTER"))
        self.assertEqual(len(ctrl.doc.pieces), 1)
        poly = list(ctrl.doc.pieces.values())[0]
        outer_path = poly.paths[poly.outer_path_id]
        self.assertEqual(len(outer_path.nodes), 3)

    def test_sew_tool_headless(self):
        ctrl = self.controller
        # 2つの矩形を追加
        p1 = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.3), name="PieceA")
        p2 = ctrl.doc.add_rectangle_piece((0.5, 0.0), (0.7, 0.3), name="PieceB")

        ctrl.set_active_tool("SEW")

        # p1 の右辺 (x=0.2, y=0.15) -> screen (200, 150) をクリック
        ctrl.handle_event(EditorEvent(screen_pos=(200.0, 150.0), is_press=True))
        self.assertEqual(len(ctrl.preview_seam_candidates), 1)

        # p2 の左辺 (x=0.5, y=0.15) -> screen (500, 150) をクリック
        ctrl.handle_event(EditorEvent(screen_pos=(500.0, 150.0), is_press=True))
        self.assertEqual(len(ctrl.preview_seam_candidates), 0)
        self.assertEqual(len(ctrl.doc.seams), 1)

    def test_select_and_drag_headless(self):
        ctrl = self.controller
        p = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2))

        ctrl.set_active_tool("SELECT")

        # ピース内部 (x=0.1, y=0.1) -> screen (100, 100) をクリックしてドラッグ開始
        ctrl.handle_event(EditorEvent(screen_pos=(100.0, 100.0), is_press=True))
        self.assertTrue(ctrl.selection.is_piece_selected(p.id))

        # +0.05m (+50px) 移動
        ctrl.handle_event(EditorEvent(screen_pos=(150.0, 150.0), is_move=True))
        # マウスアップ
        ctrl.handle_event(EditorEvent(screen_pos=(150.0, 150.0), is_release=True))

        moved_piece = ctrl.doc.pieces[p.id]
        self.assertAlmostEqual(moved_piece.transform.tx, 0.05, places=3)
        self.assertAlmostEqual(moved_piece.transform.ty, 0.05, places=3)

    def test_delete_selected_headless(self):
        ctrl = self.controller
        p = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2))
        ctrl.selection.select_piece(p.id)

        # X キーで削除
        ctrl.handle_event(EditorEvent(key="X"))
        self.assertEqual(len(ctrl.doc.pieces), 0)


if __name__ == "__main__":
    unittest.main()
