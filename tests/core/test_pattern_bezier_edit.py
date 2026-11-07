# -*- coding: utf-8 -*-
"""Phase 2: ベジェ曲線編集 (Cキー巡回, ハンドルドラッグ) と楕円ツールの単体テスト."""

import unittest
import math

from taremin_cloth.pattern.document import Document, NodeKind
from taremin_cloth.pattern.editor import EditorController, EditorEvent
from taremin_cloth.pattern.session import ViewTransform2D


class TestPatternBezierEdit(unittest.TestCase):
    """ベジェ曲線編集と楕円ツールの動作検証テスト."""

    def setUp(self):
        # 1m = 1000px, 原点 (0, 0)
        view = ViewTransform2D(pan_x=0.0, pan_y=0.0, zoom=1000.0)
        self.ctrl = EditorController(doc=Document.new(), view=view)

    def test_add_ellipse_piece(self):
        doc = self.ctrl.doc
        piece = doc.add_ellipse_piece((0.5, 0.5), (0.2, 0.1), name="Cuff")
        self.assertEqual(piece.name, "Cuff")
        self.assertEqual(len(doc.pieces), 1)

        path = piece.paths[piece.outer_path_id]
        self.assertEqual(len(path.nodes), 4)
        for nd in path.nodes:
            self.assertEqual(nd.kind, NodeKind.SMOOTH)
            self.assertIsNotNone(nd.in_handle)
            self.assertIsNotNone(nd.out_handle)

        # 4区間の評価
        segs = doc.evaluate_path_segments(piece.id, piece.outer_path_id)
        self.assertEqual(len(segs), 4)

        # 楕円の周長 (Ramanujanの近似式: pi * (3(a+b) - sqrt((3a+b)(a+3b))))
        a, b = 0.2, 0.1
        approx_perimeter = math.pi * (3.0 * (a + b) - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))
        total_len = sum(tbl.total_length for _, _, tbl in segs)
        # 近似長との相対誤差が 0.5% 以内であること
        self.assertAlmostEqual(total_len, approx_perimeter, delta=approx_perimeter * 0.005)

        # 小さすぎる半径での拒絶
        with self.assertRaises(ValueError):
            doc.add_ellipse_piece((0, 0), (0.0005, 0.1))

    def test_ellipse_tool_event_flow(self):
        ctrl = self.ctrl
        ctrl.set_active_tool("ELLIPSE")

        # 対角ドラッグ: (100, 100) -> (300, 200) [スクリーン座標]
        # ワールド座標: (0.1, 0.1) -> (0.3, 0.2)
        # 中心: (0.2, 0.15), 半径: (0.1, 0.05)
        ctrl.handle_event(EditorEvent(screen_pos=(100.0, 100.0), is_press=True))
        ctrl.handle_event(EditorEvent(screen_pos=(300.0, 200.0), is_move=True))
        self.assertEqual(len(ctrl.preview_points), 25)

        ctrl.handle_event(EditorEvent(screen_pos=(300.0, 200.0), is_release=True))
        self.assertEqual(len(ctrl.doc.pieces), 1)
        self.assertTrue(ctrl.history.can_undo)

        # Undo
        ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        self.assertEqual(len(ctrl.doc.pieces), 0)

    def test_node_kind_cycling_c_key(self):
        ctrl = self.ctrl
        piece = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2), name="Square")
        path = piece.paths[piece.outer_path_id]
        target_node = path.nodes[0]
        self.assertEqual(target_node.kind, NodeKind.CORNER)

        # ノードを選択
        ctrl.selection.select_node(piece.id, target_node.id)

        # 1回目の C キー: CORNER -> SMOOTH
        ctrl.handle_event(EditorEvent(key="C"))
        self.assertEqual(target_node.kind, NodeKind.SMOOTH)
        self.assertIsNotNone(target_node.in_handle)
        self.assertIsNotNone(target_node.out_handle)

        # 2回目の C キー: SMOOTH -> AUTO
        ctrl.handle_event(EditorEvent(key="C"))
        self.assertEqual(target_node.kind, NodeKind.AUTO)
        self.assertIsNone(target_node.in_handle)
        self.assertIsNone(target_node.out_handle)

        # 3回目の C キー: AUTO -> CORNER
        ctrl.handle_event(EditorEvent(key="C"))
        self.assertEqual(target_node.kind, NodeKind.CORNER)

        # Undo で AUTO に戻る
        ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        node_after_undo = ctrl.doc.pieces[piece.id].paths[piece.outer_path_id].nodes[0]
        self.assertEqual(node_after_undo.kind, NodeKind.AUTO)

    def test_drag_handle(self):
        ctrl = self.ctrl
        piece = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2), name="Box")
        path = piece.paths[piece.outer_path_id]
        node = path.nodes[0]  # pos: (0.0, 0.0)
        node.kind = NodeKind.SMOOTH
        node.in_handle = (-0.02, 0.0)
        node.out_handle = (0.02, 0.0)

        ctrl.set_active_tool("SELECT")
        # ノードを選択状態にしてハンドルを表示・ヒット可能にする
        ctrl.selection.select_node(piece.id, node.id)

        # out_handle 先端: (0.02, 0.0) -> screen (20, 0)
        # クリックしてドラッグ開始
        ctrl.handle_event(EditorEvent(screen_pos=(20.0, 0.0), is_press=True))

        # (0.02, 0.03) -> screen (20, 30) へドラッグ
        ctrl.handle_event(EditorEvent(screen_pos=(20.0, 30.0), is_move=True))
        ctrl.handle_event(EditorEvent(screen_pos=(20.0, 30.0), is_release=True))

        # out_handle が (0.02, 0.03) に更新され、SMOOTH なので in_handle は (-0.02, -0.03) に追従
        self.assertAlmostEqual(node.out_handle[0], 0.02, places=4)
        self.assertAlmostEqual(node.out_handle[1], 0.03, places=4)
        self.assertAlmostEqual(node.in_handle[0], -0.02, places=4)
        self.assertAlmostEqual(node.in_handle[1], -0.03, places=4)

        # Undo で元の (0.02, 0.0) に戻ること
        ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        restored_node = ctrl.doc.pieces[piece.id].paths[piece.outer_path_id].nodes[0]
        self.assertAlmostEqual(restored_node.out_handle[1], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
