# -*- coding: utf-8 -*-
"""Phase 3 Step 3.1: 裁断・ダーツ・対称展開・対称複製の単体テスト."""

import unittest

from taremin_cloth.pattern.document import Document
from taremin_cloth.pattern.editor import EditorController, EditorEvent
from taremin_cloth.pattern.tools.cut import CutTool
from taremin_cloth.pattern.tools.dart import DartTool
from taremin_cloth.pattern.tools.select import SelectTool


class TestPatternCutDartUnfold(unittest.TestCase):
    """裁断、ダーツ、対称展開・複製のテストケース."""

    def setUp(self) -> None:
        self.doc = Document.new()
        # 0.2m x 0.2m の矩形ピースを作成 (中心原点: -0.1 ~ 0.1)
        piece = self.doc.add_rectangle_piece(
            (-0.1, -0.1),
            (0.1, 0.1),
            name="Body",
        )
        self.piece_id = piece.id
        self.controller = EditorController(doc=self.doc)

    def test_cut_tool_splits_piece_and_creates_seam(self) -> None:
        """CutToolで矩形を横切ると2ピースに分割され自動縫合が生成される."""
        self.controller.set_active_tool("CUT")
        tool = self.controller.tools["CUT"]
        self.assertIsInstance(tool, CutTool)

        # ピースを左右に横切るカットライン (-0.2, 0.0) -> (0.2, 0.0)
        ev_down = EditorEvent(
            screen_pos=(0, 0),
            world_pos=(-0.2, 0.0),
            is_press=True,
            button="LEFT",
        )
        self.controller.handle_event(ev_down)

        ev_up = EditorEvent(
            screen_pos=(0, 0),
            world_pos=(0.2, 0.0),
            is_release=True,
            button="LEFT",
        )
        handled = self.controller.handle_event(ev_up)
        self.assertTrue(handled)

        # ピースが2つに分割されているはず
        self.assertEqual(len(self.doc.pieces), 2)
        piece_names = [p.name for p in self.doc.pieces.values()]
        self.assertIn("Body_A", piece_names)
        self.assertIn("Body_B", piece_names)

        # 分割線上に自動縫合が1つ生成されているはず
        self.assertEqual(len(self.doc.seams), 1)
        seam = next(iter(self.doc.seams.values()))
        self.assertNotEqual(seam.a.piece_id, seam.b.piece_id)

        # Undoで1つのピースに戻る
        self.assertTrue(self.controller.history.can_undo)
        self.controller.undo()
        self.assertEqual(len(self.controller.doc.pieces), 1)
        self.assertIn("Body", [p.name for p in self.controller.doc.pieces.values()])
        self.assertEqual(len(self.controller.doc.seams), 0)

    def test_dart_tool_creates_slit_and_seam(self) -> None:
        """DartToolで外周辺と内部Apexをクリックするとダーツと自動縫合が生成される."""
        self.controller.set_active_tool("DART")
        tool = self.controller.tools["DART"]
        self.assertIsInstance(tool, DartTool)

        # 1点目: 上辺 (-0.1, 0.1) ~ (0.1, 0.1) の中央 (0.0, 0.1) をクリック
        ev1 = EditorEvent(
            screen_pos=(0, 0),
            world_pos=(0.0, 0.1),
            is_press=True,
            button="LEFT",
        )
        self.assertTrue(self.controller.handle_event(ev1))
        self.assertEqual(tool.step, 1)

        # 2点目: 内部先端 (Apex) (0.0, 0.04) をクリック
        ev2 = EditorEvent(
            screen_pos=(0, 0),
            world_pos=(0.0, 0.04),
            is_press=True,
            button="LEFT",
        )
        self.assertTrue(self.controller.handle_event(ev2))
        self.assertEqual(tool.step, 0)

        # 外周パスのノード数が 4 から 7 (left, apex, right 追加) に増えているはず
        piece = self.doc.pieces[self.piece_id]
        outer = piece.paths[piece.outer_path_id]
        self.assertEqual(len(outer.nodes), 7)

        # ダーツ自動縫合が生成されているはず
        self.assertEqual(len(self.doc.seams), 1)
        seam = next(iter(self.doc.seams.values()))
        self.assertEqual(seam.a.piece_id, self.piece_id)
        self.assertEqual(seam.b.piece_id, self.piece_id)
        self.assertTrue(seam.reverse)

    def test_select_tool_symmetric_duplicate_with_shift_u(self) -> None:
        """SelectToolでShift+Uを押すと対称ピースが複製され、中心辺が自動縫合される."""
        self.controller.set_active_tool("SELECT")
        self.controller.selection.select_piece(self.piece_id)

        # Shift+U キーイベント
        ev = EditorEvent(
            is_press=True,
            key="U",
            shift=True,
        )
        self.assertTrue(self.controller.handle_event(ev))

        # 2つのピースが存在するはず
        self.assertEqual(len(self.doc.pieces), 2)
        piece_names = [p.name for p in self.doc.pieces.values()]
        self.assertIn("Body", piece_names)
        self.assertIn("Body_Sym", piece_names)

        # 中心辺同士を結ぶ自動縫合が1件生成されているはず
        self.assertEqual(len(self.doc.seams), 1)
        seam = next(iter(self.doc.seams.values()))
        self.assertNotEqual(seam.a.piece_id, seam.b.piece_id)

    def test_select_tool_unfold_with_u(self) -> None:
        """SelectToolでUを押すと左右対称に展開されて1つのピースになる."""
        self.controller.set_active_tool("SELECT")
        self.controller.selection.select_piece(self.piece_id)

        # U キーイベント (Shiftなし)
        ev = EditorEvent(
            is_press=True,
            key="U",
            shift=False,
        )
        self.assertTrue(self.controller.handle_event(ev))

        # ピース数は1つのまま
        self.assertEqual(len(self.doc.pieces), 1)
        piece = next(iter(self.doc.pieces.values()))
        outer = piece.paths[piece.outer_path_id]
        # 元の4ノードから展開されてノード数が増加しているはず
        self.assertGreater(len(outer.nodes), 4)

    def test_cut_curved_piece_and_remap_references(self) -> None:
        """ベジェ曲線の楕円ピースをカットした際、ベジェハンドルが保持され既存縫合・ノッチが新ピースへ引き継がれる."""
        doc = Document.new()
        ellipse = doc.add_ellipse_piece((0.0, 0.0), (0.2, 0.1), name="Sleeve")

        # ノッチを追加
        from taremin_cloth.pattern.document import Notch, NotchKind
        from taremin_cloth.pattern.ids import new_id
        notch_id = new_id()
        outer = ellipse.paths[ellipse.outer_path_id]
        doc.notches[notch_id] = Notch(
            id=notch_id,
            piece_id=ellipse.id,
            path_id=outer.id,
            segment_start_id=outer.nodes[0].id,
            t=0.2,
            kind=NotchKind.V,
        )

        controller = EditorController(doc=doc)
        controller.set_active_tool("CUT")

        # Y軸に沿って中央寄りで縦にカット (X=-0.05)
        ev_down = EditorEvent(world_pos=(-0.05, -0.3), is_press=True, button="LEFT")
        controller.handle_event(ev_down)
        ev_up = EditorEvent(world_pos=(-0.05, 0.3), is_release=True, button="LEFT")
        self.assertTrue(controller.handle_event(ev_up))

        # 2ピースに分割されていること
        self.assertEqual(len(doc.pieces), 2)
        p_a = next(p for p in doc.pieces.values() if p.name.endswith("_A"))
        p_b = next(p for p in doc.pieces.values() if p.name.endswith("_B"))

        # 新規ピースのノードにベジェハンドルが残っていること（CORNER強制変換されていないこと）
        nodes_a = p_a.paths[p_a.outer_path_id].nodes
        has_handles_a = any(n.in_handle is not None or n.out_handle is not None for n in nodes_a)
        self.assertTrue(has_handles_a, "分割後ピースAにベジェハンドルが保持されていること")

        # 既存ノッチが削除されず、新ピースへ引き継がれていること
        self.assertEqual(len(doc.notches), 1)
        remapped_notch = doc.notches[notch_id]
        self.assertIn(remapped_notch.piece_id, (p_a.id, p_b.id))
        self.assertNotEqual(remapped_notch.piece_id, ellipse.id)


if __name__ == "__main__":
    unittest.main()
