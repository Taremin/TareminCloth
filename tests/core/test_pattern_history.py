# -*- coding: utf-8 -*-
"""型紙エディタ HistoryManager (Undo/Redo) の単体テスト."""

import unittest

from taremin_cloth.pattern.document import Document
from taremin_cloth.pattern.history import HistoryManager


class TestPatternHistory(unittest.TestCase):
    """HistoryManager の動作テスト."""

    def test_empty_history(self):
        hm = HistoryManager()
        self.assertFalse(hm.can_undo)
        self.assertFalse(hm.can_redo)
        self.assertIsNone(hm.undo())
        self.assertIsNone(hm.redo())

    def test_record_change_and_undo_redo(self):
        hm = HistoryManager()
        doc0 = Document.new(name="Original")

        # ピース追加
        doc1 = Document.from_json(doc0.to_json())
        doc1.add_rectangle_piece((0.0, 0.0), (0.2, 0.2), name="Piece1")

        recorded = hm.record_change(
            description="Add Piece1",
            before_doc=doc0,
            after_doc=doc1,
        )
        self.assertTrue(recorded)
        self.assertTrue(hm.can_undo)
        self.assertFalse(hm.can_redo)

        # Undo 実行
        res = hm.undo()
        self.assertIsNotNone(res)
        restored_doc, desc = res
        self.assertEqual(desc, "Add Piece1")
        self.assertEqual(len(restored_doc.pieces), 0)
        self.assertFalse(hm.can_undo)
        self.assertTrue(hm.can_redo)

        # Redo 実行
        res2 = hm.redo()
        self.assertIsNotNone(res2)
        redo_doc, desc2 = res2
        self.assertEqual(desc2, "Add Piece1")
        self.assertEqual(len(redo_doc.pieces), 1)
        self.assertTrue(hm.can_undo)
        self.assertFalse(hm.can_redo)

    def test_identical_change_skipped(self):
        hm = HistoryManager()
        doc = Document.new()
        recorded = hm.record_change(
            description="No-op",
            before_doc=doc,
            after_doc=doc,
        )
        self.assertFalse(recorded)
        self.assertFalse(hm.can_undo)

    def test_max_entries_limit(self):
        hm = HistoryManager(max_entries=3)
        doc = Document.new()

        for i in range(5):
            before = Document.from_json(doc.to_json())
            doc.add_rectangle_piece((i * 0.1, 0.0), (i * 0.1 + 0.05, 0.05), name=f"P_{i}")
            hm.record_change(
                description=f"Add P_{i}",
                before_doc=before,
                after_doc=doc,
            )

        self.assertEqual(len(hm.undo_stack), 3)
        # 最古の P_0, P_1 は破棄され、P_2, P_3, P_4 が残る
        self.assertEqual(hm.undo_stack[0].description, "Add P_2")
        self.assertEqual(hm.undo_stack[1].description, "Add P_3")
        self.assertEqual(hm.undo_stack[2].description, "Add P_4")

    def test_branching_clears_redo(self):
        hm = HistoryManager()
        doc0 = Document.new()

        doc1 = Document.from_json(doc0.to_json())
        doc1.add_rectangle_piece((0.0, 0.0), (0.1, 0.1), name="A")
        hm.record_change(description="Add A", before_doc=doc0, after_doc=doc1)

        doc2 = Document.from_json(doc1.to_json())
        doc2.add_rectangle_piece((0.2, 0.0), (0.3, 0.1), name="B")
        hm.record_change(description="Add B", before_doc=doc1, after_doc=doc2)

        # 1回 Undo (Add B を取り消し)
        hm.undo()
        self.assertTrue(hm.can_redo)

        # 別の操作 C を記録
        doc3 = Document.from_json(doc1.to_json())
        doc3.add_rectangle_piece((0.4, 0.0), (0.5, 0.1), name="C")
        hm.record_change(description="Add C", before_doc=doc1, after_doc=doc3)

        # Redo スタックは破棄される
        self.assertFalse(hm.can_redo)
        self.assertEqual(len(hm.undo_stack), 2)
        self.assertEqual(hm.undo_stack[-1].description, "Add C")

    def test_clear(self):
        hm = HistoryManager()
        doc0 = Document.new()
        doc1 = Document.from_json(doc0.to_json())
        doc1.add_rectangle_piece((0.0, 0.0), (0.1, 0.1))
        hm.record_change(description="Add", before_doc=doc0, after_doc=doc1)
        hm.undo()

        self.assertTrue(hm.can_redo)
        hm.clear()
        self.assertFalse(hm.can_undo)
        self.assertFalse(hm.can_redo)


if __name__ == "__main__":
    unittest.main()
