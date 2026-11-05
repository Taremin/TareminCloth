# -*- coding: utf-8 -*-
"""型紙エディタの要素ヒットテスト・選択状態・スナップ計算の単体テスト."""

import unittest

from taremin_cloth.pattern.document import Document, NodeKind
from taremin_cloth.pattern.selection import HitTargetKind, hit_test
from taremin_cloth.pattern.session import (
    SelectionState,
    SnapKind,
    SnapSettings,
    ViewTransform2D,
    compute_snap,
)


class TestPatternSelectionAndSession(unittest.TestCase):
    """ヒットテスト、ビューポート変換、選択、スナップのテスト."""

    def setUp(self):
        self.doc = Document.new()
        # 0.2m x 0.2m の矩形ピースを追加 (0,0) -> (0.2, 0.2)
        self.piece = self.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2), name="TestPiece")

    def test_hit_test_piece_body(self):
        # 矩形の中央 (0.1, 0.1) をクリック -> PIECE_BODY がヒット
        res = hit_test(self.doc, (0.1, 0.1))
        self.assertIsNotNone(res)
        self.assertEqual(res.kind, HitTargetKind.PIECE_BODY)
        self.assertEqual(res.piece_id, self.piece.id)

        # 矩形の外側 (0.5, 0.5) をクリック -> ヒットなし
        res_outside = hit_test(self.doc, (0.5, 0.5))
        self.assertIsNone(res_outside)

    def test_hit_test_node_over_body(self):
        # ノード (0.0, 0.0) の至近距離 (0.002, 0.002) をクリック
        # 矩形内部でもあるが、NODE の優先度が高いため NODE がヒットすること
        res = hit_test(self.doc, (0.002, 0.002))
        self.assertIsNotNone(res)
        self.assertEqual(res.kind, HitTargetKind.NODE)
        self.assertEqual(res.piece_id, self.piece.id)

    def test_hit_test_edge_over_body(self):
        # 底辺 (0.1, 0.0) のわずかに内側 (0.1, 0.002) をクリック
        # 矩形内部でもあるが、EDGE の優先度が高いため EDGE がヒットすること
        res = hit_test(self.doc, (0.1, 0.002), edge_threshold=0.005)
        self.assertIsNotNone(res)
        self.assertEqual(res.kind, HitTargetKind.EDGE)
        self.assertEqual(res.piece_id, self.piece.id)
        self.assertAlmostEqual(res.edge_t, 0.5, delta=0.1)

    def test_hit_test_handle_over_node(self):
        # ノードにハンドルを設定
        path = self.piece.paths[self.piece.outer_path_id]
        node0 = path.nodes[0]
        node0.kind = NodeKind.SMOOTH
        node0.out_handle = (0.02, 0.0)

        # ハンドル先端 (0.02, 0.0) 近傍をクリック
        # 選択中ノードとして指定
        res = hit_test(self.doc, (0.021, 0.0), selected_node_ids={node0.id})
        self.assertIsNotNone(res)
        self.assertEqual(res.kind, HitTargetKind.HANDLE)
        self.assertEqual(res.handle_type, "out")
        self.assertEqual(res.node_id, node0.id)

    def test_view_transform(self):
        view = ViewTransform2D(pan_x=500.0, pan_y=400.0, zoom=1000.0)
        world_pt = (0.15, -0.25)
        screen_pt = view.world_to_screen(world_pt)
        self.assertEqual(screen_pt, (500.0 + 150.0, 400.0 - 250.0))

        # 往復変換
        restored_world = view.screen_to_world(screen_pt)
        self.assertAlmostEqual(restored_world[0], world_pt[0], places=5)
        self.assertAlmostEqual(restored_world[1], world_pt[1], places=5)

        # ズーム動作
        view.zoom_at((650.0, 150.0), 2.0)
        self.assertEqual(view.zoom, 2000.0)
        # ズーム中心のワールド座標が変わっていないこと
        center_world = view.screen_to_world((650.0, 150.0))
        self.assertAlmostEqual(center_world[0], world_pt[0], places=5)
        self.assertAlmostEqual(center_world[1], world_pt[1], places=5)

    def test_selection_state(self):
        sel = SelectionState()
        self.assertFalse(sel.is_piece_selected("p1"))

        sel.select_piece("p1")
        self.assertTrue(sel.is_piece_selected("p1"))
        self.assertEqual(sel.active_piece_id, "p1")

        sel.select_node("p1", "n1", add=True)
        self.assertTrue(sel.is_node_selected("n1"))
        self.assertTrue(sel.is_piece_selected("p1"))

        sel.clear()
        self.assertFalse(sel.is_piece_selected("p1"))
        self.assertFalse(sel.is_node_selected("n1"))
        self.assertIsNone(sel.active_piece_id)

    def test_compute_snap(self):
        view = ViewTransform2D(pan_x=0.0, pan_y=0.0, zoom=1000.0)
        settings = SnapSettings(
            enabled=True,
            snap_grid=True,
            snap_nodes=True,
            grid_size=0.01,  # 1cm
            snap_threshold_pixels=10.0,  # 10px = 0.01m
        )

        # ノード (0.0, 0.0) の近傍 (0.003, 0.002) -> ノード吸着
        snap1 = compute_snap((0.003, 0.002), self.doc, view=view, settings=settings)
        self.assertEqual(snap1.kind, SnapKind.NODE)
        self.assertEqual(snap1.world_pos, (0.0, 0.0))

        # ノードから遠いがグリッド (0.1, 0.1) 近傍 (0.098, 0.102) -> グリッド吸着
        snap2 = compute_snap((0.098, 0.102), self.doc, view=view, settings=settings)
        self.assertEqual(snap2.kind, SnapKind.GRID)
        self.assertAlmostEqual(snap2.world_pos[0], 0.1, places=4)
        self.assertAlmostEqual(snap2.world_pos[1], 0.1, places=4)

        # 閾値より遠い場合 -> 吸着なし
        settings_strict = SnapSettings(enabled=True, snap_threshold_pixels=1.0)  # 1px = 0.001m
        snap3 = compute_snap((0.055, 0.055), self.doc, view=view, settings=settings_strict)
        self.assertEqual(snap3.kind, SnapKind.NONE)


if __name__ == "__main__":
    unittest.main()
