# -*- coding: utf-8 -*-
"""Phase 3 Step 3.2: 部分縫合（t0, t1）および鎖縫合（chain_a, chain_b）の単体テスト."""

import unittest

from taremin_cloth.pattern.build_types import build_garment_mesh_geometry
from taremin_cloth.pattern.document import Document, EdgeRange, Seam, SeamMode
from taremin_cloth.pattern.editor import EditorController, EditorEvent
from taremin_cloth.pattern.ids import new_id
from taremin_cloth.pattern.tools.sew import SewTool


class TestPatternSeamChains(unittest.TestCase):
    """部分縫合と鎖縫合のテストケース."""

    def setUp(self) -> None:
        self.doc = Document.new()
        # ピースA: 0.1m x 0.1m
        p_a = self.doc.add_rectangle_piece((0.0, 0.0), (0.1, 0.1), name="PieceA")
        # ピースB: 0.1m x 0.1m (隣接)
        p_b = self.doc.add_rectangle_piece((0.15, 0.0), (0.25, 0.1), name="PieceB")

        self.piece_a = p_a
        self.piece_b = p_b
        self.controller = EditorController(doc=self.doc)

    def test_partial_seam_build(self) -> None:
        """EdgeRangeのt0, t1による部分縫合がメッシュ上に正しく構築される."""
        path_a = self.piece_a.paths[self.piece_a.outer_path_id]
        path_b = self.piece_b.paths[self.piece_b.outer_path_id]

        # ピースAの右辺 (node 1 -> node 2)
        node_a0 = path_a.nodes[1].id
        node_a1 = path_a.nodes[2].id

        # ピースBの左辺 (node 3 -> node 0)
        node_b0 = path_b.nodes[3].id
        node_b1 = path_b.nodes[0].id

        # A側は全体 (0.0 ~ 1.0)、B側は中央部分 (0.3 ~ 0.7)
        seam_id = new_id()
        seam = Seam(
            id=seam_id,
            a=EdgeRange(piece_id=self.piece_a.id, path_id=path_a.id, start_node_id=node_a0, end_node_id=node_a1, t0=0.0, t1=1.0),
            b=EdgeRange(piece_id=self.piece_b.id, path_id=path_b.id, start_node_id=node_b0, end_node_id=node_b1, t0=0.3, t1=0.7),
        )
        self.doc.seams[seam_id] = seam

        geom = build_garment_mesh_geometry(self.doc)
        self.assertGreater(len(geom.vertices_3d), 0)
        self.assertGreater(len(geom.sewing_edges), 0)

        # 縫合エッジが存在することを確認
        edge_pairs = geom.sewing_edges
        for u, v in edge_pairs:
            self.assertNotEqual(u, v)

    def test_chain_seam_build(self) -> None:
        """chain_bに複数EdgeRangeを持つ鎖縫合が正しく構築される."""
        path_a = self.piece_a.paths[self.piece_a.outer_path_id]
        path_b = self.piece_b.paths[self.piece_b.outer_path_id]

        # ピースAの右辺
        node_a0 = path_a.nodes[1].id
        node_a1 = path_a.nodes[2].id

        # ピースBの左辺と上辺をチェーン
        range_b1 = EdgeRange(piece_id=self.piece_b.id, path_id=path_b.id, start_node_id=path_b.nodes[3].id, end_node_id=path_b.nodes[0].id)
        range_b2 = EdgeRange(piece_id=self.piece_b.id, path_id=path_b.id, start_node_id=path_b.nodes[0].id, end_node_id=path_b.nodes[1].id)

        seam_id = new_id()
        seam = Seam(
            id=seam_id,
            a=EdgeRange(piece_id=self.piece_a.id, path_id=path_a.id, start_node_id=node_a0, end_node_id=node_a1),
            b=range_b1,
            chain_b=[range_b1, range_b2],
            mode=SeamMode.CHAIN,
        )
        self.doc.seams[seam_id] = seam

        geom = build_garment_mesh_geometry(self.doc)
        self.assertGreater(len(geom.vertices_3d), 0)
        self.assertGreater(len(geom.sewing_edges), 0)

    def test_sew_tool_shift_chain(self) -> None:
        """SewToolでShift+クリックとEnterキーによる鎖縫合の定義が行える."""
        self.controller.set_active_tool("SEW")
        tool = self.controller.tools["SEW"]
        self.assertIsInstance(tool, SewTool)

        # ピースAの下辺 (0.05, 0.0) を通常クリック
        ev1 = EditorEvent(
            screen_pos=(0, 0),
            world_pos=(0.05, 0.0),
            is_press=True,
            button="LEFT",
            shift=False,
        )
        self.assertTrue(self.controller.handle_event(ev1))
        self.assertEqual(tool.step, 1)  # B側へ

        # ピースBの下辺 (0.20, 0.0) をShift+クリック
        ev2 = EditorEvent(
            screen_pos=(0, 0),
            world_pos=(0.20, 0.0),
            is_press=True,
            button="LEFT",
            shift=True,
        )
        self.assertTrue(self.controller.handle_event(ev2))
        self.assertEqual(len(tool.chain_b), 1)

        # ピースBの右辺 (0.25, 0.05) をShift+クリック
        ev3 = EditorEvent(
            screen_pos=(0, 0),
            world_pos=(0.25, 0.05),
            is_press=True,
            button="LEFT",
            shift=True,
        )
        self.assertTrue(self.controller.handle_event(ev3))
        self.assertEqual(len(tool.chain_b), 2)

        # Enterキーで確定
        ev_enter = EditorEvent(
            is_press=True,
            key="RET",
        )
        self.assertTrue(self.controller.handle_event(ev_enter))

        # 縫合が生成されているはず
        self.assertEqual(len(self.doc.seams), 1)
        seam = next(iter(self.doc.seams.values()))
        self.assertEqual(seam.mode, SeamMode.CHAIN)
        self.assertEqual(len(seam.chain_b), 2)


if __name__ == "__main__":
    unittest.main()

