# -*- coding: utf-8 -*-
"""型紙エディタ Document データモデルの単体テスト."""

import unittest
import math

from taremin_cloth.pattern.document import (
    DOCUMENT_VERSION,
    Anchor,
    AnchorKind,
    Document,
    EdgeRange,
    Node,
    NodeKind,
    Notch,
    NotchKind,
    Path,
    PathRole,
    Piece,
    Placement,
    Seam,
    SeamMode,
    Waypoint,
)
from taremin_cloth.pattern.geometry import Affine2D


class TestPatternDocument(unittest.TestCase):
    """Document モデルとその操作、シリアライズのテスト."""

    def test_empty_document(self):
        doc = Document.new(name="My Garment")
        self.assertEqual(doc.version, DOCUMENT_VERSION)
        self.assertEqual(doc.name, "My Garment")
        self.assertEqual(len(doc.pieces), 0)
        self.assertEqual(len(doc.seams), 0)
        self.assertEqual(len(doc.notches), 0)
        self.assertEqual(len(doc.anchors), 0)
        self.assertEqual(len(doc.placements), 0)

    def test_add_rectangle_piece(self):
        doc = Document.new()
        p1 = doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.3), name="Front")
        self.assertEqual(p1.name, "Front")
        self.assertIn(p1.id, doc.pieces)
        self.assertIn(p1.id, doc.placements)
        self.assertEqual(len(p1.paths), 1)

        path = p1.paths[p1.outer_path_id]
        self.assertTrue(path.closed)
        self.assertEqual(len(path.nodes), 4)
        # 反時計回り: (0,0) -> (0.2, 0) -> (0.2, 0.3) -> (0, 0.3)
        self.assertEqual(path.nodes[0].position, (0.0, 0.0))
        self.assertEqual(path.nodes[1].position, (0.2, 0.0))
        self.assertEqual(path.nodes[2].position, (0.2, 0.3))
        self.assertEqual(path.nodes[3].position, (0.0, 0.3))

        # 名前の重複回避
        p2 = doc.add_rectangle_piece((0.5, 0.0), (0.7, 0.3), name="Front")
        self.assertEqual(p2.name, "Front_1")

        # サイズが小さすぎる場合は例外
        with self.assertRaises(ValueError):
            doc.add_rectangle_piece((0.0, 0.0), (0.0005, 0.1))

    def test_add_polygon_piece(self):
        doc = Document.new()
        # 時計回りの三角形を指定 -> 自動で反時計回りに正規化されること
        pts = [(0.0, 0.0), (0.0, 0.5), (0.5, 0.0)]
        poly = doc.add_polygon_piece(pts, name="Triangle")
        self.assertEqual(poly.name, "Triangle")

        path = poly.paths[poly.outer_path_id]
        self.assertEqual(len(path.nodes), 3)

        # 頂点数が不足している場合
        with self.assertRaises(ValueError):
            doc.add_polygon_piece([(0.0, 0.0), (1.0, 1.0)])

        # 自己交差（蝶ネクタイ型）がある場合
        bow_tie = [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]
        with self.assertRaises(ValueError):
            doc.add_polygon_piece(bow_tie)

    def test_evaluate_path_segments_and_lengths(self):
        doc = Document.new()
        piece = doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.4))
        segs = doc.evaluate_path_segments(piece.id, piece.outer_path_id)
        self.assertEqual(len(segs), 4)

        # 4つの辺の長さ: 底辺0.2, 右辺0.4, 上辺0.2, 左辺0.4
        lengths = [tbl.total_length for _, _, tbl in segs]
        self.assertAlmostEqual(lengths[0], 0.2, places=4)
        self.assertAlmostEqual(lengths[1], 0.4, places=4)
        self.assertAlmostEqual(lengths[2], 0.2, places=4)
        self.assertAlmostEqual(lengths[3], 0.4, places=4)

        # 部分区間（EdgeRange）の弧長計算
        start_id, end_id, _ = segs[0]
        er = EdgeRange(
            piece_id=piece.id,
            path_id=piece.outer_path_id,
            start_node_id=start_id,
            end_node_id=end_id,
            t0=0.25,
            t1=0.75,
        )
        sub_len = doc.compute_edge_range_length(er)
        self.assertAlmostEqual(sub_len, 0.1, places=4)

    def test_auto_handles_segment_evaluation(self):
        doc = Document.new()
        piece = doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2))
        path = piece.paths[piece.outer_path_id]
        # ノードをすべて AUTO に変更
        for nd in path.nodes:
            nd.kind = NodeKind.AUTO

        segs = doc.evaluate_path_segments(piece.id, piece.outer_path_id)
        self.assertEqual(len(segs), 4)
        # AUTO ハンドルが計算されて曲線化されるため、直線距離(0.2)より長い
        for _, _, tbl in segs:
            self.assertGreater(tbl.total_length, 0.2)

    def test_compute_seam_lengths(self):
        doc = Document.new()
        p_a = doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.4), name="A")
        p_b = doc.add_rectangle_piece((0.5, 0.0), (0.7, 0.45), name="B")

        segs_a = doc.evaluate_path_segments(p_a.id, p_a.outer_path_id)
        segs_b = doc.evaluate_path_segments(p_b.id, p_b.outer_path_id)

        # A の右辺 (0.4m) と B の左辺 (0.45m) を縫合
        er_a = EdgeRange(
            piece_id=p_a.id,
            path_id=p_a.outer_path_id,
            start_node_id=segs_a[1][0],
            end_node_id=segs_a[1][1],
        )
        er_b = EdgeRange(
            piece_id=p_b.id,
            path_id=p_b.outer_path_id,
            start_node_id=segs_b[3][0],
            end_node_id=segs_b[3][1],
        )

        seam_id = "seam_test"
        doc.seams[seam_id] = Seam(
            id=seam_id,
            a=er_a,
            b=er_b,
        )

        len_a, len_b, mismatch = doc.compute_seam_lengths(seam_id)
        self.assertAlmostEqual(len_a, 0.4, places=4)
        self.assertAlmostEqual(len_b, 0.45, places=4)
        # mismatch = |0.4 - 0.45| / 0.45 = 0.05 / 0.45 = 0.1111...
        self.assertAlmostEqual(mismatch, 0.05 / 0.45, places=4)

    def test_json_roundtrip(self):
        doc = Document.new(name="Complete Doc")
        p = doc.add_rectangle_piece((0.0, 0.0), (0.3, 0.4), name="Front")
        p.transform = Affine2D(tx=0.1, ty=0.2, rotation_rad=0.5, sx=1.0, sy=1.0)
        p.target_edge_length = 0.02
        p.layer_id = 1

        segs = doc.evaluate_path_segments(p.id, p.outer_path_id)

        # Seam
        er1 = EdgeRange(piece_id=p.id, path_id=p.outer_path_id, start_node_id=segs[0][0], end_node_id=segs[0][1])
        er2 = EdgeRange(piece_id=p.id, path_id=p.outer_path_id, start_node_id=segs[2][0], end_node_id=segs[2][1])
        seam = Seam(
            id="s1",
            a=er1,
            b=er2,
            reverse=True,
            mode=SeamMode.FREE,
            shrink=0.95,
            waypoints=[Waypoint(id="wp1", position_3d=(0.1, 0.2, 0.3))],
        )
        doc.seams[seam.id] = seam

        # Notch
        notch = Notch(
            id="n1",
            piece_id=p.id,
            path_id=p.outer_path_id,
            segment_start_id=segs[0][0],
            t=0.5,
            kind=NotchKind.V,
            depth=0.006,
            width=0.003,
        )
        doc.notches[notch.id] = notch

        # Anchor
        anchor = Anchor(
            id="a1",
            piece_id=p.id,
            position=(0.15, 0.2),
            kind=AnchorKind.PIN,
            target_object="Armature",
            target_bone="Chest",
            weight=0.8,
        )
        doc.anchors[anchor.id] = anchor

        # Placement
        placement = Placement(
            piece_id=p.id,
            offset_3d=(0.0, 0.1, 1.2),
            euler_3d=(0.0, 1.57, 0.0),
            preset="BACK",
        )
        doc.placements[p.id] = placement

        # JSON シリアライズ & デシリアライズ
        json_str = doc.to_json(indent=2)
        doc2 = Document.from_json(json_str)

        self.assertEqual(doc2.version, doc.version)
        self.assertEqual(doc2.name, doc.name)
        self.assertEqual(len(doc2.pieces), 1)
        p2 = doc2.pieces[p.id]
        self.assertEqual(p2.name, p.name)
        self.assertEqual(p2.transform.tx, 0.1)
        self.assertEqual(p2.transform.rotation_rad, 0.5)
        self.assertEqual(p2.target_edge_length, 0.02)
        self.assertEqual(p2.layer_id, 1)

        self.assertEqual(len(doc2.seams), 1)
        s2 = doc2.seams["s1"]
        self.assertEqual(s2.reverse, True)
        self.assertEqual(s2.mode, SeamMode.FREE)
        self.assertEqual(s2.shrink, 0.95)
        self.assertEqual(len(s2.waypoints), 1)
        self.assertEqual(s2.waypoints[0].position_3d, (0.1, 0.2, 0.3))

        self.assertEqual(len(doc2.notches), 1)
        n2 = doc2.notches["n1"]
        self.assertEqual(n2.kind, NotchKind.V)
        self.assertEqual(n2.depth, 0.006)

        self.assertEqual(len(doc2.anchors), 1)
        a2 = doc2.anchors["a1"]
        self.assertEqual(a2.target_bone, "Chest")
        self.assertEqual(a2.weight, 0.8)

        self.assertEqual(len(doc2.placements), 1)
        pl2 = doc2.placements[p.id]
        self.assertEqual(pl2.preset, "BACK")
        self.assertEqual(pl2.offset_3d, (0.0, 0.1, 1.2))

    def test_invalid_json(self):
        # バージョン不一致
        bad_version_json = '{"version": 999, "pieces": {}}'
        with self.assertRaises(ValueError):
            Document.from_json(bad_version_json)

        # NaN 値の拒絶
        nan_json = '{"version": 1, "name": "NaN Test", "pieces": {}, "bad": NaN}'
        with self.assertRaises(ValueError):
            Document.from_json(nan_json)

        # 不正構文
        with self.assertRaises(ValueError):
            Document.from_json("{broken json")


if __name__ == "__main__":
    unittest.main()
