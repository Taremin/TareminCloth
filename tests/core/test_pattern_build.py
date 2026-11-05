# -*- coding: utf-8 -*-
"""型紙エディタ 衣服メッシュ幾何ビルド (build_types.py) の単体テスト (Blender非依存)."""

import unittest

from taremin_cloth.pattern.build_types import (
    document_to_contours_and_seams,
    build_garment_mesh_geometry,
)
from taremin_cloth.pattern.document import (
    Anchor,
    AnchorKind,
    Document,
    EdgeRange,
    Placement,
    Seam,
)


class TestPatternBuildTypes(unittest.TestCase):
    """build_types によるメッシュ幾何・UV・Loose Edge生成テスト."""

    def test_build_single_rectangle_geometry(self):
        doc = Document.new()
        p = doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.3), name="Front")
        p.target_edge_length = 0.05

        geom = build_garment_mesh_geometry(doc)
        self.assertGreater(len(geom.vertices_3d), 4)
        self.assertGreater(len(geom.faces), 2)
        self.assertEqual(len(geom.sewing_edges), 0)

        # UV のテスト (すべての頂点UVが [0.0, 1.0] 範囲内)
        for u, v in geom.face_uvs:
            self.assertGreaterEqual(u, -1e-5)
            self.assertLessEqual(u, 1.0 + 1e-5)
            self.assertGreaterEqual(v, -1e-5)
            self.assertLessEqual(v, 1.0 + 1e-5)

        # 面属性
        self.assertEqual(len(geom.face_piece_indices), len(geom.faces))
        self.assertTrue(all(idx == 0 for idx in geom.face_piece_indices))
        self.assertEqual(len(geom.piece_vertex_map["Front"]), len(geom.vertices_3d))

    def test_build_two_rectangles_with_seam(self):
        doc = Document.new()
        p1 = doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.3), name="Front")
        p2 = doc.add_rectangle_piece((0.4, 0.0), (0.6, 0.3), name="Back")
        p1.target_edge_length = 0.05
        p2.target_edge_length = 0.05

        doc.placements[p2.id] = Placement(
            piece_id=p2.id,
            offset_3d=(0.0, 0.1, 0.0),
        )

        segs_a = doc.evaluate_path_segments(p1.id, p1.outer_path_id)
        segs_b = doc.evaluate_path_segments(p2.id, p2.outer_path_id)

        # 右辺と左辺の縫合
        seam = Seam(
            id="s1",
            a=EdgeRange(piece_id=p1.id, path_id=p1.outer_path_id, start_node_id=segs_a[1][0], end_node_id=segs_a[1][1]),
            b=EdgeRange(piece_id=p2.id, path_id=p2.outer_path_id, start_node_id=segs_b[3][0], end_node_id=segs_b[3][1]),
        )
        doc.seams[seam.id] = seam

        geom = build_garment_mesh_geometry(doc)
        self.assertGreater(len(geom.vertices_3d), 10)
        self.assertGreater(len(geom.faces), 5)
        # 縫合 Loose Edge が生成されていること
        self.assertGreater(len(geom.sewing_edges), 0)
        # 2つのピースが存在すること
        self.assertEqual(len(geom.piece_vertex_map), 2)

    def test_build_with_anchor_pin(self):
        doc = Document.new()
        p = doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2), name="Square")
        p.target_edge_length = 0.05

        # (0.1, 0.1) にピンアンカーを追加
        doc.anchors["a1"] = Anchor(
            id="a1",
            piece_id=p.id,
            position=(0.1, 0.1),
            kind=AnchorKind.PIN,
        )

        geom = build_garment_mesh_geometry(doc)
        self.assertEqual(len(geom.pin_vertex_indices), 1)
        pin_v_idx = geom.pin_vertex_indices[0]
        self.assertIn(pin_v_idx, geom.piece_vertex_map["Square"])


if __name__ == "__main__":
    unittest.main()
