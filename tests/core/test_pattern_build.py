# -*- coding: utf-8 -*-
"""型紙エディタ 衣服メッシュ幾何ビルド (build_types.py) の単体テスト (Blender非依存)."""

import unittest
import numpy as np

from taremin_cloth.pattern.build_types import (
    build_garment_mesh_geometry,
)
from taremin_cloth.pattern.document import (
    Anchor,
    AnchorKind,
    Document,
    EdgeRange,
    Placement,
    Seam,
    Waypoint,
)
from taremin_cloth.utils.validation import validate_sewing_topology


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

    def test_shared_endpoints_waypoint_seams_valid_topology(self):
        """共有端点を持つ中継点付き縫合で直接縫合にフォールバックし、validate_sewing_topology がエラー0件となること (G1検証)."""
        doc = Document.new()
        p1 = doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2), name="Front")
        p2 = doc.add_rectangle_piece((0.3, 0.0), (0.5, 0.2), name="Back")
        p1.target_edge_length = 0.05
        p2.target_edge_length = 0.05

        outer1 = p1.paths[p1.outer_path_id]
        outer2 = p2.paths[p2.outer_path_id]

        # 縫合1: 直接縫合 (右辺と左辺)
        s1 = Seam(
            id="s1",
            a=EdgeRange(piece_id=p1.id, path_id=outer1.id, start_node_id=outer1.nodes[1].id, end_node_id=outer1.nodes[2].id),
            b=EdgeRange(piece_id=p2.id, path_id=outer2.id, start_node_id=outer2.nodes[3].id, end_node_id=outer2.nodes[0].id),
        )
        # 縫合2: 中継点付き縫合 (上辺同士。ノード2を共有している)
        s2 = Seam(
            id="s2",
            a=EdgeRange(piece_id=p1.id, path_id=outer1.id, start_node_id=outer1.nodes[2].id, end_node_id=outer1.nodes[3].id),
            b=EdgeRange(piece_id=p2.id, path_id=outer2.id, start_node_id=outer2.nodes[2].id, end_node_id=outer2.nodes[1].id),
            waypoints=[Waypoint(id="wp1", position_3d=(0.25, 0.25, 0.1))],
        )
        doc.seams["s1"] = s1
        doc.seams["s2"] = s2

        geom = build_garment_mesh_geometry(doc)
        self.assertGreater(len(geom.sewing_edges), 0)

        # 面頂点集合の作成
        face_verts = set()
        for f in geom.faces:
            face_verts.update(f)

        # validate_sewing_topology でトポロジー検証を実行
        report = validate_sewing_topology(geom.sewing_edges, face_verts)
        self.assertFalse(report.has_issues, f"Sewing topology issues detected: {report.branching_issues}")

    def test_uv_normalized_with_margin_and_transforms(self):
        """UV座標がピースの配置変換を反映し、[0.02, 0.98] の余白内に収まること (G2 & 8.6節検証)."""
        doc = Document.new()
        p1 = doc.add_rectangle_piece((0.0, 0.0), (0.1, 0.1), name="Left")
        p2 = doc.add_rectangle_piece((0.0, 0.0), (0.1, 0.1), name="Right")
        # p2 を 2D キャンバス上で (0.3, 0.0) へ平行移動
        from taremin_cloth.pattern.geometry import Affine2D
        p2.transform = Affine2D(tx=0.3, ty=0.0)

        geom = build_garment_mesh_geometry(doc)
        self.assertGreater(len(geom.face_uvs), 0)

        uvs = np.array(geom.face_uvs)
        min_u, min_v = float(np.min(uvs[:, 0])), float(np.min(uvs[:, 1]))
        max_u, max_v = float(np.max(uvs[:, 0])), float(np.max(uvs[:, 1]))

        # 周囲 2% 余白 [0.02, 0.98] 内に収まっていること
        self.assertGreaterEqual(min_u, 0.019)
        self.assertGreaterEqual(min_v, 0.019)
        self.assertLessEqual(max_u, 0.981)
        self.assertLessEqual(max_v, 0.981)


if __name__ == "__main__":
    unittest.main()
