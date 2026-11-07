# -*- coding: utf-8 -*-
"""Phase 3 Step 3.3: 閉じた内部穴パス（HOLE）によるメッシュ三角形除去の単体テスト."""

import unittest

from taremin_cloth.pattern.build_types import build_garment_mesh_geometry
from taremin_cloth.pattern.document import Document, Node, NodeKind, Path, PathRole
from taremin_cloth.pattern.geometry import point_in_polygon
from taremin_cloth.pattern.ids import new_id


class TestPatternHoles(unittest.TestCase):
    """内部穴パス (HOLE) のテストケース."""

    def test_hole_removes_triangles(self) -> None:
        """矩形ピースの中央に穴パスを配置すると、穴内部の三角形が除去される."""
        # 1. 穴なしのドキュメントでメッシュ生成
        doc_solid = Document.new()
        p_solid = doc_solid.add_rectangle_piece((-0.1, -0.1), (0.1, 0.1), name="Solid")
        geom_solid = build_garment_mesh_geometry(doc_solid)
        solid_face_count = len(geom_solid.faces)
        self.assertGreater(solid_face_count, 0)

        # 2. 穴ありのドキュメントを作成
        doc_hole = Document.new()
        p_hole = doc_hole.add_rectangle_piece((-0.1, -0.1), (0.1, 0.1), name="WithHole")
        piece = doc_hole.pieces[p_hole.id]

        # 中央に 0.05m x 0.05m の穴 (-0.025 ~ 0.025)
        hole_path_id = new_id()
        hole_nodes = [
            Node(id=new_id(), position=(-0.025, -0.025), kind=NodeKind.CORNER),
            Node(id=new_id(), position=(0.025, -0.025), kind=NodeKind.CORNER),
            Node(id=new_id(), position=(0.025, 0.025), kind=NodeKind.CORNER),
            Node(id=new_id(), position=(-0.025, 0.025), kind=NodeKind.CORNER),
        ]
        piece.paths[hole_path_id] = Path(
            id=hole_path_id,
            role=PathRole.HOLE,
            nodes=hole_nodes,
            closed=True,
        )

        geom_hole = build_garment_mesh_geometry(doc_hole)
        hole_face_count = len(geom_hole.faces)

        def tri_area(v0: tuple[float, float, float], v1: tuple[float, float, float], v2: tuple[float, float, float]) -> float:
            return 0.5 * abs((v1[0] - v0[0]) * (v2[2] - v0[2]) - (v1[2] - v0[2]) * (v2[0] - v0[0]))

        area_solid = sum(tri_area(geom_solid.vertices_3d[t[0]], geom_solid.vertices_3d[t[1]], geom_solid.vertices_3d[t[2]]) for t in geom_solid.faces)
        area_hole = sum(tri_area(geom_hole.vertices_3d[t[0]], geom_hole.vertices_3d[t[1]], geom_hole.vertices_3d[t[2]]) for t in geom_hole.faces)

        # 穴があるため、総表面積が穴の面積 (0.05 x 0.05 = 0.0025 m^2) 相当分減少していること
        area_diff = area_solid - area_hole
        self.assertGreater(area_diff, 0.002)
        self.assertLess(area_diff, 0.005)

        # 穴の中心点 (0, 0, 0) を覆う三角形が存在しないことを確認
        hole_poly = [(-0.025, -0.025), (0.025, -0.025), (0.025, 0.025), (-0.025, 0.025)]
        for tri in geom_hole.faces:
            v0 = geom_hole.vertices_3d[tri[0]]
            v1 = geom_hole.vertices_3d[tri[1]]
            v2 = geom_hole.vertices_3d[tri[2]]
            # 2D XZ平面 (Yは0)
            c2d = ((v0[0] + v1[0] + v2[0]) / 3.0, (v0[2] + v1[2] + v2[2]) / 3.0)
            self.assertFalse(point_in_polygon(c2d, hole_poly))


if __name__ == "__main__":
    unittest.main()

