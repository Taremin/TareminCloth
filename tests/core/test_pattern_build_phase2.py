# -*- coding: utf-8 -*-
"""Phase 2 Step 2.3: メッシュビルドへの内部線拘束とノッチ頂点固定の反映テスト."""

import unittest
import numpy as np

from taremin_cloth.pattern.build_types import build_garment_mesh_geometry
from taremin_cloth.pattern.document import (
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
)
from taremin_cloth.pattern.ids import new_id
from taremin_cloth.pattern.session import BoneInfo3D


class TestPatternBuildPhase2(unittest.TestCase):
    """内部線拘束、ノッチ頂点固定、ベジェ曲線メッシュビルドの検証."""

    def test_internal_line_constraints_in_mesh(self) -> None:
        """内部線がメッシュの拘束としてCDTリメッシュに反映されることを検証."""
        doc = Document.new()
        p = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.4), name="Panel")

        # ピース内部に水平な内部線を追加 (0.1, 0.2) -> (0.3, 0.2)
        int_nodes = [
            Node(id=new_id(), position=(0.1, 0.2), kind=NodeKind.CORNER),
            Node(id=new_id(), position=(0.3, 0.2), kind=NodeKind.CORNER),
        ]
        path_id = new_id()
        p.paths[path_id] = Path(id=path_id, role=PathRole.INTERNAL, nodes=int_nodes, closed=False)

        geom = build_garment_mesh_geometry(doc)
        self.assertGreater(len(geom.vertices_3d), 0)
        self.assertGreater(len(geom.faces), 0)

        # 内部線の両端点 (0.1, 0.2) と (0.3, 0.2) が頂点に含まれることを確認
        v3d = np.array(geom.vertices_3d, dtype=np.float32)
        # 初期配置は (u, 0, v)
        u_coords = v3d[:, 0]
        v_coords = v3d[:, 2]

        # (0.1, 0.2) 近傍の頂点を探索
        dists_p1 = np.hypot(u_coords - 0.1, v_coords - 0.2)
        self.assertLess(np.min(dists_p1), 1e-4)

        # (0.3, 0.2) 近傍の頂点を探索
        dists_p2 = np.hypot(u_coords - 0.3, v_coords - 0.2)
        self.assertLess(np.min(dists_p2), 1e-4)

    def test_notch_boundary_vertex_pinning(self) -> None:
        """合印ノッチの位置が境界ループの頂点として固定されることを検証."""
        doc = Document.new()
        p = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.3), name="Back")
        outer_path = p.paths[p.outer_path_id]

        # 下辺 (0.0, 0.0) -> (0.4, 0.0) の中央 t=0.5 (0.2, 0.0) にノッチを追加
        start_node_id = outer_path.nodes[0].id
        notch_id = new_id()
        doc.notches[notch_id] = Notch(
            id=notch_id,
            piece_id=p.id,
            path_id=p.outer_path_id,
            segment_start_id=start_node_id,
            t=0.5,
            kind=NotchKind.V,
        )

        geom = build_garment_mesh_geometry(doc)
        self.assertGreater(len(geom.vertices_3d), 0)

        # 境界頂点の中に (0.2, 0.0) と一致する頂点が存在することを検証
        v3d = np.array(geom.vertices_3d, dtype=np.float32)
        u_coords = v3d[:, 0]
        v_coords = v3d[:, 2]

        dists = np.hypot(u_coords - 0.2, v_coords - 0.0)
        min_dist = float(np.min(dists))
        self.assertLess(min_dist, 1e-4, f"ノッチ位置 (0.2, 0.0) に頂点が見つかりません。最短距離: {min_dist}")

    def test_ellipse_piece_mesh_build(self) -> None:
        """楕円ベジェピースから妥当な3Dメッシュがビルドできることを検証."""
        doc = Document.new()
        p = doc.add_ellipse_piece((0.0, 0.0), (0.3, 0.2), name="SleeveCap")

        geom = build_garment_mesh_geometry(doc)
        self.assertGreater(len(geom.vertices_3d), 10)
        self.assertGreater(len(geom.faces), 10)
        self.assertEqual(len(geom.face_piece_indices), len(geom.faces))
        self.assertEqual(len(geom.face_uvs), len(geom.faces) * 3)

    def test_build_bone_aligned_garment_geometry(self) -> None:
        """ボーンに追従して平面配置され、かつメッシュ自体は曲面変形せず完全な平面であることの検証."""
        doc = Document.new()
        p_id = "sleeve_piece"
        path_id = "path_outer"

        nodes = [
            Node(id="n0", position=(0.0, 0.0)),
            Node(id="n1", position=(0.2, 0.0)),
            Node(id="n2", position=(0.2, 0.3)),
            Node(id="n3", position=(0.0, 0.3)),
        ]
        piece = Piece(
            id=p_id,
            name="Sleeve",
            paths={path_id: Path(id=path_id, role=PathRole.OUTER, nodes=nodes, closed=True)},
            outer_path_id=path_id,
            target_edge_length=0.05,
        )
        doc.pieces[p_id] = piece
        doc.placements[p_id] = Placement(
            piece_id=p_id,
            target_bone="Arm_Bone",
            bone_t=0.5,
            bone_angle_deg=0.0,
            surface_offset=0.08,
        )

        bones = {
            "Arm_Bone": BoneInfo3D(
                name="Arm_Bone",
                head_3d=(0.3, 0.0, 1.4),
                tail_3d=(0.3, 0.0, 1.0),
                radius=0.05,
            )
        }

        geom = build_garment_mesh_geometry(doc, bones=bones)
        self.assertGreater(len(geom.vertices_3d), 0)
        self.assertGreater(len(geom.faces), 0)

        verts = np.array(geom.vertices_3d)

        # 1. ボーン軸からの法線方向オフセット検証:
        # 正面配置 (angle=0) なので法線は -Y 方向。頂点の Y 座標はすべて -0.08 付近であること
        mean_y = float(np.mean(verts[:, 1]))
        self.assertAlmostEqual(mean_y, -0.08, places=3)

        # 2. メッシュが完全な平面であることの検証 (Y方向の分散がほぼ0):
        var_y = float(np.var(verts[:, 1]))
        self.assertLess(var_y, 1e-6, "型紙メッシュは曲面プリベンドされず、完全な平面でなければなりません")

        # 3. ボーンの中心 (t=0.5 -> Z=1.2) に配置されていること:
        mean_z = float(np.mean(verts[:, 2]))
        self.assertAlmostEqual(mean_z, 1.2, delta=0.02)

    def test_build_tubular_waypoints_sewing(self) -> None:
        """Seam.routing=BONE_WRAP 時、ボーン裏側を迂回する3本構成のLoose Edge列が生成されることの検証."""
        from taremin_cloth.pattern.document import SeamRouting

        doc = Document.new()
        p_id = "sleeve_piece"
        path_id = "path_outer"

        nodes = [
            Node(id="n0", position=(0.0, 0.0)),
            Node(id="n1", position=(0.2, 0.0)),
            Node(id="n2", position=(0.2, 0.3)),
            Node(id="n3", position=(0.0, 0.3)),
        ]
        piece = Piece(
            id=p_id,
            name="Sleeve",
            paths={path_id: Path(id=path_id, role=PathRole.OUTER, nodes=nodes, closed=True)},
            outer_path_id=path_id,
            target_edge_length=0.05,
        )
        doc.pieces[p_id] = piece
        doc.placements[p_id] = Placement(
            piece_id=p_id,
            target_bone="Arm_Bone",
            bone_t=0.5,
            bone_angle_deg=0.0,
            surface_offset=0.08,
        )

        # 自己縫合 (左辺 n3->n0 と 右辺 n1->n2 を BONE_WRAP で縫合して筒状にする)
        seam_id = "tubular_seam"
        doc.seams[seam_id] = Seam(
            id=seam_id,
            a=EdgeRange(piece_id=p_id, path_id=path_id, start_node_id="n3", end_node_id="n0"),
            b=EdgeRange(piece_id=p_id, path_id=path_id, start_node_id="n1", end_node_id="n2"),
            routing=SeamRouting.BONE_WRAP,
        )

        bones = {
            "Arm_Bone": BoneInfo3D(
                name="Arm_Bone",
                head_3d=(0.3, 0.0, 1.4),
                tail_3d=(0.3, 0.0, 1.0),
                radius=0.05,
            )
        }

        geom = build_garment_mesh_geometry(doc, bones=bones)
        verts = np.array(geom.vertices_3d)

        # 布メッシュ本体の頂点数
        cloth_verts_count = len(geom.piece_vertex_map["Sleeve"])
        # 総頂点数（中継点頂点を含む）
        total_verts_count = len(verts)
        self.assertGreater(total_verts_count, cloth_verts_count, "BONE_WRAP ではボーン裏側に中継点頂点が追加される必要があります")

        # 縫合Loose Edgeの検証
        # 直接縫合ならステップ数 S 本だが、BONE_WRAP では va -> wp1 -> wp2 -> vb で 3*S 本になる
        self.assertGreater(len(geom.sewing_edges), 0)
        self.assertEqual(len(geom.sewing_edges) % 3, 0, "中継点付き縫合エッジは3本ペアでなければなりません")

        # 追加された中継点頂点の位置検証:
        # 正面が Y = -0.08 に対し、裏側中継点は Y > 0 (ボーンの奥側) に位置していること
        wp_indices = list(range(cloth_verts_count, total_verts_count))
        for wp_idx in wp_indices:
            wp_pos = verts[wp_idx]
            self.assertGreater(wp_pos[1], 0.0, "中継点はボーン裏側 (Y > 0) を迂回しなければなりません")

    def test_tubular_off_fallback_direct(self) -> None:
        """Seam.routing=DIRECT 時は、ボーン指定があっても中継点を生成せず直接縫合Loose Edgeが生成されることの検証."""
        from taremin_cloth.pattern.document import SeamRouting

        doc = Document.new()
        p_id = "sleeve_piece"
        path_id = "path_outer"

        nodes = [
            Node(id="n0", position=(0.0, 0.0)),
            Node(id="n1", position=(0.2, 0.0)),
            Node(id="n2", position=(0.2, 0.3)),
            Node(id="n3", position=(0.0, 0.3)),
        ]
        piece = Piece(
            id=p_id,
            name="Sleeve",
            paths={path_id: Path(id=path_id, role=PathRole.OUTER, nodes=nodes, closed=True)},
            outer_path_id=path_id,
            target_edge_length=0.05,
        )
        doc.pieces[p_id] = piece
        doc.placements[p_id] = Placement(
            piece_id=p_id,
            target_bone="Arm_Bone",
            bone_t=0.5,
            bone_angle_deg=0.0,
            surface_offset=0.08,
        )

        seam_id = "direct_seam"
        doc.seams[seam_id] = Seam(
            id=seam_id,
            a=EdgeRange(piece_id=p_id, path_id=path_id, start_node_id="n3", end_node_id="n0"),
            b=EdgeRange(piece_id=p_id, path_id=path_id, start_node_id="n1", end_node_id="n2"),
            routing=SeamRouting.DIRECT,
        )

        bones = {
            "Arm_Bone": BoneInfo3D(
                name="Arm_Bone",
                head_3d=(0.3, 0.0, 1.4),
                tail_3d=(0.3, 0.0, 1.0),
                radius=0.05,
            )
        }

        geom = build_garment_mesh_geometry(doc, bones=bones)
        cloth_verts_count = len(geom.piece_vertex_map["Sleeve"])
        total_verts_count = len(geom.vertices_3d)

        # 中継点頂点が追加されない
        self.assertEqual(cloth_verts_count, total_verts_count)
        # 直接縫合エッジ（ステップ数と等しい）
        self.assertGreater(len(geom.sewing_edges), 0)
        for va, vb in geom.sewing_edges:
            self.assertLess(va, cloth_verts_count)
            self.assertLess(vb, cloth_verts_count)

    def test_mixed_direct_and_bone_wrap_and_rotated_sleeve(self) -> None:
        """横向き袖 (pattern_up_angle_deg=90) における BONE_WRAP + 別ピースとの DIRECT 混在、分岐保護、ボーン無しフォールバック検証."""
        from taremin_cloth.pattern.document import SeamRouting

        doc = Document.new()
        # 横長袖 (幅 0.4 x 高さ 0.2) -> 90°回転して高さ 0.4m の縦方向配置にする
        sleeve = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="HorizSleeve")
        sleeve.target_edge_length = 0.08
        s_nodes = sleeve.get_outer_path().nodes
        doc.placements[sleeve.id] = Placement(
            piece_id=sleeve.id,
            target_bone="Arm_Bone",
            bone_t=0.5,
            bone_angle_deg=0.0,
            surface_offset=0.08,
            pattern_up_angle_deg=90.0,
        )

        # 袖ぐり相手パーツ (非ボーン平面ピース)
        body = doc.add_rectangle_piece((0.6, 0.0), (0.8, 0.2), name="Body")
        body.target_edge_length = 0.08
        b_nodes = body.get_outer_path().nodes

        # 縫合1: 横向き袖の下辺 (n0->n1) と 上辺 (n3->n2) を BONE_WRAP で筒縫い
        wrap_seam = Seam(
            id="wrap_seam",
            a=EdgeRange(piece_id=sleeve.id, path_id=sleeve.outer_path_id, start_node_id=s_nodes[0].id, end_node_id=s_nodes[1].id),
            b=EdgeRange(piece_id=sleeve.id, path_id=sleeve.outer_path_id, start_node_id=s_nodes[3].id, end_node_id=s_nodes[2].id),
            routing=SeamRouting.BONE_WRAP,
        )
        doc.seams[wrap_seam.id] = wrap_seam

        # 縫合2: +90°時にボーン上端になる右辺 (n1->n2) と 身頃左辺 (n3->n0) を通常の DIRECT 縫合
        direct_seam = Seam(
            id="armhole_seam",
            a=EdgeRange(piece_id=sleeve.id, path_id=sleeve.outer_path_id, start_node_id=s_nodes[1].id, end_node_id=s_nodes[2].id),
            b=EdgeRange(piece_id=body.id, path_id=body.outer_path_id, start_node_id=b_nodes[3].id, end_node_id=b_nodes[0].id),
            routing=SeamRouting.DIRECT,
        )
        doc.seams[direct_seam.id] = direct_seam

        bones = {
            "Arm_Bone": BoneInfo3D(
                name="Arm_Bone",
                head_3d=(0.3, 0.0, 1.4),
                tail_3d=(0.3, 0.0, 1.0),
                radius=0.05,
            )
        }
        geom = build_garment_mesh_geometry(doc, bones=bones)
        verts = np.array(geom.vertices_3d)
        sleeve_idxs = geom.piece_vertex_map["HorizSleeve"]
        body_idxs = geom.piece_vertex_map["Body"]
        cloth_count = len(sleeve_idxs) + len(body_idxs)

        # 横長 0.4m の袖が Z 軸方向に約 0.4m の高さで配置されていること
        sleeve_z_span = float(np.max(verts[sleeve_idxs, 2]) - np.min(verts[sleeve_idxs, 2]))
        self.assertAlmostEqual(sleeve_z_span, 0.4, places=2)

        # 中継点 (wp >= cloth_count) が wrap_seam 用に生成され、すべてボーン裏 (+Y > 0) にあること
        self.assertGreater(len(verts), cloth_count)
        for wp_idx in range(cloth_count, len(verts)):
            self.assertGreater(float(verts[wp_idx, 1]), 0.0)

        # DIRECT の armhole_seam は中継点を持たず袖頂点と身頃頂点を直接結ぶこと
        body_set = set(body_idxs)
        sleeve_set = set(sleeve_idxs)
        direct_armhole_edges = [
            (u, v) for u, v in geom.sewing_edges
            if (u in sleeve_set and v in body_set) or (v in sleeve_set and u in body_set)
        ]
        self.assertGreater(len(direct_armhole_edges), 0)

        # ボーン無し時の BONE_WRAP 安全フォールバック検証
        geom_nobone = build_garment_mesh_geometry(doc, bones={})
        total_nobone = len(geom_nobone.piece_vertex_map["HorizSleeve"]) + len(geom_nobone.piece_vertex_map["Body"])
        self.assertEqual(len(geom_nobone.vertices_3d), total_nobone)
        self.assertGreater(len(geom_nobone.sewing_edges), 0)


if __name__ == "__main__":
    unittest.main()

