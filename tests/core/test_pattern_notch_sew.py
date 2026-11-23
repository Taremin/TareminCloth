# -*- coding: utf-8 -*-
"""ノッチによる辺分割・部分縫合・合印位置合わせの単体テスト (Blender非依存)."""

import unittest
import numpy as np

from taremin_cloth.pattern.document import (
    Document,
    EdgeRange,
    NodeKind,
    Notch,
    NotchSnapKind,
    Piece,
    Seam,
)
from taremin_cloth.pattern.geometry import (
    build_arc_table,
    arc_length_at_t,
)


class TestPatternNotchGeometryAndDocument(unittest.TestCase):
    """幾何計算およびDocument層のノッチ・部分区間操作のテスト."""

    def test_arc_length_at_t_linear(self) -> None:
        """直線セグメントにおいて t と弧長が線形比例することを検証."""
        table = build_arc_table((0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0))
        self.assertAlmostEqual(table.total_length, 0.3, places=5)
        self.assertAlmostEqual(arc_length_at_t(table, 0.0), 0.0, places=5)
        self.assertAlmostEqual(arc_length_at_t(table, 0.5), 0.15, places=5)
        self.assertAlmostEqual(arc_length_at_t(table, 1.0), 0.3, places=5)

    def test_compute_edge_range_length_with_notches(self) -> None:
        """EdgeRange の t0, t1 に応じて正確な部分実寸弧長が計算されることを検証."""
        doc = Document.new()
        p = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="Panel")
        outer = p.paths[p.outer_path_id]
        # 下辺 (0.0, 0.0) -> (0.4, 0.0): 長さ 0.4m
        n0_id = outer.nodes[0].id
        n1_id = outer.nodes[1].id

        # 前半 0.0 -> 0.3 (30cm)
        er_half = EdgeRange(
            piece_id=p.id,
            path_id=p.outer_path_id,
            start_node_id=n0_id,
            end_node_id=n1_id,
            t0=0.0,
            t1=0.75,
        )
        length = doc.compute_edge_range_length(er_half)
        self.assertAlmostEqual(length, 0.3, places=4)

    def test_get_segment_notch_breakpoints(self) -> None:
        """セグメント上のノッチ分割比率が昇順で正しく取得できることを検証."""
        doc = Document.new()
        p = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="Panel")
        outer = p.paths[p.outer_path_id]
        n0_id = outer.nodes[0].id

        # ノッチを 0.7 と 0.3 に順不同で追加
        doc.notches["n1"] = Notch(
            id="n1",
            piece_id=p.id,
            path_id=p.outer_path_id,
            segment_start_id=n0_id,
            t=0.7,
        )
        doc.notches["n2"] = Notch(
            id="n2",
            piece_id=p.id,
            path_id=p.outer_path_id,
            segment_start_id=n0_id,
            t=0.3,
        )

        bps = doc.get_segment_notch_breakpoints(p.id, n0_id)
        self.assertEqual(len(bps), 4)
        self.assertAlmostEqual(bps[0], 0.0)
        self.assertAlmostEqual(bps[1], 0.3)
        self.assertAlmostEqual(bps[2], 0.7)
        self.assertAlmostEqual(bps[3], 1.0)

    def test_move_notch_updates_connected_edge_range(self) -> None:
        """ノッチを移動した際に接続中の EdgeRange の t0/t1 が連動更新されることを検証."""
        doc = Document.new()
        p = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="Panel")
        outer = p.paths[p.outer_path_id]
        n0_id = outer.nodes[0].id
        n1_id = outer.nodes[1].id

        notch = Notch(
            id="notch_test",
            piece_id=p.id,
            path_id=p.outer_path_id,
            segment_start_id=n0_id,
            t=0.4,
        )
        doc.notches["notch_test"] = notch

        # ノッチ位置 t=0.4 を境界とする EdgeRange
        er = EdgeRange(
            piece_id=p.id,
            path_id=p.outer_path_id,
            start_node_id=n0_id,
            end_node_id=n1_id,
            t0=0.0,
            t1=0.4,
        )
        seam = Seam(id="seam_test", a=er, b=er)
        doc.seams["seam_test"] = seam

        # ノッチを 0.6 へ移動
        ok = doc.move_notch("notch_test", 0.6)
        self.assertTrue(ok)
        self.assertAlmostEqual(notch.t, 0.6, places=5)
        self.assertAlmostEqual(er.t1, 0.6, places=5)

    def test_sew_tool_picks_notch_sub_segment_headless(self) -> None:
        """SewTool がノッチで区切られた部分区間を選択し、Alt押下で全体を選択することを検証."""
        from taremin_cloth.pattern.editor import EditorController, EditorEvent
        from taremin_cloth.pattern.session import ViewTransform2D

        view = ViewTransform2D(pan_x=0.0, pan_y=0.0, zoom=1000.0)
        doc = Document.new()
        p1 = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="P1")
        p2 = doc.add_rectangle_piece((0.0, 0.3), (0.4, 0.5), name="P2")

        outer1 = p1.paths[p1.outer_path_id]
        # P1下辺 (0.0, 0.0) -> (0.4, 0.0) の t=0.5 (0.2, 0.0) にノッチを追加
        doc.notches["notch1"] = Notch(
            id="notch1",
            piece_id=p1.id,
            path_id=p1.outer_path_id,
            segment_start_id=outer1.nodes[0].id,
            t=0.5,
        )

        ctrl = EditorController(doc=doc, view=view)
        ctrl.set_active_tool("SEW")

        # 1. P1下辺の前半 (0.1, 0.0) -> screen (100, 0) を通常クリック
        ctrl.handle_event(EditorEvent(screen_pos=(100.0, 0.0), button="LEFT", is_press=True))
        self.assertEqual(len(ctrl.preview_seam_candidates), 1)
        er_cand = ctrl.preview_seam_candidates[0]
        self.assertAlmostEqual(er_cand.t0, 0.0)
        self.assertAlmostEqual(er_cand.t1, 0.5)

        # キャンセルして Alt+クリックをテスト
        ctrl.active_tool.cancel(ctrl)
        ctrl.handle_event(EditorEvent(screen_pos=(100.0, 0.0), button="LEFT", is_press=True, alt=True))
        self.assertEqual(len(ctrl.preview_seam_candidates), 1)
        er_cand_alt = ctrl.preview_seam_candidates[0]
        self.assertAlmostEqual(er_cand_alt.t0, 0.0)
        self.assertAlmostEqual(er_cand_alt.t1, 1.0)

    def test_select_tool_drags_notch_headless(self) -> None:
        """SelectTool でノッチをドラッグ移動した際に位置がスナップ・更新されることを検証."""
        from taremin_cloth.pattern.editor import EditorController, EditorEvent
        from taremin_cloth.pattern.session import ViewTransform2D

        view = ViewTransform2D(pan_x=0.0, pan_y=0.0, zoom=1000.0)
        doc = Document.new()
        p = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="P")
        outer = p.paths[p.outer_path_id]
        n0_id = outer.nodes[0].id

        notch = Notch(
            id="drag_notch",
            piece_id=p.id,
            path_id=p.outer_path_id,
            segment_start_id=n0_id,
            t=0.5,
        )
        doc.notches["drag_notch"] = notch

        ctrl = EditorController(doc=doc, view=view)
        ctrl.set_active_tool("SELECT")

        # ノッチ位置 (0.2, 0.0) -> screen (200, 0) でマウスダウン
        ctrl.handle_event(EditorEvent(screen_pos=(200.0, 0.0), button="LEFT", is_press=True))
        self.assertIn("drag_notch", ctrl.selection.notch_ids)

        # (0.3, 0.0) -> screen (300, 0) へドラッグ
        ctrl.handle_event(EditorEvent(screen_pos=(300.0, 0.0), is_move=True))
        # 1cm刻みスナップにより 0.3m -> t=0.75 付近へ移動
        self.assertAlmostEqual(notch.t, 0.75, delta=0.05)

        # マウスアップ
        ctrl.handle_event(EditorEvent(screen_pos=(300.0, 0.0), button="LEFT", is_release=True))
        self.assertTrue(ctrl.history.can_undo)

        # Undo
        ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        self.assertAlmostEqual(ctrl.doc.notches["drag_notch"].t, 0.5, delta=0.01)

    def test_build_multiple_partial_seams_on_single_edge(self) -> None:
        """1つの辺をノッチで2分割し、前半をPieceB、後半をPieceCとそれぞれ縫合した3Dビルド検証."""
        from taremin_cloth.pattern.build_types import build_garment_mesh_geometry

        from taremin_cloth.pattern.document import Placement

        doc = Document.new()
        # PieceA: 40cm x 20cm (下辺: 40cm)
        p_a = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="PieceA")
        # PieceB: 20cm x 20cm
        p_b = doc.add_rectangle_piece((0.0, -0.3), (0.2, -0.1), name="PieceB")
        # PieceC: 20cm x 20cm
        p_c = doc.add_rectangle_piece((0.2, -0.3), (0.4, -0.1), name="PieceC")

        # 重心中心化に対して 2D 座標位置に対応する 3D オフセットを指定
        doc.placements[p_a.id] = Placement(piece_id=p_a.id, offset_3d=(0.2, 0.0, 0.1))
        doc.placements[p_b.id] = Placement(piece_id=p_b.id, offset_3d=(0.1, 0.0, -0.2))
        doc.placements[p_c.id] = Placement(piece_id=p_c.id, offset_3d=(0.3, 0.0, -0.2))

        outer_a = p_a.paths[p_a.outer_path_id]
        outer_b = p_b.paths[p_b.outer_path_id]
        outer_c = p_c.paths[p_c.outer_path_id]

        # PieceA の下辺 (node 0 -> 1) の中央 t=0.5 にノッチを追加
        doc.notches["notch_a"] = Notch(
            id="notch_a",
            piece_id=p_a.id,
            path_id=p_a.outer_path_id,
            segment_start_id=outer_a.nodes[0].id,
            t=0.5,
        )

        # 縫合1: PieceA下辺の前半 [0.0, 0.5] と PieceB上辺 (node 3 -> 2)
        doc.seams["seam_ab"] = Seam(
            id="seam_ab",
            a=EdgeRange(piece_id=p_a.id, path_id=p_a.outer_path_id, start_node_id=outer_a.nodes[0].id, end_node_id=outer_a.nodes[1].id, t0=0.0, t1=0.5),
            b=EdgeRange(piece_id=p_b.id, path_id=p_b.outer_path_id, start_node_id=outer_b.nodes[3].id, end_node_id=outer_b.nodes[2].id, t0=0.0, t1=1.0),
        )

        # 縫合2: PieceA下辺の後半 [0.5, 1.0] と PieceC上辺 (node 3 -> 2)
        doc.seams["seam_ac"] = Seam(
            id="seam_ac",
            a=EdgeRange(piece_id=p_a.id, path_id=p_a.outer_path_id, start_node_id=outer_a.nodes[0].id, end_node_id=outer_a.nodes[1].id, t0=0.5, t1=1.0),
            b=EdgeRange(piece_id=p_c.id, path_id=p_c.outer_path_id, start_node_id=outer_c.nodes[3].id, end_node_id=outer_c.nodes[2].id, t0=0.0, t1=1.0),
        )

        geom = build_garment_mesh_geometry(doc)
        self.assertGreater(len(geom.vertices_3d), 0)
        self.assertGreater(len(geom.faces), 0)
        self.assertGreater(len(geom.sewing_edges), 0)

        # PieceA のノッチ位置 (0.2, 0.0) 付近に頂点が存在すること
        verts = np.array(geom.vertices_3d)
        dists = np.hypot(verts[:, 0] - 0.2, verts[:, 2] - 0.0)
        self.assertLess(np.min(dists), 1e-4)

        # 逆向き EdgeRange (PieceB/C の nodes[3] -> nodes[2]) でも縫合エッジがねじれていないことを検証
        for idx_a, idx_b in geom.sewing_edges:
            x_a = verts[idx_a, 0]
            x_b = verts[idx_b, 0]
            self.assertLess(
                abs(x_a - x_b),
                0.03,
                f"縫合エッジがねじれています: X座標差 = {abs(x_a - x_b):.4f}",
            )

    def test_build_notch_alignment_seam(self) -> None:
        """2つのピースにそれぞれノッチがあり、合印頂点同士が正確に Loose Edge で接続されることの検証."""
        from taremin_cloth.pattern.build_types import build_garment_mesh_geometry
        from taremin_cloth.pattern.document import Placement

        doc = Document.new()
        p1 = doc.add_rectangle_piece((0.0, 0.0), (0.3, 0.2), name="Front")
        p2 = doc.add_rectangle_piece((0.0, 0.4), (0.3, 0.6), name="Back")
        doc.placements[p1.id] = Placement(piece_id=p1.id, offset_3d=(0.15, 0.0, 0.1))
        doc.placements[p2.id] = Placement(piece_id=p2.id, offset_3d=(0.15, 0.0, 0.5))

        outer1 = p1.paths[p1.outer_path_id]
        outer2 = p2.paths[p2.outer_path_id]

        # Front下辺 (0.0, 0.0) -> (0.3, 0.0) の 0.1m 地点 (t = 1/3) にノッチ
        doc.notches["notch_f"] = Notch(
            id="notch_f",
            piece_id=p1.id,
            path_id=p1.outer_path_id,
            segment_start_id=outer1.nodes[0].id,
            t=1.0 / 3.0,
        )

        # Back下辺 (0.0, 0.4) -> (0.3, 0.4) の 0.1m 地点 (t = 1/3) にノッチ
        doc.notches["notch_b"] = Notch(
            id="notch_b",
            piece_id=p2.id,
            path_id=p2.outer_path_id,
            segment_start_id=outer2.nodes[0].id,
            t=1.0 / 3.0,
        )

        # 前半同士、後半同士を縫合
        doc.seams["seam_part1"] = Seam(
            id="seam_part1",
            a=EdgeRange(piece_id=p1.id, path_id=p1.outer_path_id, start_node_id=outer1.nodes[0].id, end_node_id=outer1.nodes[1].id, t0=0.0, t1=1.0 / 3.0),
            b=EdgeRange(piece_id=p2.id, path_id=p2.outer_path_id, start_node_id=outer2.nodes[0].id, end_node_id=outer2.nodes[1].id, t0=0.0, t1=1.0 / 3.0),
        )
        doc.seams["seam_part2"] = Seam(
            id="seam_part2",
            a=EdgeRange(piece_id=p1.id, path_id=p1.outer_path_id, start_node_id=outer1.nodes[0].id, end_node_id=outer1.nodes[1].id, t0=1.0 / 3.0, t1=1.0),
            b=EdgeRange(piece_id=p2.id, path_id=p2.outer_path_id, start_node_id=outer2.nodes[0].id, end_node_id=outer2.nodes[1].id, t0=1.0 / 3.0, t1=1.0),
        )

        geom = build_garment_mesh_geometry(doc)
        self.assertGreater(len(geom.vertices_3d), 0)
        self.assertGreater(len(geom.faces), 0)
        self.assertGreater(len(geom.sewing_edges), 0)

        # Front側のノッチ頂点 (0.1, 0, 0.0) と Back側のノッチ頂点 (0.1, 0, 0.4) が直接 Loose Edge で繋がっているか確認
        verts = np.array(geom.vertices_3d)
        dists_f = np.hypot(verts[:, 0] - 0.1, verts[:, 2] - 0.0)
        idx_f = int(np.argmin(dists_f))
        self.assertLess(dists_f[idx_f], 1e-4)

        dists_b = np.hypot(verts[:, 0] - 0.1, verts[:, 2] - 0.4)
        idx_b = int(np.argmin(dists_b))
        self.assertLess(dists_b[idx_b], 1e-4)

        edge_set = set(geom.sewing_edges)
        is_connected = (idx_f, idx_b) in edge_set or (idx_b, idx_f) in edge_set
        self.assertTrue(is_connected, f"合印ノッチ頂点 {idx_f} と {idx_b} が直接縫合エッジで接続されていません")

    def test_reverse_edge_range_no_twist_3d(self) -> None:
        """外周CCWと逆順に指定された EdgeRange でも 3D 縫合エッジが X 字にねじれないことを検証."""
        from taremin_cloth.pattern.build_types import build_garment_mesh_geometry

        doc = Document.new()
        p_a = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="PieceA")
        p_b = doc.add_rectangle_piece((0.0, 0.3), (0.4, 0.5), name="PieceB")

        outer_a = p_a.paths[p_a.outer_path_id]
        outer_b = p_b.paths[p_b.outer_path_id]

        # PieceA 下辺 (node 0 -> 1: X 0.0 -> 0.4) と PieceB 上辺 (node 3 -> 2: X 0.0 -> 0.4、外周CCWに対して逆向き) を縫合
        doc.seams["seam_rev"] = Seam(
            id="seam_rev",
            a=EdgeRange(
                piece_id=p_a.id,
                path_id=p_a.outer_path_id,
                start_node_id=outer_a.nodes[0].id,
                end_node_id=outer_a.nodes[1].id,
            ),
            b=EdgeRange(
                piece_id=p_b.id,
                path_id=p_b.outer_path_id,
                start_node_id=outer_b.nodes[3].id,
                end_node_id=outer_b.nodes[2].id,
            ),
            reverse=False,
        )

        geom = build_garment_mesh_geometry(doc)
        self.assertGreater(len(geom.sewing_edges), 0)
        verts = np.array(geom.vertices_3d)

        for idx_a, idx_b in geom.sewing_edges:
            x_a = verts[idx_a, 0]
            x_b = verts[idx_b, 0]
            self.assertLess(
                abs(x_a - x_b),
                0.03,
                f"縫合エッジがねじれています: X座標差 = {abs(x_a - x_b):.4f} (v_a={x_a}, v_b={x_b})",
            )

    def test_multiple_notches_reverse_edge_range(self) -> None:
        """同一辺上に複数ノッチがある状態で逆向き EdgeRange を縫合しても正確に整列することを検証."""
        from taremin_cloth.pattern.build_types import build_garment_mesh_geometry

        doc = Document.new()
        p_a = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="PieceA")
        p_b = doc.add_rectangle_piece((0.0, 0.3), (0.4, 0.5), name="PieceB")
        outer_a = p_a.paths[p_a.outer_path_id]
        outer_b = p_b.paths[p_b.outer_path_id]

        for t_val in (0.25, 0.5, 0.75):
            nid = f"notch_a_{t_val}"
            doc.notches[nid] = Notch(
                id=nid,
                piece_id=p_a.id,
                path_id=p_a.outer_path_id,
                segment_start_id=outer_a.nodes[0].id,
                t=t_val,
            )

        # PieceA 下辺 (node 0 -> 1) と PieceB 上辺 (node 3 -> 2: 逆向き指定) を縫合
        doc.seams["seam_multi"] = Seam(
            id="seam_multi",
            a=EdgeRange(
                piece_id=p_a.id,
                path_id=p_a.outer_path_id,
                start_node_id=outer_a.nodes[0].id,
                end_node_id=outer_a.nodes[1].id,
            ),
            b=EdgeRange(
                piece_id=p_b.id,
                path_id=p_b.outer_path_id,
                start_node_id=outer_b.nodes[3].id,
                end_node_id=outer_b.nodes[2].id,
            ),
            reverse=False,
        )

        geom = build_garment_mesh_geometry(doc)
        verts = np.array(geom.vertices_3d)
        for idx_a, idx_b in geom.sewing_edges:
            x_a = verts[idx_a, 0]
            x_b = verts[idx_b, 0]
            self.assertLess(abs(x_a - x_b), 0.03)

    def test_asymmetric_bezier_edge_range(self) -> None:
        """非対称ベジェ曲線上のノッチによる実寸弧長分割の検証."""
        doc = Document.new()
        p = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="BezierPiece")
        outer = p.paths[p.outer_path_id]
        n0 = outer.nodes[0]
        n1 = outer.nodes[1]
        n0.kind = NodeKind.CORNER
        n0.handle_out = (0.05, 0.15)
        n1.kind = NodeKind.CORNER
        n1.handle_in = (-0.1, -0.05)

        doc.notches["nb"] = Notch(
            id="nb",
            piece_id=p.id,
            path_id=p.outer_path_id,
            segment_start_id=n0.id,
            t=0.5,
        )

        er = EdgeRange(
            piece_id=p.id,
            path_id=p.outer_path_id,
            start_node_id=n0.id,
            end_node_id=n1.id,
            t0=0.0,
            t1=0.5,
        )
        sub_len = doc.compute_edge_range_length(er)
        total_len = doc.compute_edge_range_length(
            EdgeRange(
                piece_id=p.id,
                path_id=p.outer_path_id,
                start_node_id=n0.id,
                end_node_id=n1.id,
                t0=0.0,
                t1=1.0,
            )
        )
        self.assertGreater(sub_len, 0.0)
        self.assertLess(sub_len, total_len)

    def test_insert_node_splits_chain_seam(self) -> None:
        """ノード挿入によってセグメントを跨ぐ Seam が SeamMode.CHAIN に昇格・分割されることの検証."""
        doc = Document.new()
        p1 = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="P1")
        p2 = doc.add_rectangle_piece((0.0, 0.3), (0.4, 0.5), name="P2")
        outer1 = p1.paths[p1.outer_path_id]
        outer2 = p2.paths[p2.outer_path_id]

        seam = Seam(
            id="seam_split",
            a=EdgeRange(
                piece_id=p1.id,
                path_id=p1.outer_path_id,
                start_node_id=outer1.nodes[0].id,
                end_node_id=outer1.nodes[1].id,
            ),
            b=EdgeRange(
                piece_id=p2.id,
                path_id=p2.outer_path_id,
                start_node_id=outer2.nodes[0].id,
                end_node_id=outer2.nodes[1].id,
            ),
        )
        doc.seams["seam_split"] = seam

        new_node = doc.insert_node_on_segment(p1.id, p1.outer_path_id, seg_idx=0, t=0.5)
        self.assertIsNotNone(new_node)
        self.assertEqual(len(outer1.nodes), 5)

        updated_seam = doc.seams["seam_split"]
        self.assertEqual(len(updated_seam.chain_a), 2)
        self.assertEqual(updated_seam.chain_a[0].end_node_id, new_node.id)
        self.assertEqual(updated_seam.chain_a[1].start_node_id, new_node.id)

    def test_editing_tools_maintain_notches(self) -> None:
        """対称複製・対称展開・ダーツ挿入でのノッチ維持・再マッピングを検証."""
        from taremin_cloth.pattern.editor import EditorController, EditorEvent
        from taremin_cloth.pattern.session import ViewTransform2D

        view = ViewTransform2D(pan_x=0.0, pan_y=0.0, zoom=1000.0)
        doc = Document.new()
        p = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="Base")
        outer = p.paths[p.outer_path_id]
        n0 = outer.nodes[0]

        doc.notches["n_test"] = Notch(
            id="n_test",
            piece_id=p.id,
            path_id=p.outer_path_id,
            segment_start_id=n0.id,
            t=0.3,
        )

        ctrl = EditorController(doc=doc, view=view)
        ctrl.selection.select_piece(p.id)

        # 1. Shift+U で対称複製
        ctrl.set_active_tool("SELECT")
        ctrl.handle_event(EditorEvent(key="U", shift=True))
        self.assertEqual(len(ctrl.doc.pieces), 2)
        mirror_notches = [n for n in ctrl.doc.notches.values() if n.piece_id != p.id]
        self.assertEqual(len(mirror_notches), 1)
        self.assertAlmostEqual(mirror_notches[0].t, 0.7, places=4)

        # 2. ダーツ挿入でノッチが維持されることの検証
        doc_dart = Document.new()
        p_d = doc_dart.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="DartPiece")
        outer_d = p_d.paths[p_d.outer_path_id]
        doc_dart.notches["n_left"] = Notch(
            id="n_left",
            piece_id=p_d.id,
            path_id=p_d.outer_path_id,
            segment_start_id=outer_d.nodes[0].id,
            t=0.1,
        )
        ctrl_dart = EditorController(doc=doc_dart, view=view)
        ctrl_dart.set_active_tool("DART")
        ctrl_dart.handle_event(EditorEvent(screen_pos=(200.0, 0.0), button="LEFT", is_press=True))
        ctrl_dart.handle_event(EditorEvent(screen_pos=(200.0, 100.0), button="LEFT", is_press=True))
        self.assertIn("n_left", ctrl_dart.doc.notches)

    def test_hud_and_duplicate_notch_prevention(self) -> None:
        """1mm未満のノッチ重複防止およびHUDの排他制御を検証."""
        from taremin_cloth.pattern.editor import EditorController, EditorEvent
        from taremin_cloth.pattern.session import ViewTransform2D

        view = ViewTransform2D(pan_x=0.0, pan_y=0.0, zoom=1000.0)
        doc = Document.new()
        p = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="P")
        ctrl = EditorController(doc=doc, view=view)

        # 1. NotchTool で (0.2, 0.0) -> screen (200, 0) にノッチ作成
        ctrl.set_active_tool("NOTCH")
        ctrl.handle_event(EditorEvent(screen_pos=(200.0, 0.0), button="LEFT", is_press=True))
        self.assertEqual(len(ctrl.doc.notches), 1)
        orig_nid = next(iter(ctrl.doc.notches.keys()))

        # 0.5mm 離れた位置 (0.2005, 0.0) -> screen (200.5, 0) をクリックしても新規作成されず既存ノッチが選択されること
        ctrl.handle_event(EditorEvent(screen_pos=(200.5, 0.0), button="LEFT", is_press=True))
        self.assertEqual(len(ctrl.doc.notches), 1)
        self.assertIn(orig_nid, ctrl.selection.notch_ids)

        # 2. ノッチ選択時に is_piece_card_mode が False になること
        is_piece_mode = (
            not ctrl.selection.notch_ids
            and not ctrl.selection.seam_ids
            and len(ctrl.selection.piece_ids) == 1
        )
        self.assertFalse(is_piece_mode)

    def test_node_deletion_removes_both_side_notches(self) -> None:
        """ノード削除時に、当該ノードが始点および終点となっていた直前セグメントのノッチが除去されることを検証."""
        from taremin_cloth.pattern.editor import EditorController

        doc = Document.new()
        p = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="P")
        outer = p.paths[p.outer_path_id]
        # 5頂点にするため下辺にノード挿入
        new_node = doc.insert_node_on_segment(p.id, p.outer_path_id, seg_idx=0, t=0.5)
        self.assertIsNotNone(new_node)
        self.assertEqual(len(outer.nodes), 5)

        # 挿入ノードの手前セグメント (node 0 -> new_node) と直後セグメント (new_node -> node 1) にノッチを配置
        n0 = outer.nodes[0]
        doc.notches["n_prev"] = Notch(
            id="n_prev",
            piece_id=p.id,
            path_id=p.outer_path_id,
            segment_start_id=n0.id,
            t=0.5,
        )
        doc.notches["n_next"] = Notch(
            id="n_next",
            piece_id=p.id,
            path_id=p.outer_path_id,
            segment_start_id=new_node.id,
            t=0.5,
        )

        ctrl = EditorController(doc=doc)
        ctrl.selection.select_node(p.id, new_node.id)
        # new_node を削除
        ok = ctrl.delete_selected()
        self.assertTrue(ok)
        self.assertEqual(len(outer.nodes), 4)

        # new_node を終点としていた n_prev も、始点としていた n_next も両方削除されていること
        self.assertNotIn("n_prev", ctrl.doc.notches)
        self.assertNotIn("n_next", ctrl.doc.notches)

    def test_hud_rendering_notch_label_and_no_compass(self) -> None:
        """ノッチ選択時・インライン編集時にコンパスが非表示となり、From Start ラベルが描画されることを検証."""
        from taremin_cloth.pattern.editor import EditorController
        from taremin_cloth.pattern.renderer import DrawCommandKind, build_render_commands
        from taremin_cloth.pattern.session import ViewTransform2D

        view = ViewTransform2D(pan_x=0.0, pan_y=0.0, zoom=1000.0)
        doc = Document.new()
        p = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="P")
        outer = p.paths[p.outer_path_id]

        notch = Notch(
            id="n_hud",
            piece_id=p.id,
            path_id=p.outer_path_id,
            segment_start_id=outer.nodes[0].id,
            t=0.5,
        )
        doc.notches["n_hud"] = notch

        ctrl = EditorController(doc=doc, view=view)
        # ピースとノッチが同時選択された状態
        ctrl.selection.select_piece(p.id)
        ctrl.selection.select_notch(p.id, "n_hud")

        commands = build_render_commands(doc, ctrl, screen_size=(800, 600))
        # コンパス円（半径>10）および角度テキストが含まれていないこと
        compass_circles = [c for c in commands if c.kind == DrawCommandKind.CIRCLE and c.radius > 10.0]
        self.assertEqual(len(compass_circles), 0)
        has_angle = any("Angle:" in (c.text or "") for c in commands)
        self.assertFalse(has_angle)

        # インライン編集状態にする
        ctrl.editing_field = "notch_from_start"
        ctrl.editing_buffer = "20.0"
        commands_edit = build_render_commands(doc, ctrl, screen_size=(800, 600))
        text_cmds = [c.text for c in commands_edit if c.kind == DrawCommandKind.TEXT and c.text]
        # "From Start: 20.0|" または "開始点から: 20.0|" が含まれること
        has_label = any("From Start: 20.0|" in t or "開始点から: 20.0|" in t for t in text_cmds)
        self.assertTrue(has_label, f"テキスト描画に From Start が見つかりません: {text_cmds}")

    def test_resolve_notch_snap_kinds_and_split_lengths(self) -> None:
        """resolve_notch_snap が 1cm刻み / 中点 / Free / 既存ノッチ重複 / 分割弧長を正しく解決すること."""
        doc = Document.new()
        p = doc.add_rectangle_piece((0.0, 0.0), (0.35, 0.20), name="Rect")
        outer = p.paths[p.outer_path_id]
        n0 = outer.nodes[0]  # 下辺: 長さ 0.35m (35cm)

        # 1. 中点スナップ (t=0.5 付近 -> 17.5cm ずつ)
        res_mid = doc.resolve_notch_snap(p.id, n0.id, raw_t=0.495, path_id=outer.id)
        self.assertEqual(res_mid.kind, NotchSnapKind.MIDPOINT)
        self.assertAlmostEqual(res_mid.t, 0.5, places=5)
        self.assertAlmostEqual(res_mid.split_left_len, 0.175, places=5)
        self.assertAlmostEqual(res_mid.split_right_len, 0.175, places=5)
        self.assertEqual(res_mid.left_t_range, (0.0, 0.5))
        self.assertEqual(res_mid.right_t_range, (0.5, 1.0))

        # 2. 1cm 刻みスナップ (raw_t=0.3 -> 0.105m -> 0.10m へ丸め)
        res_step = doc.resolve_notch_snap(p.id, n0.id, raw_t=0.295, path_id=outer.id)
        self.assertEqual(res_step.kind, NotchSnapKind.STEP_1CM)
        self.assertAlmostEqual(res_step.arc_s, 0.10, places=4)
        self.assertAlmostEqual(res_step.split_left_len, 0.10, places=4)
        self.assertAlmostEqual(res_step.split_right_len, 0.25, places=4)

        # 3. Alt 押下相当 (snap_enabled=False) -> FREE
        raw_t_free = 0.295
        res_free = doc.resolve_notch_snap(
            p.id, n0.id, raw_t=raw_t_free, path_id=outer.id, snap_enabled=False
        )
        self.assertEqual(res_free.kind, NotchSnapKind.FREE)
        self.assertAlmostEqual(res_free.t, raw_t_free, places=5)
        self.assertAlmostEqual(res_free.split_left_len + res_free.split_right_len, 0.35, places=5)

        # 4. 既存ノッチ近傍 (< 1mm) -> EXISTING + existing_notch_id
        doc.notches["ex1"] = Notch(
            id="ex1", piece_id=p.id, path_id=outer.id, segment_start_id=n0.id, t=0.10 / 0.35
        )
        res_ex = doc.resolve_notch_snap(p.id, n0.id, raw_t=(0.10 / 0.35) + 0.001, path_id=outer.id)
        self.assertEqual(res_ex.kind, NotchSnapKind.EXISTING)
        self.assertEqual(res_ex.existing_notch_id, "ex1")

        # 5. 非対称ベジェ曲線で、既存ノッチ ex1 がある場合のサブ区間和と、単独時の全長和を検証
        outer.nodes[0].kind = NodeKind.SMOOTH
        outer.nodes[0].out_handle = (0.05, 0.15)
        res_bez_sub = doc.resolve_notch_snap(p.id, n0.id, raw_t=0.40, path_id=outer.id, snap_enabled=False)
        self.assertGreater(res_bez_sub.left_t_range[0], 0.0)
        self.assertLess(res_bez_sub.split_left_len + res_bez_sub.split_right_len, res_bez_sub.total_length)

        del doc.notches["ex1"]
        res_bez = doc.resolve_notch_snap(p.id, n0.id, raw_t=0.40, path_id=outer.id, snap_enabled=False)
        self.assertAlmostEqual(res_bez.split_left_len + res_bez.split_right_len, res_bez.total_length, places=4)

    def test_resolve_notch_snap_clipboard_and_seam_partner(self) -> None:
        """クリップボード参照長および Seam 相手ノッチ位置へ優先スナップすることを検証."""
        doc = Document.new()
        p_a = doc.add_rectangle_piece((0.0, 0.0), (0.40, 0.20), name="A")
        p_b = doc.add_rectangle_piece((0.6, 0.0), (0.30, 0.20), name="B")
        outer_a = p_a.paths[p_a.outer_path_id]
        outer_b = p_b.paths[p_b.outer_path_id]
        a0, a1 = outer_a.nodes[0], outer_a.nodes[1]  # 長さ 0.40m
        b0, b1 = outer_b.nodes[0], outer_b.nodes[1]  # 長さ 0.30m

        # 1. クリップボード長 (0.123m) へのスナップ（1cm刻み 0.120m より優先）
        res_clip = doc.resolve_notch_snap(
            p_a.id,
            a0.id,
            raw_t=0.124 / 0.40,
            path_id=outer_a.id,
            clipboard_length=0.123,
        )
        self.assertEqual(res_clip.kind, NotchSnapKind.CLIPBOARD)
        self.assertAlmostEqual(res_clip.arc_s, 0.123, places=4)
        self.assertAlmostEqual(res_clip.ref_length, 0.123, places=4)

        # snap_enabled=False 時はクリップボード長へもスナップしない
        res_clip_off = doc.resolve_notch_snap(
            p_a.id,
            a0.id,
            raw_t=0.124 / 0.40,
            path_id=outer_a.id,
            snap_enabled=False,
            clipboard_length=0.123,
        )
        self.assertEqual(res_clip_off.kind, NotchSnapKind.FREE)

        # 2. Seam 相手ノッチへのスナップ (reverse=False, B側ノッチ = 始点から 0.085m)
        doc.notches["nb"] = Notch(
            id="nb",
            piece_id=p_b.id,
            path_id=outer_b.id,
            segment_start_id=b0.id,
            t=0.085 / 0.30,
        )
        seam = Seam(
            id="s_ab",
            a=EdgeRange(piece_id=p_a.id, path_id=outer_a.id, start_node_id=a0.id, end_node_id=a1.id),
            b=EdgeRange(piece_id=p_b.id, path_id=outer_b.id, start_node_id=b0.id, end_node_id=b1.id),
            reverse=False,
        )
        doc.seams[seam.id] = seam

        res_seam = doc.resolve_notch_snap(
            p_a.id,
            a0.id,
            raw_t=0.086 / 0.40,
            path_id=outer_a.id,
        )
        self.assertEqual(res_seam.kind, NotchSnapKind.SEAM_PARTNER)
        self.assertAlmostEqual(res_seam.arc_s, 0.085, places=4)
        self.assertAlmostEqual(res_seam.ref_length, 0.085, places=4)

        # 3. reverse=True のとき: 相手全長 0.30 - 0.085 = 0.215m および 自全長 0.40 - 0.085 = 0.315m 側へスナップ
        seam.reverse = True
        res_seam_rev = doc.resolve_notch_snap(
            p_a.id,
            a0.id,
            raw_t=0.316 / 0.40,
            path_id=outer_a.id,
        )
        self.assertEqual(res_seam_rev.kind, NotchSnapKind.SEAM_PARTNER)
        self.assertAlmostEqual(res_seam_rev.arc_s, 0.315, places=4)

        # 4. 相手側が逆方向 EdgeRange (b1 -> b0) でも正しく対応弧長へマップされること
        seam.reverse = False
        seam.b = EdgeRange(piece_id=p_b.id, path_id=outer_b.id, start_node_id=b1.id, end_node_id=b0.id)
        # b 終端(b1)から nb (b0+0.085) までの距離は 0.30 - 0.085 = 0.215m
        res_seam_inv_er = doc.resolve_notch_snap(
            p_a.id,
            a0.id,
            raw_t=0.214 / 0.40,
            path_id=outer_a.id,
        )
        self.assertEqual(res_seam_inv_er.kind, NotchSnapKind.SEAM_PARTNER)
        self.assertAlmostEqual(res_seam_inv_er.arc_s, 0.215, places=4)

    def test_resolve_notch_snap_existing_preserves_both_split_lengths(self) -> None:
        """EXISTING 判定時に左右両方の分割区間と弧長が非ゼロで正しく算出されることを検証."""
        doc = Document.new()
        p = doc.add_rectangle_piece((0.0, 0.0), (0.3, 0.2), name="Rect")
        outer = p.paths[p.outer_path_id]
        n0 = outer.nodes[0]

        doc.notches["n1"] = Notch(id="n1", piece_id=p.id, path_id=outer.id, segment_start_id=n0.id, t=0.25)
        doc.notches["n2"] = Notch(id="n2", piece_id=p.id, path_id=outer.id, segment_start_id=n0.id, t=0.75)

        res1 = doc.resolve_notch_snap(p.id, n0.id, raw_t=0.251, path_id=outer.id)
        self.assertEqual(res1.kind, NotchSnapKind.EXISTING)
        self.assertEqual(res1.existing_notch_id, "n1")
        self.assertEqual(res1.left_t_range, (0.0, 0.25))
        self.assertEqual(res1.right_t_range, (0.25, 0.75))
        self.assertAlmostEqual(res1.split_left_len, 0.075, places=4)
        self.assertAlmostEqual(res1.split_right_len, 0.150, places=4)

        res2 = doc.resolve_notch_snap(p.id, n0.id, raw_t=0.749, path_id=outer.id)
        self.assertEqual(res2.kind, NotchSnapKind.EXISTING)
        self.assertEqual(res2.existing_notch_id, "n2")
        self.assertEqual(res2.left_t_range, (0.25, 0.75))
        self.assertEqual(res2.right_t_range, (0.75, 1.0))
        self.assertAlmostEqual(res2.split_left_len, 0.150, places=4)
        self.assertAlmostEqual(res2.split_right_len, 0.075, places=4)

    def test_clamp_boundary_notch_remains_in_breakpoints_on_bezier(self) -> None:
        """隣接ノッチ 1mm 限界クランプで移動した後も get_segment_notch_breakpoints から脱落しないことを検証."""
        from taremin_cloth.pattern.document import Node, NodeKind, Path, PathRole, Piece

        doc = Document.new()
        outer = Path(
            id="outer",
            role=PathRole.OUTER,
            nodes=[
                Node(id="b0", position=(0.0, 0.0), kind=NodeKind.SMOOTH, out_handle=(0.05, 0.12)),
                Node(id="b1", position=(0.3, 0.0), kind=NodeKind.SMOOTH, in_handle=(0.24, -0.08)),
                Node(id="b2", position=(0.3, 0.2), kind=NodeKind.CORNER),
                Node(id="b3", position=(0.0, 0.2), kind=NodeKind.CORNER),
            ],
            closed=True,
        )
        p = Piece(id="p_bez", name="Bez", paths={"outer": outer}, outer_path_id="outer")
        doc.pieces[p.id] = p

        doc.notches["n_fixed"] = Notch(id="n_fixed", piece_id=p.id, path_id=outer.id, segment_start_id="b0", t=0.4)
        doc.notches["n_move"] = Notch(id="n_move", piece_id=p.id, path_id=outer.id, segment_start_id="b0", t=0.6)

        snap_res = doc.resolve_notch_snap(
            p.id,
            "b0",
            raw_t=0.4001,
            path_id=outer.id,
            snap_enabled=False,
            exclude_notch_id="n_move",
        )
        moved = doc.move_notch("n_move", snap_res.t)
        self.assertTrue(moved)

        bps = doc.get_segment_notch_breakpoints(p.id, "b0", path_id=outer.id)
        self.assertEqual(len(bps), 4)
        self.assertAlmostEqual(bps[1], 0.4, places=5)
        self.assertAlmostEqual(bps[2], doc.notches["n_move"].t, places=5)
        self.assertAlmostEqual(snap_res.left_t_range[0], bps[1], places=5)
        self.assertAlmostEqual(snap_res.left_t_range[1], bps[2], places=5)
        self.assertAlmostEqual(snap_res.right_t_range[1], bps[3], places=5)

    def test_resolve_notch_snap_chain_seam_partner(self) -> None:
        """複数セグメント連結の CHAIN 縫合越しに縫合相手ノッチ距離へスナップすることを検証."""
        doc = Document.new()
        p_a = doc.add_rectangle_piece((0.0, 0.0), (0.3, 0.2), name="PieceA")
        p_b = doc.add_rectangle_piece((0.0, 0.3), (0.3, 0.5), name="PieceB")
        outer_a = p_a.paths[p_a.outer_path_id]
        outer_b = p_b.paths[p_b.outer_path_id]
        a0, a1 = outer_a.nodes[0], outer_a.nodes[1]
        b0, b1 = outer_b.nodes[0], outer_b.nodes[1]

        seam = Seam(
            id="seam_chain",
            a=EdgeRange(piece_id=p_a.id, path_id=outer_a.id, start_node_id=a0.id, end_node_id=a1.id),
            b=EdgeRange(piece_id=p_b.id, path_id=outer_b.id, start_node_id=b0.id, end_node_id=b1.id),
            reverse=False,
        )
        doc.seams[seam.id] = seam

        # Piece A の下辺 (0.30m) を t=0.5 (0.15m) で分割して CHAIN 縫合化
        mid_node = doc.insert_node_on_segment(p_a.id, outer_a.id, seg_idx=0, t=0.5)
        self.assertIsNotNone(mid_node)
        self.assertEqual(len(seam.chain_a), 2)

        # Piece A の 2 セグメント目（先頭から 0.15 + 0.07 = 0.22m）へノッチを配置
        doc.notches["na_chain"] = Notch(
            id="na_chain",
            piece_id=p_a.id,
            path_id=outer_a.id,
            segment_start_id=mid_node.id,
            t=0.07 / 0.15,
        )

        # 順方向: Piece B の先頭から 0.22m 付近へスナップすること
        res_fwd = doc.resolve_notch_snap(p_b.id, b0.id, raw_t=0.221 / 0.30, path_id=outer_b.id)
        self.assertEqual(res_fwd.kind, NotchSnapKind.SEAM_PARTNER)
        self.assertAlmostEqual(res_fwd.arc_s, 0.22, places=4)
        self.assertAlmostEqual(res_fwd.ref_length, 0.22, places=4)

        # reverse=True: Piece B の終点から 0.22m (= 先頭から 0.08m) 側へスナップすること
        seam.reverse = True
        res_rev = doc.resolve_notch_snap(p_b.id, b0.id, raw_t=0.079 / 0.30, path_id=outer_b.id)
        self.assertEqual(res_rev.kind, NotchSnapKind.SEAM_PARTNER)
        self.assertAlmostEqual(res_rev.arc_s, 0.08, places=4)


if __name__ == "__main__":
    unittest.main()
