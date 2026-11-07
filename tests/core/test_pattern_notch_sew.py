# -*- coding: utf-8 -*-
"""ノッチによる辺分割・部分縫合・合印位置合わせの単体テスト (Blender非依存)."""

import unittest
import numpy as np

from taremin_cloth.pattern.document import (
    Document,
    EdgeRange,
    NodeKind,
    Notch,
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

        doc = Document.new()
        # PieceA: 40cm x 20cm (下辺: 40cm)
        p_a = doc.add_rectangle_piece((0.0, 0.0), (0.4, 0.2), name="PieceA")
        # PieceB: 20cm x 20cm
        p_b = doc.add_rectangle_piece((0.0, -0.3), (0.2, -0.1), name="PieceB")
        # PieceC: 20cm x 20cm
        p_c = doc.add_rectangle_piece((0.2, -0.3), (0.4, -0.1), name="PieceC")

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

        doc = Document.new()
        p1 = doc.add_rectangle_piece((0.0, 0.0), (0.3, 0.2), name="Front")
        p2 = doc.add_rectangle_piece((0.0, 0.4), (0.3, 0.6), name="Back")

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


if __name__ == "__main__":
    unittest.main()
