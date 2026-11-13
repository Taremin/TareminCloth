# -*- coding: utf-8 -*-
"""型紙エディタの辺操作・対称化・反転・形状転写の単体テスト (Blender非依存)."""

import math
import unittest

from taremin_cloth.pattern.build_types import build_garment_mesh_geometry
from taremin_cloth.pattern.document import (
    Document,
    EdgeRange,
    Node,
    NodeKind,
    Notch,
    Path,
    PathRole,
    Placement,
    Seam,
    SeamMode,
)
from taremin_cloth.pattern.editor import EditorController, EditorEvent
from taremin_cloth.pattern.geometry import signed_area
from taremin_cloth.pattern.layout import compute_hud_layout
from taremin_cloth.pattern.renderer import DrawCommandKind, build_render_commands
from taremin_cloth.pattern.session import ViewTransform2D
from taremin_cloth.pattern.theme import DEFAULT_THEME


class TestPatternEdgeOperations(unittest.TestCase):
    """辺選択、中点分割、任意辺対称展開・複製、反転、ベジェ形状転写のテスト."""

    def setUp(self):
        # 1m = 1000px, 原点 (0, 0) -> screen (0, 0)
        self.view = ViewTransform2D(pan_x=0.0, pan_y=0.0, zoom=1000.0)
        self.doc = Document.new()
        self.controller = EditorController(doc=self.doc, view=self.view)

    def test_edge_selection_preserves_piece_selection_and_renders_highlight(self):
        """外周辺クリック時に辺選択とピース選択が両立し、POLYLINE数を変えずにハイライト描画されること."""
        ctrl = self.controller
        piece = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2), name="Rect")
        pid = piece.id
        outer = piece.get_outer_path()
        self.assertIsNotNone(outer)

        # 下辺 (0.0, 0.0) -> (0.2, 0.0) の中点 (0.1, 0.0) -> 画面 (100.0, 0.0) をクリック
        ctrl.set_active_tool("SELECT")
        ctrl.handle_event(EditorEvent(screen_pos=(100.0, 0.0), is_press=True))

        # 辺選択状態の検証
        self.assertEqual(len(ctrl.selection.edge_ranges), 1)
        er = ctrl.selection.get_active_edge_range()
        self.assertIsNotNone(er)
        self.assertEqual(er.piece_id, pid)
        # 辺選択時は親ピース自体を piece_ids に混入させない（辺移動・誤削除防止）
        self.assertNotIn(pid, ctrl.selection.piece_ids)
        self.assertEqual(ctrl.selection.active_piece_id, pid)

        # 描画コマンドの検証（POLYLINE 総数が4のまま、選択辺のみハイライト色・太線）
        commands = build_render_commands(ctrl.doc, ctrl, screen_size=(1280, 720))
        polylines = [cmd for cmd in commands if cmd.kind == DrawCommandKind.POLYLINE]
        self.assertEqual(len(polylines), 4)

        selected_lines = [
            cmd for cmd in polylines
            if cmd.color == DEFAULT_THEME.edge_selected and cmd.width == 3.5
        ]
        self.assertEqual(len(selected_lines), 1)

        # 辺を掴んだままのドラッグ移動（100.0 -> 150.0 px, +0.05m）: 辺の両端頂点のみが移動し piece.transform は不変
        ctrl.handle_event(EditorEvent(screen_pos=(150.0, 0.0), is_move=True))
        ctrl.handle_event(EditorEvent(screen_pos=(150.0, 0.0), is_release=True))
        self.assertAlmostEqual(piece.transform.tx, 0.0, places=5)
        self.assertAlmostEqual(outer.nodes[0].position[0], 0.05, places=5)
        self.assertAlmostEqual(outer.nodes[1].position[0], 0.25, places=5)
        self.assertAlmostEqual(outer.nodes[2].position[0], 0.2, places=5)
        self.assertAlmostEqual(outer.nodes[3].position[0], 0.0, places=5)

    def test_subdivide_edge_w_key_and_double_click(self):
        """ベジェ曲線辺を実寸弧長50%で分割し、ダブルクリックでもノードが挿入されること."""
        ctrl = self.controller
        piece = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2), name="CurvedPiece")
        pid = piece.id
        outer = piece.get_outer_path()
        self.assertIsNotNone(outer)

        # 下辺 (セグメント0) に非対称なベジェハンドルを設定して曲線化
        n0 = outer.nodes[0]
        n1 = outer.nodes[1]
        n0.out_handle = (0.05, 0.08)
        n1.in_handle = (-0.05, 0.03)
        n0.kind = NodeKind.CORNER
        n1.kind = NodeKind.CORNER

        orig_er = EdgeRange(piece_id=pid, path_id=outer.id, start_node_id=n0.id, end_node_id=n1.id)
        orig_len = ctrl.doc.compute_edge_range_length(orig_er)

        # 辺を選択して W キー押下
        ctrl.selection.select_edge(orig_er)
        handled = ctrl.handle_event(EditorEvent(key="W"))
        self.assertTrue(handled)

        # 分割後のノード数が 4 -> 5 に増加
        self.assertEqual(len(outer.nodes), 5)
        # 挿入された新ノードが選択状態になっていること
        self.assertEqual(len(ctrl.selection.node_ids), 1)
        mid_node_id = next(iter(ctrl.selection.node_ids))
        self.assertEqual(outer.nodes[1].id, mid_node_id)

        # 分割後の2区間の弧長がともに元の弧長の50%（許容誤差 1e-4m）であること
        er_left = EdgeRange(piece_id=pid, path_id=outer.id, start_node_id=n0.id, end_node_id=mid_node_id)
        er_right = EdgeRange(piece_id=pid, path_id=outer.id, start_node_id=mid_node_id, end_node_id=n1.id)
        len_left = ctrl.doc.compute_edge_range_length(er_left)
        len_right = ctrl.doc.compute_edge_range_length(er_right)
        self.assertAlmostEqual(len_left, orig_len * 0.5, places=4)
        self.assertAlmostEqual(len_right, orig_len * 0.5, places=4)

        # 上辺 (X: 0.2->0.0, Y: 0.2) 上でダブルクリック (画面 100.0, 200.0)
        handled_dbl = ctrl.handle_event(EditorEvent(
            screen_pos=(100.0, 200.0),
            is_press=True,
            is_double_click=True,
        ))
        self.assertTrue(handled_dbl)
        self.assertEqual(len(outer.nodes), 6)

    def test_unfold_and_mirror_duplicate_along_arbitrary_selected_edge(self):
        """任意に選択した上辺を軸とした対称展開、および右辺を軸とした対称複製・平行縫合の検証."""
        ctrl = self.controller
        piece = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.3), name="Rect")
        pid = piece.id
        outer = piece.get_outer_path()

        # 上辺 (セグメント2: (0.2, 0.3) -> (0.0, 0.3)) を選択
        n2 = outer.nodes[2]
        n3 = outer.nodes[3]
        er_top = EdgeRange(piece_id=pid, path_id=outer.id, start_node_id=n2.id, end_node_id=n3.id)
        ctrl.selection.select_edge(er_top)

        # U キー押下（上辺を軸に対称展開）
        self.assertTrue(ctrl.handle_event(EditorEvent(key="U")))
        self.assertEqual(len(ctrl.doc.pieces), 1)
        unfolded_piece = ctrl.doc.pieces[pid]
        unfolded_outer = unfolded_piece.get_outer_path()
        ys = [nd.position[1] for nd in unfolded_outer.nodes]
        min_y, max_y = min(ys), max(ys)
        self.assertAlmostEqual(min_y, 0.0, places=4)
        self.assertAlmostEqual(max_y, 0.6, places=4)  # 0.3基準で上側へ対称展開

        # 右辺を軸に対称複製 (Shift+U) を検証するための別ピース
        p2 = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2), name="Sleeve_L")
        pid2 = p2.id
        outer2 = p2.get_outer_path()
        # 右辺 (セグメント1: (0.2, 0.0) -> (0.2, 0.2)) を選択
        er_right = EdgeRange(piece_id=pid2, path_id=outer2.id, start_node_id=outer2.nodes[1].id, end_node_id=outer2.nodes[2].id)
        ctrl.selection.select_edge(er_right)

        self.assertTrue(ctrl.handle_event(EditorEvent(key="U", shift=True)))
        self.assertEqual(len(ctrl.doc.pieces), 3)
        sym_piece = next(p for p in ctrl.doc.pieces.values() if p.id not in (pid, pid2))
        self.assertEqual(sym_piece.name, "Sleeve_R")

        # 右辺の外側 (+X 方向) に約5cm離れて配置されていること
        self.assertGreater(sym_piece.transform.tx, p2.transform.tx + 0.04)

        # 自動生成された Seam の平行性を検証
        self.assertEqual(len(ctrl.doc.seams), 1)
        seam = next(iter(ctrl.doc.seams.values()))
        self.assertTrue(seam.reverse)
        geom = ctrl.doc.get_seam_geometry(seam)
        # A側始点・終点とB側始点・終点
        w_a0, _, w_a1, w_b0, _, w_b1 = geom
        dist_start = math.hypot(w_b0[0] - w_a0[0], w_b0[1] - w_a0[1])
        dist_end = math.hypot(w_b1[0] - w_a1[0], w_b1[1] - w_a1[1])
        self.assertAlmostEqual(dist_start, dist_end, places=3)  # 交差せず平行

    def test_duplicate_and_flip_piece_preserves_ccw_and_holes_and_notches(self):
        """反転・複製時に閉パスの反時計回り（CCW）、穴、ノッチ、ボーン配置が整合性を保つこと."""
        ctrl = self.controller
        piece = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2), name="Front_L")
        pid = piece.id
        outer = piece.get_outer_path()

        # 穴を追加 (中心 0.1, 0.1, サイズ 0.04)
        hole_id = "hole_1"
        hole_nodes = [
            Node(id="h0", position=(0.08, 0.08), kind=NodeKind.CORNER),
            Node(id="h1", position=(0.12, 0.08), kind=NodeKind.CORNER),
            Node(id="h2", position=(0.12, 0.12), kind=NodeKind.CORNER),
            Node(id="h3", position=(0.08, 0.12), kind=NodeKind.CORNER),
        ]
        piece.paths[hole_id] = Path(id=hole_id, role=PathRole.HOLE, nodes=hole_nodes, closed=True)

        # ノッチを下辺に追加
        nid = "notch_1"
        ctrl.doc.notches[nid] = Notch(
            id=nid,
            piece_id=pid,
            path_id=outer.id,
            segment_start_id=outer.nodes[0].id,
            t=0.25,
        )

        # ボーン配置を設定
        ctrl.doc.placements[pid] = Placement(
            piece_id=pid,
            target_bone="UpperArm.L",
            preset="ARM_LEFT",
            bone_angle_deg=30.0,
        )

        ctrl.selection.select_piece(pid)

        # 1. 左右反転複製 (Ctrl+Shift+D)
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, shift=True, key="D")))
        self.assertEqual(len(ctrl.doc.pieces), 2)
        dup_pid = next(p for p in ctrl.doc.pieces if p != pid)
        dup_piece = ctrl.doc.pieces[dup_pid]
        self.assertEqual(dup_piece.name, "Front_R")

        # ボーン左右反転の検証
        dup_pl = ctrl.doc.placements.get(dup_pid)
        self.assertIsNotNone(dup_pl)
        self.assertEqual(dup_pl.target_bone, "UpperArm.R")
        self.assertEqual(dup_pl.preset, "ARM_RIGHT")
        self.assertAlmostEqual(dup_pl.bone_angle_deg, -30.0, places=4)

        # 2. 左右反転 (M)
        ctrl.selection.clear()
        ctrl.selection.select_piece(pid)
        self.assertTrue(ctrl.handle_event(EditorEvent(key="M")))

        # CCW（反時計回り）符号の維持検証
        outer_poly = [n.position for n in outer.nodes]
        hole_poly = [n.position for n in piece.paths[hole_id].nodes]
        self.assertGreater(signed_area(outer_poly), 0.0)
        self.assertGreater(signed_area(hole_poly), 0.0)

        # ノッチが鏡像位置（1.0 - 0.25 = 0.75）に対応していること
        notch = ctrl.doc.notches[nid]
        self.assertAlmostEqual(notch.t, 0.75, places=4)

        # 3Dメッシュ化に成功すること（縮退三角形なし）
        geom = build_garment_mesh_geometry(ctrl.doc)
        self.assertGreater(len(geom.vertices_3d), 0)
        self.assertGreater(len(geom.faces), 0)

    def test_copy_paste_edge_shape_modes_and_ease(self):
        """ベジェ曲線形状のコピー＆ペーストと3つの合わせ方モード、イセコミ加算の検証."""
        ctrl = self.controller
        p_src = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2), name="Src")
        p_dst = ctrl.doc.add_rectangle_piece((0.3, 0.0), (0.5, 0.2), name="Dst")
        pid_src = p_src.id
        pid_dst = p_dst.id
        out_src = p_src.get_outer_path()
        out_dst = p_dst.get_outer_path()

        # ソースの下辺をベジェ曲線にする（膨らみ形状）
        out_src.nodes[0].out_handle = (0.06, 0.08)
        out_src.nodes[1].in_handle = (-0.06, 0.08)
        er_src = EdgeRange(piece_id=pid_src, path_id=out_src.id, start_node_id=out_src.nodes[0].id, end_node_id=out_src.nodes[1].id)
        src_len = ctrl.doc.compute_edge_range_length(er_src)
        self.assertGreater(src_len, 0.201)

        # 形状コピー (Ctrl+Shift+C)
        ctrl.selection.select_edge(er_src)
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, shift=True, key="C")))
        self.assertIsNotNone(ctrl.edge_clipboard)

        # ターゲットの下辺を選択
        er_dst = EdgeRange(piece_id=pid_dst, path_id=out_dst.id, start_node_id=out_dst.nodes[0].id, end_node_id=out_dst.nodes[1].id)
        ctrl.selection.select_edge(er_dst)

        # Mode 1: LENGTH_ADJUST_BULGE + イセコミ 5mm (0.005m)
        ctrl.shape_match_mode = "LENGTH_ADJUST_BULGE"
        ctrl.shape_match_ease_value = 0.005
        ctrl.shape_match_ease_is_ratio = False
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, shift=True, key="V")))

        pasted_len = ctrl.doc.compute_edge_range_length(er_dst)
        self.assertAlmostEqual(pasted_len, src_len + 0.005, places=4)
        # 隣接する右辺（セグメント1）の端点ノードハンドルが非破壊で維持されていること
        self.assertIsNone(out_dst.nodes[1].out_handle)

        # Mode 2: LENGTH_MOVE_END
        ctrl.shape_match_mode = "LENGTH_MOVE_END"
        ctrl.shape_match_ease_value = 0.010  # +1cm
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, shift=True, key="V")))
        pasted_len_move = ctrl.doc.compute_edge_range_length(er_dst)
        self.assertAlmostEqual(pasted_len_move, src_len + 0.010, places=4)
        # 終点ノードが伸張移動していること
        self.assertGreater(out_dst.nodes[1].position[0], 0.20)

        # Mode 3: ペースト先の弦長がコピー元より長いケース（直線に潰れずカーブが転写されることの回帰検証）
        p_long = ctrl.doc.add_rectangle_piece((0.6, 0.0), (1.0, 0.2), name="Long")
        out_long = p_long.get_outer_path()
        er_long = EdgeRange(piece_id=p_long.id, path_id=out_long.id, start_node_id=out_long.nodes[0].id, end_node_id=out_long.nodes[1].id)
        ctrl.selection.clear()
        ctrl.selection.select_edge(er_long)
        ctrl.shape_match_mode = "LENGTH_ADJUST_BULGE"
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, shift=True, key="V")))
        self.assertIsNotNone(out_long.nodes[0].out_handle)
        self.assertIsNotNone(out_long.nodes[1].in_handle)

    def test_seam_one_click_match_and_hud_card_clicks(self):
        """Seam選択時のワンクリック同期 (Ctrl+M / Ctrl+Shift+M) とHUDカードクリック、Undo検証."""
        ctrl = self.controller
        p_a = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.25, 0.2), name="A")
        p_b = ctrl.doc.add_rectangle_piece((0.0, 0.3), (0.20, 0.5), name="B")
        pid_a = p_a.id
        pid_b = p_b.id
        out_a = p_a.get_outer_path()
        out_b = p_b.get_outer_path()

        # Aの下辺 (長さ約0.27m, 曲線) と Bの下辺 (長さ0.20m, 直線) を結ぶ Seam
        out_a.nodes[0].out_handle = (0.06, 0.05)
        out_a.nodes[1].in_handle = (-0.06, 0.05)
        er_a = EdgeRange(piece_id=pid_a, path_id=out_a.id, start_node_id=out_a.nodes[0].id, end_node_id=out_a.nodes[1].id)
        er_b = EdgeRange(piece_id=pid_b, path_id=out_b.id, start_node_id=out_b.nodes[0].id, end_node_id=out_b.nodes[1].id)

        sid = "seam_test"
        ctrl.doc.seams[sid] = Seam(id=sid, a=er_a, b=er_b, reverse=True, mode=SeamMode.SEGMENT)

        # Seam 選択
        ctrl.selection.select_seam(sid)
        len_a, len_b, mismatch = ctrl.doc.compute_seam_lengths(sid)
        self.assertGreater(mismatch, 0.04)

        # Ctrl+M: A -> B へ形状・弧長を同期
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, key="M")))
        new_len_a, new_len_b, new_mismatch = ctrl.doc.compute_seam_lengths(sid)
        self.assertLess(new_mismatch, 0.001)

        # HUD カード上のモード切替クリック判定の検証
        layout = compute_hud_layout(1280, 720)
        mode_field = layout.card_fields["seam_match_mode"]
        ctrl.handle_event(EditorEvent(screen_pos=(mode_field.x + 10.0, mode_field.y + 5.0), is_press=True))
        self.assertEqual(ctrl.shape_match_mode, "LENGTH_MOVE_END")

        # Undo (Ctrl+Z) で同期前の形状・長さに戻ること
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, key="Z")))
        restored_len_a, restored_len_b, restored_mismatch = ctrl.doc.compute_seam_lengths(sid)
        self.assertGreater(restored_mismatch, 0.05)

    def test_copy_paste_pieces_and_clipboard(self):
        """ピースのコピー(Ctrl+C)・ペースト(Ctrl+V)および元ピース削除後の復元を検証."""
        ctrl = self.controller
        piece = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2), name="BasePiece")
        pid = piece.id
        outer = piece.get_outer_path()
        self.assertIsNotNone(outer)

        # ノッチと配置を追加
        nid = "notch_c"
        ctrl.doc.notches[nid] = Notch(
            id=nid,
            piece_id=pid,
            path_id=outer.id,
            segment_start_id=outer.nodes[0].id,
            t=0.5,
        )
        ctrl.doc.placements[pid] = Placement(
            piece_id=pid,
            target_bone="Spine",
            preset="CUSTOM",
        )

        # ピースを選択して Ctrl+C
        ctrl.selection.clear()
        ctrl.selection.select_piece(pid)
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, key="C")))
        self.assertEqual(len(ctrl.piece_clipboard), 1)

        # Ctrl+V でペースト（元ピースが存在するケース: duplicate_piece経由）
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, key="V")))
        self.assertEqual(len(ctrl.doc.pieces), 2)
        pasted_pid = next(p for p in ctrl.doc.pieces if p != pid)
        pasted_piece = ctrl.doc.pieces[pasted_pid]
        self.assertEqual(pasted_piece.name, "BasePiece_Copy")
        self.assertEqual(len(ctrl.doc.notches), 2)
        self.assertIn(pasted_pid, ctrl.doc.placements)

        # 元ピースとペーストピースを選択して削除
        ctrl.selection.clear()
        ctrl.selection.select_piece(pid, add=True)
        ctrl.selection.select_piece(pasted_pid, add=True)
        self.assertTrue(ctrl.delete_selected())
        self.assertEqual(len(ctrl.doc.pieces), 0)

        # 再度 Ctrl+V でペースト（元ピースが削除されているケース: スナップショット復元経由）
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, key="V")))
        self.assertEqual(len(ctrl.doc.pieces), 1)
        restored_piece = next(iter(ctrl.doc.pieces.values()))
        self.assertEqual(restored_piece.name, "BasePiece_Copy")
        self.assertEqual(len(ctrl.doc.notches), 1)
        self.assertEqual(len(ctrl.doc.placements), 1)

    def test_contextual_edge_copy_paste_with_ctrl_c_and_v(self):
        """Shiftなしの Ctrl+C / Ctrl+V による文脈連動の辺形状コピー＆ペーストの検証."""
        ctrl = self.controller
        p_src = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2), name="Src")
        p_dst = ctrl.doc.add_rectangle_piece((0.3, 0.0), (0.5, 0.2), name="Dst")
        out_src = p_src.get_outer_path()
        out_dst = p_dst.get_outer_path()

        # ソースの下辺をベジェ曲線にする
        out_src.nodes[0].out_handle = (0.05, 0.07)
        out_src.nodes[1].in_handle = (-0.05, 0.07)
        er_src = EdgeRange(piece_id=p_src.id, path_id=out_src.id, start_node_id=out_src.nodes[0].id, end_node_id=out_src.nodes[1].id)
        src_len = ctrl.doc.compute_edge_range_length(er_src)

        # 辺を選択して Ctrl+C (Shiftなし)
        ctrl.selection.select_edge(er_src)
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, key="C")))
        self.assertIsNotNone(ctrl.edge_clipboard)
        self.assertAlmostEqual(ctrl.edge_clipboard["length"], src_len, places=4)

        # ターゲットの辺を選択して Ctrl+V (Shiftなし)
        er_dst = EdgeRange(piece_id=p_dst.id, path_id=out_dst.id, start_node_id=out_dst.nodes[0].id, end_node_id=out_dst.nodes[1].id)
        ctrl.selection.clear()
        ctrl.selection.select_edge(er_dst)
        ctrl.shape_match_mode = "SIMILAR"
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, key="V")))

        # 形状がペーストされ、ハンドルが設定されていること
        self.assertIsNotNone(out_dst.nodes[0].out_handle)
        self.assertIsNotNone(out_dst.nodes[1].in_handle)

    def test_edge_selection_auto_match_seam_with_ctrl_m(self):
        """縫合線ではなく接続辺を選択した状態での Ctrl+M による自動Seam同期の検証."""
        ctrl = self.controller
        p_a = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.25, 0.2), name="A")
        p_b = ctrl.doc.add_rectangle_piece((0.0, 0.3), (0.20, 0.5), name="B")
        out_a = p_a.get_outer_path()
        out_b = p_b.get_outer_path()

        # Aの下辺（曲線）とBの下辺（直線）を結ぶ Seam
        out_a.nodes[0].out_handle = (0.06, 0.05)
        out_a.nodes[1].in_handle = (-0.06, 0.05)
        er_a = EdgeRange(piece_id=p_a.id, path_id=out_a.id, start_node_id=out_a.nodes[0].id, end_node_id=out_a.nodes[1].id)
        er_b = EdgeRange(piece_id=p_b.id, path_id=out_b.id, start_node_id=out_b.nodes[0].id, end_node_id=out_b.nodes[1].id)

        sid = "seam_auto"
        ctrl.doc.seams[sid] = Seam(id=sid, a=er_a, b=er_b, reverse=True, mode=SeamMode.SEGMENT)

        # 縫合線 (sid) ではなく、Aの下辺 (er_a) のみを選択
        ctrl.selection.clear()
        ctrl.selection.select_edge(er_a)
        self.assertEqual(len(ctrl.selection.seam_ids), 0)
        self.assertEqual(len(ctrl.selection.edge_ranges), 1)

        # Ctrl+M を押すと、接続された Seam が自動特定されて B 側へ同期される
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, key="M")))
        new_len_a, new_len_b, new_mismatch = ctrl.doc.compute_seam_lengths(sid)
        self.assertLess(new_mismatch, 0.001)
        self.assertIsNotNone(out_b.nodes[0].out_handle)
        self.assertIsNotNone(out_b.nodes[1].in_handle)

    def test_copy_paste_multi_edge_chain_armhole(self):
        """アームホールのような連続した2辺チェーンを、1辺へコピー＆ペーストして中間ノードが挿入されることの検証."""
        ctrl = self.controller
        # 身頃ピース (アームホール側): 連続した2つの曲線辺
        p_bodice = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.3, 0.4), name="Bodice")
        out_bodice = p_bodice.get_outer_path()

        # 右辺 (セグメント1: (0.3, 0.0) -> (0.3, 0.4)) の中間にノードを挿入して2辺にする
        new_mid = ctrl.doc.insert_node_on_segment(p_bodice.id, out_bodice.id, seg_idx=1, t=0.5)
        self.assertIsNotNone(new_mid)
        self.assertEqual(len(out_bodice.nodes), 5)

        # 2つのセグメントにそれぞれ特徴的なベジェハンドルを設定 (S字アームホールカーブ)
        n1 = out_bodice.nodes[1]
        n_mid = out_bodice.nodes[2]
        n3 = out_bodice.nodes[3]

        n1.out_handle = (0.04, 0.05)
        n_mid.in_handle = (-0.02, 0.03)
        n_mid.out_handle = (-0.03, 0.06)
        n3.in_handle = (0.01, -0.04)

        # 身頃側の連続2辺を選択
        er1 = EdgeRange(piece_id=p_bodice.id, path_id=out_bodice.id, start_node_id=n1.id, end_node_id=n_mid.id)
        er2 = EdgeRange(piece_id=p_bodice.id, path_id=out_bodice.id, start_node_id=n_mid.id, end_node_id=n3.id)
        ctrl.selection.clear()
        ctrl.selection.select_edge(er1)
        ctrl.selection.select_edge(er2, add=True)
        self.assertEqual(len(ctrl.selection.edge_ranges), 2)

        # 複数辺チェーンをコピー (Ctrl+C)
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, key="C")))
        self.assertIsNotNone(ctrl.edge_clipboard)
        self.assertEqual(ctrl.edge_clipboard["num_segments"], 2)

        # 袖ピース: 1つの直線辺を持つ
        p_sleeve = ctrl.doc.add_rectangle_piece((0.5, 0.0), (0.8, 0.4), name="Sleeve")
        out_sleeve = p_sleeve.get_outer_path()
        self.assertEqual(len(out_sleeve.nodes), 4)

        # 袖側の左辺 (セグメント3: (0.5, 0.4) -> (0.5, 0.0)) を選択
        er_target = EdgeRange(
            piece_id=p_sleeve.id,
            path_id=out_sleeve.id,
            start_node_id=out_sleeve.nodes[3].id,
            end_node_id=out_sleeve.nodes[0].id,
        )
        ctrl.selection.clear()
        ctrl.selection.select_edge(er_target)

        # ペースト (Ctrl+V)
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, key="V")))

        # 袖側のノード数が 4 -> 5 に増え、中間ノードが自動挿入されて2辺のベジェ曲線になっていること！
        self.assertEqual(len(out_sleeve.nodes), 5)
        # ハンドルが設定されていること
        inserted_mid = out_sleeve.nodes[4]  # 挿入された中間ノード
        self.assertIsNotNone(inserted_mid.in_handle)
        self.assertIsNotNone(inserted_mid.out_handle)

    def test_apply_endpoint_offset_transfers_chord_vector(self):
        """端点位置関係の適用 (apply_endpoint_offset = True) で終点ノードが相対ベクトル通りにテレポートすることの検証."""
        ctrl = self.controller
        # コピー元: 始点(0, 0) -> 終点(0.12, 0.25) の変位ベクトル
        p_src = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.12, 0.25), name="SrcDiag")
        out_src = p_src.get_outer_path()
        er_src = EdgeRange(piece_id=p_src.id, path_id=out_src.id, start_node_id=out_src.nodes[0].id, end_node_id=out_src.nodes[1].id)
        # 下辺の長さは0.12
        ctrl.selection.clear()
        ctrl.selection.select_edge(er_src)
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, key="C")))

        # コピー先: 始点(0.3, 0.0) -> 終点(0.5, 0.0)（元の長さは0.20）
        p_dst = ctrl.doc.add_rectangle_piece((0.3, 0.0), (0.5, 0.2), name="Dst")
        out_dst = p_dst.get_outer_path()
        er_dst = EdgeRange(piece_id=p_dst.id, path_id=out_dst.id, start_node_id=out_dst.nodes[0].id, end_node_id=out_dst.nodes[1].id)
        ctrl.selection.clear()
        ctrl.selection.select_edge(er_dst)

        # 同方向（下辺から下辺）への転写のため reverse=False
        ctrl.shape_match_reverse = False

        # Shift+O で「端点位置関係を適用」を ON にする
        self.assertFalse(ctrl.shape_match_apply_endpoint_offset)
        self.assertTrue(ctrl.handle_event(EditorEvent(key="O", shift=True)))
        self.assertTrue(ctrl.shape_match_apply_endpoint_offset)

        # ペースト
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, key="V")))

        # コピー先の終点ノード位置が、始点(0.3, 0.0) + (0.12, 0.0) = (0.42, 0.0) に更新されていること！
        n_dst_end = out_dst.nodes[1]
        self.assertAlmostEqual(n_dst_end.position[0], 0.42, places=4)
        self.assertAlmostEqual(n_dst_end.position[1], 0.0, places=4)

    def test_seam_match_multi_edge_chain_armhole_inserts_node_and_updates_seam(self):
        """Ctrl+M による縫合線同期で、複数辺チェーン（身頃アームホール2辺）から相手側1辺（袖）へ中間ノードが自動挿入され、Seam参照も更新されることの検証."""
        ctrl = self.controller
        # 身頃ピース: 右辺に中間ノードを持つ連続2辺
        p_bodice = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.3, 0.4), name="Bodice")
        out_bodice = p_bodice.get_outer_path()
        new_mid = ctrl.doc.insert_node_on_segment(p_bodice.id, out_bodice.id, seg_idx=1, t=0.5)
        self.assertIsNotNone(new_mid)

        n1 = out_bodice.nodes[1]
        n_mid = out_bodice.nodes[2]
        n3 = out_bodice.nodes[3]
        n1.out_handle = (0.04, 0.05)
        n_mid.in_handle = (-0.02, 0.03)
        n_mid.out_handle = (-0.03, 0.06)
        n3.in_handle = (0.01, -0.04)

        er_b1 = EdgeRange(piece_id=p_bodice.id, path_id=out_bodice.id, start_node_id=n1.id, end_node_id=n_mid.id)
        er_b2 = EdgeRange(piece_id=p_bodice.id, path_id=out_bodice.id, start_node_id=n_mid.id, end_node_id=n3.id)

        # 袖ピース: 左辺 (セグメント3) は1つの直線辺
        p_sleeve = ctrl.doc.add_rectangle_piece((0.5, 0.0), (0.8, 0.4), name="Sleeve")
        out_sleeve = p_sleeve.get_outer_path()
        self.assertEqual(len(out_sleeve.nodes), 4)

        er_sleeve = EdgeRange(
            piece_id=p_sleeve.id,
            path_id=out_sleeve.id,
            start_node_id=out_sleeve.nodes[3].id,
            end_node_id=out_sleeve.nodes[0].id,
        )

        # 2辺(A) と 1辺(B) を結ぶ Seam を作成
        sid = "seam_armhole"
        ctrl.doc.seams[sid] = Seam(
            id=sid,
            a=er_b1,
            b=er_sleeve,
            chain_a=[er_b1, er_b2],
            chain_b=[er_sleeve],
            reverse=True,
            mode=SeamMode.CHAIN,
        )

        # 身頃側のアームホール辺を選択して Ctrl+M
        ctrl.selection.clear()
        ctrl.selection.select_edge(er_b1)
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, key="M")))

        # 袖側のノード数が 4 -> 5 に増え、中間ノードが挿入されていること
        self.assertEqual(len(out_sleeve.nodes), 5)
        inserted_mid = out_sleeve.nodes[4]
        self.assertIsNotNone(inserted_mid.in_handle)
        self.assertIsNotNone(inserted_mid.out_handle)

        # Seam の chain_b が自動更新され、2つの EdgeRange に展開されていること！
        updated_seam = ctrl.doc.seams[sid]
        self.assertEqual(len(updated_seam.chain_b), 2)
        self.assertEqual(updated_seam.chain_b[0].start_node_id, out_sleeve.nodes[3].id)
        self.assertEqual(updated_seam.chain_b[0].end_node_id, inserted_mid.id)
        self.assertEqual(updated_seam.chain_b[1].start_node_id, inserted_mid.id)
        self.assertEqual(updated_seam.chain_b[1].end_node_id, out_sleeve.nodes[0].id)

    def test_seam_match_with_apply_endpoint_offset(self):
        """Ctrl+M による縫合線同期で、Shift+O（端点位置関係適用）が有効な場合に相手側終点ノードが相対ベクトル通りに移動することの検証."""
        ctrl = self.controller
        # A: 始点(0, 0) -> 終点(0.12, 0.0)（長さ0.12）
        p_a = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.12, 0.2), name="A")
        out_a = p_a.get_outer_path()
        er_a = EdgeRange(piece_id=p_a.id, path_id=out_a.id, start_node_id=out_a.nodes[0].id, end_node_id=out_a.nodes[1].id)

        # B: 始点(0.3, 0.0) -> 終点(0.5, 0.0)（長さ0.20）
        p_b = ctrl.doc.add_rectangle_piece((0.3, 0.0), (0.5, 0.2), name="B")
        out_b = p_b.get_outer_path()
        er_b = EdgeRange(piece_id=p_b.id, path_id=out_b.id, start_node_id=out_b.nodes[0].id, end_node_id=out_b.nodes[1].id)

        sid = "seam_offset"
        # 下辺同士で同じ向き（左から右）のため reverse=False
        ctrl.doc.seams[sid] = Seam(id=sid, a=er_a, b=er_b, reverse=False, mode=SeamMode.SEGMENT)

        # Shift+O で「端点位置関係を適用」を ON
        ctrl.shape_match_apply_endpoint_offset = True

        # Aを選択して Ctrl+M
        ctrl.selection.clear()
        ctrl.selection.select_edge(er_a)
        self.assertTrue(ctrl.handle_event(EditorEvent(ctrl=True, key="M")))

        # Bの終点ノード位置が、始点(0.3, 0.0) + (0.12, 0.0) = (0.42, 0.0) にテレポート移動していること！
        n_b_end = out_b.nodes[1]
        self.assertAlmostEqual(n_b_end.position[0], 0.42, places=4)
        self.assertAlmostEqual(n_b_end.position[1], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
