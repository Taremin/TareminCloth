# -*- coding: utf-8 -*-
"""Phase 3.2 新規追加機能・不具合修正の単体テスト（Blender 非起動スタンドアロン）."""

from __future__ import annotations

import math
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from taremin_cloth.ops import pattern as pattern_ops
from taremin_cloth.ops.pattern import (
    TAREMIN_OT_close_pattern_editor,
    TAREMIN_OT_pattern_editor,
)
from taremin_cloth.pattern.bridge import load_document_from_scene, save_document_to_scene
from taremin_cloth.pattern.build_types import (
    build_garment_mesh_geometry,
    transfer_vertex_weights_by_uv,
)
from taremin_cloth.pattern.canvas import PatternCanvasRenderer
from taremin_cloth.pattern.cards import format_selection_card_lines
from taremin_cloth.pattern.document import (
    Document,
    EdgeRange,
    Node,
    NodeKind,
    Notch,
    NotchKind,
    Path,
    PathRole,
    Placement,
)
from taremin_cloth.pattern.editor import EditorController, EditorEvent
from taremin_cloth.pattern.geometry import point_in_polygon
from taremin_cloth.pattern.layout import compute_hud_layout
from taremin_cloth.pattern.placement import get_quick_placement_transform
from taremin_cloth.utils.quad_remesh_prep import _pick_loop_interior_probe


def _center(rect) -> tuple[float, float]:
    """Rect2D の中心座標を返す."""
    return (rect.x + rect.w * 0.5, rect.y + rect.h * 0.5)


def _make_fake_op_context(*, has_existing_garment: bool = False):
    """TAREMIN_OT_pattern_editor 検証用の最小限コンテキストを構築する."""
    objects_dict = {"Garment_Cloth": object()} if has_existing_garment else {}
    scene = SimpleNamespace(
        name="Phase32Scene",
        taremin_pattern_editor_active=False,
        taremin_pattern_document="",
        objects=objects_dict,
    )
    rv3d = SimpleNamespace(
        view_perspective="PERSP",
        view_rotation=MagicMock(),
        view_location=MagicMock(),
        view_distance=5.0,
    )
    space = SimpleNamespace(
        type="VIEW_3D",
        show_region_toolbar=True,
        show_region_tool_header=True,
        region_3d=rv3d,
    )
    region = SimpleNamespace(width=1280, height=720)
    area = SimpleNamespace(type="VIEW_3D", spaces=SimpleNamespace(active=space), regions=[])
    wm = SimpleNamespace(modal_handler_add=MagicMock())
    return SimpleNamespace(
        scene=scene,
        area=area,
        space_data=space,
        region=region,
        region_data=rv3d,
        window_manager=wm,
    )


def _make_raw_event(*, event_type: str = "MOUSEMOVE", value: str = "NOTHING"):
    return SimpleNamespace(
        type=event_type,
        value=value,
        mouse_region_x=100,
        mouse_region_y=200,
        mouse_x=100,
        mouse_y=200,
        shift=False,
        ctrl=False,
        alt=False,
        unicode="",
    )


class TestPatternPhase32(unittest.TestCase):
    """Phase 3.2 で統合・修正された型紙ビルド・描画・エディタ機能の網羅検証."""

    def test_nonbone_build_and_canvas_centroid_alignment(self) -> None:
        """非ボーン配置時にビルド幾何重心が offset_3d と一致し、canvas の3D投影とも一致すること."""
        doc = Document.new()
        piece = doc.add_rectangle_piece((0.40, 0.20), (0.60, 0.50), name="OffsetRect")
        piece.target_edge_length = 0.05
        doc.placements[piece.id] = Placement(
            piece_id=piece.id,
            target_bone="",
            offset_3d=(0.15, -0.25, 1.10),
            euler_3d=(0.0, 0.0, 0.0),
        )

        geom = build_garment_mesh_geometry(doc, bones=None)
        self.assertGreater(len(geom.vertices_3d), 0)
        verts_arr = np.asarray(geom.vertices_3d, dtype=np.float64)
        mean_3d = np.mean(verts_arr, axis=0)
        np.testing.assert_allclose(mean_3d, [0.15, -0.25, 1.10], atol=0.03)

        # canvas 側の投影も同一ピース重心 (0.50, 0.35) を差し引いて offset_3d に一致すること
        ctrl = EditorController(doc=doc)
        canvas = PatternCanvasRenderer(ctrl)
        pt_3d = canvas._project_world2d_point_to_3d(
            piece.id,
            (0.50, 0.35),
            bone_map={},
            centroid_cache={},
        )
        np.testing.assert_allclose(pt_3d, [0.15, -0.25, 1.10], atol=1e-4)

    def test_hole_triangle_removal_and_vertex_compaction(self) -> None:
        """閉じた HOLE パス内の三角形除去後、孤立頂点がコンパクションで排除されること."""
        doc = Document.new()
        piece = doc.add_rectangle_piece((0.0, 0.0), (0.40, 0.40), name="HolePanel")
        piece.target_edge_length = 0.04

        hole_nodes = [
            Node(id="h0", position=(0.12, 0.12), kind=NodeKind.CORNER),
            Node(id="h1", position=(0.28, 0.12), kind=NodeKind.CORNER),
            Node(id="h2", position=(0.28, 0.28), kind=NodeKind.CORNER),
            Node(id="h3", position=(0.12, 0.28), kind=NodeKind.CORNER),
        ]
        piece.paths["hole_1"] = Path(
            id="hole_1",
            role=PathRole.HOLE,
            nodes=hole_nodes,
            closed=True,
        )

        geom = build_garment_mesh_geometry(doc)
        n_verts = len(geom.vertices_3d)
        self.assertGreater(n_verts, 0)
        self.assertGreater(len(geom.faces), 0)

        # 穴中央付近の三角形が除去されていること
        for a, b, c in geom.faces:
            va = geom.vertices_3d[a]
            vb = geom.vertices_3d[b]
            vc = geom.vertices_3d[c]
            # (X, Z) が元の (x, y) に対応（未設定 placement）
            cx = (va[0] + vb[0] + vc[0]) / 3.0
            cy = (va[2] + vb[2] + vc[2]) / 3.0
            # build は重心 (-0.2, -0.2) を引くため穴中心は (0.0, 0.0)
            self.assertFalse(abs(cx) < 0.04 and abs(cy) < 0.04)

        # 全頂点が少なくとも1つの有効三角形から参照され、孤立頂点が 0 であること
        referenced = {vi for tri in geom.faces for vi in tri}
        self.assertEqual(len(referenced), n_verts)
        self.assertEqual(set(geom.piece_vertex_map["HolePanel"]), set(range(n_verts)))

    def test_transfer_vertex_weights_by_uv(self) -> None:
        """UV重心補間および範囲外最近傍フォールバックで頂点ウェイトが正しく転写されること."""
        old_tri_uvs = np.array(
            [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        old_tri_weights = {
            "Pin": np.array([[0.0, 0.5, 1.0], [0.0, 1.0, 0.5]], dtype=np.float32),
        }
        new_uvs = np.array(
            [
                [0.5, 0.5],     # 中心 -> 0.5
                [0.0, 0.0],     # 左下隅 -> 0.0
                [1.0, 1.0],     # 右上隅 -> 1.0
                [-0.2, -0.2],   # 範囲外 -> 最近傍 (0,0) の 0.0
                [1.3, 1.3],     # 範囲外 -> 最近傍 (1,1) の 1.0
            ],
            dtype=np.float32,
        )
        res = transfer_vertex_weights_by_uv(
            old_tri_uvs=old_tri_uvs,
            old_tri_weights=old_tri_weights,
            new_uvs=new_uvs,
        )
        np.testing.assert_allclose(res["Pin"], [0.5, 0.0, 1.0, 0.0, 1.0], atol=1e-4)

    def test_pick_loop_interior_probe_concave_polygon(self) -> None:
        """頂点重心が外部に落ちる深いコの字型凹ポリゴンでも内部プローブ点が厳密内部を返すこと."""
        concave_u = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.8, 1.0],
                [0.8, 0.2],
                [0.2, 0.2],
                [0.2, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        probe = _pick_loop_interior_probe(concave_u)
        poly_list = [(float(p[0]), float(p[1])) for p in concave_u]
        self.assertTrue(point_in_polygon(probe, poly_list))
        # 境界頂点そのものではないこと
        for vx, vy in poly_list:
            self.assertGreater(math.hypot(probe[0] - vx, probe[1] - vy), 1e-4)

    def test_corrupted_json_protection_guard(self) -> None:
        """破損 JSON 読込時に _load_failed が付与され、通常保存での上書き破壊を防ぐこと."""
        corrupted_payload = '{"version": 1, "pieces": BROKEN_DATA'
        scene = SimpleNamespace(taremin_pattern_document=corrupted_payload)

        doc = load_document_from_scene(scene)
        self.assertTrue(getattr(doc, "_load_failed", False))
        self.assertEqual(len(doc.pieces), 0)

        # 通常保存は保護ガードによりスキップされ元データが守られる
        save_document_to_scene(doc, scene=scene, push_undo=False)
        self.assertEqual(scene.taremin_pattern_document, corrupted_payload)

        # force_overwrite=True 指定時のみ明示的に上書き可能
        save_document_to_scene(doc, scene=scene, push_undo=False, force_overwrite=True)
        self.assertNotEqual(scene.taremin_pattern_document, corrupted_payload)

    def test_internal_closed_path_and_hole_toggle(self) -> None:
        """InternalLineTool で始点クリック閉パスを作成でき、H キーおよびカードクリックで HOLE 切替できること."""
        doc = Document.new()
        piece = doc.add_rectangle_piece((0.0, 0.0), (0.5, 0.5), name="Body")
        ctrl = EditorController(doc=doc)
        ctrl.set_active_tool("INTERNAL_LINE")

        pts = [(0.15, 0.15), (0.35, 0.15), (0.25, 0.35)]
        for wp in pts:
            sp = ctrl.view.world_to_screen(wp)
            ctrl.handle_event(EditorEvent(screen_pos=sp, world_pos=wp, is_press=True, button="LEFT"))

        # 始点近く (< 1cm) をクリックして閉パス確定
        close_wp = (0.153, 0.152)
        close_sp = ctrl.view.world_to_screen(close_wp)
        ctrl.handle_event(EditorEvent(screen_pos=close_sp, world_pos=close_wp, is_press=True, button="LEFT"))

        internal_paths = [
            pid for pid, path in piece.paths.items() if pid != piece.outer_path_id
        ]
        self.assertEqual(len(internal_paths), 1)
        ipath = piece.paths[internal_paths[0]]
        self.assertEqual(ipath.role, PathRole.INTERNAL)
        self.assertTrue(ipath.closed)
        self.assertEqual(len(ipath.nodes), 3)

        # 辺を選択して H キーで HOLE へ切替
        ctrl.set_active_tool("SELECT")
        er = EdgeRange(
            piece_id=piece.id,
            path_id=ipath.id,
            start_node_id=ipath.nodes[0].id,
            end_node_id=ipath.nodes[1].id,
        )
        ctrl.selection.select_edge(er)

        self.assertTrue(ctrl.handle_event(EditorEvent(key="H", is_press=True)))
        self.assertEqual(ipath.role, PathRole.HOLE)

        # カード表示テキストに Hole: ON (H) が出力され、クリックで INTERNAL に戻ること
        lines = format_selection_card_lines(ctrl.doc, ctrl.selection)
        self.assertTrue(any("Hole: ON (H)" in ln for ln in lines))

        layout = compute_hud_layout(*ctrl.screen_size)
        hx, hy = _center(layout.card_fields["edge_toggle_hole"])
        self.assertTrue(ctrl.handle_event(EditorEvent(screen_pos=(hx, hy), is_press=True, button="LEFT")))
        ipath_after = ctrl.doc.pieces[piece.id].paths[ipath.id]
        self.assertEqual(ipath_after.role, PathRole.INTERNAL)

        # Undo で HOLE 状態に戻り選択が保持されていること
        self.assertTrue(ctrl.undo())
        self.assertEqual(ctrl.doc.pieces[piece.id].paths[ipath.id].role, PathRole.HOLE)
        self.assertEqual(len(ctrl.selection.edge_ranges), 1)

    def test_cut_tool_inherits_paths_avoids_name_collision_and_remaps_hit1_notch(self) -> None:
        """CutTool が _unique_piece_name で衝突回避し内部線/穴を継承し hit1 後半ノッチ ID を正しく付け替えること."""
        doc = Document.new()
        # あえて衝突先名 "Panel_A" を事前配置
        doc.add_rectangle_piece((1.0, 0.0), (1.2, 0.2), name="Panel_A")

        piece = doc.add_rectangle_piece((0.0, 0.0), (0.40, 0.40), name="Panel")
        outer = piece.paths[piece.outer_path_id]

        # 水平カット y=0.20 と右辺(seg_idx=1: (0.4,0)->(0.4,0.4)) で交差するため、後半 t=0.75 (y=0.30) にノッチを置く
        doc.notches["n_upper"] = Notch(
            id="n_upper",
            piece_id=piece.id,
            path_id=piece.outer_path_id,
            segment_start_id=outer.nodes[1].id,
            t=0.75,
            kind=NotchKind.SLIT,
        )
        # 下側 (y in [0.05, 0.12]) に HOLE パス、上側 (y=0.32) に INTERNAL パスを追加
        hole_nodes = [
            Node(id="hn0", position=(0.10, 0.05)),
            Node(id="hn1", position=(0.30, 0.05)),
            Node(id="hn2", position=(0.30, 0.12)),
            Node(id="hn3", position=(0.10, 0.12)),
        ]
        piece.paths["hole_sub"] = Path(id="hole_sub", role=PathRole.HOLE, nodes=hole_nodes, closed=True)
        int_nodes = [
            Node(id="in0", position=(0.10, 0.32)),
            Node(id="in1", position=(0.30, 0.32)),
        ]
        piece.paths["int_sub"] = Path(id="int_sub", role=PathRole.INTERNAL, nodes=int_nodes, closed=False)

        ctrl = EditorController(doc=doc)
        ctrl.set_active_tool("CUT")

        # 右 -> 左 への水平カットライン y=0.20 (hit0=右辺 seg 1 t=0.5, hit1=左辺 seg 3 t=0.5 / または左->右で hit1=右辺)
        # 左->右: (-0.05, 0.20) -> (0.45, 0.20) なので 左辺(seg 3) が hit0, 右辺(seg 1) が hit1 になる
        p0 = (-0.05, 0.20)
        p1 = (0.45, 0.20)
        ctrl.handle_event(EditorEvent(
            screen_pos=ctrl.view.world_to_screen(p0),
            world_pos=p0,
            is_press=True,
            button="LEFT",
        ))
        self.assertTrue(
            ctrl.handle_event(EditorEvent(
                screen_pos=ctrl.view.world_to_screen(p1),
                world_pos=p1,
                is_release=True,
                button="LEFT",
            ))
        )

        # 衝突しない一意名（既存 Panel_A + 新規2ピースで計3ピース、名前はすべて重複なし）
        names = [p.name for p in ctrl.doc.pieces.values()]
        self.assertEqual(len(ctrl.doc.pieces), 3)
        self.assertEqual(len(set(names)), 3)

        # 内部線/穴が上下それぞれのピースに継承されていること
        sub_pieces = [p for p in ctrl.doc.pieces.values() if p.name != "Panel_A"]
        inherited_roles = [
            path.role
            for sp in sub_pieces
            for pid, path in sp.paths.items()
            if pid != sp.outer_path_id
        ]
        self.assertIn(PathRole.HOLE, inherited_roles)
        self.assertIn(PathRole.INTERNAL, inherited_roles)

        # hit1 (右辺) 上後半のノッチがリマップされた先で実際に存在・解決可能なノードIDを参照していること
        remapped_notch = ctrl.doc.notches.get("n_upper")
        self.assertIsNotNone(remapped_notch)
        target_piece = ctrl.doc.pieces[remapped_notch.piece_id]
        target_path = target_piece.paths[remapped_notch.path_id]
        target_node_ids = {nd.id for nd in target_path.nodes}
        self.assertIn(remapped_notch.segment_start_id, target_node_ids)
        # 元の outer.nodes[1].id の座標 (0.40, 0.0) ではなくカット点 (0.40, 0.20) に付け替わっていること
        seg_start = next(nd for nd in target_path.nodes if nd.id == remapped_notch.segment_start_id)
        self.assertAlmostEqual(seg_start.position[0], 0.40, places=4)
        self.assertAlmostEqual(seg_start.position[1], 0.20, places=4)
        self.assertAlmostEqual(remapped_notch.t, 0.5, places=3)

    def test_nonbone_card_editing_right_click_cancel_and_undo_selection(self) -> None:
        """非ボーン配置カード操作・右クリックビジーツールキャンセル・Undo選択保持が機能すること."""
        doc = Document.new()
        p1 = doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2), name="Front")
        ctrl = EditorController(doc=doc)
        ctrl.selection.select_piece(p1.id)
        layout = compute_hud_layout(*ctrl.screen_size)

        # 1. カードテキストに非ボーン7行が出力されること
        lines = format_selection_card_lines(ctrl.doc, ctrl.selection)
        self.assertTrue(any("3D Preset" in ln for ln in lines))
        self.assertTrue(any("3D Pos X" in ln for ln in lines))
        self.assertTrue(any("3D Rot Z" in ln for ln in lines))

        # 2. placement_preset クリックで CHEST_FRONT プリセットが適用されること
        px, py = _center(layout.card_fields["placement_preset"])
        self.assertTrue(ctrl.handle_event(EditorEvent(screen_pos=(px, py), is_press=True, button="LEFT")))
        exp_off, exp_euler = get_quick_placement_transform("CHEST_FRONT")
        self.assertEqual(ctrl.doc.placements[p1.id].offset_3d, exp_off)
        self.assertEqual(ctrl.doc.placements[p1.id].euler_3d, exp_euler)

        # 3. offset_3d_x をクリック→リリースでインライン入力開始し "25" (cm=0.25m) で確定
        ox, oy = _center(layout.card_fields["offset_3d_x"])
        ctrl.handle_event(EditorEvent(screen_pos=(ox, oy), is_press=True, button="LEFT"))
        ctrl.handle_event(EditorEvent(screen_pos=(ox, oy), is_release=True, button="LEFT"))
        self.assertEqual(ctrl.editing_field, "offset_3d_x")
        ctrl.handle_event(EditorEvent(key="BACK_SPACE", is_press=True))
        ctrl.handle_event(EditorEvent(key="BACK_SPACE", is_press=True))
        ctrl.handle_event(EditorEvent(key="BACK_SPACE", is_press=True))
        ctrl.handle_event(EditorEvent(key="BACK_SPACE", is_press=True))
        ctrl.handle_event(EditorEvent(key="2", unicode="2", is_press=True))
        ctrl.handle_event(EditorEvent(key="5", unicode="5", is_press=True))
        ctrl.handle_event(EditorEvent(key="ENTER", is_press=True))
        self.assertAlmostEqual(ctrl.doc.placements[p1.id].offset_3d[0], 0.25, places=4)

        # 4. Undo 後も p1 の選択状態が維持されること (prune_missing)
        self.assertTrue(ctrl.undo())
        self.assertEqual(ctrl.doc.placements[p1.id].offset_3d[0], exp_off[0])
        self.assertIn(p1.id, ctrl.selection.piece_ids)

        # 5. 右クリックでビジー状態の InternalLineTool がキャンセルされること
        ctrl.set_active_tool("INTERNAL_LINE")
        pt = (0.05, 0.05)
        ctrl.handle_event(EditorEvent(
            screen_pos=ctrl.view.world_to_screen(pt),
            world_pos=pt,
            is_press=True,
            button="LEFT",
        ))
        self.assertTrue(ctrl._is_active_tool_busy())
        self.assertTrue(ctrl.handle_event(EditorEvent(screen_pos=(500.0, 400.0), is_press=True, button="RIGHT")))
        self.assertFalse(ctrl._is_active_tool_busy())

    def test_editor_skip_unchanged_rebuild_and_save_on_dirty_cancel(self) -> None:
        """既存メッシュあり・未変更時の通常終了は再ビルドを回避し、編集ありキャンセル時はドキュメントのみ保存すること."""
        saved_states: dict[str, object] = {"op": pattern_ops._active_pattern_editor_op, "canvas": pattern_ops._active_pattern_editor_canvas}
        pattern_ops._active_pattern_editor_op = None
        pattern_ops._active_pattern_editor_canvas = None
        try:
            # A. 既存 Garment_Cloth あり ＋ 未変更終了 -> 再ビルドスキップ
            ctx = _make_fake_op_context(has_existing_garment=True)
            op_self = SimpleNamespace(report=MagicMock(), _modal_impl=MagicMock())
            canvas_mock = MagicMock()
            initial_doc = Document.new()
            initial_doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2), name="Front")

            with patch.object(pattern_ops, "is_interactive_active", return_value=False), \
                    patch.object(pattern_ops, "stop_animation"), \
                    patch.object(pattern_ops, "load_document_from_scene", return_value=initial_doc), \
                    patch.object(pattern_ops, "_find_target_armature", return_value=None), \
                    patch.object(pattern_ops, "extract_armature_bones_for_canvas", return_value={}), \
                    patch.object(pattern_ops, "sync_view2d_from_region_3d"), \
                    patch.object(pattern_ops, "get_viewport_overlay_offsets", return_value=(0.0, 0.0, 0.0, 0.0)), \
                    patch.object(pattern_ops, "PatternCanvasRenderer", return_value=canvas_mock), \
                    patch.object(pattern_ops, "tag_redraw_view3d"):
                res = TAREMIN_OT_pattern_editor.invoke(op_self, ctx, _make_raw_event())
            self.assertEqual(res, {'RUNNING_MODAL'})

            close_op = SimpleNamespace(report=MagicMock())
            with patch.object(pattern_ops, "save_document_to_scene") as mock_save, \
                    patch.object(pattern_ops, "build_garment_mesh_in_place") as mock_build, \
                    patch.object(pattern_ops, "tag_redraw_view3d"):
                TAREMIN_OT_close_pattern_editor.execute(close_op, ctx)
            mock_save.assert_called_once()
            mock_build.assert_not_called()

            # B. 編集ありキャンセル -> save_document_to_scene のみ実行され build は走らないこと
            ctx2 = _make_fake_op_context(has_existing_garment=True)
            op_self2 = SimpleNamespace(
                report=MagicMock(),
                _modal_impl=lambda c, e: TAREMIN_OT_pattern_editor._modal_impl(op_self2, c, e),
                _finish=lambda c: TAREMIN_OT_pattern_editor._finish(op_self2, c),
            )
            doc2 = Document.new()
            with patch.object(pattern_ops, "is_interactive_active", return_value=False), \
                    patch.object(pattern_ops, "stop_animation"), \
                    patch.object(pattern_ops, "load_document_from_scene", return_value=doc2), \
                    patch.object(pattern_ops, "_find_target_armature", return_value=None), \
                    patch.object(pattern_ops, "extract_armature_bones_for_canvas", return_value={}), \
                    patch.object(pattern_ops, "sync_view2d_from_region_3d"), \
                    patch.object(pattern_ops, "get_viewport_overlay_offsets", return_value=(0.0, 0.0, 0.0, 0.0)), \
                    patch.object(pattern_ops, "PatternCanvasRenderer", return_value=MagicMock()), \
                    patch.object(pattern_ops, "tag_redraw_view3d"):
                TAREMIN_OT_pattern_editor.invoke(op_self2, ctx2, _make_raw_event())

            # RIGHTMOUSE では終了せず RUNNING_MODAL / PASS_THROUGH になること
            op_self2.controller.handle_event = MagicMock(return_value=False)
            rmb_res = op_self2._modal_impl(ctx2, _make_raw_event(event_type="RIGHTMOUSE", value="PRESS"))
            self.assertNotEqual(rmb_res, {'FINISHED'})

            # ドキュメントへ変更を加えて cancel() すると保存だけ走る
            op_self2.controller.doc.add_rectangle_piece((0.0, 0.0), (0.1, 0.1), name="Edited")
            with patch.object(pattern_ops, "save_document_to_scene") as mock_save2, \
                    patch.object(pattern_ops, "build_garment_mesh_in_place") as mock_build2, \
                    patch.object(pattern_ops, "tag_redraw_view3d"):
                TAREMIN_OT_pattern_editor.cancel(op_self2, ctx2)
            mock_save2.assert_called_once()
            mock_build2.assert_not_called()
        finally:
            pattern_ops._active_pattern_editor_op = saved_states["op"]
            pattern_ops._active_pattern_editor_canvas = saved_states["canvas"]


if __name__ == "__main__":
    unittest.main()
