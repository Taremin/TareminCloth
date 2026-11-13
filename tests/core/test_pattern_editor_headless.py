# -*- coding: utf-8 -*-
"""型紙エディタのヘッドレスイベント列テスト (Blender非依存)."""

import unittest

from taremin_cloth.pattern.document import Document
from taremin_cloth.pattern.editor import EditorController, EditorEvent
from taremin_cloth.pattern.session import ViewTransform2D


class TestPatternEditorHeadless(unittest.TestCase):
    """EditorController に対するマウス・キーボードイベント列のヘッドレステスト."""

    def setUp(self):
        # 1m = 1000px, pan=(0, 0) -> screen(100, 200) = world(0.1, 0.2)
        view = ViewTransform2D(pan_x=0.0, pan_y=0.0, zoom=1000.0)
        self.controller = EditorController(doc=Document.new(), view=view)

    def test_draw_rectangle_tool_headless(self):
        ctrl = self.controller
        ctrl.set_active_tool("RECTANGLE")

        # 1. マウスダウン (0, 0) -> world (0.0, 0.0)
        ctrl.handle_event(EditorEvent(screen_pos=(0.0, 0.0), is_press=True))
        self.assertEqual(len(ctrl.preview_points), 2)

        # 2. マウス移動 (200, 300) -> world (0.2, 0.3)
        ctrl.handle_event(EditorEvent(screen_pos=(200.0, 300.0), is_move=True))
        self.assertEqual(ctrl.preview_points[1], (0.2, 0.3))

        # 3. マウスアップ
        ctrl.handle_event(EditorEvent(screen_pos=(200.0, 300.0), is_release=True))
        self.assertEqual(len(ctrl.doc.pieces), 1)
        self.assertEqual(len(ctrl.preview_points), 0)
        self.assertTrue(ctrl.history.can_undo)

        # 4. Undo (Ctrl+Z)
        ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        self.assertEqual(len(ctrl.doc.pieces), 0)

        # 5. Redo (Ctrl+Shift+Z)
        ctrl.handle_event(EditorEvent(ctrl=True, shift=True, key="Z"))
        self.assertEqual(len(ctrl.doc.pieces), 1)

    def test_draw_polygon_tool_headless(self):
        ctrl = self.controller
        ctrl.set_active_tool("POLYGON")

        # 3点をクリック
        ctrl.handle_event(EditorEvent(screen_pos=(0.0, 0.0), is_press=True))
        ctrl.handle_event(EditorEvent(screen_pos=(200.0, 0.0), is_press=True))
        ctrl.handle_event(EditorEvent(screen_pos=(100.0, 200.0), is_press=True))

        # Enter で閉じる
        ctrl.handle_event(EditorEvent(key="ENTER"))
        self.assertEqual(len(ctrl.doc.pieces), 1)
        poly = list(ctrl.doc.pieces.values())[0]
        outer_path = poly.paths[poly.outer_path_id]
        self.assertEqual(len(outer_path.nodes), 3)

    def test_sew_tool_headless(self):
        ctrl = self.controller
        # 2つの矩形を追加
        p1 = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.3), name="PieceA")
        p2 = ctrl.doc.add_rectangle_piece((0.5, 0.0), (0.7, 0.3), name="PieceB")

        ctrl.set_active_tool("SEW")

        # p1 の右辺 (x=0.2, y=0.15) -> screen (200, 150) をクリック
        ctrl.handle_event(EditorEvent(screen_pos=(200.0, 150.0), is_press=True))
        self.assertEqual(len(ctrl.preview_seam_candidates), 1)

        # p2 の左辺 (x=0.5, y=0.15) -> screen (500, 150) をクリック
        ctrl.handle_event(EditorEvent(screen_pos=(500.0, 150.0), is_press=True))
        self.assertEqual(len(ctrl.preview_seam_candidates), 0)
        self.assertEqual(len(ctrl.doc.seams), 1)

    def test_select_and_drag_headless(self):
        ctrl = self.controller
        p = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2))

        ctrl.set_active_tool("SELECT")

        # ピース内部 (x=0.1, y=0.1) -> screen (100, 100) をクリックしてドラッグ開始
        ctrl.handle_event(EditorEvent(screen_pos=(100.0, 100.0), is_press=True))
        self.assertTrue(ctrl.selection.is_piece_selected(p.id))

        # +0.05m (+50px) 移動
        ctrl.handle_event(EditorEvent(screen_pos=(150.0, 150.0), is_move=True))
        # マウスアップ
        ctrl.handle_event(EditorEvent(screen_pos=(150.0, 150.0), is_release=True))

        moved_piece = ctrl.doc.pieces[p.id]
        self.assertAlmostEqual(moved_piece.transform.tx, 0.05, places=3)
        self.assertAlmostEqual(moved_piece.transform.ty, 0.05, places=3)

    def test_delete_selected_headless(self):
        ctrl = self.controller
        p = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2))
        ctrl.selection.select_piece(p.id)

        # X キーで削除
        ctrl.handle_event(EditorEvent(key="X"))
        self.assertEqual(len(ctrl.doc.pieces), 0)

    def test_box_selection_and_insert_node_on_edge(self):
        """ボックス選択によるノード一括選択およびCtrl+クリックによる辺上ノード挿入."""
        ctrl = self.controller
        p = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2))
        ctrl.set_active_tool("SELECT")

        # 1. ボックス選択: (-0.05, -0.05) から (0.15, 0.25) をドラッグ -> 左側の2ノード (0,0), (0,0.2) を囲む
        # screen: (-50, -50) -> (150, 250)
        ctrl.handle_event(EditorEvent(screen_pos=(-50.0, -50.0), is_press=True))
        ctrl.handle_event(EditorEvent(screen_pos=(150.0, 250.0), is_move=True))
        ctrl.handle_event(EditorEvent(screen_pos=(150.0, 250.0), is_release=True))
        # 少なくともノードが選択されていること
        self.assertGreater(len(ctrl.selection.node_ids), 0)

        # 2. Ctrl+クリックによる辺上へのノード挿入: 底辺 (0.1, 0.0) -> screen (100, 0)
        outer = p.paths[p.outer_path_id]
        orig_node_count = len(outer.nodes)
        ctrl.handle_event(EditorEvent(screen_pos=(100.0, 0.0), is_press=True, ctrl=True))
        self.assertEqual(len(outer.nodes), orig_node_count + 1)

    def test_hud_clicks_and_inline_numeric_editing(self):
        """HUDのボタンクリック（ツール切替、ビルド/閉じる要求）およびインライン数値入力の検証."""
        from taremin_cloth.pattern.layout import compute_hud_layout
        ctrl = self.controller
        layout = compute_hud_layout(*ctrl.screen_size)

        # 1. ツールレールの「POLYGON」ボタンをクリック
        poly_btn = layout.tool_buttons["POLYGON"]
        click_x = poly_btn.x + poly_btn.w * 0.5
        click_y = poly_btn.y + poly_btn.h * 0.5
        ctrl.handle_event(EditorEvent(screen_pos=(click_x, click_y), is_press=True, button="LEFT"))
        self.assertEqual(ctrl.active_tool_name, "POLYGON")

        # 2. Build 3D ボタンをクリック
        build_btn = layout.build_button
        ctrl.handle_event(EditorEvent(
            screen_pos=(build_btn.x + build_btn.w * 0.5, build_btn.y + build_btn.h * 0.5),
            is_press=True,
            button="LEFT",
        ))
        self.assertTrue(ctrl.request_build)

        # 3. Close ボタンをクリック
        close_btn = layout.close_button
        ctrl.handle_event(EditorEvent(
            screen_pos=(close_btn.x + close_btn.w * 0.5, close_btn.y + close_btn.h * 0.5),
            is_press=True,
            button="LEFT",
        ))
        self.assertTrue(ctrl.request_close)

        # 4. ピースを追加し選択状態にしてインライン数値入力をテスト
        p = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2))
        ctrl.selection.select_piece(p.id)

        target_field = layout.card_fields["target_edge_length"]
        # 解像度行の描画ベースライン (info_card.top - 56.0) が target_field の領域内に存在することを検証
        res_baseline = layout.info_card.top - 56.0
        self.assertTrue(target_field.y <= res_baseline <= target_field.top)

        ctrl.handle_event(EditorEvent(
            screen_pos=(target_field.x + target_field.w * 0.5, target_field.y + target_field.h * 0.5),
            is_press=True,
            button="LEFT",
        ))
        self.assertEqual(ctrl.editing_field, "target_edge_length")

        # バッファを一度クリアして "2.5" を入力 (2.5cm = 0.025m)
        ctrl.editing_buffer = ""
        for ch in ["2", ".", "5"]:
            ctrl.handle_event(EditorEvent(key=ch))
        self.assertEqual(ctrl.editing_buffer, "2.5")

        # Enter で確定
        ctrl.handle_event(EditorEvent(key="ENTER"))
        self.assertIsNone(ctrl.editing_field)
        self.assertAlmostEqual(p.target_edge_length, 0.025, places=4)
        # 履歴スタックに記録されていること
        self.assertGreater(len(ctrl.history.undo_stack), 0)

        # 5. キーボード上段の数字キー名（"THREE", "PERIOD", "ZERO"）による入力をテスト (3.0cm = 0.030m)
        ctrl.handle_event(EditorEvent(
            screen_pos=(target_field.x + target_field.w * 0.5, target_field.y + target_field.h * 0.5),
            is_press=True,
            button="LEFT",
        ))
        self.assertEqual(ctrl.editing_field, "target_edge_length")
        ctrl.editing_buffer = ""
        for key_name in ["THREE", "PERIOD", "ZERO"]:
            ctrl.handle_event(EditorEvent(key=key_name))
        self.assertEqual(ctrl.editing_buffer, "3.0")
        ctrl.handle_event(EditorEvent(key="NUMPAD_ENTER"))
        self.assertAlmostEqual(p.target_edge_length, 0.030, places=4)

        # 6. event.unicode による入力をテスト (1.8cm = 0.018m)
        ctrl.handle_event(EditorEvent(
            screen_pos=(target_field.x + target_field.w * 0.5, target_field.y + target_field.h * 0.5),
            is_press=True,
            button="LEFT",
        ))
        ctrl.editing_buffer = ""
        for uni in ["1", ".", "8"]:
            ctrl.handle_event(EditorEvent(unicode=uni))
        self.assertEqual(ctrl.editing_buffer, "1.8")
        ctrl.handle_event(EditorEvent(key="RET"))
        self.assertAlmostEqual(p.target_edge_length, 0.018, places=4)

    def test_hud_offsets_for_toolbar_and_sidebar(self):
        """Tパネル（Toolbar）、Nパネル（Sidebar）、ヘッダー/ツールヘッダーの4方向オフセット存在時のHUD再配置とクリック判定検証."""
        from taremin_cloth.pattern.layout import compute_hud_layout
        ctrl = self.controller
        ctrl.left_offset = 64.0
        ctrl.right_offset = 280.0
        ctrl.top_offset = 52.0
        ctrl.bottom_offset = 20.0

        layout = compute_hud_layout(
            *ctrl.screen_size,
            left_offset=ctrl.left_offset,
            right_offset=ctrl.right_offset,
            top_offset=ctrl.top_offset,
            bottom_offset=ctrl.bottom_offset,
        )

        # ツールレールが left_offset 分右にシフトしていること
        self.assertAlmostEqual(layout.left_rail.x, 12.0 + 64.0)
        # 情報カードが right_offset 分左にシフトしていること
        self.assertLessEqual(layout.info_card.right, ctrl.screen_size[0] - 12.0 - 280.0 + 1e-4)
        # ヘッダーバーが top_offset 分下にシフトしていること
        self.assertLessEqual(layout.header_bar.top, ctrl.screen_size[1] - 12.0 - 52.0 + 1e-4)
        # フッターバーが bottom_offset 分上にシフトしていること
        self.assertGreaterEqual(layout.footer_bar.y, 12.0 + 20.0 - 1e-4)
        # ツールレールの開始位置がヘッダーバーの下になっていること
        self.assertLessEqual(layout.left_rail.top, layout.header_bar.y)

        # オフセット後のツールボタン（CUT）をクリックしてツールが切り替わること
        cut_btn = layout.tool_buttons["CUT"]
        ctrl.handle_event(EditorEvent(
            screen_pos=(cut_btn.x + cut_btn.w * 0.5, cut_btn.y + cut_btn.h * 0.5),
            is_press=True,
            button="LEFT",
        ))
        self.assertEqual(ctrl.active_tool_name, "CUT")

    def test_seam_flip_direction_and_endpoints_rendering(self):
        """縫合の両端線描画、Rキーによる縫合反転（reverseトグル）、ヒットテスト選択、Xキー削除の検証."""
        from taremin_cloth.pattern.document import EdgeRange, Seam
        from taremin_cloth.pattern.renderer import build_render_commands
        from taremin_cloth.pattern.canvas import DrawCommandKind
        from taremin_cloth.pattern.cards import format_selection_card_lines

        ctrl = self.controller
        ctrl.doc = Document.new()
        p_a = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2), name="PieceA")
        p_b = ctrl.doc.add_rectangle_piece((0.3, 0.0), (0.5, 0.2), name="PieceB")

        path_a = p_a.paths[p_a.outer_path_id]
        path_b = p_b.paths[p_b.outer_path_id]

        # 縫合を追加 (PieceAの右辺とPieceBの左辺)
        seam = Seam(
            id="test_seam_1",
            a=EdgeRange(piece_id=p_a.id, path_id=path_a.id, start_node_id=path_a.nodes[1].id, end_node_id=path_a.nodes[2].id),
            b=EdgeRange(piece_id=p_b.id, path_id=path_b.id, start_node_id=path_b.nodes[0].id, end_node_id=path_b.nodes[3].id),
            reverse=False,
        )
        ctrl.doc.seams[seam.id] = seam
        ctrl.selection.select_seam(seam.id)

        # 1. カード情報に縫合方向と F キー案内が表示されること
        card_lines = format_selection_card_lines(ctrl.doc, ctrl.selection)
        self.assertTrue(any("Normal" in line or "正方向" in line for line in card_lines))
        self.assertTrue(any("Press F to flip" in line or "Fキーで反転" in line for line in card_lines))

        # 2. 描画コマンド生成: 両端を結ぶ2本の主線と中央の薄い補助線、始点マーカー2個が生成されること
        cmds = build_render_commands(ctrl.doc, ctrl, screen_size=(800, 600))
        seam_lines = [c for c in cmds if c.kind == DrawCommandKind.LINE and len(c.points) == 2 and c.points[0][0] != c.points[1][0]]
        # 始点ペア、終点ペア、中央補助線
        self.assertGreaterEqual(len(seam_lines), 3)

        # 3. F キー押下で縫合方向が反転すること (reverse: False -> True)
        handled = ctrl.handle_event(EditorEvent(key="F"))
        self.assertTrue(handled)
        self.assertTrue(seam.reverse)

        # カード情報が "Reversed" / "反転" に更新されていること
        card_lines_rev = format_selection_card_lines(ctrl.doc, ctrl.selection)
        self.assertTrue(any("Reversed" in line or "反転" in line for line in card_lines_rev))

        # 再度 F キーでトグル (reverse: True -> False)
        ctrl.handle_event(EditorEvent(key="F"))
        self.assertFalse(ctrl.doc.seams[seam.id].reverse)

        # もう一度 F を押すと反転に戻ること (reverse: False -> True)
        ctrl.handle_event(EditorEvent(key="F"))
        self.assertTrue(ctrl.doc.seams[seam.id].reverse)

        # Undo (Ctrl+Z) で反転が元に戻ること
        ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        self.assertFalse(ctrl.doc.seams[seam.id].reverse)

        # 4. 縫合線をクリックして選択できること (ヒットテスト)
        ctrl.selection.clear()
        self.assertEqual(len(ctrl.selection.seam_ids), 0)
        # 縫合線の中点付近をクリック
        w_mid = (0.25, 0.1)
        s_mid = ctrl.view.world_to_screen(w_mid)
        ctrl.set_active_tool("SELECT")
        ctrl.handle_event(EditorEvent(
            screen_pos=s_mid,
            is_press=True,
            button="LEFT",
        ))
        self.assertIn(seam.id, ctrl.selection.seam_ids)

        # 5. X キーで選択中の縫合を削除できること
        ctrl.handle_event(EditorEvent(key="X"))
        self.assertNotIn(seam.id, ctrl.doc.seams)
        self.assertEqual(len(ctrl.selection.seam_ids), 0)

        # ピース本体は削除されずに残っていること
        self.assertEqual(len(ctrl.doc.pieces), 2)

    def test_bone_placement_gui_and_inline_edit_headless(self):
        """HUD情報カードおよび角度コンパスウィジェットからのボーン配置操作検証."""
        ctrl = self.controller
        p = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.3), name="Panel")
        ctrl.selection.select_piece(p.id)

        # ダミーのボーン情報を登録
        from taremin_cloth.pattern.session import BoneInfo3D
        ctrl.bones = {
            "Spine": BoneInfo3D(name="Spine", head_3d=(0, 0, 0.8), tail_3d=(0, 0, 1.2)),
            "UpperArm.L": BoneInfo3D(name="UpperArm.L", head_3d=(0.3, 0, 1.3), tail_3d=(0.3, 0, 1.0)),
        }

        from taremin_cloth.pattern.layout import compute_hud_layout
        layout = compute_hud_layout(*ctrl.screen_size)

        # 1. "target_bone" 行をクリックしてボーン選択 (Pick Bone) モード開始
        bone_rect = layout.card_fields["target_bone"]
        click_x = bone_rect.x + bone_rect.w * 0.5
        click_y = bone_rect.y + bone_rect.h * 0.5

        self.assertFalse(ctrl.picking_bone)
        ctrl.handle_event(EditorEvent(screen_pos=(click_x, click_y), is_press=True, button="LEFT"))
        self.assertTrue(ctrl.picking_bone)

        # ホバーボーン設定後の左クリックでボーン選択が確定し履歴に記録されること
        ctrl.hovered_bone = "Spine"
        ctrl.handle_event(EditorEvent(screen_pos=(200.0, 200.0), is_press=True, button="LEFT"))
        pl = ctrl.doc.placements[p.id]
        self.assertFalse(ctrl.picking_bone)
        self.assertEqual(pl.target_bone, "Spine")

        # Bキーで再度Pick Boneモードに入り、Xキーでボーン指定を解除できること
        ctrl.handle_event(EditorEvent(key="B"))
        self.assertTrue(ctrl.picking_bone)
        ctrl.handle_event(EditorEvent(key="X"))
        self.assertFalse(ctrl.picking_bone)
        self.assertEqual(ctrl.doc.placements[p.id].target_bone, "")

        # Undoで "Spine" に戻ること (Undo実行時は選択がクリアされるため再選択)
        ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        ctrl.selection.select_piece(p.id)
        pl = ctrl.doc.placements[p.id]
        self.assertEqual(pl.target_bone, "Spine")

        # 2. "pattern_up_angle_deg" 行をクリックしてインライン編集 (-90度) ＆ T / Shift+T ショートカット検証
        up_rect = layout.card_fields["pattern_up_angle_deg"]
        u_click_x = up_rect.x + up_rect.w * 0.5
        u_click_y = up_rect.y + up_rect.h * 0.5

        self.assertAlmostEqual(pl.pattern_up_angle_deg, 0.0)
        ctrl.handle_event(EditorEvent(screen_pos=(u_click_x, u_click_y), is_press=True, button="LEFT"))
        ctrl.handle_event(EditorEvent(screen_pos=(u_click_x, u_click_y), is_release=True, button="LEFT"))
        self.assertEqual(ctrl.editing_field, "pattern_up_angle_deg")
        ctrl.handle_event(EditorEvent(key="BACKSPACE"))
        ctrl.handle_event(EditorEvent(key="-"))
        ctrl.handle_event(EditorEvent(key="9"))
        ctrl.handle_event(EditorEvent(key="0"))
        ctrl.handle_event(EditorEvent(key="ENTER"))
        self.assertAlmostEqual(pl.pattern_up_angle_deg, -90.0)

        # Tキーで +90°回転 (-90 -> 0), Shift+T で -90°回転 (0 -> -90)
        ctrl.handle_event(EditorEvent(key="T"))
        self.assertAlmostEqual(pl.pattern_up_angle_deg, 0.0)
        ctrl.handle_event(EditorEvent(key="T", shift=True))
        self.assertAlmostEqual(pl.pattern_up_angle_deg, -90.0)
        # Undo で 0° に戻る
        ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        self.assertAlmostEqual(ctrl.doc.placements[p.id].pattern_up_angle_deg, 0.0)
        ctrl.selection.select_piece(p.id)
        pl = ctrl.doc.placements[p.id]

        # 3. "bone_angle_deg" 行をクリック（単発プレス＆リリース）してインライン数値編集 (45度)
        angle_rect = layout.card_fields["bone_angle_deg"]
        a_click_x = angle_rect.x + angle_rect.w * 0.5
        a_click_y = angle_rect.y + angle_rect.h * 0.5

        ctrl.handle_event(EditorEvent(screen_pos=(a_click_x, a_click_y), is_press=True, button="LEFT"))
        ctrl.handle_event(EditorEvent(screen_pos=(a_click_x, a_click_y), is_release=True, button="LEFT"))
        self.assertEqual(ctrl.editing_field, "bone_angle_deg")

        # 45 と入力して Enter
        ctrl.handle_event(EditorEvent(key="4"))
        ctrl.handle_event(EditorEvent(key="5"))
        ctrl.handle_event(EditorEvent(key="ENTER"))

        self.assertIsNone(ctrl.editing_field)
        self.assertAlmostEqual(pl.bone_angle_deg, 45.0)

        # 4. 角度コンパス (compass_rect) のドラッグ操作
        compass = layout.compass_rect
        cx = compass.x + compass.w * 0.5
        cy = compass.y + compass.h * 0.5

        # 右側 (dx > 0, dy = 0) をクリックしてドラッグ -> 90度
        drag_pos = (cx + 30.0, cy)
        ctrl.handle_event(EditorEvent(screen_pos=drag_pos, is_press=True, button="LEFT"))
        self.assertTrue(ctrl.is_dragging_compass)
        self.assertAlmostEqual(pl.bone_angle_deg, 90.0, places=1)

        # マウス解放
        ctrl.handle_event(EditorEvent(screen_pos=drag_pos, is_release=True, button="LEFT"))
        self.assertFalse(ctrl.is_dragging_compass)

        # Undo (Ctrl+Z) で 45度 に戻ること
        ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        undone_pl = ctrl.doc.placements[p.id]
        self.assertAlmostEqual(undone_pl.bone_angle_deg, 45.0)

    def test_bone_placement_slider_drag(self):
        """ボーン配置パラメータ（bone_t, bone_angle_deg, surface_offset）のマウスドラッグスライダー操作検証."""
        ctrl = self.controller
        p = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.3), name="Panel")
        ctrl.selection.select_piece(p.id)

        from taremin_cloth.pattern.layout import compute_hud_layout
        layout = compute_hud_layout(*ctrl.screen_size)

        # 1. bone_t スライダードラッグ
        t_rect = layout.card_fields["bone_t"]
        start_x = t_rect.x + t_rect.w * 0.5
        start_y = t_rect.y + t_rect.h * 0.5

        # プレス
        ctrl.handle_event(EditorEvent(screen_pos=(start_x, start_y), is_press=True, button="LEFT"))
        self.assertEqual(ctrl.sliding_field, "bone_t")
        self.assertFalse(ctrl.is_slider_dragging)

        # 右へ 40px 移動 (40 * 0.005 = +0.20)
        ctrl.handle_event(EditorEvent(screen_pos=(start_x + 40.0, start_y), is_move=True))
        self.assertTrue(ctrl.is_slider_dragging)
        pl = ctrl.doc.placements[p.id]
        self.assertAlmostEqual(pl.bone_t, 0.5 + 0.20, places=2)

        # リリースで確定
        ctrl.handle_event(EditorEvent(screen_pos=(start_x + 40.0, start_y), is_release=True, button="LEFT"))
        self.assertIsNone(ctrl.sliding_field)
        self.assertFalse(ctrl.is_slider_dragging)

        # Undo で元の 0.5 に戻ること
        ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        undone_pl = ctrl.doc.placements[p.id]
        self.assertAlmostEqual(undone_pl.bone_t, 0.5, places=2)

    def test_seam_routing_toggle_card_and_shortcut(self):
        """縫合選択時の Routing カードクリックおよび B キーによる Direct / Bone Wrap 切替検証."""
        from taremin_cloth.pattern.document import EdgeRange, Seam, SeamRouting
        from taremin_cloth.pattern.layout import compute_hud_layout

        ctrl = self.controller
        p = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.3, 0.2), name="Sleeve")
        nodes = p.get_outer_path().nodes
        seam = Seam(
            id="s1",
            a=EdgeRange(piece_id=p.id, path_id=p.outer_path_id, start_node_id=nodes[0].id, end_node_id=nodes[1].id),
            b=EdgeRange(piece_id=p.id, path_id=p.outer_path_id, start_node_id=nodes[2].id, end_node_id=nodes[3].id),
        )
        ctrl.doc.seams[seam.id] = seam
        ctrl.selection.select_seam(seam.id)

        layout = compute_hud_layout(*ctrl.screen_size)
        r_rect = layout.card_fields["seam_routing"]
        rx = r_rect.x + r_rect.w * 0.5
        ry = r_rect.y + r_rect.h * 0.5

        # カードクリックで DIRECT -> BONE_WRAP
        self.assertEqual(seam.routing, SeamRouting.DIRECT)
        handled = ctrl.handle_event(EditorEvent(screen_pos=(rx, ry), is_press=True, button="LEFT"))
        self.assertTrue(handled)
        self.assertEqual(seam.routing, SeamRouting.BONE_WRAP)

        # B キーで BONE_WRAP -> DIRECT
        handled_b = ctrl.handle_event(EditorEvent(key="B"))
        self.assertTrue(handled_b)
        self.assertEqual(seam.routing, SeamRouting.DIRECT)
        self.assertFalse(ctrl.picking_bone)

        # Undo で BONE_WRAP に戻ること
        ctrl.handle_event(EditorEvent(ctrl=True, key="Z"))
        self.assertEqual(ctrl.doc.seams[seam.id].routing, SeamRouting.BONE_WRAP)

    def test_piece_drag_snaps_to_world_grid_and_alt_bypass(self):
        """ピース全体のドラッグ移動で基準ノードが共通グリッドに吸着し、Alt押下中は吸着しないこと."""
        ctrl = self.controller
        p = ctrl.doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.2))
        ctrl.set_active_tool("SELECT")

        # 内部 (100,100) を掴み +34.2px,+12.7px 移動 -> 基準ノード(0,0)が(0.034,0.013)付近 -> グリッド(0.03,0.01)
        ctrl.handle_event(EditorEvent(screen_pos=(100.0, 100.0), is_press=True, button="LEFT"))
        ctrl.handle_event(EditorEvent(screen_pos=(134.2, 112.7), is_move=True))
        ctrl.handle_event(EditorEvent(screen_pos=(134.2, 112.7), is_release=True, button="LEFT"))
        tr = ctrl.doc.pieces[p.id].transform
        self.assertAlmostEqual(tr.tx, 0.03, places=6)
        self.assertAlmostEqual(tr.ty, 0.01, places=6)

        # Alt 押下中は端数のまま移動
        ctrl.handle_event(EditorEvent(screen_pos=(130.0, 110.0), is_press=True, button="LEFT", alt=True))
        ctrl.handle_event(EditorEvent(screen_pos=(164.2, 122.7), is_move=True, alt=True))
        ctrl.handle_event(EditorEvent(screen_pos=(164.2, 122.7), is_release=True, button="LEFT", alt=True))
        tr = ctrl.doc.pieces[p.id].transform
        self.assertAlmostEqual(tr.tx, 0.0642, places=4)

    def test_rect_then_poly_select_and_drag_both_pieces(self):
        """Rect作成後にPolyを追加（連続クリック重複やダブルクリック確定を含む）しても、SelでRectとPolyの双方を選択・移動できること."""
        ctrl = self.controller

        # 1. Rect で (0.0, 0.0) -> (0.2, 0.2) に矩形ピース作成
        ctrl.set_active_tool("RECTANGLE")
        ctrl.handle_event(EditorEvent(screen_pos=(0.0, 0.0), is_press=True, button="LEFT"))
        ctrl.handle_event(EditorEvent(screen_pos=(200.0, 200.0), is_move=True))
        ctrl.handle_event(EditorEvent(screen_pos=(200.0, 200.0), is_release=True, button="LEFT"))
        self.assertEqual(len(ctrl.doc.pieces), 1)
        rect_id = next(iter(ctrl.doc.pieces.keys()))

        # 2. Poly で (0.4, 0.0), (0.6, 0.0), (0.5, 0.2) をクリックし、最終点をダブルクリックで確定
        ctrl.set_active_tool("POLYGON")
        ctrl.handle_event(EditorEvent(screen_pos=(400.0, 0.0), is_press=True, button="LEFT"))
        ctrl.handle_event(EditorEvent(screen_pos=(400.0, 0.0), is_release=True, button="LEFT"))
        # 2点目で同一グリッドへの重複クリックがあっても縮退辺が作られないこと
        ctrl.handle_event(EditorEvent(screen_pos=(600.0, 0.0), is_press=True, button="LEFT"))
        ctrl.handle_event(EditorEvent(screen_pos=(601.0, 0.0), is_press=True, button="LEFT"))
        # 3点目をダブルクリックしてポリゴン確定
        ctrl.handle_event(EditorEvent(screen_pos=(500.0, 200.0), is_press=True, button="LEFT"))
        ctrl.handle_event(EditorEvent(screen_pos=(500.0, 200.0), is_press=True, button="LEFT"))

        self.assertEqual(len(ctrl.doc.pieces), 2)
        poly_id = next(pid for pid in ctrl.doc.pieces.keys() if pid != rect_id)

        # 3. Sel (SELECT) に切り替え、Rect の内部 (100, 100) をクリック＆ドラッグして Rect だけが移動すること
        ctrl.set_active_tool("SELECT")
        ctrl.handle_event(EditorEvent(screen_pos=(100.0, 100.0), is_press=True, button="LEFT"))
        self.assertEqual(ctrl.selection.piece_ids, {rect_id})

        ctrl.handle_event(EditorEvent(screen_pos=(130.0, 140.0), is_move=True))
        ctrl.handle_event(EditorEvent(screen_pos=(130.0, 140.0), is_release=True, button="LEFT"))
        self.assertAlmostEqual(ctrl.doc.pieces[rect_id].transform.tx, 0.03, places=3)
        self.assertAlmostEqual(ctrl.doc.pieces[rect_id].transform.ty, 0.04, places=3)
        self.assertAlmostEqual(ctrl.doc.pieces[poly_id].transform.tx, 0.0, places=3)

        # 4. 空白領域 (-100, -100) をクリックすると選択解除されること
        ctrl.handle_event(EditorEvent(screen_pos=(-100.0, -100.0), is_press=True, button="LEFT"))
        ctrl.handle_event(EditorEvent(screen_pos=(-100.0, -100.0), is_release=True, button="LEFT"))
        self.assertEqual(len(ctrl.selection.piece_ids), 0)

        # 5. Rect の外周辺上 (上辺 x=0.13, y=0.24 -> screen (130, 240)) をクリックすると外周辺のみ選択され、
        #    ピース内部 (130, 140) をクリックすると Rect ピースを選択できること
        ctrl.handle_event(EditorEvent(screen_pos=(130.0, 240.0), is_press=True, button="LEFT"))
        ctrl.handle_event(EditorEvent(screen_pos=(130.0, 240.0), is_release=True, button="LEFT"))
        self.assertEqual(len(ctrl.selection.edge_ranges), 1)
        self.assertEqual(ctrl.selection.edge_ranges[0].piece_id, rect_id)
        self.assertEqual(len(ctrl.selection.piece_ids), 0)

        ctrl.handle_event(EditorEvent(screen_pos=(130.0, 140.0), is_press=True, button="LEFT"))
        ctrl.handle_event(EditorEvent(screen_pos=(130.0, 140.0), is_release=True, button="LEFT"))
        self.assertEqual(ctrl.selection.piece_ids, {rect_id})


if __name__ == "__main__":
    unittest.main()
