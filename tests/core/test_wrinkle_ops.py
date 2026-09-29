"""
シワフィールドオペレーター層（ops/wrinkle.py）の単体・統合ロジックテスト (test_wrinkle_ops.py)
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# プロジェクトルートおよび python/ ディレクトリをモジュール検索パスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from taremin_cloth.ops import wrinkle
from taremin_cloth import i18n


class TestWrinkleOps(unittest.TestCase):
    """シワフィールドオペレーター群の定義、データ抽出、幾何更新ロジックの検証"""

    def test_operator_registered_and_properties(self):
        """オペレーターの bl_idname, bl_translation_context, プロパティ定義を確認"""
        # 1. Add Preset
        op_add = wrinkle.TAREMIN_CLOTH_OT_add_wrinkle_preset
        self.assertEqual(op_add.bl_idname, "taremin_cloth.add_wrinkle_preset")
        self.assertEqual(op_add.bl_translation_context, i18n.CONTEXT)
        ann_add = getattr(op_add, "__annotations__", {})
        self.assertIn("preset_name", ann_add)
        self.assertIn("t_position", ann_add)
        self.assertIn("scale_radius", ann_add)
        self.assertIn("start_modal_slide", ann_add)

        # 2. Slide Modal
        op_slide = wrinkle.TAREMIN_CLOTH_OT_slide_wrinkle_curves
        self.assertEqual(op_slide.bl_idname, "taremin_cloth.slide_wrinkle_curves")
        self.assertEqual(op_slide.bl_translation_context, i18n.CONTEXT)

        # 3. Save Preset
        op_save = wrinkle.TAREMIN_CLOTH_OT_save_wrinkle_preset
        self.assertEqual(op_save.bl_idname, "taremin_cloth.save_wrinkle_preset")
        self.assertEqual(op_save.bl_translation_context, i18n.CONTEXT)
        ann_save = getattr(op_save, "__annotations__", {})
        self.assertIn("preset_name", ann_save)
        self.assertIn("bone_name", ann_save)

    def test_preset_items_enumeration(self):
        """プリセットアイテム一覧コールバックが組み込みプリセットを網羅していることを確認"""
        items = wrinkle._get_preset_items(None, None)
        self.assertGreaterEqual(len(items), 5)
        item_keys = [it[0] for it in items]
        self.assertIn("cinch_single", item_keys)
        self.assertIn("puff_single", item_keys)
        self.assertIn("pinch_and_puff", item_keys)
        self.assertIn("accordion_double", item_keys)
        self.assertIn("y_branch_joint", item_keys)

    def test_extract_bone_chain_data_from_mock_armature(self):
        """モックアーマチュアから親子関係のあるボーンチェーンデータが正しく抽出されること"""
        mock_arm = MagicMock()
        mock_arm.type = 'ARMATURE'
        mock_arm.matrix_world = MagicMock()
        mock_arm.matrix_world.__matmul__ = lambda self, vec: vec

        pb_upper = MagicMock()
        pb_upper.name = "UpperArm"
        pb_upper.head = [0.0, 0.0, 0.0]
        pb_upper.tail = [0.0, 0.0, 1.0]
        pb_upper.length = 1.0
        pb_upper.parent = None
        pb_upper.bone = MagicMock(head_radius=0.06, tail_radius=0.05)

        pb_fore = MagicMock()
        pb_fore.name = "ForeArm"
        pb_fore.head = [0.0, 0.0, 1.0]
        pb_fore.tail = [0.0, 0.0, 2.0]
        pb_fore.length = 1.0
        pb_fore.parent = pb_upper
        pb_fore.bone = MagicMock(head_radius=0.05, tail_radius=0.04)
        pb_fore.children = []
        pb_upper.children = [pb_fore]

        mock_arm.pose.bones = {"UpperArm": pb_upper, "ForeArm": pb_fore}

        chain_data = wrinkle._extract_bone_chain_data(mock_arm, "ForeArm")
        self.assertEqual(len(chain_data), 2)
        self.assertEqual(chain_data[0][0], "UpperArm")
        self.assertEqual(chain_data[1][0], "ForeArm")
        # 半径がタプルで抽出されていること
        self.assertEqual(chain_data[0][3], (0.06, 0.05))
        self.assertEqual(chain_data[1][3], (0.05, 0.04))

    def test_slide_modal_collect_and_update_geometry(self):
        """スライドオペレーターがカーブとアーマチュアを収集し、スライド幾何更新を行うこと"""
        op = wrinkle.TAREMIN_CLOTH_OT_slide_wrinkle_curves()
        op._init_state()

        # モックカーブ
        curve_obj = MagicMock()
        curve_obj.type = 'CURVE'
        curve_obj.name = "Crest_Test"
        curve_props = {
            "wrinkle_type": "crest",
            "wrinkle_armature": "Armature",
            "wrinkle_target_bone": "BoneA",
            "wrinkle_t_param": 0.5,
            "wrinkle_scale_radius": 1.0,
            "wrinkle_influence_radius": 0.03,
            "wrinkle_strength": 1.0,
        }
        curve_obj.get.side_effect = lambda k, default=None: curve_props.get(k, default)
        curve_obj.__contains__ = lambda self, k: k in curve_props
        curve_obj.__getitem__ = lambda self, k: curve_props[k]
        def setitem(self, k, v):
            curve_props[k] = v
        curve_obj.__setitem__ = setitem

        # モックスプライン頂点
        mock_point = MagicMock()
        mock_point.co = [0.05, 0.0, 0.5, 1.0]
        mock_spline = MagicMock()
        mock_spline.points = [mock_point]
        curve_obj.data.splines = [mock_spline]

        # モックアーマチュア
        mock_arm = MagicMock()
        mock_arm.type = 'ARMATURE'
        mock_arm.matrix_world = MagicMock()
        mock_arm.matrix_world.__matmul__ = lambda self, vec: vec
        pb = MagicMock()
        pb.name = "BoneA"
        pb.head = [0.0, 0.0, 0.0]
        pb.tail = [0.0, 0.0, 1.0]
        pb.length = 1.0
        pb.parent = None
        pb.children = []
        pb.bone = MagicMock(head_radius=0.05, tail_radius=0.05)
        mock_arm.pose.bones = {"BoneA": pb}

        mock_context = MagicMock()
        mock_context.selected_objects = [curve_obj]
        mock_context.scene.objects = {"Armature": mock_arm}

        with patch("bpy.data.objects.get", return_value=mock_arm):
            collected = op._collect_curves(mock_context)
            self.assertTrue(collected)
            self.assertEqual(len(op._target_curves), 1)
            self.assertIsNotNone(op._chain)

            # t=0.8 へスライド
            op._current_t = 0.8
            op._scale_radius = 1.2
            op._update_curves_geometry()

            # プロパティが更新されたこと
            self.assertAlmostEqual(curve_props["wrinkle_t_param"], 0.8, places=5)
            self.assertAlmostEqual(curve_props["wrinkle_scale_radius"], 1.2, places=5)
            # 頂点Z座標が 0.8 付近に更新されたこと
            self.assertAlmostEqual(mock_point.co[2], 0.8, places=2)

    def test_add_wrinkle_preset_execute(self):
        """Add Wrinkle Preset オペレーターの execute が例外なくカーブを生成すること"""
        op = wrinkle.TAREMIN_CLOTH_OT_add_wrinkle_preset()
        op.preset_name = "cinch_single"
        op.t_position = 0.5
        op.scale_radius = 1.0
        op.start_modal_slide = False
        op.report = MagicMock()

        # モックアーマチュア
        mock_arm = MagicMock()
        mock_arm.type = 'ARMATURE'
        mock_arm.name = "Armature"
        mock_arm.matrix_world = MagicMock()
        mock_arm.matrix_world.__matmul__ = lambda self, vec: vec

        pb = MagicMock()
        pb.name = "BoneA"
        pb.head = [0.0, 0.0, 0.0]
        pb.tail = [0.0, 0.0, 1.0]
        pb.length = 1.0
        pb.parent = None
        pb.children = []
        pb.bone = MagicMock(head_radius=0.05, tail_radius=0.05)
        mock_arm.pose.bones = {"BoneA": pb}
        mock_arm.data.bones.active = pb

        mock_context = MagicMock()
        mock_context.active_object = mock_arm
        mock_context.active_pose_bone = pb
        mock_context.selected_objects = [mock_arm]
        mock_context.scene.objects = {"Armature": mock_arm}

        # モックコレクション
        mock_col = MagicMock()
        linked_objs = []
        mock_col.objects.link = lambda o: linked_objs.append(o)

        created_props = []
        def mock_new_obj(name, object_data):
            o = MagicMock()
            p = {}
            created_props.append(p)
            o.__setitem__ = lambda self, k, v: p.__setitem__(k, v)
            o.__getitem__ = lambda self, k: p[k]
            o.name = name
            return o

        with patch("taremin_cloth.ops.wrinkle._find_or_create_collection", return_value=mock_col), \
             patch("bpy.data.objects.new", side_effect=mock_new_obj), \
             patch("taremin_cloth.ops.wrinkle.tag_redraw_view3d"):
            res = op.execute(mock_context)
            self.assertEqual(res, {'FINISHED'})
            # cinch_single のカーブ（谷1本）が生成されてリンクされていること
            self.assertGreaterEqual(len(linked_objs), 1)
            self.assertEqual(created_props[0]["wrinkle_type"], "root")
            self.assertEqual(created_props[0]["wrinkle_target_bone"], "BoneA")
            self.assertIn("Cinch Single", created_props[0]["wrinkle_preset"])

    def test_find_nearest_bone_to_mouse(self):
        """マウス座標に最も近いボーンが正しく特定されること"""
        mock_arm = MagicMock()
        mock_arm.type = 'ARMATURE'
        mock_arm.matrix_world = MagicMock()
        mock_arm.matrix_world.__matmul__ = lambda self, vec: vec

        pb_top = MagicMock()
        pb_top.name = "BoneTop"
        pb_top.head = [0.0, 0.0, 1.0]
        pb_top.tail = [0.0, 0.0, 2.0]

        pb_bottom = MagicMock()
        pb_bottom.name = "BoneBottom"
        pb_bottom.head = [0.0, 0.0, 0.0]
        pb_bottom.tail = [0.0, 0.0, 1.0]

        mock_arm.pose.bones = [pb_top, pb_bottom]

        # 3D -> 2D 投影モック
        # BoneTop: 2D (50, 150)〜(50, 250)
        # BoneBottom: 2D (50, 50)〜(50, 150)
        def mock_loc_to_reg(region, rv3d, loc):
            z = loc[2]
            return (50.0, 50.0 + z * 100.0)

        mock_context = MagicMock()
        mock_context.region = MagicMock()
        mock_context.region_data = MagicMock()

        with patch("bpy_extras.view3d_utils.location_3d_to_region_2d", side_effect=mock_loc_to_reg):
            # マウスが (50, 220) にあるときは BoneTop (150〜250) が最も近い
            nearest = wrinkle.find_nearest_bone_to_mouse(mock_context, mock_arm, (50.0, 220.0))
            self.assertEqual(nearest, "BoneTop")

            # マウスが (50, 70) にあるときは BoneBottom (50〜150) が最も近い
            nearest2 = wrinkle.find_nearest_bone_to_mouse(mock_context, mock_arm, (50.0, 70.0))
            self.assertEqual(nearest2, "BoneBottom")

    def test_auto_setup_cloth_wrinkle_settings(self):
        """未設定の布オブジェクトにシワコレクションとシワフィールドが自動設定されること"""
        mock_cloth = MagicMock()
        mock_cloth.type = 'MESH'
        mock_props = MagicMock()
        mock_props.wrinkle_collection = None
        mock_props.use_wrinkle_field = False
        mock_props.wrinkle_armature = None
        mock_props.wrinkle_bone_name = ""
        mock_cloth.taremin_cloth = mock_props

        mock_col = MagicMock()
        mock_col.name = "TareminCloth_Wrinkles"
        mock_arm = MagicMock(name="Armature")

        mock_context = MagicMock()
        mock_context.active_object = mock_cloth
        mock_context.selected_objects = [mock_cloth]
        mock_context.scene.objects = [mock_cloth]

        wrinkle._auto_setup_cloth_wrinkle_settings(mock_context, mock_col, mock_arm, "ForeArm")

        # 自動代入されたことを検証
        self.assertEqual(mock_props.wrinkle_collection, mock_col)
        self.assertTrue(mock_props.use_wrinkle_field)
        self.assertEqual(mock_props.wrinkle_armature, mock_arm)
        self.assertEqual(mock_props.wrinkle_bone_name, "ForeArm")

    def test_slide_modal_retarget_bone(self):
        """スライドオペレーターの _retarget_to_bone でボーンが切り替わること"""
        op = wrinkle.TAREMIN_CLOTH_OT_slide_wrinkle_curves()
        op._init_state()

        # モックアーマチュアに BoneA と BoneB を準備
        mock_arm = MagicMock()
        mock_arm.type = 'ARMATURE'
        mock_arm.matrix_world = MagicMock()
        mock_arm.matrix_world.__matmul__ = lambda self, vec: vec

        pb_a = MagicMock(name="BoneA")
        pb_a.name = "BoneA"
        pb_a.head = [0.0, 0.0, 0.0]
        pb_a.tail = [0.0, 0.0, 1.0]
        pb_a.length = 1.0
        pb_a.parent = None
        pb_a.children = []
        pb_a.bone = MagicMock(head_radius=0.05, tail_radius=0.05)

        pb_b = MagicMock(name="BoneB")
        pb_b.name = "BoneB"
        pb_b.head = [0.0, 0.0, 1.0]
        pb_b.tail = [0.0, 0.0, 2.0]
        pb_b.length = 1.0
        pb_b.parent = pb_a
        pb_b.children = []
        pb_b.bone = MagicMock(head_radius=0.05, tail_radius=0.04)

        mock_arm.pose.bones = {"BoneA": pb_a, "BoneB": pb_b}
        op._armature_obj = mock_arm
        op._target_bone_name = "BoneA"

        # モックカーブ
        curve_obj = MagicMock()
        curve_obj.type = 'CURVE'
        curve_props = {"wrinkle_target_bone": "BoneA"}
        curve_obj.__setitem__ = lambda self, k, v: curve_props.__setitem__(k, v)
        curve_obj.__getitem__ = lambda self, k: curve_props[k]
        mock_point = MagicMock()
        mock_point.co = [0.05, 0.0, 0.5, 1.0]
        mock_spline = MagicMock()
        mock_spline.points = [mock_point]
        curve_obj.data.splines = [mock_spline]

        op._target_curves = [curve_obj]
        op._initial_points = [np.array([[0.05, 0.0, 0.5]], dtype=np.float32)]

        mock_context = MagicMock()
        with patch("taremin_cloth.ops.wrinkle.tag_redraw_view3d"):
            op._retarget_to_bone(mock_context, "BoneB")

        self.assertEqual(op._target_bone_name, "BoneB")
        self.assertEqual(curve_props["wrinkle_target_bone"], "BoneB")
        self.assertIsNotNone(op._chain)

    def test_two_phase_modal_flow(self):
        """ボーン選択フェイズ(PICK_BONE)からスライドフェイズ(SLIDE)への移行とキャンセル削除を検証"""
        op = wrinkle.TAREMIN_CLOTH_OT_slide_wrinkle_curves()
        op.report = MagicMock()
        op.initial_phase = 'PICK_BONE'
        op.is_new_addition = True
        op._init_state()
        op._phase = 'PICK_BONE'
        op._is_new_addition = True

        # モックカーブ
        curve_obj = MagicMock()
        curve_obj.type = 'CURVE'
        curve_data = MagicMock()
        curve_data.users = 0
        curve_obj.data = curve_data

        op._target_curves = [curve_obj]
        op._target_bone_name = "ForeArm"

        mock_context = MagicMock()

        # 1. フェイズ1で LMB (クリック PRESS) -> フェイズ2 (SLIDE) へ移行
        event_lmb = MagicMock()
        event_lmb.type = 'LEFTMOUSE'
        event_lmb.value = 'PRESS'
        event_lmb.mouse_x = 100

        res = op.modal(mock_context, event_lmb)
        self.assertEqual(res, {'RUNNING_MODAL'})
        self.assertEqual(op._phase, 'SLIDE')
        self.assertTrue(op._press_tracker.pressed)

        # 1b. 指を離す (RELEASE) -> 押下追跡が解除される
        event_release = MagicMock()
        event_release.type = 'LEFTMOUSE'
        event_release.value = 'RELEASE'
        res_rel = op.modal(mock_context, event_release)
        self.assertEqual(res_rel, {'RUNNING_MODAL'})
        self.assertFalse(op._press_tracker.pressed)

        # 2. フェイズ2で Bキー -> フェイズ1 (PICK_BONE) へ再移行
        event_b = MagicMock()
        event_b.type = 'B'
        event_b.value = 'PRESS'
        res_b = op.modal(mock_context, event_b)
        self.assertEqual(res_b, {'RUNNING_MODAL'})
        self.assertEqual(op._phase, 'PICK_BONE')

        # 3. 再びクリックでフェイズ2へ移行
        res_pick2 = op.modal(mock_context, event_lmb)
        self.assertEqual(res_pick2, {'RUNNING_MODAL'})
        self.assertEqual(op._phase, 'SLIDE')

        # 4. フェイズ2で RELEASE 後の再クリック (PRESS) -> 確定 (FINISHED)
        op.modal(mock_context, event_release)
        event_confirm = MagicMock()
        event_confirm.type = 'LEFTMOUSE'
        event_confirm.value = 'PRESS'
        res_confirm = op.modal(mock_context, event_confirm)
        self.assertEqual(res_confirm, {'FINISHED'})

        # 5. フェイズ1で ESC -> キャンセル & 初期配置を維持 (カーブオブジェクトは削除しない)
        op._phase = 'PICK_BONE'
        event_esc = MagicMock()
        event_esc.type = 'ESC'
        event_esc.value = 'PRESS'
        with patch("bpy.data.objects.remove") as mock_remove_obj, \
             patch("bpy.data.curves.remove") as mock_remove_curve, \
             patch.object(op, "_restore_initial_geometry") as mock_restore:
            res_esc = op.modal(mock_context, event_esc)
            self.assertEqual(res_esc, {'CANCELLED'})
            # プリセット追加後のESCでもカーブが勝手に消えないこと (削除が呼ばれない)
            mock_remove_obj.assert_not_called()
            mock_remove_curve.assert_not_called()
            mock_restore.assert_called_once()

    def test_extract_all_armature_bones(self):
        """アーマチュアから全ボーンのワールド座標と半径が抽出されること"""
        mock_arm = MagicMock()
        mock_arm.type = 'ARMATURE'
        mock_arm.matrix_world = MagicMock()
        mock_arm.matrix_world.__matmul__ = lambda self, vec: vec

        pb1 = MagicMock(name="Bone1")
        pb1.name = "Bone1"
        pb1.head = [0.0, 0.0, 0.0]
        pb1.tail = [0.0, 1.0, 0.0]
        pb1.length = 1.0
        pb1.bone = MagicMock(head_radius=0.08, tail_radius=0.06)

        pb2 = MagicMock(name="Bone2")
        pb2.name = "Bone2"
        pb2.head = [0.0, 1.0, 0.0]
        pb2.tail = [0.0, 2.0, 0.0]
        pb2.length = 1.0
        pb2.bone = MagicMock(head_radius=0.06, tail_radius=0.04)

        mock_arm.pose.bones = {"Bone1": pb1, "Bone2": pb2}

        all_bones = wrinkle._extract_all_armature_bones(mock_arm)
        self.assertEqual(len(all_bones), 2)
        self.assertEqual(all_bones[0][0], "Bone1")
        self.assertEqual(all_bones[0][1], [0.0, 0.0, 0.0])
        self.assertEqual(all_bones[0][2], [0.0, 1.0, 0.0])
        self.assertAlmostEqual(all_bones[0][3], 0.08, places=5)
        self.assertEqual(all_bones[1][0], "Bone2")

    def test_compute_octahedron_mesh(self):
        """八面体ボーンのTRISとLINESの頂点数が正しく生成され、太さが一定に制御されること"""
        from taremin_cloth.utils.drawing import compute_octahedron_mesh
        tris, lines = compute_octahedron_mesh([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], radius=0.016)
        # 8三角形 * 3 = 24頂点
        self.assertEqual(len(tris), 24)
        # 12線分 * 2 = 24頂点
        self.assertEqual(len(lines), 24)

        # ボーン長1.0mでも太さが一定（0.016m）に制御されていること
        max_r = max(abs(pt[0]) for pt in tris)
        self.assertAlmostEqual(max_r, 0.016, places=3)

        # ゼロ長の場合は空リスト
        empty_tris, empty_lines = compute_octahedron_mesh([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], radius=0.016)
        self.assertEqual(len(empty_tris), 0)
        self.assertEqual(len(empty_lines), 0)

    def test_draw_wrinkle_armature_overlay(self):
        """draw_wrinkle_armature_overlay がシェーダーを呼び出して例外なく実行されること"""
        from taremin_cloth.utils.drawing import draw_wrinkle_armature_overlay
        bones_data = [
            ("Bone1", [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.08),
            ("Bone2", [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], 0.06),
        ]
        mock_shader = MagicMock()
        mock_batch = MagicMock()

        with patch("taremin_cloth.utils.drawing.get_3d_uniform_color_shader", return_value=mock_shader), \
             patch("taremin_cloth.utils.drawing.get_point_shader", return_value=mock_shader), \
             patch("taremin_cloth.utils.drawing.batch_for_shader", return_value=mock_batch), \
             patch("gpu.state.depth_test_set"), \
             patch("gpu.state.depth_test_get", return_value='LESS_EQUAL'), \
             patch("gpu.state.line_width_set"), \
             patch("gpu.state.line_width_get", return_value=1.0), \
             patch("gpu.state.blend_set"), \
             patch("gpu.state.blend_get", return_value='NONE'), \
             patch("gpu.state.point_size_set"), \
             patch("gpu.state.point_size_get", return_value=1.0):
            # 例外なく描画が完了すること
            draw_wrinkle_armature_overlay(bones_data, active_bone_name="Bone1", chain_bone_names=["Bone1", "Bone2"])
            self.assertTrue(mock_shader.bind.called)

    def test_retarget_to_bone_moves_curve_to_new_bone_center(self):
        """ボーンリターゲット時に、カーブが旧ボーンの位置に残らず新ボーンの中心へ正しくスナップすること"""
        op = wrinkle.TAREMIN_CLOTH_OT_slide_wrinkle_curves()
        op._init_state()

        # ボーンA: 原点 (0, 0, 0) -> (0, 0, 1)
        pb_a = MagicMock()
        pb_a.name = "BoneA"
        pb_a.head = [0.0, 0.0, 0.0]
        pb_a.tail = [0.0, 0.0, 1.0]
        pb_a.length = 1.0
        pb_a.parent = None
        pb_a.bone = MagicMock(head_radius=0.05, tail_radius=0.05)
        pb_a.children = []

        # ボーンB: 10m離れた位置 (10, 0, 0) -> (10, 0, 1)
        pb_b = MagicMock()
        pb_b.name = "BoneB"
        pb_b.head = [10.0, 0.0, 0.0]
        pb_b.tail = [10.0, 0.0, 1.0]
        pb_b.length = 1.0
        pb_b.parent = None
        pb_b.bone = MagicMock(head_radius=0.05, tail_radius=0.05)
        pb_b.children = []

        mock_arm = MagicMock()
        mock_arm.name = "Armature"
        mock_arm.type = 'ARMATURE'
        mock_arm.matrix_world = MagicMock()
        mock_arm.matrix_world.__matmul__ = lambda self, vec: vec
        mock_arm.pose.bones = {"BoneA": pb_a, "BoneB": pb_b}

        # ボーンAの周囲に初期配置されたカーブオブジェクト
        mock_curve = MagicMock()
        mock_curve.type = 'CURVE'
        mock_curve_data = MagicMock()
        mock_spline = MagicMock()
        points = []
        for _ in range(8):
            p = MagicMock()
            p.co = [0.05, 0.0, 0.5, 1.0]
            points.append(p)
        mock_spline.points = points
        mock_curve_data.splines = [mock_spline]
        mock_curve.data = mock_curve_data

        curve_props = {
            "wrinkle_type": "root",
            "wrinkle_curve_index": 0,
            "wrinkle_preset": "cinch_single",
            "wrinkle_target_bone": "BoneA",
            "wrinkle_armature": "Armature",
            "wrinkle_t_param": 0.5,
            "wrinkle_scale_radius": 1.0,
            "wrinkle_influence_radius": 0.03,
            "wrinkle_strength": 1.0,
        }
        mock_curve.get = lambda k, d=None: curve_props.get(k, d)
        mock_curve.__getitem__ = lambda self, k: curve_props[k]
        mock_curve.__setitem__ = lambda self, k, v: curve_props.__setitem__(k, v)

        mock_context = MagicMock()
        mock_context.selected_objects = [mock_curve]
        mock_context.active_object = mock_curve
        mock_context.scene.objects = [mock_curve, mock_arm]

        # 収集
        with patch.object(wrinkle, "bpy") as mock_bpy:
            mock_bpy.data.objects.get.side_effect = lambda name: mock_arm if name == "Armature" else None
            ok = op._collect_curves(mock_context)
            self.assertTrue(ok)
            self.assertEqual(op._target_bone_name, "BoneA")

            # ボーンBへリターゲット！
            op._retarget_to_bone(mock_context, "BoneB")

            self.assertEqual(op._target_bone_name, "BoneB")
            self.assertEqual(curve_props["wrinkle_target_bone"], "BoneB")

            # カーブの各点がBoneBの位置（X=10付近）に移動していることを検証
            x_coords = [p.co[0] for p in mock_spline.points]
            mean_x = float(np.mean(x_coords))
            # X座標が 10.0m 付近（例: 9.9m 〜 10.1m）にスナップしていること
            self.assertAlmostEqual(mean_x, 10.0, delta=0.2)
            # Z座標が BoneB の中央（Z=0.5付近）にあること
            z_coords = [p.co[2] for p in mock_spline.points]
            mean_z = float(np.mean(z_coords))
            self.assertAlmostEqual(mean_z, 0.5, delta=0.2)

    def test_compute_wrinkle_influence_mesh_geometry(self):
        """影響範囲メッシュ生成が有限値の滑らかなジオメトリを生成すること"""
        from taremin_cloth.utils.drawing import compute_wrinkle_influence_mesh

        # Z=0〜0.1、目標半径0.07の全周影響範囲
        tris, lines = compute_wrinkle_influence_mesh(
            origin=[0.0, 0.0, -0.1], axis=[0.0, 0.0, 1.0], normal=[1.0, 0.0, 0.0],
            theta_min=0.0, theta_max=2.0 * np.pi, z_min=0.0, z_max=0.1,
            target_r=0.07, thick_in=0.0, thick_out=0.015,
            n_theta=16, n_z=2,
        )

        self.assertGreater(len(tris), 0)
        self.assertGreater(len(lines), 0)
        # TRISは3の倍数
        self.assertEqual(len(tris) % 3, 0)
        # LINESは2の倍数
        self.assertEqual(len(lines) % 2, 0)

        # すべての頂点座標が有限値 (NaN/Inf なし) であること
        for pt in tris:
            for coord in pt:
                self.assertTrue(np.isfinite(coord))
        for pt in lines:
            for coord in pt:
                self.assertTrue(np.isfinite(coord))

        # 外面が目標半径+厚みの近傍にあること
        radii = [float(np.linalg.norm(np.array(p[:2]))) for p in tris]
        self.assertAlmostEqual(max(radii), 0.085, delta=1e-6)

    def test_wrinkle_curve_rna_sync_to_id_properties(self):
        """TareminWrinkleCurveSettings の更新ハンドラーがIDプロパティへ正しく同期すること"""
        from taremin_cloth.properties import _on_wrinkle_curve_prop_updated

        mock_obj = {}
        mock_settings = MagicMock()
        mock_settings.id_data = mock_obj
        mock_settings.curve_type = 'root'
        mock_settings.strength = 2.5
        mock_settings.influence_radius = 0.045
        mock_settings.target_radius = 0.12

        _on_wrinkle_curve_prop_updated(mock_settings, None)

        self.assertEqual(mock_obj.get("wrinkle_type"), "root")
        self.assertAlmostEqual(mock_obj.get("wrinkle_strength"), 2.5)
        self.assertAlmostEqual(mock_obj.get("wrinkle_influence_radius"), 0.045)
        self.assertAlmostEqual(mock_obj.get("wrinkle_target_radius"), 0.12)

    def test_slide_modal_updates_curve_rna(self):
        """スライドオペレーターがジオメトリ更新時にカーブの taremin_wrinkle RNA も更新すること"""
        op = wrinkle.TAREMIN_CLOTH_OT_slide_wrinkle_curves()
        op._init_state()

        mock_curve = MagicMock()
        mock_curve.type = 'CURVE'
        mock_spline = MagicMock()
        mock_spline.points = [MagicMock(co=[0.0, 0.0, 0.0, 1.0])]
        mock_curve.data.splines = [mock_spline]

        mock_tw = MagicMock()
        mock_tw.influence_radius = 0.03
        mock_tw.strength = 1.0
        mock_curve.taremin_wrinkle = mock_tw

        curve_props = {}
        mock_curve.get = lambda k, d=None: curve_props.get(k, d)
        mock_curve.__setitem__ = lambda self, k, v: curve_props.__setitem__(k, v)

        op._target_curves = [mock_curve]
        op._chain = MagicMock()
        op._base_t = 0.5
        op._current_t = 0.6
        op._scale_radius = 1.2
        op._influence_radius = 0.05
        op._strength = 2.0
        op._initial_points = [np.array([[0.0, 0.0, 0.0]], dtype=np.float32)]

        with patch("taremin_cloth.ops.wrinkle.slide_curves_along_chain") as mock_slide:
            mock_slide.return_value = [np.array([[0.1, 0.0, 0.0]], dtype=np.float32)]
            op._update_curves_geometry()

            # IDプロパティの更新確認
            self.assertEqual(curve_props["wrinkle_influence_radius"], 0.05)
            self.assertEqual(curve_props["wrinkle_strength"], 2.0)
            # RNAプロパティの更新確認
            self.assertEqual(mock_tw.influence_radius, 0.05)
            self.assertEqual(mock_tw.strength, 2.0)

    def test_restore_initial_geometry_restores_true_initial_state_after_retarget(self):
        """リターゲットで別のボーンに移動した後にESCキャンセルした場合、真の初期ボーンと初期座標に完全復元されること"""
        op = wrinkle.TAREMIN_CLOTH_OT_slide_wrinkle_curves()
        op._init_state()

        mock_curve = MagicMock()
        mock_curve.type = 'CURVE'
        mock_spline = MagicMock()
        p0 = MagicMock(co=[1.0, 2.0, 3.0, 1.0])
        mock_spline.points = [p0]
        mock_curve.data.splines = [mock_spline]

        curve_props = {
            "wrinkle_target_bone": "OriginalBone",
            "wrinkle_t_param": 0.5,
            "wrinkle_scale_radius": 1.0,
            "wrinkle_influence_radius": 0.03,
            "wrinkle_strength": 1.0,
        }
        mock_curve.get = lambda k, d=None: curve_props.get(k, d)
        mock_curve.__setitem__ = lambda self, k, v: curve_props.__setitem__(k, v)

        op._target_curves = [mock_curve]
        op._target_bone_name = "OriginalBone"
        op._base_t = 0.5
        op._current_t = 0.5
        op._initial_points = [np.array([[1.0, 2.0, 3.0]], dtype=np.float32)]

        # _collect_curves で退避される真の初期状態
        op._true_initial_points = [pts.copy() for pts in op._initial_points]
        op._true_initial_bone_name = "OriginalBone"
        op._true_initial_base_t = 0.5

        # 別のボーンにリターゲットされてカーブが別の位置に移動した状態をシミュレート
        curve_props["wrinkle_target_bone"] = "DifferentBone"
        p0.co = [10.0, 20.0, 30.0, 1.0]
        op._target_bone_name = "DifferentBone"
        op._initial_points = [np.array([[10.0, 20.0, 30.0]], dtype=np.float32)]

        # ここでESCキャンセルが呼ばれた！
        op._restore_initial_geometry()

        # 真の初期ボーンに戻っていること！
        self.assertEqual(curve_props["wrinkle_target_bone"], "OriginalBone")
        # 真の初期座標 (1.0, 2.0, 3.0) に完全復元されていること！
        self.assertAlmostEqual(p0.co[0], 1.0)
        self.assertAlmostEqual(p0.co[1], 2.0)
        self.assertAlmostEqual(p0.co[2], 3.0)

    def test_pick_bone_first_click_transitions_to_slide_and_guards_confirm(self):
        """ボーン選択フェーズでは1回目のLMBクリックで即座にSLIDEへ移行し、SLIDE側で確定誤爆ガードが有効になること"""
        op = wrinkle.TAREMIN_CLOTH_OT_slide_wrinkle_curves()
        op._init_state()
        op._phase = 'PICK_BONE'
        op._target_bone_name = "TargetBone"
        op._initial_points = [np.array([[5.0, 5.0, 5.0]], dtype=np.float32)]

        mock_context = MagicMock()

        # 1. ユーザーの1回目のクリック (PRESS): PICK_BONE から即座に SLIDE へ移行すること！
        event_press = MagicMock()
        event_press.type = 'LEFTMOUSE'
        event_press.value = 'PRESS'
        event_press.mouse_x = 100
        res = op.modal(mock_context, event_press)

        self.assertEqual(res, {'RUNNING_MODAL'})
        self.assertEqual(op._phase, 'SLIDE', "1回目のクリックで確実にSLIDEフェーズへ移行すること")
        self.assertTrue(op._press_tracker.pressed, "ボーン決定クリックの指が離されるまで確定誤爆防止フラグが立つこと")
        self.assertEqual(op._true_initial_bone_name, "TargetBone", "選択されたボーンが真の初期ボーンとして記憶されること")
        self.assertAlmostEqual(op._true_initial_points[0][0, 0], 5.0)

        # 2. まだ指が離されていない状態でのクリック確定はブロックされること
        event_press_still_held = MagicMock()
        event_press_still_held.type = 'LEFTMOUSE'
        event_press_still_held.value = 'PRESS'
        res_held = op.modal(mock_context, event_press_still_held)
        self.assertEqual(res_held, {'RUNNING_MODAL'}, "指が離される前のクリック確定は誤爆として無視されること")
        self.assertEqual(op._phase, 'SLIDE')

        # 3. 指が離された (RELEASE)
        event_release = MagicMock()
        event_release.type = 'LEFTMOUSE'
        event_release.value = 'RELEASE'
        op.modal(mock_context, event_release)
        self.assertFalse(op._press_tracker.pressed, "RELEASEで確定誤爆防止フラグが解除されること")

        # 4. スライド調整後、次の正式なクリックで確定 (FINISHED) すること
        event_confirm = MagicMock()
        event_confirm.type = 'LEFTMOUSE'
        event_confirm.value = 'PRESS'
        res_confirm = op.modal(mock_context, event_confirm)
        self.assertEqual(res_confirm, {'FINISHED'}, "次のクリックで正常に確定終了すること")

    def test_add_wrinkle_preset_launches_pick_bone_phase(self):
        """プリセット追加後に起動されるスライドモーダルが initial_phase='PICK_BONE' で起動すること"""
        op = wrinkle.TAREMIN_CLOTH_OT_add_wrinkle_preset()
        # execute 内で slide_wrinkle_curves が initial_phase='PICK_BONE' で呼ばれることを確認
        with patch("bpy.ops.taremin_cloth.slide_wrinkle_curves") as mock_slide_op, \
             patch("taremin_cloth.ops.wrinkle._find_target_armature") as mock_find_arm, \
             patch("taremin_cloth.ops.wrinkle.get_builtin_presets") as mock_presets, \
             patch("taremin_cloth.ops.wrinkle._extract_bone_chain_data") as mock_chain_data, \
             patch("taremin_cloth.ops.wrinkle.instantiate_preset_on_chain") as mock_instantiate, \
             patch("taremin_cloth.ops.wrinkle._find_or_create_collection") as mock_col, \
             patch("bpy.data.curves.new") as mock_curve_new, \
             patch("bpy.data.objects.new") as mock_obj_new:

            mock_bone = MagicMock()
            mock_bone.name = "Spine"
            mock_arm = MagicMock()
            mock_bones = MagicMock()
            mock_bones.__contains__ = lambda self, k: k == "Spine"
            mock_bones.__iter__ = lambda self: iter([mock_bone])
            mock_bones.__len__ = lambda self: 1
            mock_bones.__getitem__ = lambda self, k: mock_bone
            mock_arm.pose.bones = mock_bones
            mock_find_arm.return_value = mock_arm
            mock_presets.return_value = {"cinch_single": MagicMock(name="cinch_single")}
            mock_chain_data.return_value = [("Spine", [0, 0, 0], [0, 0, 1], (0.05, 0.05))]
            mock_instantiate.return_value = ([MagicMock(points=[[0, 0, 0]], strength=1.0, influence_radius=0.03, target_radius=None)], [])

            mock_obj = MagicMock()
            mock_obj.data = MagicMock()
            mock_obj_new.return_value = mock_obj

            mock_context = MagicMock()
            mock_context.scene = MagicMock()

            op.preset_name = "cinch_single"
            op.t_position = 0.5
            op.scale_radius = 1.0
            op.start_modal_slide = True
            op.report = MagicMock()
            res = op.execute(mock_context)

            self.assertEqual(res, {'FINISHED'})
            mock_slide_op.assert_called_once_with('INVOKE_DEFAULT', initial_phase='PICK_BONE', is_new_addition=True)


if __name__ == "__main__":
    unittest.main()


