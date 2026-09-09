import unittest
import bpy
import sys
from pathlib import Path

# pythonディレクトリをsys.pathに追加
addon_dir = Path(__file__).parent.parent / "python"
if str(addon_dir) not in sys.path:
    sys.path.insert(0, str(addon_dir))

import taremin_cloth
from taremin_cloth.panels import (
    TAREMIN_CLOTH_PT_objects_panel,
    TAREMIN_CLOTH_PT_main_panel,
    TAREMIN_CLOTH_PT_collider_panel,
)
from tests.fixtures.panel_test_utils import render_panel, MockLayout


class TestPanels(unittest.TestCase):
    """UIパネル描画およびテストユーティリティの動作検証テスト"""

    def setUp(self):
        taremin_cloth.register()
        bpy.ops.mesh.primitive_plane_add()
        self.obj = bpy.context.active_object
        self.created_objects = [self.obj]

    def tearDown(self):
        for obj in self.created_objects:
            if obj and obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        taremin_cloth.unregister()

    # -------------------------------------------------------------------------
    # オブジェクト未選択・無効オブジェクト時のテスト
    # -------------------------------------------------------------------------

    def test_draw_with_no_active_object(self):
        """アクティブオブジェクトが存在しない状態でのパネル描画テスト"""
        bpy.context.view_layer.objects.active = None

        # メインパネル
        layout_main = render_panel(TAREMIN_CLOTH_PT_main_panel)
        labels_main = layout_main.get_labels()
        self.assertTrue(any("メッシュオブジェクトを選択してください" in lbl for lbl in labels_main))

        # コライダーパネル
        layout_col = render_panel(TAREMIN_CLOTH_PT_collider_panel)
        labels_col = layout_col.get_labels()
        self.assertTrue(any("オブジェクトを選択してください" in lbl for lbl in labels_col))

    def test_draw_with_non_mesh_object(self):
        """メッシュ以外のオブジェクト（カメラ等）選択時のパネル描画テスト"""
        cam_data = bpy.data.cameras.new(name="TestCam")
        cam_obj = bpy.data.objects.new(name="TestCamObj", object_data=cam_data)
        bpy.context.collection.objects.link(cam_obj)
        self.created_objects.append(cam_obj)

        bpy.context.view_layer.objects.active = cam_obj

        # メインパネルはメッシュ以外を拒否するラベルを表示
        layout_main = render_panel(TAREMIN_CLOTH_PT_main_panel)
        labels_main = layout_main.get_labels()
        self.assertTrue(any("メッシュオブジェクトを選択してください" in lbl for lbl in labels_main))

    # -------------------------------------------------------------------------
    # GPU Cloth メインパネル (TAREMIN_CLOTH_PT_main_panel) のテスト
    # -------------------------------------------------------------------------

    def test_main_panel_cloth_disabled(self):
        """Cloth が無効な初期状態でのメインパネル描画テスト"""
        self.obj.taremin_cloth.is_cloth = False

        layout = render_panel(TAREMIN_CLOTH_PT_main_panel)
        operators = layout.get_operators()

        # Enable Cloth ボタンが表示されていること
        self.assertIn("taremin_cloth.toggle_cloth", operators)
        # 詳細パラメータ群はまだ表示されていないこと
        self.assertNotIn("taremin_cloth.interactive", operators)
        self.assertNotIn("taremin_cloth.save_preset", operators)

    def test_main_panel_cloth_enabled(self):
        """Cloth が有効な状態でのメインパネル描画テスト（全セクション・プロパティ検証）"""
        self.obj.taremin_cloth.is_cloth = True
        self.obj.taremin_cloth.enable_adaptive_substep = True
        self.obj.taremin_cloth.enable_sewing = True

        layout = render_panel(TAREMIN_CLOTH_PT_main_panel)
        operators = layout.get_operators()
        menus = layout.get_menus()
        props = layout.get_props()

        # 主要オペレーターの存在確認
        self.assertIn("taremin_cloth.toggle_cloth", operators)
        self.assertIn("taremin_cloth.interactive", operators)
        self.assertIn("taremin_cloth.reset_selected", operators)
        self.assertIn("taremin_cloth.reset_all", operators)
        self.assertIn("taremin_cloth.apply_rest_shape", operators)
        self.assertIn("taremin_cloth.save_preset", operators)
        self.assertIn("taremin_cloth.delete_preset", operators)
        self.assertIn("taremin_cloth.create_seam", operators)

        # プリセットメニューの確認
        self.assertIn("TAREMIN_CLOTH_MT_fabric_presets", menus)
        self.assertIn("TAREMIN_CLOTH_MT_simulation_presets", menus)

        # プリセットボタンのカテゴリ設定確認
        all_save_items = [
            item.details["properties"]
            for item in layout.get_all_items()
            if item.item_type == "operator" and item.details["operator"] == "taremin_cloth.save_preset"
        ]
        categories = [p.category for p in all_save_items]
        self.assertIn("fabric", categories)
        self.assertIn("simulation", categories)

        # プロパティが正しくレイアウトに組み込まれているか確認
        prop_names = [name for data, name in props]
        self.assertIn("tension_stiffness", prop_names)
        self.assertIn("bending_stiffness", prop_names)
        self.assertIn("air_damping", prop_names)
        self.assertIn("gravity", prop_names)
        self.assertIn("substeps", prop_names)
        self.assertIn("min_substeps", prop_names)
        self.assertIn("enable_adaptive_substep", prop_names)
        self.assertIn("sewing_shrink_speed", prop_names)
        self.obj.taremin_cloth.enable_edge_collision = True
        layout = render_panel(TAREMIN_CLOTH_PT_main_panel)
        props = layout.get_props()
        prop_names = [name for data, name in props]
        self.assertIn("enable_edge_collision", prop_names)
        self.assertIn("edge_margin_scale", prop_names)
        self.assertIn("edge_margin_offset", prop_names)

    # -------------------------------------------------------------------------
    # GPU Collider パネル (TAREMIN_CLOTH_PT_collider_panel) のテスト
    # -------------------------------------------------------------------------

    def test_collider_panel_disabled(self):
        """Collider が無効な状態でのコライダーパネル描画テスト"""
        self.obj.taremin_cloth_collider.is_collider = False

        layout = render_panel(TAREMIN_CLOTH_PT_collider_panel)
        props = layout.get_props()
        prop_names = [name for data, name in props]

        self.assertIn("is_collider", prop_names)
        # 詳細パラメータ群は非表示
        self.assertNotIn("collider_type", prop_names)
        self.assertEqual(len(layout.get_operators()), 0)

    def test_collider_panel_enabled_sphere(self):
        """Collider (SPHERE) が有効な状態でのコライダーパネル描画テスト"""
        self.obj.taremin_cloth_collider.is_collider = True
        self.obj.taremin_cloth_collider.collider_type = 'SPHERE'

        layout = render_panel(TAREMIN_CLOTH_PT_collider_panel)
        operators = layout.get_operators()
        menus = layout.get_menus()
        props = layout.get_props()
        prop_names = [name for data, name in props]

        # プリセット関連
        self.assertIn("TAREMIN_CLOTH_MT_collider_presets", menus)
        self.assertIn("taremin_cloth.save_preset", operators)
        self.assertIn("taremin_cloth.delete_preset", operators)

        # category が 'collider' に設定されていること
        op_save = layout.find_operator_props("taremin_cloth.save_preset")
        self.assertIsNotNone(op_save)
        self.assertEqual(op_save.category, 'collider')

        op_del = layout.find_operator_props("taremin_cloth.delete_preset")
        self.assertIsNotNone(op_del)
        self.assertEqual(op_del.category, 'collider')

        # SPHERE パラメータ
        self.assertIn("collider_type", prop_names)
        self.assertIn("radius", prop_names)
        self.assertNotIn("thickness", prop_names)
        self.assertIn("friction", prop_names)
        self.assertIn("restitution", prop_names)

    def test_collider_panel_enabled_mesh(self):
        """Collider (MESH) が有効な状態でのコライダーパネル描画テスト"""
        self.obj.taremin_cloth_collider.is_collider = True
        self.obj.taremin_cloth_collider.collider_type = 'MESH'

        layout = render_panel(TAREMIN_CLOTH_PT_collider_panel)
        props = layout.get_props()
        prop_names = [name for data, name in props]

        self.assertIn("thickness", prop_names)
        self.assertIn("single_sided", prop_names)
        self.assertNotIn("radius", prop_names)

    # -------------------------------------------------------------------------
    # オペレーターが None を返した場合の耐障害性テスト (バグ再発防止)
    # -------------------------------------------------------------------------

    def test_panels_resilience_when_operator_returns_none(self):
        """
        未登録オペレーターなどにより layout.operator() が None を返した場合でも、
        AttributeError でパネル描画がクラッシュしないことを検証するテスト。
        """
        self.obj.taremin_cloth.is_cloth = True
        self.obj.taremin_cloth_collider.is_collider = True

        # operator() が常に None を返すモックレイアウトを作成
        class NoneReturningLayout(MockLayout):
            def operator(self, *args, **kwargs):
                return None

        layout_none_main = NoneReturningLayout(
            strict_props=True,
            strict_operators=False,
            strict_menus=False,
        )
        layout_none_col = NoneReturningLayout(
            strict_props=True,
            strict_operators=False,
            strict_menus=False,
        )

        # クラッシュせずに最後まで実行完了できること
        try:
            render_panel(TAREMIN_CLOTH_PT_main_panel, layout=layout_none_main)
            render_panel(TAREMIN_CLOTH_PT_collider_panel, layout=layout_none_col)
        except AttributeError as e:
            self.fail(f"operator() が None を返した際に AttributeError が発生しました: {e}")

    # -------------------------------------------------------------------------
    # テストユーティリティ自身の検証検知能力テスト
    # -------------------------------------------------------------------------

    def test_utility_detects_unregistered_operator(self):
        """テストユーティリティが未登録のオペレーター呼び出しを正しく検知・例外化することのテスト"""
        class BrokenPanel:
            @classmethod
            def draw(cls, self, context):
                self.layout.operator("taremin_cloth.non_existent_operator")

        with self.assertRaises(RuntimeError) as ctx:
            render_panel(BrokenPanel, strict=True)
        self.assertIn("未登録のオペレーター", str(ctx.exception))

    def test_utility_detects_invalid_property(self):
        """テストユーティリティが存在しないプロパティの参照を正しく検知・例外化することのテスト"""
        class BrokenPropPanel:
            @classmethod
            def draw(cls, self, context):
                self.layout.prop(context.active_object.taremin_cloth, "non_existent_property")

        with self.assertRaises(RuntimeError) as ctx:
            render_panel(BrokenPropPanel, strict=True)
        self.assertIn("プロパティ 'non_existent_property' が存在しません", str(ctx.exception))

    def test_utility_detects_invalid_operator_property(self):
        """テストユーティリティがオペレーターに存在しないプロパティへの代入を正しく検知・例外化することのテスト"""
        class BrokenOpPropPanel:
            @classmethod
            def draw(cls, self, context):
                op = self.layout.operator("taremin_cloth.save_preset")
                op.invalid_nonexistent_attribute = "invalid_value"

        with self.assertRaises(RuntimeError) as ctx:
            render_panel(BrokenOpPropPanel, strict=True)
        self.assertIn("プロパティ 'invalid_nonexistent_attribute' は定義されていません", str(ctx.exception))

    def test_clear_cache_operator_selected_and_all(self):
        """clear_cache オペレーターによる個別および全体キャッシュクリアと最新ポーズ記憶の検証"""
        self.obj.taremin_cloth.is_cloth = True
        from taremin_cloth.operators import cache_rest_positions
        cache_rest_positions(self.obj)

        # 頂点を変形させ、変形フラグをセット
        self.obj.data.vertices[0].co.z += 2.0
        # シミュレーション変形フラグを立てる
        self.obj["_taremin_cloth_is_deformed"] = True
        self.assertTrue(self.obj.get("_taremin_cloth_is_deformed", False))

        # apply_rest_shape (対象オブジェクトのみ) 実行
        bpy.context.view_layer.objects.active = self.obj
        res = bpy.ops.taremin_cloth.apply_rest_shape(all_objects=False)
        self.assertEqual(res, {'FINISHED'})

        # is_deformed がリセットされ、新レスト形状がキャッシュされていることを検証
        self.assertFalse(self.obj.get("_taremin_cloth_is_deformed", False))
        self.assertTrue("_taremin_cloth_rest_positions" in self.obj)
        import numpy as np
        new_cached = np.frombuffer(self.obj["_taremin_cloth_rest_positions"], dtype=np.float32)
        self.assertAlmostEqual(new_cached[2], 2.0, places=4, msg="最新の変形位置が新レストポーズになっていること")

        # 全体キャッシュクリア実行
        res_all = bpy.ops.taremin_cloth.apply_rest_shape(all_objects=True)
        self.assertEqual(res_all, {'FINISHED'})

    # -------------------------------------------------------------------------
    # オブジェクト一覧パネル (TAREMIN_CLOTH_PT_objects_panel) および選択オペレーターのテスト
    # -------------------------------------------------------------------------

    def test_objects_panel_empty(self):
        """ClothおよびColliderが未設定状態でのオブジェクト一覧パネル描画テスト"""
        self.obj.taremin_cloth.is_cloth = False
        self.obj.taremin_cloth_collider.is_collider = False

        layout = render_panel(TAREMIN_CLOTH_PT_objects_panel)
        labels = layout.get_labels()

        # ヘッダーに (0) と表示され、空メッセージが表示されていること
        self.assertTrue(any("Cloth Objects (0)" in lbl for lbl in labels))
        self.assertTrue(any("Clothが設定されたオブジェクトはありません" in lbl for lbl in labels))
        self.assertTrue(any("Collider Objects (0)" in lbl for lbl in labels))
        self.assertTrue(any("Colliderが設定されたオブジェクトはありません" in lbl for lbl in labels))

    def test_objects_panel_with_cloth_and_collider(self):
        """ClothおよびColliderが存在する状態でのオブジェクト一覧パネル描画テスト"""
        # self.obj を Cloth に設定
        self.obj.taremin_cloth.is_cloth = True
        self.obj.taremin_cloth.layer_id = 1

        # コライダー用オブジェクトを作成
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        self.created_objects.append(cube)
        cube.taremin_cloth_collider.is_collider = True
        cube.taremin_cloth_collider.collider_type = 'SPHERE'

        # パネルを描画
        layout = render_panel(TAREMIN_CLOTH_PT_objects_panel)
        labels = layout.get_labels()
        operators = layout.get_operators()
        menus = layout.get_menus()
        props = layout.get_props()

        # ヘッダー件数
        self.assertTrue(any("Cloth Objects (1)" in lbl for lbl in labels))
        self.assertTrue(any("Collider Objects (1)" in lbl for lbl in labels))

        # バッジラベル（レイヤーや形状名）
        self.assertTrue(any("L1" in lbl for lbl in labels))
        self.assertTrue(any("Sphere" in lbl for lbl in labels))

        # 選択オペレーターが呼び出されていること
        self.assertIn("taremin_cloth.select_object", operators)
        # 一括操作オペレーターおよびプリセットオペレーター
        self.assertIn("taremin_cloth.batch_simulation_state", operators)
        self.assertIn("taremin_cloth.save_config_preset", operators)
        # 構成プリセットメニュー
        self.assertIn("TAREMIN_CLOTH_MT_config_presets", menus)
        prop_names = [p[1] for p in props]
        self.assertIn("enabled", prop_names)

    def test_select_object_operator(self):
        """select_object オペレーターの選択切り替え・アクティブ化・非表示解除の検証"""
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        self.created_objects.append(cube)

        # 初期状態: cube がアクティブ
        self.assertEqual(bpy.context.active_object, cube)

        # self.obj (Plane) を非表示にしてからオペレーターで選択
        self.obj.hide_set(True)
        self.assertTrue(self.obj.hide_get())

        res = bpy.ops.taremin_cloth.select_object(object_name=self.obj.name)
        self.assertEqual(res, {'FINISHED'})

        # self.obj がアクティブかつ選択されており、非表示が解除されていること
        self.assertEqual(bpy.context.active_object, self.obj)
        self.assertTrue(self.obj.select_get())
        self.assertFalse(self.obj.hide_get())
        # 他のオブジェクト (cube) の選択が解除されていること
        self.assertFalse(cube.select_get())

        # 存在しないオブジェクト名を指定した場合に CANCELLED となること
        res_invalid = bpy.ops.taremin_cloth.select_object(object_name="NonExistentObject_12345")
        self.assertEqual(res_invalid, {'CANCELLED'})


if __name__ == "__main__":
    unittest.main()

