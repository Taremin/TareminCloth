"""
インタラクティブモードの頂点ハイライト・ピン留め・クリーンアップ機能のユニットテスト
"""

import unittest
import bpy
import mathutils
import sys
from pathlib import Path

# pythonディレクトリおよびプロジェクトルートをsys.pathに追加
root_dir = Path(__file__).parent.parent
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import taremin_cloth
from taremin_cloth.utils import drawing


class TestInteractivePinning(unittest.TestCase):
    def setUp(self):
        taremin_cloth.register()
        bpy.ops.wm.read_factory_settings(use_empty=True)
        # テスト用メッシュの作成
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 0, 1))
        self.obj = bpy.context.active_object
        self.obj.taremin_cloth.is_cloth = True

    def tearDown(self):
        # クリーンアップ
        drawing.set_interactive_active(False)
        drawing.clear_active_grabbed_vertex()
        if hasattr(bpy.context.workspace, "status_text_set"):
            bpy.context.workspace.status_text_set(None)
        taremin_cloth.unregister()

    def test_pin_properties_initialized(self):
        """ピン表示色およびオーバーレイ制御プロパティが正しく初期化されるかテスト"""
        settings = self.obj.taremin_cloth
        self.assertTrue(hasattr(settings, "pin_color"))
        self.assertEqual(len(settings.pin_color), 4)
        # デフォルト色はオレンジ系 (1.0, 0.45, 0.0, 0.9)
        self.assertAlmostEqual(settings.pin_color[0], 1.0, places=2)
        self.assertAlmostEqual(settings.pin_color[1], 0.45, places=2)

        self.assertTrue(hasattr(settings, "pin_overlay_interactive_only"))
        self.assertTrue(settings.pin_overlay_interactive_only)

        self.assertTrue(hasattr(settings, "elastic_overlay_interactive_only"))
        self.assertTrue(settings.elastic_overlay_interactive_only)

        # 深度テストオプション（デフォルトはFalse）
        self.assertTrue(hasattr(settings, "overlay_depth_test"))
        self.assertFalse(settings.overlay_depth_test)

    def test_overlay_depth_test_toggle(self):
        """overlay_depth_testのトグル切り替えと永続性のテスト"""
        settings = self.obj.taremin_cloth
        self.assertFalse(settings.overlay_depth_test)

        settings.overlay_depth_test = True
        self.assertTrue(settings.overlay_depth_test)

        settings.overlay_depth_test = False
        self.assertFalse(settings.overlay_depth_test)

    def test_drawing_state_management(self):
        """drawingモジュールのインタラクティブ状態と掴んでいる頂点情報の管理テスト"""
        # 初期状態
        drawing.set_interactive_active(False)
        drawing.clear_active_grabbed_vertex()
        self.assertFalse(drawing.is_interactive_active())
        self.assertIsNone(drawing._active_grabbed_info)

        # インタラクティブ開始
        drawing.set_interactive_active(True)
        self.assertTrue(drawing.is_interactive_active())

        # 頂点掴み情報の登録
        test_pos = mathutils.Vector((0.5, 0.5, 1.0))
        drawing.set_active_grabbed_vertex(self.obj.name, 2, test_pos)
        self.assertIsNotNone(drawing._active_grabbed_info)
        self.assertEqual(drawing._active_grabbed_info["obj_name"], self.obj.name)
        self.assertEqual(drawing._active_grabbed_info["vert_idx"], 2)
        self.assertEqual(drawing._active_grabbed_info["target_world_pos"], test_pos)

        # 頂点解放
        drawing.clear_active_grabbed_vertex()
        self.assertIsNone(drawing._active_grabbed_info)

        # インタラクティブ終了
        drawing.set_interactive_active(False)
        self.assertFalse(drawing.is_interactive_active())

    def test_auto_create_pin_vertex_group(self):
        """ピン留め操作時に頂点グループが存在しない場合、自動的に'Pin'グループが作成されるかテスト"""
        settings = self.obj.taremin_cloth
        settings.pin_vertex_group = ""
        self.assertEqual(len(self.obj.vertex_groups), 0)

        # 頂点グループ取得・自動作成ロジックの検証
        vg_name = settings.pin_vertex_group or "Pin"
        vg = self.obj.vertex_groups.get(vg_name)
        if not vg:
            vg = self.obj.vertex_groups.new(name="Pin")
            settings.pin_vertex_group = "Pin"

        self.assertIsNotNone(vg)
        self.assertEqual(vg.name, "Pin")
        self.assertEqual(settings.pin_vertex_group, "Pin")

        # 頂点0をウェイト1.0でピン留め
        vg.add([0], 1.0, 'REPLACE')
        self.assertAlmostEqual(vg.weight(0), 1.0)

        # ピン解除
        vg.remove([0])
        with self.assertRaises(RuntimeError):
            vg.weight(0)

    def test_drawing_overlay_conditional_display(self):
        """インタラクティブモード非実行時にオーバーレイが非表示判定されるかテスト"""
        settings = self.obj.taremin_cloth
        settings.pin_overlay_interactive_only = True
        settings.elastic_overlay_interactive_only = True

        # モード非実行時は描画が無効判定
        drawing.set_interactive_active(False)
        show_pin = not (settings.pin_overlay_interactive_only and not drawing.is_interactive_active())
        self.assertFalse(show_pin)

        # モード実行中は描画が有効判定
        drawing.set_interactive_active(True)
        show_pin = not (settings.pin_overlay_interactive_only and not drawing.is_interactive_active())
        self.assertTrue(show_pin)

    def test_apply_view_depth_bias(self):
        """深度バイアス計算関数のテスト（透視投影・正射影・フォールバック）"""
        coords = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]

        # region_3d が None のときは座標がそのまま返る
        self.assertEqual(drawing.apply_view_depth_bias(coords, None), coords)

        # 透視投影のダミー Region3D
        class DummyPerspR3D:
            view_matrix = mathutils.Matrix.Translation((0.0, -10.0, 5.0))
            is_perspective = True

        r3d_persp = DummyPerspR3D()
        biased_persp = drawing.apply_view_depth_bias(coords, r3d_persp)
        self.assertEqual(len(biased_persp), 2)
        # カメラ位置 (0, 10, -5) に向かって手前にオフセットされているか
        # coords[0] (0, 0, 0) は Y が + 方向に変化するはず
        self.assertGreater(biased_persp[0][1], coords[0][1])

        # 正射影のダミー Region3D
        class DummyOrthoR3D:
            # 視線方向が -Z（ビュー行列単位行列）
            view_matrix = mathutils.Matrix.Identity(4)
            is_perspective = False

        r3d_ortho = DummyOrthoR3D()
        biased_ortho = drawing.apply_view_depth_bias(coords, r3d_ortho)
        self.assertEqual(len(biased_ortho), 2)
        # 視線が -Z なので、カメラ手前方向は +Z となり、Z が増加するはず
        self.assertGreater(biased_ortho[0][2], coords[0][2])


if __name__ == "__main__":
    unittest.main()
