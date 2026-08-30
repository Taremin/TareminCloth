import unittest
import bpy
import sys
from pathlib import Path

# pythonディレクトリをsys.pathに追加
addon_dir = Path(__file__).parent.parent / "python"
if str(addon_dir) not in sys.path:
    sys.path.insert(0, str(addon_dir))

import taremin_cloth


class TestAddonRegistration(unittest.TestCase):
    def setUp(self):
        # 登録
        taremin_cloth.register()

    def tearDown(self):
        # 登録解除
        taremin_cloth.unregister()

    def test_addon_registration_and_properties(self):
        """アドオン登録後にオブジェクトプロパティとオペレーターが正しく初期化されるかテスト"""
        # メッシュオブジェクトを作成
        bpy.ops.mesh.primitive_plane_add()
        obj = bpy.context.active_object
        self.assertIsNotNone(obj)

        # プロパティの存在確認
        self.assertTrue(hasattr(obj, "taremin_cloth"))
        self.assertFalse(obj.taremin_cloth.is_cloth)
        self.assertEqual(obj.taremin_cloth.stiffness, 1000.0)
        self.assertEqual(obj.taremin_cloth.substeps, 20)

        # オペレーターの実行テスト
        res = bpy.ops.taremin_cloth.toggle_cloth()
        self.assertEqual(res, {'FINISHED'})
        self.assertTrue(obj.taremin_cloth.is_cloth)

        # トグルで元に戻るか
        res = bpy.ops.taremin_cloth.toggle_cloth()
        self.assertEqual(res, {'FINISHED'})
        self.assertFalse(obj.taremin_cloth.is_cloth)

        # オブジェクト削除
        bpy.data.objects.remove(obj, do_unlink=True)


if __name__ == "__main__":
    unittest.main()
