import logging
import unittest
import sys
from pathlib import Path
import bpy

root_dir = Path(__file__).parent.parent
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import taremin_cloth
from taremin_cloth.preferences import (
    get_preferences,
    TareminClothPreferences,
    get_preferences_filepath,
    save_preferences_to_disk,
    load_preferences_from_disk,
    apply_saved_preferences,
)
from taremin_cloth.utils.logger import logger, set_log_level
from taremin_cloth.operators import cache_rest_positions, restore_rest_positions


class TestLoggingAndPreferences(unittest.TestCase):

    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        taremin_cloth.register()
        if "taremin_cloth" not in bpy.context.preferences.addons:
            addon = bpy.context.preferences.addons.new()
            addon.module = "taremin_cloth"
        # 初期状態として一度設定ファイルをクリーンアップ
        self._clean_prefs_file()

    def tearDown(self):
        self._clean_prefs_file()

    def _clean_prefs_file(self):
        try:
            fp = get_preferences_filepath()
            if fp.is_file():
                fp.unlink()
        except Exception:
            pass

    def test_logger_set_log_level(self):
        """ロガーのログレベル動的変更テスト"""
        set_log_level('DEBUG')
        self.assertEqual(logger.level, logging.DEBUG)

        set_log_level('WARNING')
        self.assertEqual(logger.level, logging.WARNING)

        set_log_level('INFO')
        self.assertEqual(logger.level, logging.INFO)

    def test_preferences_registered_and_update(self):
        """アドオンプレファレンスの取得およびログレベル更新テスト"""
        prefs = get_preferences()
        self.assertIsNotNone(prefs)
        prefs.log_level = 'DEBUG'
        self.assertEqual(logger.level, logging.DEBUG)
        prefs.log_level = 'INFO'
        self.assertEqual(logger.level, logging.INFO)

    def test_preferences_disk_save_and_load(self):
        """設定のJSONディスク保存・読み込み・適用の単体テスト"""
        prefs = get_preferences()
        self.assertIsNotNone(prefs)

        # 1. プレファレンス変更による自動ディスク保存の確認
        prefs.log_level = 'DEBUG'
        prefs.gpu_backend = 'VULKAN'

        data = load_preferences_from_disk()
        self.assertEqual(data.get("log_level"), "DEBUG")
        self.assertEqual(data.get("gpu_backend"), "VULKAN")

        # 2. ディスク上の設定ファイルを書き換えて apply_saved_preferences で復元されるかを検証
        import json
        fp = get_preferences_filepath()
        with open(fp, "w", encoding="utf-8") as f:
            json.dump({"log_level": "WARNING", "gpu_backend": "DX12", "gpu_device": "AUTO"}, f)

        self.assertTrue(apply_saved_preferences(prefs))
        self.assertEqual(prefs.log_level, 'WARNING')
        self.assertEqual(prefs.gpu_backend, 'DX12')
        self.assertEqual(logger.level, logging.WARNING)

    def test_preferences_persistence_across_addon_toggle(self):
        """アドオン無効化→有効化サイクルを挟んでも設定が維持されるかのテスト"""
        prefs = get_preferences()
        self.assertIsNotNone(prefs)

        # ログレベルとGPU設定を変更（自動でsave_preferences_to_diskが呼ばれる）
        prefs.log_level = 'DEBUG'
        prefs.gpu_backend = 'VULKAN'

        # アドオン無効化（unregister）
        taremin_cloth.unregister()

        # アドオン再有効化（register）
        taremin_cloth.register()
        if "taremin_cloth" not in bpy.context.preferences.addons:
            addon = bpy.context.preferences.addons.new()
            addon.module = "taremin_cloth"

        # 再取得したプレファレンスで値が維持されていることを検証
        prefs_reloaded = get_preferences()
        self.assertIsNotNone(prefs_reloaded)
        self.assertEqual(prefs_reloaded.log_level, 'DEBUG')
        self.assertEqual(prefs_reloaded.gpu_backend, 'VULKAN')
        self.assertEqual(logger.level, logging.DEBUG)

    def test_preferences_gpu_settings_and_operator(self):
        """プレファレンスのGPUバックエンド・デバイス設定および適用オペレーターのテスト"""
        prefs = get_preferences()
        self.assertIsNotNone(prefs)
        self.assertTrue(hasattr(prefs, "gpu_backend"))
        self.assertTrue(hasattr(prefs, "gpu_device"))
        self.assertTrue(hasattr(prefs, "active_device_name"))
        self.assertTrue(hasattr(prefs, "active_backend_name"))

        # DX12に変更して適用
        prefs.gpu_backend = 'DX12'
        self.assertEqual(prefs.gpu_backend, 'DX12')
        self.assertEqual(prefs.gpu_device, 'AUTO')

        res = bpy.ops.taremin_cloth.apply_gpu_settings()
        self.assertEqual(res, {'FINISHED'})
        self.assertIn("DirectX", prefs.active_backend_name)

        # VULKANに変更して適用（Backend変更時にDeviceがAUTOに安全リセットされること）
        prefs.gpu_backend = 'VULKAN'
        self.assertEqual(prefs.gpu_backend, 'VULKAN')
        self.assertEqual(prefs.gpu_device, 'AUTO')

        res = bpy.ops.taremin_cloth.apply_gpu_settings()
        self.assertEqual(res, {'FINISHED'})
        self.assertEqual(prefs.active_backend_name, "Vulkan")

    def test_logging_during_cache_and_restore(self):
        """キャッシュおよびリセット処理実行時にロガーが正常に呼び出されるかのテスト"""
        set_log_level('DEBUG')

        bpy.ops.mesh.primitive_plane_add(size=1.0)
        obj = bpy.context.active_object
        obj.taremin_cloth.is_cloth = True

        # キャッシュの実行
        with self.assertLogs(logger="taremin_cloth", level="DEBUG") as log:
            cache_rest_positions(obj, force=True)
            self.assertTrue(any("[Cache]" in msg for msg in log.output))

        # リセットの実行
        with self.assertLogs(logger="taremin_cloth", level="INFO") as log:
            restore_rest_positions(obj)
            self.assertTrue(any("[Reset]" in msg for msg in log.output))


if __name__ == "__main__":
    unittest.main()
