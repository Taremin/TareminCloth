"""
taremin_cloth タイムラインキャッシュ管理・変更検知の単体テスト
"""

import os
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np

# プロジェクトルートおよび python/ ディレクトリをモジュール検索パスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from taremin_cloth.engine import cache


class TestTimelineCache(unittest.TestCase):
    """タイムラインキャッシュのシグネチャ生成、変更検知、およびキャッシュ情報取得の単体テスト"""

    def setUp(self):
        cache.clear_simulators()

    def tearDown(self):
        cache.clear_simulators()

    def _create_mock_cloth_obj(self, name="TestCloth"):
        obj = MagicMock()
        obj.name = name
        obj.type = 'MESH'

        settings = MagicMock()
        settings.is_cloth = True
        settings.tension_stiffness = 1000.0
        settings.compression_stiffness = 1000.0
        settings.shear_stiffness = 1000.0
        settings.bending_stiffness = 10.0
        settings.air_damping = 0.05
        settings.tension_damping = 0.1
        settings.compression_damping = 0.1
        settings.shear_damping = 0.1
        settings.bending_damping = 0.1
        settings.gravity = 1.0
        settings.pin_vertex_group = "Pin"
        settings.pin_target_object = None
        settings.pin_target_bone = ""
        settings.enable_sewing = False
        settings.sewing_shrink_speed = 1.0
        settings.sewing_stiffness = 10000.0
        settings.enable_sewing_lock = True
        settings.thickness = 0.005
        settings.enable_self_collision = False
        settings.self_collision_purpose = "STANDARD"
        settings.substeps = 10
        settings.solver_iterations = 2
        settings.solver_mode = "COLORING"
        settings.elastic_groups = []

        obj.taremin_cloth = settings
        return obj

    def test_get_cloth_params_signature(self):
        """パラメータシグネチャが正常に生成されること"""
        obj = self._create_mock_cloth_obj()
        sig = cache.get_cloth_params_signature(obj)
        self.assertIsNotNone(sig)
        self.assertIsInstance(sig, tuple)

        # 非Clothオブジェクトの場合はNone
        obj.taremin_cloth.is_cloth = False
        self.assertIsNone(cache.get_cloth_params_signature(obj))

    def test_check_and_update_params_signature(self):
        """パラメータ変更時に正しくDirty検知してキャッシュを破棄すること"""
        obj = self._create_mock_cloth_obj("ClothA")

        # 初回呼び出し: 前回のシグネチャが存在しないためFalse
        res_initial = cache.check_and_update_params_signature(obj)
        self.assertFalse(res_initial)

        # パラメータ未変更で再呼び出し: False
        res_same = cache.check_and_update_params_signature(obj)
        self.assertFalse(res_same)

        # キャッシュにダミーデータを注入
        cache._timeline_frame_cache["ClothA"] = {
            1: np.zeros(12, dtype=np.float32),
            2: np.ones(12, dtype=np.float32),
        }
        self.assertIn("ClothA", cache._timeline_frame_cache)

        # 剛性を変更
        obj.taremin_cloth.tension_stiffness = 2500.0
        res_changed = cache.check_and_update_params_signature(obj)
        self.assertTrue(res_changed, "パラメータ変更時はTrueが返ること")

        # キャッシュが自動破棄されていること
        self.assertNotIn("ClothA", cache._timeline_frame_cache)

    def test_get_timeline_cache_info(self):
        """キャッシュ情報の正確な取得（空状態・複数フレーム・容量計算）"""
        obj_name = "ClothTest"

        # 1. 空状態
        info_empty = cache.get_timeline_cache_info(obj_name)
        self.assertEqual(info_empty["count"], 0)
        self.assertEqual(info_empty["range_str"], "Empty")
        self.assertEqual(info_empty["size_kb"], 0.0)

        # 2. 1フレームのみ
        cache._timeline_frame_cache[obj_name] = {
            1: np.zeros(300, dtype=np.float32)  # 300 floats = 1200 bytes
        }
        info_1 = cache.get_timeline_cache_info(obj_name)
        self.assertEqual(info_1["count"], 1)
        self.assertEqual(info_1["range_str"], "1")
        self.assertAlmostEqual(info_1["size_kb"], 1200.0 / 1024.0, places=2)

        # 3. 複数フレーム (1〜10)
        for f in range(2, 11):
            cache._timeline_frame_cache[obj_name][f] = np.zeros(300, dtype=np.float32)
        info_10 = cache.get_timeline_cache_info(obj_name)
        self.assertEqual(info_10["count"], 10)
        self.assertEqual(info_10["range_str"], "1 - 10")
        self.assertAlmostEqual(info_10["size_kb"], (1200.0 * 10) / 1024.0, places=2)

    def test_clear_timeline_cache(self):
        """clear_timeline_cache の動作検証"""
        cache._timeline_frame_cache["ObjA"] = {1: np.zeros(3)}
        cache._timeline_frame_cache["ObjB"] = {1: np.zeros(3)}

        # 特定オブジェクトのみクリア
        cache.clear_timeline_cache("ObjA")
        self.assertNotIn("ObjA", cache._timeline_frame_cache)
        self.assertIn("ObjB", cache._timeline_frame_cache)

        # 全クリア
        cache.clear_timeline_cache()
        self.assertEqual(len(cache._timeline_frame_cache), 0)

    def test_bake_state_and_free(self):
        """is_object_baked, is_scene_baked, free_object_bake, free_scene_bake の検証"""
        obj = self._create_mock_cloth_obj("ClothBake")
        scene = MagicMock()
        scene.objects = [obj]

        # 1. 未ベイク状態
        self.assertFalse(cache.is_object_baked(obj))
        self.assertFalse(cache.is_scene_baked(scene))

        # 1フレームのみ（初期フレーム）では未ベイク判定
        cache._timeline_frame_cache["ClothBake"] = {1: np.zeros(12, dtype=np.float32)}
        self.assertFalse(cache.is_object_baked(obj))
        self.assertFalse(cache.is_scene_baked(scene))

        # 2フレーム以上でベイク済み判定
        cache._timeline_frame_cache["ClothBake"][2] = np.ones(12, dtype=np.float32)
        self.assertTrue(cache.is_object_baked(obj))
        self.assertTrue(cache.is_scene_baked(scene))

        # 2. free_object_bake の実行
        cache.free_object_bake(obj)
        self.assertFalse(cache.is_object_baked(obj))
        self.assertNotIn("ClothBake", cache._timeline_frame_cache)

        # 再度ベイクデータを注入して free_scene_bake
        cache._timeline_frame_cache["ClothBake"] = {1: np.zeros(12), 2: np.ones(12)}
        self.assertTrue(cache.is_scene_baked(scene))
        cache.free_scene_bake(scene)
        self.assertFalse(cache.is_scene_baked(scene))
        self.assertEqual(len(cache._timeline_frame_cache), 0)


if __name__ == "__main__":
    unittest.main()

