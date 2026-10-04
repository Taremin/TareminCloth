import unittest
import numpy as np
from taremin_cloth.engine.runner import (
    get_seam_merge_threshold,
    get_connected_seam_partners,
)
from taremin_cloth.engine.cache import set_seam_cluster_map, clear_simulators


class MockSettings:
    def __init__(self, lock_dist=0.005, thickness=0.005):
        self.sewing_lock_distance = lock_dist
        self.thickness = thickness


class MockObj:
    def __init__(self, name="TestCloth", lock_dist=0.005, thickness=0.005):
        self.name = name
        self.taremin_cloth = MockSettings(lock_dist, thickness)


class TestGrabSeamPartners(unittest.TestCase):
    def setUp(self):
        clear_simulators()

    def tearDown(self):
        clear_simulators()

    def test_get_seam_merge_threshold(self):
        # 1. 設定あり（デフォルト 0.005）: 最小閾値 0.02
        obj = MockObj(lock_dist=0.005, thickness=0.005)
        thresh = get_seam_merge_threshold(obj)
        self.assertAlmostEqual(thresh, 0.02)

        # 2. 設定あり（大きめの値 0.03）: max(0.06, 0.01, 0.02) = 0.06
        obj_large = MockObj(lock_dist=0.03, thickness=0.005)
        thresh_large = get_seam_merge_threshold(obj_large)
        self.assertAlmostEqual(thresh_large, 0.06)

        # 3. オブジェクトが設定を持たない場合
        self.assertAlmostEqual(get_seam_merge_threshold(object()), 0.02)

    def test_get_connected_seam_partners(self):
        obj_name = "ClothA"
        # 頂点0と1が縫合ペア、頂点2と3が縫合ペア
        cluster_map = {
            0: [1],
            1: [0],
            2: [3],
            3: [2],
        }
        set_seam_cluster_map(obj_name, cluster_map)

        # 座標:
        # 頂点0: (0.0, 0.0, 0.0)
        # 頂点1 (離れている未結合エッジ): (0.2, 0.0, 0.0) -> 距離 0.2m
        # 頂点2: (0.5, 0.0, 0.0)
        # 頂点3 (密着している結合済みシーム): (0.502, 0.0, 0.0) -> 距離 0.002m
        coords = np.array([
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.502, 0.0, 0.0],
        ], dtype=np.float32)

        threshold = 0.02

        # 頂点0 (未結合): パートナー1は離れているため返されない
        partners_0 = get_connected_seam_partners(obj_name, 0, coords, threshold)
        self.assertEqual(partners_0, [], "離れている未結合エッジは同期対象から除外されること")

        # 頂点1 (未結合): 同様にパートナー0は返されない
        partners_1 = get_connected_seam_partners(obj_name, 1, coords, threshold)
        self.assertEqual(partners_1, [], "離れている未結合エッジは同期対象から除外されること")

        # 頂点2 (結合済み): パートナー3は 0.002m <= 0.02m なので同期対象として返される
        partners_2 = get_connected_seam_partners(obj_name, 2, coords, threshold)
        self.assertEqual(partners_2, [3], "密着している結合済みシームは同期対象として返されること")

        # 頂点3 (結合済み): パートナー2が返される
        partners_3 = get_connected_seam_partners(obj_name, 3, coords, threshold)
        self.assertEqual(partners_3, [2], "密着している結合済みシームは同期対象として返されること")


if __name__ == "__main__":
    unittest.main()
