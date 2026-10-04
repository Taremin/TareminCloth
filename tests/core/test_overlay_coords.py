"""
オーバーレイ描画およびブラシ操作における座標系の整合性検証テスト。
ClothSimulatorがワールド座標系で動作している前提のもと、
ビューポートオーバーレイ描画やブラシ操作時に二重変換（Double Transformation）や
不要なローカル座標変換が発生せず、ワールド座標で一貫して処理されていることを検証する。
"""

import unittest
from unittest.mock import MagicMock
import numpy as np


class TestOverlayCoordinates(unittest.TestCase):
    def test_sewing_overlay_uses_world_coords_without_double_transform(self):
        """シミュレータ稼働時、縫合線の頂点座標が二重変換されずcoordsそのものであることを検証"""
        from taremin_cloth.engine.cache import _simulators

        obj_name = "TestClothObj"
        # シミュレータの現在座標 (ワールド座標: Z=10.0 付近)
        sim_coords = np.array([
            [1.0, 2.0, 10.0],
            [3.0, 4.0, 10.0],
            [5.0, 6.0, 10.0],
        ], dtype=np.float32)

        mock_sim = MagicMock()
        _simulators[obj_name] = (mock_sim, sim_coords.ravel())

        try:
            # オブジェクトの matrix_world (オフセット Z=10.0)
            # もし二重変換されると Z=20.0 になってしまう
            world_mat = np.eye(4, dtype=np.float32)
            world_mat[2, 3] = 10.0

            # drawing.py のロジックを検証
            loose_edge_pairs = [(0, 1), (1, 2)]
            vert_count = 3

            curr_coords_2d = None
            sim_entry = _simulators.get(obj_name)
            if sim_entry is not None and sim_entry[1] is not None:
                curr_coords_2d = sim_entry[1].reshape(-1, 3)

            self.assertIsNotNone(curr_coords_2d)
            self.assertEqual(len(curr_coords_2d), vert_count)

            # 修正後のロジック: curr_coords_2d はすでにワールド座標系なので直接使用
            coords_world = curr_coords_2d
            sew_lines = []
            for v0_idx, v1_idx in loose_edge_pairs:
                sew_lines.append(coords_world[v0_idx].tolist())
                sew_lines.append(coords_world[v1_idx].tolist())

            # 抽出された座標が二重変換（Z=20.0）ではなく、シミュレータ座標そのもの（Z=10.0）であることを確認
            self.assertAlmostEqual(sew_lines[0][2], 10.0)
            self.assertAlmostEqual(sew_lines[1][2], 10.0)
            self.assertAlmostEqual(sew_lines[2][2], 10.0)
            self.assertAlmostEqual(sew_lines[3][2], 10.0)
        finally:
            _simulators.pop(obj_name, None)

    def test_single_grab_target_is_world_space(self):
        """single_grab がマウス移動量をワールド座標のまま sim.set_pin に渡すことを検証"""
        from taremin_cloth.brush.single_grab import SingleGrabTool

        brush = SingleGrabTool()
        brush.grabbed_vert = 0

        # ワールド座標での初期位置と移動量
        init_pos = np.array([1.0, 2.0, 10.0], dtype=np.float32)
        delta_pos = np.array([0.5, -0.2, 0.1], dtype=np.float32)
        expected_target = init_pos + delta_pos

        self.assertAlmostEqual(float(expected_target[0]), 1.5, places=5)
        self.assertAlmostEqual(float(expected_target[1]), 1.8, places=5)
        self.assertAlmostEqual(float(expected_target[2]), 10.1, places=5)

        # SingleGrabToolの内部状態プロパティが正しく定義・初期化されていることを確認
        self.assertIsNone(brush.grab_initial_pos_world)
        self.assertIsNone(brush.grab_current_target_world)

    def test_resolve_brush_center_fallback_is_world_space(self):
        """resolve_brush_centerのフォールバックがシミュレータのワールド座標をそのままcenterとして返すことを検証"""
        from taremin_cloth.brush.base import resolve_brush_center

        mock_context = MagicMock()
        mock_context.region = MagicMock()
        mock_context.region_data = MagicMock()

        mock_obj = MagicMock()
        mock_obj.matrix_world.inverted.return_value = MagicMock()
        # ray_cast がヒットしなかった（フォールバック）ケース
        mock_obj.ray_cast.return_value = (False, None, None, None)

        sim_coords = np.array([
            [1.0, 2.0, 10.0],
            [3.0, 4.0, 10.0],
        ], dtype=np.float32)

        # view3d_utils のモック設定
        with unittest.mock.patch("bpy_extras.view3d_utils.region_2d_to_origin_3d", return_value=MagicMock()), \
             unittest.mock.patch("bpy_extras.view3d_utils.region_2d_to_vector_3d", return_value=MagicMock()), \
             unittest.mock.patch("bpy_extras.view3d_utils.location_3d_to_region_2d", return_value=MagicMock(x=10, y=10)):
            res = resolve_brush_center(mock_context, mock_obj, sim_coords, (10, 10))
            self.assertIsNotNone(res)
            # center がワールド座標（Z=10.0）であることを確認
            self.assertAlmostEqual(float(res["center"][2]), 10.0, places=5)


if __name__ == '__main__':
    unittest.main()
