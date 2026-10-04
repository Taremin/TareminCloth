"""
縫合クラスタマップ構築およびGrabツール同期ドラッグの単体テスト
"""
import unittest
from unittest.mock import MagicMock
import numpy as np

from taremin_cloth.utils.mesh_extract import build_seam_cluster_map, WaypointSeam
from taremin_cloth.engine.cache import set_seam_cluster_map, get_seam_cluster_map, clear_simulator_for_object
from taremin_cloth.brush.single_grab import SingleGrabTool
from taremin_cloth.brush.range_grab import RangeGrabTool


class TestSeamClusterGrab(unittest.TestCase):
    def setUp(self):
        clear_simulator_for_object("TestCloth")

    def tearDown(self):
        clear_simulator_for_object("TestCloth")

    def test_build_seam_cluster_map(self):
        """単純縫合および中継点縫合から正しいパートナー関係が構築されること"""
        simple_edges = [(0, 1), (10, 11)]
        waypoint_seams = [
            WaypointSeam(
                vert_a=2,
                vert_b=3,
                waypoint_indices=[100, 101],
                waypoint_coords=np.zeros((2, 3), dtype=np.float32),
            )
        ]

        cluster_map = build_seam_cluster_map(simple_edges, waypoint_seams)

        # 単純縫合
        self.assertEqual(set(cluster_map[0]), {1})
        self.assertEqual(set(cluster_map[1]), {0})
        self.assertEqual(set(cluster_map[10]), {11})

        # 中継点縫合 (2, 3, 100, 101)
        self.assertEqual(set(cluster_map[2]), {3, 100, 101})
        self.assertEqual(set(cluster_map[3]), {2, 100, 101})
        self.assertEqual(set(cluster_map[100]), {2, 3, 101})
        self.assertEqual(set(cluster_map[101]), {2, 3, 100})

    @unittest.mock.patch("taremin_cloth.brush.single_grab.view3d_utils")
    def test_single_grab_cluster_sync(self, mock_v3d):
        """SingleGrabToolで縫合頂点を掴んだ際に対向頂点も同期してピン留め・移動・解放されること"""
        class FakeVec:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x = float(x)
                self.y = float(y)
                self.z = float(z)
            def dot(self, other):
                return self.x * other.x + self.y * other.y + self.z * other.z
            def __neg__(self):
                return FakeVec(-self.x, -self.y, -self.z)
            def normalized(self):
                return self
            def __sub__(self, other):
                return FakeVec(self.x - other.x, self.y - other.y, self.z - other.z)
            def __add__(self, other):
                return FakeVec(self.x + other.x, self.y + other.y, self.z + other.z)
            def __mul__(self, s):
                return FakeVec(self.x * s, self.y * s, self.z * s)
            def __rmul__(self, s):
                return FakeVec(self.x * s, self.y * s, self.z * s)

        mock_v3d.region_2d_to_origin_3d.return_value = FakeVec(0.0, 0.0, 5.0)
        mock_v3d.region_2d_to_vector_3d.return_value = FakeVec(0.0, 0.0, -1.0)

        # 頂点10のスクリーン座標 (マウス位置 100, 100 と完全一致)
        def mock_loc_to_reg(region, rv3d, world_p):
            if hasattr(world_p, "z") and world_p.z == 1.0:
                return FakeVec(100.0, 100.0)
            return FakeVec(9999.0, 9999.0)
        mock_v3d.location_3d_to_region_2d.side_effect = mock_loc_to_reg

        obj_name = "TestCloth"
        cluster_map = {10: [20, 100], 20: [10, 100], 100: [10, 20]}
        set_seam_cluster_map(obj_name, cluster_map)

        tool = SingleGrabTool()

        mock_sim = MagicMock()
        mock_obj = MagicMock()
        mock_obj.name = obj_name
        # ray_cast は外れにして最近傍探索へフォールバック
        mock_obj.ray_cast.return_value = (False, None, None, None)

        coords = np.zeros((150, 3), dtype=np.float32)
        coords[10] = [0.0, 0.0, 1.0]
        coords[20] = [0.0, 0.0, 1.0]
        coords[100] = [0.0, 0.0, 1.0]

        mock_ctx = MagicMock()
        mock_ctx.obj = mock_obj
        mock_ctx.sim = mock_sim
        mock_ctx.coords = coords.flatten()
        mock_ctx.mouse_pos = (100, 100)
        mock_ctx.context.region = MagicMock()
        mock_ctx.context.region_data = MagicMock()
        # camera_forward
        fake_col = MagicMock()
        fake_col.__getitem__.side_effect = lambda i: FakeVec(0.0, 0.0, 1.0)
        mock_ctx.context.region_data.view_matrix.inverted.return_value.to_3x3.return_value.col = fake_col

        # on_press 実行
        with unittest.mock.patch("taremin_cloth.brush.single_grab.mathutils.Vector", side_effect=lambda x: FakeVec(*x) if hasattr(x, '__iter__') else FakeVec()):
            ok = tool.on_press(mock_ctx)
        self.assertTrue(ok)
        self.assertEqual(tool.grabbed_vert, 10)
        self.assertEqual(set(tool.grabbed_cluster), {10, 20, 100})

        # 10, 20, 100 の全頂点に set_pin が呼ばれたこと
        pinned_indices = [call.args[0] for call in mock_sim.set_pin.call_args_list]
        self.assertIn(10, pinned_indices)
        self.assertIn(20, pinned_indices)
        self.assertIn(100, pinned_indices)

        # on_release 実行
        mock_sim.reset_mock()
        tool.on_release(mock_ctx, pinned_verts=set())
        released_indices = [call.args[0] for call in mock_sim.release_pin.call_args_list]
        self.assertIn(10, released_indices)
        self.assertIn(20, released_indices)
        self.assertIn(100, released_indices)

    @unittest.mock.patch("taremin_cloth.brush.range_grab.resolve_brush_center")
    @unittest.mock.patch("taremin_cloth.brush.range_grab.verts_in_brush")
    def test_range_grab_cluster_sync(self, mock_vib, mock_rbc):
        """RangeGrabToolで縫合頂点が選ばれた際、対向頂点・中継点が自動補完されてピン留め・解放されること"""
        obj_name = "TestCloth"
        cluster_map = {10: [20, 100], 20: [10, 100], 100: [10, 20]}
        set_seam_cluster_map(obj_name, cluster_map)

        tool = RangeGrabTool()

        mock_sim = MagicMock()
        mock_obj = MagicMock()
        mock_obj.name = obj_name

        coords = np.zeros((150, 3), dtype=np.float32)
        mock_rbc.return_value = {
            "center": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "origin": np.array([0.0, 0.0, 5.0], dtype=np.float32),
            "direction": np.array([0.0, 0.0, -1.0], dtype=np.float32),
            "hit_t": None,
            "pos_2d": coords,
        }

        # ブラシ範囲内として 10 のみが選ばれたと仮定 (20 と 100 はデプスフィルタや境界外で未選択)
        mock_vib.return_value = (np.array([10], dtype=np.uint32), np.array([0.8], dtype=np.float32))

        mock_ctx = MagicMock()
        mock_ctx.obj = mock_obj
        mock_ctx.sim = mock_sim
        mock_ctx.coords = coords.flatten()
        mock_ctx.mouse_pos = (100, 100)
        mock_ctx.brush = MagicMock()

        ok = tool.on_press(mock_ctx)
        self.assertTrue(ok)

        # 10 だけでなく、補完された 20 と 100 も grab 辞書に含まれること
        self.assertIn(10, tool.grab)
        self.assertIn(20, tool.grab)
        self.assertIn(100, tool.grab)

        # ウェイトが同一であること
        self.assertAlmostEqual(tool.grab[10][1], 0.8)
        self.assertAlmostEqual(tool.grab[20][1], 0.8)
        self.assertAlmostEqual(tool.grab[100][1], 0.8)

        # 全頂点がピン留めされたこと
        pinned_indices = [call.args[0] for call in mock_sim.set_pin.call_args_list]
        self.assertIn(10, pinned_indices)
        self.assertIn(20, pinned_indices)
        self.assertIn(100, pinned_indices)

        # on_release で全頂点が物理解放されること
        mock_sim.reset_mock()
        tool.on_release(mock_ctx, pinned_verts=set())
        released_indices = [call.args[0] for call in mock_sim.release_pin.call_args_list]
        self.assertIn(10, released_indices)
        self.assertIn(20, released_indices)
        self.assertIn(100, released_indices)


if __name__ == "__main__":
    unittest.main()
