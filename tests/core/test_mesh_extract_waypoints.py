# -*- coding: utf-8 -*-
"""ウェイポイント（中継点）付き縫合エッジ抽出の単体テスト."""
import unittest
import numpy as np

from taremin_cloth.utils.mesh_extract import extract_sewing_topology, WaypointSeam


class TestExtractSewingTopology(unittest.TestCase):
    def setUp(self):
        # 簡易メッシュ:
        # 面 0: (0, 1, 2)
        # 面 1: (3, 4, 5)
        # 頂点 0〜5 が面頂点
        # 頂点 6, 7, 8, 9 が面を持たない孤立頂点 (中継点候補)
        self.faces = np.array([
            [0, 1, 2],
            [3, 4, 5],
        ], dtype=np.uint32)

        self.positions = np.array([
            [-1.0, 0.0, 1.0],  # 0
            [-1.0, 1.0, 1.0],  # 1
            [-1.0, 0.0, 0.0],  # 2
            [ 1.0, 0.0, 1.0],  # 3
            [ 1.0, 1.0, 1.0],  # 4
            [ 1.0, 0.0, 0.0],  # 5
            [ 0.0, 0.0, 0.5],  # 6 (Waypoint 1)
            [ 0.0, 0.0, 0.2],  # 7 (Waypoint 2)
            [ 0.0, 1.0, 0.5],  # 8 (Waypoint 3)
            [ 0.0, 1.0, 0.2],  # 9 (Waypoint 4)
        ], dtype=np.float32)

    def test_direct_simple_sewing(self):
        """中継点のない直線縫合 (0 <-> 3)"""
        loose_edges = [(0, 3)]
        simple, waypoints = extract_sewing_topology(loose_edges, self.faces, self.positions)
        self.assertEqual(len(simple), 1)
        self.assertEqual(simple[0], (0, 3))
        self.assertEqual(len(waypoints), 0)

    def test_single_waypoint_seam(self):
        """1つの中継点を持つ縫合 (0 - 6 - 3)"""
        loose_edges = [(0, 6), (6, 3)]
        simple, waypoints = extract_sewing_topology(loose_edges, self.faces, self.positions)
        self.assertEqual(len(simple), 0)
        self.assertEqual(len(waypoints), 1)

        ws = waypoints[0]
        self.assertTrue((ws.vert_a == 0 and ws.vert_b == 3) or (ws.vert_a == 3 and ws.vert_b == 0))
        self.assertEqual(ws.waypoint_indices, [6])
        np.testing.assert_allclose(ws.waypoint_coords, [[0.0, 0.0, 0.5]])

    def test_multi_waypoint_seam_ordered(self):
        """複数の中継点を持つ縫合 (0 - 6 - 7 - 3)"""
        # エッジの並び順がシャッフルされていても順序付けられること
        loose_edges = [(7, 3), (0, 6), (6, 7)]
        simple, waypoints = extract_sewing_topology(loose_edges, self.faces, self.positions)
        self.assertEqual(len(simple), 0)
        self.assertEqual(len(waypoints), 1)

        ws = waypoints[0]
        if ws.vert_a == 0:
            self.assertEqual(ws.vert_b, 3)
            self.assertEqual(ws.waypoint_indices, [6, 7])
            np.testing.assert_allclose(ws.waypoint_coords, [
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.2],
            ])
        else:
            self.assertEqual(ws.vert_a, 3)
            self.assertEqual(ws.vert_b, 0)
            self.assertEqual(ws.waypoint_indices, [7, 6])
            np.testing.assert_allclose(ws.waypoint_coords, [
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.5],
            ])

    def test_mixed_sewing_and_waypoints(self):
        """直線縫合と中継点付き縫合が混在するケース"""
        loose_edges = [
            (1, 4),               # 直線 (1 <-> 4)
            (0, 6), (6, 7), (7, 3), # 2中継点 (0 - 6 - 7 - 3)
            (2, 8), (8, 5),       # 1中継点 (2 - 8 - 5)
        ]
        simple, waypoints = extract_sewing_topology(loose_edges, self.faces, self.positions)
        self.assertEqual(len(simple), 1)
        self.assertEqual(simple[0], (1, 4))
        self.assertEqual(len(waypoints), 2)

    def test_branching_or_loop_fallback(self):
        """分岐や閉ループなどの不正トポロジーでクラッシュせず安全に除外されること"""
        # T字分岐: 0 - 6 - 3 かつ 6 - 7 (7は行き止まり)
        loose_edges = [(0, 6), (6, 3), (6, 7)]
        simple, waypoints = extract_sewing_topology(loose_edges, self.faces, self.positions)
        # 単一経路保証のため、枝分かれした不正トポロジーはシミュレーションから安全に除外される
        self.assertEqual(len(simple), 0)
        self.assertEqual(len(waypoints), 0)


if __name__ == "__main__":
    unittest.main()
