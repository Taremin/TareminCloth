"""
メッシュ縮退面検知および縫合線トポロジー（枝分かれ防止）検証テスト
"""

import unittest
from unittest.mock import MagicMock
import numpy as np

from taremin_cloth.utils.validation import (
    detect_degenerate_faces,
    validate_sewing_topology,
    validate_cloth_mesh,
    clear_validation_cache,
    DegenerateFaceReport,
    SewingTopologyReport,
    BranchingIssue,
)


class TestMeshValidation(unittest.TestCase):
    def setUp(self):
        clear_validation_cache()

    def test_detect_degenerate_faces(self):
        """正常三角形と面積0の縮退三角形を正確に識別できることを検証"""
        # 頂点座標: 0, 1, 2 は正常三角形、3, 4, 5 は同一直線上（面積0）、6, 7, 8 は縮退点（同一座標）
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            # 同一直線
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            # 同一座標
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [6.0, 5.0, 5.0],
        ], dtype=np.float32)

        faces = np.array([
            [0, 1, 2],  # 正常面 (面積 0.5)
            [3, 4, 5],  # 縮退面 (面積 0.0)
            [6, 7, 8],  # 縮退面 (面積 0.0)
        ], dtype=np.int32)

        report = detect_degenerate_faces(positions, faces)
        self.assertTrue(report.has_issues)
        self.assertEqual(report.degenerate_tri_count, 2)
        self.assertEqual(report.degenerate_tri_indices, [1, 2])

    def test_validate_sewing_topology_direct(self):
        """直接縫合（内部頂点なし）の正常系を検証"""
        face_verts = {0, 1, 2, 3}
        # 0 と 1 を結ぶ直接縫合、2 と 3 を結ぶ直接縫合
        loose_edges = [(0, 1), (2, 3)]

        report = validate_sewing_topology(loose_edges, face_verts)
        self.assertFalse(report.has_issues)
        self.assertEqual(report.valid_simple_count, 2)
        self.assertEqual(report.valid_waypoint_count, 0)
        self.assertEqual(len(report.branching_issues), 0)

    def test_validate_sewing_topology_waypoint_path(self):
        """中継点を含む単一経路縫合線の正常系を検証"""
        face_verts = {0, 5}
        # パス: 0 (端点) - 1 (中継) - 2 (中継) - 3 (中継) - 5 (端点)
        loose_edges = [(0, 1), (1, 2), (2, 3), (3, 5)]

        report = validate_sewing_topology(loose_edges, face_verts)
        self.assertFalse(report.has_issues)
        self.assertEqual(report.valid_simple_count, 0)
        self.assertEqual(report.valid_waypoint_count, 1)
        self.assertEqual(len(report.branching_issues), 0)

    def test_validate_sewing_topology_branching_y(self):
        """Y字・T字分岐（次数3以上）が検知されエラーとなることを検証"""
        face_verts = {0, 3, 4}
        # 頂点1で分岐: 0-1, 1-2, 1-3, 2-4 -> 頂点1の次数が3
        loose_edges = [(0, 1), (1, 2), (1, 3), (2, 4)]

        report = validate_sewing_topology(loose_edges, face_verts)
        self.assertTrue(report.has_issues)
        self.assertEqual(len(report.branching_issues), 1)
        issue = report.branching_issues[0]
        self.assertEqual(issue.issue_type, "branching")
        self.assertIn(1, issue.vertices)

    def test_validate_sewing_topology_dangling(self):
        """端点が1つしかなく宙ぶらりん（行き止まり）のエッジが検知されることを検証"""
        face_verts = {0}
        # 0 (端点) - 1 (中継) - 2 (行き止まり)
        loose_edges = [(0, 1), (1, 2)]

        report = validate_sewing_topology(loose_edges, face_verts)
        self.assertTrue(report.has_issues)
        self.assertEqual(len(report.branching_issues), 1)
        self.assertEqual(report.branching_issues[0].issue_type, "dangling")

    def test_validate_sewing_topology_cycle(self):
        """布端点のない閉路（ループ）が検知されることを検証"""
        face_verts = {10, 11}
        # 10, 11 とは無関係な中継点ループ 1-2-3-1
        loose_edges = [(1, 2), (2, 3), (3, 1)]

        report = validate_sewing_topology(loose_edges, face_verts)
        self.assertTrue(report.has_issues)
        self.assertEqual(len(report.branching_issues), 1)
        self.assertEqual(report.branching_issues[0].issue_type, "cycle")

    def test_validate_sewing_topology_multi_endpoints(self):
        """3つ以上の布端点に接続されている不正トポロジーが検知されることを検証"""
        face_verts = {0, 2, 4}
        # 中継点1が 0, 2, 4 の3つの端点に接続
        loose_edges = [(0, 1), (2, 1), (4, 1)]

        report = validate_sewing_topology(loose_edges, face_verts)
        self.assertTrue(report.has_issues)
        # 次数3の分岐として検知される
        self.assertTrue(any(issue.issue_type in ("branching", "multi_endpoints") for issue in report.branching_issues))

    def test_validate_cloth_mesh_mock(self):
        """Blender Mesh モックオブジェクトに対する一括検証とキャッシュ動作を検証"""
        mock_mesh = MagicMock()
        mock_verts = MagicMock()
        mock_verts.__len__.return_value = 4
        coords = np.array([
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            2.0, 2.0, 0.0,  # 独立頂点
        ], dtype=np.float32)

        def mock_foreach_get_vert(attr, arr):
            if attr == "co":
                arr[:] = coords

        mock_verts.foreach_get.side_effect = mock_foreach_get_vert
        mock_mesh.vertices = mock_verts

        # 三角形面: 0-1-2 (正常面)
        mock_tris = MagicMock()
        mock_tris.__len__.return_value = 1
        tri_verts = np.array([0, 1, 2], dtype=np.int32)

        def mock_foreach_get_tri(attr, arr):
            if attr == "vertices":
                arr[:] = tri_verts

        mock_tris.foreach_get.side_effect = mock_foreach_get_tri
        mock_mesh.loop_triangles = mock_tris

        # エッジ: 面エッジ(0,1), (1,2), (2,0) と ルーズエッジ(0, 3) (宙ぶらりん)
        e0 = MagicMock(); e0.vertices = (0, 1)
        e1 = MagicMock(); e1.vertices = (1, 2)
        e2 = MagicMock(); e2.vertices = (2, 0)
        e3 = MagicMock(); e3.vertices = (0, 3)  # ルーズエッジ (端点0のみ)
        mock_mesh.edges = [e0, e1, e2, e3]

        poly0 = MagicMock()
        poly0.area = 0.5
        poly0.index = 0
        mock_mesh.polygons = [poly0]

        mock_obj = MagicMock()
        mock_obj.type = 'MESH'
        mock_obj.name = "ClothObject"
        mock_obj.data = mock_mesh

        report = validate_cloth_mesh(mock_obj)
        self.assertIsNotNone(report)
        self.assertFalse(report.has_warnings)  # 縮退面なし
        self.assertTrue(report.has_errors)    # 宙ぶらりんエッジあり (dangling)

        # キャッシュヒット確認（同一引数での再取得）
        report2 = validate_cloth_mesh(mock_obj)
        self.assertIs(report, report2)


if __name__ == "__main__":
    unittest.main()
