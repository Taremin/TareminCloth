"""
自己衝突パラメータ自動フィッティングモジュール (self_collision_fit.py) の単体テスト
Blender非依存のモックオブジェクトを用いて、エッジ長・頂点密度・用途に基づく
自己衝突パラメータの自動算出および設定適用ロジックを検証する。
"""

import os
import sys
import unittest
from types import SimpleNamespace

# プロジェクトルートパスの設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)

from taremin_cloth.utils.self_collision_fit import (
    calculate_mesh_edge_stats,
    compute_self_collision_params,
    fit_self_collision_for_object,
)


class MockEdge:
    def __init__(self, v0: int, v1: int):
        self.vertices = (v0, v1)


class MockVertex:
    def __init__(self, x: float, y: float, z: float):
        self.co = (x, y, z)


class MockMesh:
    def __init__(self, vertices, edges):
        self.vertices = [MockVertex(*co) for co in vertices]
        self.edges = [MockEdge(e[0], e[1]) for e in edges]


class MockClothObject:
    def __init__(self, vertices, edges, matrix_world=None):
        self.type = 'MESH'
        self.data = MockMesh(vertices, edges)
        self.matrix_world = matrix_world
        self.taremin_cloth = SimpleNamespace(
            enable_self_collision=True,
            self_collision_purpose='STANDARD',
            thickness=0.005,
            self_collision_relief_factor=0.2,
            self_collision_max_displacement_ratio=0.2,
            self_collision_max_iterations='256',
            coupled_self_collision_mode='RELAXATION',
            post_collision_relaxation_iters=2,
            enable_normal_untangling=True,
        )


class TestSelfCollisionFit(unittest.TestCase):
    """自己衝突自動フィッティング機能の単体テスト"""

    def setUp(self):
        # 1辺 0.1m (10cm) の正方形クアッドメッシュ (4頂点、4エッジ)
        # (0,0,0) - (0.1, 0, 0) - (0.1, 0.1, 0) - (0, 0.1, 0)
        self.quad_vertices = [
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.1, 0.1, 0.0),
            (0.0, 0.1, 0.0),
        ]
        self.quad_edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
        ]

    def test_calculate_mesh_edge_stats(self):
        """エッジ統計量（平均長、最小長、頂点数）の正確な算出を検証"""
        obj = MockClothObject(self.quad_vertices, self.quad_edges)
        stats = calculate_mesh_edge_stats(obj)
        self.assertIsNotNone(stats)
        avg_len, min_len, n_verts = stats
        self.assertAlmostEqual(avg_len, 0.1, places=5)
        self.assertAlmostEqual(min_len, 0.1, places=5)
        self.assertEqual(n_verts, 4)

    def test_compute_params_standard(self):
        """STANDARD（一般衣服）目的における計算パラメータの検証"""
        # 平均エッジ長 0.1m (100mm)
        params = compute_self_collision_params(
            avg_edge_len=0.1,
            min_edge_len=0.1,
            num_vertices=500,
            purpose='STANDARD',
        )
        # 厚み: 0.1 * 0.05 = 0.005m (5mm)
        self.assertAlmostEqual(params['thickness'], 0.005, places=5)
        self.assertEqual(params['self_collision_relief_factor'], 0.20)
        self.assertEqual(params['self_collision_max_displacement_ratio'], 0.20)
        self.assertEqual(params['self_collision_max_iterations'], '256')
        self.assertEqual(params['coupled_self_collision_mode'], 'RELAXATION')
        self.assertEqual(params['post_collision_relaxation_iters'], 2)
        self.assertTrue(params['enable_normal_untangling'])

    def test_compute_params_skirt(self):
        """SKIRT（プリーツ・スカート・多重折り）目的における計算パラメータの検証"""
        # スカート用: 密着折り畳みに耐えるよう厚みは3.5%、FULL_COUPLED、iterations=512
        params = compute_self_collision_params(
            avg_edge_len=0.1,
            min_edge_len=0.1,
            num_vertices=1500,
            purpose='SKIRT',
        )
        # 厚み: 0.1 * 0.035 = 0.0035m (3.5mm)
        self.assertAlmostEqual(params['thickness'], 0.0035, places=5)
        self.assertEqual(params['self_collision_relief_factor'], 0.15)
        self.assertEqual(params['self_collision_max_displacement_ratio'], 0.15)
        self.assertEqual(params['self_collision_max_iterations'], '512')
        self.assertEqual(params['coupled_self_collision_mode'], 'FULL_COUPLED')
        self.assertEqual(params['post_collision_relaxation_iters'], 2)

    def test_compute_params_thin(self):
        """THIN（薄手・シルク・フリル）目的における計算パラメータの検証"""
        # 薄手用: 極薄 2.5%、relief 0.10、マイルドな反発
        params = compute_self_collision_params(
            avg_edge_len=0.1,
            min_edge_len=0.1,
            num_vertices=500,
            purpose='THIN',
        )
        # 厚み: 0.1 * 0.025 = 0.0025m (2.5mm)
        self.assertAlmostEqual(params['thickness'], 0.0025, places=5)
        self.assertEqual(params['self_collision_relief_factor'], 0.10)
        self.assertEqual(params['self_collision_max_displacement_ratio'], 0.10)
        self.assertEqual(params['self_collision_max_iterations'], '256')
        self.assertEqual(params['coupled_self_collision_mode'], 'RELAXATION')

    def test_thickness_clamp_safety(self):
        """局所的に極小エッジがある場合、最小エッジ長の40%に厚みがクランプされ自発爆発を防ぐ"""
        # 平均は 0.1m だが、最小エッジが 0.005m (5mm) の場合
        params = compute_self_collision_params(
            avg_edge_len=0.1,
            min_edge_len=0.005,
            num_vertices=500,
            purpose='STANDARD',
        )
        # 本来 0.1 * 0.05 = 0.005 だが、min_edge_len * 0.4 = 0.002m に安全クランプされること
        self.assertAlmostEqual(params['thickness'], 0.002, places=5)

    def test_fit_self_collision_for_object(self):
        """オブジェクトの settings への一括適用を検証"""
        obj = MockClothObject(self.quad_vertices, self.quad_edges)
        obj.taremin_cloth.self_collision_purpose = 'SKIRT'

        result = fit_self_collision_for_object(obj)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(obj.taremin_cloth.thickness, 0.0035, places=5)
        self.assertEqual(obj.taremin_cloth.coupled_self_collision_mode, 'FULL_COUPLED')
        self.assertEqual(obj.taremin_cloth.self_collision_max_iterations, '512')

    def test_fit_custom_purpose_noop(self):
        """CUSTOM 目的の場合は手動値を上書きせず None を返す"""
        obj = MockClothObject(self.quad_vertices, self.quad_edges)
        obj.taremin_cloth.self_collision_purpose = 'CUSTOM'
        obj.taremin_cloth.thickness = 0.042

        result = fit_self_collision_for_object(obj)
        self.assertIsNone(result)
        self.assertEqual(obj.taremin_cloth.thickness, 0.042)

    def test_invalid_object_handling(self):
        """メッシュがない、またはエッジがないオブジェクトでも例外を起こさず安全に処理される"""
        self.assertIsNone(fit_self_collision_for_object(None))
        empty_obj = SimpleNamespace(type='EMPTY')
        self.assertIsNone(fit_self_collision_for_object(empty_obj))


if __name__ == '__main__':
    unittest.main()
