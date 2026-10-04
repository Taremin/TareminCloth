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
        # 平均エッジ長 0.04m (40mm)
        params = compute_self_collision_params(
            avg_edge_len=0.04,
            min_edge_len=0.04,
            num_vertices=500,
            purpose='STANDARD',
        )
        # 厚み: 0.04 * 0.20 = 0.008m (8mm)
        self.assertAlmostEqual(params['thickness'], 0.008, places=5)
        self.assertEqual(params['self_collision_relief_factor'], 0.20)
        self.assertEqual(params['self_collision_max_displacement_ratio'], 0.20)
        self.assertEqual(params['self_collision_max_iterations'], '256')
        self.assertEqual(params['coupled_self_collision_mode'], 'OFF')
        self.assertEqual(params['post_collision_relaxation_iters'], 0)
        self.assertTrue(params['enable_normal_untangling'])

    def test_compute_params_skirt(self):
        """SKIRT（プリーツ・スカート・多重折り）目的における計算パラメータの検証"""
        # スカート用: 密着折り畳みに耐えるよう厚みは15%、FULL_COUPLED、iterations=512
        params = compute_self_collision_params(
            avg_edge_len=0.04,
            min_edge_len=0.04,
            num_vertices=1500,
            purpose='SKIRT',
        )
        # 厚み: 0.04 * 0.15 = 0.006m (6mm)
        self.assertAlmostEqual(params['thickness'], 0.006, places=5)
        self.assertEqual(params['self_collision_relief_factor'], 0.15)
        self.assertEqual(params['self_collision_max_displacement_ratio'], 0.15)
        self.assertEqual(params['self_collision_max_iterations'], '512')
        self.assertEqual(params['coupled_self_collision_mode'], 'FULL_COUPLED')
        self.assertEqual(params['post_collision_relaxation_iters'], 2)

    def test_compute_params_thin(self):
        """THIN（薄手・シルク・フリル）目的における計算パラメータの検証"""
        # 薄手用: 極薄 10%、relief 0.10、マイルドな反発
        params = compute_self_collision_params(
            avg_edge_len=0.03,
            min_edge_len=0.03,
            num_vertices=500,
            purpose='THIN',
        )
        # 厚み: 0.03 * 0.10 = 0.003m (3mm)
        self.assertAlmostEqual(params['thickness'], 0.003, places=5)
        self.assertEqual(params['self_collision_relief_factor'], 0.10)
        self.assertEqual(params['self_collision_max_displacement_ratio'], 0.10)
        self.assertEqual(params['self_collision_max_iterations'], '256')
        self.assertEqual(params['coupled_self_collision_mode'], 'OFF')
        self.assertEqual(params['post_collision_relaxation_iters'], 0)

    def test_thickness_clamp_safety(self):
        """局所的に小エッジがある場合、最小エッジ長の40%に安全クランプされる"""
        # 平均は 0.05m (50mm) だが、小エッジが 0.006m (6mm) の場合
        params = compute_self_collision_params(
            avg_edge_len=0.05,
            min_edge_len=0.006,
            num_vertices=500,
            purpose='STANDARD',
        )
        # 本来 0.05 * 0.20 = 0.010m だが、min_edge_len * 0.4 = 0.0024m (2.4mm) に安全クランプされること
        self.assertAlmostEqual(params['thickness'], 0.0024, places=5)

    def test_outlier_min_edge_protected_by_min_floor(self):
        """襟元等に極小エッジ（0.2mm）があっても、下限フロア（1.5mm）を死守して極薄化（0.12mmバグ）を防止する"""
        params = compute_self_collision_params(
            avg_edge_len=0.0055,  # 平均 5.5mm
            min_edge_len=0.0002,  # 最小 0.2mm (極小外れ値)
            num_vertices=10000,
            purpose='STANDARD',
        )
        # 従来の0.0002 * 0.4 = 0.00008m (0.08mm) に潰れることなく、下限フロア 0.0015m (1.5mm) を死守
        self.assertAlmostEqual(params['thickness'], 0.0015, places=5)

    def test_fit_self_collision_for_object(self):
        """オブジェクトの settings への一括適用を検証"""
        obj = MockClothObject(self.quad_vertices, self.quad_edges)
        obj.taremin_cloth.self_collision_purpose = 'SKIRT'

        result = fit_self_collision_for_object(obj)
        self.assertIsNotNone(result)
        # 0.1m * 0.15 = 0.015m だが、SKIRTの max_cap 0.008m (8mm) にキャップされる
        self.assertAlmostEqual(obj.taremin_cloth.thickness, 0.008, places=5)
        self.assertEqual(obj.taremin_cloth.coupled_self_collision_mode, 'FULL_COUPLED')
        self.assertEqual(obj.taremin_cloth.self_collision_max_iterations, '512')

    def test_percentile_ignores_outlier_edges(self):
        """十分なエッジ数がある場合、下位5%パーセンタイル値により極小外れ値エッジが無視される"""
        # 100本の直列エッジ（通常長 0.01m = 10mm、1本だけ 0.0001m = 0.1mm の外れ値）
        verts = [(i * 0.01, 0.0, 0.0) for i in range(101)]
        # 最後の1頂点だけ極小距離
        verts[100] = (verts[99][0] + 0.0001, 0.0, 0.0)
        edges = [(i, i + 1) for i in range(100)]

        obj = MockClothObject(verts, edges)
        stats = calculate_mesh_edge_stats(obj)
        self.assertIsNotNone(stats)
        avg_len, effective_min, n_verts = stats
        # 最小値 0.0001m ではなく、下位5%パーセンタイル値（0.01m 近傍）が返ること
        self.assertGreater(effective_min, 0.005)

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
