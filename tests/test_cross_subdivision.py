import unittest
import bpy
import numpy as np
import sys
from pathlib import Path

# pythonディレクトリおよびプロジェクトルートをsys.pathに追加
root_dir = Path(__file__).parent.parent
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import taremin_cloth
from taremin_cloth.utils import topology
from taremin_cloth.panels import TAREMIN_CLOTH_PT_main_panel
from tests.panel_test_utils import render_panel


class TestCrossSubdivision(unittest.TestCase):
    def setUp(self):
        taremin_cloth.register()
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def tearDown(self):
        taremin_cloth.unregister()
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def test_apply_cross_subdivision_and_restore_quad(self):
        """十字分割の適用とQuad復元のライフサイクル検証"""
        # 単一Quad平面 (4頂点, 1面)
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        obj = bpy.context.active_object
        self.assertIsNotNone(obj)

        self.assertEqual(len(obj.data.vertices), 4)
        self.assertEqual(len(obj.data.polygons), 1)
        self.assertEqual(len(obj.data.polygons[0].vertices), 4)

        # 十字分割を適用
        success = topology.apply_cross_subdivision(obj)
        self.assertTrue(success)
        self.assertTrue(topology.is_cross_subdivided(obj))

        # 頂点数は 4 + 1 = 5、面数は 1 * 4 = 4 (すべて三角形)
        self.assertEqual(len(obj.data.vertices), 5)
        self.assertEqual(len(obj.data.polygons), 4)
        for poly in obj.data.polygons:
            self.assertEqual(len(poly.vertices), 3)

        # Quad復元 (Restore Quad) を実行
        res = topology.apply_post_process(obj, mode='QUAD')
        self.assertTrue(res)
        self.assertFalse(topology.is_cross_subdivided(obj))

        # 元の4頂点、1つのQuadに戻る
        self.assertEqual(len(obj.data.vertices), 4)
        self.assertEqual(len(obj.data.polygons), 1)
        self.assertEqual(len(obj.data.polygons[0].vertices), 4)

    def test_optimal_diagonal_selection(self):
        """最適対角線の幾何判定ロジック検証"""
        # p0-p2 間の山折り: p1とp3が下に下がり、p0, pc, p2 が稜線上に残る
        p0_fold = np.array([0.0, 0.0, 1.0])
        p1_fold = np.array([1.0, 0.0, 0.0])
        p2_fold = np.array([1.0, 1.0, 1.0])
        p3_fold = np.array([0.0, 1.0, 0.0])
        pc_fold = np.array([0.5, 0.5, 1.0])  # 稜線上の中心

        # (p0, p2) の中点は (0.5, 0.5, 1.0) -> pc との距離 0.0
        # (p1, p3) の中点は (0.5, 0.5, 0.0) -> pc との距離 1.0
        use_02 = topology.evaluate_optimal_diagonal(p0_fold, p1_fold, p2_fold, p3_fold, pc_fold)
        self.assertTrue(use_02, "p0-p2の稜線に沿う対角線が選ばれること")

        # 逆に p1-p3 間の山折り:
        p0_fold2 = np.array([0.0, 0.0, 0.0])
        p1_fold2 = np.array([1.0, 0.0, 1.0])
        p2_fold2 = np.array([1.0, 1.0, 0.0])
        p3_fold2 = np.array([0.0, 1.0, 1.0])
        pc_fold2 = np.array([0.5, 0.5, 1.0])  # 稜線上の中心
        use_02_rev = topology.evaluate_optimal_diagonal(p0_fold2, p1_fold2, p2_fold2, p3_fold2, pc_fold2)
        self.assertFalse(use_02_rev, "p1-p3の稜線に沿う対角線が選ばれること")

    def test_post_process_optimal_triangles(self):
        """最適2三角面へのトポロジー変換検証"""
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        obj = bpy.context.active_object
        self.assertEqual(len(obj.data.polygons), 1)  # 1つのQuad

        topology.apply_cross_subdivision(obj)
        self.assertEqual(len(obj.data.vertices), 5)
        self.assertEqual(len(obj.data.polygons), 4)

        # 最適対角線2三角面への後処理を実行
        res = topology.apply_post_process(obj, mode='OPTIMAL_TRI')
        self.assertTrue(res)

        # 頂点数は元の4頂点に戻り、面数は2つの三角形になる
        self.assertEqual(len(obj.data.vertices), 4)
        self.assertEqual(len(obj.data.polygons), 2)
        for poly in obj.data.polygons:
            self.assertEqual(len(poly.vertices), 3)

    def test_operators_and_panel_ui(self):
        """オペレーター実行およびUI描画の検証"""
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        obj = bpy.context.active_object
        obj.taremin_cloth.is_cloth = True
        obj.taremin_cloth.enable_cross_subdivision = True

        # オペレーターで十字分割を適用
        res = bpy.ops.taremin_cloth.apply_cross_subdivision()
        self.assertEqual(res, {'FINISHED'})
        self.assertTrue(topology.is_cross_subdivided(obj))

        # パネルがエラーなく描画されること
        layout = render_panel(TAREMIN_CLOTH_PT_main_panel)
        labels = layout.get_labels()
        operators = layout.get_operators()
        self.assertTrue(any("Topology & Cross-Subdivision" in lbl for lbl in labels))
        self.assertIn("taremin_cloth.apply_post_process", operators)
        self.assertIn("taremin_cloth.restore_quad_topology", operators)

        # 後処理オペレーター（OPTIMAL_TRI）を実行
        res = bpy.ops.taremin_cloth.apply_post_process()
        self.assertEqual(res, {'FINISHED'})
        self.assertFalse(topology.is_cross_subdivided(obj))
        self.assertEqual(len(obj.data.polygons), 2)  # 2つの三角形になった

        # 新規Planeで十字分割 -> Quad復元オペレーターを検証
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        obj2 = bpy.context.active_object
        obj2.taremin_cloth.is_cloth = True
        obj2.taremin_cloth.enable_cross_subdivision = True

        bpy.ops.taremin_cloth.apply_cross_subdivision()
        self.assertTrue(topology.is_cross_subdivided(obj2))
        res = bpy.ops.taremin_cloth.restore_quad_topology()
        self.assertEqual(res, {'FINISHED'})
        self.assertFalse(topology.is_cross_subdivided(obj2))
        self.assertEqual(len(obj2.data.vertices), 4)
        self.assertEqual(len(obj2.data.polygons), 1)

    def test_post_process_adaptive_three_way(self):
        """アダプティブ後処理の3分岐検証:
        1. 平坦部 -> Quad (1面, 4頂点) に復元
        2. 対角線シワ -> 最適三角2面 (2面, 3頂点) に分割
        3. 中心盛り上がり (ドーム/テント) -> 十字分割維持 (4面, 3頂点)
        """
        # --- ケース1: 平坦部 (Flat) ---
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        obj_flat = bpy.context.active_object
        topology.apply_cross_subdivision(obj_flat)
        self.assertEqual(len(obj_flat.data.polygons), 4)

        topology.apply_post_process(obj_flat, mode='ADAPTIVE', flatness_threshold=5.0)
        self.assertEqual(len(obj_flat.data.polygons), 1, "平坦な面はQuad(1面)に復元されること")
        self.assertEqual(len(obj_flat.data.polygons[0].vertices), 4)

        # --- ケース2: 対角線シワ (Ridge) ---
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        obj_ridge = bpy.context.active_object
        topology.apply_cross_subdivision(obj_ridge)
        subdiv_map = obj_ridge["_taremin_cross_subdiv_map"]
        v0_idx, v1_idx, v2_idx, v3_idx = subdiv_map[0]["quad_verts"]
        c_idx = subdiv_map[0]["center_vert_index"]
        # v0, v2, pc を持ち上げて対角線上にシワを形成（v1, v3はそのまま）
        obj_ridge.data.vertices[v0_idx].co.z += 0.5
        obj_ridge.data.vertices[v2_idx].co.z += 0.5
        obj_ridge.data.vertices[c_idx].co.z += 0.5
        obj_ridge.data.update()

        topology.apply_post_process(obj_ridge, mode='ADAPTIVE', flatness_threshold=5.0)
        self.assertEqual(len(obj_ridge.data.polygons), 2, "対角線シワは最適三角2面に分割されること")
        for p in obj_ridge.data.polygons:
            self.assertEqual(len(p.vertices), 3)

        # --- ケース3: 中心盛り上がり (Dome / Tent) ---
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        obj_dome = bpy.context.active_object
        topology.apply_cross_subdivision(obj_dome)
        subdiv_map = obj_dome["_taremin_cross_subdiv_map"]
        c_idx = subdiv_map[0]["center_vert_index"]
        # 中心点 pc だけを大きく持ち上げてドーム/テント状に変形
        obj_dome.data.vertices[c_idx].co.z += 0.6
        obj_dome.data.update()

        topology.apply_post_process(obj_dome, mode='ADAPTIVE', flatness_threshold=5.0)
        self.assertEqual(len(obj_dome.data.polygons), 4, "中心盛り上がり(ドーム状)は十字分割(4面)が維持されること")
        for p in obj_dome.data.polygons:
            self.assertEqual(len(p.vertices), 3)

    def test_reset_restores_original_quad_topology(self):
        """十字分割・後処理後のリセット操作で初期Quadメッシュ構造と座標に完全復元されるかの検証"""
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        obj = bpy.context.active_object
        obj.taremin_cloth.is_cloth = True

        # 初期座標を記録
        orig_coords = [v.co.copy() for v in obj.data.vertices]
        self.assertEqual(len(obj.data.vertices), 4)
        self.assertEqual(len(obj.data.polygons), 1)

        # 1. 十字分割を適用
        bpy.ops.taremin_cloth.apply_cross_subdivision()
        self.assertEqual(len(obj.data.vertices), 5)
        self.assertEqual(len(obj.data.polygons), 4)

        # 2. 変形させて後処理を実行（三角形2面に変換）
        obj.data.vertices[0].co.z += 1.0
        obj.data.update()
        bpy.ops.taremin_cloth.apply_post_process()
        self.assertEqual(len(obj.data.polygons), 2)  # 三角形2個

        # 3. Reset Selected を実行
        res = bpy.ops.taremin_cloth.reset_selected()
        self.assertEqual(res, {'FINISHED'})

        # 検証: 初期Quadトポロジー（4頂点、1面）および初期座標に完全復元されていること！
        self.assertEqual(len(obj.data.vertices), 4, "頂点数が元の4頂点に戻っていること")
        self.assertEqual(len(obj.data.polygons), 1, "ポリゴン数が元の1面(Quad)に戻っていること")
        self.assertEqual(len(obj.data.polygons[0].vertices), 4, "面がQuad(4頂点)であること")
        for i, v in enumerate(obj.data.vertices):
            self.assertAlmostEqual((v.co - orig_coords[i]).length, 0.0, places=5, msg="初期座標に復元されていること")

        # 4. 十字分割のまま後処理を挟まずにリセットした場合も同様にQuadに復元されること
        bpy.ops.taremin_cloth.apply_cross_subdivision()
        self.assertEqual(len(obj.data.vertices), 5)
        res = bpy.ops.taremin_cloth.reset_selected()
        self.assertEqual(res, {'FINISHED'})
        self.assertEqual(len(obj.data.vertices), 4)
        self.assertEqual(len(obj.data.polygons), 1)

    def test_reset_self_healing_without_backup(self):
        """バックアップMeshが存在しない過渡期オブジェクトでも、リセット時にQuad復元と初期座標復元が正常に機能するかの検証"""
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        obj = bpy.context.active_object
        obj.taremin_cloth.is_cloth = True

        from taremin_cloth.operators import get_or_create_simulator
        sim, coords = get_or_create_simulator(obj)

        # 十字分割されるが、古いセッションのように _taremin_backup_mesh が無い場合を再現
        topology.apply_cross_subdivision(obj)
        if "_taremin_backup_mesh" in obj:
            del obj["_taremin_backup_mesh"]
        self.assertEqual(len(obj.data.vertices), 5)

        # ユーザーが分割なしに切り替えて変形させた状態を再現
        obj.taremin_cloth.enable_cross_subdivision = False
        obj.data.vertices[0].co.z += 1.5
        obj.data.update()
        obj["_taremin_is_deformed"] = True

        # リセット実行
        res = bpy.ops.taremin_cloth.reset_selected()
        self.assertEqual(res, {'FINISHED'})

        # バックアップが無くても自己修復でQuad（4頂点、1面）および初期Z=0に戻ること！
        self.assertEqual(len(obj.data.vertices), 4, "自己修復で4頂点に戻ること")
        self.assertEqual(len(obj.data.polygons), 1, "自己修復で1面(Quad)に戻ること")
        for v in obj.data.vertices:
            self.assertAlmostEqual(v.co.z, 0.0, places=5, msg="初期Z=0に復元されること")

    def test_reset_after_adaptive_partial_subdivision(self):
        """ADAPTIVE後処理で一部のQuadのみ十字維持（中間頂点数）になった状態からリセットで初期Quad構造・座標に完全復元されるかの検証"""
        # 2x2 Grid (9頂点, 4個のQuad面)
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=2, y_subdivisions=2, size=2.0)
        obj = bpy.context.active_object
        obj.taremin_cloth.is_cloth = True
        obj.taremin_cloth.enable_cross_subdivision = True

        from taremin_cloth.operators import get_or_create_simulator, cache_rest_positions
        cache_rest_positions(obj)
        self.assertEqual(len(obj.data.vertices), 9)
        self.assertEqual(len(obj.data.polygons), 4)

        # 十字分割実行 -> 9 + 4 = 13頂点, 16ポリゴン
        res = bpy.ops.taremin_cloth.apply_cross_subdivision()
        self.assertEqual(res, {'FINISHED'})
        self.assertEqual(len(obj.data.vertices), 13)
        self.assertEqual(len(obj.data.polygons), 16)

        # 頂点を変形: Quad 0の中心点(頂点9)を大きく持ち上げてドーム状にする
        # これによりADAPTIVE判定でQuad 0のみ十字分割が維持され、他はQuad/2三角化される
        obj.data.vertices[9].co.z += 1.5
        obj.data.update()
        obj["_taremin_is_deformed"] = True

        # ADAPTIVE後処理を実行
        obj.taremin_cloth.post_process_mode = 'ADAPTIVE'
        res_post = bpy.ops.taremin_cloth.apply_post_process()
        self.assertEqual(res_post, {'FINISHED'})

        # Quad 0の中心点(1頂点)のみ残るため、頂点数は 9 + 1 = 10 頂点になる
        # (元の9でも全Pokeの13でもない、中間頂点数状態)
        self.assertEqual(len(obj.data.vertices), 10, "Quad 0の中心点のみ残り10頂点になっていること")

        # この中間頂点数の状態からリセットを実行！
        res_reset = bpy.ops.taremin_cloth.reset_selected()
        self.assertEqual(res_reset, {'FINISHED'})

        # 検証: 初期Quadメッシュ（9頂点、4面Quad）および初期Z=0に100%完全復元されること！
        self.assertEqual(len(obj.data.vertices), 9, "リセット後に初期の9頂点に完全復元されること")
        self.assertEqual(len(obj.data.polygons), 4, "リセット後に初期の4面(Quad)に完全復元されること")
        for v in obj.data.vertices:
            self.assertAlmostEqual(v.co.z, 0.0, places=5, msg="全頂点が初期座標Z=0に復元されること")


if __name__ == "__main__":
    unittest.main()
