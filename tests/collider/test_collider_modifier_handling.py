"""
コライダーおよびシミュレーション対象におけるモディファイア評価方針の検証テスト
- MESHコライダー評価時にSubsurfやSolidifyなどの面増殖・二重化モディファイアがスキップされること
- Latticeなどの形状デフォーマーが忠実に反映されること
- Fast Playbackでコライダーが常に保護され、Latticeがバイパス対象外であること
"""

import sys
from pathlib import Path
import unittest

root_dir = Path(__file__).parent.parent
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import bpy
import mathutils
import taremin_cloth
from taremin_cloth.engine.collider import (
    get_collider_eval_mesh,
    cleanup_collider_eval_mesh,
    sync_colliders,
    clear_collider_cache,
)
from taremin_cloth.engine.runner import apply_fast_playback, restore_fast_playback


class TestColliderModifierHandling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            taremin_cloth.register()
        except ValueError:
            pass

    @classmethod
    def tearDownClass(cls):
        try:
            taremin_cloth.unregister()
        except Exception:
            pass

    def setUp(self):
        clear_collider_cache()
        restore_fast_playback()
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def tearDown(self):
        clear_collider_cache()
        restore_fast_playback()

    def test_collider_lattice_deform_preserved(self):
        """MESHコライダーでLattice変形が正しく反映されることを検証"""
        # キューブ作成
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
        cube = bpy.context.active_object
        cube.name = "ColliderCubeLattice"
        cube.taremin_cloth_collider.is_collider = True
        cube.taremin_cloth_collider.collider_type = 'MESH'

        # Latticeオブジェクト作成
        lat_data = bpy.data.lattices.new("TestLattice")
        lat_data.points_u = 2
        lat_data.points_v = 2
        lat_data.points_w = 2
        lat_obj = bpy.data.objects.new("TestLatticeObj", lat_data)
        bpy.context.collection.objects.link(lat_obj)

        # Latticeの点(0,0,0)を変形
        lat_data.points[0].co_deform += mathutils.Vector((0.5, 0.5, 0.5))

        # モディファイア追加
        mod_lat = cube.modifiers.new(name="Lattice", type='LATTICE')
        mod_lat.object = lat_obj

        depsgraph = bpy.context.evaluated_depsgraph_get()
        mesh, eval_obj, disabled_mods = get_collider_eval_mesh(cube, depsgraph)
        try:
            self.assertEqual(len(disabled_mods), 0, "Latticeモディファイアは無効化されないこと")
            self.assertEqual(len(mesh.vertices), 8, "頂点数はベースメッシュと同じ8であること")

            # ベース頂点と比較して変形されているか検証
            coords = [list(v.co) for v in mesh.vertices]
            base_coords = [list(v.co) for v in cube.data.vertices]
            is_deformed = any(coords[i] != base_coords[i] for i in range(8))
            self.assertTrue(is_deformed, "Lattice変形がコライダー評価メッシュに反映されていること")
        finally:
            cleanup_collider_eval_mesh(eval_obj, disabled_mods)

    def test_collider_subsurf_solidify_skipped(self):
        """MESHコライダーでSubsurfやSolidifyがスキップされ、評価後に復元されることを検証"""
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
        cube = bpy.context.active_object
        cube.name = "ColliderCubeSubsurf"
        cube.taremin_cloth_collider.is_collider = True
        cube.taremin_cloth_collider.collider_type = 'MESH'

        # Subsurf（レベル2）とSolidifyを追加
        mod_sub = cube.modifiers.new(name="Subsurf", type='SUBSURF')
        mod_sub.levels = 2
        mod_sol = cube.modifiers.new(name="Solidify", type='SOLIDIFY')
        mod_sol.thickness = 0.1

        self.assertTrue(mod_sub.show_viewport)
        self.assertTrue(mod_sol.show_viewport)

        depsgraph = bpy.context.evaluated_depsgraph_get()
        mesh, eval_obj, disabled_mods = get_collider_eval_mesh(cube, depsgraph)
        try:
            self.assertEqual(len(disabled_mods), 2, "SubsurfとSolidifyの2つが無効化対象として記録されること")
            # Subsurfレベル2だと頂点数が98頂点以上になるが、スキップされていれば8頂点
            self.assertEqual(len(mesh.vertices), 8, "Subsurf/Solidifyがスキップされ8頂点であること")
        finally:
            cleanup_collider_eval_mesh(eval_obj, disabled_mods)

        # cleanup 後に元の表示状態に戻っていること
        self.assertTrue(mod_sub.show_viewport, "cleanup後にSubsurfのshow_viewportがTrueに復帰すること")
        self.assertTrue(mod_sol.show_viewport, "cleanup後にSolidifyのshow_viewportがTrueに復帰すること")

    def test_fast_playback_protects_collider_and_excludes_lattice(self):
        """Fast Playback実行時にコライダーが保護され、Latticeが除外されることを検証"""
        scene = bpy.context.scene

        # 1. コライダーオブジェクト（Subsurf付き）
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
        col_obj = bpy.context.active_object
        col_obj.taremin_cloth_collider.is_collider = True
        mod_col_sub = col_obj.modifiers.new(name="Subsurf", type='SUBSURF')

        # 2. 無関係な背景オブジェクト（SubsurfとLattice付き）
        bpy.ops.mesh.primitive_uv_sphere_add(location=(5, 5, 5))
        bg_obj = bpy.context.active_object
        mod_bg_sub = bg_obj.modifiers.new(name="Subsurf", type='SUBSURF')
        mod_bg_lat = bg_obj.modifiers.new(name="Lattice", type='LATTICE')

        # Fast Playback を強制適用
        apply_fast_playback(scene, force=True, include_sim_objs=False)

        # コライダーのSubsurfは保護されていること
        self.assertTrue(mod_col_sub.show_viewport, "コライダーのモディファイアは保護され無効化されないこと")

        # 背景オブジェクトのLatticeは除外（保護）されていること
        self.assertTrue(mod_bg_lat.show_viewport, "LATTICEはheavy_typesから除外されているため無効化されないこと")

        # 背景オブジェクトのSubsurfは無効化されていること
        self.assertFalse(mod_bg_sub.show_viewport, "背景オブジェクトのSubsurfは無効化されること")

        # 復元
        restore_fast_playback(scene)
        self.assertTrue(mod_bg_sub.show_viewport, "restore_fast_playbackでSubsurfが復元されること")

    def test_sync_colliders_skips_subsurf_triangles(self):
        """sync_colliders 呼び出し時にSubsurfの細分割三角形が除外され、ベース形状の三角形（12個）のみ同期されることを検証"""
        scene = bpy.context.scene

        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
        col_obj = bpy.context.active_object
        col_obj.taremin_cloth_collider.is_collider = True
        col_obj.taremin_cloth_collider.collider_type = 'MESH'

        mod_sub = col_obj.modifiers.new(name="Subsurf", type='SUBSURF')
        mod_sub.levels = 2

        passed_calls = []

        class MockSim:
            def clear_colliders(self):
                pass
            def set_mesh_collider_triangles(self, triangles, attributes):
                passed_calls.append(triangles)

        mock_sim = MockSim()
        sync_colliders(mock_sim, scene, force=True)

        self.assertEqual(len(passed_calls), 1, "set_mesh_collider_triangles が1回呼ばれること")
        # 立方体の三角形数は12。もしSubsurfレベル2が適用されていると192三角形以上になる
        num_tris = len(passed_calls[0])
        self.assertEqual(num_tris, 12, f"Subsurfが無視され12三角形のみ同期されること: 実際={num_tris}")

        # コライダーのモディファイアの表示状態が元通りTrueであること
        self.assertTrue(mod_sub.show_viewport)


if __name__ == "__main__":
    unittest.main()
