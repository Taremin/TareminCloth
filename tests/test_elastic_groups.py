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

import taremin_cloth_core
import taremin_cloth
from tests.panel_test_utils import render_panel


class TestElasticGroups(unittest.TestCase):
    def setUp(self):
        taremin_cloth.register()
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def tearDown(self):
        taremin_cloth.unregister()
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def test_elastic_group_lifecycle(self):
        """伸縮グループの作成・エッジ割り当て・選択・削除のライフサイクル検証"""
        # 1. 2x2 平面グリッドを作成
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3, size=1.0)
        obj = bpy.context.active_object
        self.assertIsNotNone(obj)

        settings = obj.taremin_cloth
        settings.is_cloth = True

        # 2. Editモードで最初のエッジを選択
        bpy.ops.object.mode_set(mode='EDIT')
        import bmesh
        bm = bmesh.from_edit_mesh(obj.data)
        bm.edges.ensure_lookup_table()
        for e in bm.edges:
            e.select = False
        # エッジ0と1を選択
        bm.edges[0].select = True
        bm.edges[1].select = True
        bmesh.update_edit_mesh(obj.data)

        # 3. 新規伸縮グループ作成オペレーターの実行
        res = bpy.ops.taremin_cloth.add_elastic_group()
        self.assertEqual(res, {'FINISHED'})
        self.assertEqual(len(settings.elastic_groups), 1)

        group = settings.elastic_groups[0]
        self.assertEqual(group.name, "Elastic 1")
        self.assertEqual(group.scale, 1.0)
        edge_indices = group.get_edge_indices()
        self.assertEqual(set(edge_indices), {0, 1})

        # 4. エッジ再選択のテスト
        for e in bm.edges:
            e.select = False
        bmesh.update_edit_mesh(obj.data)

        res_sel = bpy.ops.taremin_cloth.select_elastic_edges()
        self.assertEqual(res_sel, {'FINISHED'})
        selected = [e.index for e in bm.edges if e.select]
        self.assertEqual(set(selected), {0, 1})

        # Objectモードに戻す
        bpy.ops.object.mode_set(mode='OBJECT')

        # 5. グループ削除
        res_del = bpy.ops.taremin_cloth.remove_elastic_group()
        self.assertEqual(res_del, {'FINISHED'})
        self.assertEqual(len(settings.elastic_groups), 0)

    def test_elastic_simulation_shrink(self):
        """伸縮グループによるエッジ自然長収縮シミュレーションの検証"""
        # 2頂点を結ぶ単純な線分メッシュを作成
        mesh = bpy.data.meshes.new("LineMesh")
        mesh.from_pydata([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0, 1]], [])
        mesh.update()

        obj = bpy.data.objects.new("LineObj", mesh)
        bpy.context.scene.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        settings = obj.taremin_cloth
        settings.is_cloth = True
        settings.gravity = 0.0 # 無重力
        settings.tension_stiffness = 10000.0

        # 頂点0をPin固定
        vg = obj.vertex_groups.new(name="Pin")
        vg.add([0], 1.0, 'REPLACE')
        settings.pin_vertex_group = "Pin"

        # 伸縮グループを作成してエッジ0を登録、倍率 0.5 に設定
        grp = settings.elastic_groups.add()
        grp.name = "TestShrink"
        grp.set_edge_indices([0])
        grp.scale = 0.5

        # シミュレータ取得とフレーム進行
        from taremin_cloth.operators import get_or_create_simulator, sync_cloth_parameters
        sim, coords = get_or_create_simulator(obj)
        sync_cloth_parameters(sim, obj, bpy.context.scene)

        # 40ステップシミュレーション
        for _ in range(40):
            sync_cloth_parameters(sim, obj, bpy.context.scene)
            sim.step(dt=1.0 / 60.0, substeps=20)

        sim.get_positions(coords)
        p0 = coords[0:3]
        p1 = coords[3:6]
        dist = float(np.linalg.norm(p1 - p0))

        print(f"\n[Test Elastic Simulation Shrink] Initial: 1.0, Target: 0.5, Final: {dist:.4f}")
        self.assertLess(abs(dist - 0.5), 0.05, f"シミュレーション結果が目標長 0.5 に収縮していること (実測: {dist:.4f})")

    def test_panel_draw(self):
        """伸縮グループが存在する状態でのパネルUI描画がエラーなく実行されることを検証"""
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=2, y_subdivisions=2)
        obj = bpy.context.active_object
        obj.taremin_cloth.is_cloth = True
        grp = obj.taremin_cloth.elastic_groups.add()
        grp.name = "Test"
        grp.set_edge_indices([0])

        # render_panel ユーティリティを用いて描画を実行し、UI要素を検証
        layout = render_panel(bpy.types.TAREMIN_CLOTH_PT_main_panel)
        operators = layout.get_operators()

        # 伸縮グループ関連のオペレーターが描画されていることを確認
        self.assertIn("taremin_cloth.add_elastic_group", operators)
        self.assertIn("taremin_cloth.remove_elastic_group", operators)
        self.assertIn("taremin_cloth.assign_elastic_edges", operators)


if __name__ == "__main__":
    unittest.main()
