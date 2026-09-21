"""インタラクティブ起動の段階初期化のE2Eテスト (Blenderバックグラウンド実行用)"""
import unittest
import bpy
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import taremin_cloth
from taremin_cloth.ops import interactive as inter_mod
from taremin_cloth.utils import drawing


def _make_cloth_grid():
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=1.0)
    obj = bpy.context.active_object
    obj.taremin_cloth.is_cloth = True
    obj.taremin_cloth.isolate_viewport_view = False
    return obj


class TestInteractiveInitE2E(unittest.TestCase):
    def setUp(self):
        taremin_cloth.register()
        for obj in list(bpy.context.scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh, do_unlink=True)
        from taremin_cloth.engine.cache import clear_simulators
        clear_simulators()
        drawing.clear_init_overlay_info()
        inter_mod._interactive_running = False
        inter_mod._interactive_operator_instance = None

    def tearDown(self):
        try:
            inst = inter_mod._interactive_operator_instance
            if inst is not None:
                inst.cancel(bpy.context)
        except Exception:
            pass
        inter_mod._interactive_running = False
        inter_mod._interactive_operator_instance = None
        drawing.clear_init_overlay_info()
        from taremin_cloth.engine.cache import clear_simulators
        clear_simulators()
        for obj in list(bpy.context.scene.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh, do_unlink=True)
        taremin_cloth.unregister()

    def test_background_invoke_runs_all_stages(self):
        """バックグラウンドでは invoke が同期 execute に委譲し全段階が完走すること"""
        obj = _make_cloth_grid()
        self.assertTrue(bpy.app.background)
        ret = bpy.ops.taremin_cloth.interactive()
        self.assertIn('RUNNING_MODAL', ret)
        self.assertTrue(inter_mod.is_interactive_running())
        # 準備HUDは完了時に消去されていること (リークなし)
        self.assertIsNone(drawing.get_init_overlay_info())
        self.assertFalse(drawing.is_init_overlay_active())
        # シミュレータが生成・登録されていること
        from taremin_cloth.engine.cache import _simulators
        self.assertIn(obj.name, _simulators)
        # 6段階の所要時間が記録され、サマリー用simが保持されていること
        inst = inter_mod._interactive_operator_instance
        self.assertIsNotNone(inst)
        self.assertEqual(len(inst._init_timings), 6)
        for msg, ms in inst._init_timings:
            self.assertIn(msg, inter_mod.TAREMIN_CLOTH_OT_interactive._INIT_STAGE_MESSAGES)
            self.assertGreaterEqual(ms, 0.0)
        self.assertIsNotNone(inst._init_sim)

    def test_warm_resume_reuses_simulator(self):
        """2回目の起動が Warm Resume で同一シミュレータを再利用すること"""
        obj = _make_cloth_grid()
        bpy.ops.taremin_cloth.interactive()
        from taremin_cloth.engine.cache import _simulators
        first_sim, _ = _simulators[obj.name]
        inst = inter_mod._interactive_operator_instance
        self.assertIsNotNone(inst)
        inst.cancel(bpy.context)
        self.assertFalse(inter_mod.is_interactive_running())
        # Warm Resume: インスタンス保持のため2回目は再生成されない
        bpy.ops.taremin_cloth.interactive()
        second_sim, _ = _simulators[obj.name]
        self.assertIs(first_sim, second_sim)
        self.assertIsNone(drawing.get_init_overlay_info())

    def test_abort_init_cleans_up(self):
        """準備中断ヘルパーが進捗・HUD・実行フラグを確実に復旧すること"""
        obj = _make_cloth_grid()
        bpy.ops.taremin_cloth.interactive()
        op = inter_mod._interactive_operator_instance
        self.assertIsNotNone(op)
        # 準備途中の状態を擬似再現して中断する
        op._init_stage = 1
        drawing.set_init_overlay_info("Preparing cloth...", 1, 4)
        op._abort_init(bpy.context)
        self.assertIsNone(op._init_stage)
        self.assertFalse(inter_mod.is_interactive_running())
        self.assertIsNone(inter_mod._interactive_operator_instance)
        self.assertIsNone(drawing.get_init_overlay_info())


if __name__ == "__main__":
    unittest.main()
