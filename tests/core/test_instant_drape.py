"""Quasi-Static Instant Drape (準静的一瞬ドレープ) の単体テスト。"""

import unittest
import numpy as np
import time

try:
    import taremin_cloth_core as core
    HAS_CORE = True
except ImportError:
    try:
        from taremin_cloth import taremin_cloth_core as core
        HAS_CORE = True
    except ImportError:
        HAS_CORE = False


@unittest.skipUnless(HAS_CORE, "taremin_cloth_core not available")
class TestInstantDrape(unittest.TestCase):
    def test_instant_drape_convergence(self):
        # 3x3 グリッド布 (9頂点)
        positions = []
        for y in range(3):
            for x in range(3):
                positions.append([x * 0.1, y * 0.1, 1.0])
        positions = np.array(positions, dtype=np.float32)

        edges = []
        for y in range(3):
            for x in range(3):
                idx = y * 3 + x
                if x + 1 < 3:
                    edges.append([idx, idx + 1])
                if y + 1 < 3:
                    edges.append([idx, idx + 3])
        edges = np.array(edges, dtype=np.uint32)

        faces = []
        for y in range(2):
            for x in range(2):
                i00 = y * 3 + x
                i10 = y * 3 + (x + 1)
                i01 = (y + 1) * 3 + x
                i11 = (y + 1) * 3 + (x + 1)
                faces.append([i00, i10, i11])
                faces.append([i00, i11, i01])
        faces = np.array(faces, dtype=np.uint32)

        inv_masses = np.ones(len(positions), dtype=np.float32)
        # 上端の頂点 (y=2: 6, 7, 8) をピン留め
        inv_masses[6] = 0.0
        inv_masses[7] = 0.0
        inv_masses[8] = 0.0

        sim = core.ClothSimulator(positions, edges, faces, inv_masses)
        sim.set_gravity(0.0, 0.0, -9.81)

        # 初回パイプライン生成（シェーダー遅延コンパイル）をウォームアップ
        sim.step(0.016, 1)

        t0 = time.perf_counter()
        executed = sim.instant_drape(substeps=40, damping=0.85)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(executed, 40)
        self.assertLess(elapsed_ms, 500.0, f"Instant drape took too long: {elapsed_ms:.1f}ms")

        out_flat = np.zeros(len(positions) * 3, dtype=np.float32)
        sim.get_positions(out_flat)
        out = out_flat.reshape((-1, 3))

        # ピン留め頂点は動いていないこと
        np.testing.assert_allclose(out[6], positions[6], atol=1e-4)
        np.testing.assert_allclose(out[7], positions[7], atol=1e-4)
        np.testing.assert_allclose(out[8], positions[8], atol=1e-4)

        # 自由端頂点 (0, 1, 2) は重力でZが垂れ下がっていること
        self.assertLess(out[0, 2], 1.0 - 0.05, f"Free vertex did not drape: z={out[0, 2]}")
        self.assertLess(out[1, 2], 1.0 - 0.05, f"Free vertex did not drape: z={out[1, 2]}")
        self.assertLess(out[2, 2], 1.0 - 0.05, f"Free vertex did not drape: z={out[2, 2]}")

        print(f"[Instant Drape Test] 40 substeps completed in {elapsed_ms:.2f}ms (Z dropped to {out[0, 2]:.3f}m)")

    def test_instant_drape_operator_execution(self):
        """TAREMIN_CLOTH_OT_instant_drape オペレーターの execute() が Context 引数で安全に動作することを検証"""
        from unittest.mock import MagicMock
        from taremin_cloth.ops.basic import TAREMIN_CLOTH_OT_instant_drape
        from taremin_cloth.engine.cache import _simulators

        # モック Context と Scene の構築
        context = MagicMock()
        scene = MagicMock()
        context.scene = scene
        scene.frame_current = 1
        scene.objects = []

        # モック布オブジェクトの構築
        obj = MagicMock()
        obj.name = "TestClothDrape"
        settings = MagicMock()
        settings.is_cloth = True
        obj.taremin_cloth = settings
        obj.data.vertices = MagicMock()
        obj.data.vertices.__len__.return_value = 4
        obj.matrix_world = MagicMock()
        obj.matrix_world.__getitem__.side_effect = lambda idx: [1.0 if i == idx else 0.0 for i in range(4)]
        context.active_object = obj

        # モックシミュレーター
        mock_sim = MagicMock()
        mock_sim.instant_drape.return_value = 40
        # キャッシュには (sim, coords) のタプルが格納される規約
        _simulators[obj.name] = (mock_sim, np.zeros(12, dtype=np.float32))

        op = TAREMIN_CLOTH_OT_instant_drape()
        op.substeps = 40
        op.damping = 0.85
        op.report = MagicMock()

        try:
            # 実行 (以前は context の frame_current 参照で AttributeError になっていた)
            res = op.execute(context)
            self.assertEqual(res, {'FINISHED'})
            mock_sim.instant_drape.assert_called_once_with(40, 0.85)
            mock_sim.get_positions.assert_called_once()
            context.window_manager.progress_begin.assert_called_once_with(0, 5)
            context.window_manager.progress_end.assert_called_once()

            # 完了トーストHUDバッジが設定されていること
            from taremin_cloth.utils import drawing
            toast = drawing.get_toast_overlay_info()
            self.assertIsNotNone(toast)
            self.assertIn("40 substeps", toast["message"])
            self.assertEqual(toast["pct"], 100)
            drawing.clear_toast_overlay_info()
            self.assertIsNone(drawing.get_toast_overlay_info())
        finally:
            _simulators.pop(obj.name, None)

    def test_instant_drape_modal_lifecycle(self):
        """TAREMIN_CLOTH_OT_instant_drape の invoke() -> modal(TIMER) 進行 -> FINISHED を検証"""
        from unittest.mock import MagicMock
        from taremin_cloth.ops.basic import TAREMIN_CLOTH_OT_instant_drape
        from taremin_cloth.engine.cache import _simulators
        from taremin_cloth.utils import drawing

        context = MagicMock()
        context.scene = MagicMock()
        context.scene.objects = []
        context.window = MagicMock()
        context.window_manager = MagicMock()

        obj = MagicMock()
        obj.name = "TestClothModal"
        settings = MagicMock()
        settings.is_cloth = True
        obj.taremin_cloth = settings
        obj.data.vertices = MagicMock()
        obj.data.vertices.__len__.return_value = 4
        obj.matrix_world = MagicMock()
        obj.matrix_world.__getitem__.side_effect = lambda idx: [1.0 if i == idx else 0.0 for i in range(4)]
        context.active_object = obj

        mock_sim = MagicMock()
        mock_sim.instant_drape.return_value = 40
        _simulators[obj.name] = (mock_sim, np.zeros(12, dtype=np.float32))

        op = TAREMIN_CLOTH_OT_instant_drape()
        op.substeps = 40
        op.damping = 0.85
        op.report = MagicMock()

        try:
            # 1. invoke
            event_init = MagicMock()
            res_invoke = op.invoke(context, event_init)
            self.assertEqual(res_invoke, {'RUNNING_MODAL'})
            context.window_manager.modal_handler_add.assert_called_once_with(op)
            context.window_manager.event_timer_add.assert_called_once()

            # 2. modal TIMER 駆動によるステージ進行
            timer_event = MagicMock()
            timer_event.type = 'TIMER'

            # Stage 0 -> 1 ("Extracting mesh...")
            res = op.modal(context, timer_event)
            self.assertEqual(res, {'RUNNING_MODAL'})
            self.assertEqual(op._init_stage, 1)

            # Stage 1 -> 2 ("Initializing GPU...")
            res = op.modal(context, timer_event)
            self.assertEqual(res, {'RUNNING_MODAL'})
            self.assertEqual(op._init_stage, 2)

            # Stage 2 -> 3 ("Syncing colliders...")
            res = op.modal(context, timer_event)
            self.assertEqual(res, {'RUNNING_MODAL'})
            self.assertEqual(op._init_stage, 3)

            # Stage 3 -> 4 ("Settling cloth drape...")
            res = op.modal(context, timer_event)
            self.assertEqual(res, {'RUNNING_MODAL'})
            self.assertEqual(op._init_stage, 4)
            mock_sim.instant_drape.assert_called_once_with(40, 0.85)

            # Stage 4 -> FINISHED ("Updating mesh...")
            res = op.modal(context, timer_event)
            self.assertEqual(res, {'FINISHED'})
            mock_sim.get_positions.assert_called_once()
            self.assertIsNotNone(drawing.get_toast_overlay_info())
            drawing.clear_toast_overlay_info()

            # クリーンアップ確認
            context.window_manager.event_timer_remove.assert_called_once()
            context.window_manager.progress_end.assert_called_once()

            # 3. ESCキーによる中断キャンセル検証
            op_esc = TAREMIN_CLOTH_OT_instant_drape()
            op_esc.report = MagicMock()
            op_esc.invoke(context, event_init)
            esc_event = MagicMock()
            esc_event.type = 'ESC'
            res_esc = op_esc.modal(context, esc_event)
            self.assertEqual(res_esc, {'CANCELLED'})
            op_esc.report.assert_called_with({'INFO'}, "Instant drape cancelled")

        finally:
            _simulators.pop(obj.name, None)
            drawing.clear_toast_overlay_info()
            drawing.clear_init_overlay_info()

    def test_toast_overlay_lifecycle(self):
        """トースト通知バッジのライフサイクル（設定、取得、クリア）を検証"""
        from taremin_cloth.utils import drawing

        drawing.clear_toast_overlay_info()
        self.assertFalse(drawing.is_toast_overlay_active())

        drawing.set_toast_overlay_info("Test Toast Message", duration=2.0, pct=100)
        self.assertTrue(drawing.is_toast_overlay_active())
        toast = drawing.get_toast_overlay_info()
        self.assertEqual(toast["message"], "Test Toast Message")
        self.assertEqual(toast["pct"], 100)

        drawing.clear_toast_overlay_info()
        self.assertFalse(drawing.is_toast_overlay_active())
        self.assertIsNone(drawing.get_toast_overlay_info())


if __name__ == "__main__":
    unittest.main()
