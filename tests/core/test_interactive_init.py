"""
インタラクティブ起動の段階初期化と進捗HUDの単体テスト (Blender不要)
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from taremin_cloth.utils import drawing
from taremin_cloth import i18n
from taremin_cloth.ops.interactive import TAREMIN_CLOTH_OT_interactive


def _make_context():
    ctx = MagicMock()
    ctx.window_manager.progress_begin = MagicMock()
    ctx.window_manager.progress_update = MagicMock()
    ctx.window_manager.progress_end = MagicMock()
    ctx.window_manager.cursor_modal_set = MagicMock()
    ctx.window_manager.cursor_modal_restore = MagicMock()
    ctx.window_manager.event_timer_add = MagicMock(return_value=MagicMock())
    ctx.window_manager.event_timer_remove = MagicMock()
    ctx.window_manager.modal_handler_add = MagicMock()
    ctx.workspace.status_text_set = MagicMock()
    return ctx


class TestInitOverlayInfo(unittest.TestCase):
    def setUp(self):
        drawing.clear_init_overlay_info()
        drawing.clear_bake_overlay_info()
        drawing.set_interactive_active(False)
        # 他テストのシェーダーキャッシュ汚染に依存しないよう退避・無効化
        self._saved_shaders = (
            drawing._cached_point_shader,
            drawing._cached_line_shader,
            drawing._cached_2d_shader,
            drawing._cached_smooth_line_shader,
        )
        drawing._cached_point_shader = None
        drawing._cached_line_shader = None
        drawing._cached_2d_shader = None
        drawing._cached_smooth_line_shader = None

    def tearDown(self):
        drawing.clear_init_overlay_info()
        drawing.clear_bake_overlay_info()
        drawing.set_interactive_active(False)
        (drawing._cached_point_shader,
         drawing._cached_line_shader,
         drawing._cached_2d_shader,
         drawing._cached_smooth_line_shader) = self._saved_shaders

    def test_set_get_clear(self):
        self.assertFalse(drawing.is_init_overlay_active())
        self.assertIsNone(drawing.get_init_overlay_info())
        drawing.set_init_overlay_info("Preparing cloth...", 1, 4)
        self.assertTrue(drawing.is_init_overlay_active())
        info = drawing.get_init_overlay_info()
        self.assertEqual(info["message"], "Preparing cloth...")
        self.assertEqual(info["current"], 1)
        self.assertEqual(info["total"], 4)
        drawing.clear_init_overlay_info()
        self.assertFalse(drawing.is_init_overlay_active())
        self.assertIsNone(drawing.get_init_overlay_info())

    def test_init_stage_messages_have_translations(self):
        raw = i18n.get_raw_translations()
        self.assertIn("en_US", raw)
        self.assertIn("ja_JP", raw)
        for msgid in TAREMIN_CLOTH_OT_interactive._INIT_STAGE_MESSAGES:
            self.assertIn(msgid, raw["en_US"])
            self.assertEqual(raw["en_US"][msgid], msgid)
            self.assertIn(msgid, raw["ja_JP"])
            self.assertTrue(raw["ja_JP"][msgid].strip())
        self.assertIn("[Esc] Cancel", raw["en_US"])
        self.assertIn("[Esc] Cancel", raw["ja_JP"])

    @patch.dict("sys.modules", {"blf": MagicMock()})
    @patch("taremin_cloth.utils.drawing.batch_for_shader")
    @patch("taremin_cloth.utils.drawing.gpu")
    @patch("taremin_cloth.utils.drawing.bpy")
    def test_draw_callback_2d_init_overlay(self, mock_bpy, mock_gpu, mock_batch):
        mock_region = MagicMock()
        mock_region.width = 1920
        mock_region.height = 1080
        mock_bpy.context.region = mock_region
        mock_batch_for_shader = mock_batch
        mock_batch_for_shader.return_value = MagicMock()
        mock_gpu.shader.from_builtin.return_value = MagicMock()

        drawing.set_interactive_active(False)
        drawing.set_init_overlay_info("Loading pins...", 1, 4)
        try:
            drawing.draw_callback_2d()
        except Exception as e:
            self.fail(f"draw_callback_2d for init overlay raised: {e}")
        self.assertTrue(mock_batch_for_shader.called)


class TestStagedInitStateMachine(unittest.TestCase):
    """実BlenderでもCIスタブでも動作するよう、演算子実体を生成せず
    非束縛メソッド呼出し + SimpleNamespace self で状態遷移を検証する"""

    def setUp(self):
        drawing.clear_init_overlay_info()
        from taremin_cloth.ops import interactive as inter_mod
        inter_mod._interactive_running = False
        inter_mod._interactive_operator_instance = None

    def tearDown(self):
        drawing.clear_init_overlay_info()
        from taremin_cloth.ops import interactive as inter_mod
        inter_mod._interactive_running = False
        inter_mod._interactive_operator_instance = None

    def _make_op_self(self, **attrs):
        import types
        cls = TAREMIN_CLOTH_OT_interactive
        op = types.SimpleNamespace()
        op._init_stage = 0
        op._stop_requested = False
        op._timer = None
        op._modal_handler_added = False
        op._progress_active = False
        op._cursor_wait_set = False
        op._init_timings = []
        op._init_partial = None
        op._init_sim = None
        op._INIT_STAGE_MESSAGES = TAREMIN_CLOTH_OT_interactive._INIT_STAGE_MESSAGES
        op._SLOW_STARTUP_SEC = TAREMIN_CLOTH_OT_interactive._SLOW_STARTUP_SEC
        op.report = MagicMock()
        # 内部呼出し解決用に実メソッドを束縛（個別テストで必要に応じ上書き）
        for name in (
            "_init_progress_total",
            "_update_init_progress",
            "_end_init_ui",
            "_abort_init",
            "_start_sim_loop",
            "_report_startup_summary",
        ):
            setattr(op, name, types.MethodType(getattr(cls, name), op))
        for k, v in attrs.items():
            setattr(op, k, v)
        return op

    def test_stage_dispatch_order(self):
        op = self._make_op_self()
        op._init_stage_topo = MagicMock()
        op._init_stage_pins = MagicMock()
        op._init_stage_sim_extract = MagicMock()
        op._init_stage_sim_gpu = MagicMock()
        op._init_stage_sim_collider = MagicMock()
        op._init_stage_setup = MagicMock()
        ctx = _make_context()
        run = TAREMIN_CLOTH_OT_interactive._run_init_stage
        self.assertTrue(run(op, ctx))
        op._init_stage_topo.assert_called_once()
        self.assertTrue(run(op, ctx))
        op._init_stage_pins.assert_called_once()
        self.assertTrue(run(op, ctx))
        op._init_stage_sim_extract.assert_called_once()
        self.assertTrue(run(op, ctx))
        op._init_stage_sim_gpu.assert_called_once()
        self.assertTrue(run(op, ctx))
        op._init_stage_sim_collider.assert_called_once()
        self.assertFalse(run(op, ctx))
        op._init_stage_setup.assert_called_once()
        # 全段階の所要時間が記録されていること
        self.assertEqual(len(op._init_timings), 6)
        for msg, ms in op._init_timings:
            self.assertIn(msg, TAREMIN_CLOTH_OT_interactive._INIT_STAGE_MESSAGES)
            self.assertGreaterEqual(ms, 0.0)

    def test_timer_drives_to_finish(self):
        op = self._make_op_self()
        op._run_init_stage = MagicMock(side_effect=[True, True, True, True, True, False])
        op._update_init_progress = MagicMock()
        op._start_sim_loop = MagicMock()
        op._abort_init = MagicMock()
        ctx = _make_context()
        step = TAREMIN_CLOTH_OT_interactive._modal_init_step
        for _ in range(6):
            res = step(op, ctx, MagicMock(type='TIMER'))
            self.assertEqual(res, {'RUNNING_MODAL'})
        self.assertEqual(op._run_init_stage.call_count, 6)
        op._start_sim_loop.assert_called_once_with(ctx)
        op._abort_init.assert_not_called()
        self.assertIsNone(drawing.get_init_overlay_info())

    def test_esc_aborts_init(self):
        # 準備UI開始済みの状態を再現して中断する
        op = self._make_op_self(_init_stage=1, _progress_active=True, _cursor_wait_set=True)
        ctx = _make_context()
        run_mock = MagicMock()
        op._run_init_stage = run_mock
        res = TAREMIN_CLOTH_OT_interactive._modal_init_step(op, ctx, MagicMock(type='ESC'))
        self.assertEqual(res, {'CANCELLED'})
        run_mock.assert_not_called()
        self.assertIsNone(op._init_stage)
        ctx.window_manager.progress_end.assert_called_once()
        ctx.window_manager.cursor_modal_restore.assert_called_once()
        self.assertFalse(op._progress_active)
        self.assertFalse(op._cursor_wait_set)
        drawing_state = drawing.get_init_overlay_info()
        self.assertIsNone(drawing_state)

    def test_init_failure_reports_and_cleans_up(self):
        op = self._make_op_self(_init_stage=2)
        ctx = _make_context()
        op._run_init_stage = MagicMock(side_effect=RuntimeError("gpu boom"))
        res = TAREMIN_CLOTH_OT_interactive._modal_init_step(op, ctx, MagicMock(type='TIMER'))
        self.assertEqual(res, {'CANCELLED'})
        op.report.assert_called()
        self.assertIsNone(op._init_stage)
        self.assertIsNone(drawing.get_init_overlay_info())

    def test_update_progress_reports(self):
        op = self._make_op_self(_init_stage=2)
        ctx = _make_context()
        TAREMIN_CLOTH_OT_interactive._update_init_progress(op, ctx)
        ctx.window_manager.progress_update.assert_called_with(2)
        info = drawing.get_init_overlay_info()
        self.assertIsNotNone(info)
        self.assertEqual(info["message"], "Extracting mesh...")
        ctx.workspace.status_text_set.assert_called()
        drawing.clear_init_overlay_info()

    def _make_summary_self(self, timings, sim=None):
        op = self._make_op_self()
        op._init_timings = list(timings)
        op._init_sim = sim
        return op

    def test_startup_summary_debug_logs_breakdown(self):
        """DEBUG時は段階別内訳を出力すること"""
        import taremin_cloth.ops.interactive as inter_mod
        sim = MagicMock()
        try:
            from taremin_cloth.engine.collider import _last_sync_info
            _last_sync_info[id(sim)] = {"sdf_source": "baked"}
            with patch.object(inter_mod, "get_preferences", return_value=MagicMock(log_level='DEBUG')), \
                    patch.object(inter_mod, "logger") as mock_logger:
                op = self._make_summary_self(
                    [("Extracting mesh...", 10.0), ("Syncing colliders...", 250.0)], sim=sim
                )
                TAREMIN_CLOTH_OT_interactive._report_startup_summary(op, MagicMock())
                mock_logger.debug.assert_called_once()
                msg = mock_logger.debug.call_args[0][0]
                self.assertIn("baked", msg)
                self.assertIn("Extracting mesh...", msg)
        finally:
            from taremin_cloth.engine.collider import _last_sync_info
            _last_sync_info.pop(id(sim), None)

    def test_startup_summary_info_silent_when_fast(self):
        """INFOかつ高速起動時はサマリーを出さないこと"""
        import taremin_cloth.ops.interactive as inter_mod
        with patch.object(inter_mod, "get_preferences", return_value=MagicMock(log_level='INFO')), \
                patch.object(inter_mod, "logger") as mock_logger:
            op = self._make_summary_self([("Extracting mesh...", 5.0)], sim=None)
            TAREMIN_CLOTH_OT_interactive._report_startup_summary(op, MagicMock())
            mock_logger.info.assert_not_called()
            mock_logger.debug.assert_not_called()

    def test_startup_summary_info_when_slow(self):
        """INFOかつ低速起動時は1行サマリーを出すこと"""
        import taremin_cloth.ops.interactive as inter_mod
        with patch.object(inter_mod, "get_preferences", return_value=MagicMock(log_level='INFO')), \
                patch.object(inter_mod, "logger") as mock_logger:
            op = self._make_summary_self(
                [("Extracting mesh...", 100.0), ("Syncing colliders...", 4000.0)], sim=None
            )
            TAREMIN_CLOTH_OT_interactive._report_startup_summary(op, MagicMock())
            mock_logger.info.assert_called_once()
            msg = mock_logger.info.call_args[0][0]
            self.assertIn("warm", msg)

    def test_startup_summary_no_timings_no_output(self):
        """計測なしでは何も出力しないこと"""
        import taremin_cloth.ops.interactive as inter_mod
        with patch.object(inter_mod, "logger") as mock_logger:
            op = self._make_summary_self([], sim=None)
            TAREMIN_CLOTH_OT_interactive._report_startup_summary(op, MagicMock())
            mock_logger.info.assert_not_called()
            mock_logger.debug.assert_not_called()

    def test_no_double_modal_handler_on_finish(self):
        """invoke済み (ハンドラ登録済み) のまま完走しても二重登録されないこと"""
        op = self._make_op_self(_modal_handler_added=True)
        op.report = MagicMock()
        ctx = _make_context()
        ctx.window = MagicMock()
        op._run_init_stage = MagicMock(side_effect=[True, True, True, True, True, False])
        op._update_init_progress = MagicMock()
        step = TAREMIN_CLOTH_OT_interactive._modal_init_step
        for _ in range(6):
            step(op, ctx, MagicMock(type='TIMER'))
        ctx.window_manager.modal_handler_add.assert_not_called()
        self.assertIsNotNone(op._timer)

    def test_modal_handler_added_once_from_sync_path(self):
        """execute同期パスではハンドラがちょうど1回だけ登録されること"""
        op = self._make_op_self(_modal_handler_added=False)
        op.report = MagicMock()
        ctx = _make_context()
        ctx.window = MagicMock()
        op._run_init_stage = MagicMock(side_effect=[True, True, True, True, True, False])
        op._update_init_progress = MagicMock()
        step = TAREMIN_CLOTH_OT_interactive._modal_init_step
        for _ in range(6):
            step(op, ctx, MagicMock(type='TIMER'))
        ctx.window_manager.modal_handler_add.assert_called_once()

    def test_end_init_ui_is_idempotent(self):
        """準備UI終了処理の二重呼び出しでprogress_endが重複実行されないこと"""
        op = self._make_op_self(_progress_active=True, _cursor_wait_set=True)
        ctx = _make_context()
        end = TAREMIN_CLOTH_OT_interactive._end_init_ui
        end(op, ctx, True)
        end(op, ctx, True)
        ctx.window_manager.progress_end.assert_called_once()
        ctx.window_manager.cursor_modal_restore.assert_called_once()


if __name__ == "__main__":
    unittest.main()
