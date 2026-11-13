# -*- coding: utf-8 -*-
"""パターンエディタの不正状態検出ログとセッション状態管理の単体テスト (Blender不要)."""

import logging
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

import tests  # noqa: F401, E402  # bpyモック等のテスト基盤初期化
from taremin_cloth.ops import pattern as pattern_ops  # noqa: E402
from taremin_cloth.ops.pattern import (  # noqa: E402
    TAREMIN_OT_close_pattern_editor,
    TAREMIN_OT_pattern_editor,
    is_pattern_editor_active,
)
from taremin_cloth.pattern.document import Document  # noqa: E402
from taremin_cloth.utils.logger import logger, set_log_level  # noqa: E402


class _DeadStructRNAOp:
    """StructRNA解放済みモーダルオペレータを模擬するダミー."""

    @property
    def controller(self):
        raise ReferenceError("StructRNA of type TAREMIN_OT_pattern_editor has been removed")

    @property
    def canvas(self):
        raise ReferenceError("StructRNA of type TAREMIN_OT_pattern_editor has been removed")


class _FakePoseCopyable:
    """copy() を持つダミーの回転・位置ベクトル."""

    def __init__(self, tag: str):
        self.tag = tag

    def copy(self):
        return _FakePoseCopyable(self.tag)


def _make_context(
    *,
    scene_name: str = "Scene",
    active: bool = True,
    area_type: str = "VIEW_3D",
):
    scene = SimpleNamespace(
        name=scene_name,
        taremin_pattern_editor_active=active,
        taremin_pattern_document="",
    )
    space = SimpleNamespace(
        type="VIEW_3D",
        show_region_toolbar=True,
        show_region_tool_header=True,
        region_3d=None,
    )
    rv3d = SimpleNamespace(
        view_perspective="PERSP",
        view_rotation=_FakePoseCopyable("orig_rot"),
        view_location=_FakePoseCopyable("orig_loc"),
        view_distance=5.0,
    )
    space.region_3d = rv3d
    area = (
        SimpleNamespace(type=area_type, spaces=SimpleNamespace(active=space), regions=[])
        if area_type is not None
        else None
    )
    region = SimpleNamespace(width=1280, height=720)
    wm = SimpleNamespace(modal_handler_add=MagicMock())
    return SimpleNamespace(
        scene=scene,
        area=area,
        space_data=space,
        region=region,
        region_data=rv3d,
        window_manager=wm,
    )


def _make_event(*, event_type: str = "MOUSEMOVE", value: str = "NOTHING"):
    return SimpleNamespace(
        type=event_type,
        value=value,
        mouse_region_x=100,
        mouse_region_y=200,
        mouse_x=100,
        mouse_y=200,
        shift=False,
        ctrl=False,
        alt=False,
        unicode="",
    )


def _patch_editor_startup(*, canvas_instance=None):
    """invoke() 内の Blender 依存呼び出しを安全なフェイクへ差し替えるパッチ束."""
    canvas_mock = canvas_instance or MagicMock()
    return (
        patch.object(pattern_ops, "is_interactive_active", return_value=False),
        patch.object(pattern_ops, "stop_animation"),
        patch.object(pattern_ops, "load_document_from_scene", return_value=Document.new()),
        patch.object(pattern_ops, "_find_target_armature", return_value=None),
        patch.object(pattern_ops, "extract_armature_bones_for_canvas", return_value={}),
        patch.object(pattern_ops, "sync_view2d_from_region_3d"),
        patch.object(pattern_ops, "get_viewport_overlay_offsets", return_value=(0.0, 0.0, 0.0, 0.0)),
        patch.object(pattern_ops, "PatternCanvasRenderer", return_value=canvas_mock),
        patch.object(pattern_ops, "tag_redraw_view3d"),
    )


class TestPatternEditorActiveLoggingAndState(unittest.TestCase):
    """パターンエディタの不正状態検出ログとライフサイクル管理の単体テスト."""

    def setUp(self):
        self._saved_op = pattern_ops._active_pattern_editor_op
        self._saved_canvas = pattern_ops._active_pattern_editor_canvas
        self._saved_level = logger.level
        pattern_ops._active_pattern_editor_op = None
        pattern_ops._active_pattern_editor_canvas = None

    def tearDown(self):
        pattern_ops._active_pattern_editor_op = self._saved_op
        pattern_ops._active_pattern_editor_canvas = self._saved_canvas
        logger.setLevel(self._saved_level)
        for handler in logger.handlers:
            handler.setLevel(self._saved_level)

    def test_stale_flag_without_op_ref_logs_warning_and_recovers_to_open(self):
        """フラグのみ True で _active_pattern_editor_op が None の不正状態を検知し、警告後に復旧して開けること."""
        ctx = _make_context(scene_name="GarmentScene", active=True, area_type="VIEW_3D")
        op_self = SimpleNamespace(report=MagicMock())
        canvas_mock = MagicMock()

        patches = _patch_editor_startup(canvas_instance=canvas_mock)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            with self.assertLogs(logger="taremin_cloth", level="WARNING") as cm:
                res = TAREMIN_OT_pattern_editor.invoke(op_self, ctx, _make_event())

        self.assertEqual(res, {'RUNNING_MODAL'})
        self.assertTrue(ctx.scene.taremin_pattern_editor_active)
        self.assertIs(pattern_ops._active_pattern_editor_op, op_self)
        self.assertTrue(is_pattern_editor_active(scene=ctx.scene))

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].levelno, logging.WARNING)
        msg = cm.records[0].getMessage()
        self.assertIn("[Pattern] Detected stale pattern editor state (cleaning up):", msg)
        self.assertIn("scene='GarmentScene'", msg)
        self.assertIn("flag=True", msg)
        self.assertIn("area='VIEW_3D'", msg)
        self.assertIn("interactive=False", msg)
        self.assertIn("op_ref=None", msg)

    def test_dead_struct_rna_logs_warning_unregisters_fallback_canvas_and_recovers(self):
        """StructRNA破棄済み(ReferenceError)でも落ちずに警告を出し、退避キャンバスを解除して復旧すること."""
        dead_op = _DeadStructRNAOp()
        stale_canvas = MagicMock()
        pattern_ops._active_pattern_editor_op = dead_op
        pattern_ops._active_pattern_editor_canvas = stale_canvas
        ctx = _make_context(scene_name="Scene", active=True, area_type="VIEW_3D")

        # 診断フォーマッタ自体は副作用（グローバル参照消去）を起こさないこと
        diag = pattern_ops._format_pattern_editor_active_diagnostics(ctx)
        self.assertIs(pattern_ops._active_pattern_editor_op, dead_op)
        self.assertIn("op_ref=present", diag)
        self.assertIn("op_rna_alive=False(ReferenceError)", diag)

        new_op = SimpleNamespace(report=MagicMock())
        new_canvas = MagicMock()
        patches = _patch_editor_startup(canvas_instance=new_canvas)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            with self.assertLogs(logger="taremin_cloth", level="WARNING") as cm:
                res = TAREMIN_OT_pattern_editor.invoke(new_op, ctx, _make_event())

        self.assertEqual(res, {'RUNNING_MODAL'})
        stale_canvas.unregister_handler.assert_called_once()
        self.assertIs(pattern_ops._active_pattern_editor_op, new_op)
        self.assertEqual(len(cm.records), 1)
        msg = cm.records[0].getMessage()
        self.assertIn("op_ref=present", msg)
        self.assertIn("op_rna_alive=False(ReferenceError)", msg)

    def test_live_op_blocks_reopen_with_controller_and_canvas_diagnostics(self):
        """生存中セッションがあるときは controller / canvas 詳細付き WARNING を出して CANCELLED を返すこと."""
        ctrl = SimpleNamespace(
            active_tool_name="SEW",
            is_modal_input_active=True,
            editing_field="seam_offset",
            picking_bone=True,
            sliding_field=None,
            transform_session=object(),
            request_close=False,
        )
        canvas = SimpleNamespace(_handler=object(), _handler_3d=None)
        live_op = SimpleNamespace(controller=ctrl, canvas=canvas, _session_closed=False)
        pattern_ops._active_pattern_editor_op = live_op
        ctx = _make_context(scene_name="Scene", active=True, area_type="VIEW_3D")
        op_self = SimpleNamespace(report=MagicMock())

        with patch.object(pattern_ops, "is_interactive_active", return_value=True):
            with self.assertLogs(logger="taremin_cloth", level="WARNING") as cm:
                res = TAREMIN_OT_pattern_editor.invoke(op_self, ctx, _make_event())

        self.assertEqual(res, {'CANCELLED'})
        op_self.report.assert_called_once()
        self.assertEqual(op_self.report.call_args[0][0], {'WARNING'})
        self.assertEqual(len(cm.records), 1)
        msg = cm.records[0].getMessage()
        self.assertIn("[Pattern] Cannot open pattern editor (already active):", msg)
        self.assertIn("op_ref=present", msg)
        self.assertIn("op_rna_alive=True", msg)
        self.assertIn("closed=False", msg)
        self.assertIn("interactive=True", msg)
        self.assertIn("tool='SEW'", msg)
        self.assertIn("modal_input=True", msg)
        self.assertIn("editing_field='seam_offset'", msg)
        self.assertIn("picking_bone=True", msg)
        self.assertIn("has_transform=True", msg)
        self.assertIn("request_close=False", msg)
        self.assertIn("handler_2d=True", msg)
        self.assertIn("handler_3d=False", msg)

    def test_live_op_with_false_flag_warns_syncs_flag_and_blocks(self):
        """実体が生存中なのにフラグだけ False の逆向き不整合を検知・修復して二重起動を防ぐこと."""
        ctrl = SimpleNamespace(active_tool_name="SELECT")
        canvas = SimpleNamespace(_handler=object(), _handler_3d=object())
        live_op = SimpleNamespace(controller=ctrl, canvas=canvas)
        pattern_ops._active_pattern_editor_op = live_op
        ctx = _make_context(scene_name="Scene", active=False, area_type="VIEW_3D")
        op_self = SimpleNamespace(report=MagicMock())

        with patch.object(pattern_ops, "is_interactive_active", return_value=False):
            with self.assertLogs(logger="taremin_cloth", level="WARNING") as cm:
                res = TAREMIN_OT_pattern_editor.invoke(op_self, ctx, _make_event())

        self.assertEqual(res, {'CANCELLED'})
        self.assertTrue(ctx.scene.taremin_pattern_editor_active)
        self.assertIn("with False scene flag", cm.records[0].getMessage())

    def test_warns_even_if_diagnostics_helper_raises(self):
        """診断情報収集で予期せぬ例外が起きてもフォールバックログを出して CANCELLED を返すこと."""
        ctx = _make_context(scene_name="Scene", active=True)
        op_self = SimpleNamespace(report=MagicMock())
        live_op = SimpleNamespace(controller=SimpleNamespace(), canvas=SimpleNamespace())
        pattern_ops._active_pattern_editor_op = live_op

        with patch.object(
            pattern_ops,
            "_format_pattern_editor_active_diagnostics",
            side_effect=RuntimeError("diag boom"),
        ):
            with self.assertLogs(logger="taremin_cloth", level="WARNING") as cm:
                res = TAREMIN_OT_pattern_editor.invoke(op_self, ctx, _make_event())

        self.assertEqual(res, {'CANCELLED'})
        op_self.report.assert_called_once()
        self.assertEqual(len(cm.records), 1)
        self.assertIn("diagnostics_error=RuntimeError(diag boom)", cm.records[0].getMessage())

    def test_cancel_restores_ui_camera_and_resets_flags_without_saving(self):
        """Blender による cancel() で UI・カメラ復元とハンドラ/フラグ解除が実行されること."""
        ctx = _make_context(scene_name="Scene", active=False)
        op_self = SimpleNamespace(
            report=MagicMock(),
            _modal_impl=lambda c, e: TAREMIN_OT_pattern_editor._modal_impl(op_self, c, e),
            _finish=lambda c: TAREMIN_OT_pattern_editor._finish(op_self, c),
        )
        canvas_mock = MagicMock()

        patches = _patch_editor_startup(canvas_instance=canvas_mock)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            res = TAREMIN_OT_pattern_editor.invoke(op_self, ctx, _make_event())
        self.assertEqual(res, {'RUNNING_MODAL'})
        self.assertFalse(ctx.space_data.show_region_toolbar)
        self.assertEqual(ctx.region_data.view_perspective, "ORTHO")

        with patch.object(pattern_ops, "save_document_to_scene") as mock_save, \
                patch.object(pattern_ops, "build_garment_mesh_in_place") as mock_build, \
                patch.object(pattern_ops, "tag_redraw_view3d"):
            TAREMIN_OT_pattern_editor.cancel(op_self, ctx)

        mock_save.assert_not_called()
        mock_build.assert_not_called()
        canvas_mock.unregister_handler.assert_called_once()
        self.assertTrue(ctx.space_data.show_region_toolbar)
        self.assertTrue(ctx.space_data.show_region_tool_header)
        self.assertEqual(ctx.region_data.view_perspective, "PERSP")
        self.assertEqual(ctx.region_data.view_rotation.tag, "orig_rot")
        self.assertFalse(ctx.scene.taremin_pattern_editor_active)
        self.assertIsNone(pattern_ops._active_pattern_editor_op)
        self.assertFalse(is_pattern_editor_active(scene=ctx.scene))

    def test_modal_unhandled_exception_cleans_up_and_allows_reopen(self):
        """modal() 内で例外が起きてもクリーンアップされて再度開けること."""
        ctx = _make_context(scene_name="Scene", active=False)
        op_self = SimpleNamespace(
            report=MagicMock(),
            _modal_impl=MagicMock(side_effect=RuntimeError("modal crash")),
        )
        canvas_mock = MagicMock()

        patches = _patch_editor_startup(canvas_instance=canvas_mock)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            res = TAREMIN_OT_pattern_editor.invoke(op_self, ctx, _make_event())
        self.assertEqual(res, {'RUNNING_MODAL'})

        with patch.object(pattern_ops, "tag_redraw_view3d"):
            with self.assertLogs(logger="taremin_cloth", level="ERROR") as err_cm:
                modal_res = TAREMIN_OT_pattern_editor.modal(op_self, ctx, _make_event())

        self.assertEqual(modal_res, {'CANCELLED'})
        self.assertTrue(any("Unhandled exception in pattern editor modal" in r.getMessage() for r in err_cm.records))
        canvas_mock.unregister_handler.assert_called_once()
        self.assertFalse(ctx.scene.taremin_pattern_editor_active)
        self.assertIsNone(pattern_ops._active_pattern_editor_op)

        # 引き続き次の起動が警告なしで成功すること
        op_next = SimpleNamespace(report=MagicMock())
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            res_next = TAREMIN_OT_pattern_editor.invoke(op_next, ctx, _make_event())
        self.assertEqual(res_next, {'RUNNING_MODAL'})
        self.assertIs(pattern_ops._active_pattern_editor_op, op_next)

    def test_invoke_error_after_canvas_register_rolls_back_cleanly(self):
        """invoke() 中に modal_handler_add が失敗しても描画ハンドラやフラグをロールバックすること."""
        ctx = _make_context(scene_name="Scene", active=False)
        ctx.window_manager.modal_handler_add.side_effect = RuntimeError("wm failed")
        op_self = SimpleNamespace(report=MagicMock())
        canvas_mock = MagicMock()

        patches = _patch_editor_startup(canvas_instance=canvas_mock)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            with self.assertLogs(logger="taremin_cloth", level="ERROR"):
                res = TAREMIN_OT_pattern_editor.invoke(op_self, ctx, _make_event())

        self.assertEqual(res, {'CANCELLED'})
        canvas_mock.unregister_handler.assert_called_once()
        self.assertTrue(ctx.space_data.show_region_toolbar)
        self.assertEqual(ctx.region_data.view_perspective, "PERSP")
        self.assertFalse(ctx.scene.taremin_pattern_editor_active)
        self.assertIsNone(pattern_ops._active_pattern_editor_op)

    def test_close_operator_saves_builds_and_marks_session_closed_for_modal(self):
        """TAREMIN_OT_close_pattern_editor が正規終了処理を行い、後続の modal() が即 FINISHED を返すこと."""
        ctx = _make_context(scene_name="Scene", active=False)
        op_self = SimpleNamespace(
            report=MagicMock(),
            _modal_impl=MagicMock(),
        )
        canvas_mock = MagicMock()

        patches = _patch_editor_startup(canvas_instance=canvas_mock)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            res = TAREMIN_OT_pattern_editor.invoke(op_self, ctx, _make_event())
        self.assertEqual(res, {'RUNNING_MODAL'})
        self.assertTrue(TAREMIN_OT_close_pattern_editor.poll(ctx))

        close_op = SimpleNamespace(report=MagicMock())
        with patch.object(pattern_ops, "save_document_to_scene") as mock_save, \
                patch.object(pattern_ops, "build_garment_mesh_in_place") as mock_build, \
                patch.object(pattern_ops, "tag_redraw_view3d"):
            close_res = TAREMIN_OT_close_pattern_editor.execute(close_op, ctx)

        self.assertEqual(close_res, {'FINISHED'})
        mock_save.assert_called_once()
        mock_build.assert_called_once()
        canvas_mock.unregister_handler.assert_called_once()
        self.assertFalse(ctx.scene.taremin_pattern_editor_active)
        self.assertIsNone(pattern_ops._active_pattern_editor_op)

        # 直後に古いモーダルインスタンスへ届いたイベントでは _modal_impl を呼ばず FINISHED で抜ける
        late_modal_res = TAREMIN_OT_pattern_editor.modal(op_self, ctx, _make_event())
        self.assertEqual(late_modal_res, {'FINISHED'})
        op_self._modal_impl.assert_not_called()

    def test_load_pre_post_and_unregister_clear_stale_scene_flags_and_session(self):
        """load_pre/load_post および unregister ハンドラが残留フラグとアクティブセッションを解放すること."""
        stale_scene = SimpleNamespace(name="LoadedScene", taremin_pattern_editor_active=True)
        canvas_mock = MagicMock()
        live_op = SimpleNamespace(
            controller=SimpleNamespace(doc=Document.new()),
            canvas=canvas_mock,
            _session_closed=False,
            _session_scene=stale_scene,
        )
        pattern_ops._active_pattern_editor_op = live_op
        pattern_ops._active_pattern_editor_canvas = canvas_mock

        load_pre_list = []
        load_post_list = []
        fake_bpy = SimpleNamespace(
            data=SimpleNamespace(scenes=[stale_scene]),
            context=SimpleNamespace(scene=stale_scene),
            app=SimpleNamespace(handlers=SimpleNamespace(load_pre=load_pre_list, load_post=load_post_list)),
        )
        with patch.object(pattern_ops, "bpy", fake_bpy), \
                patch.object(pattern_ops, "tag_redraw_view3d"):
            with self.assertLogs(logger="taremin_cloth", level="WARNING") as cm:
                pattern_ops._on_pattern_editor_load_post(None)

        self.assertFalse(stale_scene.taremin_pattern_editor_active)
        self.assertIsNone(pattern_ops._active_pattern_editor_op)
        self.assertIsNone(pattern_ops._active_pattern_editor_canvas)
        canvas_mock.unregister_handler.assert_called_once()
        self.assertIn("Resetting stale pattern editor state after file load", cm.records[0].getMessage())

        # ハンドラ登録・load_pre・unregister のライフサイクル検証
        canvas_mock_2 = MagicMock()
        live_op_2 = SimpleNamespace(
            controller=SimpleNamespace(doc=Document.new()),
            canvas=canvas_mock_2,
            _session_closed=False,
            _session_scene=stale_scene,
        )
        with patch.object(pattern_ops, "bpy", fake_bpy), \
                patch.object(pattern_ops, "tag_redraw_view3d"):
            pattern_ops.register_pattern_editor_handlers()
            self.assertIn(pattern_ops._on_pattern_editor_load_pre, load_pre_list)
            self.assertIn(pattern_ops._on_pattern_editor_load_post, load_post_list)

            stale_scene.taremin_pattern_editor_active = True
            pattern_ops._active_pattern_editor_op = live_op_2
            pattern_ops._active_pattern_editor_canvas = canvas_mock_2
            pattern_ops._on_pattern_editor_load_pre(None)
            self.assertFalse(stale_scene.taremin_pattern_editor_active)
            self.assertIsNone(pattern_ops._active_pattern_editor_op)
            canvas_mock_2.unregister_handler.assert_called_once()

            # unregister 時に再びセッションが生きていても解放され、ハンドラも外れること
            canvas_mock_3 = MagicMock()
            live_op_3 = SimpleNamespace(
                controller=SimpleNamespace(doc=Document.new()),
                canvas=canvas_mock_3,
                _session_closed=False,
                _session_scene=stale_scene,
            )
            stale_scene.taremin_pattern_editor_active = True
            pattern_ops._active_pattern_editor_op = live_op_3
            pattern_ops._active_pattern_editor_canvas = canvas_mock_3
            pattern_ops.unregister_pattern_editor_handlers()
            self.assertFalse(stale_scene.taremin_pattern_editor_active)
            self.assertIsNone(pattern_ops._active_pattern_editor_op)
            canvas_mock_3.unregister_handler.assert_called_once()
            self.assertNotIn(pattern_ops._on_pattern_editor_load_pre, load_pre_list)
            self.assertNotIn(pattern_ops._on_pattern_editor_load_post, load_post_list)

    def test_panel_drafting_box_uses_live_session_truth_and_close_operator(self):
        """panels._draw_pattern_drafting_box が実体状態に応じて Open/Close オペレーターを出し分けること."""
        from taremin_cloth.panels import _draw_pattern_drafting_box

        # フラグだけ True の不正状態でも、実体が無ければ Open Pattern Editor を表示する
        ctx_stale = _make_context(scene_name="Scene", active=True)
        layout_stale = MagicMock()
        row_mock = MagicMock()
        layout_stale.box.return_value.row.return_value = row_mock
        with patch("taremin_cloth.panels._get_panel_reference_image_info", return_value=None):
            _draw_pattern_drafting_box(layout_stale, ctx_stale)
        called_ops_stale = [call.args[0] for call in row_mock.operator.call_args_list]
        self.assertIn("taremin_cloth.pattern_editor", called_ops_stale)
        self.assertNotIn("taremin_cloth.close_pattern_editor", called_ops_stale)

        # 実体が生存中のときは Close Pattern Editor オペレーターを表示する
        pattern_ops._active_pattern_editor_op = SimpleNamespace(
            controller=SimpleNamespace(),
            canvas=SimpleNamespace(),
            _session_closed=False,
        )
        layout_live = MagicMock()
        row_live = MagicMock()
        layout_live.box.return_value.row.return_value = row_live
        with patch("taremin_cloth.panels._get_panel_reference_image_info", return_value=None):
            _draw_pattern_drafting_box(layout_live, ctx_stale)
        called_ops_live = [call.args[0] for call in row_live.operator.call_args_list]
        self.assertIn("taremin_cloth.close_pattern_editor", called_ops_live)

    def test_warning_emitted_under_default_info_level(self):
        """既定の INFO 設定下でも WARNING ログが抑制されず出力可能であること."""
        set_log_level("INFO")
        self.assertTrue(logger.isEnabledFor(logging.WARNING))


if __name__ == "__main__":
    unittest.main()
