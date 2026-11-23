"""Phase 4.1 未カバー領域（SmoothTool / GUI Preview / log_tools 5コマンド）のテスト"""

import argparse
import contextlib
import gzip
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tests  # noqa: F401
from taremin_cloth import log_tools
from taremin_cloth.brush.base import BrushContext
from taremin_cloth.brush.smooth import SmoothTool
import taremin_cloth.ops.gui as gui_mod


_FIXTURE_LOG = _REPO_ROOT / "tests" / "fixtures" / "data_penetration_f46_f50.jsonl.gz"


def _make_dummy_brush_opt():
    """SmoothTool 用のダミーブラシ設定を返す"""
    return argparse.Namespace(
        radius=2.0,
        strength=0.8,
        falloff_shape="SMOOTH",
        use_active_region=False,
    )


def _make_dummy_mesh_data():
    """4頂点正方形グリッドのダミーメッシュデータ"""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return argparse.Namespace(
        coords=coords,
        num_vertices=4,
        normal_edges=[(0, 1), (0, 2), (1, 3), (2, 3)],
        rest_lengths=[1.0, 1.0, 1.0, 1.0],
        triangles=[(0, 1, 2), (1, 3, 2)],
        structural_edges=[(0, 1), (0, 2), (1, 3), (2, 3)],
        bend_edges=[],
        sewing_edges=[],
        pin_inv_masses=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


class TestSmoothToolCoverage(unittest.TestCase):
    """SmoothTool のバッチ/個別ピン設定・既存ピン保護・abort を検証"""

    def test_smooth_press_move_release_batch_protects_pinned(self):
        tool = SmoothTool()
        brush_opt = _make_dummy_brush_opt()
        obj = MagicMock()
        obj.taremin_cloth.brush = brush_opt
        coords = np.array(
            [
                [0.0, 0.0, 0.5],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        op = argparse.Namespace(_pinned_verts={3})
        sim = MagicMock()
        event = argparse.Namespace(mouse_region_x=100, mouse_region_y=100)
        ctx = BrushContext(
            op=op,
            context=MagicMock(),
            obj=obj,
            sim=sim,
            coords=coords,
            event=event,
            mouse_pos=(100, 100),
        )
        hit = {
            "center": np.array([0.0, 0.0, 0.5], dtype=np.float32),
            "origin": np.array([0.0, 0.0, 2.0], dtype=np.float32),
            "direction": np.array([0.0, 0.0, -1.0], dtype=np.float32),
            "hit_t": 1.5,
            "pos_2d": np.zeros((4, 3), dtype=np.float32),
        }

        with patch(
            "taremin_cloth.utils.mesh_extract.extract_cloth_mesh_data",
            return_value=_make_dummy_mesh_data(),
        ), patch(
            "taremin_cloth.brush.smooth.resolve_brush_center",
            return_value=hit,
        ):
            tool.on_press(ctx)
            self.assertTrue(sim.set_pins_batch.called)
            # 既存ピン (頂点3) にピン変更が波及しないこと
            batch_indices = set()
            for call in sim.set_pins_batch.call_args_list:
                args = call.args
                if args:
                    batch_indices.update(int(i) for i in args[0])
            self.assertNotIn(3, batch_indices)
            self.assertTrue(len(batch_indices) >= 1)

            sim.set_pins_batch.reset_mock()
            tool.on_move(ctx)
            self.assertTrue(sim.set_pins_batch.called)

            tool.on_release(ctx, {3})
            # 解放対象にも頂点3が含まれないこと
            released = {int(c.args[0]) for c in sim.release_pin.call_args_list}
            self.assertNotIn(3, released)
            self.assertTrue(len(released) >= 1)

    def test_smooth_fallback_set_pin_and_abort(self):
        tool = SmoothTool()
        brush_opt = _make_dummy_brush_opt()
        obj = MagicMock()
        obj.taremin_cloth.brush = brush_opt
        coords = np.array(
            [
                [0.0, 0.0, 0.4],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        op = argparse.Namespace(_pinned_verts={3})
        sim = MagicMock(spec=["set_pin", "release_pin"])
        event = argparse.Namespace(mouse_region_x=50, mouse_region_y=50)
        ctx = BrushContext(
            op=op,
            context=MagicMock(),
            obj=obj,
            sim=sim,
            coords=coords,
            event=event,
            mouse_pos=(50, 50),
        )
        hit = {
            "center": np.array([0.0, 0.0, 0.4], dtype=np.float32),
            "origin": np.array([0.0, 0.0, 2.0], dtype=np.float32),
            "direction": np.array([0.0, 0.0, -1.0], dtype=np.float32),
            "hit_t": 1.6,
            "pos_2d": np.zeros((4, 3), dtype=np.float32),
        }

        with patch(
            "taremin_cloth.utils.mesh_extract.extract_cloth_mesh_data",
            return_value=_make_dummy_mesh_data(),
        ), patch(
            "taremin_cloth.brush.smooth.resolve_brush_center",
            return_value=hit,
        ):
            tool.on_press(ctx)
            self.assertTrue(sim.set_pin.called)
            pinned_targets = {int(c.args[0]) for c in sim.set_pin.call_args_list}
            self.assertNotIn(3, pinned_targets)

        tool.abort(sim, {3})
        released = {int(c.args[0]) for c in sim.release_pin.call_args_list}
        self.assertNotIn(3, released)
        self.assertTrue(len(released) >= 1)


class TestGuiPreviewCoverage(unittest.TestCase):
    """TAREMIN_CLOTH_OT_gui_preview の invoke / modal / cancel を検証"""

    def setUp(self):
        self._orig_running = getattr(gui_mod, "_gui_preview_running", False)
        self._orig_instance = getattr(gui_mod, "_gui_preview_instance", None)
        gui_mod._gui_preview_running = False
        gui_mod._gui_preview_instance = None

    def tearDown(self):
        gui_mod._gui_preview_running = self._orig_running
        gui_mod._gui_preview_instance = self._orig_instance

    def test_gui_preview_invoke_modal_sync_and_cancel(self):
        cloth_obj = MagicMock()
        cloth_obj.name = "PreviewCloth"
        cloth_obj.type = "MESH"
        cloth_obj.taremin_cloth.is_cloth = True

        wm = MagicMock()
        wm.event_timer_add.return_value = "TIMER_HANDLE"

        ctx = MagicMock()
        ctx.active_object = cloth_obj
        ctx.window_manager = wm
        ctx.window = "WIN"

        mock_bpy = MagicMock()
        mock_bpy.data.objects.get.return_value = cloth_obj

        mock_client = MagicMock()
        mock_client.is_connected = True
        # 1回目は seq=1、2回目は同じ seq=1、3〜8回目も seq=1 (更新されない)、9回目で seq=2
        coords_1 = np.zeros(6, dtype=np.float32)
        coords_2 = np.ones(6, dtype=np.float32)
        mock_client.get_latest_coords.side_effect = [
            (1, coords_1, 60.0, 16.0),
            (1, coords_1, 60.0, 16.0),
            (1, coords_1, 60.0, 16.0),
            (1, coords_1, 60.0, 16.0),
            (1, coords_1, 60.0, 16.0),
            (1, coords_1, 60.0, 16.0),
            (1, coords_1, 60.0, 16.0),
            (1, coords_1, 60.0, 16.0),
            (2, coords_2, 62.0, 15.5),
        ]

        op = gui_mod.TAREMIN_CLOTH_OT_gui_preview()
        op.report = MagicMock()
        timer_event = argparse.Namespace(type="TIMER", value="NOTHING")

        with patch.object(gui_mod, "bpy", mock_bpy), \
             patch.object(gui_mod, "_apply_coords_to_mesh") as mock_apply, \
             patch("taremin_cloth.ops.gui.get_gui_client", return_value=mock_client):
            ret = op.invoke(ctx, timer_event)
            self.assertIn("RUNNING_MODAL", ret)
            self.assertEqual(op._target_obj_name, "PreviewCloth")
            self.assertTrue(gui_mod.is_gui_preview_running())
            wm.event_timer_add.assert_called_once()
            wm.modal_handler_add.assert_called_once_with(op)

            # 実行中の二重起動は CANCELLED を返すこと
            op_dup = gui_mod.TAREMIN_CLOTH_OT_gui_preview()
            op_dup.report = MagicMock()
            ret_dup = op_dup.invoke(ctx, timer_event)
            self.assertIn("CANCELLED", ret_dup)
            op_dup.report.assert_called_once()

            for _ in range(9):
                m_ret = op.modal(ctx, timer_event)
                self.assertIn("PASS_THROUGH", m_ret)

            # seq=1 (初回) と seq=2 (9回目) の2回のみ座標適用されること
            self.assertEqual(mock_apply.call_count, 2)
            # 8回周期のパラメータ同期が呼ばれていること
            self.assertTrue(mock_client.send_update_params_from_settings.called)
            op.cancel(ctx)
            wm.event_timer_remove.assert_called_once()
            self.assertFalse(gui_mod.is_gui_preview_running())

        # 未接続時は WARNING を report して CANCELLED を返すこと
        op_disconnected = gui_mod.TAREMIN_CLOTH_OT_gui_preview()
        op_disconnected.report = MagicMock()
        mock_disc_client = MagicMock()
        mock_disc_client.is_connected = False
        with patch.object(gui_mod, "bpy", mock_bpy), \
             patch("taremin_cloth.ops.gui.get_gui_client", return_value=mock_disc_client):
            ret_disc = op_disconnected.invoke(ctx, timer_event)
            self.assertIn("CANCELLED", ret_disc)
            op_disconnected.report.assert_called_once()
            self.assertIn("WARNING", op_disconnected.report.call_args.args[0])


class TestLogToolsCommandsCoverage(unittest.TestCase):
    """log_tools の未テスト CLI サブコマンド5種を実フィクスチャで検証"""

    def test_cmd_export_obj(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_obj = Path(tmpdir) / "out_f46.obj"
            args = argparse.Namespace(
                log_file=str(_FIXTURE_LOG),
                frame=46,
                output=str(out_obj),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                log_tools.cmd_export_obj(args)
            self.assertTrue(out_obj.exists())
            txt = out_obj.read_text(encoding="utf-8")
            self.assertIn("\nv ", "\n" + txt)
            self.assertIn("\nf ", "\n" + txt)

    def test_cmd_check_intersections(self):
        args = argparse.Namespace(
            log_file=str(_FIXTURE_LOG),
            frame=46,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            log_tools.cmd_check_intersections(args)
        out = buf.getvalue()
        self.assertTrue(len(out.strip()) > 0)

    def test_cmd_render_coloring(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_png = Path(tmpdir) / "coloring.png"
            args = argparse.Namespace(
                log_file=str(_FIXTURE_LOG),
                output=str(out_png),
                type="distance",
                frame=46,
                width=128,
                height=128,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                log_tools.cmd_render_coloring(args)
            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 100)

    def test_cmd_diff_same_and_perturbed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            perturbed = Path(tmpdir) / "perturbed.jsonl.gz"
            with gzip.open(_FIXTURE_LOG, "rt", encoding="utf-8") as src, \
                 gzip.open(perturbed, "wt", encoding="utf-8") as dst:
                for line in src:
                    obj = json.loads(line)
                    if obj.get("frame_index") == 47 and "positions" in obj:
                        pos = list(obj["positions"])
                        pos[0] = [pos[0][0], pos[0][1], float(pos[0][2]) + 5.0]
                        obj["positions"] = pos
                    dst.write(json.dumps(obj) + "\n")

            args_same = argparse.Namespace(
                log1=str(_FIXTURE_LOG),
                log2=str(_FIXTURE_LOG),
                tolerance=0.01,
            )
            buf_same = io.StringIO()
            with contextlib.redirect_stdout(buf_same):
                log_tools.cmd_diff(args_same)

            args_diff = argparse.Namespace(
                log1=str(_FIXTURE_LOG),
                log2=str(perturbed),
                tolerance=0.01,
            )
            buf_diff = io.StringIO()
            with contextlib.redirect_stdout(buf_diff):
                try:
                    log_tools.cmd_diff(args_diff)
                except SystemExit as exc:
                    self.assertNotEqual(exc.code, 0)
            self.assertNotEqual(buf_same.getvalue(), buf_diff.getvalue())

    def test_cmd_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_trace = Path(tmpdir) / "trace.json"
            args = argparse.Namespace(
                log_file=str(_FIXTURE_LOG),
                frame=48,
                lookback=1,
                output=str(out_trace),
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                log_tools.cmd_profile(args)
            out = buf.getvalue()
            self.assertTrue(len(out.strip()) > 0)


if __name__ == "__main__":
    unittest.main()
