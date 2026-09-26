"""デバッグ記録ファイル出力の確認フロー (Blender非依存スタンドアロンテスト)。

巨大ファイル誤出力を防ぐため、対話UI終了時は即時自動保存せず
確認ダイアログ用の待機情報 (_pending_debug_save) を登録すること、
ヘッドレス時は従来通り自動保存すること、破棄パスがバッファを解放することを検証する。
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

# tests パッケージのbpyモックを先に有効化する
root_dir = Path(__file__).parent.parent.parent
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import tests  # noqa: F401  (bpyモック登録の副作用用)
import bpy

from taremin_cloth.ops import interactive as inter_mod


class FakeSim:
    """デバッグ記録APIだけを持つ軽量シミュレータスタブ"""

    def __init__(self, frames=5):
        self._recording = False
        self._frames = int(frames)
        self.saved_to = None
        self.stopped = False

    def start_debug_recording(self, name, max_frames=None):
        self._recording = True
        self.stopped = False

    def is_debug_recording(self):
        return self._recording

    def get_debug_frame_count(self):
        return self._frames if self._recording else 0

    def save_debug_recording(self, filepath):
        self.saved_to = filepath
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("fake-debug-recording\n")
        return filepath

    def stop_debug_recording(self):
        self._recording = False
        self.stopped = True


def _headless_context():
    ctx = types.SimpleNamespace(window_manager=None, window=None)
    return ctx


def _ui_context():
    wm = types.SimpleNamespace(
        invoke_props_dialog=lambda *a, **k: {'RUNNING_MODAL'},
    )
    return types.SimpleNamespace(window_manager=wm, window=object())


class TestFormatBytes(unittest.TestCase):
    def test_basic_units(self):
        self.assertEqual(inter_mod.format_bytes(0), "0 B")
        self.assertEqual(inter_mod.format_bytes(512), "512 B")
        self.assertEqual(inter_mod.format_bytes(1024), "1.0 KB")
        self.assertEqual(inter_mod.format_bytes(1536), "1.5 KB")
        self.assertEqual(inter_mod.format_bytes(5 * 1024 * 1024), "5.0 MB")
        self.assertEqual(inter_mod.format_bytes(3 * 1024**3), "3.0 GB")

    def test_invalid(self):
        self.assertEqual(inter_mod.format_bytes(-1), "unknown")
        self.assertEqual(inter_mod.format_bytes(None), "unknown")


class TestEstimateRaw(unittest.TestCase):
    def test_user_heavy_file(self):
        # 実ファイル (11618頂点・50フレーム・55ボーン・256^3相当、非圧縮実測約140MB)
        raw = inter_mod.estimate_debug_raw_bytes(50, 11618, 1, 16777216, 55)
        self.assertGreater(raw, 120000000)
        self.assertLess(raw, 150000000)

    def test_calibration_grid(self):
        # 実測 (40x40・30フレーム、非圧縮実測約3.17MB)
        raw = inter_mod.estimate_debug_raw_bytes(30, 1600)
        self.assertGreater(raw, 3000000)
        self.assertLess(raw, 4500000)

    def test_invalid_inputs(self):
        self.assertEqual(inter_mod.estimate_debug_raw_bytes(0, 10), 0)
        self.assertEqual(inter_mod.estimate_debug_raw_bytes(10, 0), 0)
        self.assertEqual(inter_mod.estimate_debug_raw_bytes("x", 10), 0)
        self.assertEqual(inter_mod.estimate_debug_raw_bytes(10, 10, 0), inter_mod.estimate_debug_raw_bytes(10, 10, 1))
        self.assertEqual(inter_mod.estimate_debug_raw_bytes(10, 10, 1, -5, -2), inter_mod.estimate_debug_raw_bytes(10, 10, 1, 0, 0))


class TestEstimate(unittest.TestCase):
    def test_full_recording_near_measured(self):
        # 実測 (40x40グリッド・30フレーム・自由落下で約242KB) に対し
        # 安全側の過大概算になること
        est = inter_mod.estimate_debug_file_bytes(30, 1600)
        self.assertGreater(est, 400000)
        self.assertLess(est, 900000)

    def test_sparse_stride_reduces_estimate(self):
        full = inter_mod.estimate_debug_file_bytes(120, 1600, 1)
        sparse = inter_mod.estimate_debug_file_bytes(120, 1600, 5)
        # 実測 sparse (約214KB) に対する概算は安全側の過大評価
        self.assertLess(sparse, full)
        self.assertGreater(sparse, 400000)
        self.assertLess(sparse, 750000)

    def test_bone_sdf_dominates(self):
        # 実測 (11618頂点・50フレーム・55ボーン・256^3相当で約61.75MB) に対し
        # 次数が合う概算になること
        est = inter_mod.estimate_debug_file_bytes(50, 11618, 1, 16777216, 55)
        self.assertGreater(est, 50000000)
        self.assertLess(est, 65000000)

    def test_no_bone_by_default(self):
        self.assertEqual(
            inter_mod.estimate_debug_file_bytes(10, 100),
            inter_mod.estimate_debug_file_bytes(10, 100, 1, 0, 0),
        )

    def test_invalid_inputs(self):
        self.assertEqual(inter_mod.estimate_debug_file_bytes(0, 10), 0)
        self.assertEqual(inter_mod.estimate_debug_file_bytes(10, 0), 0)
        self.assertEqual(inter_mod.estimate_debug_file_bytes("x", 10), 0)
        self.assertEqual(inter_mod.estimate_debug_file_bytes(10, 10, 0), inter_mod.estimate_debug_file_bytes(10, 10, 1))
        self.assertEqual(inter_mod.estimate_debug_file_bytes(10, 10, 1, -5, -2), inter_mod.estimate_debug_file_bytes(10, 10, 1, 0, 0))


class TestSimVertexCount(unittest.TestCase):
    """頂点数はRust APIを正とし、旧ビルドではcoordsフォールバックに倒れること"""

    def test_rust_api_primary(self):
        sim = FakeSim()
        sim.get_num_vertices = 42  # #[getter]相当 (非callable属性値)
        self.assertEqual(inter_mod._sim_vertex_count(sim, [0.0] * 30), 42)

    def test_coords_fallback_without_api(self):
        sim = FakeSim()  # get_num_verticesを持たない旧ビルド相当
        self.assertEqual(inter_mod._sim_vertex_count(sim, [0.0] * 30), 10)

    def test_zero_api_falls_back_to_coords(self):
        sim = FakeSim()
        sim.get_num_vertices = 0
        self.assertEqual(inter_mod._sim_vertex_count(sim, [0.0] * 30), 10)

    def test_shaped_coords(self):
        import numpy as np
        sim = FakeSim()
        self.assertEqual(
            inter_mod._sim_vertex_count(sim, np.zeros((7, 3), dtype=np.float32)), 7
        )

    def test_unresolvable_returns_zero(self):
        self.assertEqual(inter_mod._sim_vertex_count(FakeSim()), 0)


class TestDebugSaveConfirmationFlow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="taremin_cloth_confirm_")
        # bpy.app.background の現在値を退避
        try:
            self._orig_background = bpy.app.background
        except Exception:
            self._orig_background = None
        inter_mod.clear_pending_debug_save()
        # 他テストの残骸があれば除去
        inter_mod._simulators.pop("TestObj", None)

    def tearDown(self):
        try:
            if self._orig_background is not None:
                bpy.app.background = self._orig_background
        except Exception:
            pass
        inter_mod.clear_pending_debug_save()
        inter_mod._simulators.pop("TestObj", None)
        if os.path.isdir(self.temp_dir):
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except Exception:
                        pass
            try:
                os.rmdir(self.temp_dir)
            except Exception:
                pass

    def test_headless_auto_saves(self):
        """ヘッドレス時は従来通り即時自動保存され、待機情報が残らないこと"""
        bpy.app.background = True
        sim = FakeSim(frames=5)
        sim.start_debug_recording("TestObj")
        inter_mod._simulators["TestObj"] = (sim, [0.0] * (50 * 3))
        filepath = os.path.join(self.temp_dir, "headless.jsonl.gz")

        res = inter_mod.request_debug_save_confirmation(
            _headless_context(), "TestObj", sim, 5, filepath
        )
        self.assertEqual(res, 'fallback-saved')
        self.assertTrue(os.path.isfile(filepath))
        self.assertIsNone(inter_mod.get_pending_debug_save())
        self.assertTrue(sim.stopped)

    def test_ui_defers_and_discards(self):
        """対話UI時は保存せず待機登録し、破棄でバッファが解放されること"""
        bpy.app.background = False
        sim = FakeSim(frames=7)
        sim.start_debug_recording("TestObj")
        inter_mod._simulators["TestObj"] = (sim, [0.0] * (200 * 3))
        filepath = os.path.join(self.temp_dir, "ui.jsonl.gz")

        res = inter_mod.request_debug_save_confirmation(
            _ui_context(), "TestObj", sim, 7, filepath
        )
        self.assertEqual(res, 'deferred')
        # 自動保存されていないこと
        self.assertFalse(os.path.exists(filepath))
        self.assertTrue(sim.is_debug_recording())
        pending = inter_mod.get_pending_debug_save()
        self.assertIsNotNone(pending)
        self.assertEqual(pending["obj_name"], "TestObj")
        self.assertEqual(pending["frame_count"], 7)
        self.assertEqual(pending["vertex_count"], 200)
        self.assertEqual(pending["full_stride"], 1)
        self.assertEqual(pending["bone_voxels"], 0)
        self.assertEqual(pending["bone_count"], 0)

        # 確認オペレーターのpollは待機ありでTrue
        self.assertTrue(inter_mod.TAREMIN_CLOTH_OT_confirm_debug_save.poll(_ui_context()))

        # 破棄パス
        self.assertTrue(inter_mod.discard_pending_debug_recording(reason="test"))
        self.assertTrue(sim.stopped)
        self.assertIsNone(inter_mod.get_pending_debug_save())

    def test_pending_bone_info_from_cache(self):
        """_bone_sdf_cacheからテクスチャ寸法・ボーン数がpendingに載ること"""
        from taremin_cloth.engine import collider as collider_mod

        class FakeBake:
            width = 256
            height = 256
            depth = 256
            bone_infos = [[0.0] * 20 for _ in range(55)]

        bpy.app.background = False
        sim = FakeSim(frames=7)
        sim.start_debug_recording("TestObj")
        inter_mod._simulators["TestObj"] = (sim, [0.0] * (200 * 3))
        collider_mod._bone_sdf_cache[id(sim)] = (None, FakeBake(), 'STATIC')
        try:
            filepath = os.path.join(self.temp_dir, "ui_bone.jsonl.gz")
            res = inter_mod.request_debug_save_confirmation(
                _ui_context(), "TestObj", sim, 7, filepath
            )
            self.assertEqual(res, 'deferred')
            pending = inter_mod.get_pending_debug_save()
            self.assertEqual(pending["bone_voxels"], 256 * 256 * 256)
            self.assertEqual(pending["bone_count"], 55)
            est = inter_mod.estimate_debug_file_bytes(7, 200, 1, 256**3, 55)
            self.assertGreater(est, 40000000)
        finally:
            collider_mod._bone_sdf_cache.pop(id(sim), None)

    def test_ui_defers_with_coords(self):
        """coordsから頂点数が求まり待機登録されること"""
        bpy.app.background = False
        sim = FakeSim(frames=7)
        sim.start_debug_recording("TestObj")
        inter_mod._simulators["TestObj"] = (sim, [0.0] * (200 * 3))
        filepath = os.path.join(self.temp_dir, "ui_coords.jsonl.gz")

        res = inter_mod.request_debug_save_confirmation(
            _ui_context(), "TestObj", sim, 7, filepath
        )
        self.assertEqual(res, 'deferred')
        pending = inter_mod.get_pending_debug_save()
        self.assertIsNotNone(pending)
        self.assertEqual(pending["vertex_count"], 200)

    def test_confirm_operator_execute_discard(self):
        """確認ダイアログのキャンセル (cancel) でファイル出力なくバッファが解放されること"""
        bpy.app.background = False
        sim = FakeSim(frames=3)
        sim.start_debug_recording("TestObj")
        inter_mod._simulators["TestObj"] = (sim, [0.0] * (10 * 3))
        filepath = os.path.join(self.temp_dir, "discard.jsonl.gz")
        inter_mod.request_debug_save_confirmation(_ui_context(), "TestObj", sim, 3, filepath)

        op = object.__new__(inter_mod.TAREMIN_CLOTH_OT_confirm_debug_save)
        op.obj_name = "TestObj"
        op.filepath = filepath
        op.frame_count = 3
        op.vertex_count = 10
        op.report = lambda *a, **k: None

        op.cancel(_ui_context())
        self.assertFalse(os.path.exists(filepath))
        self.assertTrue(sim.stopped)
        self.assertIsNone(inter_mod.get_pending_debug_save())

    def test_confirm_operator_execute_save(self):
        """確認ダイアログのOK (execute) でファイルが出力されバッファが解放されること"""
        bpy.app.background = False
        sim = FakeSim(frames=4)
        sim.start_debug_recording("TestObj")
        inter_mod._simulators["TestObj"] = (sim, [0.0] * (10 * 3))
        filepath = os.path.join(self.temp_dir, "saved.jsonl.gz")
        inter_mod.request_debug_save_confirmation(_ui_context(), "TestObj", sim, 4, filepath)

        op = object.__new__(inter_mod.TAREMIN_CLOTH_OT_confirm_debug_save)
        op.obj_name = "TestObj"
        op.filepath = filepath
        op.frame_count = 4
        op.vertex_count = 10
        op.report = lambda *a, **k: None

        res = op.execute(_ui_context())
        self.assertEqual(res, {'FINISHED'})
        self.assertTrue(os.path.isfile(filepath))
        self.assertTrue(sim.stopped)
        self.assertIsNone(inter_mod.get_pending_debug_save())

    def test_confirm_operator_rejects_stale_buffer(self):
        """ダイアログ表示後に入れ替わった別セッションのバッファを誤保存しないこと"""
        bpy.app.background = False
        sim = FakeSim(frames=4)
        sim.start_debug_recording("TestObj")
        inter_mod._simulators["TestObj"] = (sim, [0.0] * (10 * 3))
        filepath = os.path.join(self.temp_dir, "stale.jsonl.gz")
        inter_mod.request_debug_save_confirmation(_ui_context(), "TestObj", sim, 4, filepath)

        # 新規セッション開始でバッファが増えた状況を再現
        sim._frames = 9

        op = object.__new__(inter_mod.TAREMIN_CLOTH_OT_confirm_debug_save)
        op.obj_name = "TestObj"
        op.filepath = filepath
        op.frame_count = 4
        op.vertex_count = 10
        reports = []
        op.report = lambda *a, **k: reports.append(a)

        res = op.execute(_ui_context())
        self.assertEqual(res, {'CANCELLED'})
        self.assertFalse(os.path.exists(filepath))
        self.assertTrue(sim.is_debug_recording())


if __name__ == "__main__":
    unittest.main()
