"""
Blender非依存の完全再現リプレイヤー (replayer) とCLIツール (log_tools) の統合テスト
ログ生成 -> 完全再現再生 -> サブステップトレース -> スライス -> make-test -> レンダリング
"""

import os
import sys
import tempfile
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)

import taremin_cloth_core
from taremin_cloth.replayer import ClothReplayer, read_metadata, iter_frames, get_frame
from taremin_cloth import log_tools


class TestReplayerStandalone(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "standalone_sim.jsonl.gz")

        # 4頂点プレーンを生成し、球コライダーとピンを設定してシミュレーション実行・ログ記録
        positions = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ], dtype=np.float32)
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=np.uint32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        inv_masses = np.ones(4, dtype=np.float32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.005,
            stiffness=1000.0,
            bending_stiffness=10.0,
        )
        sim.set_gravity(0.0, 0.0, -9.81)

        # 球コライダー
        sim.add_sphere_collider([0.5, 0.5, 0.0], 0.4, 0.2, 0.0)

        # 動的ピン (頂点0を固定)
        sim.set_pin(0, [0.0, 0.0, 1.0], 1.0)

        # 記録開始
        sim.start_debug_recording("StandaloneTestCloth", max_frames=20)

        # 5フレーム進める
        for _ in range(5):
            sim.step(dt=1.0 / 60.0, substeps=6)

        sim.save_debug_recording(self.log_path)
        sim.stop_debug_recording()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_log_metadata_and_frames(self):
        """生成された .jsonl.gz のメタデータおよびフレームレコードの検証"""
        meta = read_metadata(self.log_path)
        self.assertEqual(meta["object_name"], "StandaloneTestCloth")
        self.assertEqual(meta["num_vertices"], 4)
        self.assertEqual(meta["gravity"], [0.0, 0.0, -9.81])

        frames = list(iter_frames(self.log_path))
        self.assertEqual(len(frames), 5)
        for idx, f in enumerate(frames):
            self.assertEqual(f["frame_index"], idx)
            # 各フレームにピンとコライダーが記録されていること
            self.assertEqual(len(f["pins"]), 1)
            self.assertEqual(f["pins"][0]["vertex_idx"], 0)
            self.assertEqual(len(f["colliders"]), 1)
            self.assertEqual(f["colliders"][0]["type"], "sphere")

    def test_replayer_exact_match(self):
        """ClothReplayer による完全再現と座標完全一致の検証"""
        replayer = ClothReplayer(self.log_path)
        results = replayer.replay_range(start_frame_idx=0, end_frame_idx=4, compare=True)

        self.assertEqual(len(results), 4) # Frame 1..4
        for res in results:
            # 同一マシン・決定論的実行のため、誤差はほぼ0（float精度以内）
            self.assertLess(res["max_diff"], 1e-4, f"Frame {res['frame_index']} で再現誤差が大きすぎます: {res['max_diff']}")

    def test_trace_substeps(self):
        """サブステップ顕微鏡解析の検証"""
        replayer = ClothReplayer(self.log_path)
        substep_traces = replayer.trace_substeps(target_frame_idx=2)

        self.assertEqual(len(substep_traces), 6) # substeps=6
        for trace in substep_traces:
            self.assertIn("max_disp_mm", trace)
            self.assertIn("max_disp_vert", trace)
            self.assertEqual(len(trace["positions"]), 4)

    def test_log_tools_cli_commands(self):
        """log_tools の slice, make-test, render コマンドの検証"""
        # 1. slice
        slice_path = os.path.join(self.temp_dir, "slice.jsonl.gz")
        args_slice = type("Args", (), {
            "log_file": self.log_path,
            "start": 1,
            "end": 3,
            "output": slice_path,
        })()
        log_tools.cmd_slice(args_slice)
        self.assertTrue(os.path.exists(slice_path))
        sliced_frames = list(iter_frames(slice_path))
        self.assertEqual(len(sliced_frames), 3)

        # 2. make-test
        test_py_path = os.path.join(self.temp_dir, "test_auto_generated.py")
        args_make_test = type("Args", (), {
            "log_file": self.log_path,
            "frame": 3,
            "lookback": 2,
            "output": test_py_path,
        })()
        log_tools.cmd_make_test(args_make_test)
        self.assertTrue(os.path.exists(test_py_path))

        # 3. render
        render_png_path = os.path.join(self.temp_dir, "render_f2.png")
        args_render = type("Args", (), {
            "log_file": self.log_path,
            "frame": 2,
            "output": render_png_path,
            "width": 200,
            "height": 150,
        })()
        log_tools.cmd_render(args_render)
        self.assertTrue(os.path.exists(render_png_path))
        self.assertTrue(os.path.getsize(render_png_path) > 0)


if __name__ == "__main__":
    unittest.main()
