import os
import sys
import gzip
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

# プロジェクトルートと python ディレクトリをパスに追加
root_dir = Path(__file__).parent.parent
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import taremin_cloth_core
from taremin_cloth.operators import resolve_debug_filepath


class TestDebugRecorder(unittest.TestCase):
    """シミュレーションデバッグ状態記録機能のユニットテスト"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="taremin_test_debug_")

    def tearDown(self):
        # 作成した一時ディレクトリとファイルを確実にクリーンアップ
        if os.path.exists(self.temp_dir):
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

    def test_simulator_debug_recording_lifecycle(self):
        """ClothSimulatorでのデバッグ記録開始、フレーム進行、ファイル保存、停止のライフサイクルテスト"""
        # 単純な平面（4頂点、2面）
        positions = np.array([
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ], dtype=np.float32)
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0], [0, 2]
        ], dtype=np.uint32)
        faces = np.array([
            [0, 1, 2], [0, 2, 3]
        ], dtype=np.uint32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            faces=faces,
            workgroup_size=32,
            solver_mode=0,
        )

        self.assertFalse(sim.is_debug_recording())
        self.assertEqual(sim.get_debug_frame_count(), 0)

        # 記録開始
        sim.start_debug_recording("TestClothMesh", max_frames=10)
        self.assertTrue(sim.is_debug_recording())

        # 3ステップ実行
        for _ in range(3):
            sim.step(dt=1.0 / 60.0, substeps=10)

        self.assertEqual(sim.get_debug_frame_count(), 3)

        # ファイル保存
        output_file = os.path.join(self.temp_dir, "test_trace.json.gz")
        saved_path = sim.save_debug_recording(output_file)
        self.assertTrue(os.path.exists(saved_path))
        self.assertTrue(os.path.getsize(saved_path) > 0)

        # gzip解凍とJSON検証
        with gzip.open(saved_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["version"], 1)
        self.assertEqual(data["metadata"]["object_name"], "TestClothMesh")
        self.assertEqual(data["metadata"]["num_vertices"], 4)
        self.assertEqual(data["metadata"]["num_edges"], len(data["metadata"]["edges"]))
        self.assertEqual(data["metadata"]["num_faces"], 2)
        self.assertEqual(len(data["frames"]), 3)

        # 各フレームの構造検証
        for idx, frame in enumerate(data["frames"]):
            self.assertEqual(frame["frame_index"], idx)
            self.assertAlmostEqual(frame["dt"], 1.0 / 60.0, places=5)
            self.assertEqual(frame["substeps"], 10)
            self.assertEqual(len(frame["positions"]), 4)
            self.assertEqual(len(frame["velocities"]), 4)
            self.assertIn("stats", frame)
            stats = frame["stats"]
            self.assertFalse(stats["has_nan_or_inf"])
            self.assertIn("aabb_min", stats)
            self.assertIn("aabb_max", stats)

        # 停止とクリア
        sim.stop_debug_recording()
        self.assertFalse(sim.is_debug_recording())
        self.assertEqual(sim.get_debug_frame_count(), 0)

    def test_resolve_debug_filepath_template(self):
        """ファイル名テンプレート解決関数の検証"""
        class DummyPrefs:
            debug_output_dir = self.temp_dir
            debug_filename_template = "dump_{date}_{time}_{object}_{frames}f.{ext}"

        filepath = resolve_debug_filepath(DummyPrefs(), "Cloth:Special/1", frame_count=120, ext="json.gz")
        self.assertTrue(filepath.startswith(self.temp_dir))
        filename = os.path.basename(filepath)

        # 特殊文字 (:) や (/) がサニタイズされていること
        self.assertIn("Cloth_Special_1", filename)
        self.assertIn("120f", filename)
        self.assertTrue(filename.endswith(".json.gz"))

    def test_max_frames_protection(self):
        """最大記録フレーム数（メモリ保護上限）の動作検証"""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        edges = np.array([[0, 1]], dtype=np.uint32)

        sim = taremin_cloth_core.ClothSimulator(positions=positions, edges=edges)

        # 上限を2フレームに設定
        sim.start_debug_recording("ShortSim", max_frames=2)

        for _ in range(5):
            sim.step(dt=1.0 / 60.0, substeps=5)

        # 上限の2フレームで止まっていること
        self.assertEqual(sim.get_debug_frame_count(), 2)
        sim.stop_debug_recording()


if __name__ == "__main__":
    unittest.main()
