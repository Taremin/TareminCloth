"""共有パイプラインキャッシュ + 遅延生成 + ビルド計測の単体テスト (Blender不要)"""

import json
import os
import subprocess
import sys
import unittest

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import taremin_cloth_core


def _grid(n=12):
    xs = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    ys = np.linspace(-0.5, 0.5, n, dtype=np.float32)
    pos = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float32)
    edges = []
    for iy in range(n):
        for ix in range(n):
            if ix + 1 < n:
                edges.append([iy * n + ix, iy * n + ix + 1])
            if iy + 1 < n:
                edges.append([iy * n + ix, (iy + 1) * n + ix])
    return pos, np.array(edges, dtype=np.uint32)


def _require_gpu(test):
    if not taremin_cloth_core.is_gpu_available():
        test.skipTest("GPU が利用不可のためスキップ (CI環境の可能性)")
    return True


_COLD_PROBE = r"""
import json, time
import numpy as np
import taremin_cloth_core as core
n = 12
xs = np.linspace(-0.5, 0.5, n, dtype=np.float32)
ys = np.linspace(-0.5, 0.5, n, dtype=np.float32)
pos = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float32)
edges = []
for iy in range(n):
    for ix in range(n):
        if ix + 1 < n: edges.append([iy*n+ix, iy*n+ix+1])
        if iy + 1 < n: edges.append([iy*n+ix, (iy+1)*n+ix])
edges = np.array(edges, dtype=np.uint32)
t0 = time.perf_counter(); s1 = core.ClothSimulator(positions=pos, edges=edges); t1 = time.perf_counter()
t2 = time.perf_counter(); s2 = core.ClothSimulator(positions=pos, edges=edges); t3 = time.perf_counter()
print(json.dumps({
    "first_ms": (t1-t0)*1000.0,
    "second_ms": (t3-t2)*1000.0,
    "first_timings": [[k, float(v)] for k, v in s1.get_build_timings()],
    "second_timings": [[k, float(v)] for k, v in s2.get_build_timings()],
    "cache": core.shared_pipeline_cache_info(),
}))
"""


class TestPipelineCache(unittest.TestCase):
    def test_cold_build_shares_cache(self):
        """冷間プロセスで初回ビルド→2個目が共有ヒットし、API計測で裏付けられる"""
        _require_gpu(self)
        proc = subprocess.run(
            [sys.executable, "-c", _COLD_PROBE],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, f"cold probe failed: {proc.stderr[-2000:]}")
        data = json.loads(proc.stdout.strip().splitlines()[-1])

        first = dict(data["first_timings"])
        second = dict(data["second_timings"])
        self.assertIn("shared_cache_build", first)
        self.assertIn("shared_cache_hit", second)
        # 2個目はパイプラインコンパイルなし (相対比較で flaky 回避)
        self.assertLess(data["second_ms"], data["first_ms"])
        self.assertLess(second["shared_cache_hit"], 100.0)

        entries = data["cache"]
        self.assertGreaterEqual(len(entries), 1)
        for key in ("epoch", "workgroup_size", "eager_ms", "lazy_built", "lazy_ms"):
            self.assertIn(key, entries[0])

    def test_build_timings_api(self):
        """get_build_timings がフェーズ別内訳を返す"""
        _require_gpu(self)
        pos, edges = _grid()
        sim = taremin_cloth_core.ClothSimulator(positions=pos, edges=edges)
        timings = dict(sim.get_build_timings())
        head = sim.get_build_timings()[0][0]
        self.assertIn(head, ("shared_cache_build", "shared_cache_hit"))
        for key in ("buffers_ms", "bind_groups_ms"):
            self.assertIn(key, timings)
            self.assertGreaterEqual(timings[key], 0.0)

    def test_mid_sim_feature_toggles(self):
        """シミュレーション途中の機能ON/OFFとsolver切替が完走する"""
        _require_gpu(self)
        pos, edges = _grid()
        sim = taremin_cloth_core.ClothSimulator(positions=pos, edges=edges)
        sim.set_enable_self_collision(True)
        sim.set_enable_pair_cache(True)
        sim.set_enable_edge_collision(True)
        sim.set_solver_mode(1)
        sim.step(dt=1.0 / 60.0, substeps=2)
        sim.set_solver_mode(0)
        sim.set_enable_edge_collision(False)
        sim.set_enable_pair_cache(False)
        sim.set_enable_self_collision(False)
        sim.step(dt=1.0 / 60.0, substeps=2)

        out = np.zeros(len(pos) * 3, dtype=np.float32)
        sim.get_positions(out)
        self.assertTrue(bool(np.isfinite(out).all()))

        info = taremin_cloth_core.shared_pipeline_cache_info()
        self.assertGreaterEqual(len(info), 1)
        self.assertIn("self_collision", info[0]["lazy_built"])

    def test_shared_cache_info_shape(self):
        """shared_pipeline_cache_info の戻り値形式"""
        _require_gpu(self)
        pos, edges = _grid()
        taremin_cloth_core.ClothSimulator(positions=pos, edges=edges)
        info = taremin_cloth_core.shared_pipeline_cache_info()
        self.assertIsInstance(info, list)
        self.assertGreaterEqual(len(info), 1)
        entry = info[0]
        self.assertIsInstance(entry["epoch"], int)
        self.assertIsInstance(entry["workgroup_size"], int)
        self.assertIsInstance(entry["eager_ms"], float)
        self.assertIsInstance(entry["lazy_built"], list)
        self.assertIsInstance(entry["lazy_ms"], float)


if __name__ == "__main__":
    unittest.main()
