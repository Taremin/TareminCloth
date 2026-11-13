"""tools/parallel_unittest.py の単体・統合テスト (Blender不要)"""

import sys
import tempfile
import unittest
from pathlib import Path

from tools.parallel_unittest import (
    REPO_ROOT,
    TestUnit,
    _is_blender_required_file,
    _run_targets_via_temp_file,
    discover_test_units,
    resolve_worker_count,
    run_parallel_python_tests,
)


class TestParallelUnittestDiscovery(unittest.TestCase):
    def test_keeps_setup_class_and_ordered_classes_grouped(self):
        """setUpClass や連番テストを持つクラスはメソッド分割されずクラス単位維持になること"""
        units = discover_test_units(
            REPO_ROOT,
            pattern="test_gui_ipc.py",
            granularity="auto",
            scope="all",
        )
        targets = [u.target for u in units]
        self.assertEqual(targets, ["tests.core.test_gui_ipc.TestGuiIpc"])

    def test_splits_independent_heavy_classes_into_methods_and_sorts_gpu_reset_last(self):
        """独立クラスはメソッド分割され、set_gpu_device を呼ぶテストが末尾優先度になること"""
        demo_units = discover_test_units(
            REPO_ROOT,
            pattern="test_demo_scenes.py",
            granularity="auto",
            scope="all",
        )
        self.assertEqual(len(demo_units), 8)
        self.assertTrue(
            all(u.target.startswith("tests.physics.test_demo_scenes.TestDemoScenes.test_") for u in demo_units)
        )

        core_units = discover_test_units(
            REPO_ROOT,
            pattern="test_cloth_core.py",
            granularity="auto",
            scope="all",
        )
        self.assertEqual(len(core_units), 17)
        self.assertEqual(
            core_units[-1].target,
            "tests.core.test_cloth_core.TestClothCoreBasic.test_gpu_devices_and_backend_selection",
        )
        self.assertEqual(core_units[-1].priority, 3)

    def test_standalone_discovery_matches_unittest_discover_count(self):
        """standalone スコープの収集総テスト数が従来の unittest discover と完全一致すること"""
        units = discover_test_units(REPO_ROOT, granularity="auto", scope="standalone")
        loader = unittest.TestLoader()
        parallel_total = sum(loader.loadTestsFromName(u.target).countTestCases() for u in units)

        traditional_total = 0
        for d in ("tests/core", "tests/physics", "tests/collider", "tests/interactive", "tests/regression"):
            traditional_total += loader.discover(d, pattern="test_*.py", top_level_dir=".").countTestCases()

        self.assertEqual(parallel_total, traditional_total)

    def test_blender_scope_excludes_non_blender_tests(self):
        """blender スコープは e2e と _IS_REAL_BLENDER 依存ファイルのみを正確に抽出すること"""
        units = discover_test_units(REPO_ROOT, granularity="file", scope="blender")
        rel_paths = {u.file_path.relative_to(REPO_ROOT).as_posix() for u in units}

        all_units = discover_test_units(REPO_ROOT, granularity="file", scope="all")
        expected_blender_paths = {
            u.file_path.relative_to(REPO_ROOT).as_posix()
            for u in all_units
            if _is_blender_required_file(
                u.file_path, u.file_path.read_text(encoding="utf-8", errors="ignore")
            )
        }
        self.assertEqual(rel_paths, expected_blender_paths)
        self.assertIn("tests/e2e/test_simulation_e2e.py", rel_paths)
        self.assertIn("tests/interactive/test_panels.py", rel_paths)
        self.assertIn("tests/physics/test_adaptive_substep.py", rel_paths)
        self.assertNotIn("tests/core/test_cloth_core.py", rel_paths)
        self.assertNotIn("tests/core/test_i18n.py", rel_paths)

    def test_keep_class_directive_does_not_leak_to_following_class(self):
        """# parallel: keep-class は対象クラス直上連続コメント/本体のみに効き、直前クラスから漏れないこと"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pkg_dir = Path(tmp_dir) / "keep_tests"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
            (pkg_dir / "test_keep_boundary.py").write_text(
                (
                    "import unittest\n"
                    "class TestFirst(unittest.TestCase):\n"
                    "    def test_a(self):\n"
                    "        pass\n"
                    "    # parallel: keep-class\n"
                    "    def test_b(self):\n"
                    "        pass\n"
                    "\n"
                    "class TestSecond(unittest.TestCase):\n"
                    "    def test_c(self):\n"
                    "        pass\n"
                    "    def test_d(self):\n"
                    "        pass\n"
                    "\n"
                    "# parallel: keep-class\n"
                    "class TestThird(unittest.TestCase):\n"
                    "    def test_e(self):\n"
                    "        pass\n"
                    "    def test_f(self):\n"
                    "        pass\n"
                ),
                encoding="utf-8",
            )
            units = discover_test_units(
                Path(tmp_dir),
                test_roots=[pkg_dir],
                granularity="auto",
                scope="all",
            )
            targets = [u.target for u in units]
            self.assertIn("keep_tests.test_keep_boundary.TestFirst", targets)
            self.assertIn("keep_tests.test_keep_boundary.TestSecond.test_c", targets)
            self.assertIn("keep_tests.test_keep_boundary.TestSecond.test_d", targets)
            self.assertIn("keep_tests.test_keep_boundary.TestThird", targets)
            self.assertNotIn("keep_tests.test_keep_boundary.TestSecond", targets)

    def test_pattern_matching_supports_glob_and_module_names(self):
        """glob パターンやドット区切りモジュール名で正しくフィルタできること"""
        e2e_units = discover_test_units(REPO_ROOT, pattern="*_e2e.py", granularity="file", scope="all")
        e2e_names = {u.file_path.name for u in e2e_units}
        self.assertIn("test_simulation_e2e.py", e2e_names)
        self.assertIn("test_bone_sdf_e2e.py", e2e_names)

        dot_units = discover_test_units(
            REPO_ROOT,
            pattern="tests.core.test_quick_pinning",
            granularity="file",
            scope="all",
        )
        self.assertEqual([u.target for u in dot_units], ["tests.core.test_quick_pinning"])

    def test_resolve_worker_count_bounds(self):
        """ワーカー数解決がユニット数および要求値に従って適切に丸められること"""
        self.assertEqual(resolve_worker_count(requested=None, unit_count=1), 1)
        self.assertEqual(resolve_worker_count(requested=8, unit_count=3), 3)
        self.assertEqual(resolve_worker_count(requested=1, unit_count=10), 1)


class TestParallelUnittestExecution(unittest.TestCase):
    def test_parallel_pool_pass_and_temp_isolation(self):
        """2ワーカー並列実行で全件成功し、各ワーカーのTEMPディレクトリが隔離されていること"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pkg_dir = Path(tmp_dir) / "sample_tests"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
            (pkg_dir / "test_alpha.py").write_text(
                (
                    "import os, tempfile, unittest\n"
                    "class TestAlpha(unittest.TestCase):\n"
                    "    def test_one(self):\n"
                    "        self.assertTrue(os.path.isdir(tempfile.gettempdir()))\n"
                    "    def test_two(self):\n"
                    "        self.assertEqual(os.environ['TEMP'], tempfile.gettempdir())\n"
                ),
                encoding="utf-8",
            )
            (pkg_dir / "test_beta.py").write_text(
                (
                    "import unittest\n"
                    "class TestBeta(unittest.TestCase):\n"
                    "    @unittest.skip('expected skip')\n"
                    "    def test_skip(self):\n"
                    "        pass\n"
                    "    def test_pass(self):\n"
                    "        self.assertEqual(1 + 1, 2)\n"
                ),
                encoding="utf-8",
            )

            rc = run_parallel_python_tests(
                Path(tmp_dir),
                test_roots=[pkg_dir],
                workers=2,
                granularity="method",
                scope="all",
                verbosity=0,
            )
            self.assertEqual(rc, 0)

    def test_parallel_pool_reports_failure_exit_code(self):
        """失敗テストが含まれる場合に非0の終了コードを返すこと"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pkg_dir = Path(tmp_dir) / "failing_tests"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
            (pkg_dir / "test_fail_case.py").write_text(
                (
                    "import unittest\n"
                    "class TestFailCase(unittest.TestCase):\n"
                    "    def test_ok(self):\n"
                    "        self.assertTrue(True)\n"
                    "    def test_ng(self):\n"
                    "        self.fail('intentional failure')\n"
                ),
                encoding="utf-8",
            )

            rc = run_parallel_python_tests(
                Path(tmp_dir),
                test_roots=[pkg_dir],
                workers=2,
                granularity="method",
                scope="all",
                verbosity=0,
            )
            self.assertEqual(rc, 1)

    def test_sequential_subprocess_handles_many_targets_without_cli_overflow(self):
        """多数のターゲット(Windows CLI長上限32KB超)でも一時targets.json経由で正常実行できること"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pkg_dir = Path(tmp_dir) / "many_tests"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
            test_file = pkg_dir / "test_many.py"
            test_file.write_text(
                (
                    "import unittest\n"
                    "class TestManyTargetCase(unittest.TestCase):\n"
                    "    def test_ok(self):\n"
                    "        self.assertTrue(True)\n"
                ),
                encoding="utf-8",
            )
            units = [
                TestUnit(
                    target="many_tests.test_many.TestManyTargetCase.test_ok",
                    file_path=test_file,
                    priority=0,
                    weight=1,
                    requires_blender=False,
                )
                for _ in range(1200)
            ]
            # 1200 件 * ~48 文字 ≒ 57KB (> Windows 32KB コマンドライン制限) でも成功すること
            rc = _run_targets_via_temp_file(
                units,
                cmd_prefix=[sys.executable],
                cwd=Path(tmp_dir),
                verbosity=0,
                pass_separator=False,
                isolate_blender_config=False,
                prefix="taremin_test_many_",
            )
            self.assertEqual(rc, 0)

    def test_bone_sdf_baker_classified_as_gpu_unit(self):
        """bake_bone_sdf 呼び出しテストが GPU ユニットとして正しく分類されること"""
        units = discover_test_units(
            REPO_ROOT,
            pattern="test_bone_sdf_baker.py",
            granularity="file",
            scope="standalone",
        )
        self.assertEqual(len(units), 1)
        self.assertTrue(units[0].uses_gpu)
        self.assertEqual(units[0].priority, 1)

    def test_software_gpu_lane_runs_gpu_units_on_single_worker_and_retires_on_device_lost(self):
        """Software GPU モードでは GPU ユニットを専任ワーカーで順次実行し、Device Lost 後は別プロセスへ交代すること"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pkg_dir = tmp_path / "lane_tests"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
            log_file = tmp_path / "pids.log"

            # GPU ユニット群 (1・2件目は同一ワーカー PID で実行され、Device Lost 後は別 PID へ交代)
            (pkg_dir / "test_gpu_sim.py").write_text(
                (
                    "import os, unittest\n"
                    "import taremin_cloth_core  # noqa: F401\n"
                    f"LOG_PATH = {str(log_file)!r}\n"
                    "class TestGpuFirst(unittest.TestCase):\n"
                    "    def test_first(self):\n"
                    "        with open(LOG_PATH, 'a', encoding='utf-8') as f:\n"
                    "            f.write(f'gpu_1:{os.getpid()}\\n')\n"
                    "        self.assertTrue(True)\n"
                    "class TestGpuLost(unittest.TestCase):\n"
                    "    def test_lost(self):\n"
                    "        with open(LOG_PATH, 'a', encoding='utf-8') as f:\n"
                    "            f.write(f'gpu_lost:{os.getpid()}\\n')\n"
                    "        self.fail('Parent device is lost')\n"
                    "class TestGpuRecover(unittest.TestCase):\n"
                    "    def test_ok(self):\n"
                    "        with open(LOG_PATH, 'a', encoding='utf-8') as f:\n"
                    "            f.write(f'gpu_ok:{os.getpid()}\\n')\n"
                    "        self.assertTrue(True)\n"
                ),
                encoding="utf-8",
            )
            (pkg_dir / "test_cpu_work.py").write_text(
                (
                    "import os, unittest\n"
                    f"LOG_PATH = {str(log_file)!r}\n"
                    "class TestCpuLane(unittest.TestCase):\n"
                    "    def test_cpu(self):\n"
                    "        with open(LOG_PATH, 'a', encoding='utf-8') as f:\n"
                    "            f.write(f'cpu:{os.getpid()}\\n')\n"
                    "        self.assertTrue(True)\n"
                ),
                encoding="utf-8",
            )

            rc = run_parallel_python_tests(
                tmp_path,
                test_roots=[pkg_dir],
                workers=2,
                granularity="method",
                scope="all",
                verbosity=0,
                serialize_gpu=True,
            )
            self.assertEqual(rc, 1)
            records = dict(
                line.split(":", 1)
                for line in log_file.read_text(encoding="utf-8").splitlines()
                if ":" in line
            )
            self.assertIn("gpu_1", records)
            self.assertIn("gpu_lost", records)
            self.assertIn("gpu_ok", records)
            self.assertIn("cpu", records)
            # 同一専任 GPU ワーカーが連続処理すること
            self.assertEqual(records["gpu_1"], records["gpu_lost"])
            # Device Lost 発生後、後続の GPU ユニットは交代した別プロセス (PID) で実行されること
            self.assertNotEqual(records["gpu_lost"], records["gpu_ok"])
            self.assertNotEqual(records["gpu_1"], records["cpu"])


if __name__ == "__main__":
    unittest.main()
