#!/usr/bin/env python3
"""
Taremin Cloth 共通並列 unittest ランナー基盤。

特徴:
1. AST ベースのハイブリッド粒度分割:
   - setUpClass / tearDownClass / 連番テスト (test_01_ 等) を持つクラスはクラス単位を維持
   - 独立した TestCase メソッドは個別メソッド単位に分割して負荷を平準化
2. GPU パイプラインキャッシュ (PIPELINE_CACHE) に最適化した常駐ワーカープール:
   - Python / Blender ともに N 個の常駐ワーカープロセスを再利用し、
     プロセス内でコンパイル済みパイプラインを再利用してシェーダ再コンパイルを最小化
   - GPU コンテキスト再初期化を含むテストを末尾にスケジュールしキャッシュ無効化を回避
3. ローカル TCP キューサーバーによる統一プロトコル:
   - Python テストと Blender バックグラウンドテストで同一の動的配布・クラッシュ監視・
     タイムアウト制御・一時ディレクトリ隔離 (TEMP / TMP / BLENDER_USER_CONFIG) を共有
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fnmatch
import io
import json
import logging
import os
import re
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# リポジトリルートおよび作業ディレクトリを sys.path に自動追加（Blender --python 実行時等に対応）
REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(Path.cwd())):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Windows 環境の文字化け防止
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# 既知の重量級テストファイル（LPT スケジューリングで優先投入）
_HEAVY_TEST_FILES: Tuple[str, ...] = (
    "test_demo_scenes.py",
    "test_cloth_core.py",
    "test_simulation_e2e.py",
    "test_pipeline_cache.py",
    "test_coupled_self_collision.py",
    "test_internal_ee_ccd.py",
    "test_async_readback.py",
    "test_dynamic_sdf_collider.py",
    "test_gui_ipc.py",
    "test_mesh_sdf_baker.py",
    "test_pair_caching.py",
    "test_pair_audit.py",
    "test_multi_collider_options.py",
    "test_simconfig_sync.py",
    "test_garment_standalone.py",
    "test_golden_regression.py",
    "test_issue_f23_f25_penetration.py",
    "test_issue_penetration_replay.py",
    "test_single_cloth_untangling.py",
    "test_wrinkle_inversion_guard.py",
    "test_waypoint_sewing.py",
    "test_waypoint_release.py",
    "test_sparse_recording.py",
    "test_substep_decoupling.py",
    "test_string_drawstring.py",
    "test_mesh_triangle_cluster_culling.py",
    "test_sewing_lock_distance.py",
    "test_wrinkle_field_standalone.py",
)
_HEAVY_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(_HEAVY_TEST_FILES)}

# GPU デバイス世代 (DEVICE_EPOCH) を進め、プロセス内キャッシュを無効化するテストメソッド
_GPU_RESET_METHODS: Tuple[str, ...] = ("test_gpu_devices_and_backend_selection",)

# スタンドアロン (--ci) テストの標準ルートディレクトリ
DEFAULT_STANDALONE_DIRS: Tuple[str, ...] = (
    "tests/core",
    "tests/physics",
    "tests/collider",
    "tests/interactive",
    "tests/regression",
)

# 連番依存メソッド判定パターン (例: test_01_connect)
_ORDERED_METHOD_RE = re.compile(r"^test_\d+")


@dataclass(frozen=True)
class TestUnit:
    """並列ディスパッチの最小実行単位"""

    target: str
    file_path: Path
    priority: int
    weight: int
    requires_blender: bool
    uses_gpu: bool = False


@dataclass
class UnitResult:
    """1 テストユニットの実行結果"""

    target: str
    worker_id: str
    ok: bool
    tests_run: int
    skipped: int
    elapsed: float
    output: str = ""
    error_summary: str = ""


def _normalize_rel_path(path: Path, repo_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
        return rel.as_posix()
    except Exception:
        return path.as_posix()


def _matches_pattern(file_path: Path, *, repo_root: Path, pattern: str) -> bool:
    """ファイル名・相対パス・glob・ドット区切りモジュール指定に柔軟一致させる"""
    pat = pattern.strip().replace("\\", "/")
    if not pat:
        return True

    rel_posix = _normalize_rel_path(file_path, repo_root)
    name = file_path.name
    stem = file_path.stem
    mod_name = ".".join(Path(rel_posix).with_suffix("").parts)

    # ドット区切りモジュール指定 (例: tests.core.test_foo)
    if pat == mod_name or pat.startswith(mod_name + "."):
        return True

    # 完全一致 / パス一致 / glob 一致
    if pat in (name, stem, rel_posix):
        return True
    if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel_posix, pat):
        return True

    # tests/core/foo.py 形式の部分一致
    if not pat.endswith(".py") and fnmatch.fnmatch(stem, pat):
        return True

    return False


_BLENDER_SKIP_RE = re.compile(r"skipUnless\([^\n]*_IS_REAL_BLENDER")


def _is_blender_required_file(file_path: Path, text: str) -> bool:
    """実 Blender 環境での実行が必須のテストファイルかを判定する"""
    if "e2e" in file_path.parts:
        return True
    if _BLENDER_SKIP_RE.search(text):
        return True
    return False


_GPU_IDENTIFIERS = frozenset(
    {
        "ClothSimulator",
        "ClothSimulatorCore",
        "ClothReplayer",
        "GpuSdfBaker",
        "MeshSdfBaker",
        "bake_bone_sdf",
        "bake_mesh_sdf",
        "bake_bone_sdf_from_data",
        "bake_mesh_sdf_from_data",
        "bake_bone_sdf_gpu",
        "bake_mesh_sdf_gpu",
        "step_cloth_scene",
        "create_replayer",
        "is_gpu_available",
        "get_gpu_device_name",
        "get_current_gpu_device",
        "list_gpu_devices",
        "shared_pipeline_cache_info",
    }
)


def _uses_gpu(text: str) -> bool:
    """AST 走査により実際に GPU 実行 API・ラッパーを参照しているか判定する。

    docstring・コメント・テスト用ダミー文字列・@patch("...") モック文字列では True にしない。
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return any(ident in text for ident in _GPU_IDENTIFIERS)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _GPU_IDENTIFIERS:
            return True
        if isinstance(node, ast.Attribute) and node.attr in _GPU_IDENTIFIERS:
            return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")[-1]
                if head in _GPU_IDENTIFIERS or head == "replayer":
                    return True
        elif isinstance(node, ast.ImportFrom):
            mod_tail = (node.module or "").split(".")[-1]
            for alias in node.names:
                if alias.name == "*" and node.module == "taremin_cloth_core":
                    return True
                if alias.name in _GPU_IDENTIFIERS or alias.name == "replayer":
                    return True
            if mod_tail == "replayer":
                return True
    return False


def _detect_software_gpu(*, python_exe: Optional[str] = None) -> bool:
    """実行環境のアクティブGPUがソフトウェア実装 (WARP/Lavapipe等) かを判定する。"""
    override = os.environ.get("TAREMIN_CLOTH_GPU_SERIALIZE", "").strip().lower()
    if override in ("1", "true", "yes", "on"):
        return True
    if override in ("0", "false", "no", "off"):
        return False

    exe = python_exe or sys.executable
    probe = (
        "import json, taremin_cloth_core as c; "
        "print(json.dumps(c.get_current_gpu_device()))"
    )
    try:
        proc = subprocess.run(
            [exe, "-c", probe],
            capture_output=True,
            text=True,
            timeout=15.0,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            info = json.loads(proc.stdout.strip().splitlines()[-1])
            return str(info.get("device_type", "")).upper() == "CPU"
    except Exception:
        pass
    return os.environ.get("CI", "").strip().lower() == "true"


def _has_testcase_base(node: ast.ClassDef) -> bool:
    for base in node.bases:
        try:
            base_src = ast.unparse(base)
        except Exception:
            base_src = ""
        if "TestCase" in base_src:
            return True
    return False


def _collect_test_methods(node: ast.ClassDef) -> List[str]:
    methods: List[str] = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name.startswith("test_"):
                methods.append(item.name)
    return methods


def _class_has_keep_directive(
    node: ast.ClassDef,
    *,
    source_lines: Sequence[str],
) -> bool:
    """クラス直上の連続コメント行群またはクラス本体内の `# parallel: keep-class` 指定を検出する。"""
    dec_lines = [d.lineno for d in node.decorator_list] if node.decorator_list else []
    first_lineno = min([node.lineno] + dec_lines)
    # クラス（または最初のデコレータ）直前の連続するコメント行のみを遡り、
    # 直前クラスの末尾コメントを取り込まないよう空行・コード行で停止する。
    idx = first_lineno - 2
    leading_comments: List[str] = []
    while idx >= 0:
        line = source_lines[idx]
        if line.lstrip().startswith("#"):
            leading_comments.append(line)
            idx -= 1
        else:
            break
    end_lineno = min(len(source_lines), getattr(node, "end_lineno", node.lineno))
    body_slice = source_lines[max(0, first_lineno - 1) : end_lineno]
    snippet = "\n".join(reversed(leading_comments)) + "\n" + "\n".join(body_slice)
    return "parallel: keep-class" in snippet


def _class_requires_grouping(
    node: ast.ClassDef,
    *,
    methods: Sequence[str],
    source_lines: Sequence[str],
) -> bool:
    """クラス分割を禁止しクラス単位実行を維持すべきか判定する"""
    # setUpClass / tearDownClass の存在
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name in ("setUpClass", "tearDownClass"):
                return True

    # test_01_... のような連番順序依存
    if any(_ORDERED_METHOD_RE.match(m) for m in methods):
        return True

    # 明示ディレクティブ (# parallel: keep-class)
    if _class_has_keep_directive(node, source_lines=source_lines):
        return True

    return False


def _extract_units_from_file(
    file_path: Path,
    repo_root: Path,
    *,
    granularity: str = "auto",
    source_text: Optional[str] = None,
) -> List[TestUnit]:
    """単一の test_*.py ファイルから実行ユニット一覧を構築する"""
    text = (
        source_text
        if source_text is not None
        else file_path.read_text(encoding="utf-8", errors="ignore")
    )
    rel_posix = _normalize_rel_path(file_path, repo_root)
    mod_name = ".".join(Path(rel_posix).with_suffix("").parts)
    requires_blender = _is_blender_required_file(file_path, text)
    file_name = file_path.name
    file_size = len(text)

    is_gpu_unit = _uses_gpu(text)
    if file_name in _HEAVY_INDEX:
        base_priority = 0
        base_weight = 1_000_000 - _HEAVY_INDEX[file_name] * 1000
    elif is_gpu_unit:
        base_priority = 1
        base_weight = file_size
    else:
        base_priority = 2
        base_weight = file_size

    if granularity == "file" or "parallel: keep-module" in text:
        return [
            TestUnit(
                target=mod_name,
                file_path=file_path,
                priority=base_priority,
                weight=base_weight,
                requires_blender=requires_blender,
                uses_gpu=is_gpu_unit,
            )
        ]

    try:
        tree = ast.parse(text, filename=str(file_path))
    except SyntaxError:
        return [
            TestUnit(
                target=mod_name,
                file_path=file_path,
                priority=base_priority,
                weight=base_weight,
                requires_blender=requires_blender,
                uses_gpu=is_gpu_unit,
            )
        ]

    source_lines = text.splitlines()
    units: List[TestUnit] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = _collect_test_methods(node)
        if not methods and not _has_testcase_base(node):
            continue
        if not methods:
            continue

        class_target = f"{mod_name}.{node.name}"
        if granularity == "auto" and _class_requires_grouping(
            node,
            methods=methods,
            source_lines=source_lines,
        ):
            units.append(
                TestUnit(
                    target=class_target,
                    file_path=file_path,
                    priority=base_priority,
                    weight=base_weight,
                    requires_blender=requires_blender,
                    uses_gpu=is_gpu_unit,
                )
            )
            continue

        # メソッド単位に分割（各メソッドに安定した重みを付与）
        method_count = max(1, len(methods))
        per_method_weight = max(1, base_weight // method_count)
        for m_name in methods:
            prio = 3 if m_name in _GPU_RESET_METHODS else base_priority
            units.append(
                TestUnit(
                    target=f"{class_target}.{m_name}",
                    file_path=file_path,
                    priority=prio,
                    weight=per_method_weight,
                    requires_blender=requires_blender,
                    uses_gpu=is_gpu_unit,
                )
            )

    if not units:
        return [
            TestUnit(
                target=mod_name,
                file_path=file_path,
                priority=base_priority,
                weight=base_weight,
                requires_blender=requires_blender,
                uses_gpu=is_gpu_unit,
            )
        ]

    return units


def discover_test_units(
    repo_root: Path,
    *,
    test_roots: Optional[Sequence[Path]] = None,
    pattern: Optional[str] = None,
    granularity: str = "auto",
    scope: str = "standalone",
) -> List[TestUnit]:
    """
    指定スコープ・パターンに従ってテストユニットを収集し、LPT 順にソートして返す。

    Args:
        repo_root: リポジトリルートパス
        test_roots: 走査対象ディレクトリ群（省略時は scope に応じて自動決定）
        pattern: ファイル名・相対パス・glob パターン（指定時は tests/ 全体も探索対象）
        granularity: 分割粒度 ('auto' | 'file' | 'method')
        scope: 対象スコープ ('standalone' | 'blender' | 'all')
    """
    if test_roots is not None:
        roots = [r if r.is_absolute() else (repo_root / r) for r in test_roots]
    elif scope == "standalone" and not pattern:
        roots = [repo_root / d for d in DEFAULT_STANDALONE_DIRS]
    else:
        roots = [repo_root / "tests"]

    seen_files: set[Path] = set()
    matched_files: List[Path] = []
    matched_texts: Dict[Path, str] = {}
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = sorted(root.rglob("test_*.py"))
        for f in candidates:
            resolved = f.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)

            if pattern:
                if not _matches_pattern(f, repo_root=repo_root, pattern=pattern):
                    continue
            else:
                txt = f.read_text(encoding="utf-8", errors="ignore")
                is_blender_req = _is_blender_required_file(f, txt)
                if scope == "standalone" and is_blender_req:
                    continue
                if scope == "blender" and not is_blender_req:
                    continue
                matched_texts[f] = txt

            matched_files.append(f)

    units: List[TestUnit] = []
    for f in matched_files:
        units.extend(
            _extract_units_from_file(
                f,
                repo_root,
                granularity=granularity,
                source_text=matched_texts.get(f),
            )
        )

    # LPT + 同一ファイル分散 + GPUリセット末尾配置
    units.sort(key=lambda u: (u.priority, -u.weight, u.target))
    return units


def resolve_worker_count(
    *,
    requested: Optional[int],
    unit_count: int,
    default_cap: int = 4,
) -> int:
    """利用するワーカープロセス数を決定する"""
    if unit_count <= 1:
        return 1
    if requested is not None and requested > 0:
        return min(requested, unit_count)
    cpu = os.cpu_count() or 1
    return min(default_cap, max(1, cpu), unit_count)


# ---------------------------------------------------------------------------
# ワーカー内実行ロジック（Python / Blender 共通）
# ---------------------------------------------------------------------------


def _run_single_unit_locally(target: str, *, worker_id: str) -> UnitResult:
    """同一プロセス内で 1 テストユニットを実行し、出力を捕捉して UnitResult を返す"""
    buf = io.StringIO()
    loader = unittest.TestLoader()
    t0 = time.perf_counter()

    # taremin_cloth ロガーが以前の stdout を握っている場合もテスト単位で捕捉する
    taremin_logger = logging.getLogger("taremin_cloth")
    saved_streams: List[Tuple[logging.StreamHandler, Any]] = []
    for h in list(taremin_logger.handlers) + list(logging.getLogger().handlers):
        if isinstance(h, logging.StreamHandler):
            saved_streams.append((h, h.stream))
            h.stream = buf

    ok = False
    tests_run = 0
    skipped = 0
    error_summary = ""
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            suite = loader.loadTestsFromName(target)
            runner = unittest.TextTestRunner(stream=buf, verbosity=2)
            res = runner.run(suite)
            ok = res.wasSuccessful()
            tests_run = res.testsRun
            skipped = len(res.skipped)
            if not ok:
                error_summary = f"failures={len(res.failures)}, errors={len(res.errors)}"
    except Exception as exc:
        import traceback

        traceback.print_exc(file=buf)
        error_summary = f"Exception: {exc!r}"
        ok = False
    finally:
        for h, orig in saved_streams:
            h.stream = orig

    elapsed = time.perf_counter() - t0
    return UnitResult(
        target=target,
        worker_id=worker_id,
        ok=ok,
        tests_run=tests_run,
        skipped=skipped,
        elapsed=elapsed,
        output=buf.getvalue(),
        error_summary=error_summary,
    )


def _run_sequential_local(
    units: Sequence[TestUnit],
    *,
    verbosity: int = 2,
) -> int:
    """1 ワーカー相当のローカル直列実行（単一テストや --workers 1 用）"""
    loader = unittest.TestLoader()
    master_suite = unittest.TestSuite()
    for u in units:
        master_suite.addTests(loader.loadTestsFromName(u.target))
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(master_suite)
    return 0 if result.wasSuccessful() else 1


# ---------------------------------------------------------------------------
# ローカルループバック TCP キューサーバー & ワーカープロセス管理
# ---------------------------------------------------------------------------


class _CoordinatorState:
    """テスト配布キューと進捗・実行中タスクを管理するスレッドセーフ状態"""

    def __init__(
        self,
        units: Sequence[TestUnit],
        *,
        gpu_worker_ids: Optional[Sequence[str]] = None,
    ):
        self._lock = threading.Lock()
        self._queue: List[TestUnit] = list(units)
        self._total = len(units)
        self._in_flight: Dict[str, Tuple[str, float]] = {}
        self._results: List[UnitResult] = []
        self._worker_busy: Dict[str, float] = {}
        self._gpu_worker_bases: Optional[set[str]] = (
            set(gpu_worker_ids) if gpu_worker_ids is not None else None
        )
        self._retire_requested: set[str] = set()

    def _is_gpu_worker(self, worker_id: str) -> bool:
        if self._gpu_worker_bases is None:
            return True
        base_wid = worker_id.split("_r", 1)[0]
        return base_wid in self._gpu_worker_bases

    def pop_next(self, worker_id: str) -> Optional[str]:
        with self._lock:
            if not self._queue:
                self._in_flight.pop(worker_id, None)
                return None

            idx: Optional[int] = None
            if self._gpu_worker_bases is None:
                idx = 0
            elif self._is_gpu_worker(worker_id):
                # 専任GPUワーカーはGPUテストを優先消化し、残余のCPUテストも最後に手伝う
                for i, u in enumerate(self._queue):
                    if u.uses_gpu:
                        idx = i
                        break
                if idx is None:
                    idx = 0
            else:
                # CPUワーカーはGPUテストを取得せずCPUテストのみを並行消化する
                for i, u in enumerate(self._queue):
                    if not u.uses_gpu:
                        idx = i
                        break

            if idx is None:
                self._in_flight.pop(worker_id, None)
                return None

            unit = self._queue.pop(idx)
            self._in_flight[worker_id] = (unit.target, time.monotonic())
            return unit.target

    def request_worker_retire(self, worker_id: str) -> None:
        with self._lock:
            self._retire_requested.add(worker_id)

    def consume_retire_request(self, worker_id: str) -> bool:
        with self._lock:
            if worker_id in self._retire_requested:
                self._retire_requested.remove(worker_id)
                return True
            return False

    def has_pending_work_for_worker(self, worker_id: str) -> bool:
        with self._lock:
            if not self._queue:
                return False
            if self._gpu_worker_bases is None or self._is_gpu_worker(worker_id):
                return True
            return any(not u.uses_gpu for u in self._queue)

    def record_result(self, result: UnitResult) -> Tuple[int, int]:
        with self._lock:
            self._in_flight.pop(result.worker_id, None)
            self._results.append(result)
            self._worker_busy[result.worker_id] = (
                self._worker_busy.get(result.worker_id, 0.0) + result.elapsed
            )
            return len(self._results), self._total

    def mark_worker_aborted(
        self,
        worker_id: str,
        *,
        error_summary: str,
        output: str = "",
    ) -> Optional[UnitResult]:
        with self._lock:
            inflight = self._in_flight.pop(worker_id, None)
            if inflight is None:
                return None
            target, started_at = inflight
            res = UnitResult(
                target=target,
                worker_id=worker_id,
                ok=False,
                tests_run=1,
                skipped=0,
                elapsed=max(0.0, time.monotonic() - started_at),
                output=output,
                error_summary=error_summary,
            )
            self._results.append(res)
            return res

    def active_in_flight(self) -> List[Tuple[str, str, float]]:
        now = time.monotonic()
        with self._lock:
            return [
                (wid, target, now - started_at)
                for wid, (target, started_at) in self._in_flight.items()
            ]

    def has_pending_work(self) -> bool:
        with self._lock:
            return bool(self._queue) or bool(self._in_flight)

    def snapshot(self) -> Tuple[List[UnitResult], Dict[str, float]]:
        with self._lock:
            return list(self._results), dict(self._worker_busy)


def _send_json_line(conn: socket.socket, payload: Dict[str, Any]) -> None:
    raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    conn.sendall(raw)


def _recv_json_line(conn: socket.socket) -> Dict[str, Any]:
    chunks: List[bytes] = []
    while True:
        data = conn.recv(65536)
        if not data:
            break
        nl = data.find(b"\n")
        if nl != -1:
            chunks.append(data[:nl])
            break
        chunks.append(data)
    if not chunks:
        return {}
    return json.loads(b"".join(chunks).decode("utf-8", errors="replace"))


_FATAL_GPU_ERROR_MARKERS = (
    "Parent device is lost",
    "Device is lost",
    "GetDeviceRemovedReason",
    "0x887A0005",
)


def _has_fatal_gpu_error(result: UnitResult) -> bool:
    if result.ok:
        return False
    blob = f"{result.error_summary}\n{result.output}"
    return any(marker in blob for marker in _FATAL_GPU_ERROR_MARKERS)


def _make_queue_server(
    state: _CoordinatorState,
    *,
    on_result: Callable[[UnitResult, int, int], None],
) -> socketserver.ThreadingTCPServer:
    class _Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            try:
                req = _recv_json_line(self.request)
            except Exception:
                return
            op = req.get("op")
            if op == "next":
                wid = str(req.get("worker_id", ""))
                target = state.pop_next(wid)
                if target is None:
                    _send_json_line(self.request, {"done": True})
                else:
                    _send_json_line(self.request, {"done": False, "target": target})
            elif op == "report":
                res = UnitResult(
                    target=str(req.get("target", "")),
                    worker_id=str(req.get("worker_id", "")),
                    ok=bool(req.get("ok", False)),
                    tests_run=int(req.get("tests_run", 0)),
                    skipped=int(req.get("skipped", 0)),
                    elapsed=float(req.get("elapsed", 0.0)),
                    output=str(req.get("output", "")),
                    error_summary=str(req.get("error_summary", "")),
                )
                retire = _has_fatal_gpu_error(res)
                if retire:
                    state.request_worker_retire(res.worker_id)
                done_cnt, total_cnt = state.record_result(res)
                on_result(res, done_cnt=done_cnt, total_cnt=total_cnt)
                _send_json_line(self.request, {"ack": True, "retire": retire})

    class _Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    return _Server(("127.0.0.1", 0), _Handler)


def _worker_client_loop(
    *,
    host: str,
    port: int,
    worker_id: str,
    temp_dir: Optional[Path] = None,
    user_config_dir: Optional[Path] = None,
) -> int:
    """常駐ワーカー側の実行ループ（Python / Blender 共通）"""
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TEMP"] = str(temp_dir)
        os.environ["TMP"] = str(temp_dir)
        tempfile.tempdir = str(temp_dir)
    if user_config_dir is not None:
        user_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["BLENDER_USER_CONFIG"] = str(user_config_dir)

    while True:
        with socket.create_connection((host, port), timeout=30.0) as sock:
            _send_json_line(sock, {"op": "next", "worker_id": worker_id})
            resp = _recv_json_line(sock)
        if not resp or resp.get("done"):
            break
        target = str(resp["target"])
        res = _run_single_unit_locally(target, worker_id=worker_id)
        with socket.create_connection((host, port), timeout=30.0) as sock:
            _send_json_line(
                sock,
                {
                    "op": "report",
                    "worker_id": res.worker_id,
                    "target": res.target,
                    "ok": res.ok,
                    "tests_run": res.tests_run,
                    "skipped": res.skipped,
                    "elapsed": res.elapsed,
                    "output": res.output,
                    "error_summary": res.error_summary,
                },
            )
            ack = _recv_json_line(sock)
        if ack.get("retire"):
            # Device Lost 等の異常状態が後続テストへ連鎖しないようワーカープロセスを交代させる
            break
    return 0


def _tail_file(path: Path, *, max_chars: int = 4000) -> str:
    if not path.is_file():
        return ""
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
        return txt[-max_chars:]
    except Exception:
        return ""


def _run_with_worker_pool(
    units: Sequence[TestUnit],
    *,
    workers: int,
    build_worker_cmd: Callable[..., Tuple[List[str], Dict[str, str]]],
    cwd: Path,
    verbosity: int = 1,
    timeout_sec: float = 300.0,
    label: str = "ParallelTest",
    gpu_worker_ids: Optional[Sequence[str]] = None,
) -> int:
    """キューサーバー + ワーカープロセス群を統括し並列実行する共通エンジン"""
    state = _CoordinatorState(units, gpu_worker_ids=gpu_worker_ids)
    print_lock = threading.Lock()
    wall_t0 = time.perf_counter()

    def on_result(res: UnitResult, *, done_cnt: int, total_cnt: int) -> None:
        status = "OK" if res.ok else "FAIL"
        if res.ok and res.skipped > 0 and res.skipped == res.tests_run:
            status = "SKIP"
        with print_lock:
            if not res.ok or verbosity >= 2:
                print(
                    f"[{label}] [{done_cnt}/{total_cnt}] {status:<4} "
                    f"({res.elapsed:5.2f}s, {res.worker_id}) {res.target}",
                    flush=True,
                )
            elif done_cnt == total_cnt or done_cnt % 15 == 0:
                print(
                    f"[{label}] 進行中: {done_cnt}/{total_cnt} 完了 "
                    f"(経過 {time.perf_counter() - wall_t0:5.1f}s)",
                    flush=True,
                )

    server = _make_queue_server(state, on_result=on_result)
    port = int(server.server_address[1])
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()

    procs: Dict[str, Tuple[subprocess.Popen, Path, Any]] = {}

    def spawn_worker(wid: str, tmp_root: Path) -> None:
        w_dir = tmp_root / wid
        w_tmp = w_dir / "tmp"
        w_tmp.mkdir(parents=True, exist_ok=True)
        log_path = w_dir / "worker.log"
        cmd, extra_env = build_worker_cmd(wid, port=port, w_dir=w_dir)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["TEMP"] = str(w_tmp)
        env["TMP"] = str(w_tmp)
        env.update(extra_env)
        log_fp = open(log_path, "w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
        )
        procs[wid] = (proc, log_path, log_fp)

    try:
        with tempfile.TemporaryDirectory(prefix="taremin_parallel_test_") as tmp_dir_str:
            tmp_root = Path(tmp_dir_str)
            for idx in range(workers):
                spawn_worker(f"w{idx + 1}", tmp_root)

            # ワーカー監視ループ（異常終了・Device Lost引退・タイムアウトを検出）
            crash_restarts = 0
            max_restarts = max(4, workers * 2)
            while procs:
                finished_wids: List[str] = []
                for wid, (proc, log_path, log_fp) in list(procs.items()):
                    rc = proc.poll()
                    if rc is not None:
                        log_fp.close()
                        finished_wids.append(wid)
                        retired = state.consume_retire_request(wid)
                        base_wid = wid.split("_r", 1)[0]
                        if rc != 0:
                            tail = _tail_file(log_path)
                            aborted = state.mark_worker_aborted(
                                wid,
                                error_summary=f"ワーカー異常終了 (exit={rc})",
                                output=tail,
                            )
                            with print_lock:
                                failed_name = aborted.target if aborted else "(待機/初期化中)"
                                print(
                                    f"[{label}] [ERROR] ワーカー {wid} が異常終了しました "
                                    f"(exit={rc}, 対象={failed_name})",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            if state.has_pending_work_for_worker(base_wid) and crash_restarts < max_restarts:
                                crash_restarts += 1
                                new_wid = f"{base_wid}_r{crash_restarts}"
                                spawn_worker(new_wid, tmp_root)
                        elif retired and state.has_pending_work_for_worker(base_wid) and crash_restarts < max_restarts:
                            crash_restarts += 1
                            new_wid = f"{base_wid}_r{crash_restarts}"
                            with print_lock:
                                print(
                                    f"[{label}] [INFO] GPU異常後のワーカー {wid} を {new_wid} へ交代します",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            spawn_worker(new_wid, tmp_root)
                for wid in finished_wids:
                    procs.pop(wid, None)

                # タスク単位タイムアウト監視
                for wid, target, elapsed in state.active_in_flight():
                    if elapsed > timeout_sec and wid in procs:
                        proc, log_path, log_fp = procs.pop(wid)
                        proc.kill()
                        proc.wait(timeout=5.0)
                        log_fp.close()
                        tail = _tail_file(log_path)
                        state.mark_worker_aborted(
                            wid,
                            error_summary=f"タイムアウト ({elapsed:.1f}s > {timeout_sec:.0f}s)",
                            output=tail,
                        )
                        with print_lock:
                            print(
                                f"[{label}] [TIMEOUT] {wid} がタイムアウトしました: {target}",
                                file=sys.stderr,
                                flush=True,
                            )
                        base_wid = wid.split("_r", 1)[0]
                        if state.has_pending_work_for_worker(base_wid) and crash_restarts < max_restarts:
                            crash_restarts += 1
                            spawn_worker(f"{base_wid}_r{crash_restarts}", tmp_root)

                if procs:
                    time.sleep(0.05)

    except KeyboardInterrupt:
        print(f"\n[{label}] 中断シグナルを受信したためワーカーを停止します...", file=sys.stderr)
        for wid, (proc, _lp, log_fp) in procs.items():
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                log_fp.close()
            except Exception:
                pass
        return 130
    finally:
        server.shutdown()
        server.server_close()

    wall = time.perf_counter() - wall_t0
    results, worker_busy = state.snapshot()
    failures = [r for r in results if not r.ok]
    total_tests = sum(r.tests_run for r in results)
    total_skipped = sum(r.skipped for r in results)

    # 失敗時はキャプチャログをすべて表示
    if failures:
        print("\n" + "=" * 70, file=sys.stderr)
        print(f"[{label}] 失敗テスト詳細 ({len(failures)} 件)", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        for fail in failures:
            print(f"\n--- FAIL: {fail.target} ({fail.worker_id}, {fail.elapsed:.2f}s) ---", file=sys.stderr)
            if fail.error_summary:
                print(f"Summary: {fail.error_summary}", file=sys.stderr)
            if fail.output:
                print(fail.output.rstrip(), file=sys.stderr)
    elif verbosity >= 2:
        for res in results:
            if res.output.strip():
                print(f"\n--- OUTPUT: {res.target} ---\n{res.output.rstrip()}")

    busy_str = ", ".join(f"{k}:{v:.1f}s" for k, v in sorted(worker_busy.items()))
    print("\n" + "-" * 70)
    print(
        f"[{label}] 完了: {total_tests} tests (skipped={total_skipped}, "
        f"units={len(results)}, fail={len(failures)}) in {wall:.2f}s "
        f"[workers={workers}: {busy_str}]"
    )
    print("-" * 70)

    if len(results) < len(units):
        print(
            f"[{label}] [ERROR] 未完了ユニットが残っています ({len(results)}/{len(units)})",
            file=sys.stderr,
        )
        return 1

    return 0 if not failures else 1


# ---------------------------------------------------------------------------
# 公開エントリポイント (Python / Blender)
# ---------------------------------------------------------------------------


def _run_targets_via_temp_file(
    units: Sequence[TestUnit],
    *,
    cmd_prefix: Sequence[str],
    cwd: Path,
    verbosity: int,
    pass_separator: bool = False,
    isolate_blender_config: bool = False,
    prefix: str = "taremin_single_",
) -> int:
    """単一ワーカー向けのサブプロセス実行を一時 targets.json 経由で行い、CLI 長上限を回避する。"""
    script_path = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp_dir_str:
        tmp_root = Path(tmp_dir_str)
        targets_file = tmp_root / "targets.json"
        targets_file.write_text(json.dumps([u.target for u in units]), encoding="utf-8")
        tmp_sub = tmp_root / "tmp"
        tmp_sub.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["TEMP"] = str(tmp_sub)
        env["TMP"] = str(tmp_sub)
        if isolate_blender_config:
            cfg_dir = tmp_root / "config"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            env["BLENDER_USER_CONFIG"] = str(cfg_dir)
        sep = ["--"] if pass_separator else []
        cmd = [
            *cmd_prefix,
            str(script_path),
            *sep,
            "--targets-file",
            str(targets_file),
            "--verbosity",
            str(verbosity),
        ]
        return subprocess.run(cmd, cwd=str(cwd), env=env).returncode


def run_parallel_python_tests(
    repo_root: Path,
    *,
    python_exe: Optional[Path] = None,
    test_roots: Optional[Sequence[Path]] = None,
    pattern: Optional[str] = None,
    workers: Optional[int] = None,
    granularity: str = "auto",
    scope: str = "standalone",
    verbosity: int = 1,
    timeout_sec: Optional[float] = None,
    serialize_gpu: Optional[bool] = None,
) -> int:
    """Python スタンドアロンテストを常駐ワーカープールで並列実行する"""
    units = discover_test_units(
        repo_root,
        test_roots=test_roots,
        pattern=pattern,
        granularity=granularity,
        scope=scope,
    )
    if not units:
        print(f"[ParallelTest] 対象テストが見つかりませんでした (pattern={pattern!r}, scope={scope})")
        return 0

    eff_workers = resolve_worker_count(requested=workers, unit_count=len(units), default_cap=4)
    py_exe = str(python_exe) if python_exe else sys.executable

    has_gpu_units = any(u.uses_gpu for u in units)
    has_cpu_units = any(not u.uses_gpu for u in units)
    if serialize_gpu is None:
        is_sw_gpu = has_gpu_units and _detect_software_gpu(python_exe=py_exe)
    else:
        is_sw_gpu = serialize_gpu

    eff_timeout = timeout_sec if timeout_sec is not None else (480.0 if is_sw_gpu else 300.0)
    gpu_worker_ids: Optional[List[str]] = None
    if is_sw_gpu and eff_workers > 1 and has_gpu_units:
        if not has_cpu_units:
            # GPUユニットのみの場合は1専任プロセスで順次実行しコンパイル共有と競合回避を図る
            eff_workers = 1
        else:
            gpu_worker_ids = ["w1"]

    if eff_workers <= 1:
        # 単一ワーカー時: 指定 python_exe が自身ならインプロセス実行、別環境なら一時 targets.json 経由で実行
        if Path(py_exe).resolve() == Path(sys.executable).resolve():
            return _run_sequential_local(units, verbosity=max(verbosity, 2))
        return _run_targets_via_temp_file(
            units,
            cmd_prefix=[py_exe],
            cwd=repo_root,
            verbosity=max(verbosity, 2),
            pass_separator=False,
            isolate_blender_config=False,
            prefix="taremin_python_single_",
        )

    script_path = Path(__file__).resolve()

    def build_cmd(wid: str, *, port: int, w_dir: Path) -> Tuple[List[str], Dict[str, str]]:
        cmd = [
            py_exe,
            str(script_path),
            "--worker",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--worker-id",
            wid,
            "--temp-dir",
            str(w_dir / "tmp"),
        ]
        return cmd, {}

    lane_note = (
        f", mode=software-gpu-lane(gpu=w1, cpu=w2..w{eff_workers}, timeout={eff_timeout:.0f}s)"
        if gpu_worker_ids is not None
        else f", timeout={eff_timeout:.0f}s"
    )
    print(
        f"[ParallelTest] Python テスト実行開始: {len(units)} ユニット / "
        f"{eff_workers} ワーカー (granularity={granularity}, scope={scope}{lane_note})"
    )
    return _run_with_worker_pool(
        units,
        workers=eff_workers,
        build_worker_cmd=build_cmd,
        cwd=repo_root,
        verbosity=verbosity,
        timeout_sec=eff_timeout,
        label="PythonParallel",
        gpu_worker_ids=gpu_worker_ids,
    )


def run_parallel_blender_tests(
    repo_root: Path,
    *,
    blender_path: Path,
    pattern: Optional[str] = None,
    workers: Optional[int] = None,
    include_all: bool = False,
    granularity: str = "auto",
    verbosity: int = 2,
    timeout_sec: float = 300.0,
) -> int:
    """Blender バックグラウンドテストを実行する（既定で Blender 必須テストのみ・並列対応）"""
    scope = "all" if include_all else "blender"
    units = discover_test_units(
        repo_root,
        pattern=pattern,
        granularity=granularity,
        scope=scope,
    )
    if not units:
        print(f"[BlenderParallel] 対象テストが見つかりませんでした (pattern={pattern!r}, scope={scope})")
        return 0

    eff_workers = resolve_worker_count(requested=workers, unit_count=len(units), default_cap=2)
    script_path = Path(__file__).resolve()

    if eff_workers <= 1:
        print(
            f"[BlenderParallel] 単一 Blender 実行: {len(units)} ユニット "
            f"(scope={scope}, blender={blender_path.name})"
        )
        return _run_targets_via_temp_file(
            units,
            cmd_prefix=[
                str(blender_path),
                "--background",
                "--factory-startup",
                "--python",
            ],
            cwd=repo_root,
            verbosity=verbosity,
            pass_separator=True,
            isolate_blender_config=True,
            prefix="taremin_blender_single_",
        )

    def build_cmd(wid: str, *, port: int, w_dir: Path) -> Tuple[List[str], Dict[str, str]]:
        cfg_dir = w_dir / "config"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(blender_path),
            "--background",
            "--factory-startup",
            "--python",
            str(script_path),
            "--",
            "--worker",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--worker-id",
            wid,
            "--temp-dir",
            str(w_dir / "tmp"),
            "--user-config-dir",
            str(cfg_dir),
        ]
        return cmd, {"BLENDER_USER_CONFIG": str(cfg_dir)}

    print(
        f"[BlenderParallel] Blender テスト並列実行開始: {len(units)} ユニット / "
        f"{eff_workers} ワーカー (granularity={granularity}, scope={scope})"
    )
    return _run_with_worker_pool(
        units,
        workers=eff_workers,
        build_worker_cmd=build_cmd,
        cwd=repo_root,
        verbosity=verbosity,
        timeout_sec=timeout_sec,
        label="BlenderParallel",
    )


def _parse_cli_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    # Blender 経由で起動された場合、'--' より後ろだけを受け取る
    if argv is None:
        raw_argv = sys.argv[1:]
        if "--" in sys.argv:
            idx = sys.argv.index("--")
            raw_argv = sys.argv[idx + 1 :]
    else:
        raw_argv = list(argv)

    parser = argparse.ArgumentParser(description="Taremin Cloth 並列 unittest ランナー")
    parser.add_argument("--standalone", action="store_true", help="スタンドアロン全テストを実行")
    parser.add_argument("--test", "-t", type=str, default=None, help="テストパターンまたはファイル名")
    parser.add_argument("--workers", "-j", type=int, default=None, help="ワーカープロセス数 (1 で直列)")
    parser.add_argument(
        "--granularity",
        choices=["auto", "file", "method"],
        default="auto",
        help="タスク分割粒度",
    )
    parser.add_argument("--verbosity", "-v", type=int, default=1, help="出力詳細度 (1 or 2)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="1ユニットあたりのタイムアウト秒数 (既定: 通常300 / Software GPU時480)",
    )

    # 内部ワーカー・バッチ実行用オプション
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--host", type=str, default="127.0.0.1", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-id", type=str, default="w0", help=argparse.SUPPRESS)
    parser.add_argument("--temp-dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--user-config-dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--targets-file", type=str, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(raw_argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_cli_args(argv)

    if args.worker:
        return _worker_client_loop(
            host=args.host,
            port=args.port,
            worker_id=args.worker_id,
            temp_dir=Path(args.temp_dir) if args.temp_dir else None,
            user_config_dir=Path(args.user_config_dir) if args.user_config_dir else None,
        )

    if args.targets_file:
        targets = json.loads(Path(args.targets_file).read_text(encoding="utf-8"))
        units = [
            TestUnit(target=t, file_path=REPO_ROOT, priority=0, weight=1, requires_blender=True)
            for t in targets
        ]
        return _run_sequential_local(units, verbosity=args.verbosity)

    return run_parallel_python_tests(
        REPO_ROOT,
        pattern=args.test,
        workers=args.workers,
        granularity=args.granularity,
        scope="standalone",
        verbosity=args.verbosity,
        timeout_sec=args.timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
