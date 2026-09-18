#!/usr/bin/env python3
"""
Taremin Cloth CI再現テストランナー
GitHub Actions CI（test-python）と全く同一の最小クリーン環境をローカルに自動構築し、
Blender非依存のテスト（tests/core）をスタンドアロンで実行・検証するツール。
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def get_ci_venv_dir(repo_root: Path) -> Path:
    """CI再現用仮想環境のディレクトリパスを取得"""
    return repo_root / ".venv_ci"


def get_venv_python(venv_dir: Path) -> Path:
    """仮想環境内の Python 実行可能ファイルパスを取得"""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def clean_ci_venv(repo_root: Path) -> None:
    """CI再現用仮想環境を削除してクリーンアップ"""
    venv_dir = get_ci_venv_dir(repo_root)
    if venv_dir.exists():
        print(f"[CI Runner] CI仮想環境を削除中: {venv_dir}")
        shutil.rmtree(venv_dir, ignore_errors=True)
        print("[CI Runner] 削除完了")
    else:
        print("[CI Runner] 削除対象のCI仮想環境は存在しません")


def ensure_ci_venv(repo_root: Path) -> Path:
    """CI再現用仮想環境を作成・整備し、Python実行可能ファイルのパスを返す"""
    venv_dir = get_ci_venv_dir(repo_root)
    py_exe = get_venv_python(venv_dir)

    # 仮想環境の新規作成
    if not py_exe.exists():
        print(f"[CI Runner] CI再現用仮想環境を作成中: {venv_dir}")
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True,
            cwd=repo_root,
        )
        print("[CI Runner] 仮想環境の作成完了")

    # 必須依存関係（ci.yml と完全に同一の最小構成）のチェック
    required_packages = ["numpy", "Pillow"]
    check_code = (
        "import sys\n"
        "try:\n"
        "    import numpy\n"
        "    import PIL\n"
        "    sys.exit(0)\n"
        "except ImportError:\n"
        "    sys.exit(1)\n"
    )
    res = subprocess.run(
        [str(py_exe), "-c", check_code],
        capture_output=True,
    )

    if res.returncode != 0:
        print(f"[CI Runner] 必須パッケージ（{', '.join(required_packages)}）をインストール中...")
        subprocess.run(
            [str(py_exe), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
            check=True,
            cwd=repo_root,
        )
        subprocess.run(
            [str(py_exe), "-m", "pip", "install", "--quiet"] + required_packages,
            check=True,
            cwd=repo_root,
        )
        print("[CI Runner] 依存パッケージのセットアップ完了")

    return py_exe


def run_ci_tests(
    repo_root: Optional[Path] = None,
    test_pattern: Optional[str] = None,
    clean: bool = False,
) -> int:
    """CI再現環境上でテストを実行する"""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    if clean:
        clean_ci_venv(repo_root)
        return 0

    py_exe = ensure_ci_venv(repo_root)

    # 実行時環境変数の構築（Windows cp1252等の文字化け防止）
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    # テストコマンドの構築
    if test_pattern:
        # 指定された単体テストファイルまたはパターンの実行
        target = test_pattern
        if not target.startswith("tests") and not target.startswith("tests/core"):
            candidate = repo_root / "tests" / "core" / target
            if candidate.exists():
                target = f"tests/core/{target}"
            else:
                candidate = repo_root / "tests" / target
                if candidate.exists():
                    target = f"tests/{target}"

        print(f"\n[CI Runner] 単体テスト実行: {target}")
        cmd = [str(py_exe), "-m", "unittest", target]
    else:
        # CI（ci.yml）と同一の全件ディスカバリテスト実行
        print("\n[CI Runner] CI全件テスト実行: python -m unittest discover -s tests/core -t .")
        cmd = [str(py_exe), "-m", "unittest", "discover", "-s", "tests/core", "-t", "."]

    result = subprocess.run(cmd, cwd=repo_root, env=env)
    return result.returncode


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Taremin Cloth CI再現テストランナー")
    parser.add_argument(
        "--test",
        "-t",
        type=str,
        default=None,
        help="実行するテストファイル名またはパス（例: test_quick_pinning.py）",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="CI再現仮想環境（.venv_ci）を削除します",
    )
    args = parser.parse_args()

    exit_code = run_ci_tests(test_pattern=args.test, clean=args.clean)
    sys.exit(exit_code)
