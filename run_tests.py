#!/usr/bin/env python3
"""
Taremin Cloth テストランナー
Blenderのバージョン解決、ダウンロード、バックグラウンドテスト実行を統括するスクリプト。
"""

import sys
import argparse
from pathlib import Path

# tools ディレクトリをインポートパスに追加
sys.path.insert(0, str(Path(__file__).parent))
from tools.blender_manager import (
    resolve_blender,
    find_installed_blenders,
    download_blender,
    clean_cached_blender,
    get_directory_size_mb,
    run_blender_unittest,
    get_cache_dir,
    get_latest_lts_version,
)


def main():
    parser = argparse.ArgumentParser(
        description="Taremin Cloth テストランナー"
    )
    parser.add_argument(
        "--blender",
        "-b",
        type=str,
        default=None,
        help="実行するBlenderのバージョン（例: 3.6, 4.2, latest-lts）。未指定時はダウンロード済み/ローカルの最新版を使用。",
    )
    parser.add_argument(
        "--test",
        "-t",
        type=str,
        default=None,
        help="実行するテストのファイル名パターン（例: test_e2e_pipeline.py, test_*.py）。",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="指定したバージョンのダウンロードのみを行い、テストは実行しません。",
    )
    parser.add_argument(
        "--list-blenders",
        action="store_true",
        help="ローカル（システムおよびキャッシュ）に存在するBlender一覧と使用容量を表示します。",
    )
    parser.add_argument(
        "--clean",
        nargs="?",
        const="all",
        default=None,
        metavar="VERSION",
        help="ダウンロードされたBlenderのキャッシュを削除します（例: --clean で全削除、--clean 4.2 で特定バージョンのみ削除）。",
    )

    args = parser.parse_args()

    # キャッシュ削除オプション
    if args.clean is not None:
        clean_cached_blender(args.clean)
        return 0

    if args.list_blenders:
        blenders = find_installed_blenders()
        cache_dir = get_cache_dir()
        latest_lts = get_latest_lts_version()
        print("\n=== ダウンロード済みBlender一覧 ===")
        print(f"キャッシュディレクトリ: {cache_dir}")
        print(f"公式最新LTSバージョン: {latest_lts}\n")
        if not blenders:
            print("  (ダウンロード済みのBlenderはありません)")
        else:
            for ver, path in blenders:
                ver_str = ".".join(str(x) for x in ver)
                size_mb = get_directory_size_mb(path.parent)
                print(f"  - Blender v{ver_str:<8}: {path} ({size_mb:.1f} MB)")
        print()
        return 0

    if args.download_only:
        version_to_dl = args.blender or "latest-lts"
        exe_path = download_blender(version_to_dl)
        print(f"[RunTests] ダウンロード完了: {exe_path}")
        return 0

    # Blender実行バイナリを解決
    blender_exe = resolve_blender(args.blender)

    # テストを実行
    returncode = run_blender_unittest(
        blender_path=blender_exe,
        test_pattern=args.test,
        cwd=Path(__file__).parent,
    )

    if returncode == 0:
        print("\n🎉 全てのテストが正常にパスしました！\n")
    else:
        print(f"\n❌ テストに失敗しました (終了コード: {returncode})\n")

    return returncode


if __name__ == "__main__":
    sys.exit(main())
