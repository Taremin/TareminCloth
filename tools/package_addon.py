#!/usr/bin/env python3
"""
Taremin Cloth Blenderアドオン パッケージングスクリプト

Blenderアドオンの配布用zipアーカイブを生成します。
不要なソースコード、テスト、キャッシュファイルを除外し、
アドオンの動作に必要な最小限の構成でパッケージングします。
"""

import argparse
import ast
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def extract_bl_info_version(init_file: Path) -> str:
    """__init__.py から bl_info の version タプルを抽出して文字列化する (例: '0.1.0')"""
    if not init_file.exists():
        return "0.1.0"
    content = init_file.read_text(encoding="utf-8")
    match = re.search(r'"version"\s*:\s*\(([0-9,\s]+)\)', content)
    if match:
        parts = [p.strip() for p in match.group(1).split(",") if p.strip()]
        return ".".join(parts)
    return "0.1.0"


def find_binary_in_wheel(wheel_path: Path, temp_dir: Path) -> Path:
    """Wheel (.whl) ファイル（またはディレクトリ）を展開し、taremin_cloth_core のバイナリ (.pyd または .so) を探す"""
    actual_wheel = wheel_path
    if actual_wheel.is_dir():
        wheels = list(actual_wheel.glob("*.whl"))
        if not wheels:
            raise FileNotFoundError(f"指定されたディレクトリ ({wheel_path}) 内に .whl ファイルが見つかりませんでした。")
        actual_wheel = wheels[0]

    with zipfile.ZipFile(actual_wheel, "r") as z:
        z.extractall(temp_dir)

    for item in temp_dir.rglob("*"):
        if item.is_file() and item.name.startswith("taremin_cloth_core"):
            if item.suffix in [".pyd", ".so"]:
                return item
    raise FileNotFoundError(f"Wheel ({actual_wheel}) 内に taremin_cloth_core のバイナリ (.pyd/.so) が見つかりませんでした。")


def package_addon(
    repo_root: Path,
    output_zip: Path,
    binary_path: Path | None = None,
    wheel_path: Path | None = None,
    version: str | None = None,
) -> Path:
    """アドオンの zip アーカイブをビルドする"""
    init_py = repo_root / "__init__.py"
    if not init_py.exists():
        raise FileNotFoundError(f"アドオンの __init__.py が見つかりません: {init_py}")

    addon_version = version or extract_bl_info_version(init_py)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        staging_addon_dir = tmp_dir / "taremin_cloth"
        staging_addon_dir.mkdir(parents=True, exist_ok=True)

        # 1. バイナリの特定と配置
        target_binary_name = None
        source_binary = None

        if wheel_path:
            wheel_extracted_dir = tmp_dir / "wheel_extracted"
            wheel_extracted_dir.mkdir()
            source_binary = find_binary_in_wheel(wheel_path, wheel_extracted_dir)
        elif binary_path:
            source_binary = binary_path
        else:
            # 指定がない場合はリポジトリルートから自動探索
            candidates = list(repo_root.glob("taremin_cloth_core*.pyd")) + list(repo_root.glob("taremin_cloth_core*.so"))
            if candidates:
                source_binary = candidates[0]

        if source_binary and source_binary.exists():
            # 拡張子に合わせて統一名称にリネーム
            ext = source_binary.suffix.lower()
            dest_binary = staging_addon_dir / f"taremin_cloth_core{ext}"
            shutil.copy2(source_binary, dest_binary)
            print(f"[*] バイナリ配置: {source_binary.name} -> {dest_binary.name}")
        else:
            print("[!] 警告: コンパイル済みバイナリ (.pyd / .so) が見つかりませんでした。純Python部分のみパッケージングされます。")

        # 2. __init__.py のコピー
        shutil.copy2(init_py, staging_addon_dir / "__init__.py")

        # 3. python ディレクトリのコピー (pycache等の除外)
        source_python = repo_root / "python"
        if source_python.exists():
            dest_python = staging_addon_dir / "python"
            shutil.copytree(
                source_python,
                dest_python,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.pyd", "*.so", ".DS_Store"),
            )

        # 4. メタ情報ファイル (LICENSE, README.md)
        for meta_file in ["LICENSE", "README.md"]:
            meta_path = repo_root / meta_file
            if meta_path.exists():
                shutil.copy2(meta_path, staging_addon_dir / meta_file)

        # 5. 出力先ディレクトリの準備
        output_zip.parent.mkdir(parents=True, exist_ok=True)

        # 6. zip アーカイブの生成
        base_name = str(output_zip.with_suffix(""))
        archive_format = "zip"
        shutil.make_archive(
            base_name=base_name,
            format=archive_format,
            root_dir=tmp_dir,
            base_dir="taremin_cloth",
        )

        final_zip = output_zip.with_suffix(".zip")
        print(f"[+] アドオンパッケージ作成完了: {final_zip} ({final_zip.stat().st_size:,} bytes)")
        return final_zip


def main():
    parser = argparse.ArgumentParser(description="Taremin Cloth Blenderアドオン パッケージングツール")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="出力する zip ファイルパス（未指定時は dist/taremin_cloth-<version>.zip）",
    )
    parser.add_argument(
        "--binary",
        "-b",
        type=Path,
        default=None,
        help="同梱するコンパイル済みバイナリ (.pyd または .so) のパス",
    )
    parser.add_argument(
        "--wheel",
        "-w",
        type=Path,
        default=None,
        help="バイナリ抽出元の Wheel (.whl) ファイルパス",
    )
    parser.add_argument(
        "--version",
        "-v",
        type=str,
        default=None,
        help="バージョン文字列（指定しない場合は __init__.py から抽出）",
    )

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    output_path = args.output
    if output_path is None:
        ver = args.version or extract_bl_info_version(repo_root / "__init__.py")
        output_path = repo_root / "dist" / f"taremin_cloth-v{ver}.zip"

    try:
        package_addon(
            repo_root=repo_root,
            output_zip=output_path,
            binary_path=args.binary,
            wheel_path=args.wheel,
            version=args.version,
        )
    except Exception as e:
        print(f"[!] エラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
