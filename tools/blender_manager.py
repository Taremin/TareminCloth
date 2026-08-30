"""
Blenderの検出・公式からの自動ダウンロード・キャッシュ管理・バックグラウンド実行モジュール
Python標準ライブラリのみで動作し、追加依存関係を必要としません。
"""

import os
import re
import sys
import shutil
import urllib.request
import zipfile
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 最新LTSバージョンとダウンロード先URLのマッピング（既知の安定版）
KNOWN_LTS_VERSIONS = {
    "4.2": {
        "full_version": "4.2.0",
        "url": "https://download.blender.org/release/Blender4.2/blender-4.2.0-windows-x64.zip",
    },
    "3.6": {
        "full_version": "3.6.18",
        "url": "https://download.blender.org/release/Blender3.6/blender-3.6.18-windows-x64.zip",
    },
    "2.93": {
        "full_version": "2.93.18",
        "url": "https://download.blender.org/release/Blender2.93/blender-2.93.18-windows-x64.zip",
    },
}

LATEST_LTS = "4.2"


def get_cache_dir() -> Path:
    """Blenderバイナリのキャッシュディレクトリを取得（作成）"""
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            cache_base = Path(local_app_data)
        else:
            cache_base = Path.home() / ".cache"
    else:
        cache_base = Path.home() / ".cache"

    cache_dir = cache_base / "blender_test_binaries"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_directory_size_mb(path: Path) -> float:
    """ディレクトリの合計サイズ（MB）を計算"""
    total_bytes = 0
    try:
        for p in path.glob("**/*"):
            if p.is_file():
                total_bytes += p.stat().st_size
    except Exception:
        pass
    return total_bytes / (1024 * 1024)


def clean_cached_blender(version_spec: Optional[str] = None) -> List[Path]:
    """
    ダウンロード・キャッシュされたBlenderを削除する。
    version_spec が指定されている場合はそのバージョンのみ、
    None または 'all' の場合はキャッシュ全体を削除する。
    削除されたパスのリストを返す。
    """
    cache_dir = get_cache_dir()
    deleted = []

    if not cache_dir.exists():
        print(f"[BlenderManager] キャッシュディレクトリが存在しません: {cache_dir}")
        return deleted

    # キャッシュ内のサブディレクトリ（各Blenderポータブルフォルダ）を探索
    subdirs = [d for d in cache_dir.iterdir() if d.is_dir()]

    if version_spec is None or version_spec.strip().lower() in ("all", "*"):
        # 全削除
        for d in subdirs:
            size_mb = get_directory_size_mb(d)
            shutil.rmtree(d, ignore_errors=True)
            print(f"[BlenderManager] キャッシュ削除: {d.name} ({size_mb:.1f} MB)")
            deleted.append(d)
        # 残っているzip等も削除
        for f in cache_dir.glob("*.zip"):
            f.unlink(missing_ok=True)
    else:
        # 指定バージョンの削除
        v_clean = version_spec.strip().lower()
        for d in subdirs:
            if v_clean in d.name.lower():
                size_mb = get_directory_size_mb(d)
                shutil.rmtree(d, ignore_errors=True)
                print(f"[BlenderManager] キャッシュ削除: {d.name} ({size_mb:.1f} MB)")
                deleted.append(d)

    if not deleted:
        print(f"[BlenderManager] 削除対象のキャッシュが見つかりませんでした (指定: {version_spec})")

    return deleted


def parse_version_from_path(path: Path) -> Tuple[int, ...]:
    """パスまたはフォルダ名からバージョン番号のタプルを抽出（例: '2.90.1' -> (2, 90, 1)）"""
    text = str(path)
    # 'Blender 4.2', 'blender-3.6.18', '2.90' などのパターンをマッチ
    match = re.search(r"blender[ -]?(\d+)(?:\.(\d+))?(?:\.(\d+))?", text, re.IGNORECASE)
    if match:
        parts = [int(p) for p in match.groups() if p is not None]
        # 引数が1個だけ（例: 2.90 -> 2, 90）の場合は末尾に0を補う
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts)
    # 'Blender\blender.exe' 単体の場合は2.80と推定
    if "blender foundation\\blender\\blender.exe" in text.lower():
        return (2, 80, 0)
    return (0, 0, 0)


def get_blender_version(blender_exe: Path) -> Tuple[int, ...]:
    """Blender実行ファイルを実行してバージョン番号を取得"""
    try:
        result = subprocess.run(
            [str(blender_exe), "--background", "--factory-startup", "--python-expr", "import bpy; print('BLENDER_VERSION:', bpy.app.version)"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            if "BLENDER_VERSION:" in line:
                # 例: BLENDER_VERSION: (2, 90, 1)
                match = re.search(r"\((\d+),\s*(\d+),\s*(\d+)\)", line)
                if match:
                    return tuple(int(x) for x in match.groups())
    except Exception:
        pass
    return parse_version_from_path(blender_exe)


def fetch_active_lts_versions() -> List[str]:
    """
    Blender公式のLTSページ (https://www.blender.org/download/lts/) から
    現在および歴代のLTSバージョン（例: ['5.2', '4.5', '4.2', '3.6', ...]）を動的に取得する。
    """
    url = "https://www.blender.org/download/lts/"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            # 'Blender 4.2 LTS' などの表記を抽出
            found = re.findall(r"Blender\s+(\d+\.\d+)\s+LTS", html, re.IGNORECASE)
            if found:
                # 順序を維持しつつ重複排除
                seen = set()
                lts_list = []
                for v in found:
                    if v not in seen:
                        seen.add(v)
                        lts_list.append(v)
                return lts_list
    except Exception:
        pass

    # オフライン時フォールバック
    return ["4.2", "3.6", "3.3", "2.93"]


def get_latest_lts_version() -> str:
    """動的に解決された最新のLTSバージョン文字列を返す（例: '5.2' または '4.2'）"""
    lts_list = fetch_active_lts_versions()
    return lts_list[0] if lts_list else "4.2"


def find_installed_blenders() -> List[Tuple[Tuple[int, ...], Path]]:
    """テスト用キャッシュディレクトリ内にダウンロード済みのBlenderのみを検索してバージョン降順で返す"""
    candidates = set()

    # テスト用キャッシュディレクトリのみを探索
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        for b_dir in cache_dir.iterdir():
            if b_dir.is_dir():
                exe = b_dir / "blender.exe"
                if exe.is_file():
                    candidates.add(exe)

    # バージョン情報を取得して降順ソート
    found = []
    for exe in candidates:
        if exe.is_file():
            ver = parse_version_from_path(exe)
            found.append((ver, exe))

    # バージョン降順でソート（同じバージョンの場合はパス順）
    found.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    return found


def download_file_with_progress(url: str, dest_path: Path):
    """User-Agentヘッダーを付与してプログレスバー付きでファイルをダウンロード"""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
    )
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        block_size = 1024 * 1024  # 1MB
        downloaded = 0

        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            downloaded += len(buffer)
            out_file.write(buffer)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                dl_mb = downloaded / (1024 * 1024)
                tot_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\rダウンロード進行度: {percent:.1f}% ({dl_mb:.1f} MB / {tot_mb:.1f} MB)")
                sys.stdout.flush()
    print()


def fetch_latest_patch_zip(version_spec: str) -> Tuple[str, str]:
    """
    指定されたバージョン（例: '4.2', '3.6', '4.2.0'）に対して、
    Blender公式リリースサーバー上の最新のWindows用zipファイル名とURLを動的に解決する
    """
    parts = version_spec.strip().split(".")
    major_minor = ".".join(parts[:2])
    dir_url = f"https://download.blender.org/release/Blender{major_minor}/"

    req = urllib.request.Request(
        dir_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            # blender-x.x.x-windows-x64.zip または windows64.zip にマッチ
            matches = re.findall(
                r'href=["\'](blender-[\d\.]+-windows(?:-x64|64)\.zip)["\']',
                html,
                re.IGNORECASE,
            )
            if matches:
                # バージョン番号順にソートして最新のものを選択
                def sort_key(fn):
                    v_match = re.search(r"blender-(\d+)\.(\d+)(?:\.(\d+))?", fn)
                    if v_match:
                        return tuple(int(x) for x in v_match.groups() if x is not None)
                    return (0, 0, 0)

                matches.sort(key=sort_key)
                best_zip = matches[-1]
                return best_zip, f"{dir_url}{best_zip}"
    except Exception:
        pass

    # フォールバック（既知の固定URLまたは推定URL）
    if version_spec in KNOWN_LTS_VERSIONS:
        info = KNOWN_LTS_VERSIONS[version_spec]
        return f"blender-{info['full_version']}-windows-x64.zip", info["url"]

    full_ver = version_spec if len(parts) >= 3 else f"{major_minor}.0"
    default_zip = f"blender-{full_ver}-windows-x64.zip"
    return default_zip, f"{dir_url}{default_zip}"


def download_blender(version: str) -> Path:
    """指定されたバージョンのBlenderを動的に検索・ダウンロード・解凍してblender.exeのパスを返す"""
    version_key = version.strip()
    zip_filename, url = fetch_latest_patch_zip(version_key)
    target_name = zip_filename.replace(".zip", "")

    cache_dir = get_cache_dir()
    extracted_dir = cache_dir / target_name
    blender_exe = extracted_dir / "blender.exe"

    if blender_exe.exists():
        print(f"[BlenderManager] キャッシュ済みのBlenderを使用: {blender_exe}")
        return blender_exe

    # 別名で既に解凍されているかチェック
    for exe in cache_dir.glob(f"**/{target_name}/**/blender.exe"):
        if exe.exists():
            return exe

    zip_path = cache_dir / zip_filename
    print(f"[BlenderManager] Blender {version} をダウンロード中...")
    print(f"URL: {url}")
    print(f"保存先: {zip_path}")

    try:
        download_file_with_progress(url, zip_path)
        print("[BlenderManager] ダウンロード完了。解凍中...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(cache_dir)

        # ダウンロードしたzipは削除（ディスク容量節約）
        if zip_path.exists():
            zip_path.unlink()

        # 解凍後のblender.exeを探す
        found_exes = list(cache_dir.glob(f"**/{target_name}/**/blender.exe"))
        if not found_exes:
            found_exes = list(cache_dir.glob(f"**/blender-{version_key}*/**/blender.exe"))
        if not found_exes:
            found_exes = list(extracted_dir.glob("**/blender.exe"))

        if found_exes:
            print(f"[BlenderManager] 解凍完了: {found_exes[0]}")
            return found_exes[0]
        else:
            raise FileNotFoundError(f"解凍されたディレクトリ内に blender.exe が見つかりませんでした: {cache_dir}")
    except Exception as e:
        if zip_path.exists():
            zip_path.unlink()
        raise RuntimeError(f"Blender {version} のダウンロードまたは解凍に失敗しました: {e}")


def resolve_blender(version_spec: Optional[str] = None) -> Path:
    """
    指定のバージョン要件に基づいてBlender実行バイナリを解決する。
    1. version_spec が指定されている場合（例: '4.2', '3.6', 'latest-lts'）:
       - そのバージョン（または最新LTS）を使用（ローカルにあればそれを使い、なければ自動DL）
    2. version_spec が未指定の場合:
       - 公式の最新LTSバージョン（例: 5.2）を自動解決して使用（キャッシュにあればそれを使い、なければ自動DL）
    """
    target_spec = version_spec.strip().lower() if version_spec else "latest-lts"
    if target_spec in ("latest-lts", "lts"):
        target_spec = get_latest_lts_version()

    installed = find_installed_blenders()

    # キャッシュ内に該当バージョンが存在するか検索
    for ver, path in installed:
        ver_str = ".".join(str(x) for x in ver)
        if ver_str.startswith(target_spec) or target_spec in str(path).lower():
            print(f"[BlenderManager] 要求バージョン {target_spec} に一致するBlenderを使用: {path} (v{ver_str})")
            return path

    # なければ公式から自動ダウンロード
    print(f"[BlenderManager] 要求バージョン {target_spec} がキャッシュに見つかりません。公式から自動ダウンロードを開始します。")
    return download_blender(target_spec)


def run_blender_unittest(
    blender_path: Path,
    test_pattern: Optional[str] = None,
    cwd: Optional[Path] = None,
    verbosity: int = 2,
) -> int:
    """Blenderのバックグラウンドプロセスでunittestを実行する"""
    if cwd is None:
        cwd = Path.cwd()

    pattern_arg = f"'{test_pattern}'" if test_pattern else "'*'"

    python_expr = f"""
import sys, unittest
sys.path.insert(0, '.')
suite = unittest.defaultTestLoader.discover('tests', pattern={pattern_arg})
runner = unittest.TextTestRunner(verbosity={verbosity})
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
""".strip().replace("\n", "; ")

    cmd = [
        str(blender_path),
        "--background",
        "--factory-startup",
        "--python-expr",
        python_expr,
    ]

    print(f"\n[BlenderManager] テスト実行コマンド:")
    print(f"  Blender: {blender_path}")
    print(f"  作業ディレクトリ: {cwd}")
    print("-" * 60)

    res = subprocess.run(cmd, cwd=str(cwd))
    return res.returncode
