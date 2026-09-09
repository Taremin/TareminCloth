#!/usr/bin/env python3
"""Markdownファイル内のリンク実在性・健全性自動検証ツール.

検証項目:
    1. HTTP/HTTPS 外部リンクの到達可能性 (200 OK / 302 Found)
    2. 禁止されたURIスキームの検出 (file:/// や Windows絶対パス C:\\, D:\\ の混入検知)
    3. ローカル相対パスリンク (例: [text](../path/to/file.rs)) の実在性検証

使用例:
    python tools/check_markdown_links.py docs/algorithms.md
    python tools/check_markdown_links.py docs/algorithms.md AGENTS.md docs/architecture.md
"""

import os
import re
import sys
import urllib.request
import urllib.error
from typing import List, Tuple, Set, Dict


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # リダイレクトを自動追従せず停止するハンドラ（DOIリゾルバの302確認用）
        return None


def check_external_url(url: str, timeout: float = 10.0) -> Tuple[bool, int, str]:
    """外部リンク (HTTP/HTTPS) の到達可能性を検証する."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            return (200 <= status < 400, status, "OK")
    except urllib.error.HTTPError as e:
        # doi.org の場合は302リダイレクトが返ればDOIが有効に登録・解決されている
        if "doi.org" in url and e.code == 403:
            opener = urllib.request.build_opener(NoRedirectHandler)
            try:
                doi_req = urllib.request.Request(url, headers=headers)
                with opener.open(doi_req, timeout=timeout) as doi_resp:
                    return (True, doi_resp.status, "DOI Resolved (302 Found)")
            except urllib.error.HTTPError as doi_e:
                if doi_e.code in (301, 302, 303, 307, 308):
                    return (True, doi_e.code, f"DOI Resolved to {doi_e.headers.get('Location', '')}")
        return (False, e.code, str(e.reason))
    except urllib.error.URLError as e:
        return (False, 0, str(e.reason))
    except Exception as e:
        return (False, 0, str(e))


def inspect_markdown_file(file_path: str) -> Dict[str, any]:
    """Markdownファイルをスキャンしてリンク・禁止構文を抽出する."""
    external_urls: List[str] = []
    local_links: List[Tuple[int, str]] = []
    forbidden_links: List[Tuple[int, str, str]] = []

    # [text](target) の正規表現
    md_link_pattern = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')
    # 生の URL
    raw_url_pattern = re.compile(r'https?://[^\s\)\]\"\'<>]+')

    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            # 生の外部URL抽出
            for m in raw_url_pattern.findall(line):
                clean_url = m.rstrip(".,;:)")
                external_urls.append(clean_url)

            # Markdownリンク抽出
            for text, target in md_link_pattern.findall(line):
                target = target.strip()

                # 1. 禁止されたスキーム/絶対パスのチェック
                if target.startswith("file://") or re.match(r'^[a-zA-Z]:[\\/]', target):
                    forbidden_links.append((line_no, target, "絶対パス・file:/// スキームが使用されています（環境依存リンク）"))
                    continue

                # 2. 外部URL
                if target.startswith("http://") or target.startswith("https://"):
                    clean_target = target.rstrip(".,;:)")
                    if clean_target not in external_urls:
                        external_urls.append(clean_target)
                    continue

                # 3. ページ内アンカー (#...)
                if target.startswith("#"):
                    continue

                # 4. ローカル相対パスリンク
                local_links.append((line_no, target))

    return {
        "external_urls": external_urls,
        "local_links": local_links,
        "forbidden_links": forbidden_links,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/check_markdown_links.py <file1.md> [file2.md ...]")
        sys.exit(1)

    files = sys.argv[1:]
    all_success = True
    checked_external_urls: Set[str] = set()

    print("=" * 65)
    print("Markdown Link & Integrity Checker")
    print("=" * 65)

    for file_path in files:
        print(f"\n[Scanning File]: {file_path}")
        if not os.path.exists(file_path):
            print(f"  [ERROR] File not found: {file_path}")
            all_success = False
            continue

        base_dir = os.path.dirname(os.path.abspath(file_path))
        scan_res = inspect_markdown_file(file_path)

        # 1. 禁止スキーム検証
        forbidden = scan_res["forbidden_links"]
        if forbidden:
            print(f"  [FAIL] {len(forbidden)} forbidden absolute/file link(s) detected:")
            for line_no, target, reason in forbidden:
                print(f"    Line {line_no}: {target} -> {reason}")
            all_success = False
        else:
            print("  [PASS] No forbidden absolute / file:/// links found.")

        # 2. ローカル相対パス検証
        local_links = scan_res["local_links"]
        if local_links:
            print(f"  Checking {len(local_links)} local relative link(s)...")
            for line_no, target in local_links:
                # アンカー (#...) を除いたファイルパスを取得
                clean_target = target.split("#")[0]
                if not clean_target:
                    continue
                target_abs = os.path.normpath(os.path.join(base_dir, clean_target))
                if os.path.exists(target_abs):
                    rel_to_repo = os.path.relpath(target_abs, os.getcwd())
                    print(f"    [PASS] (Line {line_no}) {target} -> exists ({rel_to_repo})")
                else:
                    print(f"    [FAIL] (Line {line_no}) {target} -> NOT FOUND ({target_abs})")
                    all_success = False
        else:
            print("  No local relative links found.")

        # 3. 外部リンク検証
        external_urls = scan_res["external_urls"]
        unique_urls = [u for u in external_urls if u not in checked_external_urls]
        if unique_urls:
            print(f"  Checking {len(unique_urls)} unique external link(s)...")
            for url in unique_urls:
                checked_external_urls.add(url)
                success, code, msg = check_external_url(url)
                if success:
                    print(f"    [PASS] ({code}) {url}")
                else:
                    print(f"    [FAIL] ({code}: {msg}) {url}")
                    all_success = False
        else:
            print("  No new external links to check.")

    print("\n" + "=" * 65)
    if all_success:
        print("ALL MARKDOWN CHECKS PASSED SUCCESSFULLY!")
        print("=" * 65)
        sys.exit(0)
    else:
        print("SOME MARKDOWN CHECKS FAILED!")
        print("=" * 65)
        sys.exit(1)


if __name__ == "__main__":
    main()
