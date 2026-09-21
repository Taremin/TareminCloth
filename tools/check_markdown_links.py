#!/usr/bin/env python3
"""
Markdown Link, Mermaid & Math Integrity Checker (tools/check_markdown_links.py)

GitHub Flavored Markdown (GFM) 上での表示破綻や構文エラーを包括的に事前検知します：
1. 絶対パス / file:/// スキームの混入検知
2. ローカル相対リンクの存在確認
3. 外部リンク (HTTP/HTTPS) の到達性確認
4. Mermaid ダイアグラムの GFM レンダリング構文エラー検知（未クォート括弧・br等）
5. KaTeX / MathJax 数式構文エラー検知 ('_' allowed only in math mode 等)
6. KaTeX 下付き添字の表記ミス検知（例: \mathbf{x}{old} の '_' 脱落）
7. 日本語強調（太字）と鉤括弧の組み合わせミス検知（例: **「...」** は 「**...**」 に）
"""

import glob
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Dict, List, Set, Tuple

# 単一引数コマンド（直後に {..} が来たら下付き '_' 脱落の疑い。\frac 等の2引数系は除外）
_SINGLE_ARG_MATH_CMDS = (
    "mathbf|mathrm|mathit|mathcal|mathbb|mathsf|mathtt|textbf|textit|"
    "boldsymbol|vec|hat|tilde|bar|dot|ddot|overline|underline|text"
)
_MISSING_SUBSCRIPT_RE = re.compile(
    r"\\(?:" + _SINGLE_ARG_MATH_CMDS + r")\{[^}]+\}\{"
)
# 素朴な下付き脱落 (例: x{old}。x_{old} は除外)
_BARE_SUBSCRIPT_RE = re.compile(r"(?<![_^\\{])\b[A-Za-z]\{[A-Za-z0-9]+\}")
# 太字スパン抽出
_BOLD_SPAN_RE = re.compile(r"\*\*(.+?)\*\*")
# インラインコード除去
_INLINE_CODE_RE = re.compile(r"`[^`]*`")

# Windows cp932 環境対策
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
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
        if "doi.org" in url and e.code == 403:
            opener = urllib.request.build_opener(NoRedirectHandler)
            try:
                doi_req = urllib.request.Request(url, headers=headers)
                with opener.open(doi_req, timeout=timeout) as doi_resp:
                    return (True, doi_resp.status, "DOI Resolved (302 Found)")
            except urllib.error.HTTPError as doi_e:
                if doi_e.code in (301, 302, 303, 307, 308):
                    return (True, doi_e.code, f"DOI Resolved to {doi_e.headers.get('Location', '')}")

        # 自リポジトリまたは GitHub Releases の場合（プライベートまたは初回リリース前は404が返るため許容）
        if "github.com" in url:
            if "taremin" in url.lower() and ("taremincloth" in url.lower() or "taremin_cloth" in url.lower()):
                return (True, 200, "Own repository link (Private or pending first release)")
            if url.endswith("/releases") and e.code == 404:
                repo_base = url[: -len("/releases")]
                try:
                    base_req = urllib.request.Request(repo_base, headers=headers)
                    with urllib.request.urlopen(base_req, timeout=timeout) as base_resp:
                        if 200 <= base_resp.status < 400:
                            return (True, 200, "Repository exists (Releases page will be created upon first release)")
                except Exception:
                    pass
        return (False, e.code, str(e.reason))
    except urllib.error.URLError as e:
        return (False, 0, str(e.reason))
    except Exception as e:
        return (False, 0, str(e))


def lint_markdown_content(file_path: str, lines: List[str]) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """Mermaid構文・KaTeX数式・日本語強調のGFM非互換エラーを検知する.

    戻り値: (mermaid_errors, math_errors, emphasis_errors)
    """
    mermaid_errors: List[Tuple[int, str]] = []
    math_errors: List[Tuple[int, str]] = []
    emphasis_errors: List[Tuple[int, str]] = []

    in_mermaid = False
    in_fenced_code = False
    in_math_block = False

    for line_no, line in enumerate(lines, 1):
        stripped = line.strip()

        # 0. フェンスコードブロックの追跡 (```mermaid 以外も強調・数式チェックから除外)
        if stripped.startswith("```"):
            if stripped.startswith("```mermaid"):
                in_mermaid = True
                in_fenced_code = True
            elif in_mermaid and stripped.startswith("```"):
                in_mermaid = False
                in_fenced_code = False
            else:
                in_fenced_code = not in_fenced_code
            continue
        if in_fenced_code:
            if in_mermaid:
                # 2-a: 矢印ラベル内の未クォート括弧検知 (例: -->|Label (with parens)|)
                label_matches = re.findall(r'\|([^\|]+)\|', line)
                for lbl in label_matches:
                    lbl_stripped = lbl.strip()
                    # ダブルクォートで囲まれていない場合
                    if not (lbl_stripped.startswith('"') and lbl_stripped.endswith('"')):
                        if any(c in lbl_stripped for c in ["(", ")"]):
                            mermaid_errors.append((
                                line_no,
                                f"Mermaidパイプラインラベルに未クォートの丸括弧が含まれています: '|{lbl}|' -> '|\"{lbl_stripped}\"|' に修正してください"
                            ))

                # 2-b: ノードテキスト内の未クォート <br> または括弧検知 (例: Node[text<br>O(1)])
                node_matches = re.findall(r'[a-zA-Z0-9_]+\[([^\]]+)\]', line)
                for n_txt in node_matches:
                    n_stripped = n_txt.strip()
                    if not (n_stripped.startswith('"') and n_stripped.endswith('"')):
                        if "<br>" in n_stripped and any(c in n_stripped for c in ["(", ")"]):
                            mermaid_errors.append((
                                line_no,
                                f"Mermaidノードテキスト内に未クォートの'<br>'および丸括弧が含まれています: '[{n_txt}]' -> '[\"{n_stripped}\"]' に修正してください"
                            ))
            continue

        # 3. KaTeX / MathJax 数式検証
        # インラインコードと表示数式ブロックを考慮して数式セグメントを抽出する
        code_stripped = _INLINE_CODE_RE.sub("", line)
        math_segments: List[str] = []
        tmp = code_stripped
        # $$...$$ ディスプレイ数式を先に抽出
        for m in re.findall(r"\$\$([^\$]+)\$\$", tmp):
            math_segments.append(m)
        tmp = re.sub(r"\$\$[^\$]+\$\$", " ", tmp)
        # 複数行 $$ ブロックの継続行は行全体を数式として扱う
        if code_stripped.count("$$") % 2 == 1:
            in_math_block = not in_math_block
        if in_math_block:
            math_segments.append(code_stripped.replace("$", " "))
        else:
            math_segments.extend(re.findall(r"\$([^$\n]+?)\$", tmp))
        for m_expr in math_segments:
            # \text{...} 内のアンダースコア検知
            text_blocks = re.findall(r'\\text\{([^}]+)\}', m_expr)
            for tb in text_blocks:
                if "_" in tb or r"\_" in tb:
                    math_errors.append((
                        line_no,
                        f"KaTeXエラー ('_' allowed only in math mode): \\text{{{tb}}} 内でアンダースコアが使われています。数式変数名 (例: d_{{eff}}) やハイフンに置き換えてください"
                    ))
            # 3-a: 下付き '_' 脱落検知 (例: \mathbf{x}{old} -> \mathbf{x}_{old})
            for bad in _MISSING_SUBSCRIPT_RE.findall(m_expr):
                math_errors.append((
                    line_no,
                    f"KaTeX下付き表記ミス: '{bad}{{...}}' のように単一引数コマンドの直後に '{{' が続いています。'_{{...}}' (例: \\mathbf{{x}}_{{old}}) に修正してください: ${m_expr.strip()}$"
                ))
            # 素朴な下付き脱落 (例: x{old} -> x_{old})。\frac 等の2引数系は _MISSING_SUBSCRIPT_RE で除外済み
            if "\\frac" not in m_expr and "\\binom" not in m_expr and "\\tfrac" not in m_expr and "\\dfrac" not in m_expr:
                for bad in _BARE_SUBSCRIPT_RE.findall(m_expr):
                    # 正規の \command{...} 引数 ([...]{...} 等) は除外: 直前が '\' または '{' の場合は上記正規表現で既に除外
                    math_errors.append((
                        line_no,
                        f"KaTeX下付き表記ミス: '{bad}' のように下付き '_' または上付き '^' が脱落しています。'{bad[0]}_{{{bad[2:-1]}}}' に修正してください: ${m_expr.strip()}$"
                    ))

        # 4. 日本語強調（太字）と鉤括弧の組み合わせミス検知
        # 正しい形: 「**...**」。誤りは以下の2パターンのみを FAIL とする:
        #   (a) 複数用語の助詞巻き込み: **「A」と「B」...** (」と「/」や「 を太字内に含む)
        #   (b) 長文巻き込み: 太字スパン内の鉤括弧外に長い日本語節が含まれる
        #       (例: **「GPU...」を結局...消滅** / **精度...ため、...では「不採用」**)
        # 単一用語の **「...」** や **「...」ボタン** 程度の短い接尾辞は許容する
        # (README/AGENTS の UI ラベル表記との互換性のため)。
        emphasis_target = _INLINE_CODE_RE.sub("", line)
        for span in _BOLD_SPAN_RE.findall(emphasis_target):
            if "「" not in span and "」" not in span:
                continue
            if "」と「" in span or "」や「" in span:
                emphasis_errors.append((
                    line_no,
                    f"日本語強調ミス: 太字 '**{span}**' が複数の鉤括弧用語を助詞ごと巻き込んでいます。「**A**」と「**B**」のように用語のみを太字にしてください"
                ))
                break
            # 鉤括弧部分を除去し、太字内に残る地の文の長さを評価
            outside = re.sub(r"「[^」]*」", "", span)
            # パンくず (>) やスラッシュ区切り等の記号のみは無視
            outside_stripped = re.sub(r"[\s>/／・|｜\-–—]+", "", outside)
            if len(outside_stripped) > 8:
                emphasis_errors.append((
                    line_no,
                    f"日本語強調ミス: 太字 '**{span}**' の内側に鉤括弧と長い説明文が混在しています。「**...**」のように用語のみを太字にし、説明文は太字の外に書いてください"
                ))
                break

    return mermaid_errors, math_errors, emphasis_errors


def inspect_markdown_file(file_path: str) -> Dict[str, any]:
    """Markdownファイルをスキャンしてリンク・Mermaid・数式構文を抽出・検証する."""
    external_urls: List[str] = []
    local_links: List[Tuple[int, str]] = []
    forbidden_links: List[Tuple[int, str, str]] = []

    md_link_pattern = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')
    raw_url_pattern = re.compile(r'https?://[^\s\)\]\"\'<>]+')

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_no, line in enumerate(lines, 1):
        for m in raw_url_pattern.findall(line):
            clean_url = m.rstrip(".,;:)")
            external_urls.append(clean_url)

        for text, target in md_link_pattern.findall(line):
            target = target.strip()
            if target.startswith("file://") or re.match(r'^[a-zA-Z]:[\\/]', target):
                forbidden_links.append((line_no, target, "絶対パス・file:/// スキームが使用されています（環境依存リンク）"))
                continue
            if target.startswith("http://") or target.startswith("https://"):
                clean_target = target.rstrip(".,;:)")
                if clean_target not in external_urls:
                    external_urls.append(clean_target)
                continue
            if target.startswith("#"):
                continue
            local_links.append((line_no, target))

    mermaid_errors, math_errors, emphasis_errors = lint_markdown_content(file_path, lines)

    return {
        "external_urls": external_urls,
        "local_links": local_links,
        "forbidden_links": forbidden_links,
        "mermaid_errors": mermaid_errors,
        "math_errors": math_errors,
        "emphasis_errors": emphasis_errors,
    }


def expand_globs(args: List[str]) -> List[str]:
    """Windows等のシェルで未展開のワイルドカード引数を展開する."""
    expanded: List[str] = []
    for arg in args:
        if any(c in arg for c in ["*", "?", "["]):
            matched = glob.glob(arg, recursive=True)
            if matched:
                expanded.extend(matched)
            else:
                expanded.append(arg)
        else:
            expanded.append(arg)
    return expanded


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/check_markdown_links.py <file1.md> [file2.md ...]")
        sys.exit(1)

    raw_args = sys.argv[1:]
    files = expand_globs(raw_args)

    all_success = True
    checked_external_urls: Set[str] = set()

    print("=" * 65)
    print("GFM Markdown Integrity & Link Checker (Links / Mermaid / Math / Emphasis)")
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

        # 2. Mermaid 構文検証
        mermaid_errs = scan_res["mermaid_errors"]
        if mermaid_errs:
            print(f"  [FAIL] {len(mermaid_errs)} Mermaid syntax error(s) detected (GitHub Render Fail):")
            for line_no, err_msg in mermaid_errs:
                print(f"    Line {line_no}: {err_msg}")
            all_success = False
        else:
            print("  [PASS] Mermaid diagrams syntax valid.")

        # 3. KaTeX / Math 構文検証
        math_errs = scan_res["math_errors"]
        if math_errs:
            print(f"  [FAIL] {len(math_errs)} KaTeX / Math syntax error(s) detected:")
            for line_no, err_msg in math_errs:
                print(f"    Line {line_no}: {err_msg}")
            all_success = False
        else:
            print("  [PASS] Math formulas syntax valid.")

        # 3b. 日本語強調（太字・鉤括弧）検証
        emphasis_errs = scan_res.get("emphasis_errors", [])
        if emphasis_errs:
            print(f"  [FAIL] {len(emphasis_errs)} Japanese emphasis error(s) detected:")
            for line_no, err_msg in emphasis_errs:
                print(f"    Line {line_no}: {err_msg}")
            all_success = False
        else:
            print("  [PASS] Japanese emphasis syntax valid.")

        # 4. ローカル相対パス検証
        local_links = scan_res["local_links"]
        if local_links:
            print(f"  Checking {len(local_links)} local relative link(s)...")
            for line_no, target in local_links:
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

        # 5. 外部リンク検証
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
        print("ALL GFM MARKDOWN CHECKS PASSED SUCCESSFULLY!")
        print("=" * 65)
        sys.exit(0)
    else:
        print("SOME MARKDOWN CHECKS FAILED! Please fix the errors above.")
        print("=" * 65)
        sys.exit(1)


if __name__ == "__main__":
    main()
