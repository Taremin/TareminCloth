#!/usr/bin/env python3
"""
Taremin Cloth Changelog 生成ツール (tools/generate_changelog.py)

Git のコミット履歴から Conventional Commits 形式 (feat, fix, perf, etc.) を解析し、
カテゴリ別に分類した Markdown 形式のリリースノートを自動生成します。

使用方法:
  python tools/generate_changelog.py                       # 直近タグ〜HEAD のチェンジログを標準出力に表示
  python tools/generate_changelog.py --tag v0.0.2          # 指定タグ用のチェンジログを生成
  python tools/generate_changelog.py --output changelog.md # ファイルへ出力
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Windows cp932 環境対策
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


CATEGORIES = [
    ("feat", "🚀 新機能 (Features)"),
    ("fix", "🐛 不具合修正 (Bug Fixes)"),
    ("perf", "⚡ パフォーマンス改善 (Performance Improvements)"),
    ("refactor", "♻️ リファクタリング (Refactoring)"),
    ("docs", "📝 ドキュメント (Documentation)"),
    ("test", "🧪 テスト (Testing)"),
    ("chore", "🔧 その他・保守 (Maintenance & Chores)"),
    ("other", "📦 その他の変更 (Other Changes)"),
]

PREFIX_MAP = {
    "feat": "feat",
    "feature": "feat",
    "fix": "fix",
    "bugfix": "fix",
    "perf": "perf",
    "performance": "perf",
    "refactor": "refactor",
    "docs": "docs",
    "doc": "docs",
    "test": "test",
    "tests": "test",
    "chore": "chore",
    "ci": "chore",
    "build": "chore",
    "style": "chore",
}


def run_git(cmd: list[str], repo_root: Path) -> str:
    res = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        return ""
    return res.stdout.strip()


def find_previous_tag(current_tag: str | None, repo_root: Path) -> str | None:
    """現在のタグの直前にあるリリースタグを探す"""
    if current_tag and "-" in current_tag:
        # プレリリースタグの場合: HEADの直前の任意のタグを取得
        prev = run_git(["git", "describe", "--tags", "--abbrev=0", "HEAD^"], repo_root)
        if prev:
            return prev

    # すべてのタグをバージョン順に取得
    tags_output = run_git(["git", "tag", "-l", "v*", "--sort=-v:refname"], repo_root)
    if not tags_output:
        return None

    all_tags = [t.strip() for t in tags_output.splitlines() if t.strip()]

    # current_tag を除外
    if current_tag and current_tag in all_tags:
        all_tags.remove(current_tag)

    if not all_tags:
        return None

    # current_tag がプレリリースでない場合は、プレリリースタグを除外した直近本番タグを優先
    if not current_tag or "-" not in current_tag:
        release_tags = [t for t in all_tags if not any(pre in t.lower() for pre in ["-alpha", "-beta", "-rc"])]
        if release_tags:
            return release_tags[0]

    return all_tags[0]


def parse_commit_line(line: str) -> tuple[str, str, str, str]:
    """コミット行 (hash, subject) をパースして (hash, category, scope, clean_subject) を返す"""
    parts = line.split(" ", 1)
    commit_hash = parts[0]
    subject = parts[1] if len(parts) > 1 else ""

    # 例: "feat(simulation): 自己衝突の改善" または "feat: 自己衝突の改善"
    pattern = r"^([a-zA-Z]+)(?:\(([^)]+)\))?\s*:\s*(.+)$"
    match = re.match(pattern, subject)

    if match:
        raw_type, scope, clean_subj = match.groups()
        cat = PREFIX_MAP.get(raw_type.lower(), "other")
        return commit_hash, cat, scope or "", clean_subj.strip()
    else:
        return commit_hash, "other", "", subject.strip()


def generate_changelog(
    repo_root: Path,
    current_tag: str | None = None,
    previous_tag: str | None = None,
    repo_slug: str = "Taremin/TareminCloth",
) -> str:
    if previous_tag is None:
        previous_tag = find_previous_tag(current_tag, repo_root)

    # コミット範囲の決定
    if previous_tag:
        revision_range = f"{previous_tag}..HEAD"
        header_range_str = f"{previous_tag} ... {current_tag or 'HEAD'}"
    else:
        revision_range = "HEAD"
        header_range_str = f"{current_tag or 'Initial Release'}"

    # コミット一覧の取得 (マージコミットを除外: --no-merges)
    git_log_cmd = ["git", "log", "--pretty=format:%h %s", "--no-merges"]
    if previous_tag:
        git_log_cmd.append(revision_range)

    log_output = run_git(git_log_cmd, repo_root)
    if not log_output:
        return f"## 変更内容 ({header_range_str})\n\n- 主な更新はありません。"

    categorized_commits: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for line in log_output.splitlines():
        line = line.strip()
        if not line:
            continue
        commit_hash, cat, scope, clean_subj = parse_commit_line(line)

        # release コミット自体は除外
        if clean_subj.startswith("chore(release):") or clean_subj.startswith("Release v"):
            continue

        categorized_commits[cat].append((commit_hash, scope, clean_subj))

    # Markdown 生成
    lines: list[str] = []
    lines.append(f"## 変更内容 ({header_range_str})\n")

    has_content = False
    for cat_key, cat_title in CATEGORIES:
        commits = categorized_commits.get(cat_key, [])
        if not commits:
            continue
        has_content = True
        lines.append(f"### {cat_title}")
        for chash, scope, subj in commits:
            scope_prefix = f"**{scope}**: " if scope else ""
            lines.append(f"- {scope_prefix}{subj} ({chash})")
        lines.append("")

    if not has_content:
        lines.append("- 各種機能の改善と調整\n")

    # 差分比較リンク
    if previous_tag and current_tag:
        lines.append("---")
        compare_url = f"https://github.com/{repo_slug}/compare/{previous_tag}...{current_tag}"
        lines.append(f"**詳細な差分比較**: [{previous_tag}...{current_tag}]({compare_url})")

    return "\n".join(lines).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Taremin Cloth リリースノート・Changelog 自動生成ツール")
    parser.add_argument("--tag", "-t", type=str, default=None, help="リリース対象タグ名 (例: v0.0.1)")
    parser.add_argument("--previous-tag", "-p", type=str, default=None, help="比較元となる前のタグ名")
    parser.add_argument("--output", "-o", type=Path, default=None, help="出力先ファイルパス (未指定時は標準出力)")
    parser.add_argument("--repo", type=str, default="Taremin/TareminCloth", help="GitHub リポジトリ名 (owner/repo)")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    changelog_md = generate_changelog(
        repo_root=repo_root,
        current_tag=args.tag,
        previous_tag=args.previous_tag,
        repo_slug=args.repo,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(changelog_md, encoding="utf-8")
        print(f"[+] Changelog を出力しました: {args.output}")
    else:
        print(changelog_md)


if __name__ == "__main__":
    main()
