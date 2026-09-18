#!/usr/bin/env python3
"""
Taremin Cloth ワンストップリリースCLIツール (tools/release.py)

バージョン更新、品質検証テスト、Gitコミット、タグ付け、リモートプッシュ案内を
一括で実行する自動化スクリプトです。

使用方法:
  python tools/release.py                     # 対話型メニューで選択
  python tools/release.py patch               # パッチバージョンアップ (0.0.1 -> 0.0.2)
  python tools/release.py minor               # マイナーバージョンアップ (0.0.1 -> 0.1.0)
  python tools/release.py major               # メジャーバージョンアップ (0.0.1 -> 1.0.0)
  python tools/release.py major-beta          # メジャーベータ (0.0.1 -> 1.0.0-beta.1)
  python tools/release.py major-rc            # メジャーRC (0.0.1 -> 1.0.0-rc.1)
  python tools/release.py minor-beta          # マイナーベータ (0.0.1 -> 0.1.0-beta.1)
  python tools/release.py minor-rc            # マイナーRC (0.0.1 -> 0.1.0-rc.1)
  python tools/release.py patch-beta          # パッチベータ (0.0.1 -> 0.0.2-beta.1)
  python tools/release.py next                # プレリリース番号インクリメント (1.0.0-beta.1 -> 1.0.0-beta.2)
  python tools/release.py release             # プレリリースから本番昇格 (1.0.0-rc.1 -> 1.0.0)
  python tools/release.py 0.0.1               # 任意バージョン直接指定

オプション:
  --dry-run       ファイル変更やGit操作を行わず、動作内容をシミュレーション表示
  --skip-tests    事前テスト（cargo test, python unit test, link check）をスキップ
  --no-push       Git push を行わずローカルコミット・タグ作成のみで終了
  -y, --yes       すべての確認プロンプトに 'yes' と回答（非対話実行向け）
  --check         リポジトリ内の各ファイルのバージョン整合性チェックのみ実行
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Windows cp932 環境対策
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class SemVer:
    major: int
    minor: int
    patch: int
    prerelease: str | None = None  # 例: "beta.1", "rc.2", "alpha.1"

    @classmethod
    def parse(cls, version_str: str) -> SemVer:
        clean_str = version_str.strip().lstrip("v")
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?$"
        match = re.match(pattern, clean_str)
        if not match:
            raise ValueError(f"無効なセマンティックバージョン形式です: '{version_str}' (例: '0.1.0', '1.0.0-beta.1')")
        major, minor, patch, prerelease = match.groups()
        return cls(int(major), int(minor), int(patch), prerelease)

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            return f"{base}-{self.prerelease}"
        return base

    @property
    def tag_name(self) -> str:
        return f"v{self}"

    @property
    def bl_info_tuple(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    def bump(self, bump_type: str, pre_id: str | None = None) -> SemVer:
        """指定された種別に基づいて新しい SemVer を生成する"""
        bt = bump_type.lower().replace("_", "-")

        # 1. 複合キーワードの分解 (例: "major-beta" -> bump="major", pre_id="beta")
        for pre in ["alpha", "beta", "rc"]:
            if bt.endswith(f"-{pre}"):
                base_part = bt[: -len(f"-{pre}")]
                return self._bump_base(base_part, pre)

        # 2. pre_id が別途指定されている場合
        if pre_id:
            return self._bump_base(bt, pre_id.lower())

        # 3. 通常の bump
        if bt == "major":
            return SemVer(self.major + 1, 0, 0, None)
        elif bt == "minor":
            return SemVer(self.major, self.minor + 1, 0, None)
        elif bt == "patch":
            return SemVer(self.major, self.minor, self.patch + 1, None)
        elif bt in ["next", "prerelease"]:
            if not self.prerelease:
                raise ValueError("現在のバージョンにプレリリース識別子がないため 'next' は指定できません。")
            # 例: "beta.1" -> "beta.2" または "rc" -> "rc.2"
            parts = self.prerelease.split(".")
            pre_name = parts[0]
            num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            return SemVer(self.major, self.minor, self.patch, f"{pre_name}.{num + 1}")
        elif bt == "release":
            if not self.prerelease:
                print(f"[*] 現在のバージョン ({self}) は既に本番リリース版です。")
                return SemVer(self.major, self.minor, self.patch, None)
            return SemVer(self.major, self.minor, self.patch, None)
        else:
            # 直接バージョン指定としてパース試行
            return SemVer.parse(bump_type)

    def _bump_base(self, base_type: str, pre_type: str) -> SemVer:
        if base_type == "major":
            return SemVer(self.major + 1, 0, 0, f"{pre_type}.1")
        elif base_type == "minor":
            return SemVer(self.major, self.minor + 1, 0, f"{pre_type}.1")
        elif base_type == "patch":
            return SemVer(self.major, self.minor, self.patch + 1, f"{pre_type}.1")
        else:
            raise ValueError(f"不明なベースリリース種別です: '{base_type}'")


# バージョン管理対象ファイル定義
def get_version_files(repo_root: Path) -> list[tuple[str, Path, str]]:
    """(ファイル種別, ファイルパス, 抽出・置換パターン種別)"""
    return [
        ("Blender Addon", repo_root / "__init__.py", "init_py"),
        ("PyPI / Maturin", repo_root / "pyproject.toml", "pyproject_toml"),
        ("Cargo PyO3 Core", repo_root / "crates" / "cloth_py" / "Cargo.toml", "cargo_toml"),
        ("Cargo Core", repo_root / "crates" / "cloth_core" / "Cargo.toml", "cargo_toml"),
        ("Cargo GUI", repo_root / "crates" / "cloth_gui" / "Cargo.toml", "cargo_toml"),
    ]


def read_file_version(path: Path, kind: str) -> str | None:
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8")
    if kind == "init_py":
        match = re.search(r'"version"\s*:\s*\(([0-9,\s]+)\)', content)
        if match:
            nums = [p.strip() for p in match.group(1).split(",") if p.strip()]
            return ".".join(nums)
    elif kind in ["pyproject_toml", "cargo_toml"]:
        match = re.search(r'version\s*=\s*"([^"]+)"', content)
        if match:
            return match.group(1)
    return None


def write_file_version(path: Path, kind: str, new_ver: SemVer) -> None:
    content = path.read_text(encoding="utf-8")
    if kind == "init_py":
        t = new_ver.bl_info_tuple
        new_content = re.sub(
            r'"version"\s*:\s*\([0-9,\s]+\)',
            f'"version": ({t[0]}, {t[1]}, {t[2]})',
            content,
            count=1,
        )
    elif kind in ["pyproject_toml", "cargo_toml"]:
        new_content = re.sub(
            r'version\s*=\s*"[^"]+"',
            f'version = "{new_ver}"',
            content,
            count=1,
        )
    else:
        return
    path.write_text(new_content, encoding="utf-8")


def check_version_consistency(repo_root: Path) -> tuple[bool, SemVer | None, dict[str, str]]:
    files = get_version_files(repo_root)
    versions: dict[str, str] = {}
    parsed_semvers: list[SemVer] = []

    for name, path, kind in files:
        v = read_file_version(path, kind)
        versions[name] = v or "未検出"
        if v:
            try:
                parsed_semvers.append(SemVer.parse(v))
            except Exception:
                pass

    if not parsed_semvers:
        return False, None, versions

    primary_ver = parsed_semvers[1] if len(parsed_semvers) > 1 else parsed_semvers[0]
    all_match = True
    for name, path, kind in files:
        cur_v = versions.get(name)
        if kind == "init_py":
            expected = f"{primary_ver.major}.{primary_ver.minor}.{primary_ver.patch}"
            if cur_v != expected:
                all_match = False
        else:
            if cur_v != str(primary_ver):
                all_match = False

    return all_match, primary_ver, versions


def run_cmd(cmd: list[str], cwd: Path, desc: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"[*] 実行中: {desc} ({' '.join(cmd)})")
    res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if check and res.returncode != 0:
        print(f"[!] エラーが発生しました ({desc}):", file=sys.stderr)
        if res.stdout:
            print("--- 標準出力 ---", file=sys.stderr)
            print(res.stdout, file=sys.stderr)
        if res.stderr:
            print("--- 標準エラー出力 ---", file=sys.stderr)
            print(res.stderr, file=sys.stderr)
        sys.exit(1)
    return res


def prompt_yes_no(question: str, default: bool = False, auto_yes: bool = False) -> bool:
    if auto_yes:
        print(f"{question} [Y/n]: Y (自動承認)")
        return True
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        try:
            resp = input(question + suffix).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n[!] 処理を中断しました。")
            sys.exit(1)
        if not resp:
            return default
        if resp in ["y", "yes"]:
            return True
        if resp in ["n", "no"]:
            return False
        print("  'y' または 'n' で回答してください。")


def interactive_select_version(current: SemVer) -> SemVer:
    print("\n" + "=" * 62)
    print(f"  Taremin Cloth リリースウィザード (現在: v{current})")
    print("=" * 62)
    options = [
        ("Patch リリース", current.bump("patch")),
        ("Minor リリース", current.bump("minor")),
        ("Major リリース", current.bump("major")),
        ("Minor Beta (プレリリース)", current.bump("minor-beta")),
        ("Minor RC (リリース候補)", current.bump("minor-rc")),
        ("Major Beta (プレリリース)", current.bump("major-beta")),
        ("Major RC (リリース候補)", current.bump("major-rc")),
    ]
    if current.prerelease:
        options.insert(0, ("プレリリース番号進捗 (Next)", current.bump("next")))
        options.insert(1, ("正式リリース版へ昇格 (Release)", current.bump("release")))

    for i, (label, target) in enumerate(options, 1):
        print(f"  {i}) {label:<28} -> {target} (tag: {target.tag_name})")
    custom_idx = len(options) + 1
    print(f"  {custom_idx}) 任意のバージョン文字列を直接入力")

    while True:
        try:
            choice = input(f"\n番号を選択してください [1-{custom_idx}]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[!] 中断しました。")
            sys.exit(1)
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1][1]
            elif idx == custom_idx:
                val = input("新しいバージョン文字列を入力 (例: 1.0.0-beta.1): ").strip()
                try:
                    return SemVer.parse(val)
                except ValueError as e:
                    print(f"  [!] {e}")
                    continue
        print(f"  1 から {custom_idx} の数字を入力してください。")


def main():
    parser = argparse.ArgumentParser(
        description="Taremin Cloth ワンストップリリースCLIツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="リリース種別 (patch, minor, major, major-beta, major-rc, minor-beta, minor-rc, next, release) またはバージョン (例: 0.0.1)",
    )
    parser.add_argument("--beta", action="store_true", help="ベータプレリリース (--pre beta と等価)")
    parser.add_argument("--rc", action="store_true", help="リリース候補プレリリース (--pre rc と等価)")
    parser.add_argument("--pre", type=str, default=None, help="プレリリース識別子 (例: alpha, beta, rc)")
    parser.add_argument("--check", action="store_true", help="バージョン整合性チェックのみ実行して終了")
    parser.add_argument("--dry-run", action="store_true", help="ファイル変更やGit操作を行わずシミュレーション表示")
    parser.add_argument("--skip-tests", action="store_true", help="品質検証テスト（Rust/Python/Links）をスキップ")
    parser.add_argument("--no-push", action="store_true", help="Gitリモートプッシュを行わない")
    parser.add_argument("-y", "--yes", action="store_true", help="すべての確認プロンプトを自動承認")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    # 1. バージョン整合性チェック
    consistent, current_semver, version_map = check_version_consistency(repo_root)

    print("\n[1/6] [*] バージョン整合性ステータス:")
    for name, ver in version_map.items():
        print(f"  - {name:<20}: {ver}")

    if args.check:
        if consistent and current_semver:
            print(f"\n[+] 全ファイルのバージョンが正常に一致しています: {current_semver}")
            sys.exit(0)
        else:
            print("\n[!] 警告: バージョンの不整合が検出されました。", file=sys.stderr)
            sys.exit(1)

    if not current_semver:
        print("[!] 現在のバージョンを特定できませんでした。__init__.py を確認してください。", file=sys.stderr)
        sys.exit(1)

    # 2. 新バージョンの決定
    target_arg = args.target
    pre_id = args.pre
    if args.beta:
        pre_id = "beta"
    elif args.rc:
        pre_id = "rc"

    if target_arg is None and not pre_id:
        target_semver = interactive_select_version(current_semver)
    else:
        target_type = target_arg or "patch"
        try:
            target_semver = current_semver.bump(target_type, pre_id=pre_id)
        except ValueError as e:
            print(f"[!] バージョン計算エラー: {e}", file=sys.stderr)
            sys.exit(1)

    print("\n" + "-" * 50)
    print(f"  変更前バージョン : {current_semver}")
    print(f"  変更後バージョン : {target_semver} (Git Tag: {target_semver.tag_name})")
    print("-" * 50)

    # Changelog プレビュー表示
    try:
        from tools.generate_changelog import generate_changelog
        cl_preview = generate_changelog(repo_root, current_tag=target_semver.tag_name)
        print("\n--- [Changelog プレビュー] ---")
        print(cl_preview.strip())
        print("------------------------------\n")
    except Exception as e:
        pass

    if not args.yes and not args.dry_run:
        if not prompt_yes_no("このバージョンでリリース作業を開始しますか？", default=True, auto_yes=args.yes):
            print("[*] リリースを中止しました。")
            sys.exit(0)

    # 3. 事前検証 (Gitワーキングツリー & ブランチ)
    print("\n[2/6] [*] Git 状態確認...")
    status_res = run_cmd(["git", "status", "--porcelain"], repo_root, "Git ステータス確認")
    dirty_files = [line for line in status_res.stdout.splitlines() if not line.strip().startswith("??")]
    if dirty_files and not args.dry_run:
        print("[!] 未コミットの変更が存在します:", file=sys.stderr)
        for df in dirty_files[:5]:
            print(f"    {df}", file=sys.stderr)
        if not prompt_yes_no("未コミットの変更を維持したまま続行しますか？", default=False, auto_yes=args.yes):
            print("[*] 変更をコミットまたは退避してから再実行してください。")
            sys.exit(1)

    branch_res = run_cmd(["git", "branch", "--show-current"], repo_root, "ブランチ確認")
    current_branch = branch_res.stdout.strip()
    if current_branch not in ["master", "main"] and not args.dry_run:
        print(f"[!] 現在のブランチは '{current_branch}' です（リリースは通常 master/main で行います）。")
        if not prompt_yes_no(f"ブランチ '{current_branch}' でリリースを続行しますか？", default=True, auto_yes=args.yes):
            sys.exit(1)

    # 4. バージョンの書き換え
    print(f"\n[3/6] [*] バージョンファイルを更新中 -> {target_semver}")
    files_to_stage: list[Path] = []
    for name, path, kind in get_version_files(repo_root):
        if not args.dry_run:
            write_file_version(path, kind, target_semver)
            files_to_stage.append(path)
            print(f"  [+] 更新: {path.relative_to(repo_root)}")
        else:
            print(f"  [DRY-RUN] 更新予定: {path.relative_to(repo_root)}")

    # Cargo.lock の更新
    print("[*] Cargo.lock を同期中 (cargo check --workspace)...")
    if not args.dry_run:
        run_cmd(["cargo", "check", "--workspace"], repo_root, "Cargo lock 同期")
        cargo_lock = repo_root / "Cargo.lock"
        if cargo_lock.exists():
            files_to_stage.append(cargo_lock)
    else:
        print("  [DRY-RUN] cargo check --workspace 実行予定")

    # 5. 品質検証テストの実行
    if not args.skip_tests:
        print("\n[4/6] [*] 品質検証テストの自動実行...")
        if not args.dry_run:
            # 5-1: Rust tests
            print("[*] Rust 単体テスト & WGSLアライメント検証を実行中 (cargo test)...")
            run_cmd(["cargo", "test"], repo_root, "Rust 単体テスト")

            # 5-2: Python tests
            print("[*] Python 単体テストを実行中 (python -m unittest discover -s tests/core -t .)...")
            run_cmd([sys.executable, "-m", "unittest", "discover", "-s", "tests/core", "-t", "."], repo_root, "Python 単体テスト")

            # 5-3: Markdown Link, Mermaid & Math Check
            print("[*] ドキュメント整合性を検証中 (check_markdown_links)...")
            chk_script = repo_root / "tools" / "check_markdown_links.py"
            if chk_script.exists():
                run_cmd([sys.executable, str(chk_script), "README.md", "AGENTS.md", "docs/*.md"], repo_root, "ドキュメント整合性・リンク検証")
            print("[+] すべての事前テストに合格しました！")
        else:
            print("  [DRY-RUN] cargo test, python unittest, link check 実行予定")
    else:
        print("\n[4/6] [*] テスト実行をスキップしました (--skip-tests)")

    # 6. Git コミット & タグ作成
    print(f"\n[5/6] [*] Git コミットおよびタグ作成 (Tag: {target_semver.tag_name})...")
    commit_msg = f"chore(release): {target_semver.tag_name}"
    tag_name = target_semver.tag_name

    if not args.dry_run:
        for f in files_to_stage:
            run_cmd(["git", "add", str(f.relative_to(repo_root))], repo_root, f"Git add {f.name}")

        run_cmd(["git", "commit", "-m", commit_msg], repo_root, "Git コミット")
        print(f"  [+] コミット作成完了: {commit_msg}")

        tag_msg = f"Release {tag_name}"
        run_cmd(["git", "tag", "-a", tag_name, "-m", tag_msg], repo_root, f"Git タグ作成 ({tag_name})")
        print(f"  [+] タグ作成完了: {tag_name}")
    else:
        print(f"  [DRY-RUN] git add {' '.join(str(f.relative_to(repo_root)) for f in files_to_stage)}")
        print(f"  [DRY-RUN] git commit -m '{commit_msg}'")
        print(f"  [DRY-RUN] git tag -a {tag_name} -m 'Release {tag_name}'")

    # 7. リモートプッシュ
    print(f"\n[6/6] [*] リモートプッシュおよび GitHub Actions リリース起動...")
    push_cmd = ["git", "push", "origin", current_branch, "--tags"]

    if args.dry_run:
        print(f"  [DRY-RUN] 実行予定コマンド: {' '.join(push_cmd)}")
        print("\n[+] [DRY-RUN] シミュレーション完了！実際のリクエスト時は --dry-run を外して実行してください。")
        return

    if args.no_push:
        print("  [*] --no-push が指定されているため、プッシュはスキップされました。")
        print(f"\n  GitHub Releases への自動公開を開始するには、以下のコマンドを手動で実行してください:")
        print(f"    {' '.join(push_cmd)}\n")
        return

    should_push = prompt_yes_no(
        f"今すぐリモートへプッシュして GitHub Actions を起動しますか？\n({' '.join(push_cmd)})",
        default=True,
        auto_yes=args.yes,
    )

    if should_push:
        run_cmd(push_cmd, repo_root, "リモートへプッシュ")
        print("\n" + "=" * 62)
        print(f"[+] リリース {tag_name} のプッシュが正常に完了しました！")
        print("=" * 62)
        print("GitHub Actions (Release Addon ワークフロー) が自動起動されました。")
        print(f"ビルド完了後、以下でアドオン ZIP が公開されます:")
        print(f"  https://github.com/Taremin/TareminCloth/releases/tag/{tag_name}")
    else:
        print("\nプッシュを保留しました。準備が整い次第、以下のコマンドを実行してください:")
        print(f"  {' '.join(push_cmd)}\n")


if __name__ == "__main__":
    main()
