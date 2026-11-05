"""
Taremin Cloth 関数引数設計（キーワード専用引数・ブールトラップ防止）監査CLIツール

Blender非依存でPython AST（抽象構文木）を解析し、
アドオンコード内の全関数・メソッドに対して以下の引数安全規約を監査します:

1. [Boolean Trap]: bool型の引数が位置引数として定義されていないか (* 以降にすべき)
2. [Optional Positional]: デフォルト値を持つオプショナル引数が位置引数に残っていないか (* 以降にすべき)
3. [Too Many Positional]: 位置引数が3個以上連続していないか (* 以降に配置するか @allow_positional で除外)

使用例:
    python tools/audit_function_args.py              # 詳細レポート表示
    python tools/audit_function_args.py --check      # 違反があれば終了コード1（CI向け）
    python tools/audit_function_args.py --json       # JSON形式で結果出力
    python tools/audit_function_args.py --verbose    # スキップされた関数も表示
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_DIR = REPO_ROOT / "python" / "taremin_cloth"

# Blender APIオペレーター・UI規約のオーバーライドメソッド（位置引数が必須）
BLENDER_METHOD_WHITELIST: Set[str] = {
    "execute",
    "invoke",
    "modal",
    "draw",
    "draw_header",
    "poll",
    "cancel",
    "description",
    "check",
    "draw_item",
    "filter_items",
}

# Python組み込み特殊メソッド（位置引数が自然または必須）
SPECIAL_METHOD_WHITELIST: Set[str] = {
    "__init__",
    "__new__",
    "__call__",
    "__repr__",
    "__str__",
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__hash__",
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "__len__",
    "__iter__",
    "__next__",
    "__enter__",
    "__exit__",
    "__contains__",
}

# 除外マーカーコメント
IGNORE_COMMENT_DIRECTIVES = (
    "# audit: allow-positional",
    "# audit: ignore-args",
    "# noqa: args",
    "# noqa: FBT",
)


def _is_bool_annotation(node: Optional[ast.AST]) -> bool:
    """型アノテーションが bool または Optional[bool] であるかを判定する"""
    if node is None:
        return False
    if isinstance(node, ast.Name) and node.id == "bool":
        return True
    if isinstance(node, ast.Subscript):
        # Optional[bool] または Union[bool, None] の検出
        if isinstance(node.value, ast.Name) and node.value.id in ("Optional", "Union"):
            if isinstance(node.slice, ast.Name) and node.slice.id == "bool":
                return True
            if isinstance(node.slice, ast.Tuple):
                for elt in node.slice.elts:
                    if isinstance(elt, ast.Name) and elt.id == "bool":
                        return True
    return False


def _is_bool_default(node: ast.AST) -> bool:
    """デフォルト値が True または False リテラルであるかを判定する"""
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return True
    return False


class FunctionArgAuditor(ast.NodeVisitor):
    """単一ファイルのASTを走査し、関数の引数定義を監査するビジター"""

    def __init__(self, filepath: Path, source_lines: List[str]):
        self.filepath = filepath
        self.source_lines = source_lines
        self.violations: List[Dict[str, Any]] = []
        self.total_functions = 0
        self.skipped_functions = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._audit_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._audit_function(node)
        self.generic_visit(node)

    def _audit_function(self, node: ast.AST) -> None:
        self.total_functions += 1
        fn_name = getattr(node, "name", "<anonymous>")
        lineno = getattr(node, "lineno", 1)

        # 1. Blender規定メソッド・特殊メソッドの除外
        if fn_name in BLENDER_METHOD_WHITELIST or fn_name in SPECIAL_METHOD_WHITELIST:
            self.skipped_functions += 1
            return

        # 2. @allow_positional デコレータの除外判定
        for decorator in getattr(node, "decorator_list", []):
            dec_name = None
            if isinstance(decorator, ast.Name):
                dec_name = decorator.id
            elif isinstance(decorator, ast.Attribute):
                dec_name = decorator.attr
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    dec_name = decorator.func.id
                elif isinstance(decorator.func, ast.Attribute):
                    dec_name = decorator.func.attr
            if dec_name in ("allow_positional", "classmethod", "staticmethod"):
                # classmethod/staticmethod の場合はデコレータだけでは除外せず引数を評価
                if dec_name == "allow_positional":
                    self.skipped_functions += 1
                    return

        # 3. インラインコメント除外判定 (# audit: allow-positional)
        if 0 < lineno <= len(self.source_lines):
            line_str = self.source_lines[lineno - 1]
            if any(directive in line_str for directive in IGNORE_COMMENT_DIRECTIVES):
                self.skipped_functions += 1
                return

        # 4. Docstring内ディレクティブ判定
        docstring = ast.get_docstring(node)
        if docstring and "audit: allow-positional" in docstring:
            self.skipped_functions += 1
            return

        args_obj = getattr(node, "args", None)
        if not args_obj:
            return

        # レシーバー引数 (self, cls) を除外した位置引数リスト
        pos_args = [a for a in args_obj.args if a.arg not in ("self", "cls")]

        # 5. Blenderプロパティ更新関数のパターン判定: (self, context) または (scene, context) の2引数のみ
        all_raw_args = [a.arg for a in args_obj.args]
        if all_raw_args in (["self", "context"], ["scene", "context"]):
            self.skipped_functions += 1
            return

        # 位置引数に対応するデフォルト値のマッピング
        # node.args.defaults は位置引数の末尾 n_defaults 個に対応
        n_defaults = len(args_obj.defaults)
        defaults_map: Dict[str, ast.AST] = {}
        if n_defaults > 0:
            defaulted_args = args_obj.args[-n_defaults:]
            for arg_node, def_node in zip(defaulted_args, args_obj.defaults):
                defaults_map[arg_node.arg] = def_node

        # --- ルール検査 ---

        # ルール 1: ブールトラップ（bool引数が位置引数として定義されている）
        for arg in pos_args:
            is_bool = _is_bool_annotation(arg.annotation)
            def_node = defaults_map.get(arg.arg)
            if not is_bool and def_node is not None:
                is_bool = _is_bool_default(def_node)

            if is_bool:
                self.violations.append({
                    "rule": "boolean_trap",
                    "file": str(self.filepath),
                    "line": arg.lineno,
                    "function": fn_name,
                    "arg": arg.arg,
                    "message": f"ブール型引数 '{arg.arg}' が位置引数になっています。'*' 以降のキーワード専用引数にしてください。",
                })

        # ルール 2: オプショナル引数の位置指定
        # 全体で2引数以上ある関数で、デフォルト値を持つ引数が位置引数に残っている場合
        # （※関数全体で引数が1個だけ、かつそれが context=None などの主要対象デフォルトは許容）
        if len(pos_args) > 1:
            for arg in pos_args:
                if arg.arg in defaults_map:
                    # ブールトラップで既に検出済みの場合は重複を避ける
                    if not (_is_bool_annotation(arg.annotation) or _is_bool_default(defaults_map[arg.arg])):
                        self.violations.append({
                            "rule": "optional_positional",
                            "file": str(self.filepath),
                            "line": arg.lineno,
                            "function": fn_name,
                            "arg": arg.arg,
                            "message": f"デフォルト値を持つ引数 '{arg.arg}' が位置引数になっています。'*' 以降のキーワード専用引数にしてください。",
                        })

        # ルール 3: 多すぎる位置引数 (3個以上)
        if len(pos_args) >= 3:
            self.violations.append({
                "rule": "too_many_positional",
                "file": str(self.filepath),
                "line": lineno,
                "function": fn_name,
                "arg": ", ".join(a.arg for a in pos_args),
                "message": f"位置引数が {len(pos_args)} 個あります（上限 2）。主要対象以外の引数を '*' 以降に配置するか、@allow_positional を付与してください。",
            })


def audit_path(target_path: Path) -> Tuple[List[Dict[str, Any]], int, int]:
    """指定パス（ディレクトリまたは単一ファイル）を走査して違反を抽出する"""
    all_violations: List[Dict[str, Any]] = []
    total_funcs = 0
    total_skipped = 0

    if target_path.is_file():
        py_files = [target_path]
    elif target_path.is_dir():
        py_files = sorted(target_path.rglob("*.py"))
    else:
        return [], 0, 0

    for py_file in py_files:
        # __pycache__ やビルド成果物はスキップ
        if "__pycache__" in py_file.parts or ".venv" in py_file.parts:
            continue
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
                source_lines = content.splitlines()
                tree = ast.parse(content, filename=str(py_file))
        except Exception as e:
            print(f"[WARN] パース失敗: {py_file}: {e}", file=sys.stderr)
            continue

        auditor = FunctionArgAuditor(py_file, source_lines)
        auditor.visit(tree)
        all_violations.extend(auditor.violations)
        total_funcs += auditor.total_functions
        total_skipped += auditor.skipped_functions

    return all_violations, total_funcs, total_skipped


def format_report(violations: List[Dict[str, Any]], total_funcs: int, total_skipped: int, repo_root: Path) -> str:
    """監査結果の人間向けテキストレポートを生成する"""
    lines = []
    lines.append("=" * 70)
    lines.append("Taremin Cloth 関数引数安全規約 監査レポート")
    lines.append("=" * 70)

    if not violations:
        lines.append("[PASS] 違反は検出されませんでした（すべての関数がキーワード安全規約を満たしています）。")
    else:
        # ファイル別にグループ化
        by_file: Dict[str, List[Dict[str, Any]]] = {}
        for v in violations:
            f_rel = str(Path(v["file"]).relative_to(repo_root)) if Path(v["file"]).is_relative_to(repo_root) else v["file"]
            by_file.setdefault(f_rel, []).append(v)

        for file_path, file_violations in by_file.items():
            lines.append(f"\n[FILE] {file_path}")
            for v in file_violations:
                rule_tag = f"[{v['rule']}]"
                lines.append(f"  Line {v['line']:>4} | {v['function']}(): {rule_tag} {v['message']}")

    lines.append("\n" + "-" * 70)
    lines.append(f"総検査関数数: {total_funcs} 件 | スキップ/免除: {total_skipped} 件 | 検出違反: {len(violations)} 件")
    lines.append("-" * 70)
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Taremin Cloth 関数引数安全規約 監査CLIツール")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_TARGET_DIR), help="走査対象のディレクトリまたはファイルパス")
    parser.add_argument("--check", action="store_true", help="違反が1件以上ある場合に終了コード1で終了（CI向け）")
    parser.add_argument("--json", action="store_true", help="結果をJSON形式で出力")
    parser.add_argument("--verbose", action="store_true", help="詳細情報を出力")
    parser.add_argument("--rule", choices=["boolean_trap", "optional_positional", "too_many_positional"], help="指定したルールのみをフィルタ表示")

    args = parser.parse_args()
    target_path = Path(args.path).resolve()

    violations, total_funcs, total_skipped = audit_path(target_path)
    if args.rule:
        violations = [v for v in violations if v["rule"] == args.rule]

    if args.json:
        result = {
            "target": str(target_path),
            "total_functions": total_funcs,
            "skipped_functions": total_skipped,
            "violations_count": len(violations),
            "violations": violations,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        report = format_report(violations, total_funcs, total_skipped, REPO_ROOT)
        print(report)

    if args.check and violations:
        sys.exit(1)


if __name__ == "__main__":
    main()
