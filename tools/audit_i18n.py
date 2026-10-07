"""
Taremin Cloth 国際化（i18n）監査・完全性検査CLIツール

Blender非依存でPython AST（抽象構文木）を解析し、
アドオンコード内の全プロパティ（name, description）、Enum選択肢、
オペレーター・パネルのラベル/説明文、UIテキストを抽出して翻訳辞書と照合します。

使用例:
    python tools/audit_i18n.py              # 詳細レポート表示
    python tools/audit_i18n.py --check      # 未翻訳があれば終了コード1（CI向け）
    python tools/audit_i18n.py --json       # JSON形式で結果出力
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ADDON_DIR = REPO_ROOT / "python" / "taremin_cloth"
TRANSLATIONS_DIR = ADDON_DIR / "translations"


def extract_terms_from_ast(addon_dir: Path = ADDON_DIR) -> Dict[str, List[Dict[str, Any]]]:
    """AST走査によってアドオンコード内の全UI用語・説明文および生のbpy.props使用を抽出する"""
    results = {
        "descriptions": [],
        "names": [],
        "enum_items": [],
        "bl_labels": [],
        "bl_descriptions": [],
        "ui_texts": [],
        "raw_props": [],
    }

    if not addon_dir.is_dir():
        return results

    for py_file in sorted(addon_dir.rglob("*.py")):
        if py_file.name == "i18n.py":
            continue
        rel = py_file.relative_to(addon_dir)

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_file))
        except Exception as e:
            print(f"[WARN] パース失敗: {rel}: {e}", file=sys.stderr)
            continue

        # 生の bpy.props 利用（translation_context未注入）の検知
        is_prop_def_file = py_file.name in ("properties.py", "preferences.py")

        for node in ast.walk(tree):
            if not is_prop_def_file:
                # from bpy.props import X の検出
                if isinstance(node, ast.ImportFrom) and node.module == "bpy.props":
                    results["raw_props"].append({
                        "file": str(rel),
                        "line": node.lineno,
                        "detail": f"from bpy.props import {', '.join(a.name for a in node.names)}",
                    })
                # bpy.props.XProperty の検出
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
                    if getattr(node.value.value, "id", None) == "bpy" and node.value.attr == "props":
                        results["raw_props"].append({
                            "file": str(rel),
                            "line": node.lineno,
                            "detail": f"bpy.props.{node.attr}",
                        })

            # 1. Property 定義の呼び出し (FloatProperty, BoolProperty, etc.)
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if "Property" in func_name and func_name not in ("PointerProperty", "CollectionProperty"):
                    kw_dict = {}
                    for kw in node.keywords:
                        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            kw_dict[kw.arg] = kw.value.value.strip()

                    if "description" in kw_dict and kw_dict["description"]:
                        results["descriptions"].append({
                            "file": str(rel),
                            "line": node.lineno,
                            "text": kw_dict["description"],
                        })

                    if "name" in kw_dict and kw_dict["name"]:
                        results["names"].append({
                            "file": str(rel),
                            "line": node.lineno,
                            "text": kw_dict["name"],
                        })

                    # Enum items
                    for kw in node.keywords:
                        if kw.arg == "items":
                            # リストまたはタプルのリテラル走査
                            if isinstance(kw.value, (ast.List, ast.Tuple)):
                                for elt in kw.value.elts:
                                    if isinstance(elt, (ast.List, ast.Tuple)) and len(elt.elts) >= 3:
                                        # (identifier, name, description)
                                        name_node = elt.elts[1]
                                        desc_node = elt.elts[2]
                                        if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str):
                                            n_val = name_node.value.strip()
                                            if n_val:
                                                results["enum_items"].append({
                                                    "file": str(rel),
                                                    "line": elt.lineno,
                                                    "text": n_val,
                                                    "type": "enum_name",
                                                })
                                        if isinstance(desc_node, ast.Constant) and isinstance(desc_node.value, str):
                                            d_val = desc_node.value.strip()
                                            if d_val:
                                                results["enum_items"].append({
                                                    "file": str(rel),
                                                    "line": elt.lineno,
                                                    "text": d_val,
                                                    "type": "enum_description",
                                                })

                # UI呼び出し (layout.label, layout.prop text, i18n.trans)
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("trans", "pgettext", "pgettext_iface"):
                        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                            t_val = node.args[0].value.strip()
                            if t_val:
                                results["ui_texts"].append({
                                    "file": str(rel),
                                    "line": node.lineno,
                                    "text": t_val,
                                    "source": "trans",
                                })
                    elif node.func.attr in ("label", "prop", "operator", "menu"):
                        for kw in node.keywords:
                            if kw.arg == "text" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                t_val = kw.value.value.strip()
                                if t_val:
                                    results["ui_texts"].append({
                                        "file": str(rel),
                                        "line": node.lineno,
                                        "text": t_val,
                                        "source": f"layout.{node.func.attr}",
                                    })

            # 2. クラス定義 (Panel, Operator, Menu) の属性 (bl_label, bl_description)
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                if target.id == "bl_label" and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                                    val = item.value.value.strip()
                                    if val:
                                        results["bl_labels"].append({
                                            "file": str(rel),
                                            "line": item.lineno,
                                            "class": node.name,
                                            "text": val,
                                        })
                                elif target.id == "bl_description" and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str):
                                    val = item.value.value.strip()
                                    if val:
                                        results["bl_descriptions"].append({
                                            "file": str(rel),
                                            "line": item.lineno,
                                            "class": node.name,
                                            "text": val,
                                        })

    return results


def run_audit(translations_dir: Path = TRANSLATIONS_DIR) -> Dict[str, Any]:
    """辞書ファイルとAST抽出文字列を照合して完全性レポートを生成"""
    extracted = extract_terms_from_ast()

    ja_path = translations_dir / "ja_JP.json"
    en_path = translations_dir / "en_US.json"

    ja_dict = {}
    en_dict = {}
    if ja_path.is_file():
        with open(ja_path, "r", encoding="utf-8") as f:
            ja_dict = json.load(f)
    if en_path.is_file():
        with open(en_path, "r", encoding="utf-8") as f:
            en_dict = json.load(f)

    def find_missing(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        missing = []
        seen = set()
        for itm in items:
            t = itm["text"]
            if t not in ja_dict and t not in seen:
                seen.add(t)
                missing.append(itm)
        return missing

    missing_desc = find_missing(extracted["descriptions"])
    missing_name = find_missing(extracted["names"])
    missing_enum = find_missing(extracted["enum_items"])
    missing_bl_label = find_missing(extracted["bl_labels"])
    missing_bl_desc = find_missing(extracted["bl_descriptions"])
    missing_ui = find_missing(extracted["ui_texts"])

    # 英語マスターと日本語のキー差分
    en_keys = set(en_dict.keys())
    ja_keys = set(ja_dict.keys())
    missing_in_ja = sorted(list(en_keys - ja_keys))
    extra_in_ja = sorted(list(ja_keys - en_keys))

    # コードで使われていないキー（デッドキー）
    all_code_texts = set()
    for cat_name, cat in extracted.items():
        if cat_name == "raw_props":
            continue
        for itm in cat:
            all_code_texts.add(itm["text"])

    dead_keys = sorted([k for k in ja_keys if k not in all_code_texts])

    # descriptions.json との整合性検査
    desc_file = translations_dir / "descriptions.json"
    ast_descriptions = set(
        [x["text"] for x in extracted["descriptions"]]
        + [x["text"] for x in extracted["bl_descriptions"]]
        + [x["text"] for x in extracted["enum_items"] if x["type"] == "enum_description"]
    )
    file_descriptions = set()
    if desc_file.is_file():
        try:
            with open(desc_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    file_descriptions = set(data)
        except Exception:
            pass

    missing_in_desc_file = sorted(list(ast_descriptions - file_descriptions))
    extra_in_desc_file = sorted(list(file_descriptions - ast_descriptions))

    return {
        "counts": {
            "en_entries": len(en_dict),
            "ja_entries": len(ja_dict),
            "desc_file_entries": len(file_descriptions),
            "missing_descriptions": len(missing_desc),
            "missing_names": len(missing_name),
            "missing_enum_items": len(missing_enum),
            "missing_bl_labels": len(missing_bl_label),
            "missing_bl_descriptions": len(missing_bl_desc),
            "missing_ui_texts": len(missing_ui),
            "missing_in_ja": len(missing_in_ja),
            "extra_in_ja": len(extra_in_ja),
            "dead_keys": len(dead_keys),
            "raw_props": len(extracted["raw_props"]),
            "missing_in_desc_file": len(missing_in_desc_file),
            "extra_in_desc_file": len(extra_in_desc_file),
        },
        "missing": {
            "descriptions": missing_desc,
            "names": missing_name,
            "enum_items": missing_enum,
            "bl_labels": missing_bl_label,
            "bl_descriptions": missing_bl_desc,
            "ui_texts": missing_ui,
        },
        "raw_props": extracted["raw_props"],
        "key_diff": {
            "missing_in_ja": missing_in_ja,
            "extra_in_ja": extra_in_ja,
        },
        "dead_keys": dead_keys,
        "descriptions_sync": {
            "missing_in_desc_file": missing_in_desc_file,
            "extra_in_desc_file": extra_in_desc_file,
            "ast_descriptions": sorted(list(ast_descriptions)),
        },
    }


def update_descriptions_file(translations_dir: Path = TRANSLATIONS_DIR) -> int:
    """AST走査で抽出した全descriptionを translations/descriptions.json に自動更新"""
    extracted = extract_terms_from_ast()
    ast_descriptions = sorted(list(set(
        [x["text"] for x in extracted["descriptions"]]
        + [x["text"] for x in extracted["bl_descriptions"]]
        + [x["text"] for x in extracted["enum_items"] if x["type"] == "enum_description"]
    )))
    desc_file = translations_dir / "descriptions.json"
    with open(desc_file, "w", encoding="utf-8") as f:
        json.dump(ast_descriptions, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return len(ast_descriptions)


def main():
    parser = argparse.ArgumentParser(description="Taremin Cloth i18n 翻訳完全性監査ツール")
    parser.add_argument("--check", action="store_true", help="未翻訳項目やコンテキスト欠落が存在する場合に終了コード1を返す")
    parser.add_argument("--json", action="store_true", help="結果をJSON形式で標準出力に出力")
    parser.add_argument("--dead-keys", action="store_true", help="未使用のデッドキー一覧を表示")
    parser.add_argument("--update-descriptions", action="store_true", help="AST抽出結果から descriptions.json を自動更新・同期する")
    args = parser.parse_args()

    if args.update_descriptions:
        count = update_descriptions_file()
        print(f"[OK] {count} 件のツールチップ説明文を translations/descriptions.json に同期しました。")
        sys.exit(0)

    report = run_audit()
    counts = report["counts"]

    total_missing = (
        counts["missing_descriptions"]
        + counts["missing_names"]
        + counts["missing_enum_items"]
        + counts["missing_bl_labels"]
        + counts["missing_bl_descriptions"]
        + counts["missing_in_ja"]
        + counts["extra_in_ja"]
        + counts["raw_props"]
        + counts["missing_in_desc_file"]
        + counts["extra_in_desc_file"]
    )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        sys.exit(1 if (args.check and total_missing > 0) else 0)

    print("==================================================")
    print("      Taremin Cloth i18n 翻訳完全性監査レポート      ")
    print("==================================================")
    print(f"辞書エントリ数: en_US: {counts['en_entries']}, ja_JP: {counts['ja_entries']}")
    print(f"ツールチップ説明文 (descriptions.json): {counts['desc_file_entries']} 件")
    print("--------------------------------------------------")
    print(f"・未登録 プロパティ Description:  {counts['missing_descriptions']} 件")
    print(f"・未登録 プロパティ Name:         {counts['missing_names']} 件")
    print(f"・未登録 Enum 項目:              {counts['missing_enum_items']} 件")
    print(f"・未登録 パネル/オペレーター Label: {counts['missing_bl_labels']} 件")
    print(f"・未登録 オペレーター Description:  {counts['missing_bl_descriptions']} 件")
    print(f"・未登録 UIテキスト (trans等):    {counts['missing_ui_texts']} 件")
    print(f"・コンテキスト未注入 生 bpy.props: {counts['raw_props']} 件")
    print(f"・en_US にあるが ja_JP に未登録:  {counts['missing_in_ja']} 件")
    print(f"・ja_JP にあるが en_US に未登録:  {counts['extra_in_ja']} 件")
    print(f"・descriptions.json 未同期 (不足): {counts['missing_in_desc_file']} 件")
    print(f"・descriptions.json 未同期 (過剰): {counts['extra_in_desc_file']} 件")
    print("==================================================")

    if report["raw_props"]:
        print("\n[ALERT: コンテキスト欠落の恐れがある生の bpy.props 直接使用]")
        for itm in report["raw_props"]:
            print(f"  {itm['file']}:{itm['line']} -> {itm['detail']} (properties.py 経由のラップを使用してください)")

    if report["missing"]["descriptions"]:
        print("\n[未登録 プロパティ Description 一覧]")
        for itm in report["missing"]["descriptions"]:
            print(f"  {itm['file']}:{itm['line']} -> \"{itm['text']}\"")

    if report["missing"]["names"]:
        print("\n[未登録 プロパティ Name 一覧]")
        for itm in report["missing"]["names"]:
            print(f"  {itm['file']}:{itm['line']} -> \"{itm['text']}\"")

    if report["missing"]["enum_items"]:
        print("\n[未登録 Enum 項目 一覧]")
        for itm in report["missing"]["enum_items"]:
            print(f"  {itm['file']}:{itm['line']} ({itm['type']}) -> \"{itm['text']}\"")

    if report["descriptions_sync"]["missing_in_desc_file"]:
        print("\n[ALERT: descriptions.json に未同期の新しい説明文があります (python tools/audit_i18n.py --update-descriptions で更新してください)]")
        for itm in report["descriptions_sync"]["missing_in_desc_file"]:
            print(f"  + \"{itm}\"")

    if report["descriptions_sync"]["extra_in_desc_file"]:
        print("\n[ALERT: descriptions.json にコード上に存在しない不要な説明文があります (python tools/audit_i18n.py --update-descriptions で更新してください)]")
        for itm in report["descriptions_sync"]["extra_in_desc_file"]:
            print(f"  - \"{itm}\"")

    if args.dead_keys and report["dead_keys"]:
        print(f"\n[コード内で直接見当たらないデッドキー候補: {len(report['dead_keys'])} 件]")
        for k in report["dead_keys"][:30]:
            print(f"  \"{k}\"")
        if len(report["dead_keys"]) > 30:
            print(f"  ...他 {len(report['dead_keys']) - 30} 件")

    if total_missing == 0:
        print("\n[OK] 全てのプロパティおよびUI用語が翻訳辞書に完全登録され、コンテキスト・descriptions.jsonも正常です。")
        sys.exit(0)
    else:
        print(f"\n[ALERT] 合計 {total_missing} 件の翻訳関連の不整合・コンテキスト欠落が検出されました。")
        if args.check:
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()

