#!/usr/bin/env python3
"""
Taremin Cloth 国際化（i18n）語彙・辞書カバレッジ検査ツール

アドオン内の全パネル、メニュー、オペレーター、プロパティ、プリセット等から
英語のUI表示テキスト（bl_label, bl_description, RNAプロパティ名等）を網羅的に走査・抽出し、
翻訳辞書（translations/ja_JP.json）との突合結果（未登録語彙、未使用語彙）をレポートします。

使用例:
  python tools/check_i18n.py
"""

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.blender_manager import resolve_blender


BLENDER_EXTRACT_SCRIPT = r"""
import sys
import inspect
import json
from pathlib import Path
import bpy

if "--" in sys.argv:
    addon_path = sys.argv[sys.argv.index("--") + 1]
    if addon_path not in sys.path:
        sys.path.insert(0, addon_path)

try:
    import taremin_cloth
    taremin_cloth.register()
except Exception as e:
    print(f"[!] register note: {e}", file=sys.stderr)

from taremin_cloth import i18n, panels, properties, preferences, presets, ops

all_terms = {}

def record(term, source):
    if term and isinstance(term, str):
        t = term.strip()
        if t:
            if t not in all_terms:
                all_terms[t] = []
            all_terms[t].append(source)

# 1. 全登録クラス (Panel, Menu, Operator) の bl_label, bl_description
for mod in [panels, presets, preferences, ops.basic, ops.gui, ops.interactive, ops.pose, ops.tools]:
    for name, cls in inspect.getmembers(mod, inspect.isclass):
        if issubclass(cls, (bpy.types.Panel, bpy.types.Operator, bpy.types.Menu)):
            lbl = getattr(cls, "bl_label", None)
            if lbl:
                record(lbl, f"{cls.__name__}.bl_label")
            desc = getattr(cls, "bl_description", None)
            if desc:
                record(desc, f"{cls.__name__}.bl_description")

# 2. 全オペレータープロパティ
if hasattr(bpy.ops, "taremin_cloth"):
    for op_name in dir(bpy.ops.taremin_cloth):
        if op_name.startswith("_"):
            continue
        op_func = getattr(bpy.ops.taremin_cloth, op_name)
        if hasattr(op_func, "get_rna_type"):
            op_rna = op_func.get_rna_type()
            for prop in op_rna.properties:
                if prop.identifier == "rna_type":
                    continue
                record(prop.name, f"Operator taremin_cloth.{op_name}.{prop.identifier}.name")
                record(prop.description, f"Operator taremin_cloth.{op_name}.{prop.identifier}.description")
                if prop.type == 'ENUM':
                    for itm in prop.enum_items:
                        record(itm.name, f"Operator taremin_cloth.{op_name}.{prop.identifier}.{itm.identifier}.name")
                        record(itm.description, f"Operator taremin_cloth.{op_name}.{prop.identifier}.{itm.identifier}.description")

# 3. 全 PropertyGroup
for p_cls in [
    properties.TareminClothObjectSettings,
    properties.TareminClothColliderSettings,
    properties.TareminClothColliderAnimSettings,
    properties.TareminClothElasticGroup,
    preferences.TareminClothPreferences,
]:
    if hasattr(p_cls, "bl_rna"):
        for prop in p_cls.bl_rna.properties:
            if prop.identifier == "rna_type":
                continue
            record(prop.name, f"{p_cls.__name__}.{prop.identifier}.name")
            record(prop.description, f"{p_cls.__name__}.{prop.identifier}.description")
            if prop.type == 'ENUM':
                for itm in prop.enum_items:
                    record(itm.name, f"{p_cls.__name__}.{prop.identifier}.{itm.identifier}.name")
                    record(itm.description, f"{p_cls.__name__}.{prop.identifier}.{itm.identifier}.description")

# 4. Scene に追加されたプロパティ
for prop_id in [
    "taremin_cloth_fast_playback",
    "taremin_cloth_ui_mode",
    "taremin_cloth_active_config_preset",
    "taremin_cloth_config_presets_json",
]:
    prop = bpy.types.Scene.bl_rna.properties.get(prop_id)
    if prop:
        record(prop.name, f"Scene.{prop_id}.name")
        record(prop.description, f"Scene.{prop_id}.description")
        if prop.type == 'ENUM':
            for itm in prop.enum_items:
                record(itm.name, f"Scene.{prop_id}.{itm.identifier}.name")
                record(itm.description, f"Scene.{prop_id}.{itm.identifier}.description")

# 5. preferences 動的Enum
try:
    pref = bpy.context.preferences.addons.get("taremin_cloth")
    if pref and hasattr(pref, "preferences"):
        items = preferences._get_gpu_device_items(pref.preferences, bpy.context)
        for itm in items:
            record(itm[1], "DynamicEnum _get_gpu_device_items.name")
            record(itm[2], "DynamicEnum _get_gpu_device_items.description")
except Exception:
    pass

# 6. panels.py 内のフォールバックタイトル・定数
record("Config Preset", "panels.py fallback preset title")
record("Material Preset", "panels.py fallback fabric preset title")
record("Quality Preset", "panels.py fallback sim preset title")
record("Bone SDF (Character Body)", "panels.py type_names")
record("Mesh SDF (Mannequin/Rigid)", "panels.py type_names")
record("Plane (Floor/Ground)", "panels.py type_names")
record("Sphere", "panels.py type_names")
record("Capsule", "panels.py type_names")
record("Mesh (Simple)", "panels.py type_names")
record("Taremin Cloth", "panels.py bl_category")

# JSON 出力
print("===JSON_START===")
print(json.dumps(all_terms, ensure_ascii=False))
print("===JSON_END===")
"""


def extract_terms_via_blender() -> dict:
    """ヘッドレスBlenderを起動して用語を抽出する"""
    blender_exe = resolve_blender()
    cmd = [
        str(blender_exe),
        "--background",
        "--factory-startup",
        "--python-expr",
        BLENDER_EXTRACT_SCRIPT,
        "--",
        str(PYTHON_DIR),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        print(f"[!] Blender実行エラー: {proc.stderr}", file=sys.stderr)
        return {}

    out = proc.stdout
    if "===JSON_START===" in out and "===JSON_END===" in out:
        json_str = out.split("===JSON_START===")[1].split("===JSON_END===")[0].strip()
        return json.loads(json_str)
    return {}


def extract_terms_from_panels_ast() -> dict:
    """AST解析で panels.py の layout.label / layout.prop text 引数を抽出する"""
    terms = {}
    panels_path = PYTHON_DIR / "taremin_cloth" / "panels.py"
    if not panels_path.exists():
        return terms

    with open(panels_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(panels_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "text" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    val = kw.value.value.strip()
                    if val:
                        terms.setdefault(val, []).append("panels.py text arg")
            if isinstance(node.func, ast.Attribute) and node.func.attr == "label":
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    val = node.args[0].value.strip()
                    if val:
                        terms.setdefault(val, []).append("panels.py label arg")
    return terms


def main() -> int:
    print("Taremin Cloth 国際化辞書カバレッジ検査を実行中...")

    # 1. 抽出
    terms = extract_terms_via_blender()
    ast_terms = extract_terms_from_panels_ast()
    for k, v in ast_terms.items():
        terms.setdefault(k, []).extend(v)

    if not terms:
        print("[!] 用語の抽出に失敗しました。", file=sys.stderr)
        return 1

    # 2. 辞書読み込み
    ja_path = PYTHON_DIR / "taremin_cloth" / "translations" / "ja_JP.json"
    with open(ja_path, "r", encoding="utf-8") as f:
        ja_dict = json.load(f)

    dict_keys = set(ja_dict.keys())
    used_keys = set(terms.keys())

    in_dict_and_used = dict_keys.intersection(used_keys)
    in_dict_but_unused = dict_keys - used_keys
    used_but_not_in_dict = used_keys - dict_keys

    coverage = len(in_dict_and_used) / len(used_keys) * 100.0 if used_keys else 0.0

    print("=" * 60)
    print("【国際化カバレッジ結果】")
    print(f"  Blender UIから検出された全用語数: {len(used_keys)}")
    print(f"  辞書に登録済みの有効用語数:       {len(in_dict_and_used)} ({coverage:.1f}%)")
    print(f"  辞書にあるが未検出の用語数:       {len(in_dict_but_unused)}")
    print(f"  UIで使用中だが辞書未登録の用語数: {len(used_but_not_in_dict)}")
    print("=" * 60)

    if used_but_not_in_dict:
        print(f"\n[未登録用語 ({len(used_but_not_in_dict)} 件)]")
        for k in sorted(used_but_not_in_dict):
            print(f"  - {k!r}  (参照元: {terms[k][0]})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
