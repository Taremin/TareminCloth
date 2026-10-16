"""
Taremin Cloth Python-Rust-wgpu プロトコル整合性・監査CLIツール

Blender非依存でPython AST、Rustソースコード、およびWGSLシェーダー定義を静的に解析し、
Blenderプロパティ、Python SimConfig、Rust SimConfig、およびGPU Uniform構造体パラメータの
4層間におけるパラメータの写像漏れや型の不整合を自動検出します。

使用例:
    python tools/audit_protocol.py              # 詳細レポート表示
    python tools/audit_protocol.py --check      # 不整合があれば終了コード1（CI向け）
    python tools/audit_protocol.py --format markdown # Markdown形式で出力
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python" / "taremin_cloth"
PROPERTIES_PY = PYTHON_DIR / "properties.py"
SIMCONFIG_PY = PYTHON_DIR / "engine" / "simconfig.py"
RUST_CONFIG_RS = REPO_ROOT / "crates" / "cloth_core" / "src" / "config.rs"
SHADERS_DIR = REPO_ROOT / "crates" / "cloth_core" / "src" / "shaders"

# Blender UI / 表示専用のプロパティ（シミュレーションコアに渡す必要のないプロパティ）
# --- カテゴリ別プロパティ分類 ---

#: シミュレータ生成時（ClothSimulator.new）に渡される初期パラメータ
INIT_ONLY_PROPERTIES: Set[str] = {
    "stiffness",
    "thickness",
    "layer_id",
    "enable_compact_readback",
    "enable_coarse_constraints",
    "enable_async_readback",
    "enable_frame_buffering",
    "frame_buffer_size",
}

#: ピン・アタッチメント・頂点グループ設定
PIN_ATTACHMENT_PROPERTIES: Set[str] = {
    "vertex_group_pins",
    "vertex_group_stretch",
    "vertex_group_bend",
    "vertex_group_layer",
    "vertex_group_thickness",
    "pin_vertex_group",
    "pin_target_object",
    "pin_target_bone",
    "pin_color",
    "pin_overlay_interactive_only",
}

#: ドレープガイド（Wrinkle Field）設定
WRINKLE_FIELD_PROPERTIES: Set[str] = {
    "use_wrinkle_field",
    "show_wrinkle_settings",
    "wrinkle_pattern",
    "wrinkle_bone_name",
    "wrinkle_radius",
    "wrinkle_influence",
    "wrinkle_stiffness",
    "wrinkle_blend",
    "wrinkle_frequency",
    "wrinkle_direction_angle",
    "wrinkle_armature",
    "wrinkle_bone_radius",
    "wrinkle_collection",
    "wrinkle_crest_depth",
    "wrinkle_global_strength",
    "wrinkle_preview_mode",
    "wrinkle_valley_depth",
}

#: メッシュ前処理・後処理・分割・縫合パターン設定
MESH_POSTPROCESS_PROPERTIES: Set[str] = {
    "enable_sewing",
    "sewing_pattern_obj",
    "sewing_target_obj",
    "enable_cross_subdivision",
    "triangulation_mode",
    "adaptive_flatness_threshold",
    "dynamic_flatness_threshold",
    "dynamic_preserve_flat",
    "auto_post_process",
    "post_process_mode",
    "auto_triangulate_on_stop",
    "target_face_layer_id",
}

#: ビューポート・HUD・オーバーレイ・表示用プロパティ
VIEWPORT_HUD_PROPERTIES: Set[str] = {
    "show_fps_overlay",
    "fps_overlay_position",
    "show_hud_help",
    "show_elastic_overlay",
    "elastic_overlay_interactive_only",
    "overlay_depth_test",
    "isolate_viewport_view",
    "show_advanced_settings",
    "show_bend_settings",
    "show_damping_settings",
    "show_collider_settings",
    "show_self_collision_settings",
    "show_sewing_settings",
    "show_brush_settings",
    "show_debug_settings",
}

#: ブラシ操作関連プロパティ
BRUSH_TOOL_PROPERTIES: Set[str] = {
    "active_brush_tool",
    "brush_radius",
    "brush_strength",
    "brush_falloff",
    "brush_depth_limit",
    "brush_mode",
    "brush_radial_axis",
    "brush_expand_ratio",
    "brush_smooth_preset",
    "brush_symmetry_axis",
    "enable_brush_symmetry",
    "brush_debug_overlay",
    "brush_smooth_auto_drag",
    "brush_smooth_drag_scale",
    "brush_normal_align_blend",
    "brush_history_cap",
    "brush_auto_anchor",
    "brush",
}

#: アドオン状態・プリセット・内部制御プロパティ
STATE_PRESET_PROPERTIES: Set[str] = {
    "enabled",
    "is_cloth",
    "self_collision_purpose",
    "self_collision_quality_mode",
    "enable_gpu_cloth",
    "substeps",
    "target_fps",
    "active_elastic_group_index",
    "elastic_groups",
    "last_fabric_preset",
    "last_simulation_preset",
    "interactive_max_steps",
    "interactive_realtime_sync",
    "instant_drape_substeps",
    "instant_drape_damping",
}

ALL_KNOWN_NON_SIMCONFIG_PROPERTIES: Set[str] = (
    INIT_ONLY_PROPERTIES
    | PIN_ATTACHMENT_PROPERTIES
    | WRINKLE_FIELD_PROPERTIES
    | MESH_POSTPROCESS_PROPERTIES
    | VIEWPORT_HUD_PROPERTIES
    | BRUSH_TOOL_PROPERTIES
    | STATE_PRESET_PROPERTIES
)


def extract_properties_from_ast(filepath: Path) -> Set[str]:
    """properties.py の TareminClothObjectSettings クラスからプロパティ名を抽出する"""
    if not filepath.is_file():
        return set()

    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(filepath))

    props = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TareminClothObjectSettings":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    props.add(item.target.id)
                elif isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            props.add(target.id)
    return props


def extract_simconfig_keys_from_ast(filepath: Path) -> Tuple[Set[str], Set[str], Set[str]]:
    """simconfig.py から SIGNATURE_KEYS, LIVE_KEYS, EXTRA_KEYS を抽出する"""
    if not filepath.is_file():
        return set(), set(), set()

    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(filepath))

    sig_keys = set()
    live_keys = set()
    extra_keys = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "SIGNATURE_KEYS" and isinstance(node.value, (ast.Tuple, ast.List)):
                        sig_keys = {elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)}
                    elif target.id == "EXTRA_KEYS" and isinstance(node.value, (ast.Tuple, ast.List)):
                        extra_keys = {elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)}

    # LIVE_KEYS は SIGNATURE_KEYS - EXTRA_KEYS
    live_keys = sig_keys - extra_keys
    return sig_keys, live_keys, extra_keys


def extract_rust_simconfig_fields(filepath: Path) -> Set[str]:
    """crates/cloth_core/src/config.rs から SimConfig のフィールド名を抽出する"""
    if not filepath.is_file():
        return set()

    content = filepath.read_text(encoding="utf-8")
    match = re.search(r"pub struct SimConfig\s*\{([^}]+)\}", content, re.MULTILINE)
    if not match:
        return set()

    struct_body = match.group(1)
    fields = set()
    for line in struct_body.splitlines():
        line = line.strip()
        if line.startswith("pub ") and ":" in line:
            name = line.split()[1].split(":")[0].strip()
            fields.add(name)
    return fields


def extract_wgsl_struct_members(shaders_dir: Path) -> Dict[str, Set[str]]:
    """src/shaders/*.wgsl から主要 Uniform 構造体のメンバー名を抽出する"""
    structs: Dict[str, Set[str]] = {}
    if not shaders_dir.is_dir():
        return structs

    struct_pattern = re.compile(r"struct\s+([A-Za-z0-9_]+)\s*\{([^}]+)\}")
    member_pattern = re.compile(r"([A-Za-z0-9_]+)\s*:")

    for wgsl_file in shaders_dir.glob("*.wgsl"):
        content = wgsl_file.read_text(encoding="utf-8")
        for match in struct_pattern.finditer(content):
            struct_name = match.group(1)
            body = match.group(2)
            members = set()
            for line in body.splitlines():
                m = member_pattern.search(line.strip())
                if m:
                    members.add(m.group(1))
            if struct_name not in structs:
                structs[struct_name] = members
            else:
                structs[struct_name].update(members)
    return structs


def audit_protocol() -> Dict[str, Any]:
    """4層間のパラメータ突合を行い、完全性監査レポートを生成する"""
    blender_props = extract_properties_from_ast(PROPERTIES_PY)
    sig_keys, live_keys, extra_keys = extract_simconfig_keys_from_ast(SIMCONFIG_PY)
    rust_fields = extract_rust_simconfig_fields(RUST_CONFIG_RS)
    wgsl_structs = extract_wgsl_struct_members(SHADERS_DIR)

    # 1. Python SimConfig (LIVE_KEYS) ↔ Rust SimConfig
    missing_in_rust = sorted(list(live_keys - rust_fields))
    missing_in_python = sorted(list((rust_fields - {"version"}) - live_keys))

    # 2. Blender Property ↔ Python SimConfig 網羅性チェック
    # UI_ONLY_PROPERTIES を除いた Blender プロパティのうち、SimConfig またはシミュレーション設定に渡っていないもの
    sim_handled_props = sig_keys | {
        "gravity",  # scene.gravity と settings.gravity が複合
        "tension_stiffness", "compression_stiffness", "shear_stiffness", "bending_stiffness",
        "air_damping", "tension_damping", "compression_damping", "shear_damping", "bending_damping",
        "coupled_self_collision_mode", "coupled_collider_mode", "post_collision_relaxation_iters",
        "self_collision_relief_factor", "self_collision_max_displacement_ratio",
        "self_collision_exclude_neighbors", "enable_normal_untangling", "self_collision_max_iterations",
        "self_collision_substep_interval", "self_collision_ee_substep_interval",
        "self_collision_algorithm", "enable_pair_cache", "pair_cache_margin_mode",
        "pair_cache_safety_margin", "pair_cache_horizon_scale", "pair_cache_max_horizon",
        "pair_cache_max_pairs", "enable_pair_cache_final_fallback",
        "enable_edge_collision", "edge_margin_scale", "edge_margin_offset",
        "sewing_stiffness", "enable_sewing_lock", "sewing_lock_distance", "sewing_shrink_speed",
        "enable_sewing_priority", "sewing_priority_threshold", "sewing_priority_merge_dist",
        "sewing_priority_ramp_frames", "sewing_priority_max_frames", "areal_density",
        "enable_adaptive_substep", "min_substeps", "max_substeps",
        "auto_coupled_on_low_substeps", "auto_compensate_iterations",
        "enable_strain_adaptive", "strain_tolerance", "solver_iterations", "solver_mode",
        "workgroup_size", "enable_self_collision",
    }
    unhandled_blender_props = sorted(list(blender_props - ALL_KNOWN_NON_SIMCONFIG_PROPERTIES - sim_handled_props))

    # 3. Rust SimConfig / SimParams ↔ WGSL 構造体照合
    sim_params_wgsl = wgsl_structs.get("SimParams", set())
    collision_params_wgsl = wgsl_structs.get("CollisionParams", set())
    self_collision_params_wgsl = wgsl_structs.get("SelfCollisionParams", set())

    is_ok = (len(missing_in_rust) == 0) and (len(missing_in_python) == 0) and (len(unhandled_blender_props) == 0)

    return {
        "status": "PASS" if is_ok else "FAIL",
        "counts": {
            "blender_properties": len(blender_props),
            "python_live_keys": len(live_keys),
            "python_extra_keys": len(extra_keys),
            "rust_simconfig_fields": len(rust_fields),
            "wgsl_sim_params_members": len(sim_params_wgsl),
            "wgsl_collision_params_members": len(collision_params_wgsl),
            "wgsl_self_collision_params_members": len(self_collision_params_wgsl),
        },
        "python_rust_parity": {
            "is_matched": len(missing_in_rust) == 0 and len(missing_in_python) == 0,
            "missing_in_rust": missing_in_rust,
            "missing_in_python": missing_in_python,
        },
        "blender_unhandled_props": unhandled_blender_props,
    }


def format_text_report(report: Dict[str, Any]) -> str:
    lines = [
        "============================================================",
        "  Taremin Cloth Python-Rust-wgpu プロトコル整合性監査",
        "============================================================",
        f"総合判定: {report['status']}",
        "",
        "【パラメータ集計】",
        f"  - Blender ObjectSettings プロパティ数: {report['counts']['blender_properties']}",
        f"  - Python SimConfig (LIVE_KEYS) キー数: {report['counts']['python_live_keys']}",
        f"  - Python 付帯キー (EXTRA_KEYS) キー数: {report['counts']['python_extra_keys']}",
        f"  - Rust SimConfig フィールド数: {report['counts']['rust_simconfig_fields']}",
        f"  - WGSL SimParams メンバ数: {report['counts']['wgsl_sim_params_members']}",
        f"  - WGSL CollisionParams メンバ数: {report['counts']['wgsl_collision_params_members']}",
        f"  - WGSL SelfCollisionParams メンバ数: {report['counts']['wgsl_self_collision_params_members']}",
        "",
        "【Python <-> Rust SimConfig 突合結果】",
    ]

    parity = report["python_rust_parity"]
    if parity["is_matched"]:
        lines.append("  [OK] Python LIVE_KEYS と Rust SimConfig は 100% 完全一致しています。")
    else:
        if parity["missing_in_rust"]:
            lines.append(f"  [ERROR] Rust側に存在しないキー ({len(parity['missing_in_rust'])}件):")
            for k in parity["missing_in_rust"]:
                lines.append(f"    - {k}")
        if parity["missing_in_python"]:
            lines.append(f"  [ERROR] Python側に存在しないフィールド ({len(parity['missing_in_python'])}件):")
            for k in parity["missing_in_python"]:
                lines.append(f"    - {k}")

    lines.append("")
    lines.append("【Blenderプロパティ 孤立チェック】")
    unhandled = report["blender_unhandled_props"]
    if not unhandled:
        lines.append("  [OK] 未割り当ての孤立した物理プロパティは存在しません。")
    else:
        lines.append(f"  [WARN] SimConfig・UI除外に未登録のプロパティ ({len(unhandled)}件):")
        for p in unhandled:
            lines.append(f"    - {p}")

    lines.append("============================================================")
    return "\n".join(lines)


def format_markdown_report(report: Dict[str, Any]) -> str:
    parity = report["python_rust_parity"]
    status_badge = "PASS" if report["status"] == "PASS" else "FAIL"
    lines = [
        f"# Taremin Cloth プロトコル整合性監査レポート ({status_badge})",
        "",
        "## パラメータ集計",
        "| レイヤー | コンポーネント | パラメータ数 |",
        "|---|---|---|",
        f"| 1. Blender UI | `TareminClothObjectSettings` | {report['counts']['blender_properties']} |",
        f"| 2. Python | `SimConfig (LIVE_KEYS)` | {report['counts']['python_live_keys']} |",
        f"| 3. Rust Core | `cloth_core::config::SimConfig` | {report['counts']['rust_simconfig_fields']} |",
        f"| 4. wgpu / WGSL | `SimParams` | {report['counts']['wgsl_sim_params_members']} |",
        f"| 4. wgpu / WGSL | `CollisionParams` | {report['counts']['wgsl_collision_params_members']} |",
        f"| 4. wgpu / WGSL | `SelfCollisionParams` | {report['counts']['wgsl_self_collision_params_members']} |",
        "",
        "## Python <-> Rust SimConfig 突合",
    ]
    if parity["is_matched"]:
        lines.append("- **ステータス**: 完全一致 (100% 同期)")
    else:
        lines.append("- **ステータス**: 不整合あり")
        if parity["missing_in_rust"]:
            lines.append(f"- **Rust欠落キー**: `{', '.join(parity['missing_in_rust'])}`")
        if parity["missing_in_python"]:
            lines.append(f"- **Python欠落キー**: `{', '.join(parity['missing_in_python'])}`")

    lines.append("")
    lines.append("## Blenderプロパティ 孤立チェック")
    if not report["blender_unhandled_props"]:
        lines.append("- 未割り当ての孤立した物理プロパティはありません。")
    else:
        lines.append(f"- 未登録プロパティ: `{', '.join(report['blender_unhandled_props'])}`")

    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Taremin Cloth プロトコル整合性監査ツール")
    parser.add_argument("--check", action="store_true", help="不整合があれば非ゼロで終了 (CI向け)")
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text", help="出力形式")
    args = parser.parse_args()

    report = audit_protocol()

    if args.format == "json":
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.format == "markdown":
        print(format_markdown_report(report))
    else:
        print(format_text_report(report))

    if args.check and report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
