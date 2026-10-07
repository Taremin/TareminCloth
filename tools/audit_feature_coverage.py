#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taremin Cloth - エンジンコア機能・クライアント統合カバレッジ監査ツール

エンジンコア (crates/cloth_core) に対し、
Blenderアドオン (python/taremin_cloth) と 独立GUI (crates/cloth_gui) が
どれだけ機能を網羅・実装しているかを3つの軸で静的解析・測定・比較します。

3つの測定軸:
  軸 A: SimConfig パラメータ・カバレッジ (全52項目)
  軸 B: シミュレータ API / メソッド・カバレッジ (ClothSimulator メソッド)
  軸 C: コライダー・操作・機能種別カバレッジ (Sphere/Capsule/SDF, Grab, Pin, Profile等)

使用例:
  python tools/audit_feature_coverage.py
  python tools/audit_feature_coverage.py --format markdown
  python tools/audit_feature_coverage.py --format json
  python tools/audit_feature_coverage.py --check
"""

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Windows コンソールでの UTF-8 文字出力対応
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# プロジェクトルートの解決
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==============================================================================
# エイリアス解決マッピング
# ==============================================================================

# メタデータフィールド (物理パラメータではなくスキーマバージョン等のため別枠扱い)
METADATA_FIELDS = {"version"}

# SimConfigキー名 ↔ 各クライアント層での識別名候補
PARAM_ALIASES = {
    "tension_stiffness": ["stiffness", "tension_stiffness"],
    "damping": ["damping", "air_damping"],
    "enable_self_collision": ["use_self_collision", "enable_self_collision", "self_collision_enabled", "enabled"],
    "enable_edge_collision": ["use_edge_collision", "enable_edge_collision", "edge_collision_enabled"],
    "enable_sewing_lock": ["use_sewing_lock", "enable_sewing_lock"],
    "enable_adaptive_substep": ["use_adaptive_substep", "enable_adaptive_substep"],
    "enable_strain_adaptive": ["use_strain_adaptive", "enable_strain_adaptive"],
    "enable_pair_cache": ["use_pair_cache", "enable_pair_cache"],
    "enable_pair_final_fallback": [
        "use_pair_cache_final_fallback",
        "enable_pair_final_fallback",
        "enable_pair_cache_final_fallback",
    ],
    "auto_coupled_on_low_substeps": ["use_auto_coupled_on_low_substeps", "auto_coupled_on_low_substeps"],
    "auto_compensate_iterations": ["use_auto_compensate_iterations", "auto_compensate_iterations"],
    "enable_normal_untangling": ["use_normal_untangling", "enable_normal_untangling"],
    "solver_iterations": ["solver_iterations", "iterations"],
    "self_collision_max_iterations": [
        "self_collision_max_iterations",
        "max_iterations",
        "max_search_iterations",
    ],
    "pair_margin_mode": ["pair_margin_mode", "pair_cache_margin_mode"],
    "pair_safety_margin": ["pair_safety_margin", "pair_cache_safety_margin"],
    "pair_horizon_scale": ["pair_horizon_scale", "pair_cache_horizon_scale"],
    "pair_max_horizon": ["pair_max_horizon", "pair_cache_max_horizon"],
    "pair_max_pairs": ["pair_max_pairs", "pair_cache_max_pairs"],
    "sewing_priority_enabled": ["sewing_priority_enabled", "use_sewing_priority"],
    "sewing_priority_merge_dist": ["sewing_priority_merge_dist", "sewing_priority_merge_distance"],
    "sewing_priority_ramp_frames": ["sewing_priority_ramp_frames"],
    "sewing_priority_max_frames": ["sewing_priority_max_frames"],
    "sewing_priority_threshold": ["sewing_priority_threshold"],
}


# ==============================================================================
# 軸 A: パラメータ・カバレッジ (SimConfig) 解析
# ==============================================================================

def parse_sim_config(root: str) -> List[Tuple[str, str]]:
    """crates/cloth_core/src/config.rs から SimConfig フィールド (名前, 型) を抽出"""
    config_path = os.path.join(root, "crates", "cloth_core", "src", "config.rs")
    if not os.path.exists(config_path):
        return []

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    # pub struct SimConfig { ... } の中身を抽出
    m = re.search(r"pub\s+struct\s+SimConfig\s*\{([^}]+)\}", content)
    if not m:
        return []

    struct_body = m.group(1)
    fields = []
    for line in struct_body.splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        field_match = re.match(r"pub\s+([a-zA-Z0-9_]+)\s*:\s*([^,;]+)", line)
        if field_match:
            name = field_match.group(1).strip()
            ftype = field_match.group(2).strip()
            fields.append((name, ftype))
    return fields


def parse_blender_engine_simconfig(root: str) -> Set[str]:
    """python/taremin_cloth/engine/simconfig.py から SIGNATURE_KEYS / LIVE_KEYS を抽出"""
    path = os.path.join(root, "python", "taremin_cloth", "engine", "simconfig.py")
    if not os.path.exists(path):
        return set()

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    keys = set()
    m = re.search(r"SIGNATURE_KEYS\s*=\s*\(([^)]+)\)", content)
    if m:
        for item in re.findall(r'"([^"]+)"', m.group(1)):
            keys.add(item)
    return keys


def parse_blender_properties(root: str) -> Set[str]:
    """python/taremin_cloth/properties.py から TareminClothObjectSettings のプロパティ一覧を抽出"""
    path = os.path.join(root, "python", "taremin_cloth", "properties.py")
    if not os.path.exists(path):
        return set()

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    props = set()
    # TareminClothObjectSettings クラス定義から抽出
    m = re.search(r"class\s+TareminClothObjectSettings\b.*?:(.*?)(?=\nclass\s+|\Z)", content, re.DOTALL)
    if m:
        body = m.group(1)
        for line in body.splitlines():
            prop_match = re.match(r"^\s*([a-zA-Z0-9_]+)\s*:\s*[A-Z][a-zA-Z0-9_]*Property", line)
            if prop_match:
                props.add(prop_match.group(1))
    return props


def parse_blender_panels(root: str) -> Set[str]:
    """python/taremin_cloth/panels.py から描画されているプロパティ名を抽出"""
    path = os.path.join(root, "python", "taremin_cloth", "panels.py")
    if not os.path.exists(path):
        return set()

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    props = set()
    # layout.prop(settings, "xxx") または .prop(..., "xxx")
    for m in re.finditer(r'\.prop\s*\([^,]+,\s*["\']([a-zA-Z0-9_]+)["\']', content):
        props.add(m.group(1))
    return props


def parse_gui_protocol(root: str) -> Tuple[Set[str], Set[str]]:
    """crates/cloth_gui/src/protocol.rs から (SceneInitDataフィールド群, GuiParamsUpdateフィールド群) を抽出"""
    path = os.path.join(root, "crates", "cloth_gui", "src", "protocol.rs")
    if not os.path.exists(path):
        return set(), set()

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    init_keys = set()
    update_keys = set()

    # 1. SceneInitData
    m_init = re.search(r"pub\s+struct\s+SceneInitData\s*\{([^}]+)\}", content)
    if m_init:
        for line in m_init.group(1).splitlines():
            fm = re.match(r"\s*pub\s+([a-zA-Z0-9_]+)\s*:", line)
            if fm:
                init_keys.add(fm.group(1))

    # 2. GuiSelfCollisionData (SceneInitData.self_collision 内にネスト)
    m_sc = re.search(r"pub\s+struct\s+GuiSelfCollisionData\s*\{([^}]+)\}", content)
    if m_sc:
        for line in m_sc.group(1).splitlines():
            fm = re.match(r"\s*pub\s+([a-zA-Z0-9_]+)\s*:", line)
            if fm:
                init_keys.add(fm.group(1))

    # 3. GuiParamsUpdate
    m_update = re.search(r"pub\s+struct\s+GuiParamsUpdate\s*\{([^}]+)\}", content)
    if m_update:
        for line in m_update.group(1).splitlines():
            fm = re.match(r"\s*pub\s+([a-zA-Z0-9_]+)\s*:", line)
            if fm:
                update_keys.add(fm.group(1))

    return init_keys, update_keys


def parse_gui_standalone_ui(root: str) -> Set[str]:
    """crates/cloth_gui/src/app.rs から単体UIで操作可能なフィールド・変数を抽出"""
    path = os.path.join(root, "crates", "cloth_gui", "src", "app.rs")
    if not os.path.exists(path):
        return set()

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    fields = set()
    for m in re.finditer(r"self\.([a-zA-Z0-9_]+)", content):
        fields.add(m.group(1))
    for m in re.finditer(r"&mut\s+self\.([a-zA-Z0-9_]+)", content):
        fields.add(m.group(1))

    return fields


def check_param_match(param: str, candidate_set: Set[str]) -> bool:
    """エイリアスを考慮してパラメータが指定のキー集合に含まれるか判定"""
    if param in candidate_set:
        return True
    aliases = PARAM_ALIASES.get(param, [])
    for a in aliases:
        if a in candidate_set:
            return True
    return False


# ==============================================================================
# 軸 B: シミュレータ API / メソッド・カバレッジ 解析
# ==============================================================================

def parse_cloth_simulator_api(root: str) -> List[str]:
    """crates/cloth_py/src/lib.rs の impl ClothSimulator から公開メソッド一覧を抽出"""
    path = os.path.join(root, "crates", "cloth_py", "src", "lib.rs")
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    m = re.search(r"impl\s+ClothSimulator\s*\{([\s\S]+?)\n\}", content)
    if not m:
        return []

    impl_body = m.group(1)
    methods = []
    for line in impl_body.splitlines():
        fm = re.match(r"^\s*fn\s+([a-zA-Z0-9_]+)\s*(?:<[^>]+>)?\s*\(", line)
        if fm:
            name = fm.group(1)
            if name not in ("new", "__repr__"):
                methods.append(name)
    return sorted(list(set(methods)))


def scan_blender_api_usage(root: str, methods: List[str]) -> Set[str]:
    """python/taremin_cloth 配下の Python コードで呼び出されているメソッドを検出"""
    py_dir = os.path.join(root, "python", "taremin_cloth")
    used = set()
    if not os.path.exists(py_dir):
        return used

    for dirpath, _, filenames in os.walk(py_dir):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fn)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
            for m in methods:
                pattern = r"(?:\b" + re.escape(m) + r"\b|\bgetattr\([^,]+,\s*[\"']" + re.escape(m) + r"[\"']|\bhasattr\([^,]+,\s*[\"']" + re.escape(m) + r"[\"'])"
                if re.search(pattern, code):
                    used.add(m)
    return used


def scan_gui_api_usage(root: str, methods: List[str]) -> Set[str]:
    """crates/cloth_gui 配下の Rust コードで呼び出されているメソッドを検出"""
    gui_dir = os.path.join(root, "crates", "cloth_gui")
    used = set()
    if not os.path.exists(gui_dir):
        return used

    for dirpath, _, filenames in os.walk(gui_dir):
        for fn in filenames:
            if not fn.endswith(".rs"):
                continue
            fpath = os.path.join(dirpath, fn)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
            for m in methods:
                pattern = r"\." + re.escape(m) + r"\s*\("
                if re.search(pattern, code):
                    used.add(m)
    return used


# ==============================================================================
# 軸 C: コライダー・操作・機能種別 カバレッジ
# ==============================================================================

@dataclass
class FeatureSpec:
    category: str
    name: str
    description: str
    blender_check: str
    gui_check: str


FEATURES_LIST = [
    # コライダー種別
    FeatureSpec("Colliders", "Sphere Collider", "解析球体コライダー", "sphere", "GuiColliderData"),
    FeatureSpec("Colliders", "Capsule Collider", "解析カプセルコライダー", "capsule", "GuiColliderData"),
    FeatureSpec("Colliders", "Plane Collider", "解析平面コライダー", "plane", "GuiColliderData"),
    FeatureSpec("Colliders", "Mesh Collider", "BVH三角形メッシュコライダー", "set_mesh_collider_triangles", "GuiMeshTriangleData"),
    FeatureSpec("Colliders", "Bone SDF Collider", "ボーンSDFボリュームテクスチャコライダー", "bake_bone_sdf_gpu", "GuiBoneSdfData"),
    FeatureSpec("Colliders", "Dynamic Bone SDF Tracking", "ボーン行列動的追従・Dirty更新", "dynamic_bone_sdf", "UpdateBoneTransforms"),

    # 対話・操作機能
    FeatureSpec("Interactive", "Play / Pause / Step", "シミュレーション進行・一時停止・コマ送り", "step_cloth_scene", "step"),
    FeatureSpec("Interactive", "Reset", "初期レストポーズへのリセット", "cache_rest_positions", "reset"),
    FeatureSpec("Interactive", "Cold Resume", "停止時の変形ポーズを保持して再開", "set_positions_and_velocities", "cached_positions"),
    FeatureSpec("Interactive", "Grab Dragging", "マウスドラッグによる布頂点操作", "grab", "grab_initial_pos"),
    FeatureSpec("Interactive", "Pin Toggle / Management", "頂点ピン留め設定・リアルタイム追加/解除", "pin", "pinned_verts"),
    FeatureSpec("Interactive", "Dynamic Pin Tracking", "外部オブジェクト・ボーンへのピン追従", "sync_attachment_pins", "UpdatePins"),
    FeatureSpec("Interactive", "Apply Pose (Writeback)", "変形結果をメッシュ形状として確定", "apply_modifier", "ApplyPose"),
    FeatureSpec("Interactive", "In-flight Param Tuning", "シミュレーション実行中のリアルタイム調整", "sync_cloth_parameters", "SetParams"),

    # 高度機能・ツール
    FeatureSpec("Advanced", "Debug Recording (.jsonl.gz)", "2階層スパースデバッグ録画", "debug_record", "debug_recorder"),
    FeatureSpec("Advanced", "GPU Profiling", "Timestamp QueryによるGPUパス別計測", "set_profiling_enabled", "set_profiling_enabled"),
    FeatureSpec("Advanced", "Drape Guide (2D-SDF)", "ドレープガイド・折り目テクスチャ誘導", "set_wrinkle_field_texture_2d", "wrinkle"),
    FeatureSpec("Advanced", "Pair Audit", "自己衝突ペアのリアルタイム診断・見逃し監査", "audit_pairs", "audit_pairs"),
    FeatureSpec("Advanced", "Virtual Mesh (V-V Mode)", "仮想球衝突によるセルフコリジョン", "virtual", "num_virtual_vertices"),
    FeatureSpec("Advanced", "Elastic Bands", "辺の自然長スケーリング・ゴム紐グループ", "elastic", "UpdateElasticScales"),
]


def audit_features(root: str) -> List[Dict[str, Any]]:
    """軸 C の各機能が Blender側および GUI側でサポートされているか静的走査して判定"""
    py_dir = os.path.join(root, "python", "taremin_cloth")
    gui_dir = os.path.join(root, "crates", "cloth_gui")

    py_content = []
    if os.path.exists(py_dir):
        for dirpath, _, filenames in os.walk(py_dir):
            for fn in filenames:
                if fn.endswith(".py"):
                    with open(os.path.join(dirpath, fn), "r", encoding="utf-8", errors="ignore") as f:
                        py_content.append(f.read())
    all_py_lower = "\n".join(py_content).lower()

    gui_content = []
    if os.path.exists(gui_dir):
        for dirpath, _, filenames in os.walk(gui_dir):
            for fn in filenames:
                if fn.endswith(".rs"):
                    with open(os.path.join(dirpath, fn), "r", encoding="utf-8", errors="ignore") as f:
                        gui_content.append(f.read())
    all_gui_lower = "\n".join(gui_content).lower()

    results = []
    for spec in FEATURES_LIST:
        b_ok = spec.blender_check.lower() in all_py_lower
        g_ok = spec.gui_check.lower() in all_gui_lower

        results.append({
            "category": spec.category,
            "name": spec.name,
            "description": spec.description,
            "blender": b_ok,
            "gui": g_ok,
        })
    return results


# ==============================================================================
# 総合集計 & レポート生成
# ==============================================================================

@dataclass
class AuditReport:
    params: List[Dict[str, Any]] = field(default_factory=list)
    param_counts: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    apis: List[Dict[str, Any]] = field(default_factory=list)
    api_counts: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    features: List[Dict[str, Any]] = field(default_factory=list)
    feature_counts: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    actionable_items: List[str] = field(default_factory=list)


def run_full_audit(root: str) -> AuditReport:
    report = AuditReport()

    # 1. 軸 A 解析
    sim_fields = parse_sim_config(root)
    b_eng_keys = parse_blender_engine_simconfig(root)
    b_prop_keys = parse_blender_properties(root)
    b_panel_keys = parse_blender_panels(root)
    g_init_keys, g_update_keys = parse_gui_protocol(root)
    g_ui_keys = parse_gui_standalone_ui(root)

    total_params = len(sim_fields)
    c_b_eng = 0
    c_b_prop = 0
    c_b_panel = 0
    c_g_init = 0
    c_g_update = 0
    c_g_ui = 0

    missing_in_g_update = []
    missing_in_g_ui = []
    missing_in_g_init = []

    for name, ftype in sim_fields:
        ok_b_eng = check_param_match(name, b_eng_keys)
        ok_b_prop = check_param_match(name, b_prop_keys)
        ok_b_panel = check_param_match(name, b_panel_keys)
        ok_g_init = check_param_match(name, g_init_keys)
        ok_g_update = check_param_match(name, g_update_keys)
        ok_g_ui = check_param_match(name, g_ui_keys)

        if ok_b_eng: c_b_eng += 1
        if ok_b_prop: c_b_prop += 1
        if ok_b_panel: c_b_panel += 1
        if ok_g_init: c_g_init += 1
        else: missing_in_g_init.append(name)
        if ok_g_update: c_g_update += 1
        else: missing_in_g_update.append(name)
        if ok_g_ui: c_g_ui += 1
        else: missing_in_g_ui.append(name)

        report.params.append({
            "name": name,
            "type": ftype,
            "blender_engine": ok_b_eng,
            "blender_prop": ok_b_prop,
            "blender_panel": ok_b_panel,
            "gui_init": ok_g_init,
            "gui_update": ok_g_update,
            "gui_ui": ok_g_ui,
        })

    report.param_counts = {
        "blender_engine": (c_b_eng, total_params),
        "blender_prop": (c_b_prop, total_params),
        "blender_panel": (c_b_panel, total_params),
        "gui_init": (c_g_init, total_params),
        "gui_update": (c_g_update, total_params),
        "gui_ui": (c_g_ui, total_params),
    }

    # 2. 軸 B 解析
    api_methods = parse_cloth_simulator_api(root)
    total_apis = len(api_methods)
    used_py_apis = scan_blender_api_usage(root, api_methods)
    used_gui_apis = scan_gui_api_usage(root, api_methods)

    c_b_api = len(used_py_apis)
    c_g_api = len(used_gui_apis)

    for m in api_methods:
        ok_b = m in used_py_apis
        ok_g = m in used_gui_apis
        report.apis.append({
            "method": m,
            "blender": ok_b,
            "gui": ok_g,
        })

    report.api_counts = {
        "blender_api": (c_b_api, total_apis),
        "gui_api": (c_g_api, total_apis),
    }

    # 3. 軸 C 解析
    feat_results = audit_features(root)
    total_feats = len(feat_results)
    c_b_feat = sum(1 for r in feat_results if r["blender"])
    c_g_feat = sum(1 for r in feat_results if r["gui"])

    report.features = feat_results
    report.feature_counts = {
        "blender_feature": (c_b_feat, total_feats),
        "gui_feature": (c_g_feat, total_feats),
    }

    if missing_in_g_update:
        report.actionable_items.append(
            f"GUI IPC Update (GuiParamsUpdate) に自己衝突・ソルバー等の動的変更項目が欠落 ({len(missing_in_g_update)}項目: {', '.join(missing_in_g_update[:5])}...)"
        )
    if missing_in_g_init:
        report.actionable_items.append(
            f"GUI IPC Init (SceneInitData) に最新のひずみ適応・Coupledコライダー等が未定義 ({len(missing_in_g_init)}項目: {', '.join(missing_in_g_init[:5])}...)"
        )
    if missing_in_g_ui:
        report.actionable_items.append(
            f"独立GUI 単体UI (app.rs) に縫合・詳細自己衝突等のスライダーが未実装 ({len(missing_in_g_ui)}項目: {', '.join(missing_in_g_ui[:5])}...)"
        )

    return report


def pct(n: int, total: int) -> float:
    return (n / total * 100.0) if total > 0 else 0.0


def icon(ok: bool) -> str:
    return "✅" if ok else "❌"


def format_console(report: AuditReport) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(" Taremin Cloth - エンジンコア機能・クライアント統合カバレッジ監査レポート")
    lines.append("=" * 80)
    lines.append("")

    lines.append("【総合カバレッジ・サマリー】")
    lines.append("  レイヤー / 軸                       実装数 / 全数    カバレッジ (%)   状態")
    lines.append("  " + "-" * 72)

    # 軸 A
    p_be, t_p = report.param_counts["blender_engine"]
    p_bp, _ = report.param_counts["blender_prop"]
    p_bpan, _ = report.param_counts["blender_panel"]
    p_gi, _ = report.param_counts["gui_init"]
    p_gu, _ = report.param_counts["gui_update"]
    p_gui, _ = report.param_counts["gui_ui"]

    # 物理パラメータ (メタデータ除外) での実効数
    phys_total = len([p for p in report.params if p["name"] not in METADATA_FIELDS])
    phys_be = len([p for p in report.params if p["name"] not in METADATA_FIELDS and p["blender_engine"]])
    phys_gi = len([p for p in report.params if p["name"] not in METADATA_FIELDS and p["gui_init"]])
    phys_gu = len([p for p in report.params if p["name"] not in METADATA_FIELDS and p["gui_update"]])
    phys_gui = len([p for p in report.params if p["name"] not in METADATA_FIELDS and p["gui_ui"]])

    lines.append(f"  [軸 A: SimConfig 物理パラメータ (全{phys_total}項目)]")
    lines.append(f"    - Blender Engine (simconfig.py)     {phys_be:3d} / {phys_total:3d}        {pct(phys_be, phys_total):5.1f}%       {'✅ 完全同期 (100%)' if phys_be == phys_total else '⚠️ 欠落あり'}")
    lines.append(f"    - Blender Properties (設定定義)      {p_bp:3d} / {phys_total:3d}        {pct(p_bp, phys_total):5.1f}%       {'✅ 良好' if pct(p_bp, phys_total) > 85 else '⚠️'}")
    lines.append(f"    - Blender Panels (UIウィジェット)     {p_bpan:3d} / {phys_total:3d}        {pct(p_bpan, phys_total):5.1f}%       {'✅ 良好' if pct(p_bpan, phys_total) > 80 else '⚠️'}")
    lines.append(f"    - GUI IPC Init (SceneInitData)       {phys_gi:3d} / {phys_total:3d}        {pct(phys_gi, phys_total):5.1f}%       {'⚠️ 同期遅れ' if pct(phys_gi, phys_total) < 90 else '✅'}")
    lines.append(f"    - GUI IPC Update (GuiParamsUpdate)   {phys_gu:3d} / {phys_total:3d}        {pct(phys_gu, phys_total):5.1f}%       {'❌ 大幅欠落' if pct(phys_gu, phys_total) < 50 else '⚠️'}")
    lines.append(f"    - 独立GUI 単体UI (app.rs)            {phys_gui:3d} / {phys_total:3d}        {pct(phys_gui, phys_total):5.1f}%       {'❌ 未実装多数' if pct(phys_gui, phys_total) < 60 else '⚠️'}")

    # 軸 B
    a_b, t_a = report.api_counts["blender_api"]
    a_g, _ = report.api_counts["gui_api"]
    lines.append("")
    lines.append(f"  [軸 B: シミュレータ API メソッド (全{t_a}メソッド)]")
    lines.append(f"    - Blender Python アドオン            {a_b:3d} / {t_a:3d}        {pct(a_b, t_a):5.1f}%       {'✅ 活用中' if pct(a_b, t_a) > 60 else '⚠️'}")
    lines.append(f"    - 独立GUI (crates/cloth_gui)         {a_g:3d} / {t_a:3d}        {pct(a_g, t_a):5.1f}%       {'⚠️ 最小利用' if pct(a_g, t_a) < 40 else '✅'}")

    # 軸 C
    f_b, t_f = report.feature_counts["blender_feature"]
    f_g, _ = report.feature_counts["gui_feature"]
    lines.append("")
    lines.append(f"  [軸 C: コライダー & 機能種別 (全{t_f}機能)]")
    lines.append(f"    - Blender アドオン対応機能           {f_b:3d} / {t_f:3d}        {pct(f_b, t_f):5.1f}%       {'✅ 網羅' if pct(f_b, t_f) > 85 else '⚠️'}")
    lines.append(f"    - 独立GUI 対応機能                   {f_g:3d} / {t_f:3d}        {pct(f_g, t_f):5.1f}%       {'✅ 良好' if pct(f_g, t_f) > 75 else '⚠️'}")

    lines.append("")
    lines.append("-" * 80)
    lines.append("【主なアクション・未実装項目】")
    for act in report.actionable_items:
        lines.append(f"  • {act}")

    lines.append("=" * 80)
    return "\n".join(lines)


def format_markdown(report: AuditReport) -> str:
    lines = []
    lines.append("# Taremin Cloth - エンジンコア機能・クライアント統合カバレッジ監査レポート")
    lines.append("")

    lines.append("## 1. 総合カバレッジ・サマリー")
    lines.append("")
    lines.append("| レイヤー / 軸 | 実装数 / 全数 | カバレッジ (%) | 状態 |")
    lines.append("|---|:---:|:---:|:---:|")

    p_be, t_p = report.param_counts["blender_engine"]
    p_bp, _ = report.param_counts["blender_prop"]
    p_bpan, _ = report.param_counts["blender_panel"]
    p_gi, _ = report.param_counts["gui_init"]
    p_gu, _ = report.param_counts["gui_update"]
    p_gui, _ = report.param_counts["gui_ui"]

    lines.append(f"| **軸 A: Blender Engine** (`simconfig.py`) | {p_be} / {t_p} | **{pct(p_be, t_p):.1f}%** | {'✅ 完全網羅' if p_be == t_p else '⚠️'} |")
    lines.append(f"| **軸 A: Blender Properties** (`properties.py`) | {p_bp} / {t_p} | **{pct(p_bp, t_p):.1f}%** | {'✅' if pct(p_bp, t_p) > 90 else '⚠️'} |")
    lines.append(f"| **軸 A: Blender Panels** (`panels.py`) | {p_bpan} / {t_p} | **{pct(p_bpan, t_p):.1f}%** | {'✅' if pct(p_bpan, t_p) > 80 else '⚠️'} |")
    lines.append(f"| **軸 A: GUI IPC Init** (`SceneInitData`) | {p_gi} / {t_p} | **{pct(p_gi, t_p):.1f}%** | {'⚠️ 同期遅れ' if pct(p_gi, t_p) < 85 else '✅'} |")
    lines.append(f"| **軸 A: GUI IPC Update** (`GuiParamsUpdate`) | {p_gu} / {t_p} | **{pct(p_gu, t_p):.1f}%** | {'❌ 大幅欠落' if pct(p_gu, t_p) < 50 else '⚠️'} |")
    lines.append(f"| **軸 A: 独立GUI 単体UI** (`app.rs`) | {p_gui} / {t_p} | **{pct(p_gui, t_p):.1f}%** | {'❌ 未実装多数' if pct(p_gui, t_p) < 60 else '⚠️'} |")

    a_b, t_a = report.api_counts["blender_api"]
    a_g, _ = report.api_counts["gui_api"]
    lines.append(f"| **軸 B: Blender API 利用率** | {a_b} / {t_a} | **{pct(a_b, t_a):.1f}%** | {'✅' if pct(a_b, t_a) > 75 else '⚠️'} |")
    lines.append(f"| **軸 B: 独立GUI API 利用率** | {a_g} / {t_a} | **{pct(a_g, t_a):.1f}%** | {'⚠️ 最小利用' if pct(a_g, t_a) < 60 else '✅'} |")

    f_b, t_f = report.feature_counts["blender_feature"]
    f_g, _ = report.feature_counts["gui_feature"]
    lines.append(f"| **軸 C: Blender 機能・コライダー網羅** | {f_b} / {t_f} | **{pct(f_b, t_f):.1f}%** | {'✅' if pct(f_b, t_f) > 85 else '⚠️'} |")
    lines.append(f"| **軸 C: 独立GUI 機能・コライダー網羅** | {f_g} / {t_f} | **{pct(f_g, t_f):.1f}%** | {'⚠️ 一部未対応' if pct(f_g, t_f) < 70 else '✅'} |")

    lines.append("")
    lines.append("## 2. 軸 A: パラメータ対応マトリクス詳細")
    lines.append("")
    lines.append("| パラメータ名 | 型 | Blender Engine | Blender UI | GUI Init | GUI Update | GUI 単体UI |")
    lines.append("|---|---|:---:|:---:|:---:|:---:|:---:|")
    for p in report.params:
        lines.append(
            f"| `{p['name']}` | `{p['type']}` | {icon(p['blender_engine'])} | {icon(p['blender_panel'])} | "
            f"{icon(p['gui_init'])} | {icon(p['gui_update'])} | {icon(p['gui_ui'])} |"
        )

    lines.append("")
    lines.append("## 3. 軸 C: コライダー・対話機能対応マトリクス")
    lines.append("")
    lines.append("| 分類 | 機能名 | 説明 | Blender | 独立GUI |")
    lines.append("|---|---|---|:---:|:---:|")
    for f in report.features:
        lines.append(f"| {f['category']} | **{f['name']}** | {f['description']} | {icon(f['blender'])} | {icon(f['gui'])} |")

    lines.append("")
    lines.append("## 4. 主な改善アクション")
    lines.append("")
    for act in report.actionable_items:
        lines.append(f"- ⚠️ {act}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Taremin Cloth エンジンコア機能・クライアント統合カバレッジ監査ツール")
    parser.add_argument("--format", choices=["console", "markdown", "json"], default="console", help="出力フォーマット")
    parser.add_argument("--check", action="store_true", help="主要な同期漏れ (Blender Engineパリティ欠落等) があれば非ゼロで終了")
    parser.add_argument("--output", "-o", type=str, default="", help="出力先ファイルパス (省略時は標準出力)")
    args = parser.parse_args()

    report = run_full_audit(PROJECT_ROOT)

    if args.format == "console":
        out_text = format_console(report)
    elif args.format == "markdown":
        out_text = format_markdown(report)
    elif args.format == "json":
        out_text = json.dumps({
            "param_counts": report.param_counts,
            "api_counts": report.api_counts,
            "feature_counts": report.feature_counts,
            "actionable_items": report.actionable_items,
            "params": report.params,
            "apis": report.apis,
            "features": report.features,
        }, indent=2, ensure_ascii=False)
    else:
        out_text = format_console(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_text)
        print(f"Report saved to {args.output}")
    else:
        print(out_text)

    if args.check:
        phys_total = len([p for p in report.params if p["name"] not in METADATA_FIELDS])
        phys_be = len([p for p in report.params if p["name"] not in METADATA_FIELDS and p["blender_engine"]])
        if phys_be < phys_total:
            print(f"[ERROR] Blender Engine 物理パラメータ欠落: {phys_be}/{phys_total}", file=sys.stderr)
            sys.exit(1)
        print("[CHECK PASSED] Blender Engine 物理パラメータ完全同期 (100%)")


if __name__ == "__main__":
    main()
