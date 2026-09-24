"""
taremin_cloth SimConfig収集・適用モジュール (Single Source of Truth)

Blender設定 (`taremin_cloth` プロパティ) から Rust `SimConfig`
(`crates/cloth_core/src/config.rs`) 互換の辞書を組み立てる唯一の経路。
`engine/params.py` (In-proc同期)、`engine/gui_client.py` (GUI転送)、
`engine/cache.py` (署名) の3層はすべて本モジュール経由で値を取得し、
新パラメータ追加時の写像漏れを防ぐ。

Rust側に存在しないPython側付帯キー (`EXTRA_KEYS`) は転送・署名専用であり、
`apply_config_json` 送付時は除外される (serdeは未知キーを無視するが、
明示的に除去してペイロードを安定させる)。
"""

import json
from typing import Any, Dict, Optional, Tuple

#: Rust SimConfigに存在しないPython側付帯キー (転送・署名専用)
EXTRA_KEYS = ("gravity_scale", "sewing_shrink_speed")

#: 署名に含めるLiveキー (固定順序・丸め済みタプル化される)
SIGNATURE_KEYS = (
    "gravity",
    "damping",
    "tension_damping",
    "compression_damping",
    "shear_damping",
    "bending_damping",
    "tension_stiffness",
    "compression_stiffness",
    "shear_stiffness",
    "bending_stiffness",
    "solver_iterations",
    "solver_mode",
    "workgroup_size",
    "enable_self_collision",
    "relief_factor",
    "max_displacement_ratio",
    "exclude_neighbors",
    "enable_normal_untangling",
    "self_collision_max_iterations",
    "coupled_mode",
    "post_relaxation_iters",
    "substep_interval",
    "enable_pair_cache",
    "pair_margin_mode",
    "pair_safety_margin",
    "pair_horizon_scale",
    "pair_max_horizon",
    "pair_max_pairs",
    "enable_pair_final_fallback",
    "enable_edge_collision",
    "edge_margin_scale",
    "edge_margin_offset",
    "sewing_stiffness",
    "enable_sewing_lock",
    "sewing_priority_enabled",
    "sewing_priority_threshold",
    "sewing_priority_merge_dist",
    "sewing_priority_ramp_frames",
    "sewing_priority_max_frames",
    "gravity_scale",
    "sewing_shrink_speed",
)

#: apply_config_json に渡すLiveキー (= SIGNATURE_KEYS から付帯キーを除いたもの)
LIVE_KEYS = tuple(k for k in SIGNATURE_KEYS if k not in EXTRA_KEYS)


def _get(settings: Any, name: str, default: Any) -> Any:
    return getattr(settings, name, default)


def _f(settings: Any, name: str, default: float) -> float:
    try:
        return float(_get(settings, name, default))
    except (TypeError, ValueError):
        return float(default)


def _i(settings: Any, name: str, default: int) -> int:
    try:
        return int(_get(settings, name, default))
    except (TypeError, ValueError):
        return int(default)


def coupled_mode_from_settings(settings: Any) -> Tuple[int, int]:
    """coupled_self_collision_mode設定を (mode_int, relax_iters) に変換する。

    params.py / gui_client.py に重複していた同一分岐の一本化。
    """
    mode_str = _get(settings, "coupled_self_collision_mode", "RELAXATION")
    if mode_str == "OFF":
        return 0, 0
    try:
        relax_iters = int(_get(settings, "post_collision_relaxation_iters", 2))
    except (TypeError, ValueError):
        relax_iters = 2
    if mode_str == "FULL_COUPLED":
        return 3, relax_iters if relax_iters != 0 else 1
    return 1, relax_iters if relax_iters != 0 else 2


def gravity_from_settings(settings: Any, scene: Any = None) -> Tuple[list, float]:
    """(重力ベクトル, 倍率) を算出する。scene=None時はフォールバック式。"""
    scale = _f(settings, "gravity", 1.0)
    if scene is not None and bool(getattr(scene, "use_gravity", True)):
        sg = scene.gravity
        return [float(sg[0] * scale), float(sg[1] * scale), float(sg[2] * scale)], scale
    if scene is not None:
        return [0.0, 0.0, 0.0], 0.0
    return [0.0, 0.0, -9.81 * scale], scale


def collect_sim_config(settings: Any, scene: Any = None) -> Dict[str, Any]:
    """Blender設定からSimConfig互換辞書を組み立てる (付帯キー付き)。"""
    gravity_vec, scale = gravity_from_settings(settings, scene)
    coupled_mode, relax_iters = coupled_mode_from_settings(settings)
    return {
        "version": 1,
        "gravity": gravity_vec,
        "damping": _f(settings, "air_damping", 1.0),
        "tension_damping": _f(settings, "tension_damping", 5.0),
        "compression_damping": _f(settings, "compression_damping", 5.0),
        "shear_damping": _f(settings, "shear_damping", 5.0),
        "bending_damping": _f(settings, "bending_damping", 0.5),
        "tension_stiffness": _f(settings, "tension_stiffness", 1000.0),
        "compression_stiffness": _f(settings, "compression_stiffness", 100.0),
        "shear_stiffness": _f(settings, "shear_stiffness", 100.0),
        "bending_stiffness": _f(settings, "bending_stiffness", 10.0),
        "solver_iterations": _i(settings, "solver_iterations", 2),
        "solver_mode": 1 if _get(settings, "solver_mode", "COLORING") == "ATOMIC" else 0,
        "workgroup_size": _i(settings, "workgroup_size", 32),
        "enable_self_collision": bool(_get(settings, "enable_self_collision", False)),
        "relief_factor": _f(settings, "self_collision_relief_factor", 0.2),
        "max_displacement_ratio": _f(settings, "self_collision_max_displacement_ratio", 0.2),
        "exclude_neighbors": bool(_get(settings, "self_collision_exclude_neighbors", True)),
        "enable_normal_untangling": bool(_get(settings, "enable_normal_untangling", True)),
        "self_collision_max_iterations": _i(settings, "self_collision_max_iterations", 256),
        "coupled_mode": coupled_mode,
        "post_relaxation_iters": relax_iters,
        "substep_interval": max(1, _i(settings, "self_collision_substep_interval", 1)),
        "enable_pair_cache": bool(_get(settings, "enable_pair_cache", False)),
        "pair_margin_mode": 0 if _get(settings, "pair_cache_margin_mode", "AUTO") == "FIXED" else 1,
        "pair_safety_margin": _f(settings, "pair_cache_safety_margin", 0.005),
        "pair_horizon_scale": _f(settings, "pair_cache_horizon_scale", 1.3),
        "pair_max_horizon": _f(settings, "pair_cache_max_horizon", 0.02),
        "pair_max_pairs": _i(settings, "pair_cache_max_pairs", 65536),
        "enable_pair_final_fallback": bool(_get(settings, "enable_pair_cache_final_fallback", True)),
        "enable_edge_collision": bool(_get(settings, "enable_edge_collision", False)),
        "edge_margin_scale": _f(settings, "edge_margin_scale", 1.0),
        "edge_margin_offset": _f(settings, "edge_margin_offset", 0.0),
        "sewing_stiffness": _f(settings, "sewing_stiffness", 10000.0),
        "enable_sewing_lock": bool(_get(settings, "enable_sewing_lock", True)),
        "sewing_priority_enabled": bool(_get(settings, "enable_sewing_priority", False)),
        "sewing_priority_threshold": _f(settings, "sewing_priority_threshold", 0.9),
        "sewing_priority_merge_dist": _f(settings, "sewing_priority_merge_dist", 0.005),
        "sewing_priority_ramp_frames": _i(settings, "sewing_priority_ramp_frames", 3),
        "sewing_priority_max_frames": _i(settings, "sewing_priority_max_frames", 600),
        # 付帯キー (転送・署名専用、Rust apply時は除外)
        "gravity_scale": scale,
        "sewing_shrink_speed": _f(settings, "sewing_shrink_speed", 1.0),
    }


def live_config_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """apply_config_json送付用にLiveキーのみ抽出する。"""
    return {k: cfg[k] for k in LIVE_KEYS if k in cfg}


def _round_value(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return round(v, 5)
    if isinstance(v, (list, tuple)):
        return tuple(round(float(x), 5) for x in v)
    return v


def sim_config_signature(cfg: Dict[str, Any]) -> tuple:
    """キャッシュ無効化用の正規署名タプル (丸め済み・順序固定)。"""
    return tuple(_round_value(cfg.get(k)) for k in SIGNATURE_KEYS)


def apply_sim_config(sim: Any, cfg: Dict[str, Any]) -> str:
    """SimConfig辞書をシミュレータに適用する。戻り値は適用経路 ("config"/"legacy")。

    新バイナリは `apply_config_json` 一発適用。旧バイナリ (メソッド欠如) や
    適用失敗時は従来の個別setter経路にフォールバックする。
    """
    live = live_config_dict(cfg)
    applier = getattr(sim, "apply_config_json", None)
    if callable(applier):
        try:
            applier(json.dumps(live))
            return "config"
        except Exception:
            pass
    apply_sim_config_legacy(sim, cfg)
    return "legacy"


def apply_sim_config_legacy(sim: Any, cfg: Dict[str, Any]) -> None:
    """旧バイナリ互換の個別setter適用 (フォールバック専用、新規追加禁止)。

    マッピング元は settings ではなく収集済み cfg 辞書とし、
    設定→値の変換ロジックの重複を生まないようにする。
    """
    if "gravity" in cfg and hasattr(sim, "set_gravity"):
        g = cfg["gravity"]
        sim.set_gravity(float(g[0]), float(g[1]), float(g[2]))
    if hasattr(sim, "set_damping"):
        sim.set_damping(float(cfg.get("damping", 0.01)))
    if hasattr(sim, "set_damping_all"):
        sim.set_damping_all(
            float(cfg.get("tension_damping", 0.0)),
            float(cfg.get("compression_damping", 0.0)),
            float(cfg.get("shear_damping", 0.0)),
            float(cfg.get("bending_damping", 0.0)),
        )
    if hasattr(sim, "set_solver_iterations"):
        sim.set_solver_iterations(int(cfg.get("solver_iterations", 2)))
    if hasattr(sim, "set_solver_mode"):
        sim.set_solver_mode(int(cfg.get("solver_mode", 0)))
    if hasattr(sim, "set_workgroup_size"):
        try:
            sim.set_workgroup_size(int(cfg.get("workgroup_size", 32)))
        except (TypeError, ValueError):
            pass
    if hasattr(sim, "set_enable_self_collision"):
        sim.set_enable_self_collision(bool(cfg.get("enable_self_collision", False)))
    if hasattr(sim, "set_self_collision_options"):
        sim.set_self_collision_options(
            relief_factor=float(cfg.get("relief_factor", 0.2)),
            max_displacement_ratio=float(cfg.get("max_displacement_ratio", 0.2)),
            exclude_neighbors=bool(cfg.get("exclude_neighbors", True)),
            enable_normal_untangling=bool(cfg.get("enable_normal_untangling", True)),
            max_iterations=int(cfg.get("self_collision_max_iterations", 256)),
        )
    if hasattr(sim, "set_coupled_self_collision_options"):
        sim.set_coupled_self_collision_options(
            int(cfg.get("coupled_mode", 1)), int(cfg.get("post_relaxation_iters", 2))
        )
    if hasattr(sim, "set_self_collision_substep_interval"):
        sim.set_self_collision_substep_interval(max(1, int(cfg.get("substep_interval", 1))))
    if hasattr(sim, "set_enable_pair_cache"):
        sim.set_enable_pair_cache(bool(cfg.get("enable_pair_cache", False)))
    if hasattr(sim, "set_pair_cache_options"):
        max_pairs = int(cfg.get("pair_max_pairs", 65536))
        sim.set_pair_cache_options(
            max_pairs,
            max_pairs,
            int(cfg.get("pair_margin_mode", 1)),
            float(cfg.get("pair_safety_margin", 0.005)),
            float(cfg.get("pair_horizon_scale", 1.3)),
            float(cfg.get("pair_max_horizon", 0.02)),
        )
    if hasattr(sim, "set_enable_pair_cache_final_fallback"):
        sim.set_enable_pair_cache_final_fallback(bool(cfg.get("enable_pair_final_fallback", True)))
    if hasattr(sim, "set_enable_edge_collision"):
        sim.set_enable_edge_collision(bool(cfg.get("enable_edge_collision", False)))
    if hasattr(sim, "set_edge_margin_scale"):
        sim.set_edge_margin_scale(float(cfg.get("edge_margin_scale", 1.0)))
    if hasattr(sim, "set_edge_margin_offset"):
        sim.set_edge_margin_offset(float(cfg.get("edge_margin_offset", 0.0)))
    if hasattr(sim, "set_stiffness_all"):
        sim.set_stiffness_all(
            float(cfg.get("tension_stiffness", 1000.0)),
            float(cfg.get("compression_stiffness", 100.0)),
            float(cfg.get("shear_stiffness", 100.0)),
            float(cfg.get("bending_stiffness", 10.0)),
        )
    if hasattr(sim, "set_sewing_stiffness"):
        sim.set_sewing_stiffness(float(cfg.get("sewing_stiffness", 10000.0)))
    if hasattr(sim, "set_enable_sewing_lock"):
        sim.set_enable_sewing_lock(bool(cfg.get("enable_sewing_lock", True)))
    if hasattr(sim, "set_sewing_priority_options"):
        try:
            sim.set_sewing_priority_options(
                bool(cfg.get("sewing_priority_enabled", False)),
                float(cfg.get("sewing_priority_threshold", 0.9)),
                float(cfg.get("sewing_priority_merge_dist", 0.005)),
                int(cfg.get("sewing_priority_ramp_frames", 3)),
                int(cfg.get("sewing_priority_max_frames", 600)),
            )
        except (TypeError, ValueError):
            pass


# オブジェクト名 -> 最終適用署名 (毎フレーム差分スキップ用)
_last_applied_signatures: Dict[str, tuple] = {}


def purge_applied_signature(obj_name: Optional[str] = None) -> None:
    """シミュレータ破棄時に適用済み署名を消去する (次回syncで必ず再適用)。"""
    if obj_name is None:
        _last_applied_signatures.clear()
    else:
        _last_applied_signatures.pop(obj_name, None)


def is_config_current(obj_name: str, sig: tuple) -> bool:
    return _last_applied_signatures.get(obj_name) == sig


def mark_config_applied(obj_name: str, sig: tuple) -> None:
    _last_applied_signatures[obj_name] = sig


# --- デバッグ記録オプション (2階層ロギング) ---

RECORDING_OPTION_DEFAULTS = {
    "full_stride": 1,
    "ring_size": 0,
    "lookahead": 0,
    "enable_triggers": False,
}

#: プリファレンス属性名 -> 記録オプション名
RECORDING_PREF_MAP = (
    ("debug_sparse_stride", "full_stride"),
    ("debug_sparse_ring", "ring_size"),
    ("debug_sparse_lookahead", "lookahead"),
    ("debug_sparse_triggers", "enable_triggers"),
)


def recording_options_from_prefs(prefs: Any) -> Dict[str, Any]:
    """アドオンプリファレンスから記録オプション辞書を組み立てる。"""
    opts = dict(RECORDING_OPTION_DEFAULTS)
    if prefs is None:
        return opts
    for pref_name, opt_name in RECORDING_PREF_MAP:
        try:
            v = getattr(prefs, pref_name, opts[opt_name])
        except Exception:
            continue
        default = RECORDING_OPTION_DEFAULTS[opt_name]
        if isinstance(default, bool):
            opts[opt_name] = bool(v)
        else:
            try:
                opts[opt_name] = max(0, int(v))
            except (TypeError, ValueError):
                pass
    opts["full_stride"] = max(1, opts["full_stride"])
    return opts


def apply_recording_options(sim: Any, opts: Dict[str, Any]) -> bool:
    """記録オプション辞書をシミュレータに適用する。適用したらTrue。

    旧バイナリ (set_debug_recording_options欠如) では何もせずFalseを返す。
    """
    setter = getattr(sim, "set_debug_recording_options", None)
    if not callable(setter):
        return False
    try:
        setter(
            full_stride=max(1, int(opts.get("full_stride", 1))),
            ring_size=max(0, int(opts.get("ring_size", 0))),
            lookahead=max(0, int(opts.get("lookahead", 0))),
            enable_triggers=bool(opts.get("enable_triggers", False)),
        )
        return True
    except Exception:
        return False
