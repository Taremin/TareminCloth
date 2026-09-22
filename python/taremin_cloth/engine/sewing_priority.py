"""縫合優先モードの表示・検証用ヘルパー。

重力ゲーティングの真実源は Rust コア
(`GpuClothSimulator::update_sewing_priority_from_positions`) にあり、
本モジュールは追加のGPU同期なしに済む用途に限定する:

- Nパネル進捗表示用の結合率計算 (既知の ``coords`` から算出)
- Blender非依存の単体テスト用純粋関数
"""

import numpy as np


def closure_ratio(positions_2d, sewing_edges, merge_dist: float) -> float:
    """縫合ペアの結合率 (0.0〜1.0) を返す。縫合なし・空配列は 1.0。"""
    if sewing_edges is None or len(sewing_edges) == 0:
        return 1.0
    pos = np.asarray(positions_2d, dtype=np.float64).reshape(-1, 3)
    pairs = np.asarray(sewing_edges, dtype=np.int64).reshape(-1, 2)
    n_verts = pos.shape[0]
    valid = (pairs[:, 0] >= 0) & (pairs[:, 0] < n_verts) & (pairs[:, 1] >= 0) & (pairs[:, 1] < n_verts)
    pairs = pairs[valid]
    if len(pairs) == 0:
        return 1.0
    diff = pos[pairs[:, 0]] - pos[pairs[:, 1]]
    dist_sq = np.sum(diff * diff, axis=1)
    closed = np.count_nonzero(dist_sq <= float(merge_dist) ** 2)
    return float(closed) / float(len(pairs))


def default_merge_dist(thickness: float) -> float:
    """結合判定距離の既定値。縫合ロック閾値と整合させる。"""
    return max(float(thickness) * 2.0, 0.005)


def sim_phase(sim):
    """simの縫合優先フェーズ状態を返す (active, ratio, scale)。GPU同期なし。

    旧バイナリ等でAPI欠落時は None を返す。
    """
    try:
        active = bool(sim.is_sewing_priority_active())
        ratio = float(sim.get_sewing_closure_ratio())
        scale = float(sim.get_sewing_priority_scale())
    except Exception:
        return None
    return (active, ratio, scale)


def phase_text(active: bool, ratio: float, scale: float, trans) -> str:
    """フェーズ表示文を組み立てる (Nパネル・HUD共通書式)。trans: i18n.trans相当。"""
    if active:
        return f"{trans('Phase: Sewing')} {float(ratio) * 100.0:.0f}% (gravity {float(scale):.2f}x)"
    return f"{trans('Phase: Normal')} (gravity {float(scale):.2f}x)"
