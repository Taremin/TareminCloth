"""GPUプロファイル集計ヘルパ (bpy非依存)。

`take_profile` の平坦列は親子二重計上を含むため、単純合計では
フレーム時間を過大評価する。本モジュールは包含・排他の区別を
一元化し、runner / log_tools / テストから共用する。
"""

from collections import defaultdict
from typing import Dict, List, Tuple

# 親スコープ名の接頭辞。子合計と重複するため包含扱いにする。
PARENT_PREFIXES = ("self_collision_",)

# 包含スコープ（子と重なるため合計から除外すべき親）の判定。
def is_parent_scope(label: str) -> bool:
    return label.startswith(PARENT_PREFIXES)


def aggregate_profile(
    entries: List[Tuple[str, float]],
) -> Tuple[Dict[str, float], Dict[str, int], float, float]:
    """平坦列を集計し (合計, 件数, 正味合計, 平坦合計) を返す。

    正味合計は親スコープを除外した非重複の推定フレーム時間。
    平坦合計は親子二重計上の参考値。
    """
    totals: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for name, ms in entries:
        totals[name] += float(ms)
        counts[name] += 1
    flat_total = float(sum(totals.values()))
    net_total = float(sum(v for k, v in totals.items() if not is_parent_scope(k)))
    return dict(totals), dict(counts), net_total, flat_total


def format_profile_table(entries: List[Tuple[str, float]]) -> str:
    """集計表を文字列化する。空列時はその旨を返す。"""
    if not entries:
        return "profile: no entries (disabled or unsupported)"
    totals, counts, net_total, flat_total = aggregate_profile(entries)
    lines = []
    lines.append(f"{'label':24s} {'total_ms':>9s} {'count':>5s} {'avg_ms':>9s} {'share':>6s}")
    for name in sorted(totals, key=lambda n: -totals[n]):
        share = totals[name] / net_total * 100.0 if net_total > 0 else 0.0
        marker = " (parent)" if is_parent_scope(name) else ""
        lines.append(
            f"{name:24s} {totals[name]:9.3f} {counts[name]:5d}"
            f" {totals[name] / counts[name]:9.4f} {share:5.1f}%{marker}"
        )
    lines.append(f"net ms (excl. parents): {net_total:.3f} / flat ms: {flat_total:.3f}")
    return "\n".join(lines)
