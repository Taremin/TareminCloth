# -*- coding: utf-8 -*-
"""2D多角形幾何計算・符号付き面積・内外判定・自己交差検査モジュール (Blender非依存)."""

import math
from typing import Sequence, Tuple

Vec2 = Tuple[float, float]


def signed_area(points: Sequence[Vec2]) -> float:
    """Shoelace公式による2D多角形の符号付き面積 (反時計回り CCW > 0)."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1] - points[j][0] * points[i][1]
    return 0.5 * area


def ensure_ccw(points: Sequence[Vec2]) -> list[Vec2]:
    """多角形点列を反時計回り (CCW) に正規化した新しいリストを返す."""
    pts = list(points)
    if signed_area(pts) < 0.0:
        pts.reverse()
    return pts


def point_in_polygon(
    point: Vec2,
    polygon: Sequence[Vec2],
) -> bool:
    """Winding Number 法による2D点・多角形の内外判定 (境界上または内部なら True)."""
    n = len(polygon)
    if n < 3:
        return False

    px, py = float(point[0]), float(point[1])
    wn = 0

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # 頂点または線分上の厳密判定
        dx = x2 - x1
        dy = y2 - y1
        d_px = px - x1
        d_py = py - y1
        cross = dx * d_py - dy * d_px
        dot = d_px * dx + d_py * dy
        seg_len_sq = dx * dx + dy * dy

        if abs(cross) < 1e-9 and 0.0 <= dot <= seg_len_sq:
            return True  # 境界線上

        if y1 <= py:
            if y2 > py and cross > 0.0:
                wn += 1
        else:
            if y2 <= py and cross < 0.0:
                wn -= 1

    return wn != 0


def _segments_intersect(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> bool:  # audit: allow-positional
    """2線分 p1-p2 と p3-p4 が真に交差（端点共有を除く）するかを判定する."""
    def ccw(a: Vec2, b: Vec2, c: Vec2) -> float:  # audit: allow-positional
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)

    if ((d1 > 1e-9 and d2 < -1e-9) or (d1 < -1e-9 and d2 > 1e-9)) and \
       ((d3 > 1e-9 and d4 < -1e-9) or (d3 < -1e-9 and d4 > 1e-9)):
        return True

    return False


def has_self_intersection(
    points: Sequence[Vec2],
    *,
    closed: bool = True,
) -> bool:
    """点列で構成される折れ線または閉多角形に自己交差が存在するかを判定する."""
    n = len(points)
    if n < 4:
        return False

    num_segs = n if closed else n - 1

    for i in range(num_segs):
        p1 = points[i]
        p2 = points[(i + 1) % n]

        # 隣接辺および自身を除外して検査
        j_start = i + 2
        j_end = num_segs
        if closed and i == 0:
            j_end = num_segs - 1  # 最終辺 (n-1 -> 0) は辺0 (0 -> 1) と隣接

        for j in range(j_start, j_end):
            p3 = points[j]
            p4 = points[(j + 1) % n]
            if _segments_intersect(p1, p2, p3, p4):
                return True

    return False
