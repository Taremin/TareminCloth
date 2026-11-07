# -*- coding: utf-8 -*-
"""2D多角形幾何計算・符号付き面積・内外判定・自己交差検査モジュール (Blender非依存)."""

from typing import Optional, Sequence, Tuple

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

        # 頂点または線分上の厳密判定（長さゼロの縮退辺による全域誤判定を防止）
        dx = x2 - x1
        dy = y2 - y1
        d_px = px - x1
        d_py = py - y1
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq <= 1e-18:
            if d_px * d_px + d_py * d_py <= 1e-18:
                return True  # 縮退頂点と一致
            continue

        cross = dx * d_py - dy * d_px
        dot = d_px * dx + d_py * dy

        if (cross * cross) <= 1e-18 * seg_len_sq and 0.0 <= dot <= seg_len_sq:
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


def segment_intersection(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> Optional[tuple[Vec2, float, float]]:  # audit: allow-positional
    """2線分 p1-p2 と p3-p4 の交点座標、および各線分上の比率 (t, u) を算出する. 交差しない場合は None を返す."""
    d1x = p2[0] - p1[0]
    d1y = p2[1] - p1[1]
    d2x = p4[0] - p3[0]
    d2y = p4[1] - p3[1]

    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-9:
        return None  # 平行または同一直線

    dx = p3[0] - p1[0]
    dy = p3[1] - p1[1]

    t = (dx * d2y - dy * d2x) / denom
    u = (dx * d1y - dy * d1x) / denom

    if -1e-6 <= t <= 1.0 + 1e-6 and -1e-6 <= u <= 1.0 + 1e-6:
        t_clamped = max(0.0, min(1.0, float(t)))
        u_clamped = max(0.0, min(1.0, float(u)))
        ix = p1[0] + t_clamped * d1x
        iy = p1[1] + t_clamped * d1y
        return ((float(ix), float(iy)), t_clamped, u_clamped)

    return None


def triangulate_polygon_2d(points: Sequence[Vec2]) -> list[tuple[int, int, int]]:
    """単純多角形（凸・凹対応）を耳刈り法（Ear-Clipping）で三角形分割し、頂点インデックス組のリストを返す."""
    n = len(points)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    # 符号付き面積が負（時計回り CW）の場合は頂点インデックス順を反転してCCWとして処理
    indices = list(range(n))
    if signed_area(points) < 0.0:
        indices.reverse()

    triangles: list[tuple[int, int, int]] = []

    def is_point_inside_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool:  # audit: allow-positional
        # 三角形 a-b-c (CCW) に対する点 p の内外判定
        cp1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        cp2 = (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0])
        cp3 = (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0])
        eps = -1e-9
        return cp1 >= eps and cp2 >= eps and cp3 >= eps

    max_iterations = n * n
    it_count = 0

    while len(indices) > 3 and it_count < max_iterations:
        it_count += 1
        m = len(indices)
        ear_found = False

        for i in range(m):
            u = indices[(i - 1) % m]
            v = indices[i]
            w = indices[(i + 1) % m]

            pu = points[u]
            pv = points[v]
            pw = points[w]

            # 凸角チェック (外積 > 0)
            cross = (pv[0] - pu[0]) * (pw[1] - pv[1]) - (pv[1] - pu[1]) * (pw[0] - pv[0])
            if cross <= 1e-9:
                continue

            # 耳の内部に他の多角形頂点が含まれていないかを検査
            has_inside = False
            for k in range(m):
                if k in ((i - 1) % m, i, (i + 1) % m):
                    continue
                pk = points[indices[k]]
                if is_point_inside_triangle(pk, pu, pv, pw):
                    has_inside = True
                    break

            if not has_inside:
                triangles.append((u, v, w))
                indices.pop(i)
                ear_found = True
                break

        if not ear_found:
            # 縮退・平坦などにより厳密な耳が見つからない場合、最初の凸っぽい頂点を切り出す
            best_idx = 0
            best_cross = -float("inf")
            for i in range(m):
                u = indices[(i - 1) % m]
                v = indices[i]
                w = indices[(i + 1) % m]
                c = (points[v][0] - points[u][0]) * (points[w][1] - points[v][1]) - (points[v][1] - points[u][1]) * (points[w][0] - points[v][0])
                if c > best_cross:
                    best_cross = c
                    best_idx = i
            u = indices[(best_idx - 1) % m]
            v = indices[best_idx]
            w = indices[(best_idx + 1) % m]
            triangles.append((u, v, w))
            indices.pop(best_idx)

    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))

    return triangles
