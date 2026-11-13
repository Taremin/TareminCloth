//! 2D平面幾何ユーティリティ（パターンリメッシュ・Quadリメッシュ共通）

/// 2D多角形の符号付き面積を算出（反時計回りで正）
pub fn polygon_signed_area_2d(pts: &[[f32; 2]]) -> f32 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut area2 = 0.0_f64;
    for i in 0..n {
        let p0 = pts[i];
        let p1 = pts[(i + 1) % n];
        area2 += (p0[0] as f64) * (p1[1] as f64) - (p1[0] as f64) * (p0[1] as f64);
    }
    (0.5 * area2) as f32
}

/// 2D三角形の符号付き面積を算出（反時計回りで正）
#[inline]
pub fn triangle_signed_area_2d(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2]) -> f32 {
    0.5 * cross_2d(p0, p1, p2)
}

/// 2Dベクトル外積 `(b - a) x (c - a)` を算出
#[inline]
pub fn cross_2d(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// 2D三角形の最小内角（ラジアン）を算出
pub fn triangle_min_angle_rad(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2]) -> f32 {
    let a_sq = (p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2);
    let b_sq = (p2[0] - p0[0]).powi(2) + (p2[1] - p0[1]).powi(2);
    let c_sq = (p0[0] - p1[0]).powi(2) + (p0[1] - p1[1]).powi(2);
    let a = a_sq.sqrt();
    let b = b_sq.sqrt();
    let c = c_sq.sqrt();
    if a <= 1e-12 || b <= 1e-12 || c <= 1e-12 {
        return 0.0;
    }
    let cos_a = ((b_sq + c_sq - a_sq) / (2.0 * b * c)).clamp(-1.0, 1.0);
    let cos_b = ((c_sq + a_sq - b_sq) / (2.0 * c * a)).clamp(-1.0, 1.0);
    let cos_c = ((a_sq + b_sq - c_sq) / (2.0 * a * b)).clamp(-1.0, 1.0);
    cos_a.acos().min(cos_b.acos()).min(cos_c.acos())
}

/// 点 `p` に対する多角形の回転数 (Winding Number) を算出
pub fn winding_number_2d(p: [f32; 2], polygon: &[[f32; 2]]) -> i32 {
    let n = polygon.len();
    if n < 3 {
        return 0;
    }
    let mut wn = 0_i32;
    for i in 0..n {
        let a = polygon[i];
        let b = polygon[(i + 1) % n];
        if a[1] <= p[1] {
            if b[1] > p[1] && cross_2d(a, b, p) > 0.0 {
                wn += 1;
            }
        } else if b[1] <= p[1] && cross_2d(a, b, p) < 0.0 {
            wn -= 1;
        }
    }
    wn
}

/// 線分 `[a, b]` 上で点 `p` に最も近い点とパラメータ `t in [0, 1]` を算出
#[inline]
pub fn closest_point_on_segment_2d(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    let ab_x = b[0] - a[0];
    let ab_y = b[1] - a[1];
    let len_sq = ab_x * ab_x + ab_y * ab_y;
    if len_sq <= 1e-20 {
        return a;
    }
    let t = (((p[0] - a[0]) * ab_x + (p[1] - a[1]) * ab_y) / len_sq).clamp(0.0, 1.0);
    [a[0] + ab_x * t, a[1] + ab_y * t]
}

/// 点 `p` から線分 `[a, b]` までの距離の2乗を算出
#[inline]
pub fn point_to_segment_dist_sq(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let q = closest_point_on_segment_2d(p, a, b);
    let dx = p[0] - q[0];
    let dy = p[1] - q[1];
    dx * dx + dy * dy
}

/// 点 `p` が許容誤差 `eps` 以内で線分 `[a, b]` 上に乗っているか判定
#[inline]
pub fn is_point_on_segment_2d(p: [f32; 2], a: [f32; 2], b: [f32; 2], eps: f32) -> bool {
    point_to_segment_dist_sq(p, a, b) <= eps * eps
}

/// 2つの線分 `[a0, a1]` と `[b0, b1]` が交差（接触・重なり含む）するか判定
pub fn segments_intersect_2d(
    a0: [f32; 2],
    a1: [f32; 2],
    b0: [f32; 2],
    b1: [f32; 2],
    eps: f32,
) -> bool {
    // 軸平行バウンディングボックスによる早期棄却
    let a_min_x = a0[0].min(a1[0]) - eps;
    let a_max_x = a0[0].max(a1[0]) + eps;
    let a_min_y = a0[1].min(a1[1]) - eps;
    let a_max_y = a0[1].max(a1[1]) + eps;
    let b_min_x = b0[0].min(b1[0]) - eps;
    let b_max_x = b0[0].max(b1[0]) + eps;
    let b_min_y = b0[1].min(b1[1]) - eps;
    let b_max_y = b0[1].max(b1[1]) + eps;
    if a_max_x < b_min_x || b_max_x < a_min_x || a_max_y < b_min_y || b_max_y < a_min_y {
        return false;
    }

    if is_point_on_segment_2d(a0, b0, b1, eps)
        || is_point_on_segment_2d(a1, b0, b1, eps)
        || is_point_on_segment_2d(b0, a0, a1, eps)
        || is_point_on_segment_2d(b1, a0, a1, eps)
    {
        return true;
    }

    let d1 = cross_2d(a0, a1, b0);
    let d2 = cross_2d(a0, a1, b1);
    let d3 = cross_2d(b0, b1, a0);
    let d4 = cross_2d(b0, b1, a1);

    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

/// 2つの線分 `[a0, a1]` と `[b0, b1]` が内部で真に交差するか判定（端点の単純接続は除外）
pub fn segments_properly_intersect_2d(
    a0: [f32; 2],
    a1: [f32; 2],
    b0: [f32; 2],
    b1: [f32; 2],
    endpoint_eps: f32,
) -> bool {
    let eps_sq = endpoint_eps * endpoint_eps;
    let same = |p: [f32; 2], q: [f32; 2]| -> bool {
        let dx = p[0] - q[0];
        let dy = p[1] - q[1];
        dx * dx + dy * dy <= eps_sq
    };
    // 端点を共有している場合の共線重なりチェックは簡略化し、隣接辺同士の交差誤爆を防ぐ
    let shares_endpoint =
        same(a0, b0) || same(a0, b1) || same(a1, b0) || same(a1, b1);

    let d1 = cross_2d(a0, a1, b0);
    let d2 = cross_2d(a0, a1, b1);
    let d3 = cross_2d(b0, b1, a0);
    let d4 = cross_2d(b0, b1, a1);

    let straddles_a = (d1 > endpoint_eps && d2 < -endpoint_eps)
        || (d1 < -endpoint_eps && d2 > endpoint_eps);
    let straddles_b = (d3 > endpoint_eps && d4 < -endpoint_eps)
        || (d3 < -endpoint_eps && d4 > endpoint_eps);
    if straddles_a && straddles_b {
        return true;
    }

    if shares_endpoint {
        return false;
    }

    // 端点を共有しない場合の T 字接触・重なりを検出
    is_point_on_segment_2d(a0, b0, b1, endpoint_eps)
        || is_point_on_segment_2d(a1, b0, b1, endpoint_eps)
        || is_point_on_segment_2d(b0, a0, a1, endpoint_eps)
        || is_point_on_segment_2d(b1, a0, a1, endpoint_eps)
}

/// 点 `p` が外周ループ内かつ全穴ループ外にあるか判定（境界上も外部扱いしない）
pub fn is_point_in_polygon_with_holes(
    p: [f32; 2],
    outer: &[[f32; 2]],
    holes: &[Vec<[f32; 2]>],
) -> bool {
    if winding_number_2d(p, outer) == 0 {
        return false;
    }
    for hole in holes {
        if hole.len() >= 3 && winding_number_2d(p, hole) != 0 {
            return false;
        }
    }
    true
}

/// 点 `p` が外周・穴の境界から `boundary_eps` 以上離れた厳密内部にあるか判定
pub fn is_point_strictly_inside_polygon_with_holes(
    p: [f32; 2],
    outer: &[[f32; 2]],
    holes: &[Vec<[f32; 2]>],
    boundary_eps: f32,
) -> bool {
    if !is_point_in_polygon_with_holes(p, outer, holes) {
        return false;
    }
    point_to_polygon_boundary_dist_sq(p, outer, holes) > boundary_eps * boundary_eps
}

/// 点 `p` から外周および穴の全境界辺までの最短距離の2乗を算出
pub fn point_to_polygon_boundary_dist_sq(
    p: [f32; 2],
    outer: &[[f32; 2]],
    holes: &[Vec<[f32; 2]>],
) -> f32 {
    let mut best = f32::INFINITY;
    for i in 0..outer.len() {
        let next = (i + 1) % outer.len();
        best = best.min(point_to_segment_dist_sq(p, outer[i], outer[next]));
    }
    for hole in holes {
        for i in 0..hole.len() {
            let next = (i + 1) % hole.len();
            best = best.min(point_to_segment_dist_sq(p, hole[i], hole[next]));
        }
    }
    best
}

/// 4点 `(p0, p1, p2, p3)` が厳密な凸四角形（CW/CCW 問わず全角 < 180°・非退化）か判定
pub fn is_strictly_convex_quad(
    p0: [f32; 2],
    p1: [f32; 2],
    p2: [f32; 2],
    p3: [f32; 2],
    area_eps: f32,
) -> bool {
    let c0 = cross_2d(p0, p1, p2);
    let c1 = cross_2d(p1, p2, p3);
    let c2 = cross_2d(p2, p3, p0);
    let c3 = cross_2d(p3, p0, p1);
    let all_pos = c0 > area_eps && c1 > area_eps && c2 > area_eps && c3 > area_eps;
    let all_neg = c0 < -area_eps && c1 < -area_eps && c2 < -area_eps && c3 < -area_eps;
    all_pos || all_neg
}

/// 4点 `(p0, p1, p2, p3)` が反時計回りかつ厳密な凸四角形か判定
pub fn is_strictly_convex_ccw_quad(
    p0: [f32; 2],
    p1: [f32; 2],
    p2: [f32; 2],
    p3: [f32; 2],
    area_eps: f32,
) -> bool {
    let c0 = cross_2d(p0, p1, p2);
    let c1 = cross_2d(p1, p2, p3);
    let c2 = cross_2d(p2, p3, p0);
    let c3 = cross_2d(p3, p0, p1);
    c0 > area_eps && c1 > area_eps && c2 > area_eps && c3 > area_eps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convex_quad_and_concave_quad() {
        assert!(is_strictly_convex_quad(
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            1e-6,
        ));
        // 矢じり型凹四角形
        assert!(!is_strictly_convex_quad(
            [0.0, 0.0],
            [2.0, 0.0],
            [0.6, 0.6],
            [0.0, 2.0],
            1e-6,
        ));
    }

    #[test]
    fn test_polygon_with_holes_containment() {
        let outer = vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let holes = vec![vec![[1.0, 1.0], [1.0, 3.0], [3.0, 3.0], [3.0, 1.0]]];
        assert!(is_point_in_polygon_with_holes([0.5, 0.5], &outer, &holes));
        assert!(!is_point_in_polygon_with_holes([2.0, 2.0], &outer, &holes));
        assert!(!is_point_in_polygon_with_holes([-0.5, 2.0], &outer, &holes));
    }

    #[test]
    fn test_segments_intersect() {
        assert!(segments_intersect_2d([0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0], 1e-6));
        assert!(!segments_properly_intersect_2d(
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            1e-6,
        ));
    }
}
