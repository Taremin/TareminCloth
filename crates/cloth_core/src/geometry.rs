//! 検証・監査用CPU幾何判定 (Blender非依存・GPU不要)。
//!
//! Python `analysis.py` を正本の式から移植した決定的参照実装。
//! `log_tools check-intersections / watch --stop-on intersections` の
//! 毎フレーム呼び出しにおけるPython三重ループを置き換える。

use std::collections::{HashMap, HashSet};

const EPS: f64 = 1e-5;

type V3 = [f64; 3];

#[inline]
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn orient2(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn edge_cross_2d(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let o1 = orient2(a, b, c);
    let o2 = orient2(a, b, d);
    let o3 = orient2(c, d, a);
    let o4 = orient2(c, d, b);
    (o1 * o2 < -EPS * EPS) && (o3 * o4 < -EPS * EPS)
}

fn point_in_tri2(p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    let o1 = orient2(a, b, p);
    let o2 = orient2(b, c, p);
    let o3 = orient2(c, a, p);
    (o1 > EPS && o2 > EPS && o3 > EPS) || (o1 < -EPS && o2 < -EPS && o3 < -EPS)
}

fn coplanar_tri_tri_2d(p0: V3, p1: V3, p2: V3, q0: V3, q1: V3, q2: V3, ax0: usize, ax1: usize) -> bool {
    let pr = |v: V3| [v[ax0], v[ax1]];
    let (p0, p1, p2) = (pr(p0), pr(p1), pr(p2));
    let (q0, q1, q2) = (pr(q0), pr(q1), pr(q2));
    let edges_p = [(p0, p1), (p1, p2), (p2, p0)];
    let edges_q = [(q0, q1), (q1, q2), (q2, q0)];
    for &(ep0, ep1) in &edges_p {
        for &(eq0, eq1) in &edges_q {
            if edge_cross_2d(ep0, ep1, eq0, eq1) {
                return true;
            }
        }
    }
    for &q in &[q0, q1, q2] {
        if point_in_tri2(q, p0, p1, p2) {
            return true;
        }
    }
    for &p in &[p0, p1, p2] {
        if point_in_tri2(p, q0, q1, q2) {
            return true;
        }
    }
    false
}

/// Möller (1997) 高速三角形交差判定。Python `tri_tri_intersection_moller` と同一式。
pub fn tri_tri_intersect(
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    u0: [f32; 3],
    u1: [f32; 3],
    u2: [f32; 3],
) -> bool {
    let v0 = [v0[0] as f64, v0[1] as f64, v0[2] as f64];
    let v1 = [v1[0] as f64, v1[1] as f64, v1[2] as f64];
    let v2 = [v2[0] as f64, v2[1] as f64, v2[2] as f64];
    let u0 = [u0[0] as f64, u0[1] as f64, u0[2] as f64];
    let u1 = [u1[0] as f64, u1[1] as f64, u1[2] as f64];
    let u2 = [u2[0] as f64, u2[1] as f64, u2[2] as f64];

    let e1 = sub(v1, v0);
    let e2 = sub(v2, v0);
    let n1 = cross(e1, e2);
    let d1 = -dot(n1, v0);
    let mut du = [dot(n1, u0) + d1, dot(n1, u1) + d1, dot(n1, u2) + d1];
    for d in du.iter_mut() {
        if d.abs() < EPS {
            *d = 0.0;
        }
    }
    if du[0] * du[1] > 0.0 && du[0] * du[2] > 0.0 {
        return false;
    }

    let e1u = sub(u1, u0);
    let e2u = sub(u2, u0);
    let n2 = cross(e1u, e2u);
    let d2 = -dot(n2, u0);
    let mut dv = [dot(n2, v0) + d2, dot(n2, v1) + d2, dot(n2, v2) + d2];
    for d in dv.iter_mut() {
        if d.abs() < EPS {
            *d = 0.0;
        }
    }
    if dv[0] * dv[1] > 0.0 && dv[0] * dv[2] > 0.0 {
        return false;
    }

    let d = cross(n1, n2);
    if dot(d, d) < 1e-10 {
        let abs_n = [n1[0].abs(), n1[1].abs(), n1[2].abs()];
        let max_idx = if abs_n[0] >= abs_n[1] && abs_n[0] >= abs_n[2] {
            0
        } else if abs_n[1] >= abs_n[2] {
            1
        } else {
            2
        };
        let axes: Vec<usize> = (0..3).filter(|&i| i != max_idx).collect();
        return coplanar_tri_tri_2d(v0, v1, v2, u0, u1, u2, axes[0], axes[1]);
    }

    let ad = [d[0].abs(), d[1].abs(), d[2].abs()];
    let max_idx = if ad[0] >= ad[1] && ad[0] >= ad[2] {
        0
    } else if ad[1] >= ad[2] {
        1
    } else {
        2
    };
    let vp = [v0[max_idx], v1[max_idx], v2[max_idx]];
    let up = [u0[max_idx], u1[max_idx], u2[max_idx]];
    fn intervals(p0: f64, p1: f64, p2: f64, d0: f64, d1: f64, d2: f64) -> (f64, f64) {
        let d0d1 = d0 * d1;
        let d0d2 = d0 * d2;
        if d0d1 > 0.0 {
            (
                p2 + (p0 - p2) * (d2 / (d2 - d0)),
                p2 + (p1 - p2) * (d2 / (d1 - d2)),
            )
        } else if d0d2 > 0.0 {
            (
                p1 + (p0 - p1) * (d1 / (d1 - d0)),
                p1 + (p2 - p1) * (d1 / (d2 - d1)),
            )
        } else {
            (
                p0 + (p1 - p0) * (d0 / (d0 - d1)),
                p0 + (p2 - p0) * (d0 / (d0 - d2)),
            )
        }
    }
    let (a0, a1) = intervals(vp[0], vp[1], vp[2], dv[0], dv[1], dv[2]);
    let (b0, b1) = intervals(up[0], up[1], up[2], du[0], du[1], du[2]);
    let (t1min, t1max) = (a0.min(a1), a0.max(a1));
    let (t2min, t2max) = (b0.min(b1), b0.max(b1));
    !(t1max < t2min - EPS || t2max < t1min - EPS)
}

/// 自己交差三角形ペア検出。Python `find_triangle_intersections` と同一式。
/// 戻り値は `(f0, f1)`（`f0 < f1`）の昇順ソート列。
pub fn find_triangle_intersections(
    positions: &[[f32; 3]],
    faces: &[[u32; 3]],
    ignore_adjacent: bool,
    cell_size: Option<f32>,
) -> Vec<[u32; 2]> {
    if faces.is_empty() {
        return Vec::new();
    }
    let n = positions.len();
    let pos64: Vec<V3> = positions
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();
    // 有効面のみ残す（範囲外・退化面は除外）。元の面番号を保持する。
    let mut valid_faces: Vec<(u32, [u32; 3])> = Vec::with_capacity(faces.len());
    for (fi, &[a, b, c]) in faces.iter().enumerate() {
        let (ia, ib, ic) = (a as usize, b as usize, c as usize);
        if a == b || b == c || c == a || ia >= n || ib >= n || ic >= n {
            continue;
        }
        valid_faces.push((fi as u32, [a, b, c]));
    }
    if valid_faces.is_empty() {
        return Vec::new();
    }
    let m = valid_faces.len();
    let mut mins = vec![[0.0f64; 3]; m];
    let mut maxs = vec![[0.0f64; 3]; m];
    let mut tris = vec![[[0.0f64; 3]; 3]; m];
    let mut edge_lens = Vec::with_capacity(m);
    for (k, &(_, [a, b, c])) in valid_faces.iter().enumerate() {
        let t = [pos64[a as usize], pos64[b as usize], pos64[c as usize]];
        tris[k] = t;
        let mut mn = t[0];
        let mut mx = t[0];
        for v in t.iter().skip(1) {
            for ax in 0..3 {
                if v[ax] < mn[ax] {
                    mn[ax] = v[ax];
                }
                if v[ax] > mx[ax] {
                    mx[ax] = v[ax];
                }
            }
        }
        mins[k] = mn;
        maxs[k] = mx;
        let dx = t[1][0] - t[0][0];
        let dy = t[1][1] - t[0][1];
        let dz = t[1][2] - t[0][2];
        edge_lens.push((dx * dx + dy * dy + dz * dz).sqrt());
    }
    let cell = match cell_size {
        Some(c) if c > 0.0 => c as f64,
        _ => {
            let mean = if edge_lens.is_empty() {
                0.05
            } else {
                edge_lens.iter().sum::<f64>() / edge_lens.len() as f64
            };
            (mean * 2.0).max(1e-4)
        }
    };
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (k, (mn, mx)) in mins.iter().zip(maxs.iter()).enumerate() {
        let cmin = [
            (mn[0] / cell).floor() as i64,
            (mn[1] / cell).floor() as i64,
            (mn[2] / cell).floor() as i64,
        ];
        let cmax = [
            (mx[0] / cell).floor() as i64,
            (mx[1] / cell).floor() as i64,
            (mx[2] / cell).floor() as i64,
        ];
        for cx in cmin[0]..=cmax[0] {
            for cy in cmin[1]..=cmax[1] {
                for cz in cmin[2]..=cmax[2] {
                    grid.entry((cx, cy, cz)).or_default().push(k);
                }
            }
        }
    }
    let mut tested: HashSet<(u32, u32)> = HashSet::new();
    let mut out: Vec<[u32; 2]> = Vec::new();
    for cell_faces in grid.values() {
        if cell_faces.len() < 2 {
            continue;
        }
        for i in 0..cell_faces.len() {
            for j in (i + 1)..cell_faces.len() {
                let ka = cell_faces[i];
                let kb = cell_faces[j];
                let (fa, _) = valid_faces[ka];
                let (fb, _) = valid_faces[kb];
                let pair = if fa < fb { (fa, fb) } else { (fb, fa) };
                if !tested.insert(pair) {
                    continue;
                }
                if ignore_adjacent {
                    let (_, ea) = valid_faces[ka];
                    let (_, eb) = valid_faces[kb];
                    if ea[0] == eb[0]
                        || ea[0] == eb[1]
                        || ea[0] == eb[2]
                        || ea[1] == eb[0]
                        || ea[1] == eb[1]
                        || ea[1] == eb[2]
                        || ea[2] == eb[0]
                        || ea[2] == eb[1]
                        || ea[2] == eb[2]
                    {
                        continue;
                    }
                }
                // AABB重複判定
                let separated = (0..3).any(|ax| mins[ka][ax] > maxs[kb][ax])
                    || (0..3).any(|ax| mins[kb][ax] > maxs[ka][ax]);
                if separated {
                    continue;
                }
                let t0 = tris[ka];
                let t1 = tris[kb];
                let p = [
                    [t0[0][0] as f32, t0[0][1] as f32, t0[0][2] as f32],
                    [t0[1][0] as f32, t0[1][1] as f32, t0[1][2] as f32],
                    [t0[2][0] as f32, t0[2][1] as f32, t0[2][2] as f32],
                ];
                let q = [
                    [t1[0][0] as f32, t1[0][1] as f32, t1[0][2] as f32],
                    [t1[1][0] as f32, t1[1][1] as f32, t1[1][2] as f32],
                    [t1[2][0] as f32, t1[2][1] as f32, t1[2][2] as f32],
                ];
                if tri_tri_intersect(p[0], p[1], p[2], q[0], q[1], q[2]) {
                    out.push([pair.0, pair.1]);
                }
            }
        }
    }
    out.sort();
    out
}

/// 厚み未満に接近した頂点-面ペア検出。Python `find_proximity_violations` と同一式。
/// 簡易重心判定であり、戻り値は `(vertex_idx, face_idx, distance)`。
pub fn find_proximity_violations(
    positions: &[[f32; 3]],
    faces: &[[u32; 3]],
    thickness: f32,
    ignore_adjacent: bool,
) -> Vec<(u32, u32, f32)> {
    let n = positions.len();
    let pos64: Vec<V3> = positions
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();
    let valid: Vec<(u32, [u32; 3])> = faces
        .iter()
        .enumerate()
        .filter_map(|(fi, &[a, b, c])| {
            let (ia, ib, ic) = (a as usize, b as usize, c as usize);
            if a == b || b == c || c == a || ia >= n || ib >= n || ic >= n {
                None
            } else {
                Some((fi as u32, [a, b, c]))
            }
        })
        .collect();
    if valid.is_empty() || n == 0 {
        return Vec::new();
    }
    let centers: Vec<V3> = valid
        .iter()
        .map(|&(_, [a, b, c])| {
            let (pa, pb, pc) = (pos64[a as usize], pos64[b as usize], pos64[c as usize]);
            [
                (pa[0] + pb[0] + pc[0]) / 3.0,
                (pa[1] + pb[1] + pc[1]) / 3.0,
                (pa[2] + pb[2] + pc[2]) / 3.0,
            ]
        })
        .collect();
    let th = thickness as f64;
    let mut out = Vec::new();
    for (vi, p) in pos64.iter().enumerate() {
        for (k, &(_, f)) in valid.iter().enumerate() {
            let c = centers[k];
            let dx = c[0] - p[0];
            let dy = c[1] - p[1];
            let dz = c[2] - p[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < th {
                if ignore_adjacent
                    && (vi as u32 == f[0] || vi as u32 == f[1] || vi as u32 == f[2])
                {
                    continue;
                }
                out.push((vi as u32, valid[k].0, dist as f32));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_crossing_pair_intersects() {
        // XY平面の三角形を貫く垂直三角形 (Python参照実装でTrueを確認済み)
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let u0 = [0.2, -1.0, -1.0];
        let u1 = [0.2, 2.0, -1.0];
        let u2 = [0.2, 0.5, 2.0];
        assert!(tri_tri_intersect(v0, v1, v2, u0, u1, u2));
    }

    #[test]
    fn separated_pairs_do_not_intersect() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let u0 = [0.0, 0.0, 5.0];
        let u1 = [1.0, 0.0, 5.0];
        let u2 = [0.0, 1.0, 5.0];
        assert!(!tri_tri_intersect(v0, v1, v2, u0, u1, u2));
    }

    #[test]
    fn coplanar_overlap_intersects() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [2.0, 0.0, 0.0];
        let v2 = [0.0, 2.0, 0.0];
        let u0 = [0.5, 0.5, 0.0];
        let u1 = [1.5, 0.5, 0.0];
        let u2 = [0.5, 1.5, 0.0];
        assert!(tri_tri_intersect(v0, v1, v2, u0, u1, u2));
    }

    #[test]
    fn adjacent_pair_skipped_when_requested() {
        let pos = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let faces = vec![[0, 1, 2], [1, 3, 2]];
        let out = find_triangle_intersections(&pos, &faces, true, None);
        assert!(out.is_empty());
    }

    #[test]
    fn proximity_detects_close_vertex() {
        let pos = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.33, 0.33, 0.001],
        ];
        let faces = vec![[0, 1, 2]];
        let out = find_proximity_violations(&pos, &faces, 0.005, true);
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].0, out[0].1), (3, 0));
    }
}
