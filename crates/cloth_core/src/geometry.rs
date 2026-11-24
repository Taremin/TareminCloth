//! 検証・監査用CPU幾何判定 (Blender非依存・GPU不要)。
//!
//! Python `analysis.py` を正本の式から移植した決定的参照実装。
//! `log_tools check-intersections / watch --stop-on intersections` の
//! 毎フレーム呼び出しにおけるPython三重ループを置き換える。

use std::collections::{HashMap, HashSet};

const COPLANAR_DIST_EPS: f64 = 1e-6;
const PLANE_ZERO_EPS: f64 = 1e-12;

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
    let tol_ab = (b[0] - a[0]).hypot(b[1] - a[1]) * 1e-6;
    let tol_cd = (d[0] - c[0]).hypot(d[1] - c[1]) * 1e-6;
    ((o1 > tol_ab && o2 < -tol_ab) || (o1 < -tol_ab && o2 > tol_ab))
        && ((o3 > tol_cd && o4 < -tol_cd) || (o3 < -tol_cd && o4 > tol_cd))
}

fn point_in_tri2(p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    let o1 = orient2(a, b, p);
    let o2 = orient2(b, c, p);
    let o3 = orient2(c, a, p);
    let tol_ab = (b[0] - a[0]).hypot(b[1] - a[1]) * 1e-6;
    let tol_bc = (c[0] - b[0]).hypot(c[1] - b[1]) * 1e-6;
    let tol_ca = (a[0] - c[0]).hypot(a[1] - c[1]) * 1e-6;
    (o1 > tol_ab && o2 > tol_bc && o3 > tol_ca) || (o1 < -tol_ab && o2 < -tol_bc && o3 < -tol_ca)
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

/// 同一平面上で交差する2つの三角形から、可視化用の代表3D交差線分を抽出する。
fn coplanar_intersection_segment_3d(
    v0: V3,
    v1: V3,
    v2: V3,
    u0: V3,
    u1: V3,
    u2: V3,
    ax0: usize,
    ax1: usize,
) -> ([f32; 3], [f32; 3]) {
    let pr = |v: V3| [v[ax0], v[ax1]];
    let edges_v = [(v0, v1), (v1, v2), (v2, v0)];
    let edges_u = [(u0, u1), (u1, u2), (u2, u0)];
    let (pv0, pv1, pv2) = (pr(v0), pr(v1), pr(v2));
    let (pu0, pu1, pu2) = (pr(u0), pr(u1), pr(u2));

    let mut pts: Vec<V3> = Vec::with_capacity(12);
    for &(ep0, ep1) in &edges_v {
        let a = pr(ep0);
        let b = pr(ep1);
        for &(eq0, eq1) in &edges_u {
            let c = pr(eq0);
            let d = pr(eq1);
            if edge_cross_2d(a, b, c, d) {
                let r0 = b[0] - a[0];
                let r1 = b[1] - a[1];
                let s0 = d[0] - c[0];
                let s1 = d[1] - c[1];
                let denom = r0 * s1 - r1 * s0;
                if denom.abs() > 1e-15 {
                    let t = (((c[0] - a[0]) * s1 - (c[1] - a[1]) * s0) / denom).clamp(0.0, 1.0);
                    pts.push([
                        ep0[0] + t * (ep1[0] - ep0[0]),
                        ep0[1] + t * (ep1[1] - ep0[1]),
                        ep0[2] + t * (ep1[2] - ep0[2]),
                    ]);
                }
            }
        }
    }
    for &q in &[u0, u1, u2] {
        if point_in_tri2(pr(q), pv0, pv1, pv2) {
            pts.push(q);
        }
    }
    for &p in &[v0, v1, v2] {
        if point_in_tri2(pr(p), pu0, pu1, pu2) {
            pts.push(p);
        }
    }

    let to_f32 = |p: V3| [p[0] as f32, p[1] as f32, p[2] as f32];
    if pts.len() >= 2 {
        let mut best_d2 = -1.0f64;
        let mut best_a = pts[0];
        let mut best_b = pts[1];
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let dx = pts[j][0] - pts[i][0];
                let dy = pts[j][1] - pts[i][1];
                let dz = pts[j][2] - pts[i][2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 > best_d2 {
                    best_d2 = d2;
                    best_a = pts[i];
                    best_b = pts[j];
                }
            }
        }
        (to_f32(best_a), to_f32(best_b))
    } else if pts.len() == 1 {
        let p = to_f32(pts[0]);
        (p, p)
    } else {
        let cv = [
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0,
        ];
        let cu = [
            (u0[0] + u1[0] + u2[0]) / 3.0,
            (u0[1] + u1[1] + u2[1]) / 3.0,
            (u0[2] + u1[2] + u2[2]) / 3.0,
        ];
        (to_f32(cv), to_f32(cu))
    }
}

/// Möller (1997) 高速三角形交差判定。交差している場合は 3D 交差線分の両端点を返す。
/// `tri_tri_intersect` と完全に同一の幾何判定式を使用し、`is_some() == tri_tri_intersect(...)` を保証する。
pub fn tri_tri_intersection_segment(
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    u0: [f32; 3],
    u1: [f32; 3],
    u2: [f32; 3],
) -> Option<([f32; 3], [f32; 3])> {
    let v0 = [v0[0] as f64, v0[1] as f64, v0[2] as f64];
    let v1 = [v1[0] as f64, v1[1] as f64, v1[2] as f64];
    let v2 = [v2[0] as f64, v2[1] as f64, v2[2] as f64];
    let u0 = [u0[0] as f64, u0[1] as f64, u0[2] as f64];
    let u1 = [u1[0] as f64, u1[1] as f64, u1[2] as f64];
    let u2 = [u2[0] as f64, u2[1] as f64, u2[2] as f64];

    let e1 = sub(v1, v0);
    let e2 = sub(v2, v0);
    let n1_raw = cross(e1, e2);
    let n1_len = dot(n1_raw, n1_raw).sqrt();
    if n1_len <= 1e-15 {
        return None;
    }
    let n1 = [n1_raw[0] / n1_len, n1_raw[1] / n1_len, n1_raw[2] / n1_len];
    let d1 = -dot(n1, v0);
    let du_raw = [dot(n1, u0) + d1, dot(n1, u1) + d1, dot(n1, u2) + d1];

    let e1u = sub(u1, u0);
    let e2u = sub(u2, u0);
    let n2_raw = cross(e1u, e2u);
    let n2_len = dot(n2_raw, n2_raw).sqrt();
    if n2_len <= 1e-15 {
        return None;
    }
    let n2 = [n2_raw[0] / n2_len, n2_raw[1] / n2_len, n2_raw[2] / n2_len];
    let d2 = -dot(n2, u0);
    let dv_raw = [dot(n2, v0) + d2, dot(n2, v1) + d2, dot(n2, v2) + d2];

    let d = cross(n1, n2);
    if dot(d, d) < 1e-10 {
        // ほぼ平行な2平面: 全頂点が COPLANAR_DIST_EPS 以内にある場合のみ同一平面判定へ進む
        let coplanar_u = du_raw.iter().all(|x| x.abs() < COPLANAR_DIST_EPS);
        let coplanar_v = dv_raw.iter().all(|x| x.abs() < COPLANAR_DIST_EPS);
        if !(coplanar_u && coplanar_v) {
            return None;
        }
        let abs_n = [n1[0].abs(), n1[1].abs(), n1[2].abs()];
        let max_idx = if abs_n[0] >= abs_n[1] && abs_n[0] >= abs_n[2] {
            0
        } else if abs_n[1] >= abs_n[2] {
            1
        } else {
            2
        };
        let axes: Vec<usize> = (0..3).filter(|&i| i != max_idx).collect();
        if coplanar_tri_tri_2d(v0, v1, v2, u0, u1, u2, axes[0], axes[1]) {
            return Some(coplanar_intersection_segment_3d(
                v0, v1, v2, u0, u1, u2, axes[0], axes[1],
            ));
        }
        return None;
    }

    // 非平行平面: 1e-5 等の粗いスナップを行うと浅い角度で近傍非交差面を誤検出するため
    // 数値丸め下限 (PLANE_ZERO_EPS = 1e-12) 未満のみゼロとみなす。
    let mut du = du_raw;
    for x in du.iter_mut() {
        if x.abs() < PLANE_ZERO_EPS {
            *x = 0.0;
        }
    }
    if du[0] * du[1] > 0.0 && du[0] * du[2] > 0.0 {
        return None;
    }

    let mut dv = dv_raw;
    for x in dv.iter_mut() {
        if x.abs() < PLANE_ZERO_EPS {
            *x = 0.0;
        }
    }
    if dv[0] * dv[1] > 0.0 && dv[0] * dv[2] > 0.0 {
        return None;
    }

    let all_du_zero = du[0] == 0.0 && du[1] == 0.0 && du[2] == 0.0;
    let all_dv_zero = dv[0] == 0.0 && dv[1] == 0.0 && dv[2] == 0.0;
    if all_du_zero || all_dv_zero {
        let abs_n = [n1[0].abs(), n1[1].abs(), n1[2].abs()];
        let max_idx = if abs_n[0] >= abs_n[1] && abs_n[0] >= abs_n[2] {
            0
        } else if abs_n[1] >= abs_n[2] {
            1
        } else {
            2
        };
        let axes: Vec<usize> = (0..3).filter(|&i| i != max_idx).collect();
        if coplanar_tri_tri_2d(v0, v1, v2, u0, u1, u2, axes[0], axes[1]) {
            return Some(coplanar_intersection_segment_3d(
                v0, v1, v2, u0, u1, u2, axes[0], axes[1],
            ));
        }
        return None;
    }

    let ad = [d[0].abs(), d[1].abs(), d[2].abs()];
    let max_idx = if ad[0] >= ad[1] && ad[0] >= ad[2] {
        0
    } else if ad[1] >= ad[2] {
        1
    } else {
        2
    };

    #[inline]
    fn lerp3(p: V3, q: V3, t: f64) -> V3 {
        [
            p[0] + (q[0] - p[0]) * t,
            p[1] + (q[1] - p[1]) * t,
            p[2] + (q[2] - p[2]) * t,
        ]
    }

    fn intervals_3d(p0: V3, p1: V3, p2: V3, d0: f64, d1: f64, d2: f64) -> (V3, V3) {
        let d0d1 = d0 * d1;
        let d0d2 = d0 * d2;
        if d0d1 > 0.0 {
            (lerp3(p2, p0, d2 / (d2 - d0)), lerp3(p2, p1, d2 / (d2 - d1)))
        } else if d0d2 > 0.0 {
            (lerp3(p1, p0, d1 / (d1 - d0)), lerp3(p1, p2, d1 / (d1 - d2)))
        } else if d1 * d2 > 0.0 || d0 != 0.0 {
            (lerp3(p0, p1, d0 / (d0 - d1)), lerp3(p0, p2, d0 / (d0 - d2)))
        } else if d1 != 0.0 {
            (lerp3(p1, p0, d1 / (d1 - d0)), lerp3(p1, p2, d1 / (d1 - d2)))
        } else {
            (lerp3(p2, p0, d2 / (d2 - d0)), lerp3(p2, p1, d2 / (d2 - d1)))
        }
    }

    let (pa0, pa1) = intervals_3d(v0, v1, v2, dv[0], dv[1], dv[2]);
    let (pb0, pb1) = intervals_3d(u0, u1, u2, du[0], du[1], du[2]);

    let (t1min, p1min, t1max, p1max) = if pa0[max_idx] <= pa1[max_idx] {
        (pa0[max_idx], pa0, pa1[max_idx], pa1)
    } else {
        (pa1[max_idx], pa1, pa0[max_idx], pa0)
    };
    let (t2min, p2min, t2max, p2max) = if pb0[max_idx] <= pb1[max_idx] {
        (pb0[max_idx], pb0, pb1[max_idx], pb1)
    } else {
        (pb1[max_idx], pb1, pb0[max_idx], pb0)
    };

    if t1max < t2min - PLANE_ZERO_EPS || t2max < t1min - PLANE_ZERO_EPS {
        return None;
    }

    let seg_start = if t1min >= t2min { p1min } else { p2min };
    let seg_end = if t1max <= t2max { p1max } else { p2max };
    Some((
        [seg_start[0] as f32, seg_start[1] as f32, seg_start[2] as f32],
        [seg_end[0] as f32, seg_end[1] as f32, seg_end[2] as f32],
    ))
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
    tri_tri_intersection_segment(v0, v1, v2, u0, u1, u2).is_some()
}

/// 有効布面に対して空間ハッシュを構築し、検索に必要な事前計算データを保持する。
struct ClothSpatialGrid {
    valid_faces: Vec<(u32, [u32; 3])>,
    mins: Vec<[f64; 3]>,
    maxs: Vec<[f64; 3]>,
    tris: Vec<[V3; 3]>,
    global_min: [f64; 3],
    global_max: [f64; 3],
    cell: f64,
    grid: HashMap<(i64, i64, i64), Vec<usize>>,
}

impl ClothSpatialGrid {
    fn build(positions: &[[f32; 3]], faces: &[[u32; 3]], cell_size: Option<f32>) -> Option<Self> {
        if faces.is_empty() {
            return None;
        }
        let n = positions.len();
        let pos64: Vec<V3> = positions
            .iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
        let mut valid_faces: Vec<(u32, [u32; 3])> = Vec::with_capacity(faces.len());
        for (fi, &[a, b, c]) in faces.iter().enumerate() {
            let (ia, ib, ic) = (a as usize, b as usize, c as usize);
            if a == b || b == c || c == a || ia >= n || ib >= n || ic >= n {
                continue;
            }
            valid_faces.push((fi as u32, [a, b, c]));
        }
        if valid_faces.is_empty() {
            return None;
        }
        let m = valid_faces.len();
        let mut mins = vec![[0.0f64; 3]; m];
        let mut maxs = vec![[0.0f64; 3]; m];
        let mut tris = vec![[[0.0f64; 3]; 3]; m];
        let mut edge_lens = Vec::with_capacity(m);
        let mut global_min = [f64::INFINITY; 3];
        let mut global_max = [f64::NEG_INFINITY; 3];
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
            for ax in 0..3 {
                global_min[ax] = global_min[ax].min(mn[ax]);
                global_max[ax] = global_max[ax].max(mx[ax]);
            }
            let d01 = ((t[1][0] - t[0][0]).powi(2)
                + (t[1][1] - t[0][1]).powi(2)
                + (t[1][2] - t[0][2]).powi(2))
            .sqrt();
            let d12 = ((t[2][0] - t[1][0]).powi(2)
                + (t[2][1] - t[1][1]).powi(2)
                + (t[2][2] - t[1][2]).powi(2))
            .sqrt();
            let d20 = ((t[0][0] - t[2][0]).powi(2)
                + (t[0][1] - t[2][1]).powi(2)
                + (t[0][2] - t[2][2]).powi(2))
            .sqrt();
            edge_lens.push((d01 + d12 + d20) / 3.0);
        }
        let max_span = (global_max[0] - global_min[0])
            .max(global_max[1] - global_min[1])
            .max(global_max[2] - global_min[2])
            .max(0.0);
        let cell = match cell_size {
            Some(c) if c > 0.0 => (c as f64).max(max_span / 128.0).max(1e-4),
            _ => {
                let mean = if edge_lens.is_empty() {
                    0.05
                } else {
                    edge_lens.iter().sum::<f64>() / edge_lens.len() as f64
                };
                (mean * 2.0).max(max_span / 128.0).max(1e-4)
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
        Some(Self {
            valid_faces,
            mins,
            maxs,
            tris,
            global_min,
            global_max,
            cell,
            grid,
        })
    }
}

#[inline]
fn to_f32_tri(t: [V3; 3]) -> [[f32; 3]; 3] {
    [
        [t[0][0] as f32, t[0][1] as f32, t[0][2] as f32],
        [t[1][0] as f32, t[1][1] as f32, t[1][2] as f32],
        [t[2][0] as f32, t[2][1] as f32, t[2][2] as f32],
    ]
}

/// 縫合スプリングで連結された頂点を Union-Find でクラスタ代表IDへ統合する。
fn build_sewing_vertex_clusters(num_vertices: usize, sewing_springs: &[[u32; 2]]) -> Vec<u32> {
    let mut parent: Vec<u32> = (0..num_vertices as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            let p = parent[x as usize];
            parent[x as usize] = parent[p as usize];
            x = p;
        }
        x
    }
    for &[a, b] in sewing_springs {
        if (a as usize) < num_vertices && (b as usize) < num_vertices {
            let ra = find(&mut parent, a);
            let rb = find(&mut parent, b);
            if ra != rb {
                if ra < rb {
                    parent[rb as usize] = ra;
                } else {
                    parent[ra as usize] = rb;
                }
            }
        }
    }
    for v in 0..num_vertices {
        let r = find(&mut parent, v as u32);
        parent[v] = r;
    }
    parent
}

/// 自己交差三角形ペアと 3D 交差線分を同時検出する（縫合結合クラスタの隣接除外対応版）。
/// `sewing_springs` が `Some` かつ非空で `ignore_adjacent == true` の場合、
/// 縫合スプリングで結ばれた頂点クラスタを共有する面ペアも隣接面として除外する。
pub fn find_triangle_intersection_segments_with_sewing(
    positions: &[[f32; 3]],
    faces: &[[u32; 3]],
    ignore_adjacent: bool,
    cell_size: Option<f32>,
    sewing_springs: Option<&[[u32; 2]]>,
) -> Vec<([u32; 2], [[f32; 3]; 2])> {
    let Some(sg) = ClothSpatialGrid::build(positions, faces, cell_size) else {
        return Vec::new();
    };
    let clusters = if ignore_adjacent {
        sewing_springs
            .filter(|sew| !sew.is_empty())
            .map(|sew| build_sewing_vertex_clusters(positions.len(), sew))
    } else {
        None
    };

    let mut tested: HashSet<(u32, u32)> = HashSet::new();
    let mut out: Vec<([u32; 2], [[f32; 3]; 2])> = Vec::new();
    for cell_faces in sg.grid.values() {
        if cell_faces.len() < 2 {
            continue;
        }
        for i in 0..cell_faces.len() {
            for j in (i + 1)..cell_faces.len() {
                let ka = cell_faces[i];
                let kb = cell_faces[j];
                let (fa, ea) = sg.valid_faces[ka];
                let (fb, eb) = sg.valid_faces[kb];
                let (pair, k0, k1) = if fa < fb {
                    ((fa, fb), ka, kb)
                } else {
                    ((fb, fa), kb, ka)
                };
                if !tested.insert(pair) {
                    continue;
                }
                if ignore_adjacent {
                    let (ca, cb) = if let Some(ref rep) = clusters {
                        (
                            [rep[ea[0] as usize], rep[ea[1] as usize], rep[ea[2] as usize]],
                            [rep[eb[0] as usize], rep[eb[1] as usize], rep[eb[2] as usize]],
                        )
                    } else {
                        (ea, eb)
                    };
                    if ca[0] == cb[0]
                        || ca[0] == cb[1]
                        || ca[0] == cb[2]
                        || ca[1] == cb[0]
                        || ca[1] == cb[1]
                        || ca[1] == cb[2]
                        || ca[2] == cb[0]
                        || ca[2] == cb[1]
                        || ca[2] == cb[2]
                    {
                        continue;
                    }
                }
                let separated = (0..3).any(|ax| sg.mins[k0][ax] > sg.maxs[k1][ax])
                    || (0..3).any(|ax| sg.mins[k1][ax] > sg.maxs[k0][ax]);
                if separated {
                    continue;
                }
                let p = to_f32_tri(sg.tris[k0]);
                let q = to_f32_tri(sg.tris[k1]);
                if let Some((s0, s1)) = tri_tri_intersection_segment(p[0], p[1], p[2], q[0], q[1], q[2]) {
                    out.push(([pair.0, pair.1], [s0, s1]));
                }
            }
        }
    }
    out.sort_by_key(|&(pair, _)| pair);
    out
}

/// 自己交差三角形ペアと 3D 交差線分を同時検出する。
/// 戻り値は `([f0, f1], [p_start, p_end])`（`f0 < f1` の昇順ソート列）。
pub fn find_triangle_intersection_segments(
    positions: &[[f32; 3]],
    faces: &[[u32; 3]],
    ignore_adjacent: bool,
    cell_size: Option<f32>,
) -> Vec<([u32; 2], [[f32; 3]; 2])> {
    find_triangle_intersection_segments_with_sewing(positions, faces, ignore_adjacent, cell_size, None)
}

/// 自己交差三角形ペア検出（縫合結合クラスタの隣接除外対応版）。
pub fn find_triangle_intersections_with_sewing(
    positions: &[[f32; 3]],
    faces: &[[u32; 3]],
    ignore_adjacent: bool,
    cell_size: Option<f32>,
    sewing_springs: Option<&[[u32; 2]]>,
) -> Vec<[u32; 2]> {
    find_triangle_intersection_segments_with_sewing(
        positions,
        faces,
        ignore_adjacent,
        cell_size,
        sewing_springs,
    )
    .into_iter()
    .map(|(pair, _)| pair)
    .collect()
}

/// 自己交差三角形ペア検出。Python `find_triangle_intersections` と同一式。
/// 戻り値は `(f0, f1)`（`f0 < f1`）の昇順ソート列。
pub fn find_triangle_intersections(
    positions: &[[f32; 3]],
    faces: &[[u32; 3]],
    ignore_adjacent: bool,
    cell_size: Option<f32>,
) -> Vec<[u32; 2]> {
    find_triangle_intersections_with_sewing(positions, faces, ignore_adjacent, cell_size, None)
}

/// 布メッシュとコライダー三角形メッシュ間の交差ペアと 3D 交差線分を検出する。
/// 戻り値は `([cloth_face_idx, collider_tri_idx], [p_start, p_end])` の昇順ソート列。
pub fn find_mesh_collider_intersections(
    positions: &[[f32; 3]],
    faces: &[[u32; 3]],
    collider_tris: &[[f32; 9]],
    cell_size: Option<f32>,
) -> Vec<([u32; 2], [[f32; 3]; 2])> {
    if collider_tris.is_empty() {
        return Vec::new();
    }
    let Some(sg) = ClothSpatialGrid::build(positions, faces, cell_size) else {
        return Vec::new();
    };
    let m = sg.valid_faces.len();
    let mut visited_stamp = vec![0u32; m];
    let mut out: Vec<([u32; 2], [[f32; 3]; 2])> = Vec::new();

    for (ci, tri_raw) in collider_tris.iter().enumerate() {
        if !tri_raw.iter().all(|v| v.is_finite()) {
            continue;
        }
        let q = [
            [tri_raw[0], tri_raw[1], tri_raw[2]],
            [tri_raw[3], tri_raw[4], tri_raw[5]],
            [tri_raw[6], tri_raw[7], tri_raw[8]],
        ];
        let mut c_min = [q[0][0] as f64, q[0][1] as f64, q[0][2] as f64];
        let mut c_max = c_min;
        for v in q.iter().skip(1) {
            for ax in 0..3 {
                let val = v[ax] as f64;
                c_min[ax] = c_min[ax].min(val);
                c_max[ax] = c_max[ax].max(val);
            }
        }
        // 布全体 AABB と重ならなければスキップ
        if (0..3).any(|ax| c_min[ax] > sg.global_max[ax] || sg.global_min[ax] > c_max[ax]) {
            continue;
        }
        // 巨大平面コライダー等によるセル爆発を防ぐため布 AABB にクランプ
        let clamped_min = [
            c_min[0].max(sg.global_min[0]),
            c_min[1].max(sg.global_min[1]),
            c_min[2].max(sg.global_min[2]),
        ];
        let clamped_max = [
            c_max[0].min(sg.global_max[0]),
            c_max[1].min(sg.global_max[1]),
            c_max[2].min(sg.global_max[2]),
        ];
        let cmin = [
            (clamped_min[0] / sg.cell).floor() as i64,
            (clamped_min[1] / sg.cell).floor() as i64,
            (clamped_min[2] / sg.cell).floor() as i64,
        ];
        let cmax = [
            (clamped_max[0] / sg.cell).floor() as i64,
            (clamped_max[1] / sg.cell).floor() as i64,
            (clamped_max[2] / sg.cell).floor() as i64,
        ];
        let nx = (cmax[0] - cmin[0] + 1).max(0) as u128;
        let ny = (cmax[1] - cmin[1] + 1).max(0) as u128;
        let nz = (cmax[2] - cmin[2] + 1).max(0) as u128;
        let cell_count = nx.saturating_mul(ny).saturating_mul(nz);
        let stamp = (ci as u32) + 1;

        if cell_count > 4096 {
            for k in 0..m {
                let separated = (0..3).any(|ax| sg.mins[k][ax] > c_max[ax] || c_min[ax] > sg.maxs[k][ax]);
                if separated {
                    continue;
                }
                let p = to_f32_tri(sg.tris[k]);
                if let Some((s0, s1)) = tri_tri_intersection_segment(p[0], p[1], p[2], q[0], q[1], q[2]) {
                    out.push(([sg.valid_faces[k].0, ci as u32], [s0, s1]));
                }
            }
        } else {
            for cx in cmin[0]..=cmax[0] {
                for cy in cmin[1]..=cmax[1] {
                    for cz in cmin[2]..=cmax[2] {
                        if let Some(cell_faces) = sg.grid.get(&(cx, cy, cz)) {
                            for &k in cell_faces {
                                if visited_stamp[k] == stamp {
                                    continue;
                                }
                                visited_stamp[k] = stamp;
                                let separated = (0..3)
                                    .any(|ax| sg.mins[k][ax] > c_max[ax] || c_min[ax] > sg.maxs[k][ax]);
                                if separated {
                                    continue;
                                }
                                let p = to_f32_tri(sg.tris[k]);
                                if let Some((s0, s1)) =
                                    tri_tri_intersection_segment(p[0], p[1], p[2], q[0], q[1], q[2])
                                {
                                    out.push(([sg.valid_faces[k].0, ci as u32], [s0, s1]));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out.sort_by_key(|&(pair, _)| pair);
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

    #[test]
    fn test_small_parallel_separated_triangles_not_false_positive() {
        // 辺長 1cm の平行三角形ペアが Z=2mm 離れている場合は非交差であること
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [0.01, 0.0, 0.0];
        let v2 = [0.0, 0.01, 0.0];
        let u0 = [0.0, 0.0, 0.002];
        let u1 = [0.01, 0.0, 0.002];
        let u2 = [0.0, 0.01, 0.002];
        assert!(!tri_tri_intersect(v0, v1, v2, u0, u1, u2));

        // 辺長 1cm で同一平面重なりがある場合は交差を検出すること
        let o0 = [0.002, 0.002, 0.0];
        let o1 = [0.012, 0.002, 0.0];
        let o2 = [0.002, 0.012, 0.0];
        assert!(tri_tri_intersect(v0, v1, v2, o0, o1, o2));
    }

    #[test]
    fn test_moller_denominator_sign_and_vertex_on_plane() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];

        // d0*d1 > 0 分岐と d0*d2 > 0 分岐（頂点入替）の双方で真の交差を正しく検出すること
        let u0 = [1.2, 0.2, 0.5];
        let u1 = [1.2, 0.25, 0.5];
        let u2 = [-0.5, 0.2, -1.0];
        assert!(tri_tri_intersect(v0, v1, v2, u0, u1, u2));
        assert!(tri_tri_intersect(v0, v1, v2, u0, u2, u1));

        // X+Y > 1 の外側へ抜ける非交差ペア
        let nu0 = [2.0, 0.2, 0.5];
        let nu1 = [2.0, 0.25, 0.5];
        let nu2 = [0.5, 0.2, -1.0];
        assert!(!tri_tri_intersect(v0, v1, v2, nu0, nu1, nu2));
        assert!(!tri_tri_intersect(v0, v1, v2, nu0, nu2, nu1));

        // 1頂点が相手平面上 (d0 == 0) で他2頂点がZ両側に跨ぐ場合
        let p_in0 = [0.2, 0.2, 0.0];
        let p_in1 = [0.3, 0.3, 1.0];
        let p_in2 = [0.4, 0.2, -1.0];
        assert!(tri_tri_intersect(v0, v1, v2, p_in0, p_in1, p_in2));

        let p_out0 = [2.0, 2.0, 0.0];
        let p_out1 = [2.0, 2.0, 1.0];
        let p_out2 = [2.5, 2.5, -1.0];
        assert!(!tri_tri_intersect(v0, v1, v2, p_out0, p_out1, p_out2));
    }

    #[test]
    fn test_intersection_segments_and_parity() {
        // 1. XY平面三角形と垂直三角形の交差線分検証
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let u0 = [0.2, -1.0, -1.0];
        let u1 = [0.2, 2.0, -1.0];
        let u2 = [0.2, 0.5, 2.0];
        let seg = tri_tri_intersection_segment(v0, v1, v2, u0, u1, u2)
            .expect("交差線分が返るべき");
        // x=0.2, z=0 で、y は [0.0, 0.8] の範囲で交差（u三角形は z=0 で y in [-0.5, 1.5]）
        let (ya, yb) = (seg.0[1].min(seg.1[1]), seg.0[1].max(seg.1[1]));
        assert!((seg.0[0] - 0.2).abs() < 1e-4 && (seg.1[0] - 0.2).abs() < 1e-4);
        assert!(seg.0[2].abs() < 1e-4 && seg.1[2].abs() < 1e-4);
        assert!(ya.abs() < 1e-4 && (yb - 0.8).abs() < 1e-4);

        // 2. 非交差・coplanar・頂点接触で tri_tri_intersect と厳密一致
        let cases = [
            (v0, v1, v2, [0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]),
            (v0, v1, v2, [0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [0.5, 1.5, 0.0]),
            (v0, v1, v2, [1.2, 0.2, 0.5], [1.2, 0.25, 0.5], [-0.5, 0.2, -1.0]),
            (v0, v1, v2, [2.0, 0.2, 0.5], [2.0, 0.25, 0.5], [0.5, 0.2, -1.0]),
            (v0, v1, v2, [0.2, 0.2, 0.0], [0.3, 0.3, 1.0], [0.4, 0.2, -1.0]),
        ];
        for (a0, a1, a2, b0, b1, b2) in cases {
            assert_eq!(
                tri_tri_intersection_segment(a0, a1, a2, b0, b1, b2).is_some(),
                tri_tri_intersect(a0, a1, a2, b0, b1, b2)
            );
        }

        // 3. メッシュ交差線分APIがペア一覧と一致すること
        let pos = vec![v0, v1, v2, u0, u1, u2];
        let faces = vec![[0, 1, 2], [3, 4, 5]];
        let pairs_only = find_triangle_intersections(&pos, &faces, true, None);
        let segs = find_triangle_intersection_segments(&pos, &faces, true, None);
        assert_eq!(pairs_only, vec![[0, 1]]);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].0, [0, 1]);
    }

    #[test]
    fn test_mesh_collider_intersections_and_large_plane() {
        let pos = vec![
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let faces = vec![[0, 1, 2], [3, 4, 5]];
        // 巨大平面コライダー (100m 四方、z=0 で面0と交差、面1とは非交差)
        let collider_tris = vec![
            [-50.0, -50.0, 0.0, 50.0, -50.0, 0.0, 50.0, 50.0, 0.0],
            [-50.0, -50.0, 0.0, 50.0, 50.0, 0.0, -50.0, 50.0, 0.0],
        ];
        // Z=0 同一平面重なり
        let hit = find_mesh_collider_intersections(&pos, &faces, &collider_tris, None);
        assert!(!hit.is_empty());
        assert!(hit.iter().all(|&(pair, _)| pair[0] == 0));

        // 斜め交差コライダー
        let cross_col = vec![
            [0.0, -1.0, -0.5, 0.0, 1.0, -0.5, 0.0, 0.0, 0.5],
        ];
        let cross_hit = find_mesh_collider_intersections(&pos, &faces, &cross_col, None);
        assert_eq!(cross_hit.len(), 1);
        assert_eq!(cross_hit[0].0, [0, 0]);
        let s = cross_hit[0].1;
        assert!(s[0][0].abs() < 1e-4 && s[1][0].abs() < 1e-4);
        assert!(s[0][2].abs() < 1e-4 && s[1][2].abs() < 1e-4);
    }

    #[test]
    fn test_sewn_adjacent_faces_excluded() {
        // 別頂点IDで同一座標エッジに密着しつつ交差する2枚（面0・面1）を sewing_springs で接続、
        // さらにそれらと交差する独立した第3面（面2）を配置する。
        let pos = vec![
            // 面0 (Z=0)
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            // 面1 (頂点3,4は頂点0,1と縫合結合されてエッジ付近が重なり交差する別パネル)
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.3, 0.3, 0.0],
            // 面2 (Zを跨いで面0を貫通する第3面)
            [0.2, -0.5, -0.5],
            [0.2, 1.5, -0.5],
            [0.2, 0.5, 0.5],
        ];
        let faces = vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]];
        let sewing = vec![[0, 3], [1, 4]];

        // sewing_springs なしでは (0, 1) も (0, 2) も検出される
        let raw_pairs = find_triangle_intersections_with_sewing(&pos, &faces, true, None, None);
        assert!(raw_pairs.contains(&[0, 1]));
        assert!(raw_pairs.contains(&[0, 2]));

        // sewing_springs ありでは縫合結合された隣接ペア (0, 1) は除外され、(0, 2) は残る
        let filtered_pairs =
            find_triangle_intersections_with_sewing(&pos, &faces, true, None, Some(&sewing));
        assert!(!filtered_pairs.contains(&[0, 1]));
        assert!(filtered_pairs.contains(&[0, 2]));
    }

    #[test]
    fn test_shallow_curved_surface_separated_triangles_not_false_positive() {
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [0.005, 0.0, 0.0];
        let v2 = [0.0, 0.005, 0.0];

        // du が全正 (+0.006〜+0.017mm) で横に 5mm 離れた非交差ペア
        let u0 = [0.006, 0.0, 6e-6];
        let u1 = [0.011, 0.0, 1.0e-5];
        let u2 = [0.006, 0.005, 1.7e-5];
        assert!(!tri_tri_intersect(v0, v1, v2, u0, u1, u2));
        assert!(tri_tri_intersection_segment(v0, v1, v2, u0, u1, u2).is_none());

        // わずかに平面を跨ぐが横に 5mm 離れた非交差ペア
        let w0 = [0.006, 0.0, -1.6e-5];
        let w1 = [0.011, 0.0, 8.0e-6];
        let w2 = [0.006, 0.005, -3.0e-6];
        assert!(!tri_tri_intersect(v0, v1, v2, w0, w1, w2));
        assert!(tri_tri_intersection_segment(v0, v1, v2, w0, w1, w2).is_none());

        // 斜め平面 x == z 上で f32 丸めを含んで重なる同一平面交差ペア
        let s = (1.0f32 / 2.0f32.sqrt()) as f32;
        let cv0 = [0.0, 0.0, 0.0];
        let cv1 = [0.01, 0.0, 0.01];
        let cv2 = [0.0, 0.01, 0.0];
        let cu0 = [0.002 * s, 0.002, 0.002 * s];
        let cu1 = [0.012 * s, 0.002, 0.012 * s];
        let cu2 = [0.002 * s, 0.012, 0.002 * s];
        assert!(tri_tri_intersect(cv0, cv1, cv2, cu0, cu1, cu2));
        assert!(tri_tri_intersection_segment(cv0, cv1, cv2, cu0, cu1, cu2).is_some());
    }
}
