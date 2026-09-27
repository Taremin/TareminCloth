//! ペアキャッシュ監査用CPU参照実装 (Blender非依存・GPU不要)。
//!
//! Python `pair_audit.audit_frame` を正本の式から移植した決定的実装。
//! `log_tools audit-pairs` の毎フレーム呼び出しにおけるPython外周ループと
//! 格子 broadphase を置き換える。方策実験系
//! (`collect_ground_truth_details` / `simulate_quota_policy`) は対象外とし、
//! Python側に残す。

use std::collections::{HashMap, HashSet};

type V3 = [f64; 3];

#[inline]
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm(v: V3) -> f64 {
    dot(v, v).sqrt()
}

#[inline]
fn clamp01(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

/// 各頂点の2ホップ以内近傍集合 (自身を含む)。Python `build_two_hop_sets` と同一式。
fn two_hop_sets(edges: &[[u32; 2]], n: usize) -> Vec<HashSet<usize>> {
    let mut adj: Vec<HashSet<usize>> = (0..n).map(|_| HashSet::new()).collect();
    for &[a, b] in edges {
        let (a, b) = (a as usize, b as usize);
        if a < n && b < n && a != b {
            adj[a].insert(b);
            adj[b].insert(a);
        }
    }
    let mut near: Vec<HashSet<usize>> = (0..n).map(|_| HashSet::new()).collect();
    for v in 0..n {
        let mut seen = HashSet::new();
        seen.insert(v);
        for &u in &adj[v] {
            seen.insert(u);
        }
        // borrow回避のため隣接を複写して走査
        let nbrs: Vec<usize> = adj[v].iter().copied().collect();
        for u in nbrs {
            for &w in &adj[u] {
                seen.insert(w);
            }
        }
        near[v] = seen;
    }
    near
}

/// 点-三角形距離 (Ericson 5.1.5 スカラー版)。Python `_closest_pt_single` の距離部分と同一式。
fn point_tri_distance(p: V3, a: V3, b: V3, c: V3) -> f64 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return norm(sub(p, a));
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return norm(sub(p, b));
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = if d1 != d3 { d1 / (d1 - d3) } else { 0.0 };
        let q = [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]];
        return norm(sub(p, q));
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return norm(sub(p, c));
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = if d2 != d6 { d2 / (d2 - d6) } else { 0.0 };
        let q = [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]];
        return norm(sub(p, q));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        let w = if denom != 0.0 {
            (d4 - d3) / denom
        } else {
            0.0
        };
        let q = [b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2])];
        return norm(sub(p, q));
    }
    let denom = va + vb + vc;
    let (v, w) = if denom != 0.0 {
        (vb / denom, vc / denom)
    } else {
        (0.0, 0.0)
    };
    let q = [
        a[0] + ab[0] * v + ac[0] * w,
        a[1] + ab[1] * v + ac[1] * w,
        a[2] + ab[2] * v + ac[2] * w,
    ];
    norm(sub(p, q))
}

/// 線分-線分距離 (Ericson 5.1.9 スカラー版)。Python `segment_segment_distances` と同一式。
fn seg_seg_distance(p1: V3, q1: V3, p2: V3, q2: V3) -> f64 {
    const EPS: f64 = 1e-12;
    let d1 = sub(q1, p1);
    let d2 = sub(q2, p2);
    let r = sub(p1, p2);
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);
    if a <= EPS {
        let t = (f / e.max(EPS)).clamp(0.0, 1.0);
        let c2 = [p2[0] + t * d2[0], p2[1] + t * d2[1], p2[2] + t * d2[2]];
        return norm(sub(p1, c2));
    }
    let c = dot(d1, r);
    if e <= EPS {
        let s = (-c / a).clamp(0.0, 1.0);
        let c1 = [p1[0] + s * d1[0], p1[1] + s * d1[1], p1[2] + s * d1[2]];
        return norm(sub(c1, p2));
    }
    let b = dot(d1, d2);
    let denom = a * e - b * b;
    let mut s = 0.0;
    let mut t;
    if denom.abs() > EPS {
        s = clamp01((b * f - c * e) / denom);
    }
    t = (b * s + f) / e;
    if t < 0.0 {
        t = 0.0;
        s = clamp01(-c / a);
    } else if t > 1.0 {
        t = 1.0;
        s = clamp01((b - c) / a);
    }
    let c1 = [p1[0] + s * d1[0], p1[1] + s * d1[1], p1[2] + s * d1[2]];
    let c2 = [p2[0] + t * d2[0], p2[1] + t * d2[1], p2[2] + t * d2[2]];
    norm(sub(c1, c2))
}

/// 点群に対するAABB格子 (単一セル参照)。Python `_grid_candidates_points` と同一式。
struct PointGrid {
    cell_inv: f64,
    map: HashMap<(i64, i64, i64), Vec<usize>>,
}

impl PointGrid {
    fn build(boxes_min: &[V3], boxes_max: &[V3], cell: f64) -> Self {
        let inv = 1.0 / cell.max(1e-9);
        let mut map: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (bi, (bmin, bmax)) in boxes_min.iter().zip(boxes_max.iter()).enumerate() {
            let cmin = [
                (bmin[0] * inv).floor() as i64,
                (bmin[1] * inv).floor() as i64,
                (bmin[2] * inv).floor() as i64,
            ];
            let cmax = [
                (bmax[0] * inv).floor() as i64,
                (bmax[1] * inv).floor() as i64,
                (bmax[2] * inv).floor() as i64,
            ];
            for cx in cmin[0]..=cmax[0] {
                for cy in cmin[1]..=cmax[1] {
                    for cz in cmin[2]..=cmax[2] {
                        map.entry((cx, cy, cz)).or_default().push(bi);
                    }
                }
            }
        }
        Self { cell_inv: inv, map }
    }

    fn query(&self, p: V3) -> &[usize] {
        let key = (
            (p[0] * self.cell_inv).floor() as i64,
            (p[1] * self.cell_inv).floor() as i64,
            (p[2] * self.cell_inv).floor() as i64,
        );
        self.map.get(&key).map(Vec::as_slice).unwrap_or(&[])
    }
}

fn dedup_keep_order(ids: &[usize]) -> Vec<usize> {
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(ids.len());
    for &id in ids {
        if seen.insert(id) {
            out.push(id);
        }
    }
    out
}

/// 監査結果の集計値。Python `audit_frame` の戻り辞書と同一キー。
#[derive(Debug, Clone, Default)]
pub struct AuditResult {
    pub vt_ground: usize,
    pub vt_hit: usize,
    pub vt_coverage: f64,
    pub vt_missed: usize,
    pub vt_miss_horizon: usize,
    pub vt_miss_other: usize,
    pub vt_stale: usize,
    pub vt_cached: usize,
    pub vt_dropped: usize,
    pub ee_ground: usize,
    pub ee_hit: usize,
    pub ee_coverage: f64,
    pub ee_missed: usize,
    pub ee_miss_horizon: usize,
    pub ee_miss_other: usize,
    pub ee_stale: usize,
    pub ee_cached: usize,
    pub ee_dropped: usize,
    pub check_static: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn audit_pairs(
    prev_pos: &[V3],
    prev_vel: Option<&[V3]>,
    curr_pos: &[V3],
    vt_pairs: &[[u32; 4]],
    ee_pairs: &[[u32; 4]],
    vt_count: u32,
    ee_count: u32,
    faces: &[[u32; 3]],
    edges: &[[u32; 2]],
    thickness: f64,
    safety_margin: f64,
    horizon_scale: f64,
    max_horizon: f64,
    margin_mode: u32,
    dt_frame: f64,
    max_vt_pairs: u32,
    max_ee_pairs: u32,
) -> AuditResult {
    let n = curr_pos.len();
    let eff_thick = thickness * 2.0;
    let check_static = eff_thick + safety_margin;
    let mode_f = if margin_mode == 0 { 0.0 } else { 1.0 };
    let zero = [0.0, 0.0, 0.0];
    let mut horizon = vec![0.0f64; n];
    for (i, h) in horizon.iter_mut().enumerate() {
        let v = prev_vel.map(|vv| vv[i]).unwrap_or(zero);
        let speed = norm(v);
        *h = (speed * dt_frame * horizon_scale).min(max_horizon) * mode_f;
    }
    let near = two_hop_sets(edges, n);

    let mut face_of: HashMap<(u32, u32, u32), usize> = HashMap::new();
    for (fi, &[a, b, c]) in faces.iter().enumerate() {
        let mut k = [a, b, c];
        k.sort_unstable();
        face_of.insert((k[0], k[1], k[2]), fi);
    }
    let mut edge_of: HashMap<(u32, u32), usize> = HashMap::new();
    for (ei, &[a, b]) in edges.iter().enumerate() {
        edge_of.insert((a.min(b), a.max(b)), ei);
    }
    let face_tris: Vec<[V3; 3]> = faces
        .iter()
        .map(|&[a, b, c]| [curr_pos[a as usize], curr_pos[b as usize], curr_pos[c as usize]])
        .collect();
    let prev_tris: Vec<[V3; 3]> = faces
        .iter()
        .map(|&[a, b, c]| [prev_pos[a as usize], prev_pos[b as usize], prev_pos[c as usize]])
        .collect();

    // ---- V-T グラウンドトゥルース ----
    let cell = check_static.max(1e-4);
    let (fmin, fmax): (Vec<V3>, Vec<V3>) = face_tris
        .iter()
        .map(|t| {
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
            (
                [mn[0] - check_static, mn[1] - check_static, mn[2] - check_static],
                [mx[0] + check_static, mx[1] + check_static, mx[2] + check_static],
            )
        })
        .unzip();
    let grid = PointGrid::build(&fmin, &fmax, cell);
    let mut gt_vt: HashSet<(usize, usize)> = HashSet::new();
    for (i, p) in curr_pos.iter().enumerate() {
        let cands = dedup_keep_order(grid.query(*p));
        if cands.is_empty() {
            continue;
        }
        for fi in cands {
            let f = faces[fi];
            if i as u32 == f[0] || i as u32 == f[1] || i as u32 == f[2] {
                continue;
            }
            if near[i].contains(&(f[0] as usize))
                && near[i].contains(&(f[1] as usize))
                && near[i].contains(&(f[2] as usize))
            {
                continue;
            }
            let t = face_tris[fi];
            if point_tri_distance(*p, t[0], t[1], t[2]) < check_static {
                gt_vt.insert((i, fi));
            }
        }
    }
    let mut cached_vt: HashSet<(usize, usize)> = HashSet::new();
    for &[i, j, v0, v1] in vt_pairs {
        let mut k = [j, v0, v1];
        k.sort_unstable();
        if let Some(&fi) = face_of.get(&(k[0], k[1], k[2])) {
            cached_vt.insert((i as usize, fi));
        }
    }
    let vt_hit = gt_vt.intersection(&cached_vt).count();
    let vt_missed: Vec<(usize, usize)> = gt_vt.difference(&cached_vt).copied().collect();
    let (mut vt_miss_horizon, mut vt_miss_other) = (0usize, 0usize);
    for (i, fi) in &vt_missed {
        let f = faces[*fi];
        let t = prev_tris[*fi];
        let d0 = point_tri_distance(prev_pos[*i], t[0], t[1], t[2]);
        let hj = f.iter().map(|v| horizon[*v as usize]).fold(0.0f64, f64::max);
        if d0 > eff_thick + safety_margin + horizon[*i] + hj {
            vt_miss_horizon += 1;
        } else {
            vt_miss_other += 1;
        }
    }
    let mut stale_vt = 0usize;
    for (i, fi) in &cached_vt {
        if *i >= n || *fi >= faces.len() {
            continue;
        }
        let f = faces[*fi];
        if *i as u32 == f[0] || *i as u32 == f[1] || *i as u32 == f[2] {
            continue;
        }
        let t = face_tris[*fi];
        if point_tri_distance(curr_pos[*i], t[0], t[1], t[2]) > check_static {
            stale_vt += 1;
        }
    }

    // ---- E-E グラウンドトゥルース ----
    let emid: Vec<V3> = edges
        .iter()
        .map(|&[a, b]| {
            let (pa, pb) = (curr_pos[a as usize], curr_pos[b as usize]);
            [
                (pa[0] + pb[0]) * 0.5,
                (pa[1] + pb[1]) * 0.5,
                (pa[2] + pb[2]) * 0.5,
            ]
        })
        .collect();
    let (emin, emax): (Vec<V3>, Vec<V3>) = edges
        .iter()
        .map(|&[a, b]| {
            let (pa, pb) = (curr_pos[a as usize], curr_pos[b as usize]);
            (
                [
                    pa[0].min(pb[0]) - check_static,
                    pa[1].min(pb[1]) - check_static,
                    pa[2].min(pb[2]) - check_static,
                ],
                [
                    pa[0].max(pb[0]) + check_static,
                    pa[1].max(pb[1]) + check_static,
                    pa[2].max(pb[2]) + check_static,
                ],
            )
        })
        .unzip();
    let egrid = PointGrid::build(&emin, &emax, cell);
    let ep0: Vec<V3> = edges.iter().map(|&[a, _]| curr_pos[a as usize]).collect();
    let ep1: Vec<V3> = edges.iter().map(|&[_, b]| curr_pos[b as usize]).collect();
    let mut gt_ee: HashSet<(usize, usize)> = HashSet::new();
    for (ei, m) in emid.iter().enumerate() {
        let cands: Vec<usize> = dedup_keep_order(egrid.query(*m))
            .into_iter()
            .filter(|&c| c != ei)
            .collect();
        if cands.is_empty() {
            continue;
        }
        let (a, b) = (edges[ei][0] as usize, edges[ei][1] as usize);
        for ej in cands {
            if ej <= ei {
                continue;
            }
            let (c, dd) = (edges[ej][0] as usize, edges[ej][1] as usize);
            if a == c || a == dd || b == c || b == dd {
                continue;
            }
            if near[a].contains(&c) {
                continue;
            }
            let dist = seg_seg_distance(ep0[ei], ep1[ei], ep0[ej], ep1[ej]);
            if dist < check_static && dist > 1e-9 {
                gt_ee.insert((ei, ej));
            }
        }
    }
    let mut cached_ee: HashSet<(usize, usize)> = HashSet::new();
    for &[vi, vui, vj, vvj] in ee_pairs {
        let ea = edge_of.get(&(vi.min(vui), vi.max(vui))).copied();
        let eb = edge_of.get(&(vj.min(vvj), vj.max(vvj))).copied();
        if let (Some(x), Some(y)) = (ea, eb) {
            if x != y {
                cached_ee.insert((x.min(y), x.max(y)));
            }
        }
    }
    let ee_hit = gt_ee.intersection(&cached_ee).count();
    let ee_missed: Vec<(usize, usize)> = gt_ee.difference(&cached_ee).copied().collect();
    let prev_ep0: Vec<V3> = edges.iter().map(|&[a, _]| prev_pos[a as usize]).collect();
    let prev_ep1: Vec<V3> = edges.iter().map(|&[_, b]| prev_pos[b as usize]).collect();
    let (mut ee_miss_horizon, mut ee_miss_other) = (0usize, 0usize);
    for (ei, ej) in &ee_missed {
        let a = edges[*ei][0] as usize;
        let d0 = seg_seg_distance(prev_ep0[*ei], prev_ep1[*ei], prev_ep0[*ej], prev_ep1[*ej]);
        let check_start = eff_thick
            + safety_margin
            + horizon[a]
            + horizon[edges[*ej][0] as usize];
        if d0 > check_start {
            ee_miss_horizon += 1;
        } else {
            ee_miss_other += 1;
        }
    }
    let mut stale_ee = 0usize;
    for (ei, ej) in &cached_ee {
        if *ei >= edges.len() || *ej >= edges.len() {
            continue;
        }
        if seg_seg_distance(ep0[*ei], ep1[*ei], ep0[*ej], ep1[*ej]) > check_static {
            stale_ee += 1;
        }
    }

    let vt_dropped = ((vt_count as usize).saturating_sub(vt_pairs.len()))
        .max((vt_count as usize).saturating_sub(max_vt_pairs as usize));
    let ee_dropped = ((ee_count as usize).saturating_sub(ee_pairs.len()))
        .max((ee_count as usize).saturating_sub(max_ee_pairs as usize));

    let ratio = |a: usize, b: usize| if b > 0 { a as f64 / b as f64 } else { 1.0 };
    AuditResult {
        vt_ground: gt_vt.len(),
        vt_hit,
        vt_coverage: ratio(vt_hit, gt_vt.len()),
        vt_missed: vt_missed.len(),
        vt_miss_horizon,
        vt_miss_other,
        vt_stale: stale_vt,
        vt_cached: cached_vt.len(),
        vt_dropped,
        ee_ground: gt_ee.len(),
        ee_hit,
        ee_coverage: ratio(ee_hit, gt_ee.len()),
        ee_missed: ee_missed.len(),
        ee_miss_horizon,
        ee_miss_other,
        ee_stale: stale_ee,
        ee_cached: cached_ee.len(),
        ee_dropped,
        check_static,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> (Vec<V3>, Vec<[u32; 3]>, Vec<[u32; 2]>) {
        let pos = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.001],
        ];
        let faces = vec![[0, 1, 2], [1, 3, 2]];
        let edges = vec![[0, 1], [1, 2], [2, 0], [1, 3], [3, 2]];
        (pos, faces, edges)
    }

    #[test]
    fn empty_pairs_yield_empty_ground_stats() {
        let (pos, faces, edges) = quad();
        let r = audit_pairs(
            &pos, None, &pos, &[], &[], 0, 0, &faces, &edges, 0.005, 0.005, 1.3, 0.02, 1,
            1.0 / 60.0, 65536, 65536,
        );
        assert_eq!(r.vt_cached, 0);
        assert_eq!(r.ee_cached, 0);
        assert_eq!(r.vt_dropped, 0);
        // 平面クアッドは2ホップ内で全頂点が近傍のため真値は空
        assert_eq!(r.vt_ground, 0);
    }

    #[test]
    fn point_tri_known_distances() {
        let d = point_tri_distance([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!((d - 1.0).abs() < 1e-9);
        let d = point_tri_distance([0.25, 0.25, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(d < 1e-9);
    }

    #[test]
    fn seg_seg_known_distances() {
        let d = seg_seg_distance(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        );
        assert!((d - 1.0).abs() < 1e-9);
        // 縮退 (点-線分)
        let d = seg_seg_distance(
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        );
        assert!((d - 0.5).abs() < 1e-9);
    }
}
