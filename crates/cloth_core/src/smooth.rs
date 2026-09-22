//! 平滑化ブラシ用ポータブルCPUカーネル
//!
//! Pythonブラシ (`python/taremin_cloth/brush/math.py`) と同一仕様の単一真実源。
//! Python側はフォールバック実装を保持し、parity は
//! `tests/core/test_smooth_parity.py` で検証する。
//! GPU隣接CSR (`ClothMesh::adj_*`) ではなく明示CSR入力を取り、縫合展開の有無に
//! 依存しない決定性を保証する。

use std::collections::HashSet;

/// 張力アシストの既定ゲイン (Python `TENSION_EXPAND` と一致させること)
pub const TENSION_EXPAND: f32 = 0.05;

/// フォールバック形状ID (Python `FALLOFF_SHAPES` 順序と一致させること)
pub const FALLOFF_SMOOTH: u32 = 0;
pub const FALLOFF_SPHERE: u32 = 1;
pub const FALLOFF_SHARP: u32 = 2;
pub const FALLOFF_LINEAR: u32 = 3;
pub const FALLOFF_CONSTANT: u32 = 4;

/// グラブ選択の既定裾野しきい値 (Python `select_grab_pins` 既定値と一致)
pub const GRAB_PIN_THRESHOLD: f32 = 0.01;

fn clamp01(v: f32) -> f32 {
    if !v.is_finite() {
        return 0.0;
    }
    v.clamp(0.0, 1.0)
}

/// 無向辺配列から1-ring隣接CSRを構築する。`(offsets, indices)` を返す。
/// offsets: [N+1]、indices: [2E] で頂点毎に昇順ソート済み。
/// 範囲外・自己ループ辺は除外する (Python `build_adjacency_csr` と同一)。
pub fn build_adjacency_csr(num_vertices: usize, edges: &[[u32; 2]]) -> (Vec<u32>, Vec<u32>) {
    if num_vertices == 0 {
        return (vec![0u32], Vec::new());
    }
    let n = num_vertices;
    let mut degrees = vec![0usize; n];
    let mut valid: Vec<[u32; 2]> = Vec::with_capacity(edges.len());
    for &[a, b] in edges {
        let (ia, ib) = (a as usize, b as usize);
        if a != b && ia < n && ib < n {
            valid.push([a, b]);
            degrees[ia] += 1;
            degrees[ib] += 1;
        }
    }
    let mut offsets = vec![0u32; n + 1];
    let mut acc = 0u32;
    for (i, d) in degrees.iter().enumerate() {
        offsets[i] = acc;
        acc += *d as u32;
    }
    offsets[n] = acc;
    let mut indices = vec![0u32; acc as usize];
    let mut cursor = offsets[..n].to_vec();
    for &[a, b] in &valid {
        let (ia, ib) = (a as usize, b as usize);
        indices[cursor[ia] as usize] = b;
        cursor[ia] += 1;
        indices[cursor[ib] as usize] = a;
        cursor[ib] += 1;
    }
    for v in 0..n {
        let (s, t) = (offsets[v] as usize, offsets[v + 1] as usize);
        if t - s > 1 {
            indices[s..t].sort_unstable();
        }
    }
    (offsets, indices)
}

/// 1-ring平均への減衰ブレンド目標を計算する。
/// `target[v] = pos[v] + (mean(pos[nbrs]) - pos[v]) * clamp(w) * clamp(strength)`。
/// 孤立・除外頂点は現位置を返す。`exclude` はターゲット側のみに適用する。
/// 不正入力時は `Err` (Python側 `ValueError` に対応)。
pub fn laplacian_smooth_targets(
    positions: &[[f32; 3]],
    adj_offsets: &[u32],
    adj_indices: &[u32],
    brush_idx: &[u32],
    brush_weights: &[f32],
    strength: f32,
    exclude: &[u32],
) -> Result<Vec<[f32; 3]>, String> {
    let n = positions.len();
    if brush_idx.len() != brush_weights.len() {
        return Err("brush_idx and brush_weights must have the same length".to_string());
    }
    let k_global = clamp01(strength);
    let excluded: HashSet<u32> = exclude.iter().copied().collect();
    let csr_ok = adj_offsets.len() == n + 1;
    let mut out = Vec::with_capacity(brush_idx.len());
    for (&v_raw, &w_raw) in brush_idx.iter().zip(brush_weights.iter()) {
        let v = v_raw as usize;
        if v >= n {
            return Err(format!("brush vertex index out of range: {v_raw}"));
        }
        let pv = positions[v];
        if excluded.contains(&v_raw) || !csr_ok {
            out.push(pv);
            continue;
        }
        let k = clamp01(w_raw) * k_global;
        if k <= 0.0 {
            out.push(pv);
            continue;
        }
        let (s, t) = (adj_offsets[v] as usize, adj_offsets[v + 1] as usize);
        if s > t || t > adj_indices.len() {
            out.push(pv);
            continue;
        }
        let mut sx = 0.0f64;
        let mut sy = 0.0f64;
        let mut sz = 0.0f64;
        let mut count = 0usize;
        for &u_raw in &adj_indices[s..t] {
            let u = u_raw as usize;
            if u < n {
                sx += positions[u][0] as f64;
                sy += positions[u][1] as f64;
                sz += positions[u][2] as f64;
                count += 1;
            }
        }
        if count == 0 {
            out.push(pv);
            continue;
        }
        let inv = 1.0f64 / count as f64;
        let kf = k as f64;
        out.push([
            (pv[0] as f64 + (sx * inv - pv[0] as f64) * kf) as f32,
            (pv[1] as f64 + (sy * inv - pv[1] as f64) * kf) as f32,
            (pv[2] as f64 + (sz * inv - pv[2] as f64) * kf) as f32,
        ]);
    }
    Ok(out)
}

/// ブラシ中心からの放射方向へ目標をわずかに拡張する (張力アシスト)。
/// `t' = c + (t - c) * (1 + expand * w * strength)`。
pub fn radial_expand_targets(
    positions: &[[f32; 3]],
    center: [f32; 3],
    brush_idx: &[u32],
    brush_weights: &[f32],
    strength: f32,
    expand: f32,
) -> Result<Vec<[f32; 3]>, String> {
    let n = positions.len();
    if brush_idx.len() != brush_weights.len() {
        return Err("brush_idx and brush_weights must have the same length".to_string());
    }
    let k_global = clamp01(strength);
    let gain = if expand.is_finite() { expand.max(0.0) } else { 0.0 };
    let mut out = Vec::with_capacity(brush_idx.len());
    for (&v_raw, &w_raw) in brush_idx.iter().zip(brush_weights.iter()) {
        let v = v_raw as usize;
        if v >= n {
            return Err(format!("brush vertex index out of range: {v_raw}"));
        }
        let pv = positions[v];
        let k = gain * clamp01(w_raw) * k_global;
        if k <= 0.0 {
            out.push(pv);
            continue;
        }
        let dx = pv[0] - center[0];
        let dy = pv[1] - center[1];
        let dz = pv[2] - center[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist <= 1e-9 {
            out.push(pv);
            continue;
        }
        let s = 1.0 + k;
        out.push([
            center[0] + dx * s,
            center[1] + dy * s,
            center[2] + dz * s,
        ]);
    }
    Ok(out)
}

/// 距離配列から減衰重み (0..1) を求める。中心=1、半径外=0。
/// Python `falloff_weights` と同一仕様 (`shape` は FALLOFF_* 定数)。
pub fn falloff_weights(distances: &[f32], radius: f32, shape: u32) -> Vec<f32> {
    let r = radius.max(1e-9);
    distances
        .iter()
        .map(|&d| {
            let t = (1.0 - d / r).clamp(0.0, 1.0);
            match shape {
                FALLOFF_LINEAR => t,
                FALLOFF_SPHERE => (1.0 - (1.0 - t) * (1.0 - t)).max(0.0).sqrt(),
                FALLOFF_SHARP => t * t,
                FALLOFF_CONSTANT => {
                    if d <= r {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => t * t * (3.0 - 2.0 * t),
            }
        })
        .collect()
}

/// ブラシ球内の頂点を列挙する。戻り値は `(indices, weights)`。
/// Python `verts_in_brush` と同一仕様 (マスクは非クランプ半径で判定)。
pub fn verts_in_brush(
    positions: &[[f32; 3]],
    center: [f32; 3],
    radius: f32,
    shape: u32,
) -> (Vec<u32>, Vec<f32>) {
    let mut found = Vec::new();
    let mut dists = Vec::new();
    for (i, p) in positions.iter().enumerate() {
        let dx = p[0] - center[0];
        let dy = p[1] - center[1];
        let dz = p[2] - center[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        if d <= radius {
            found.push(i as u32);
            dists.push(d);
        }
    }
    let weights = falloff_weights(&dists, radius, shape);
    (found, weights)
}

/// レイ命中面より奥の点を除外するマスク。`hit_t=None` 時は全て true。
/// Python `depth_keep_mask` と同一仕様。
pub fn depth_keep_mask(
    points: &[[f32; 3]],
    origin: [f32; 3],
    direction: [f32; 3],
    hit_t: Option<f32>,
    eps: f32,
) -> Vec<bool> {
    match hit_t {
        None => vec![true; points.len()],
        Some(h) => {
            let limit = h + eps;
            points
                .iter()
                .map(|p| {
                    (p[0] - origin[0]) * direction[0]
                        + (p[1] - origin[1]) * direction[1]
                        + (p[2] - origin[2]) * direction[2]
                        <= limit
                })
                .collect()
        }
    }
}

/// グラブ対象の選別: 閾値未満の裾野は除外する。
/// Python `select_grab_pins` と同一仕様。
pub fn select_grab_pins(
    found: &[u32],
    weights: &[f32],
    threshold: f32,
) -> Result<(Vec<u32>, Vec<f32>), String> {
    if found.len() != weights.len() {
        return Err("found and weights must have the same length".to_string());
    }
    let mut out_idx = Vec::new();
    let mut out_w = Vec::new();
    for (&i, &w) in found.iter().zip(weights.iter()) {
        let wc = clamp01(w);
        if wc >= threshold {
            out_idx.push(i);
            out_w.push(wc);
        }
    }
    Ok((out_idx, out_w))
}

/// ラジアル操作の相対式: 開始値×(1+dx/px_scale)を範囲へクリップする。
/// Python `radial_adjust` と同一仕様。
pub fn radial_adjust(start_value: f64, dx_px: f64, min_value: f64, max_value: f64) -> f64 {
    radial_adjust_with_scale(start_value, dx_px, min_value, max_value, 200.0)
}

pub fn radial_adjust_with_scale(
    start_value: f64,
    dx_px: f64,
    min_value: f64,
    max_value: f64,
    px_scale: f64,
) -> f64 {
    let v = start_value * (1.0 + dx_px / px_scale.max(1.0));
    if !v.is_finite() {
        return min_value;
    }
    v.clamp(min_value, max_value)
}

/// 範囲グラブのドラッグ目標: `target = init + delta * (w * strength)`。
/// Python `RangeGrabTool._move` 内ループと同一仕様。
pub fn grab_drag_targets(
    initials: &[[f32; 3]],
    delta: [f32; 3],
    weights: &[f32],
    strength: f32,
) -> Result<Vec<[f32; 3]>, String> {
    if initials.len() != weights.len() {
        return Err("initials and weights must have the same length".to_string());
    }
    let ks = clamp01(strength);
    Ok(initials
        .iter()
        .zip(weights.iter())
        .map(|(init, &w)| {
            let k = clamp01(w) * ks;
            [
                init[0] + delta[0] * k,
                init[1] + delta[1] * k,
                init[2] + delta[2] * k,
            ]
        })
        .collect())
}

/// レイと平面の交点。平行時は `None`。
/// Python 単一グラブのビュー平面交点計算と同一仕様 (|denom|<=1e-6 は無効)。
pub fn ray_plane_hit(
    origin: [f32; 3],
    direction: [f32; 3],
    plane_point: [f32; 3],
    plane_normal: [f32; 3],
) -> Option<[f32; 3]> {
    let denom = direction[0] * plane_normal[0]
        + direction[1] * plane_normal[1]
        + direction[2] * plane_normal[2];
    if denom.abs() <= 1e-6 {
        return None;
    }
    let t = ((plane_point[0] - origin[0]) * plane_normal[0]
        + (plane_point[1] - origin[1]) * plane_normal[1]
        + (plane_point[2] - origin[2]) * plane_normal[2])
        / denom;
    Some([
        origin[0] + direction[0] * t,
        origin[1] + direction[1] * t,
        origin[2] + direction[2] * t,
    ])
}

#[cfg(test)]
mod tests {    use super::*;

    fn approx(a: [f32; 3], b: [f32; 3]) {
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-5, "mismatch {a:?} vs {b:?}");
        }
    }

    #[test]
    fn csr_triangle_sorted() {
        let (off, idx) = build_adjacency_csr(3, &[[0, 1], [1, 2], [2, 0]]);
        assert_eq!(off, vec![0, 2, 4, 6]);
        assert_eq!(&idx[0..2], &[1, 2]);
        assert_eq!(&idx[2..4], &[0, 2]);
        assert_eq!(&idx[4..6], &[0, 1]);
    }

    #[test]
    fn csr_filters_invalid() {
        let (off, idx) = build_adjacency_csr(2, &[[0, 0], [0, 5], [0, 1]]);
        assert_eq!(off, vec![0, 1, 2]);
        assert_eq!(idx, vec![1, 0]);
    }

    #[test]
    fn smooth_centers_peak() {
        let pos = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]];
        let (off, idx) = build_adjacency_csr(3, &[[0, 1], [0, 2]]);
        let t = laplacian_smooth_targets(&pos, &off, &idx, &[0], &[1.0], 1.0, &[]).unwrap();
        approx(t[0], [0.0, 0.0, 0.0]);
        // w=0.5 -> 半分
        let t = laplacian_smooth_targets(&pos, &off, &idx, &[0], &[0.5], 1.0, &[]).unwrap();
        approx(t[0], [0.0, 0.0, 0.5]);
        // 除外・孤立は不変
        let t = laplacian_smooth_targets(&pos, &off, &idx, &[0], &[1.0], 1.0, &[0]).unwrap();
        approx(t[0], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn smooth_rejects_mismatch_and_oob() {
        let pos = [[0.0, 0.0, 0.0]];
        let (off, idx) = build_adjacency_csr(1, &[]);
        assert!(laplacian_smooth_targets(&pos, &off, &idx, &[0], &[1.0, 0.5], 1.0, &[]).is_err());
        assert!(laplacian_smooth_targets(&pos, &off, &idx, &[9], &[1.0], 1.0, &[]).is_err());
    }

    #[test]
    fn expand_outward() {
        let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let t = radial_expand_targets(&pos, [0.0, 0.0, 0.0], &[1, 2], &[1.0, 1.0], 1.0, 0.1).unwrap();
        approx(t[0], [1.1, 0.0, 0.0]);
        approx(t[1], [0.0, 2.2, 0.0]);
        // 中心・強度0は不変
        let t = radial_expand_targets(&pos, [0.0, 0.0, 0.0], &[0], &[1.0], 1.0, 0.1).unwrap();
        approx(t[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn falloff_shapes_match_python() {
        // smoothstep(0.5)=0.5、境界 d==r で CONSTANT のみ 1.0
        let d = [0.0, 0.05, 0.1, 0.2];
        let w = falloff_weights(&d, 0.1, FALLOFF_SMOOTH);
        assert!((w[0] - 1.0).abs() < 1e-6);
        assert!((w[1] - 0.5).abs() < 1e-5);
        assert_eq!(w[3], 0.0);
        for shape in [FALLOFF_SMOOTH, FALLOFF_SPHERE, FALLOFF_SHARP, FALLOFF_LINEAR] {
            let w = falloff_weights(&[0.0, 0.2], 0.1, shape);
            assert_eq!(w[0], 1.0);
            assert_eq!(w[1], 0.0);
        }
        let w = falloff_weights(&[0.1], 0.1, FALLOFF_CONSTANT);
        assert_eq!(w[0], 1.0);
        let w = falloff_weights(&[0.11], 0.1, FALLOFF_CONSTANT);
        assert_eq!(w[0], 0.0);
        // 単調性
        for shape in [FALLOFF_SMOOTH, FALLOFF_SPHERE, FALLOFF_SHARP, FALLOFF_LINEAR] {
            let ds: Vec<f32> = (0..20).map(|i| i as f32 * 0.005).collect();
            let w = falloff_weights(&ds, 0.1, shape);
            for pair in w.windows(2) {
                assert!(pair[0] + 1e-6 >= pair[1]);
            }
        }
    }

    #[test]
    fn brush_pick_and_depth() {
        let pos = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let (found, w) = verts_in_brush(&pos, [0.0, 0.0, 0.0], 0.6, FALLOFF_SMOOTH);
        assert_eq!(found, vec![0]);
        assert!((w[0] - 1.0).abs() < 1e-6);
        let (found, _) = verts_in_brush(&pos, [5.0, 5.0, 5.0], 0.1, FALLOFF_SMOOTH);
        assert!(found.is_empty());
        // デプスマスク: z=1 は手前、z=-1 は命中面の奥
        let pts = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
        let m = depth_keep_mask(&pts, [0.0, 0.0, 5.0], [0.0, 0.0, -1.0], Some(5.0), 0.1);
        assert_eq!(m, vec![true, true, false]);
        let m = depth_keep_mask(&pts, [0.0, 0.0, 0.0], [0.0, 0.0, -1.0], None, 0.1);
        assert_eq!(m, vec![true, true, true]);
    }

    #[test]
    fn grab_pin_selection() {
        let (idx, w) = select_grab_pins(&[0, 1, 2, 3], &[1.0, 0.6, 0.05, 0.005], 0.01).unwrap();
        assert_eq!(idx, vec![0, 1, 2]);
        assert!((w[0] - 1.0).abs() < 1e-6);
        assert!(select_grab_pins(&[0], &[1.0, 0.5], 0.01).is_err());
    }

    #[test]
    fn radial_adjust_matches_python() {
        assert!((radial_adjust(0.05, 200.0, 0.005, 0.5) - 0.1).abs() < 1e-12);
        assert!((radial_adjust(0.05, -100.0, 0.005, 0.5) - 0.025).abs() < 1e-12);
        assert_eq!(radial_adjust(0.5, 10000.0, 0.005, 0.5), 0.5);
        assert_eq!(radial_adjust(0.005, -10000.0, 0.005, 0.5), 0.005);
    }

    #[test]
    fn grab_drag_and_ray_plane() {
        let init = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let t = grab_drag_targets(&init, [1.0, 0.0, 0.0], &[1.0, 0.5], 0.8).unwrap();
        approx(t[0], [0.8, 0.0, 0.0]);
        approx(t[1], [1.4, 0.0, 0.0]);
        assert!(grab_drag_targets(&init, [0.0, 0.0, 0.0], &[1.0], 1.0).is_err());
        let hit = ray_plane_hit(
            [0.0, 0.0, 5.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        )
        .unwrap();
        approx(hit, [0.0, 0.0, 0.0]);
        assert!(ray_plane_hit(
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        )
        .is_none());
    }
}
