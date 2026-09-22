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

#[cfg(test)]
mod tests {
    use super::*;

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
}
