// pattern_remesh.rs
// 2D CDT + Botsch-Kobbelt 等方性リメッシュ実装

use std::collections::{HashMap, HashSet};
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

/// 2D符号付き面積 (Shoelace formula)
pub fn polygon_signed_area_2d(pts: &[[f32; 2]]) -> f32 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1];
    }
    0.5 * area
}

/// 三角形の2D符号付き面積 (p0 -> p1 -> p2, CCW > 0)
#[inline]
pub fn triangle_signed_area_2d(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2]) -> f32 {
    0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
}

/// 三角形の最小内角（ラジアン）を計算
pub fn triangle_min_angle_rad(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2]) -> f32 {
    let d01 = [(p1[0] - p0[0]), (p1[1] - p0[1])];
    let d12 = [(p2[0] - p1[0]), (p2[1] - p1[1])];
    let d20 = [(p0[0] - p2[0]), (p0[1] - p2[1])];

    let l01 = (d01[0] * d01[0] + d01[1] * d01[1]).sqrt();
    let l12 = (d12[0] * d12[0] + d12[1] * d12[1]).sqrt();
    let l20 = (d20[0] * d20[0] + d20[1] * d20[1]).sqrt();

    if l01 < 1e-12 || l12 < 1e-12 || l20 < 1e-12 {
        return 0.0;
    }

    let cos0 = ((l01 * l01 + l20 * l20 - l12 * l12) / (2.0 * l01 * l20)).clamp(-1.0, 1.0);
    let cos1 = ((l01 * l01 + l12 * l12 - l20 * l20) / (2.0 * l01 * l12)).clamp(-1.0, 1.0);
    let cos2 = ((l12 * l12 + l20 * l20 - l01 * l01) / (2.0 * l12 * l20)).clamp(-1.0, 1.0);

    let a0 = cos0.acos();
    let a1 = cos1.acos();
    let a2 = cos2.acos();

    a0.min(a1).min(a2)
}

/// Winding Number 法による2D点多角形内外判定 (非ゼロなら内部)
pub fn winding_number_2d(p: [f32; 2], polygon: &[[f32; 2]]) -> i32 {
    let mut wn = 0;
    let n = polygon.len();
    for i in 0..n {
        let p1 = polygon[i];
        let p2 = polygon[(i + 1) % n];
        if p1[1] <= p[1] {
            if p2[1] > p[1] {
                let cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p[0] - p1[0]) * (p2[1] - p1[1]);
                if cross > 0.0 {
                    wn += 1;
                }
            }
        } else if p2[1] <= p[1] {
            let cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p[0] - p1[0]) * (p2[1] - p1[1]);
            if cross < 0.0 {
                wn -= 1;
            }
        }
    }
    wn
}

/// 点と線分の最短距離の二乗
pub fn point_to_segment_dist_sq(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [p[0] - a[0], p[1] - a[1]];
    let ab_len_sq = ab[0] * ab[0] + ab[1] * ab[1];
    if ab_len_sq < 1e-12 {
        return ap[0] * ap[0] + ap[1] * ap[1];
    }
    let t = ((ap[0] * ab[0] + ap[1] * ab[1]) / ab_len_sq).clamp(0.0, 1.0);
    let proj = [a[0] + t * ab[0], a[1] + t * ab[1]];
    let d = [p[0] - proj[0], p[1] - proj[1]];
    d[0] * d[0] + d[1] * d[1]
}

/// 2D型紙のリメッシュ内部データ構造
struct RemeshMesh2D {
    verts: Vec<[f32; 2]>,
    is_boundary: Vec<bool>,
    tris: Vec<[usize; 3]>,
    tri_alive: Vec<bool>,
    boundary_edges: HashSet<(usize, usize)>,
}

impl RemeshMesh2D {
    fn new(verts: Vec<[f32; 2]>, boundary_count: usize, tris: Vec<[usize; 3]>) -> Self {
        let n = verts.len();
        let mut is_boundary = vec![false; n];
        for i in 0..boundary_count.min(n) {
            is_boundary[i] = true;
        }
        let tri_alive = vec![true; tris.len()];
        let mut boundary_edges = HashSet::new();
        for i in 0..boundary_count {
            let next = (i + 1) % boundary_count;
            let e = if i < next { (i, next) } else { (next, i) };
            boundary_edges.insert(e);
        }
        Self {
            verts,
            is_boundary,
            tris,
            tri_alive,
            boundary_edges,
        }
    }

    fn is_boundary_edge(&self, u: usize, v: usize) -> bool {
        let e = if u < v { (u, v) } else { (v, u) };
        self.boundary_edges.contains(&e)
    }

    /// 頂点接続三角形リストを構築
    fn build_vertex_triangles(&self) -> Vec<Vec<usize>> {
        let mut v2t = vec![Vec::new(); self.verts.len()];
        for (t_idx, &t) in self.tris.iter().enumerate() {
            if !self.tri_alive[t_idx] {
                continue;
            }
            v2t[t[0]].push(t_idx);
            v2t[t[1]].push(t_idx);
            v2t[t[2]].push(t_idx);
        }
        v2t
    }

    /// 頂点の隣接頂点リストを構築
    fn build_vertex_neighbors(&self) -> Vec<Vec<usize>> {
        let mut neighbors = vec![HashSet::new(); self.verts.len()];
        for (t_idx, &t) in self.tris.iter().enumerate() {
            if !self.tri_alive[t_idx] {
                continue;
            }
            for i in 0..3 {
                let u = t[i];
                let v = t[(i + 1) % 3];
                neighbors[u].insert(v);
                neighbors[v].insert(u);
            }
        }
        neighbors.into_iter().map(|s| s.into_iter().collect()).collect()
    }

    /// エッジ一覧を収集: (u, v, t1, t2)
    fn build_edge_list(&self, _v2t: &[Vec<usize>]) -> Vec<(usize, usize, usize, Option<usize>)> {
        let mut edges_map: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (t_idx, &t) in self.tris.iter().enumerate() {
            if !self.tri_alive[t_idx] {
                continue;
            }
            for i in 0..3 {
                let u = t[i];
                let v = t[(i + 1) % 3];
                let e = if u < v { (u, v) } else { (v, u) };
                edges_map.entry(e).or_default().push(t_idx);
            }
        }
        let mut result = Vec::with_capacity(edges_map.len());
        for ((u, v), t_list) in edges_map {
            if t_list.is_empty() {
                continue;
            }
            let t1 = t_list[0];
            let t2 = if t_list.len() > 1 { Some(t_list[1]) } else { None };
            result.push((u, v, t1, t2));
        }
        result
    }

    /// Botsch-Kobbelt 反復: Split, Collapse, Flip, Smooth
    fn isotropic_remesh_step(
        &mut self,
        _target_len: f32,
        min_len: f32,
        max_len: f32,
        smoothing_strength: f32,
        polygon: &[[f32; 2]],
    ) {
        // --- 1. Split Long Edges ---
        let mut v2t = self.build_vertex_triangles();
        let edges = self.build_edge_list(&v2t);
        let max_len_sq = max_len * max_len;
        let mut split_tris = HashSet::new();

        for (u, v, t1, t2_opt) in edges {
            if !self.tri_alive[t1] || t2_opt.map_or(false, |t2| !self.tri_alive[t2]) {
                continue;
            }
            if split_tris.contains(&t1) || t2_opt.map_or(false, |t2| split_tris.contains(&t2)) {
                continue;
            }
            // 境界エッジおよび両端が境界頂点のエッジは分割しない
            if self.is_boundary_edge(u, v) || (self.is_boundary[u] && self.is_boundary[v]) {
                continue;
            }
            let pu = self.verts[u];
            let pv = self.verts[v];
            let len_sq = (pu[0] - pv[0]).powi(2) + (pu[1] - pv[1]).powi(2);
            if len_sq > max_len_sq {
                let new_pos = [0.5 * (pu[0] + pv[0]), 0.5 * (pu[1] + pv[1])];
                if winding_number_2d(new_pos, polygon) == 0 {
                    continue;
                }
                let new_idx = self.verts.len();
                self.verts.push(new_pos);
                self.is_boundary.push(false);

                split_tris.insert(t1);
                self.split_triangle_at_edge(t1, u, v, new_idx);
                if let Some(t2) = t2_opt {
                    if self.tri_alive[t2] {
                        split_tris.insert(t2);
                        self.split_triangle_at_edge(t2, u, v, new_idx);
                    }
                }
            }
        }

        // --- 2. Collapse Short Edges (内部頂点同士のみ) ---
        v2t = self.build_vertex_triangles();
        let edges = self.build_edge_list(&v2t);
        let min_len_sq = min_len * min_len;
        let mut collapsed_verts = HashSet::new();

        for (u, v, t1, t2_opt) in edges {
            if !self.tri_alive[t1] || t2_opt.map_or(false, |t2| !self.tri_alive[t2]) {
                continue;
            }
            // 境界頂点に接続するエッジは縮約不可
            if self.is_boundary[u] || self.is_boundary[v] {
                continue;
            }
            if collapsed_verts.contains(&u) || collapsed_verts.contains(&v) {
                continue;
            }

            let pu = self.verts[u];
            let pv = self.verts[v];
            let len_sq = (pu[0] - pv[0]).powi(2) + (pu[1] - pv[1]).powi(2);
            if len_sq < min_len_sq {
                let keep = u;
                let remove = v;
                let new_pos = [0.5 * (pu[0] + pv[0]), 0.5 * (pu[1] + pv[1])];

                let u_tris: HashSet<usize> = v2t[u].iter().copied().filter(|&t| self.tri_alive[t]).collect();
                let v_tris: HashSet<usize> = v2t[v].iter().copied().filter(|&t| self.tri_alive[t]).collect();
                let shared_tris: Vec<usize> = u_tris.intersection(&v_tris).copied().collect();
                if shared_tris.len() != 2 {
                    continue;
                }

                // 幾何チェック
                let mut valid = true;
                for &t_idx in &v_tris {
                    if shared_tris.contains(&t_idx) {
                        continue;
                    }
                    let t = self.tris[t_idx];
                    let mut pts = [self.verts[t[0]], self.verts[t[1]], self.verts[t[2]]];
                    for i in 0..3 {
                        if t[i] == remove || t[i] == keep {
                            pts[i] = new_pos;
                        }
                    }
                    if triangle_signed_area_2d(pts[0], pts[1], pts[2]) < 1e-8 {
                        valid = false;
                        break;
                    }
                    if triangle_min_angle_rad(pts[0], pts[1], pts[2]) < 0.314 {
                        valid = false;
                        break;
                    }
                }
                for &t_idx in &u_tris {
                    if shared_tris.contains(&t_idx) {
                        continue;
                    }
                    let t = self.tris[t_idx];
                    let mut pts = [self.verts[t[0]], self.verts[t[1]], self.verts[t[2]]];
                    for i in 0..3 {
                        if t[i] == remove || t[i] == keep {
                            pts[i] = new_pos;
                        }
                    }
                    if triangle_signed_area_2d(pts[0], pts[1], pts[2]) < 1e-8 {
                        valid = false;
                        break;
                    }
                    if triangle_min_angle_rad(pts[0], pts[1], pts[2]) < 0.314 {
                        valid = false;
                        break;
                    }
                }
                if winding_number_2d(new_pos, polygon) == 0 {
                    valid = false;
                }

                if valid {
                    self.verts[keep] = new_pos;
                    for &t_idx in &shared_tris {
                        self.tri_alive[t_idx] = false;
                    }
                    for &t_idx in &v_tris {
                        if !shared_tris.contains(&t_idx) {
                            for i in 0..3 {
                                if self.tris[t_idx][i] == remove {
                                    self.tris[t_idx][i] = keep;
                                }
                            }
                            v2t[keep].push(t_idx);
                        }
                    }
                    collapsed_verts.insert(keep);
                    collapsed_verts.insert(remove);
                }
            }
        }

        // --- 3. Equalize Valences & Improve Min Angles (Edge Flip) ---
        v2t = self.build_vertex_triangles();
        let edges = self.build_edge_list(&v2t);
        let neighbors = self.build_vertex_neighbors();
        let mut flipped_tris = HashSet::new();

        for (u, v, t1, t2_opt) in edges {
            let t2 = match t2_opt {
                Some(t) => t,
                None => continue,
            };
            if !self.tri_alive[t1] || !self.tri_alive[t2] {
                continue;
            }
            if flipped_tris.contains(&t1) || flipped_tris.contains(&t2) {
                continue;
            }
            if self.is_boundary_edge(u, v) {
                continue;
            }

            let tri1 = self.tris[t1];
            let tri2 = self.tris[t2];
            let w1 = match tri1.iter().copied().find(|&idx| idx != u && idx != v) {
                Some(idx) => idx,
                None => continue,
            };
            let w2 = match tri2.iter().copied().find(|&idx| idx != u && idx != v) {
                Some(idx) => idx,
                None => continue,
            };
            if w1 == w2 {
                continue;
            }

            let val_u = neighbors[u].len() as i32;
            let val_v = neighbors[v].len() as i32;
            let val_w1 = neighbors[w1].len() as i32;
            let val_w2 = neighbors[w2].len() as i32;

            let opt_u = if self.is_boundary[u] { 4 } else { 6 };
            let opt_v = if self.is_boundary[v] { 4 } else { 6 };
            let opt_w1 = if self.is_boundary[w1] { 4 } else { 6 };
            let opt_w2 = if self.is_boundary[w2] { 4 } else { 6 };

            let err_before = (val_u - opt_u).pow(2)
                + (val_v - opt_v).pow(2)
                + (val_w1 - opt_w1).pow(2)
                + (val_w2 - opt_w2).pow(2);
            let err_after = (val_u - 1 - opt_u).pow(2)
                + (val_v - 1 - opt_v).pow(2)
                + (val_w1 + 1 - opt_w1).pow(2)
                + (val_w2 + 1 - opt_w2).pow(2);

            let pw1 = self.verts[w1];
            let pw2 = self.verts[w2];
            let pu = self.verts[u];
            let pv = self.verts[v];

            let area1 = triangle_signed_area_2d(pw1, pw2, pv);
            let area2 = triangle_signed_area_2d(pw2, pw1, pu);

            if area1 > 1e-8 && area2 > 1e-8 {
                let min_ang_before = triangle_min_angle_rad(pu, pv, pw1).min(triangle_min_angle_rad(pv, pu, pw2));
                let min_ang_after = triangle_min_angle_rad(pw1, pw2, pv).min(triangle_min_angle_rad(pw2, pw1, pu));

                if min_ang_after < 0.314 && min_ang_after <= min_ang_before {
                    continue;
                }

                if min_ang_after > min_ang_before + 1e-4
                    || (err_after < err_before && min_ang_after >= 0.35)
                {
                    self.tris[t1] = [w1, w2, v];
                    self.tris[t2] = [w2, w1, u];
                    flipped_tris.insert(t1);
                    flipped_tris.insert(t2);
                }
            }
        }

        // --- 4. Laplacian Smoothing (内部頂点のみ) ---
        let neighbors = self.build_vertex_neighbors();
        let v2t = self.build_vertex_triangles();

        for i in 0..self.verts.len() {
            if self.is_boundary[i] {
                continue;
            }
            let n_list = &neighbors[i];
            if n_list.is_empty() {
                continue;
            }
            let mut centroid = [0.0f32, 0.0f32];
            for &nb in n_list {
                centroid[0] += self.verts[nb][0];
                centroid[1] += self.verts[nb][1];
            }
            centroid[0] /= n_list.len() as f32;
            centroid[1] /= n_list.len() as f32;

            let cur_pos = self.verts[i];
            let target_pos = [
                cur_pos[0] * (1.0 - smoothing_strength) + centroid[0] * smoothing_strength,
                cur_pos[1] * (1.0 - smoothing_strength) + centroid[1] * smoothing_strength,
            ];

            // 多角形内かつ全接続三角形の正面積・最小内角が保たれるかチェック
            if winding_number_2d(target_pos, polygon) == 0 {
                continue;
            }
            let mut valid = true;
            for &t_idx in &v2t[i] {
                if !self.tri_alive[t_idx] {
                    continue;
                }
                let t = self.tris[t_idx];
                let mut pts = [self.verts[t[0]], self.verts[t[1]], self.verts[t[2]]];
                for k in 0..3 {
                    if t[k] == i {
                        pts[k] = target_pos;
                    }
                }
                let area = triangle_signed_area_2d(pts[0], pts[1], pts[2]);
                if area < 1e-8 {
                    valid = false;
                    break;
                }
                let min_ang = triangle_min_angle_rad(pts[0], pts[1], pts[2]);
                if min_ang < 0.314 {
                    // スリバー三角形になる平滑化は拒絶
                    valid = false;
                    break;
                }
            }
            if valid {
                self.verts[i] = target_pos;
            }
        }
    }

    /// t_idx の三角形をエッジ (u, v) の中点 new_idx で分割
    fn split_triangle_at_edge(&mut self, t_idx: usize, u: usize, v: usize, new_idx: usize) {
        let t = self.tris[t_idx];
        let w = match t.iter().copied().find(|&idx| idx != u && idx != v) {
            Some(idx) => idx,
            None => return,
        };

        // (u, v, w) の向きを保持して2分割
        let u_pos = match t.iter().position(|&idx| idx == u) {
            Some(pos) => pos,
            None => return,
        };
        let is_u_to_v = t[(u_pos + 1) % 3] == v;

        let (t1_new, t2_new) = if is_u_to_v {
            ([u, new_idx, w], [new_idx, v, w])
        } else {
            ([v, new_idx, w], [new_idx, u, w])
        };

        self.tris[t_idx] = t1_new;
        self.tris.push(t2_new);
        self.tri_alive.push(true);
    }

    /// 有効な頂点と三角形をコンパクト化して返却
    fn compact(self, boundary_count: usize) -> (Vec<[f32; 2]>, Vec<[u32; 3]>) {
        let mut old_to_new = vec![None; self.verts.len()];
        let mut out_verts = Vec::with_capacity(self.verts.len());

        // 境界頂点 0..boundary_count は 0..boundary_count へ厳密に 1:1 コピー
        for i in 0..boundary_count {
            old_to_new[i] = Some(i);
            out_verts.push(self.verts[i]);
        }

        // 内部頂点で生存している三角形から参照されているものを追加
        let mut referenced = vec![false; self.verts.len()];
        for (t_idx, &t) in self.tris.iter().enumerate() {
            if !self.tri_alive[t_idx] {
                continue;
            }
            referenced[t[0]] = true;
            referenced[t[1]] = true;
            referenced[t[2]] = true;
        }

        for i in boundary_count..self.verts.len() {
            if referenced[i] {
                let new_idx = out_verts.len();
                old_to_new[i] = Some(new_idx);
                out_verts.push(self.verts[i]);
            }
        }

        let mut out_tris = Vec::new();
        for (t_idx, &t) in self.tris.iter().enumerate() {
            if !self.tri_alive[t_idx] {
                continue;
            }
            if let (Some(v0), Some(v1), Some(v2)) = (old_to_new[t[0]], old_to_new[t[1]], old_to_new[t[2]]) {
                // CCW面積が正であることを確認
                let p0 = out_verts[v0];
                let p1 = out_verts[v1];
                let p2 = out_verts[v2];
                let area = triangle_signed_area_2d(p0, p1, p2);
                if area > 1e-8 {
                    out_tris.push([v0 as u32, v1 as u32, v2 as u32]);
                } else if area < -1e-8 {
                    out_tris.push([v0 as u32, v2 as u32, v1 as u32]);
                }
            }
        }

        (out_verts, out_tris)
    }
}

/// 2D型紙のリメッシュ関数コアロジック
pub fn remesh_pattern_2d_core(
    boundary_loop: &[[f32; 2]],
    internal_constraints: Option<&[Vec<[f32; 2]>]>,
    target_edge_length: f32,
    iterations: u32,
    min_edge_ratio: f32,
    max_edge_ratio: f32,
    smoothing_strength: f32,
) -> Result<(Vec<[f32; 2]>, Vec<[u32; 3]>), String> {
    let b_count = boundary_loop.len();
    if b_count < 3 {
        return Err("Boundary loop must have at least 3 points".to_string());
    }

    // 1. バウンディングボックスの算出
    let mut u_min = f32::INFINITY;
    let mut u_max = f32::NEG_INFINITY;
    let mut v_min = f32::INFINITY;
    let mut v_max = f32::NEG_INFINITY;
    for pt in boundary_loop {
        u_min = u_min.min(pt[0]);
        u_max = u_max.max(pt[0]);
        v_min = v_min.min(pt[1]);
        v_max = v_max.max(pt[1]);
    }

    // 2. spade CDT 構築
    let mut cdt = ConstrainedDelaunayTriangulation::<Point2<f64>>::default();
    let mut b_handles = Vec::with_capacity(b_count);

    for pt in boundary_loop {
        let h = cdt
            .insert(Point2::new(pt[0] as f64, pt[1] as f64))
            .map_err(|e| format!("Failed to insert boundary point: {:?}", e))?;
        b_handles.push(h);
    }

    // 境界閉ループ拘束エッジ
    for i in 0..b_count {
        let next = (i + 1) % b_count;
        if cdt.can_add_constraint(b_handles[i], b_handles[next]) {
            cdt.add_constraint(b_handles[i], b_handles[next]);
        }
    }

    // 内部拘束線があれば追加
    if let Some(constraints) = internal_constraints {
        for poly in constraints {
            if poly.len() < 2 {
                continue;
            }
            let mut prev_h = None;
            for pt in poly {
                let h = cdt
                    .insert(Point2::new(pt[0] as f64, pt[1] as f64))
                    .map_err(|e| format!("Failed to insert constraint point: {:?}", e))?;
                if let Some(ph) = prev_h {
                    if cdt.can_add_constraint(ph, h) {
                        cdt.add_constraint(ph, h);
                    }
                }
                prev_h = Some(h);
            }
        }
    }

    // 3. 正三角形格子による内部スタイナー点の配置
    let l_eff = target_edge_length.max(0.001);
    let dy = l_eff * (3.0f32.sqrt() * 0.5);
    let dx = l_eff;
    let min_boundary_dist = 0.55 * l_eff;
    let min_boundary_dist_sq = min_boundary_dist * min_boundary_dist;

    let mut row = 0;
    let mut y = v_min + 0.5 * dy;
    while y < v_max {
        let offset = if row % 2 == 1 { 0.5 * dx } else { 0.0 };
        let mut x = u_min + offset;
        while x < u_max {
            let p = [x, y];
            // 多角形内部かつ境界から十分離れているか
            if winding_number_2d(p, boundary_loop) != 0 {
                let mut too_close = false;
                for i in 0..b_count {
                    let next = (i + 1) % b_count;
                    let d_sq = point_to_segment_dist_sq(p, boundary_loop[i], boundary_loop[next]);
                    if d_sq < min_boundary_dist_sq {
                        too_close = true;
                        break;
                    }
                }
                if !too_close {
                    let _ = cdt.insert(Point2::new(x as f64, y as f64));
                }
            }
            x += dx;
        }
        y += dy;
        row += 1;
    }

    // 4. CDT から初期メッシュを抽出 (0..B を厳密に先頭に配置)
    let mut handle_to_idx = HashMap::new();
    let mut initial_verts = Vec::new();

    // 境界頂点 0..B を確実に出力配列の先頭 0..B に配置
    for i in 0..b_count {
        handle_to_idx.insert(b_handles[i], i);
        initial_verts.push(boundary_loop[i]);
    }

    for v_handle in cdt.vertices() {
        let fix = v_handle.fix();
        if !handle_to_idx.contains_key(&fix) {
            let pos = v_handle.position();
            let idx = initial_verts.len();
            handle_to_idx.insert(fix, idx);
            initial_verts.push([pos.x as f32, pos.y as f32]);
        }
    }

    // 外部三角形を除去しつつ初期三角形を収集
    let mut initial_tris = Vec::new();
    for face in cdt.inner_faces() {
        let [va, vb, vc] = face.vertices();
        let ia = handle_to_idx[&va.fix()];
        let ib = handle_to_idx[&vb.fix()];
        let ic = handle_to_idx[&vc.fix()];

        let pa = initial_verts[ia];
        let pb = initial_verts[ib];
        let pc = initial_verts[ic];

        // 重心点
        let c = [(pa[0] + pb[0] + pc[0]) / 3.0, (pa[1] + pb[1] + pc[1]) / 3.0];
        if winding_number_2d(c, boundary_loop) == 0 {
            // 外部三角形
            continue;
        }

        let area = triangle_signed_area_2d(pa, pb, pc);
        if area > 1e-8 {
            initial_tris.push([ia, ib, ic]);
        } else if area < -1e-8 {
            initial_tris.push([ia, ic, ib]);
        }
    }

    // 5. Botsch-Kobbelt 等方性リメッシュの実行
    let mut mesh = RemeshMesh2D::new(initial_verts, b_count, initial_tris);

    let min_len = target_edge_length * min_edge_ratio;
    let max_len = target_edge_length * max_edge_ratio;

    for _ in 0..iterations {
        mesh.isotropic_remesh_step(
            target_edge_length,
            min_len,
            max_len,
            smoothing_strength,
            boundary_loop,
        );
    }

    let (final_verts, final_tris) = mesh.compact(b_count);
    Ok((final_verts, final_tris))
}
