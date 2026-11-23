// pattern_remesh.rs
// 2D CDT + Botsch-Kobbelt 等方性リメッシュ実装

use std::collections::{HashMap, HashSet};
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

pub use crate::geometry_2d::{
    point_to_segment_dist_sq, polygon_signed_area_2d, triangle_min_angle_rad,
    triangle_signed_area_2d, winding_number_2d,
};

/// 2D型紙のリメッシュ内部データ構造
struct RemeshMesh2D {
    verts: Vec<[f32; 2]>,
    is_boundary: Vec<bool>,
    is_constrained: Vec<bool>,
    tris: Vec<[usize; 3]>,
    tri_alive: Vec<bool>,
    boundary_edges: HashSet<(usize, usize)>,
    constrained_edges: HashSet<(usize, usize)>,
}

impl RemeshMesh2D {
    #[inline]
    fn edge_key(u: usize, v: usize) -> (usize, usize) {
        if u < v { (u, v) } else { (v, u) }
    }

    fn new(
        verts: Vec<[f32; 2]>,
        boundary_count: usize,
        tris: Vec<[usize; 3]>,
        internal_constrained_verts: HashSet<usize>,
        internal_constrained_edges: HashSet<(usize, usize)>,
    ) -> Self {
        let n = verts.len();
        let mut is_boundary = vec![false; n];
        for i in 0..boundary_count.min(n) {
            is_boundary[i] = true;
        }
        let mut is_constrained = vec![false; n];
        for idx in internal_constrained_verts {
            if idx < n && !is_boundary[idx] {
                is_constrained[idx] = true;
            }
        }
        let tri_alive = vec![true; tris.len()];
        let mut boundary_edges = HashSet::new();
        for i in 0..boundary_count {
            let next = (i + 1) % boundary_count;
            boundary_edges.insert(Self::edge_key(i, next));
        }
        let mut constrained_edges = HashSet::new();
        for (u, v) in internal_constrained_edges {
            let key = Self::edge_key(u, v);
            if !boundary_edges.contains(&key) {
                constrained_edges.insert(key);
                if u < n && !is_boundary[u] {
                    is_constrained[u] = true;
                }
                if v < n && !is_boundary[v] {
                    is_constrained[v] = true;
                }
            }
        }
        Self {
            verts,
            is_boundary,
            is_constrained,
            tris,
            tri_alive,
            boundary_edges,
            constrained_edges,
        }
    }

    #[inline]
    fn is_locked_vert(&self, idx: usize) -> bool {
        self.is_boundary[idx] || self.is_constrained[idx]
    }

    #[inline]
    fn is_boundary_edge(&self, u: usize, v: usize) -> bool {
        self.boundary_edges.contains(&Self::edge_key(u, v))
    }

    #[inline]
    fn is_protected_edge(&self, u: usize, v: usize) -> bool {
        let e = Self::edge_key(u, v);
        self.boundary_edges.contains(&e) || self.constrained_edges.contains(&e)
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
            // 境界エッジ、内部拘束エッジ、および両端が境界頂点のエッジは分割禁止（内部拘束エッジ自体は長ければ拘束継承して分割）
            let is_bnd_edge = self.is_boundary_edge(u, v) || (self.is_boundary[u] && self.is_boundary[v]);
            if is_bnd_edge {
                continue;
            }
            let e_key = Self::edge_key(u, v);
            let is_internal_constrained = self.constrained_edges.contains(&e_key);
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
                self.is_constrained.push(is_internal_constrained);
                if is_internal_constrained {
                    self.constrained_edges.remove(&e_key);
                    self.constrained_edges.insert(Self::edge_key(u, new_idx));
                    self.constrained_edges.insert(Self::edge_key(new_idx, v));
                }

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

        // --- 2. Collapse Short Edges (自由内部頂点同士のみ) ---
        v2t = self.build_vertex_triangles();
        let edges = self.build_edge_list(&v2t);
        let min_len_sq = min_len * min_len;
        let mut collapsed_verts = HashSet::new();

        for (u, v, t1, t2_opt) in edges {
            if !self.tri_alive[t1] || t2_opt.map_or(false, |t2| !self.tri_alive[t2]) {
                continue;
            }
            // 境界頂点・内部拘束頂点・保護エッジは縮約不可
            if self.is_locked_vert(u) || self.is_locked_vert(v) || self.is_protected_edge(u, v) {
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

                // Link Condition: |N(u) ∩ N(v)| == 2 を満たさないエッジ縮約は非多様体を生成するため拒否
                let mut u_neighbors = HashSet::new();
                for &t_idx in &u_tris {
                    for &vi in &self.tris[t_idx] {
                        if vi != u {
                            u_neighbors.insert(vi);
                        }
                    }
                }
                let mut v_neighbors = HashSet::new();
                for &t_idx in &v_tris {
                    for &vi in &self.tris[t_idx] {
                        if vi != v {
                            v_neighbors.insert(vi);
                        }
                    }
                }
                if u_neighbors.intersection(&v_neighbors).count() != 2 {
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
            if self.is_protected_edge(u, v) {
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
            if w1 == w2 || neighbors[w1].contains(&w2) {
                continue;
            }

            // tri1 における共有辺の向きを基準に (a, b, w1) が CCW になるよう正規化
            let u_pos = match tri1.iter().position(|&idx| idx == u) {
                Some(pos) => pos,
                None => continue,
            };
            let is_u_to_v = tri1[(u_pos + 1) % 3] == v;
            let (a, b) = if is_u_to_v { (u, v) } else { (v, u) };

            let val_a = neighbors[a].len() as i32;
            let val_b = neighbors[b].len() as i32;
            let val_w1 = neighbors[w1].len() as i32;
            let val_w2 = neighbors[w2].len() as i32;

            let opt_a = if self.is_boundary[a] { 4 } else { 6 };
            let opt_b = if self.is_boundary[b] { 4 } else { 6 };
            let opt_w1 = if self.is_boundary[w1] { 4 } else { 6 };
            let opt_w2 = if self.is_boundary[w2] { 4 } else { 6 };

            let err_before = (val_a - opt_a).pow(2)
                + (val_b - opt_b).pow(2)
                + (val_w1 - opt_w1).pow(2)
                + (val_w2 - opt_w2).pow(2);
            let err_after = (val_a - 1 - opt_a).pow(2)
                + (val_b - 1 - opt_b).pow(2)
                + (val_w1 + 1 - opt_w1).pow(2)
                + (val_w2 + 1 - opt_w2).pow(2);

            let pw1 = self.verts[w1];
            let pw2 = self.verts[w2];
            let pa = self.verts[a];
            let pb = self.verts[b];

            let area1 = triangle_signed_area_2d(pw1, pw2, pb);
            let area2 = triangle_signed_area_2d(pw2, pw1, pa);

            if area1 > 1e-8 && area2 > 1e-8 {
                let min_ang_before = triangle_min_angle_rad(pa, pb, pw1).min(triangle_min_angle_rad(pb, pa, pw2));
                let min_ang_after = triangle_min_angle_rad(pw1, pw2, pb).min(triangle_min_angle_rad(pw2, pw1, pa));

                if min_ang_after < 0.314 && min_ang_after <= min_ang_before {
                    continue;
                }

                if min_ang_after > min_ang_before + 1e-4
                    || (err_after < err_before && min_ang_after >= 0.35)
                {
                    self.tris[t1] = [w1, w2, b];
                    self.tris[t2] = [w2, w1, a];
                    flipped_tris.insert(t1);
                    flipped_tris.insert(t2);
                }
            }
        }

        // --- 4. Laplacian Smoothing (自由内部頂点のみ) ---
        let neighbors = self.build_vertex_neighbors();
        let v2t = self.build_vertex_triangles();

        for i in 0..self.verts.len() {
            if self.is_locked_vert(i) {
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
    for (i, pt) in boundary_loop.iter().enumerate() {
        if !pt[0].is_finite() || !pt[1].is_finite() {
            return Err(format!("Boundary loop point {i} has non-finite coordinates"));
        }
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

    // 内部拘束線があれば追加（後段の破棄防止用に挿入ハンドルを記録）
    let mut constraint_handles = Vec::new();
    if let Some(constraints) = internal_constraints {
        for poly in constraints {
            if poly.len() < 2 {
                continue;
            }
            let mut prev_h = None;
            for pt in poly {
                if !pt[0].is_finite() || !pt[1].is_finite() {
                    return Err("Constraint point has non-finite coordinates".to_string());
                }
                let h = cdt
                    .insert(Point2::new(pt[0] as f64, pt[1] as f64))
                    .map_err(|e| format!("Failed to insert constraint point: {:?}", e))?;
                constraint_handles.push(h);
                if let Some(ph) = prev_h {
                    if cdt.can_add_constraint(ph, h) {
                        cdt.add_constraint(ph, h);
                    }
                }
                prev_h = Some(h);
            }
        }
    }

    // 3. 正三角形格子による内部スタイナー点の配置（整数カウンタ＋総点数上限ガードでULP無限ループを防止）
    let l_eff = target_edge_length.max(0.001);
    let min_boundary_dist = 0.55 * l_eff;
    let min_boundary_dist_sq = min_boundary_dist * min_boundary_dist;

    const MAX_STEINER_GRID_POINTS: f64 = 250_000.0;
    let span_u = (u_max as f64 - u_min as f64).max(0.0);
    let span_v = (v_max as f64 - v_min as f64).max(0.0);
    let tri_row_ratio = 3.0f64.sqrt() * 0.5;
    let min_dx_for_cap = ((span_u * span_v) / (MAX_STEINER_GRID_POINTS * tri_row_ratio)).sqrt();
    let dx_f64 = (l_eff as f64).max(min_dx_for_cap).max(1e-6);
    let dy_f64 = dx_f64 * tri_row_ratio;

    let ny = ((span_v / dy_f64).ceil() as usize).min(200_000);
    let nx = ((span_u / dx_f64).ceil() as usize).min(200_000);

    for row in 0..ny {
        let y = (v_min as f64 + (row as f64 + 0.5) * dy_f64) as f32;
        if y >= v_max {
            break;
        }
        let offset = if row % 2 == 1 { 0.5 * dx_f64 } else { 0.0 };
        for col in 0..nx {
            let x = (u_min as f64 + offset + (col as f64) * dx_f64) as f32;
            if x >= u_max {
                break;
            }
            let p = [x, y];
            // 多角形内部かつ外周境界・内部拘束線から十分離れているか
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
                    if let Some(constraints) = internal_constraints {
                        'seg_loop: for poly in constraints {
                            for seg in poly.windows(2) {
                                if point_to_segment_dist_sq(p, seg[0], seg[1]) < min_boundary_dist_sq {
                                    too_close = true;
                                    break 'seg_loop;
                                }
                            }
                        }
                    }
                }
                if !too_close {
                    let _ = cdt.insert(Point2::new(x as f64, y as f64));
                }
            }
        }
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

    // CDT に保持された全内部拘束エッジ・拘束頂点を収集して Botsch-Kobbelt 反復から保護
    let mut boundary_edges = HashSet::with_capacity(b_count);
    for i in 0..b_count {
        let next = (i + 1) % b_count;
        boundary_edges.insert(RemeshMesh2D::edge_key(i, next));
    }

    let mut internal_constrained_verts = HashSet::new();
    for ch in constraint_handles {
        if let Some(&idx) = handle_to_idx.get(&ch) {
            if idx >= b_count {
                internal_constrained_verts.insert(idx);
            }
        }
    }
    let mut internal_constrained_edges = HashSet::new();
    for edge in cdt.undirected_edges() {
        if edge.is_constraint_edge() {
            let [va, vb] = edge.vertices();
            if let (Some(&ia), Some(&ib)) = (
                handle_to_idx.get(&va.fix()),
                handle_to_idx.get(&vb.fix()),
            ) {
                let key = RemeshMesh2D::edge_key(ia, ib);
                if !boundary_edges.contains(&key) {
                    internal_constrained_edges.insert(key);
                    if ia >= b_count {
                        internal_constrained_verts.insert(ia);
                    }
                    if ib >= b_count {
                        internal_constrained_verts.insert(ib);
                    }
                }
            }
        }
    }

    // 5. Botsch-Kobbelt 等方性リメッシュの実行
    let mut mesh = RemeshMesh2D::new(
        initial_verts,
        b_count,
        initial_tris,
        internal_constrained_verts,
        internal_constrained_edges,
    );

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_internal_constraint_edges_preserved_after_remesh() {
        // 0.2 x 0.2 の正方形外周と中央を横断する内部拘束線 y=0.1 (x=0.04 -> 0.10 -> 0.16)
        let mut boundary = Vec::new();
        let step = 0.02f32;
        let n = 10usize;
        for i in 0..n {
            boundary.push([i as f32 * step, 0.0]);
        }
        for i in 0..n {
            boundary.push([0.2, i as f32 * step]);
        }
        for i in 0..n {
            boundary.push([0.2 - i as f32 * step, 0.2]);
        }
        for i in 0..n {
            boundary.push([0.0, 0.2 - i as f32 * step]);
        }

        let constraint_pts = vec![vec![[0.04, 0.10], [0.10, 0.10], [0.16, 0.10]]];
        let (verts, tris) = remesh_pattern_2d_core(
            &boundary,
            Some(&constraint_pts),
            0.02,
            6,
            0.75,
            1.35,
            0.5,
        )
        .expect("内部拘束付きリメッシュが成功すること");

        assert!(!tris.is_empty());

        // 1. すべての三角形が正面積かつ全辺の共有面数が 2 以下（非多様体なし）であること
        let mut edge_face_counts: HashMap<(u32, u32), usize> = HashMap::new();
        for t in &tris {
            let area = triangle_signed_area_2d(verts[t[0] as usize], verts[t[1] as usize], verts[t[2] as usize]);
            assert!(area > 1e-8, "退化または反転三角形が存在: area={area}");
            for k in 0..3 {
                let a = t[k];
                let b = t[(k + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                *edge_face_counts.entry(key).or_insert(0) += 1;
            }
        }
        for (edge, count) in &edge_face_counts {
            assert!(*count <= 2, "非多様体エッジが生成されました: {:?} count={}", edge, count);
        }

        // 2. 拘束線上の頂点が不動のまま維持され、始点 (0.04, 0.10) から終点 (0.16, 0.10) まで拘束線上の辺列が連結していること
        let mut on_line_adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(a, b) in edge_face_counts.keys() {
            let pa = verts[a as usize];
            let pb = verts[b as usize];
            if (pa[1] - 0.10).abs() < 1e-4
                && (pb[1] - 0.10).abs() < 1e-4
                && pa[0] >= 0.04 - 1e-4
                && pa[0] <= 0.16 + 1e-4
                && pb[0] >= 0.04 - 1e-4
                && pb[0] <= 0.16 + 1e-4
            {
                on_line_adj.entry(a as usize).or_default().push(b as usize);
                on_line_adj.entry(b as usize).or_default().push(a as usize);
            }
        }
        let start_v = verts
            .iter()
            .position(|p| (p[0] - 0.04).abs() < 1e-4 && (p[1] - 0.10).abs() < 1e-4)
            .expect("拘束始点 (0.04, 0.10) が保持されていること");
        let end_v = verts
            .iter()
            .position(|p| (p[0] - 0.16).abs() < 1e-4 && (p[1] - 0.10).abs() < 1e-4)
            .expect("拘束終点 (0.16, 0.10) が保持されていること");
        let _mid_v = verts
            .iter()
            .position(|p| (p[0] - 0.10).abs() < 1e-4 && (p[1] - 0.10).abs() < 1e-4)
            .expect("拘束中間点 (0.10, 0.10) が保持されていること");

        let mut visited = HashSet::new();
        let mut stack = vec![start_v];
        visited.insert(start_v);
        while let Some(cur) = stack.pop() {
            if let Some(nbs) = on_line_adj.get(&cur) {
                for &nb in nbs {
                    if visited.insert(nb) {
                        stack.push(nb);
                    }
                }
            }
        }
        assert!(
            visited.contains(&end_v),
            "リメッシュ後に内部拘束エッジ列が分断・消滅しました"
        );
    }

    #[test]
    fn test_large_coords_and_tiny_target_length_no_infinite_loop() {
        // 大座標オフセット (1e5) + 極小 target_edge_length でも無限ループ・OOM せず完了すること
        let off = 100_000.0f32;
        let boundary = vec![
            [off, off],
            [off + 0.2, off],
            [off + 0.2, off + 0.2],
            [off, off + 0.2],
        ];
        let (verts, tris) = remesh_pattern_2d_core(
            &boundary,
            None,
            1e-6,
            2,
            0.75,
            1.35,
            0.5,
        )
        .expect("大座標・極小辺長でも安全に完了すること");

        assert!(!verts.is_empty());
        assert!(!tris.is_empty());
        assert!(verts.iter().all(|p| p[0].is_finite() && p[1].is_finite()));
    }
}
