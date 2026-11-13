// -*- coding: utf-8 -*-
//! Milestone 5: プロダクション品質 Quad リメッシュ & 3D Barycentric 逆写像 (quad_remesh.rs)
//!
//! シミュレーション済み布メッシュの UV 座標および 3D 変形座標から、
//! 外周 Boundary Quad Belt を保持した Quad トポロジーメッシュを生成し、
//! 重心座標逆写像により 3D ドレープ曲面を高精度に復元する。

use std::collections::HashMap;
use spade::handles::FixedVertexHandle;
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

use crate::geometry_2d::{
    is_point_in_polygon_with_holes, is_point_strictly_inside_polygon_with_holes,
    is_strictly_convex_ccw_quad, point_to_segment_dist_sq, polygon_signed_area_2d,
    segments_properly_intersect_2d, triangle_min_angle_rad, triangle_signed_area_2d,
    winding_number_2d,
};

/// Quad リメッシュ結果構造体
#[derive(Debug, Clone)]
pub struct QuadRemeshOutput {
    pub positions_3d: Vec<[f32; 3]>,
    pub uvs_2d: Vec<[f32; 2]>,
    pub quads: Vec<[u32; 4]>,
    pub triangles: Vec<[u32; 3]>,
    pub boundary_vertex_indices: Vec<u32>,
}

/// 2D 空間ハッシュによる Barycentric 逆写像アクセラレータ
struct BarycentricMapper<'a> {
    pos_3d: &'a [[f32; 3]],
    uv_2d: &'a [[f32; 2]],
    tris: &'a [[u32; 3]],
    min_u: f32,
    min_v: f32,
    cell_size: f32,
    grid_w: usize,
    grid_h: usize,
    grid: Vec<Vec<usize>>,
}

impl<'a> BarycentricMapper<'a> {
    pub fn new(
        pos_3d: &'a [[f32; 3]],
        uv_2d: &'a [[f32; 2]],
        tris: &'a [[u32; 3]],
        target_size: f32,
    ) -> Self {
        let mut min_u = f32::INFINITY;
        let mut max_u = f32::NEG_INFINITY;
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;

        // 未使用頂点（縫い代・他ピース等）でグリッドAABBが膨張しないよう参照三角形の頂点のみ集計
        for tri in tris {
            for &vi in tri {
                if let Some(uv) = uv_2d.get(vi as usize) {
                    if uv[0].is_finite() && uv[1].is_finite() {
                        min_u = min_u.min(uv[0]);
                        max_u = max_u.max(uv[0]);
                        min_v = min_v.min(uv[1]);
                        max_v = max_v.max(uv[1]);
                    }
                }
            }
        }
        if !min_u.is_finite() || !max_u.is_finite() || !min_v.is_finite() || !max_v.is_finite() {
            min_u = 0.0;
            max_u = 1.0;
            min_v = 0.0;
            max_v = 1.0;
        }

        let span_u = (max_u - min_u).max(1e-6);
        let span_v = (max_v - min_v).max(1e-6);
        // 巨大UV範囲や極小目標サイズ指定によるOOMを防ぐため空間グリッド解像度を上限256に抑える
        let min_safe_cell = span_u.max(span_v) / 256.0;
        let cell_size = target_size.max(1e-5).max(min_safe_cell);
        let grid_w = ((span_u / cell_size).ceil() as usize).clamp(1, 256) + 2;
        let grid_h = ((span_v / cell_size).ceil() as usize).clamp(1, 256) + 2;
        let mut grid = vec![Vec::new(); grid_w * grid_h];

        for (ti, tri) in tris.iter().enumerate() {
            let Some(&uva) = uv_2d.get(tri[0] as usize) else { continue; };
            let Some(&uvb) = uv_2d.get(tri[1] as usize) else { continue; };
            let Some(&uvc) = uv_2d.get(tri[2] as usize) else { continue; };

            let t_min_u = uva[0].min(uvb[0]).min(uvc[0]);
            let t_max_u = uva[0].max(uvb[0]).max(uvc[0]);
            let t_min_v = uva[1].min(uvb[1]).min(uvc[1]);
            let t_max_v = uva[1].max(uvb[1]).max(uvc[1]);

            let gx0 = (((t_min_u - min_u) / cell_size).floor() as isize).clamp(0, grid_w as isize - 1) as usize;
            let gx1 = (((t_max_u - min_u) / cell_size).ceil() as isize).clamp(0, grid_w as isize - 1) as usize;
            let gy0 = (((t_min_v - min_v) / cell_size).floor() as isize).clamp(0, grid_h as isize - 1) as usize;
            let gy1 = (((t_max_v - min_v) / cell_size).ceil() as isize).clamp(0, grid_h as isize - 1) as usize;

            for gy in gy0..=gy1 {
                for gx in gx0..=gx1 {
                    grid[gy * grid_w + gx].push(ti);
                }
            }
        }

        Self {
            pos_3d,
            uv_2d,
            tris,
            min_u,
            min_v,
            cell_size,
            grid_w,
            grid_h,
            grid,
        }
    }

    /// 2D UV 座標 (u, v) を包含する三角形を見つけ、3D 座標を重心補間して返す
    pub fn map_uv_to_3d(&self, uv: [f32; 2]) -> [f32; 3] {
        if self.tris.is_empty() || self.pos_3d.is_empty() || self.uv_2d.is_empty() {
            return [0.0, 0.0, 0.0];
        }
        if !uv[0].is_finite() || !uv[1].is_finite() {
            return self.pos_3d[0];
        }

        let gx = (((uv[0] - self.min_u) / self.cell_size).floor() as isize)
            .clamp(0, self.grid_w as isize - 1) as usize;
        let gy = (((uv[1] - self.min_v) / self.cell_size).floor() as isize)
            .clamp(0, self.grid_h as isize - 1) as usize;

        let eps = 1e-4_f32;

        // 1. 同一セルおよび隣接セルを順次探索
        for rad in 0..=3 {
            let x0 = (gx as isize - rad).max(0) as usize;
            let x1 = (gx as isize + rad).min(self.grid_w as isize - 1) as usize;
            let y0 = (gy as isize - rad).max(0) as usize;
            let y1 = (gy as isize + rad).min(self.grid_h as isize - 1) as usize;

            for y in y0..=y1 {
                for x in x0..=x1 {
                    for &ti in &self.grid[y * self.grid_w + x] {
                        let tri = self.tris[ti];
                        let a = self.uv_2d[tri[0] as usize];
                        let b = self.uv_2d[tri[1] as usize];
                        let c = self.uv_2d[tri[2] as usize];

                        let det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
                        if det.abs() < 1e-12 {
                            continue;
                        }

                        let lambda_b = ((uv[0] - a[0]) * (c[1] - a[1]) - (uv[1] - a[1]) * (c[0] - a[0])) / det;
                        let lambda_c = ((b[0] - a[0]) * (uv[1] - a[1]) - (b[1] - a[1]) * (uv[0] - a[0])) / det;
                        let lambda_a = 1.0 - lambda_b - lambda_c;

                        if lambda_a >= -eps && lambda_b >= -eps && lambda_c >= -eps {
                            let la = lambda_a.max(0.0);
                            let lb = lambda_b.max(0.0);
                            let lc = lambda_c.max(0.0);
                            let sum = (la + lb + lc).max(1e-12);
                            let wa = la / sum;
                            let wb = lb / sum;
                            let wc = lc / sum;

                            let pa = self.pos_3d[tri[0] as usize];
                            let pb = self.pos_3d[tri[1] as usize];
                            let pc = self.pos_3d[tri[2] as usize];

                            return [
                                wa * pa[0] + wb * pb[0] + wc * pc[0],
                                wa * pa[1] + wb * pb[1] + wc * pc[1],
                                wa * pa[2] + wb * pb[2] + wc * pc[2],
                            ];
                        }
                    }
                }
            }
        }

        // 2. フォールバック: 全三角形から真の最近接点を持つ三角形を探索
        let mut best_dist_sq = f32::INFINITY;
        let mut best_weights = [1.0_f32, 0.0, 0.0];
        let mut best_tri = self.tris[0];

        for &tri in self.tris {
            let a = self.uv_2d[tri[0] as usize];
            let b = self.uv_2d[tri[1] as usize];
            let c = self.uv_2d[tri[2] as usize];

            let (weights, dist_sq) = closest_barycentric_on_triangle_2d(uv, a, b, c);
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_weights = weights;
                best_tri = tri;
            }
        }

        let pa = self.pos_3d[best_tri[0] as usize];
        let pb = self.pos_3d[best_tri[1] as usize];
        let pc = self.pos_3d[best_tri[2] as usize];

        [
            best_weights[0] * pa[0] + best_weights[1] * pb[0] + best_weights[2] * pc[0],
            best_weights[0] * pa[1] + best_weights[1] * pb[1] + best_weights[2] * pc[1],
            best_weights[0] * pa[2] + best_weights[1] * pb[2] + best_weights[2] * pc[2],
        ]
    }
}

/// 2D三角形上の点 `p` への最近接点に対する重心座標と距離2乗を算出
fn closest_barycentric_on_triangle_2d(
    p: [f32; 2],
    a: [f32; 2],
    b: [f32; 2],
    c: [f32; 2],
) -> ([f32; 3], f32) {
    let det = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if det.abs() >= 1e-12 {
        let lb = ((p[0] - a[0]) * (c[1] - a[1]) - (p[1] - a[1]) * (c[0] - a[0])) / det;
        let lc = ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) / det;
        let la = 1.0 - lb - lc;
        if la >= 0.0 && lb >= 0.0 && lc >= 0.0 {
            return ([la, lb, lc], 0.0);
        }
    }

    // 3辺それぞれへの直交射影で最小距離を採用
    let edges = [(a, b, 0, 1), (b, c, 1, 2), (c, a, 2, 0)];
    let mut best_d2 = f32::INFINITY;
    let mut best_w = [1.0_f32, 0.0, 0.0];
    for (u, v, iu, iv) in edges {
        let dx = v[0] - u[0];
        let dy = v[1] - u[1];
        let l2 = dx * dx + dy * dy;
        let t = if l2 > 1e-20 {
            (((p[0] - u[0]) * dx + (p[1] - u[1]) * dy) / l2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let qx = u[0] + dx * t;
        let qy = u[1] + dy * t;
        let d2 = (p[0] - qx) * (p[0] - qx) + (p[1] - qy) * (p[1] - qy);
        if d2 < best_d2 {
            best_d2 = d2;
            let mut w = [0.0_f32; 3];
            w[iu] = 1.0 - t;
            w[iv] = t;
            best_w = w;
        }
    }
    (best_w, best_d2)
}

/// 入力メッシュと境界ループの整合性・有限値チェック
fn validate_quad_remesh_inputs(
    sim_positions_3d: &[[f32; 3]],
    sim_uvs_2d: &[[f32; 2]],
    sim_triangles: &[[u32; 3]],
    boundary_loop_2d: &[[f32; 2]],
    hole_loops_2d: &[Vec<[f32; 2]>],
    target_quad_size: f32,
) -> Result<(), String> {
    if sim_positions_3d.is_empty() {
        return Err("入力 3D 頂点配列が空です".to_string());
    }
    if sim_positions_3d.len() != sim_uvs_2d.len() {
        return Err(format!(
            "3D 頂点数 ({}) と UV 頂点数 ({}) が一致しません",
            sim_positions_3d.len(),
            sim_uvs_2d.len()
        ));
    }
    if sim_triangles.is_empty() {
        return Err("入力三角形配列が空です".to_string());
    }
    if boundary_loop_2d.len() < 4 {
        return Err("境界ループの頂点数が少なすぎます (4点以上必要)".to_string());
    }
    if !target_quad_size.is_finite() || target_quad_size <= 1e-6 {
        return Err(format!(
            "target_quad_size は正の有限値である必要があります (指定値: {target_quad_size})"
        ));
    }

    for (vi, p) in sim_positions_3d.iter().enumerate() {
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            return Err(format!("頂点 {vi} の 3D 座標に非有限値が含まれています"));
        }
    }
    for (vi, uv) in sim_uvs_2d.iter().enumerate() {
        if !uv[0].is_finite() || !uv[1].is_finite() {
            return Err(format!("頂点 {vi} の UV 座標に非有限値が含まれています"));
        }
    }
    let n_verts = sim_positions_3d.len() as u32;
    for (ti, tri) in sim_triangles.iter().enumerate() {
        if tri[0] >= n_verts || tri[1] >= n_verts || tri[2] >= n_verts {
            return Err(format!(
                "三角形 {ti} が範囲外の頂点インデックス ({:?}, 頂点数 {n_verts}) を参照しています",
                tri
            ));
        }
    }
    for (bi, p) in boundary_loop_2d.iter().enumerate() {
        if !p[0].is_finite() || !p[1].is_finite() {
            return Err(format!("境界頂点 {bi} の UV 座標に非有限値が含まれています"));
        }
    }
    for (hi, hole) in hole_loops_2d.iter().enumerate() {
        if hole.len() < 3 {
            return Err(format!("穴ループ {hi} の頂点数が 3 未満です"));
        }
        for (bi, p) in hole.iter().enumerate() {
            if !p[0].is_finite() || !p[1].is_finite() {
                return Err(format!("穴 {hi} 頂点 {bi} の UV 座標に非有限値が含まれています"));
            }
        }
    }
    Ok(())
}

/// Quad リメッシュ詳細設定オプション
#[derive(Debug, Clone, Copy)]
pub struct QuadRemeshOptions {
    pub target_quad_size: f32,
    pub boundary_belt_loops: u32,
    pub smoothing_iterations: u32,
    pub keep_structured_when_possible: bool,
    pub target_size_in_3d: bool,
}

impl Default for QuadRemeshOptions {
    fn default() -> Self {
        Self {
            target_quad_size: 0.010,
            boundary_belt_loops: 1,
            smoothing_iterations: 10,
            keep_structured_when_possible: true,
            target_size_in_3d: true,
        }
    }
}

/// 入力メッシュの 3D 面積・UV 面積・符号付き UV 面積を集計
fn compute_uv_scale_and_orientation(
    pos_3d: &[[f32; 3]],
    uv_2d: &[[f32; 2]],
    tris: &[[u32; 3]],
) -> (f32, f32, f32) {
    let mut sum_area_3d = 0.0_f64;
    let mut sum_abs_area_uv = 0.0_f64;
    let mut sum_signed_area_uv = 0.0_f64;

    for tri in tris {
        let p0 = pos_3d[tri[0] as usize];
        let p1 = pos_3d[tri[1] as usize];
        let p2 = pos_3d[tri[2] as usize];
        let e1 = [
            (p1[0] - p0[0]) as f64,
            (p1[1] - p0[1]) as f64,
            (p1[2] - p0[2]) as f64,
        ];
        let e2 = [
            (p2[0] - p0[0]) as f64,
            (p2[1] - p0[1]) as f64,
            (p2[2] - p0[2]) as f64,
        ];
        let cx = e1[1] * e2[2] - e1[2] * e2[1];
        let cy = e1[2] * e2[0] - e1[0] * e2[2];
        let cz = e1[0] * e2[1] - e1[1] * e2[0];
        sum_area_3d += 0.5 * (cx * cx + cy * cy + cz * cz).sqrt();

        let uv0 = uv_2d[tri[0] as usize];
        let uv1 = uv_2d[tri[1] as usize];
        let uv2 = uv_2d[tri[2] as usize];
        let area2_uv = ((uv1[0] - uv0[0]) as f64) * ((uv2[1] - uv0[1]) as f64)
            - ((uv1[1] - uv0[1]) as f64) * ((uv2[0] - uv0[0]) as f64);
        let signed_uv = 0.5 * area2_uv;
        sum_abs_area_uv += signed_uv.abs();
        sum_signed_area_uv += signed_uv;
    }

    (
        sum_area_3d as f32,
        sum_abs_area_uv as f32,
        sum_signed_area_uv as f32,
    )
}

/// 境界ループからコーナー頂点（折れ曲がり角 > 45度）を重複なく検出する
fn detect_loop_corners(boundary_loop: &[[f32; 2]]) -> Vec<usize> {
    let b_len = boundary_loop.len();
    if b_len < 4 {
        return Vec::new();
    }

    let mut dot_products = Vec::with_capacity(b_len);
    for i in 0..b_len {
        let prev = boundary_loop[(i + b_len - 1) % b_len];
        let curr = boundary_loop[i];
        let next = boundary_loop[(i + 1) % b_len];

        let v1 = [curr[0] - prev[0], curr[1] - prev[1]];
        let v2 = [next[0] - curr[0], next[1] - curr[1]];
        let l1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt().max(1e-8);
        let l2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt().max(1e-8);

        let cos_theta = ((v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)).clamp(-1.0, 1.0);
        dot_products.push(cos_theta);
    }

    // cos_theta < 0.707 (約 45度以上の折れ曲がり) かつ厳密な局所極小（同値連続は先頭のみ）を検出
    let mut corners = Vec::new();
    for i in 0..b_len {
        let cos_val = dot_products[i];
        if cos_val < 0.707 {
            let prev_val = dot_products[(i + b_len - 1) % b_len];
            let next_val = dot_products[(i + 1) % b_len];
            // 等値プラトーでは先頭側のみ採用して重複登録を防ぐ
            if cos_val < prev_val - 1e-6 && cos_val <= next_val + 1e-6 {
                if let Some(&last) = corners.last() {
                    if i - last <= 1 {
                        continue;
                    }
                }
                corners.push(i);
            }
        }
    }
    if corners.len() >= 2 && corners[0] == 0 && corners[corners.len() - 1] == b_len - 1 {
        corners.pop();
    }

    corners
}

/// ループの平均 UV 辺長を算出
fn average_loop_edge_length(pts: &[[f32; 2]]) -> f32 {
    if pts.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0_f64;
    for i in 0..pts.len() {
        let a = pts[i];
        let b = pts[(i + 1) % pts.len()];
        let dx = (b[0] - a[0]) as f64;
        let dy = (b[1] - a[1]) as f64;
        total += (dx * dx + dy * dy).sqrt();
    }
    (total / pts.len() as f64) as f32
}

/// 既存API（後方互換維持: UV スケール指定・構造化優先）
pub fn remesh_cloth_to_quad_3d_core(
    sim_positions_3d: &[[f32; 3]],
    sim_uvs_2d: &[[f32; 2]],
    sim_triangles: &[[u32; 3]],
    boundary_loop_2d: &[[f32; 2]],
    target_quad_size: f32,
    boundary_belt_loops: u32,
    smoothing_iterations: u32,
) -> Result<QuadRemeshOutput, String> {
    remesh_cloth_to_quad_3d_ext_core(
        sim_positions_3d,
        sim_uvs_2d,
        sim_triangles,
        boundary_loop_2d,
        &[],
        QuadRemeshOptions {
            target_quad_size,
            boundary_belt_loops,
            smoothing_iterations,
            keep_structured_when_possible: true,
            target_size_in_3d: false,
        },
    )
}

/// 拡張 Quad リメッシュ実行関数（穴ループ・3Dスケール換算・ハイブリッド充填対応）
pub fn remesh_cloth_to_quad_3d_ext_core(
    sim_positions_3d: &[[f32; 3]],
    sim_uvs_2d: &[[f32; 2]],
    sim_triangles: &[[u32; 3]],
    boundary_loop_2d: &[[f32; 2]],
    hole_loops_2d: &[Vec<[f32; 2]>],
    options: QuadRemeshOptions,
) -> Result<QuadRemeshOutput, String> {
    validate_quad_remesh_inputs(
        sim_positions_3d,
        sim_uvs_2d,
        sim_triangles,
        boundary_loop_2d,
        hole_loops_2d,
        options.target_quad_size,
    )?;

    let outer_area = polygon_signed_area_2d(boundary_loop_2d);
    if outer_area.abs() < 1e-12 {
        return Err("外周境界ループの面積がゼロ（退化多角形）です".to_string());
    }

    let (sum_area_3d, sum_abs_area_uv, sum_signed_area_uv) =
        compute_uv_scale_and_orientation(sim_positions_3d, sim_uvs_2d, sim_triangles);

    let target_uv_size = if options.target_size_in_3d
        && sum_area_3d > 1e-12
        && sum_abs_area_uv > 1e-12
    {
        let uv_per_3d = (sum_abs_area_uv / sum_area_3d).sqrt();
        (options.target_quad_size * uv_per_3d).max(1e-5)
    } else {
        options.target_quad_size
    };

    let mapper = BarycentricMapper::new(
        sim_positions_3d,
        sim_uvs_2d,
        sim_triangles,
        target_uv_size,
    );

    // 入力境界ループの向きを正規化（外周=CCW, 穴=CW）しつつ元の 0..b_len インデックス写像を保持
    let b_len = boundary_loop_2d.len();
    let is_outer_cw = outer_area < 0.0;
    let mut work_outer = Vec::with_capacity(b_len);
    let mut work_outer_orig_indices = Vec::with_capacity(b_len);
    for k in 0..b_len {
        let orig_i = if is_outer_cw { b_len - 1 - k } else { k };
        work_outer.push(boundary_loop_2d[orig_i]);
        work_outer_orig_indices.push(orig_i as u32);
    }

    let mut work_holes = Vec::with_capacity(hole_loops_2d.len());
    for hole in hole_loops_2d {
        let h_area = polygon_signed_area_2d(hole);
        if h_area.abs() < 1e-12 {
            continue;
        }
        // 穴は CW（左法線が穴の外側＝布内部を向く）に揃える
        let mut h_pts = hole.clone();
        if h_area > 0.0 {
            h_pts.reverse();
        }
        work_holes.push(h_pts);
    }

    // 構造化グリッド経路の適用判定
    if options.keep_structured_when_possible && work_holes.is_empty() {
        let corners = detect_loop_corners(&work_outer);
        if corners.len() == 4 {
            let c0 = corners[0];
            let c1 = corners[1];
            let c2 = corners[2];
            let c3 = corners[3];

            let n0 = c1 - c0;
            let n1 = c2 - c1;
            let n2 = c3 - c2;
            let n3 = b_len - c3 + c0;

            if n0 == n2 && n1 == n3 && n0 >= 1 && n1 >= 1 {
                let p0 = work_outer[c0];
                let p1 = work_outer[c1];
                let p2 = work_outer[c2];
                let p3 = work_outer[c3];

                let avg_edge = average_loop_edge_length(&work_outer);
                let size_ratio = avg_edge / target_uv_size.max(1e-8);
                let is_convex = is_strictly_convex_ccw_quad(p0, p1, p2, p3, 1e-12);

                if is_convex && (0.65..=1.5).contains(&size_ratio) {
                    if let Some(structured) = try_generate_structured_quad_grid(
                        boundary_loop_2d,
                        &work_outer,
                        &work_outer_orig_indices,
                        c0,
                        c1,
                        c2,
                        c3,
                        n0,
                        n1,
                        &mapper,
                        options.smoothing_iterations,
                    ) {
                        let mut out = structured;
                        if sum_signed_area_uv < 0.0 {
                            flip_output_winding(&mut out);
                        }
                        return Ok(out);
                    }
                }
            }
        }
    }

    // 一般多角形・凹形・穴あり向け Hybrid Quad 充填
    let mut out = generate_hybrid_quad_dominant(
        boundary_loop_2d,
        &work_outer,
        &work_outer_orig_indices,
        &work_holes,
        target_uv_size,
        options.boundary_belt_loops,
        options.smoothing_iterations,
        &mapper,
    )?;

    if sum_signed_area_uv < 0.0 {
        flip_output_winding(&mut out);
    }
    Ok(out)
}

/// 元のメッシュが UV 鏡像反転している場合、生成された Quad/Tri の向きを合わせる
fn flip_output_winding(out: &mut QuadRemeshOutput) {
    for q in &mut out.quads {
        *q = [q[0], q[3], q[2], q[1]];
    }
    for t in &mut out.triangles {
        *t = [t[0], t[2], t[1]];
    }
}

/// 4コーナー構造化グリッドの生成・幾何安全検証（領域外・凹Quad時は None でフォールバック）
#[allow(clippy::too_many_arguments)]
fn try_generate_structured_quad_grid(
    orig_boundary_loop: &[[f32; 2]],
    work_outer: &[[f32; 2]],
    work_outer_orig_indices: &[u32],
    c0: usize,
    c1: usize,
    c2: usize,
    c3: usize,
    nu: usize,
    nv: usize,
    mapper: &BarycentricMapper,
    smoothing_iterations: u32,
) -> Option<QuadRemeshOutput> {
    let b_len = work_outer.len();

    let mut bottom = Vec::with_capacity(nu + 1);
    for i in 0..=nu {
        bottom.push(work_outer[(c0 + i) % b_len]);
    }
    let mut right = Vec::with_capacity(nv + 1);
    for j in 0..=nv {
        right.push(work_outer[(c1 + j) % b_len]);
    }
    let mut top = Vec::with_capacity(nu + 1);
    for i in 0..=nu {
        top.push(work_outer[(c2 + i) % b_len]);
    }
    let mut left = Vec::with_capacity(nv + 1);
    for j in 0..=nv {
        left.push(work_outer[(c3 + j) % b_len]);
    }

    let c00 = bottom[0];
    let c10 = bottom[nu];
    let c11 = top[0];
    let c01 = top[nu];

    // Transfinite Interpolation (TFI)
    let mut grid_uv = vec![vec![[0.0_f32, 0.0_f32]; nv + 1]; nu + 1];
    for i in 0..=nu {
        let s = i as f32 / nu as f32;
        for j in 0..=nv {
            let t = j as f32 / nv as f32;
            if j == 0 {
                grid_uv[i][j] = bottom[i];
            } else if j == nv {
                grid_uv[i][j] = top[nu - i];
            } else if i == 0 {
                grid_uv[i][j] = left[nv - j];
            } else if i == nu {
                grid_uv[i][j] = right[j];
            } else {
                let pb = bottom[i];
                let pt = top[nu - i];
                let pl = left[nv - j];
                let pr = right[j];

                let cu = (1.0 - s) * (1.0 - t) * c00[0]
                    + s * (1.0 - t) * c10[0]
                    + s * t * c11[0]
                    + (1.0 - s) * t * c01[0];
                let cv = (1.0 - s) * (1.0 - t) * c00[1]
                    + s * (1.0 - t) * c10[1]
                    + s * t * c11[1]
                    + (1.0 - s) * t * c01[1];

                let u = (1.0 - t) * pb[0] + t * pt[0] + (1.0 - s) * pl[0] + s * pr[0] - cu;
                let v = (1.0 - t) * pb[1] + t * pt[1] + (1.0 - s) * pl[1] + s * pr[1] - cv;
                grid_uv[i][j] = [u, v];
            }
        }
    }

    // 内部頂点のラプラシアン平滑化
    for _ in 0..smoothing_iterations {
        let mut next_uv = grid_uv.clone();
        for i in 1..nu {
            for j in 1..nv {
                let u = 0.25
                    * (grid_uv[i - 1][j][0]
                        + grid_uv[i + 1][j][0]
                        + grid_uv[i][j - 1][0]
                        + grid_uv[i][j + 1][0]);
                let v = 0.25
                    * (grid_uv[i - 1][j][1]
                        + grid_uv[i + 1][j][1]
                        + grid_uv[i][j - 1][1]
                        + grid_uv[i][j + 1][1]);
                next_uv[i][j] = [u, v];
            }
        }
        grid_uv = next_uv;
    }

    // 全内部頂点が厳密に外周ポリゴン内に収まり、全セルが凸 CCW Quad か検証
    for i in 1..nu {
        for j in 1..nv {
            if winding_number_2d(grid_uv[i][j], work_outer) == 0 {
                return None;
            }
        }
    }
    for i in 0..nu {
        for j in 0..nv {
            if !is_strictly_convex_ccw_quad(
                grid_uv[i][j],
                grid_uv[i + 1][j],
                grid_uv[i + 1][j + 1],
                grid_uv[i][j + 1],
                1e-12,
            ) {
                return None;
            }
        }
    }

    // 先頭 0..b_len は入力 orig_boundary_loop と 1対1 一致させる
    let mut out_uvs_2d = Vec::with_capacity((nu + 1) * (nv + 1));
    out_uvs_2d.extend_from_slice(orig_boundary_loop);
    let boundary_vertex_indices: Vec<u32> = (0..b_len as u32).collect();

    let mut grid_idx = vec![vec![0_u32; nv + 1]; nu + 1];
    for i in 0..nu {
        let k = (c0 + i) % b_len;
        grid_idx[i][0] = work_outer_orig_indices[k];
    }
    for j in 0..nv {
        let k = (c1 + j) % b_len;
        grid_idx[nu][j] = work_outer_orig_indices[k];
    }
    for i in 0..nu {
        let k = (c2 + i) % b_len;
        grid_idx[nu - i][nv] = work_outer_orig_indices[k];
    }
    for j in 0..nv {
        let k = (c3 + j) % b_len;
        grid_idx[0][nv - j] = work_outer_orig_indices[k];
    }

    for i in 1..nu {
        for j in 1..nv {
            let idx = out_uvs_2d.len() as u32;
            out_uvs_2d.push(grid_uv[i][j]);
            grid_idx[i][j] = idx;
        }
    }

    let mut quads = Vec::with_capacity(nu * nv);
    for i in 0..nu {
        for j in 0..nv {
            quads.push([
                grid_idx[i][j],
                grid_idx[i + 1][j],
                grid_idx[i + 1][j + 1],
                grid_idx[i][j + 1],
            ]);
        }
    }

    let positions_3d: Vec<[f32; 3]> = out_uvs_2d.iter().map(|uv| mapper.map_uv_to_3d(*uv)).collect();

    Some(QuadRemeshOutput {
        positions_3d,
        uvs_2d: out_uvs_2d,
        quads,
        triangles: Vec::new(),
        boundary_vertex_indices,
    })
}

/// 1ループ分の安全付き Boundary Quad Belt を構築する（外周=CCW、穴=CW 共用で進行左側が領域内）
#[allow(clippy::too_many_arguments)]
fn build_safe_boundary_belt_for_loop(
    loop_pts: &[[f32; 2]],
    loop_indices: &[u32],
    domain_outer: &[[f32; 2]],
    domain_holes: &[Vec<[f32; 2]>],
    obstacle_loops: &[&[[f32; 2]]],
    target_uv_size: f32,
    out_uvs_2d: &mut Vec<[f32; 2]>,
    quads: &mut Vec<[u32; 4]>,
    triangles: &mut Vec<[u32; 3]>,
) -> (Vec<[f32; 2]>, Vec<u32>) {
    let n = loop_pts.len();
    if n < 3 {
        return (loop_pts.to_vec(), loop_indices.to_vec());
    }

    // 1. 各頂点でのマイター方向・最大許容ステップ長を算出
    let mut dirs = Vec::with_capacity(n);
    let mut base_steps = Vec::with_capacity(n);

    for i in 0..n {
        let prev = loop_pts[(i + n - 1) % n];
        let curr = loop_pts[i];
        let next = loop_pts[(i + 1) % n];

        let ep = [curr[0] - prev[0], curr[1] - prev[1]];
        let en = [next[0] - curr[0], next[1] - curr[1]];
        let lp = (ep[0] * ep[0] + ep[1] * ep[1]).sqrt();
        let ln = (en[0] * en[0] + en[1] * en[1]).sqrt();

        if lp < 1e-8 || ln < 1e-8 {
            dirs.push([0.0, 0.0]);
            base_steps.push(0.0);
            continue;
        }

        let tp = [ep[0] / lp, ep[1] / lp];
        let tn = [en[0] / ln, en[1] / ln];
        let np = [-tp[1], tp[0]];
        let nn = [-tn[1], tn[0]];

        let bx = np[0] + nn[0];
        let by = np[1] + nn[1];
        let bl = (bx * bx + by * by).sqrt();

        let (dir, miter_scale) = if bl > 1e-4 {
            let mx = bx / bl;
            let my = by / bl;
            let denom = mx * nn[0] + my * nn[1];
            let scale = if denom > 0.35 {
                (1.0 / denom).min(2.0)
            } else {
                1.0
            };
            ([mx, my], scale)
        } else {
            // 180度折り返し時の特異点保護: 非ゼロ法線または角の二等分内側方向を使う
            let avg_t = [tn[0] - tp[0], tn[1] - tp[1]];
            let al = (avg_t[0] * avg_t[0] + avg_t[1] * avg_t[1]).sqrt();
            if al > 1e-4 {
                ([avg_t[0] / al, avg_t[1] / al], 1.0)
            } else {
                (np, 1.0)
            }
        };

        // 隣接しない境界辺までの最短距離を調べ、反対側壁との衝突を防ぐ (0.42倍上限)
        let mut min_non_adj_d2 = f32::INFINITY;
        for &obs in obstacle_loops {
            let m = obs.len();
            let same_loop = std::ptr::eq(obs.as_ptr(), loop_pts.as_ptr());
            for j in 0..m {
                if same_loop && (j == i || j == (i + n - 1) % n) {
                    continue;
                }
                let a = obs[j];
                let b = obs[(j + 1) % m];
                let d2 = point_to_segment_dist_sq(curr, a, b);
                if d2 < min_non_adj_d2 {
                    min_non_adj_d2 = d2;
                }
            }
        }
        let non_adj_limit = if min_non_adj_d2.is_finite() {
            0.40 * min_non_adj_d2.sqrt()
        } else {
            target_uv_size
        };

        let local_edge = 0.5 * (lp + ln);
        let step = (target_uv_size * miter_scale)
            .min(local_edge * 1.35)
            .min(non_adj_limit);

        dirs.push(dir);
        base_steps.push(step);
    }

    // 2. 段階縮小しつつ非自己交差・領域内・凸Quad条件を満たす最大のベルト幅を決定
    let min_step_thresh = target_uv_size * 0.12;
    let mut best_active = vec![false; n];
    let mut best_pts = loop_pts.to_vec();

    for &shrink in &[1.0_f32, 0.72, 0.5, 0.35, 0.22] {
        let mut active = vec![false; n];
        let mut cand_pts = loop_pts.to_vec();

        for i in 0..n {
            let s = base_steps[i] * shrink;
            if s < min_step_thresh {
                continue;
            }
            let p = [
                loop_pts[i][0] + dirs[i][0] * s,
                loop_pts[i][1] + dirs[i][1] * s,
            ];
            if is_point_strictly_inside_polygon_with_holes(p, domain_outer, domain_holes, 1e-6) {
                active[i] = true;
                cand_pts[i] = p;
            }
        }

        // 各ベルトパネルが厳密凸 CCW Quad / 正面積 Tri になるか検証し、違反する頂点を無効化
        let mut changed = true;
        let mut iter = 0;
        while changed && iter < 4 {
            changed = false;
            iter += 1;
            for i in 0..n {
                let next_i = (i + 1) % n;
                if !active[i] && !active[next_i] {
                    continue;
                }
                let b0 = loop_pts[i];
                let b1 = loop_pts[next_i];
                let valid = if active[i] && active[next_i] {
                    is_strictly_convex_ccw_quad(b0, b1, cand_pts[next_i], cand_pts[i], 1e-12)
                        && quad_angles_within(
                            [b0, b1, cand_pts[next_i], cand_pts[i]],
                            20.0_f32.to_radians(),
                            160.0_f32.to_radians(),
                        )
                } else if active[i] {
                    triangle_signed_area_2d(b0, b1, cand_pts[i]) > 1e-12
                } else {
                    triangle_signed_area_2d(b0, b1, cand_pts[next_i]) > 1e-12
                };
                if !valid {
                    if active[i] {
                        active[i] = false;
                        cand_pts[i] = loop_pts[i];
                        changed = true;
                    }
                    if active[next_i] {
                        active[next_i] = false;
                        cand_pts[next_i] = loop_pts[next_i];
                        changed = true;
                    }
                }
            }
        }

        // 自己交差のない単純ループができているか検査
        let has_self_intersect = has_loop_self_intersection(&cand_pts, 1e-6);
        let any_active = active.iter().any(|&a| a);
        if !has_self_intersect && any_active {
            // ベルト側辺 (b_i -> cand_i) 同士や境界への食い込みも無いこと
            let mut spoke_intersect = false;
            for i in 0..n {
                if !active[i] {
                    continue;
                }
                let a0 = loop_pts[i];
                let a1 = cand_pts[i];
                for j in 0..n {
                    if j == i || j == (i + n - 1) % n {
                        continue;
                    }
                    if segments_properly_intersect_2d(
                        a0,
                        a1,
                        loop_pts[j],
                        loop_pts[(j + 1) % n],
                        1e-6,
                    ) {
                        spoke_intersect = true;
                        break;
                    }
                    if active[j]
                        && segments_properly_intersect_2d(a0, a1, loop_pts[j], cand_pts[j], 1e-6)
                    {
                        spoke_intersect = true;
                        break;
                    }
                }
                if spoke_intersect {
                    break;
                }
            }
            if !spoke_intersect {
                best_active = active;
                best_pts = cand_pts;
                break;
            }
        }
    }

    // 確定した内側ベルト頂点だけを新規登録し、面と新しい CDT 閉ループを構築
    let mut ring_indices = Vec::with_capacity(n);
    for i in 0..n {
        if best_active[i] {
            let idx = out_uvs_2d.len() as u32;
            out_uvs_2d.push(best_pts[i]);
            ring_indices.push(idx);
        } else {
            ring_indices.push(loop_indices[i]);
        }
    }

    for i in 0..n {
        let next_i = (i + 1) % n;
        let b0 = loop_indices[i];
        let b1 = loop_indices[next_i];
        let n0 = ring_indices[i];
        let n1 = ring_indices[next_i];

        if best_active[i] && best_active[next_i] {
            quads.push([b0, b1, n1, n0]);
        } else if best_active[i] {
            triangles.push([b0, b1, n0]);
        } else if best_active[next_i] {
            triangles.push([b0, b1, n1]);
        }
    }

    // 連続重複を除いたユニーク頂点ループを返す（CDT の制約辺用）
    let mut next_pts = Vec::with_capacity(n);
    let mut next_indices = Vec::with_capacity(n);
    for i in 0..n {
        if let Some(&prev_idx) = next_indices.last() {
            if prev_idx == ring_indices[i] {
                continue;
            }
        }
        next_pts.push(best_pts[i]);
        next_indices.push(ring_indices[i]);
    }
    if next_indices.len() > 1 && next_indices.first() == next_indices.last() {
        next_pts.pop();
        next_indices.pop();
    }

    (next_pts, next_indices)
}

/// 閉ループに非隣接辺同士の自己交差があるか判定
fn has_loop_self_intersection(pts: &[[f32; 2]], eps: f32) -> bool {
    let n = pts.len();
    if n < 4 {
        return false;
    }
    for i in 0..n {
        let i_next = (i + 1) % n;
        for j in (i + 2)..n {
            if i == 0 && j == n - 1 {
                continue;
            }
            let j_next = (j + 1) % n;
            if segments_properly_intersect_2d(pts[i], pts[i_next], pts[j], pts[j_next], eps) {
                return true;
            }
        }
    }
    false
}

/// 2つの三角形の対角線統合による Quad 品質スコアを評価（厳密凸・内角・アスペクト基準未達なら None）
fn evaluate_triangle_pair_quad(
    p0: [f32; 2],
    p1: [f32; 2],
    p2: [f32; 2],
    p3: [f32; 2],
    degree_bonus: f32,
) -> Option<f32> {
    if !is_strictly_convex_ccw_quad(p0, p1, p2, p3, 1e-12) {
        return None;
    }

    let pts = [p0, p1, p2, p3];
    let mut lens = [0.0_f32; 4];
    for k in 0..4 {
        let a = pts[k];
        let b = pts[(k + 1) % 4];
        let l = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        if l < 1e-8 {
            return None;
        }
        lens[k] = l;
    }

    let min_e = lens[0].min(lens[1]).min(lens[2]).min(lens[3]);
    let max_e = lens[0].max(lens[1]).max(lens[2]).max(lens[3]);
    let aspect = min_e / max_e;
    if aspect < 1.0 / 3.5 {
        return None;
    }

    // 4つの内角チェック: 22°〜158° をハード制約
    let min_rad = 22.0_f32.to_radians();
    let max_rad = 158.0_f32.to_radians();
    let half_pi = std::f32::consts::FRAC_PI_2;
    let mut max_dev = 0.0_f32;

    for k in 0..4 {
        let prev = pts[(k + 3) % 4];
        let curr = pts[k];
        let next = pts[(k + 1) % 4];
        let v1 = [prev[0] - curr[0], prev[1] - curr[1]];
        let v2 = [next[0] - curr[0], next[1] - curr[1]];
        let l1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt().max(1e-8);
        let l2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt().max(1e-8);
        let cos_ang = ((v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)).clamp(-1.0, 1.0);
        let ang = cos_ang.acos();
        if ang < min_rad || ang > max_rad {
            return None;
        }
        max_dev = max_dev.max((ang - half_pi).abs() / half_pi);
    }

    let angle_quality = (1.0 - max_dev).max(0.0);
    Some(0.70 * angle_quality + 0.30 * aspect + degree_bonus)
}

/// 隣接三角形ペアの最適マッチング + 長さ3の増加路スワップによる Quad 化
fn pair_cdt_triangles_to_quads(
    uvs: &[[f32; 2]],
    cdt_tris: &[[u32; 3]],
) -> (Vec<[u32; 4]>, Vec<[u32; 3]>) {
    let nt = cdt_tris.len();
    if nt == 0 {
        return (Vec::new(), Vec::new());
    }

    // 無向辺 -> 隣接する (三角形インデックス, 辺元始点u, 辺元終点v, 対向頂点w) リスト
    let mut edge_map: HashMap<(u32, u32), Vec<(usize, u32, u32, u32)>> = HashMap::new();
    for (ti, tri) in cdt_tris.iter().enumerate() {
        for e in 0..3 {
            let u = tri[e];
            let v = tri[(e + 1) % 3];
            let w = tri[(e + 2) % 3];
            let key = if u < v { (u, v) } else { (v, u) };
            edge_map.entry(key).or_default().push((ti, u, v, w));
        }
    }

    // 各三角形の隣接三角形数（孤立しやすい低次数三角形のペアリング優遇用）
    let mut tri_degree = vec![0_usize; nt];
    for inc in edge_map.values() {
        if inc.len() == 2 {
            tri_degree[inc[0].0] += 1;
            tri_degree[inc[1].0] += 1;
        }
    }

    #[derive(Clone, Copy)]
    struct PairEdge {
        t1: usize,
        t2: usize,
        quad: [u32; 4],
        score: f32,
    }

    let mut candidates = Vec::new();
    let mut adj_edges: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nt];

    for inc in edge_map.values() {
        if inc.len() != 2 {
            continue;
        }
        let (t1, u1, v1, w1) = inc[0];
        let (t2, u2, v2, w2) = inc[1];
        if t1 == t2 {
            continue;
        }
        // CCW 三角形同士は共有辺が逆向き (u1 == v2 && v1 == u2)
        if !(u1 == v2 && v1 == u2) {
            continue;
        }
        // CCW 順序: u1 -> w2 -> v1 -> w1
        let quad = [u1, w2, v1, w1];
        let deg_sum = tri_degree[t1] + tri_degree[t2];
        let degree_bonus = if deg_sum <= 3 { 0.08 } else if deg_sum == 4 { 0.04 } else { 0.0 };

        if let Some(score) = evaluate_triangle_pair_quad(
            uvs[quad[0] as usize],
            uvs[quad[1] as usize],
            uvs[quad[2] as usize],
            uvs[quad[3] as usize],
            degree_bonus,
        ) {
            let edge_id = candidates.len();
            candidates.push(PairEdge { t1, t2, quad, score });
            adj_edges[t1].push((t2, edge_id));
            adj_edges[t2].push((t1, edge_id));
        }
    }

    let mut sorted_ids: Vec<usize> = (0..candidates.len()).collect();
    sorted_ids.sort_by(|&a, &b| {
        candidates[b]
            .score
            .partial_cmp(&candidates[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 1. Greedy 初期マッチング
    let mut matched_edge: Vec<Option<usize>> = vec![None; nt];
    let mut partner: Vec<Option<usize>> = vec![None; nt];
    for eid in sorted_ids {
        let e = candidates[eid];
        if partner[e.t1].is_none() && partner[e.t2].is_none() {
            partner[e.t1] = Some(e.t2);
            partner[e.t2] = Some(e.t1);
            matched_edge[e.t1] = Some(eid);
            matched_edge[e.t2] = Some(eid);
        }
    }

    // 2. 長さ3の増加路スワップ (未マッチ t - 既マッチ a=b - 未マッチ u -> (t,a) + (b,u) でペア数+1)
    for _ in 0..3 {
        let mut improved = false;
        for t in 0..nt {
            if partner[t].is_some() {
                continue;
            }
            let mut swap_found = None;
            'outer_nbr: for &(a, eid_ta) in &adj_edges[t] {
                let Some(b) = partner[a] else { continue; };
                if b == t {
                    continue;
                }
                for &(u, eid_bu) in &adj_edges[b] {
                    if u != t && partner[u].is_none() {
                        swap_found = Some((a, eid_ta, b, u, eid_bu));
                        break 'outer_nbr;
                    }
                }
            }
            if let Some((a, eid_ta, b, u, eid_bu)) = swap_found {
                partner[t] = Some(a);
                partner[a] = Some(t);
                matched_edge[t] = Some(eid_ta);
                matched_edge[a] = Some(eid_ta);

                partner[b] = Some(u);
                partner[u] = Some(b);
                matched_edge[b] = Some(eid_bu);
                matched_edge[u] = Some(eid_bu);
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }

    let mut out_quads = Vec::new();
    let mut out_tris = Vec::new();
    for ti in 0..nt {
        if let Some(p) = partner[ti] {
            if ti < p {
                if let Some(eid) = matched_edge[ti] {
                    out_quads.push(candidates[eid].quad);
                }
            }
        } else {
            out_tris.push(cdt_tris[ti]);
        }
    }

    (out_quads, out_tris)
}

/// 四角形の全内角が `[min_rad, max_rad]` の範囲に収まるか判定
fn quad_angles_within(pts: [[f32; 2]; 4], min_rad: f32, max_rad: f32) -> bool {
    for k in 0..4 {
        let prev = pts[(k + 3) % 4];
        let curr = pts[k];
        let next = pts[(k + 1) % 4];
        let v1 = [prev[0] - curr[0], prev[1] - curr[1]];
        let v2 = [next[0] - curr[0], next[1] - curr[1]];
        let l1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
        let l2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();
        if l1 < 1e-8 || l2 < 1e-8 {
            return false;
        }
        let cos_ang = ((v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)).clamp(-1.0, 1.0);
        let ang = cos_ang.acos();
        if ang < min_rad || ang > max_rad {
            return false;
        }
    }
    true
}

/// 制約付きラプラシアン平滑化（固定境界保護 ＋ 全隣接面凸性・内角品質・正面積・領域内ロールバック保証）
fn smooth_interior_vertices_guarded(
    uvs: &mut [[f32; 2]],
    fixed: &[bool],
    quads: &[[u32; 4]],
    tris: &[[u32; 3]],
    domain_outer: &[[f32; 2]],
    domain_holes: &[Vec<[f32; 2]>],
    smoothing_iterations: u32,
) {
    if smoothing_iterations == 0 || uvs.is_empty() {
        return;
    }

    let min_quad_rad = 22.0_f32.to_radians();
    let max_quad_rad = 158.0_f32.to_radians();

    let nv = uvs.len();
    let mut adj_verts: Vec<Vec<u32>> = vec![Vec::new(); nv];
    let mut inc_quads: Vec<Vec<usize>> = vec![Vec::new(); nv];
    let mut inc_tris: Vec<Vec<usize>> = vec![Vec::new(); nv];

    let mut add_edge = |u: u32, v: u32| {
        if u != v {
            let ui = u as usize;
            if !adj_verts[ui].contains(&v) {
                adj_verts[ui].push(v);
            }
            let vi = v as usize;
            if !adj_verts[vi].contains(&u) {
                adj_verts[vi].push(u);
            }
        }
    };

    for (qi, q) in quads.iter().enumerate() {
        for k in 0..4 {
            inc_quads[q[k] as usize].push(qi);
            add_edge(q[k], q[(k + 1) % 4]);
        }
    }
    for (ti, t) in tris.iter().enumerate() {
        for k in 0..3 {
            inc_tris[t[k] as usize].push(ti);
            add_edge(t[k], t[(k + 1) % 3]);
        }
    }

    for _ in 0..smoothing_iterations {
        for vi in 0..nv {
            if fixed[vi] || adj_verts[vi].is_empty() {
                continue;
            }
            let mut sum_x = 0.0_f32;
            let mut sum_y = 0.0_f32;
            for &nb in &adj_verts[vi] {
                sum_x += uvs[nb as usize][0];
                sum_y += uvs[nb as usize][1];
            }
            let deg = adj_verts[vi].len() as f32;
            let target = [sum_x / deg, sum_y / deg];
            let orig = uvs[vi];

            let mut orig_quad_ok = vec![true; inc_quads[vi].len()];
            for (idx, &qi) in inc_quads[vi].iter().enumerate() {
                let q = quads[qi];
                orig_quad_ok[idx] = quad_angles_within(
                    [
                        uvs[q[0] as usize],
                        uvs[q[1] as usize],
                        uvs[q[2] as usize],
                        uvs[q[3] as usize],
                    ],
                    min_quad_rad,
                    max_quad_rad,
                );
            }

            // 緩和係数 0.50 -> 0.25 の順で安全判定に合格する位置のみ採用
            for &alpha in &[0.50_f32, 0.25] {
                let cand = [
                    orig[0] + alpha * (target[0] - orig[0]),
                    orig[1] + alpha * (target[1] - orig[1]),
                ];
                if !is_point_strictly_inside_polygon_with_holes(
                    cand,
                    domain_outer,
                    domain_holes,
                    1e-6,
                ) {
                    continue;
                }
                uvs[vi] = cand;
                let mut valid = true;
                for (idx, &qi) in inc_quads[vi].iter().enumerate() {
                    let q = quads[qi];
                    let pts = [
                        uvs[q[0] as usize],
                        uvs[q[1] as usize],
                        uvs[q[2] as usize],
                        uvs[q[3] as usize],
                    ];
                    if !is_strictly_convex_ccw_quad(pts[0], pts[1], pts[2], pts[3], 1e-12) {
                        valid = false;
                        break;
                    }
                    if orig_quad_ok[idx] && !quad_angles_within(pts, min_quad_rad, max_quad_rad) {
                        valid = false;
                        break;
                    }
                }
                if valid {
                    for &ti in &inc_tris[vi] {
                        let t = tris[ti];
                        if triangle_signed_area_2d(
                            uvs[t[0] as usize],
                            uvs[t[1] as usize],
                            uvs[t[2] as usize],
                        ) <= 1e-12
                            || triangle_min_angle_rad(
                                uvs[t[0] as usize],
                                uvs[t[1] as usize],
                                uvs[t[2] as usize],
                            ) < 0.20
                        {
                            valid = false;
                            break;
                        }
                    }
                }
                if valid {
                    break;
                }
                uvs[vi] = orig;
            }
        }
    }
}

/// 一般多角形・穴付き・凹形状向け Hybrid Quad 充填リメッシュ
#[allow(clippy::too_many_arguments)]
fn generate_hybrid_quad_dominant(
    orig_boundary_loop: &[[f32; 2]],
    work_outer: &[[f32; 2]],
    work_outer_orig_indices: &[u32],
    work_holes: &[Vec<[f32; 2]>],
    target_uv_size: f32,
    boundary_belt_loops: u32,
    smoothing_iterations: u32,
    mapper: &BarycentricMapper,
) -> Result<QuadRemeshOutput, String> {
    let b_len = orig_boundary_loop.len();

    // 出力頂点の先頭 0..b_len は入力 boundary_loop と 1対1 一致させる
    let mut out_uvs_2d = Vec::with_capacity(b_len * 4);
    out_uvs_2d.extend_from_slice(orig_boundary_loop);
    let boundary_vertex_indices: Vec<u32> = (0..b_len as u32).collect();

    // 穴頂点を登録
    let mut cur_hole_indices: Vec<Vec<u32>> = Vec::with_capacity(work_holes.len());
    let mut cur_hole_pts: Vec<Vec<[f32; 2]>> = work_holes.to_vec();
    for hole in work_holes {
        let mut h_idx = Vec::with_capacity(hole.len());
        for p in hole {
            h_idx.push(out_uvs_2d.len() as u32);
            out_uvs_2d.push(*p);
        }
        cur_hole_indices.push(h_idx);
    }

    let n_fixed_boundary = out_uvs_2d.len();

    let mut cur_outer_pts = work_outer.to_vec();
    let mut cur_outer_indices = work_outer_orig_indices.to_vec();

    let mut quads = Vec::new();
    let mut belt_triangles = Vec::new();

    // 1. 安全付き Boundary Quad Belt（外周＋穴）を最大 2段まで構築
    let n_belts = boundary_belt_loops.min(2);
    for _ in 0..n_belts {
        let mut next_outer_pts = cur_outer_pts.clone();
        let mut next_outer_indices = cur_outer_indices.clone();

        {
            let mut obs: Vec<&[[f32; 2]]> = Vec::with_capacity(1 + cur_hole_pts.len());
            obs.push(&cur_outer_pts);
            for h in &cur_hole_pts {
                obs.push(h);
            }
            let (np, ni) = build_safe_boundary_belt_for_loop(
                &cur_outer_pts,
                &cur_outer_indices,
                &cur_outer_pts,
                &cur_hole_pts,
                &obs,
                target_uv_size,
                &mut out_uvs_2d,
                &mut quads,
                &mut belt_triangles,
            );
            if np.len() >= 3 && !has_loop_self_intersection(&np, 1e-6) {
                next_outer_pts = np;
                next_outer_indices = ni;
            }
        }

        let mut next_holes_pts: Vec<Vec<[f32; 2]>> = Vec::with_capacity(cur_hole_pts.len());
        let mut next_holes_indices: Vec<Vec<u32>> = Vec::with_capacity(cur_hole_indices.len());
        for hi in 0..cur_hole_pts.len() {
            let mut obs: Vec<&[[f32; 2]]> = Vec::with_capacity(1 + cur_hole_pts.len());
            obs.push(&next_outer_pts);
            for (hj, h) in cur_hole_pts.iter().enumerate() {
                if hj < hi {
                    obs.push(&next_holes_pts[hj]);
                } else {
                    obs.push(h);
                }
            }
            let (np, ni) = build_safe_boundary_belt_for_loop(
                &cur_hole_pts[hi],
                &cur_hole_indices[hi],
                &next_outer_pts,
                &cur_hole_pts,
                &obs,
                target_uv_size,
                &mut out_uvs_2d,
                &mut quads,
                &mut belt_triangles,
            );
            if np.len() >= 3 && !has_loop_self_intersection(&np, 1e-6) {
                next_holes_pts.push(np);
                next_holes_indices.push(ni);
            } else {
                next_holes_pts.push(cur_hole_pts[hi].clone());
                next_holes_indices.push(cur_hole_indices[hi].clone());
            }
        }

        cur_outer_pts = next_outer_pts;
        cur_outer_indices = next_outer_indices;
        cur_hole_pts = next_holes_pts;
        cur_hole_indices = next_holes_indices;
    }

    // 2. 残余領域 AABB と内部正方格子点の生成
    let mut min_u = f32::INFINITY;
    let mut max_u = f32::NEG_INFINITY;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for p in &cur_outer_pts {
        min_u = min_u.min(p[0]);
        max_u = max_u.max(p[0]);
        min_v = min_v.min(p[1]);
        max_v = max_v.max(p[1]);
    }
    let span = (max_u - min_u).max(max_v - min_v).max(1e-6);
    let step = target_uv_size.max(span / 128.0).max(1e-5);
    let min_clear = 0.42 * step;

    let mut interior_grid_pts = Vec::new();
    let mut y = min_v + 0.5 * step;
    while y < max_v {
        let mut x = min_u + 0.5 * step;
        while x < max_u {
            let p = [x, y];
            if is_point_strictly_inside_polygon_with_holes(
                p,
                &cur_outer_pts,
                &cur_hole_pts,
                min_clear,
            ) {
                interior_grid_pts.push(p);
            }
            x += step;
        }
        y += step;
    }

    // 3. spade CDT 構築と頂点ハンドル管理
    let mut cdt = ConstrainedDelaunayTriangulation::<Point2<f64>>::default();
    let mut handle_to_out_idx: HashMap<FixedVertexHandle, u32> = HashMap::new();

    let insert_boundary_loop = |cdt: &mut ConstrainedDelaunayTriangulation<Point2<f64>>,
                                map: &mut HashMap<FixedVertexHandle, u32>,
                                pts: &[[f32; 2]],
                                idxs: &[u32]|
     -> Result<(), String> {
        let m = pts.len();
        if m < 3 {
            return Ok(());
        }
        let mut handles = Vec::with_capacity(m);
        for (&p, &out_idx) in pts.iter().zip(idxs.iter()) {
            let h = cdt
                .insert(Point2::new(p[0] as f64, p[1] as f64))
                .map_err(|e| format!("CDT 境界頂点挿入エラー: {:?}", e))?;
            map.entry(h).or_insert(out_idx);
            handles.push(h);
        }
        for i in 0..m {
            let h0 = handles[i];
            let h1 = handles[(i + 1) % m];
            if h0 != h1 && cdt.can_add_constraint(h0, h1) {
                cdt.add_constraint(h0, h1);
            }
        }
        Ok(())
    };

    insert_boundary_loop(
        &mut cdt,
        &mut handle_to_out_idx,
        &cur_outer_pts,
        &cur_outer_indices,
    )?;
    for (hp, hi) in cur_hole_pts.iter().zip(cur_hole_indices.iter()) {
        insert_boundary_loop(&mut cdt, &mut handle_to_out_idx, hp, hi)?;
    }

    for p in interior_grid_pts {
        if let Ok(h) = cdt.insert(Point2::new(p[0] as f64, p[1] as f64)) {
            handle_to_out_idx.entry(h).or_insert_with(|| {
                let idx = out_uvs_2d.len() as u32;
                out_uvs_2d.push(p);
                idx
            });
        }
    }

    // 4. 残余領域内の CCW 三角形のみ抽出
    let mut cdt_tris = Vec::new();
    for face in cdt.inner_faces() {
        let verts = face.vertices();
        let h0 = verts[0].fix();
        let h1 = verts[1].fix();
        let h2 = verts[2].fix();
        let (Some(&i0), Some(&i1), Some(&i2)) = (
            handle_to_out_idx.get(&h0),
            handle_to_out_idx.get(&h1),
            handle_to_out_idx.get(&h2),
        ) else {
            continue;
        };
        if i0 == i1 || i1 == i2 || i2 == i0 {
            continue;
        }
        let p0 = out_uvs_2d[i0 as usize];
        let p1 = out_uvs_2d[i1 as usize];
        let p2 = out_uvs_2d[i2 as usize];
        let c = [(p0[0] + p1[0] + p2[0]) / 3.0, (p0[1] + p1[1] + p2[1]) / 3.0];
        if !is_point_in_polygon_with_holes(c, &cur_outer_pts, &cur_hole_pts) {
            continue;
        }
        let area = triangle_signed_area_2d(p0, p1, p2);
        if area > 1e-12 {
            cdt_tris.push([i0, i1, i2]);
        } else if area < -1e-12 {
            cdt_tris.push([i0, i2, i1]);
        }
    }

    // 5. 最適三角形ペアリングによる Quad 化
    let (paired_quads, rem_tris) = pair_cdt_triangles_to_quads(&out_uvs_2d, &cdt_tris);
    quads.extend(paired_quads);

    let mut all_tris = belt_triangles;
    all_tris.extend(rem_tris);

    // 6. 制約付きラプラシアン平滑化（外周・穴境界は完全固定）
    let mut fixed = vec![false; out_uvs_2d.len()];
    for f in fixed.iter_mut().take(n_fixed_boundary) {
        *f = true;
    }
    smooth_interior_vertices_guarded(
        &mut out_uvs_2d,
        &fixed,
        &quads,
        &all_tris,
        work_outer,
        work_holes,
        smoothing_iterations,
    );

    // 7. 重心座標逆写像で 3D 座標復元
    let positions_3d: Vec<[f32; 3]> = out_uvs_2d.iter().map(|uv| mapper.map_uv_to_3d(*uv)).collect();

    Ok(QuadRemeshOutput {
        positions_3d,
        uvs_2d: out_uvs_2d,
        quads,
        triangles: all_tris,
        boundary_vertex_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rect_sim_mesh(w: f32, h: f32, nx: usize, ny: usize) -> (Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<[u32; 3]>) {
        let mut pos = Vec::new();
        let mut uvs = Vec::new();
        for j in 0..=ny {
            for i in 0..=nx {
                let u = w * (i as f32 / nx as f32);
                let v = h * (j as f32 / ny as f32);
                pos.push([u, 0.0, v]);
                uvs.push([u, v]);
            }
        }
        let mut tris = Vec::new();
        let stride = nx + 1;
        for j in 0..ny {
            for i in 0..nx {
                let v0 = (j * stride + i) as u32;
                let v1 = v0 + 1;
                let v2 = v0 + stride as u32 + 1;
                let v3 = v0 + stride as u32;
                tris.push([v0, v1, v2]);
                tris.push([v0, v2, v3]);
            }
        }
        (pos, uvs, tris)
    }

    #[test]
    fn test_validation_rejects_empty_and_invalid_inputs() {
        let pos = vec![[0.0, 0.0, 0.0]; 4];
        let uv = vec![[0.0, 0.0]; 4];
        let bnd = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

        // 空の sim_triangles を拒否
        assert!(remesh_cloth_to_quad_3d_core(&pos, &uv, &[], &bnd, 0.1, 1, 5).is_err());

        // 範囲外インデックスを拒否
        assert!(remesh_cloth_to_quad_3d_core(&pos, &uv, &[[0, 1, 99]], &bnd, 0.1, 1, 5).is_err());

        // NaN を拒否
        let bad_pos = vec![[f32::NAN, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(remesh_cloth_to_quad_3d_core(&bad_pos, &uv, &[[0, 1, 2]], &bnd, 0.1, 1, 5).is_err());
    }

    #[test]
    fn test_barycentric_mapper_huge_unused_uv_no_oom() {
        let mut pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let mut uv = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        // 未使用頂点に極大値
        pos.push([1e6, 0.0, 0.0]);
        uv.push([1e6, 1e6]);
        let tris = vec![[0, 1, 2], [0, 2, 3]];
        let mapper = BarycentricMapper::new(&pos, &uv, &tris, 0.001);
        assert!(mapper.grid_w <= 258);
        assert!(mapper.grid_h <= 258);
        let mapped = mapper.map_uv_to_3d([0.5, 0.5]);
        assert!((mapped[0] - 0.5).abs() < 1e-4);
        assert!((mapped[1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_concave_arrow_quad_falls_back_cleanly() {
        let (pos, uv, tris) = make_rect_sim_mesh(2.0, 2.0, 4, 4);
        let arrow = vec![
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
            [1.3, 0.3], [0.6, 0.6],
            [0.3, 1.3], [0.0, 2.0], [0.0, 1.0],
        ];
        let out = remesh_cloth_to_quad_3d_core(&pos, &uv, &tris, &arrow, 0.2, 1, 10).unwrap();
        assert!(!out.quads.is_empty());
        // すべての生成面が正面積で頂点が矢じり多角形内部に収まっていること
        for v in &out.uvs_2d[arrow.len()..] {
            assert!(
                winding_number_2d(*v, &arrow) != 0,
                "凹四角形から頂点が飛び出しました: {:?}",
                v
            );
        }
        for q in &out.quads {
            assert!(is_strictly_convex_ccw_quad(
                out.uvs_2d[q[0] as usize],
                out.uvs_2d[q[1] as usize],
                out.uvs_2d[q[2] as usize],
                out.uvs_2d[q[3] as usize],
                1e-12,
            ));
        }
        for t in &out.triangles {
            assert!(
                triangle_signed_area_2d(
                    out.uvs_2d[t[0] as usize],
                    out.uvs_2d[t[1] as usize],
                    out.uvs_2d[t[2] as usize],
                ) > 0.0
            );
        }
    }

    #[test]
    fn test_cw_u_shape_and_holes_quad_quality() {
        let (pos, uv, tris) = make_rect_sim_mesh(1.0, 1.0, 4, 4);
        // CW U字型（高解像度サブディビジョン済み）
        let mut u_ccw = Vec::new();
        let pts_base: [[f32; 2]; 8] = [
            [0.0, 0.0], [0.4, 0.0], [0.4, 0.6], [0.6, 0.6],
            [0.6, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        ];
        for i in 0..pts_base.len() {
            let a = pts_base[i];
            let b = pts_base[(i + 1) % pts_base.len()];
            let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            let steps = (len / 0.1).round().max(1.0) as usize;
            for k in 0..steps {
                let t = k as f32 / steps as f32;
                u_ccw.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
            }
        }
        let mut u_cw = u_ccw.clone();
        u_cw.reverse();

        let out = remesh_cloth_to_quad_3d_core(&pos, &uv, &tris, &u_cw, 0.1, 1, 8).unwrap();
        let total_faces = out.quads.len() + out.triangles.len();
        let quad_ratio = out.quads.len() as f32 / total_faces as f32;
        assert!(quad_ratio >= 0.75, "Quad 率が低すぎます: {quad_ratio}");
        assert_eq!(out.boundary_vertex_indices.len(), u_cw.len());
        for (i, &vi) in out.boundary_vertex_indices.iter().enumerate() {
            assert_eq!(out.uvs_2d[vi as usize], u_cw[i]);
        }

        // 穴付き矩形 (1.0 x 1.0 外周 + 0.3..0.7 穴)
        let mut sq_outer = Vec::new();
        let sq_base = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        for i in 0..4 {
            let a = sq_base[i];
            let b = sq_base[(i + 1) % 4];
            for k in 0..10 {
                let t = k as f32 / 10.0;
                sq_outer.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
            }
        }
        let mut hole = Vec::new();
        let h_base = [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]];
        for i in 0..4 {
            let a = h_base[i];
            let b = h_base[(i + 1) % 4];
            for k in 0..4 {
                let t = k as f32 / 4.0;
                hole.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
            }
        }
        let out_hole = remesh_cloth_to_quad_3d_ext_core(
            &pos,
            &uv,
            &tris,
            &sq_outer,
            &[hole.clone()],
            QuadRemeshOptions {
                target_quad_size: 0.1,
                boundary_belt_loops: 1,
                smoothing_iterations: 6,
                keep_structured_when_possible: true,
                target_size_in_3d: false,
            },
        )
        .unwrap();
        let hole_ratio = out_hole.quads.len() as f32
            / (out_hole.quads.len() + out_hole.triangles.len()) as f32;
        assert!(hole_ratio >= 0.75, "穴付き矩形の Quad 率が低すぎます: {hole_ratio}");
        for q in &out_hole.quads {
            let c = [
                0.25 * (out_hole.uvs_2d[q[0] as usize][0]
                    + out_hole.uvs_2d[q[1] as usize][0]
                    + out_hole.uvs_2d[q[2] as usize][0]
                    + out_hole.uvs_2d[q[3] as usize][0]),
                0.25 * (out_hole.uvs_2d[q[0] as usize][1]
                    + out_hole.uvs_2d[q[1] as usize][1]
                    + out_hole.uvs_2d[q[2] as usize][1]
                    + out_hole.uvs_2d[q[3] as usize][1]),
            ];
            assert_eq!(winding_number_2d(c, &hole), 0, "穴内部に Quad が生成されました");
        }
    }

    #[test]
    fn test_3d_scale_conversion_and_mirrored_uv() {
        // 3D は 2m x 2m、UV は正規化 [0, 1] かつ鏡像反転 (u を反転)
        let mut pos = Vec::new();
        let mut uvs = Vec::new();
        for j in 0..=4 {
            for i in 0..=4 {
                let x = 2.0 * (i as f32 / 4.0);
                let y = 2.0 * (j as f32 / 4.0);
                pos.push([x, y, 0.0]);
                // u 反転で UV 符号付き面積は負
                uvs.push([1.0 - (i as f32 / 4.0), j as f32 / 4.0]);
            }
        }
        let mut tris = Vec::new();
        for j in 0..4 {
            for i in 0..4 {
                let v0 = (j * 5 + i) as u32;
                let v1 = v0 + 1;
                let v2 = v0 + 6;
                let v3 = v0 + 5;
                tris.push([v0, v1, v2]);
                tris.push([v0, v2, v3]);
            }
        }
        let mut bnd = Vec::new();
        for k in 0..8 {
            bnd.push([k as f32 / 8.0, 0.0]);
        }
        for k in 0..8 {
            bnd.push([1.0, k as f32 / 8.0]);
        }
        for k in 0..8 {
            bnd.push([1.0 - k as f32 / 8.0, 1.0]);
        }
        for k in 0..8 {
            bnd.push([0.0, 1.0 - k as f32 / 8.0]);
        }

        // target_quad_size = 0.25m (3D) -> 2m 辺に 8x8 分割相当になること
        let out = remesh_cloth_to_quad_3d_ext_core(
            &pos,
            &uvs,
            &tris,
            &bnd,
            &[],
            QuadRemeshOptions {
                target_quad_size: 0.25,
                boundary_belt_loops: 1,
                smoothing_iterations: 5,
                keep_structured_when_possible: true,
                target_size_in_3d: true,
            },
        )
        .unwrap();
        assert_eq!(out.quads.len(), 64);
        assert!(out.triangles.is_empty());

        // 全 3D Quad 法線の Z 成分が元メッシュ (+Z) と同方向であること
        for q in &out.quads {
            let p0 = out.positions_3d[q[0] as usize];
            let p1 = out.positions_3d[q[1] as usize];
            let p2 = out.positions_3d[q[2] as usize];
            let nz = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
            assert!(nz > 0.0, "鏡像UV変換後の3D法線が反転しました: nz={nz}");
        }
    }
}
