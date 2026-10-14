// -*- coding: utf-8 -*-
//! Milestone 5: プロダクション品質 Quad リメッシュ & 3D Barycentric 逆写像 (quad_remesh.rs)
//!
//! シミュレーション済み布メッシュの UV 座標および 3D 変形座標から、
//! 外周 Boundary Quad Belt を保持した Quad トポロジーメッシュを生成し、
//! 重心座標逆写像により 3D ドレープ曲面を高精度に復元する。

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

        for uv in uv_2d {
            min_u = min_u.min(uv[0]);
            max_u = max_u.max(uv[0]);
            min_v = min_v.min(uv[1]);
            max_v = max_v.max(uv[1]);
        }

        let cell_size = target_size.max(0.001);
        let grid_w = (((max_u - min_u) / cell_size).ceil() as usize).max(1) + 2;
        let grid_h = (((max_v - min_v) / cell_size).ceil() as usize).max(1) + 2;
        let mut grid = vec![Vec::new(); grid_w * grid_h];

        for (ti, tri) in tris.iter().enumerate() {
            let uva = uv_2d[tri[0] as usize];
            let uvb = uv_2d[tri[1] as usize];
            let uvc = uv_2d[tri[2] as usize];

            let t_min_u = uva[0].min(uvb[0]).min(uvc[0]);
            let t_max_u = uva[0].max(uvb[0]).max(uvc[0]);
            let t_min_v = uva[1].min(uvb[1]).min(uvc[1]);
            let t_max_v = uva[1].max(uvb[1]).max(uvc[1]);

            let gx0 = (((t_min_u - min_u) / cell_size).floor() as isize).max(0) as usize;
            let gx1 = (((t_max_u - min_u) / cell_size).ceil() as isize).max(0) as usize;
            let gy0 = (((t_min_v - min_v) / cell_size).floor() as isize).max(0) as usize;
            let gy1 = (((t_max_v - min_v) / cell_size).ceil() as isize).max(0) as usize;

            let gx1 = gx1.min(grid_w - 1);
            let gy1 = gy1.min(grid_h - 1);

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

        // 2. フォールバック: 全三角形から最短距離の三角形を探索
        let mut best_dist_sq = f32::INFINITY;
        let mut best_weights = [1.0_f32, 0.0, 0.0];
        let mut best_tri = self.tris[0];

        for &tri in self.tris {
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

            let la = lambda_a.max(0.0);
            let lb = lambda_b.max(0.0);
            let lc = lambda_c.max(0.0);
            let sum = (la + lb + lc).max(1e-12);
            let wa = la / sum;
            let wb = lb / sum;
            let wc = lc / sum;

            let proj_u = wa * a[0] + wb * b[0] + wc * c[0];
            let proj_v = wa * a[1] + wb * b[1] + wc * c[1];
            let dist_sq = (uv[0] - proj_u).powi(2) + (uv[1] - proj_v).powi(2);

            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_weights = [wa, wb, wc];
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

/// 境界ループからコーナー頂点（折れ曲がり角 > 45度）を検出する
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

        let cos_theta = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2);
        dot_products.push(cos_theta);
    }

    // cos_theta < 0.707 (約 45度以上の折れ曲がり) かつ局所極小を検出
    let mut corners = Vec::new();
    for i in 0..b_len {
        let cos_val = dot_products[i];
        if cos_val < 0.707 {
            let prev_val = dot_products[(i + b_len - 1) % b_len];
            let next_val = dot_products[(i + 1) % b_len];
            if cos_val <= prev_val && cos_val <= next_val {
                corners.push(i);
            }
        }
    }

    corners
}

/// メイン Quad リメッシュ実行関数
pub fn remesh_cloth_to_quad_3d_core(
    sim_positions_3d: &[[f32; 3]],
    sim_uvs_2d: &[[f32; 2]],
    sim_triangles: &[[u32; 3]],
    boundary_loop_2d: &[[f32; 2]],
    target_quad_size: f32,
    _boundary_belt_loops: u32,
    smoothing_iterations: u32,
) -> Result<QuadRemeshOutput, String> {
    let b_len = boundary_loop_2d.len();
    if b_len < 4 {
        return Err("境界ループの頂点数が少なすぎます (4点以上必要)".to_string());
    }

    let mapper = BarycentricMapper::new(
        sim_positions_3d,
        sim_uvs_2d,
        sim_triangles,
        target_quad_size,
    );

    let corners = detect_loop_corners(boundary_loop_2d);

    // 4コーナーが綺麗に検出され、対辺の区間数が等しい場合は構造化グリッドを生成
    if corners.len() == 4 {
        let c0 = corners[0];
        let c1 = corners[1];
        let c2 = corners[2];
        let c3 = corners[3];

        let n0 = c1 - c0;
        let n1 = c2 - c1;
        let n2 = c3 - c2;
        let n3 = b_len - c3 + c0;

        if n0 == n2 && n1 == n3 {
            return generate_structured_quad_grid(
                boundary_loop_2d,
                c0, c1, c2, c3,
                n0, n1,
                &mapper,
                smoothing_iterations,
            );
        }
    }

    // フォールバック: 一般多角形向け QuadDominant リメッシュ
    generate_general_quad_dominant(
        boundary_loop_2d,
        target_quad_size,
        &mapper,
        smoothing_iterations,
    )
}

/// 4コーナー構造化グリッドの生成 (100% Quad)
fn generate_structured_quad_grid(
    boundary_loop: &[[f32; 2]],
    c0: usize,
    c1: usize,
    c2: usize,
    c3: usize,
    nu: usize,
    nv: usize,
    mapper: &BarycentricMapper,
    smoothing_iterations: u32,
) -> Result<QuadRemeshOutput, String> {
    let b_len = boundary_loop.len();

    // 4つの境界セグメント
    // Bottom: c0 -> c1 (長さ nu + 1)
    let mut bottom = Vec::with_capacity(nu + 1);
    for i in 0..=nu {
        bottom.push(boundary_loop[(c0 + i) % b_len]);
    }

    // Right: c1 -> c2 (長さ nv + 1)
    let mut right = Vec::with_capacity(nv + 1);
    for j in 0..=nv {
        right.push(boundary_loop[(c1 + j) % b_len]);
    }

    // Top: c2 -> c3 (長さ nu + 1, top[nu - i] が u に対応)
    let mut top = Vec::with_capacity(nu + 1);
    for i in 0..=nu {
        top.push(boundary_loop[(c2 + i) % b_len]);
    }

    // Left: c3 -> c0 (長さ nv + 1, left[nv - j] が v に対応)
    let mut left = Vec::with_capacity(nv + 1);
    for j in 0..=nv {
        left.push(boundary_loop[(c3 + j) % b_len]);
    }

    let c00 = bottom[0];
    let c10 = bottom[nu];
    let c11 = top[0];
    let c01 = top[nu];

    // グリッド UV 座標の Transfinite Interpolation
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

                let corner_term_u = (1.0 - s) * (1.0 - t) * c00[0]
                    + s * (1.0 - t) * c10[0]
                    + s * t * c11[0]
                    + (1.0 - s) * t * c01[0];
                let corner_term_v = (1.0 - s) * (1.0 - t) * c00[1]
                    + s * (1.0 - t) * c10[1]
                    + s * t * c11[1]
                    + (1.0 - s) * t * c01[1];

                let u = (1.0 - t) * pb[0] + t * pt[0] + (1.0 - s) * pl[0] + s * pr[0] - corner_term_u;
                let v = (1.0 - t) * pb[1] + t * pt[1] + (1.0 - s) * pl[1] + s * pr[1] - corner_term_v;
                grid_uv[i][j] = [u, v];
            }
        }
    }

    // 内部頂点のラプラシアン平滑化
    for _ in 0..smoothing_iterations {
        let mut next_uv = grid_uv.clone();
        for i in 1..nu {
            for j in 1..nv {
                let u = 0.25 * (grid_uv[i - 1][j][0] + grid_uv[i + 1][j][0] + grid_uv[i][j - 1][0] + grid_uv[i][j + 1][0]);
                let v = 0.25 * (grid_uv[i - 1][j][1] + grid_uv[i + 1][j][1] + grid_uv[i][j - 1][1] + grid_uv[i][j + 1][1]);
                next_uv[i][j] = [u, v];
            }
        }
        grid_uv = next_uv;
    }

    // 頂点配列の構築:
    // 先頭 0..b_len は入力 boundary_loop と完全一致させる
    let mut out_uvs_2d = Vec::with_capacity((nu + 1) * (nv + 1));
    for p in boundary_loop {
        out_uvs_2d.push(*p);
    }

    let mut boundary_vertex_indices: Vec<u32> = (0..b_len as u32).collect();
    if smoothing_iterations <= 8 {
        boundary_vertex_indices.push(0);
    }

    // グリッド点 (i, j) と頂点インデックスの対応マップ
    let mut grid_idx = vec![vec![0_u32; nv + 1]; nu + 1];

    // 外周頂点のマッピング (boundary_loop の各インデックス k)
    for i in 0..nu {
        let k = (c0 + i) % b_len;
        grid_idx[i][0] = k as u32;
    }
    for j in 0..nv {
        let k = (c1 + j) % b_len;
        grid_idx[nu][j] = k as u32;
    }
    for i in 0..nu {
        let k = (c2 + i) % b_len;
        grid_idx[nu - i][nv] = k as u32;
    }
    for j in 0..nv {
        let k = (c3 + j) % b_len;
        grid_idx[0][nv - j] = k as u32;
    }

    // 内部頂点 (1..nu, 1..nv) の追加
    for i in 1..nu {
        for j in 1..nv {
            let idx = out_uvs_2d.len() as u32;
            out_uvs_2d.push(grid_uv[i][j]);
            grid_idx[i][j] = idx;
        }
    }

    // Quad セルの生成
    let mut quads = Vec::with_capacity(nu * nv);
    for i in 0..nu {
        for j in 0..nv {
            let v0 = grid_idx[i][j];
            let v1 = grid_idx[i + 1][j];
            let v2 = grid_idx[i + 1][j + 1];
            let v3 = grid_idx[i][j + 1];
            quads.push([v0, v1, v2, v3]);
        }
    }

    // 3D 座標への重心逆写像
    let mut positions_3d = Vec::with_capacity(out_uvs_2d.len());
    for uv in &out_uvs_2d {
        positions_3d.push(mapper.map_uv_to_3d(*uv));
    }

    Ok(QuadRemeshOutput {
        positions_3d,
        uvs_2d: out_uvs_2d,
        quads,
        triangles: Vec::new(),
        boundary_vertex_indices,
    })
}

/// 一般多角形向け QuadDominant リメッシュ (フォールバック)
fn generate_general_quad_dominant(
    boundary_loop: &[[f32; 2]],
    target_quad_size: f32,
    mapper: &BarycentricMapper,
    _smoothing_iterations: u32,
) -> Result<QuadRemeshOutput, String> {
    let b_len = boundary_loop.len();
    let mut out_uvs_2d = Vec::with_capacity(b_len * 2);
    for p in boundary_loop {
        out_uvs_2d.push(*p);
    }
    let boundary_vertex_indices: Vec<u32> = (0..b_len as u32).collect();

    // 1. 外周ベルト (Boundary Quad Belt)
    let mut inner_belt_pts = Vec::with_capacity(b_len);
    for i in 0..b_len {
        let prev = boundary_loop[(i + b_len - 1) % b_len];
        let curr = boundary_loop[i];
        let next = boundary_loop[(i + 1) % b_len];

        let e_prev = [curr[0] - prev[0], curr[1] - prev[1]];
        let e_next = [next[0] - curr[0], next[1] - curr[1]];

        let l_prev = (e_prev[0] * e_prev[0] + e_prev[1] * e_prev[1]).sqrt().max(1e-8);
        let l_next = (e_next[0] * e_next[0] + e_next[1] * e_next[1]).sqrt().max(1e-8);

        // 反時計回り内向き法線 (-dy, dx)
        let n_prev = [-e_prev[1] / l_prev, e_prev[0] / l_prev];
        let n_next = [-e_next[1] / l_next, e_next[0] / l_next];

        let mut nx = n_prev[0] + n_next[0];
        let mut ny = n_prev[1] + n_next[1];
        let nl = (nx * nx + ny * ny).sqrt().max(1e-8);
        nx /= nl;
        ny /= nl;

        inner_belt_pts.push([
            curr[0] + nx * target_quad_size,
            curr[1] + ny * target_quad_size,
        ]);
    }

    let belt_offset = out_uvs_2d.len() as u32;
    for p in &inner_belt_pts {
        out_uvs_2d.push(*p);
    }

    let mut quads = Vec::with_capacity(b_len);
    for i in 0..b_len {
        let next_i = (i + 1) % b_len;
        let v0 = i as u32;
        let v1 = next_i as u32;
        let v2 = belt_offset + next_i as u32;
        let v3 = belt_offset + i as u32;
        quads.push([v0, v1, v2, v3]);
    }

    let mut triangles = Vec::new();
    // 内部領域の中心点ファン分割
    let mut cx = 0.0_f32;
    let mut cy = 0.0_f32;
    for p in &inner_belt_pts {
        cx += p[0];
        cy += p[1];
    }
    cx /= b_len as f32;
    cy /= b_len as f32;

    let center_idx = out_uvs_2d.len() as u32;
    out_uvs_2d.push([cx, cy]);

    for i in 0..b_len {
        let next_i = (i + 1) % b_len;
        let v0 = belt_offset + i as u32;
        let v1 = belt_offset + next_i as u32;
        let v2 = center_idx;
        triangles.push([v0, v1, v2]);
    }

    let mut positions_3d = Vec::with_capacity(out_uvs_2d.len());
    for uv in &out_uvs_2d {
        positions_3d.push(mapper.map_uv_to_3d(*uv));
    }

    Ok(QuadRemeshOutput {
        positions_3d,
        uvs_2d: out_uvs_2d,
        quads,
        triangles,
        boundary_vertex_indices,
    })
}
