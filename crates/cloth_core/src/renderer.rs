//! 軽量ソフトウェアZ-bufferレンダラー
//! 表面（表）を白、裏面（裏・裏返り面）を赤、ライティング付きで高速描画し、
//! 縫合エッジ（水色）やコライダー（スレートグレー）の同時描画、
//! 均一1pxワイヤーフレーム描画をサポートします。

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// レンダリング描画オプション
#[derive(Clone, Debug)]
pub struct RenderOptions {
    pub width: u32,
    pub height: u32,
    pub camera_pos: Option<[f32; 3]>,
    pub camera_target: Option<[f32; 3]>,
    pub fov_deg: f32,
    pub bg_color: [u8; 3],
    pub front_color: [u8; 3],
    pub back_color: [u8; 3],
    /// コライダー描画色 (デフォルト: スレートグレー [140, 160, 180])
    pub collider_color: [u8; 3],
    /// 縫合エッジ描画色 (デフォルト: Blender一致の水色・シアン [0, 204, 255])
    pub sewing_color: [u8; 3],
    /// ワイヤーフレームを描画するか
    pub draw_wireframe: bool,
    /// ワイヤーフレームの太さ (ピクセル単位, デフォルト 1.0px)
    pub wire_width_px: f32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            camera_pos: None,
            camera_target: None,
            fov_deg: 45.0,
            bg_color: [35, 35, 35],
            front_color: [255, 255, 255],
            back_color: [255, 0, 0],
            collider_color: [140, 160, 180],
            sewing_color: [0, 204, 255],
            draw_wireframe: true,
            wire_width_px: 1.0,
        }
    }
}

/// レンダリング入力シーンデータ
#[derive(Clone, Debug, Default)]
pub struct SceneData<'a> {
    pub positions: &'a [[f32; 3]],
    pub faces: &'a [[u32; 3]],
    pub sewing_springs: &'a [[u32; 2]],
    pub mesh_triangles: &'a [[f32; 9]], // コライダー三角形 [p0x..z, p1x..z, p2x..z]
    pub vertex_colors: Option<&'a [[u8; 3]]>, // 頂点ごとのRGBカラー (ヒートマップ用)
    pub extra_lines: &'a [[f32; 9]],    // 追加3Dライン [p0x..z, p1x..z, r, g, b] (SDF BBOX等)
}

pub struct RenderResult {
    pub red_pixels: usize,
    pub width: u32,
    pub height: u32,
    pub framebuffer_rgb: Vec<u8>,
}

fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec3_norm(a: [f32; 3]) -> f32 {
    vec3_dot(a, a).sqrt()
}

fn vec3_normalize(a: [f32; 3]) -> [f32; 3] {
    let n = vec3_norm(a);
    if n > 1e-7 {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        [0.0, 0.0, 0.0]
    }
}

type Mat4 = [f32; 16];

fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [0.0; 16];
    for r in 0..4 {
        for c in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[r * 4 + k] * b[k * 4 + c];
            }
            out[r * 4 + c] = sum;
        }
    }
    out
}

fn create_view_matrix(eye: [f32; 3], target: [f32; 3], mut up: [f32; 3]) -> Mat4 {
    let f = vec3_normalize(vec3_sub(target, eye));
    let norm_up = vec3_norm(up);
    let mut up_unit = if norm_up > 1e-7 {
        [up[0] / norm_up, up[1] / norm_up, up[2] / norm_up]
    } else {
        [0.0, 0.0, 1.0]
    };

    if vec3_dot(f, up_unit).abs() > 0.99 {
        up = if f[1].abs() < 0.9 {
            [0.0, 1.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        };
        up_unit = up;
    }

    let s = vec3_normalize(vec3_cross(f, up_unit));
    let u = vec3_cross(s, f);

    [
        s[0], s[1], s[2], -vec3_dot(s, eye),
        u[0], u[1], u[2], -vec3_dot(u, eye),
        -f[0], -f[1], -f[2], vec3_dot(f, eye),
        0.0, 0.0, 0.0, 1.0,
    ]
}

fn create_perspective_matrix(fov_deg: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let fov_rad = fov_deg.to_radians();
    let tan_half_fov = (fov_rad * 0.5).tan();
    let mut m = [0.0; 16];
    m[0] = 1.0 / (aspect * tan_half_fov);
    m[5] = 1.0 / tan_half_fov;
    m[10] = -(far + near) / (far - near);
    m[11] = -(2.0 * far * near) / (far - near);
    m[14] = -1.0;
    m
}

/// スクリーン座標での点 p から線分 ab への最短自乗距離を計算する
#[inline(always)]
fn dist_sq_point_to_segment(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [p[0] - a[0], p[1] - a[1]];
    let ab_len_sq = ab[0] * ab[0] + ab[1] * ab[1];
    if ab_len_sq < 1e-7 {
        return ap[0] * ap[0] + ap[1] * ap[1];
    }
    let t = ((ap[0] * ab[0] + ap[1] * ab[1]) / ab_len_sq).clamp(0.0, 1.0);
    let proj = [a[0] + t * ab[0], a[1] + t * ab[1]];
    let dx = p[0] - proj[0];
    let dy = p[1] - proj[1];
    dx * dx + dy * dy
}

/// 3D空間の頂点をスクリーン座標とNDC深度に変換する
#[inline(always)]
fn project_point(p: [f32; 3], vp: &Mat4, width_f: f32, height_f: f32) -> ([f32; 2], f32, bool) {
    let cx = vp[0] * p[0] + vp[1] * p[1] + vp[2] * p[2] + vp[3];
    let cy = vp[4] * p[0] + vp[5] * p[1] + vp[6] * p[2] + vp[7];
    let cz = vp[8] * p[0] + vp[9] * p[1] + vp[10] * p[2] + vp[11];
    let cw = vp[12] * p[0] + vp[13] * p[1] + vp[14] * p[2] + vp[15];

    if cw <= 1e-6 {
        return ([0.0, 0.0], 1e7, false);
    }

    let inv_w = 1.0 / cw;
    let nx = cx * inv_w;
    let ny = cy * inv_w;
    let nz = cz * inv_w;

    let sx = (nx + 1.0) * 0.5 * (width_f - 1.0);
    let sy = (1.0 - ny) * 0.5 * (height_f - 1.0);

    ([sx, sy], nz, true)
}

/// 3D線分をZバッファテスト付きでラスタライズ描画する
fn draw_line_3d(
    p0: [f32; 2],
    z0: f32,
    p1: [f32; 2],
    z1: f32,
    color: [u8; 3],
    width_px: f32,
    width: u32,
    height: u32,
    fb: &mut [u8],
    z_buffer: &mut [f32],
) {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let dist = (dx * dx + dy * dy).sqrt();
    let steps = (dist * 1.5).ceil() as usize;
    if steps == 0 {
        return;
    }

    let z_bias = -0.000005; // 布表面の上に極わずかに浮かせてZ-fightingを防止（約0.3mm相当）
    let radius = (width_px * 0.5).max(0.5);
    let r_ceil = radius.ceil() as i32;

    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let cx = p0[0] + t * dx;
        let cy = p0[1] + t * dy;
        let pz = (z0 + t * (z1 - z0)) + z_bias;

        for oy in -r_ceil..=r_ceil {
            for ox in -r_ceil..=r_ceil {
                let px = cx.floor() as i32 + ox;
                let py = cy.floor() as i32 + oy;
                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let d_sq = (px as f32 + 0.5 - cx).powi(2) + (py as f32 + 0.5 - cy).powi(2);
                    if d_sq <= radius * radius {
                        let idx = (py as u32 * width + px as u32) as usize;
                        if pz <= z_buffer[idx] {
                            z_buffer[idx] = pz;
                            fb[idx * 3] = color[0];
                            fb[idx * 3 + 1] = color[1];
                            fb[idx * 3 + 2] = color[2];
                        }
                    }
                }
            }
        }
    }
}

/// 単一の三角形をラスタライズ描画する
fn rasterize_triangle(
    p0: [f32; 2],
    z0: f32,
    p1: [f32; 2],
    z1: f32,
    p2: [f32; 2],
    z2: f32,
    norm: [f32; 3],
    _is_front: bool,
    base_color: [u8; 3],
    vert_colors: Option<[[u8; 3]; 3]>,
    light_dir: [f32; 3],
    options: &RenderOptions,
    fb: &mut [u8],
    z_buffer: &mut [f32],
) {
    let width = options.width;
    let height = options.height;

    // バウンディングボックスのクリッピング
    let min_x = (p0[0].min(p1[0]).min(p2[0]).floor() as i32).max(0).min((width - 1) as i32) as u32;
    let max_x = (p0[0].max(p1[0]).max(p2[0]).ceil() as i32).max(0).min((width - 1) as i32) as u32;
    let min_y = (p0[1].min(p1[1]).min(p2[1]).floor() as i32).max(0).min((height - 1) as i32) as u32;
    let max_y = (p0[1].max(p1[1]).max(p2[1]).ceil() as i32).max(0).min((height - 1) as i32) as u32;

    if min_x >= max_x || min_y >= max_y {
        return;
    }

    let denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1]);
    if denom.abs() < 1e-7 {
        return;
    }
    let inv_denom = 1.0 / denom;

    // 半球ディレクショナル・シェーディング
    let mut ndotl = vec3_dot(norm, light_dir).abs();
    ndotl = ndotl * 0.5 + 0.5;

    let default_shaded_color = [
        (base_color[0] as f32 * ndotl).min(255.0) as u8,
        (base_color[1] as f32 * ndotl).min(255.0) as u8,
        (base_color[2] as f32 * ndotl).min(255.0) as u8,
    ];

    let default_wire_color = [
        (default_shaded_color[0] as f32 * 0.45) as u8,
        (default_shaded_color[1] as f32 * 0.45) as u8,
        (default_shaded_color[2] as f32 * 0.45) as u8,
    ];

    let half_w = options.wire_width_px * 0.5;
    let wire_thresh_sq = half_w * half_w;

    for y in min_y..=max_y {
        let py = y as f32 + 0.5;
        let row_offset = (y * width) as usize;
        for x in min_x..=max_x {
            let px = x as f32 + 0.5;

            let w0 = ((p1[1] - p2[1]) * (px - p2[0]) + (p2[0] - p1[0]) * (py - p2[1])) * inv_denom;
            let w1 = ((p2[1] - p0[1]) * (px - p2[0]) + (p0[0] - p2[0]) * (py - p2[1])) * inv_denom;
            let w2 = 1.0 - w0 - w1;

            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                let pz = w0 * z0 + w1 * z1 + w2 * z2;
                let pixel_idx = row_offset + x as usize;
                if pz < z_buffer[pixel_idx] {
                    z_buffer[pixel_idx] = pz;
                    let rgb_idx = pixel_idx * 3;

                    // 頂点カラー補間 (Gouraud Shading) または 単色陰影
                    let (shaded_color, wire_color) = if let Some([c0, c1, c2]) = vert_colors {
                        let r = (w0 * c0[0] as f32 + w1 * c1[0] as f32 + w2 * c2[0] as f32) * ndotl;
                        let g = (w0 * c0[1] as f32 + w1 * c1[1] as f32 + w2 * c2[1] as f32) * ndotl;
                        let b = (w0 * c0[2] as f32 + w1 * c1[2] as f32 + w2 * c2[2] as f32) * ndotl;
                        let sc = [r.min(255.0) as u8, g.min(255.0) as u8, b.min(255.0) as u8];
                        let wc = [
                            (sc[0] as f32 * 0.45) as u8,
                            (sc[1] as f32 * 0.45) as u8,
                            (sc[2] as f32 * 0.45) as u8,
                        ];
                        (sc, wc)
                    } else {
                        (default_shaded_color, default_wire_color)
                    };

                    // ピクセル空間距離による均一エッジ判定（1px）
                    let is_wire = if options.draw_wireframe {
                        let p = [px, py];
                        let d01_sq = dist_sq_point_to_segment(p, p0, p1);
                        let d12_sq = dist_sq_point_to_segment(p, p1, p2);
                        let d20_sq = dist_sq_point_to_segment(p, p2, p0);
                        d01_sq <= wire_thresh_sq || d12_sq <= wire_thresh_sq || d20_sq <= wire_thresh_sq
                    } else {
                        false
                    };

                    let final_pix = if is_wire { wire_color } else { shaded_color };
                    fb[rgb_idx] = final_pix[0];
                    fb[rgb_idx + 1] = final_pix[1];
                    fb[rgb_idx + 2] = final_pix[2];
                }
            }
        }
    }
}

/// シーン全体（布、縫合エッジ、コライダー）をメモリ上のフレームバッファへラスタライズ描画する
pub fn render_scene_to_buffer(scene: &SceneData, options: &RenderOptions) -> RenderResult {
    let width = options.width;
    let height = options.height;
    let total_pixels = (width * height) as usize;

    let has_cloth = !scene.positions.is_empty() && !scene.faces.is_empty();
    let has_colliders = !scene.mesh_triangles.is_empty();

    if !has_cloth && !has_colliders {
        let mut fb = Vec::with_capacity(total_pixels * 3);
        for _ in 0..total_pixels {
            fb.extend_from_slice(&options.bg_color);
        }
        return RenderResult {
            red_pixels: 0,
            width,
            height,
            framebuffer_rgb: fb,
        };
    }

    // AABB と注視点・カメラの自動計算（布とコライダーの両方を含む）
    let mut min_pt = [f32::INFINITY; 3];
    let mut max_pt = [f32::NEG_INFINITY; 3];

    for p in scene.positions {
        min_pt[0] = min_pt[0].min(p[0]);
        min_pt[1] = min_pt[1].min(p[1]);
        min_pt[2] = min_pt[2].min(p[2]);
        max_pt[0] = max_pt[0].max(p[0]);
        max_pt[1] = max_pt[1].max(p[1]);
        max_pt[2] = max_pt[2].max(p[2]);
    }

    for tri in scene.mesh_triangles {
        for v in 0..3 {
            let idx = v * 3;
            min_pt[0] = min_pt[0].min(tri[idx]);
            min_pt[1] = min_pt[1].min(tri[idx + 1]);
            min_pt[2] = min_pt[2].min(tri[idx + 2]);
            max_pt[0] = max_pt[0].max(tri[idx]);
            max_pt[1] = max_pt[1].max(tri[idx + 1]);
            max_pt[2] = max_pt[2].max(tri[idx + 2]);
        }
    }

    if min_pt[0].is_infinite() {
        min_pt = [-1.0, -1.0, -1.0];
        max_pt = [1.0, 1.0, 1.0];
    }

    let center = [
        (min_pt[0] + max_pt[0]) * 0.5,
        (min_pt[1] + max_pt[1]) * 0.5,
        (min_pt[2] + max_pt[2]) * 0.5,
    ];
    let size = (max_pt[0] - min_pt[0])
        .max(max_pt[1] - min_pt[1])
        .max(max_pt[2] - min_pt[2])
        .max(0.1);

    let target = options.camera_target.unwrap_or(center);
    let eye = options.camera_pos.unwrap_or([
        center[0] + size * 0.9,
        center[1] - size * 1.3,
        center[2] + size * 0.8,
    ]);

    let dist = vec3_norm(vec3_sub(eye, target));
    let near = (dist * 0.01).max(0.01);
    let far = dist * 10.0 + size * 5.0;

    let view = create_view_matrix(eye, target, [0.0, 0.0, 1.0]);
    let proj = create_perspective_matrix(options.fov_deg, width as f32 / height as f32, near, far);
    let vp = mat4_mul(&proj, &view);

    let w_f = width as f32;
    let h_f = height as f32;

    // 頂点変換 (ワールド -> スクリーン)
    let mut screen_pts = Vec::with_capacity(scene.positions.len());
    let mut ndc_zs = Vec::with_capacity(scene.positions.len());

    for p in scene.positions {
        let (s, z, _valid) = project_point(*p, &vp, w_f, h_f);
        screen_pts.push(s);
        ndc_zs.push(z);
    }

    let mut fb = vec![options.bg_color[0]; total_pixels * 3];
    for i in 0..total_pixels {
        fb[i * 3] = options.bg_color[0];
        fb[i * 3 + 1] = options.bg_color[1];
        fb[i * 3 + 2] = options.bg_color[2];
    }
    let mut z_buffer = vec![f32::INFINITY; total_pixels];

    // 光源方向 (カメラ斜め上方からのキーライト)
    let light_dir = vec3_normalize([
        eye[0] - target[0] + size * 0.5,
        eye[1] - target[1] - size * 0.5,
        eye[2] - target[2] + size * 1.0,
    ]);

    // 1. コライダーメッシュの描画（スレートグレー）
    for tri in scene.mesh_triangles {
        let p0_w = [tri[0], tri[1], tri[2]];
        let p1_w = [tri[3], tri[4], tri[5]];
        let p2_w = [tri[6], tri[7], tri[8]];

        let (p0_s, z0, v0) = project_point(p0_w, &vp, w_f, h_f);
        let (p1_s, z1, v1) = project_point(p1_w, &vp, w_f, h_f);
        let (p2_s, z2, v2) = project_point(p2_w, &vp, w_f, h_f);
        if !v0 || !v1 || !v2 {
            continue;
        }

        let cross = (p1_s[0] - p0_s[0]) * (p2_s[1] - p0_s[1]) - (p1_s[1] - p0_s[1]) * (p2_s[0] - p0_s[0]);
        let is_front = cross < 0.0;

        let e1 = vec3_sub(p1_w, p0_w);
        let e2 = vec3_sub(p2_w, p0_w);
        let norm = vec3_normalize(vec3_cross(e1, e2));

        rasterize_triangle(
            p0_s, z0, p1_s, z1, p2_s, z2,
            norm, is_front, options.collider_color, None,
            light_dir, options, &mut fb, &mut z_buffer,
        );
    }

    // 2. 布メッシュの描画（表面:白、裏面:赤、または頂点カラー）
    for tri in scene.faces {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        if i0 >= screen_pts.len() || i1 >= screen_pts.len() || i2 >= screen_pts.len() {
            continue;
        }

        let p0 = screen_pts[i0];
        let p1 = screen_pts[i1];
        let p2 = screen_pts[i2];

        let cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
        let is_front = cross < 0.0;
        let base_color = if is_front {
            options.front_color
        } else {
            options.back_color
        };

        let vert_colors = if let Some(vc) = scene.vertex_colors {
            if i0 < vc.len() && i1 < vc.len() && i2 < vc.len() {
                Some([vc[i0], vc[i1], vc[i2]])
            } else {
                None
            }
        } else {
            None
        };

        let p0_w = scene.positions[i0];
        let p1_w = scene.positions[i1];
        let p2_w = scene.positions[i2];
        let e1 = vec3_sub(p1_w, p0_w);
        let e2 = vec3_sub(p2_w, p0_w);
        let norm = vec3_normalize(vec3_cross(e1, e2));

        rasterize_triangle(
            p0, ndc_zs[i0], p1, ndc_zs[i1], p2, ndc_zs[i2],
            norm, is_front, base_color, vert_colors,
            light_dir, options, &mut fb, &mut z_buffer,
        );
    }

    // 3. 縫合エッジ（sewing_springs）の3Dライン描画（水色・シアン）
    for &spring in scene.sewing_springs {
        let v0 = spring[0] as usize;
        let v1 = spring[1] as usize;
        if v0 >= screen_pts.len() || v1 >= screen_pts.len() {
            continue;
        }

        let p0 = screen_pts[v0];
        let z0 = ndc_zs[v0];
        let p1 = screen_pts[v1];
        let z1 = ndc_zs[v1];

        draw_line_3d(
            p0, z0, p1, z1,
            options.sewing_color,
            2.0, // 縫合ライン太さ 2px
            width, height,
            &mut fb, &mut z_buffer,
        );
    }

    // 4. 追加3Dライン（SDF BBOX等）の描画
    for line in scene.extra_lines {
        let p0_w = [line[0], line[1], line[2]];
        let p1_w = [line[3], line[4], line[5]];
        let line_color = [line[6] as u8, line[7] as u8, line[8] as u8];
        let (p0_s, z0, v0) = project_point(p0_w, &vp, w_f, h_f);
        let (p1_s, z1, v1) = project_point(p1_w, &vp, w_f, h_f);
        if v0 && v1 {
            draw_line_3d(
                p0_s, z0, p1_s, z1,
                line_color,
                1.5,
                width, height,
                &mut fb, &mut z_buffer,
            );
        }
    }

    // 赤色ピクセル数（裏面露出面）をカウント
    let mut red_pixels = 0;
    for i in 0..total_pixels {
        let r = fb[i * 3] as u32;
        let g = fb[i * 3 + 1] as u32;
        let b = fb[i * 3 + 2] as u32;
        if r >= 90 && r > (g + b + 1) * 2 {
            red_pixels += 1;
        }
    }

    RenderResult {
        red_pixels,
        width,
        height,
        framebuffer_rgb: fb,
    }
}

/// メッシュのみを描画する互換・簡易関数
pub fn render_mesh_to_buffer(
    positions: &[[f32; 3]],
    faces: &[[u32; 3]],
    options: &RenderOptions,
) -> RenderResult {
    let scene = SceneData {
        positions,
        faces,
        sewing_springs: &[],
        mesh_triangles: &[],
        vertex_colors: None,
        extra_lines: &[],
    };
    render_scene_to_buffer(&scene, options)
}

/// シーンを描画し、PNGファイルとして保存する
pub fn render_scene_to_png_file(
    filepath: &str,
    scene: &SceneData,
    options: &RenderOptions,
) -> Result<RenderResult, String> {
    let result = render_scene_to_buffer(scene, options);

    let path = Path::new(filepath);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let file = File::create(path).map_err(|e| e.to_string())?;
    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, result.width, result.height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);

    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    writer
        .write_image_data(&result.framebuffer_rgb)
        .map_err(|e| e.to_string())?;

    Ok(result)
}

/// メッシュを描画し、PNGファイルとして保存する互換関数
pub fn render_mesh_to_png_file(
    filepath: &str,
    positions: &[[f32; 3]],
    faces: &[[u32; 3]],
    options: &RenderOptions,
) -> Result<RenderResult, String> {
    let scene = SceneData {
        positions,
        faces,
        sewing_springs: &[],
        mesh_triangles: &[],
        vertex_colors: None,
        extra_lines: &[],
    };
    render_scene_to_png_file(filepath, &scene, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rasterizer_front_back() {
        let options = RenderOptions {
            width: 100,
            height: 100,
            camera_pos: Some([0.0, 0.0, 3.0]),
            camera_target: Some([0.0, 0.0, 0.0]),
            fov_deg: 45.0,
            ..Default::default()
        };

        // 1. 表面（CCW）
        let pos_front = vec![
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces = vec![[0, 1, 2]];
        let res_front = render_mesh_to_buffer(&pos_front, &faces, &options);
        assert_eq!(res_front.red_pixels, 0, "表面描画で赤色ピクセルが検出されました");

        // 2. 裏面（CW）
        let pos_back = vec![
            [-1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ];
        let faces = vec![[0, 1, 2]];
        let res_back = render_mesh_to_buffer(&pos_back, &faces, &options);
        assert!(res_back.red_pixels > 50, "裏面描画で赤色ピクセルが検出されませんでした");
    }

    #[test]
    fn test_sewing_and_collider_render() {
        let options = RenderOptions {
            width: 100,
            height: 100,
            camera_pos: Some([0.0, 0.0, 4.0]),
            camera_target: Some([0.0, 0.0, 0.0]),
            sewing_color: [0, 204, 255],
            collider_color: [140, 160, 180],
            ..Default::default()
        };

        let positions = vec![
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 1.5, 0.0],
        ];
        let faces = vec![[0, 1, 2]];
        let sewing_springs = vec![[2, 3]]; // 頂点2と3を結ぶ縫合ライン

        // メッシュコライダー三角形
        let mesh_triangles = vec![
            [-1.5, -1.5, -0.5,  -0.5, -1.5, -0.5,  -1.0, -0.5, -0.5],
        ];

        // 追加3Dライン (SDF BBOX線など: ゴールド [255, 170, 0])
        let extra_lines = vec![
            [0.5, 0.5, 0.0, 1.5, 1.5, 0.0, 255.0, 170.0, 0.0],
        ];

        let scene = SceneData {
            positions: &positions,
            faces: &faces,
            sewing_springs: &sewing_springs,
            mesh_triangles: &mesh_triangles,
            vertex_colors: None,
            extra_lines: &extra_lines,
        };

        let res = render_scene_to_buffer(&scene, &options);
        assert_eq!(res.width, 100);
        assert_eq!(res.height, 100);

        // 縫合色 (0, 204, 255) および追加ライン色 (255, 170, 0) のピクセルが存在することを確認
        let mut found_sewing = false;
        let mut found_collider = false;
        let mut found_extra_line = false;
        for i in 0..(100 * 100) {
            let r = res.framebuffer_rgb[i * 3];
            let g = res.framebuffer_rgb[i * 3 + 1];
            let b = res.framebuffer_rgb[i * 3 + 2];

            if r == 0 && g == 204 && b == 255 {
                found_sewing = true;
            }
            if r == 255 && g == 170 && b == 0 {
                found_extra_line = true;
            }
            // コライダー色 [140, 160, 180] (シェーディングで明暗あり)
            if (r as i32 - g as i32).abs() < 25 && b > r && b > 80 {
                found_collider = true;
            }
        }

        assert!(found_sewing, "縫合エッジ（水色）がレンダリングされていません");
        assert!(found_collider, "メッシュコライダーがレンダリングされていません");
        assert!(found_extra_line, "追加3Dライン（ゴールド）がレンダリングされていません");
    }

    #[test]
    fn test_vertex_colors_render() {
        let options = RenderOptions {
            width: 100,
            height: 100,
            camera_pos: Some([0.0, 0.0, 3.0]),
            camera_target: Some([0.0, 0.0, 0.0]),
            draw_wireframe: false,
            ..Default::default()
        };

        let positions = vec![
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces = vec![[0, 1, 2]];
        // 頂点カラー: 赤、緑、青
        let v_colors = vec![
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ];

        let scene = SceneData {
            positions: &positions,
            faces: &faces,
            sewing_springs: &[],
            mesh_triangles: &[],
            vertex_colors: Some(&v_colors),
            extra_lines: &[],
        };

        let res = render_scene_to_buffer(&scene, &options);
        // 緑成分や青成分が優勢なピクセルが存在することを確認（頂点カラー補間が効いている）
        let mut found_greenish = false;
        let mut found_blueish = false;
        for i in 0..(100 * 100) {
            let r = res.framebuffer_rgb[i * 3];
            let g = res.framebuffer_rgb[i * 3 + 1];
            let b = res.framebuffer_rgb[i * 3 + 2];
            if g > 150 && g > r && g > b {
                found_greenish = true;
            }
            if b > 150 && b > r && b > g {
                found_blueish = true;
            }
        }
        assert!(found_greenish, "頂点カラー補間の緑ピクセルが検出されませんでした");
        assert!(found_blueish, "頂点カラー補間の青ピクセルが検出されませんでした");
    }

    #[test]
    fn test_rasterizer_pixel_width() {
        let options = RenderOptions {
            width: 100,
            height: 100,
            camera_pos: Some([0.0, 0.0, 2.0]),
            camera_target: Some([0.0, 0.0, 0.0]),
            wire_width_px: 1.0,
            draw_wireframe: true,
            ..Default::default()
        };

        // 細長いスライバー三角形 (アスペクト比 1:50)
        let pos_sliver = vec![
            [-0.01, -0.5, 0.0],
            [0.01, -0.5, 0.0],
            [0.0, 0.5, 0.0],
        ];
        let faces = vec![[0, 1, 2]];
        let res = render_mesh_to_buffer(&pos_sliver, &faces, &options);

        // 背景以外のピクセルが極端に黒塗りつぶし（太エッジ）にならず、正常にレンダリングされること
        let mut non_bg_count = 0;
        for i in 0..(100 * 100) {
            let r = res.framebuffer_rgb[i * 3];
            let g = res.framebuffer_rgb[i * 3 + 1];
            let b = res.framebuffer_rgb[i * 3 + 2];
            if [r, g, b] != options.bg_color {
                non_bg_count += 1;
            }
        }
        assert!(non_bg_count > 10 && non_bg_count < 200, "細長い三角形の描画面積が異常です: {}", non_bg_count);
    }
}
