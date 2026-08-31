//! 軽量ソフトウェアZ-bufferレンダラー
//! 表面（表）を白、裏面（裏・裏返り面）を赤、ライティングなし(Unlit)で高速描画し、
//! PNG画像ファイルを出力するとともに赤色ピクセル数を返却します。

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

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
        }
    }
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

    // 視線方向 f と up がほぼ平行な場合は別軸を仮の up とする
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

/// メッシュをメモリ上のフレームバッファへラスタライズ描画する
pub fn render_mesh_to_buffer(
    positions: &[[f32; 3]],
    faces: &[[u32; 3]],
    options: &RenderOptions,
) -> RenderResult {
    let width = options.width;
    let height = options.height;
    let total_pixels = (width * height) as usize;

    if positions.is_empty() || faces.is_empty() {
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

    // AABB と注視点・カメラの自動計算
    let mut min_pt = positions[0];
    let mut max_pt = positions[0];
    for p in positions {
        min_pt[0] = min_pt[0].min(p[0]);
        min_pt[1] = min_pt[1].min(p[1]);
        min_pt[2] = min_pt[2].min(p[2]);
        max_pt[0] = max_pt[0].max(p[0]);
        max_pt[1] = max_pt[1].max(p[1]);
        max_pt[2] = max_pt[2].max(p[2]);
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
        center[0] + size * 1.5,
        center[1] - size * 2.0,
        center[2] + size * 1.4,
    ]);

    let dist = vec3_norm(vec3_sub(eye, target));
    let near = (dist * 0.01).max(0.01);
    let far = dist * 10.0 + size * 5.0;

    let view = create_view_matrix(eye, target, [0.0, 0.0, 1.0]);
    let proj = create_perspective_matrix(options.fov_deg, width as f32 / height as f32, near, far);
    let vp = mat4_mul(&proj, &view);

    // 頂点変換 (ワールド -> クリップ -> NDC -> スクリーン)
    let mut screen_pts = Vec::with_capacity(positions.len());
    let mut ndc_zs = Vec::with_capacity(positions.len());

    let w_f = width as f32;
    let h_f = height as f32;

    for p in positions {
        let cx = vp[0] * p[0] + vp[1] * p[1] + vp[2] * p[2] + vp[3];
        let cy = vp[4] * p[0] + vp[5] * p[1] + vp[6] * p[2] + vp[7];
        let cz = vp[8] * p[0] + vp[9] * p[1] + vp[10] * p[2] + vp[11];
        let cw = vp[12] * p[0] + vp[13] * p[1] + vp[14] * p[2] + vp[15];

        let inv_w = if cw.abs() > 1e-7 { 1.0 / cw } else { 1e7 };
        let nx = cx * inv_w;
        let ny = cy * inv_w;
        let nz = cz * inv_w;

        let sx = (nx + 1.0) * 0.5 * (w_f - 1.0);
        let sy = (1.0 - ny) * 0.5 * (h_f - 1.0);

        screen_pts.push([sx, sy]);
        ndc_zs.push(nz);
    }

    let mut fb = vec![options.bg_color[0]; total_pixels * 3];
    for i in 0..total_pixels {
        fb[i * 3] = options.bg_color[0];
        fb[i * 3 + 1] = options.bg_color[1];
        fb[i * 3 + 2] = options.bg_color[2];
    }
    let mut z_buffer = vec![f32::INFINITY; total_pixels];

    // 三角形ラスタライズ
    for tri in faces {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        if i0 >= screen_pts.len() || i1 >= screen_pts.len() || i2 >= screen_pts.len() {
            continue;
        }

        let p0 = screen_pts[i0];
        let p1 = screen_pts[i1];
        let p2 = screen_pts[i2];

        // 符号付きスクリーン面積（カリング・表裏判定）
        let cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
        let is_front = cross < 0.0;
        let color = if is_front {
            options.front_color
        } else {
            options.back_color
        };

        // バウンディングボックス
        let min_x = (p0[0].min(p1[0]).min(p2[0]).floor() as i32).max(0).min((width - 1) as i32) as u32;
        let max_x = (p0[0].max(p1[0]).max(p2[0]).ceil() as i32).max(0).min((width - 1) as i32) as u32;
        let min_y = (p0[1].min(p1[1]).min(p2[1]).floor() as i32).max(0).min((height - 1) as i32) as u32;
        let max_y = (p0[1].max(p1[1]).max(p2[1]).ceil() as i32).max(0).min((height - 1) as i32) as u32;

        if min_x >= max_x || min_y >= max_y {
            continue;
        }

        let denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1]);
        if denom.abs() < 1e-7 {
            continue;
        }
        let inv_denom = 1.0 / denom;

        let z0 = ndc_zs[i0];
        let z1 = ndc_zs[i1];
        let z2 = ndc_zs[i2];

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
                        fb[rgb_idx] = color[0];
                        fb[rgb_idx + 1] = color[1];
                        fb[rgb_idx + 2] = color[2];
                    }
                }
            }
        }
    }

    // 赤色ピクセル数（裏面露出面）をカウント
    let mut red_pixels = 0;
    for i in 0..total_pixels {
        let r = fb[i * 3];
        let g = fb[i * 3 + 1];
        let b = fb[i * 3 + 2];
        if r >= 200 && g <= 50 && b <= 50 {
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

/// メッシュを描画し、PNGファイルとして保存する
pub fn render_mesh_to_png_file(
    filepath: &str,
    positions: &[[f32; 3]],
    faces: &[[u32; 3]],
    options: &RenderOptions,
) -> Result<RenderResult, String> {
    let result = render_mesh_to_buffer(positions, faces, options);

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
        let res_back = render_mesh_to_buffer(&pos_back, &faces, &options);
        assert!(res_back.red_pixels > 50, "裏面描画で赤色ピクセルが検出されませんでした");
    }
}
