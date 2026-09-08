//! taremin_cloth ボーンSDF高速GPUベーカー (sdf_baker.rs)
//!
//! 素体メッシュとアーマチュア情報から、wgpu コンピュートシェーダーを用いて
//! 各ボーンのローカルSDF (Rg16Float) を3Dテクスチャアトラスとして並列ベイクする。

use bytemuck::{Pod, Zeroable};
use crate::context::GpuContext;

/// シェーダー転送用ボーンベイクパラメータ（アライメント: 16バイト境界、サイズ: 64バイト）
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuBakeParams {
    pub local_min: [f32; 3],
    pub tri_start: u32,
    pub local_max: [f32; 3],
    pub tri_count: u32,
    pub tile_col: u32,
    pub tile_row: u32,
    pub tile_layer: u32,
    pub res: u32,
    pub total_width: u32,
    pub total_height: u32,
    pub total_depth: u32,
    pub row_pitch: u32,
}

/// シェーダー転送用三角形データ（アライメント: 16バイト境界、サイズ: 64バイト）
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuBakeTriangle {
    pub p0: [f32; 3],
    pub w0: f32,
    pub p1: [f32; 3],
    pub w1: f32,
    pub p2: [f32; 3],
    pub w2: f32,
    pub normal: [f32; 3],
    pub _pad: f32,
}

/// 単一ボーンの入力データ
pub struct BoneInput {
    pub name: String,
    pub weights: Vec<f32>,
    /// メッシュ空間 -> ボーンローカル空間への変換行列 (4x4, row-major)
    pub bind_matrix: [f32; 16],
}

/// ベイク結果
pub struct GpuBoneSdfBakeResult {
    /// Rg16Float (Z -> Y -> X -> [dist, alpha]) の生バイト列
    pub texture_bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// 各ボーンの20要素パラメータ [local_min(3), b_idx(1), local_max(3), res(1), uvw_scale(4), uvw_offset(4), friction, thickness, restitution, weight_th]
    pub bone_infos: Vec<[f32; 20]>,
    pub active_bones: Vec<String>,
    pub bind_matrices: Vec<[f32; 16]>,
}

/// 3Dアトラスタイルのレイアウト (cols, rows, layers) を計算する
pub fn compute_3d_atlas_layout(
    n_bones: usize,
    resolution: usize,
    max_dim: usize,
) -> (usize, usize, usize, usize, usize, usize, usize) {
    if n_bones == 0 {
        return (0, 0, 0, resolution, 0, 0, 0);
    }
    let n_cols = (n_bones as f64).powf(1.0 / 3.0).ceil() as usize;
    let n_cols = n_cols.max(1);
    let n_rows = (((n_bones + n_cols - 1) / n_cols) as f64).sqrt().ceil() as usize;
    let n_rows = n_rows.max(1);
    let n_layers = ((n_bones + n_cols * n_rows - 1) / (n_cols * n_rows)).max(1);

    let max_tile = n_cols.max(n_rows).max(n_layers);
    let safe_res = if max_tile * resolution > max_dim {
        (max_dim / max_tile).max(16)
    } else {
        resolution
    };

    let total_width = n_cols * safe_res;
    let total_height = n_rows * safe_res;
    let total_depth = n_layers * safe_res;

    (n_cols, n_rows, n_layers, safe_res, total_width, total_height, total_depth)
}

/// 4x4 行列 (row-major) による 3D 頂点の変換
#[inline]
fn transform_point(mat: &[f32; 16], p: &[f32; 3]) -> [f32; 3] {
    let x = mat[0] * p[0] + mat[1] * p[1] + mat[2] * p[2] + mat[3];
    let y = mat[4] * p[0] + mat[5] * p[1] + mat[6] * p[2] + mat[7];
    let z = mat[8] * p[0] + mat[9] * p[1] + mat[10] * p[2] + mat[11];
    let w = mat[12] * p[0] + mat[13] * p[1] + mat[14] * p[2] + mat[15];
    if w.abs() > 1e-7 && (w - 1.0).abs() > 1e-5 {
        let inv_w = 1.0 / w;
        [x * inv_w, y * inv_w, z * inv_w]
    } else {
        [x, y, z]
    }
}

/// 3D ベクトルの外積
#[inline]
fn cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// 3D ベクトルの正規化
#[inline]
fn normalize(v: &[f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq > 1e-12 {
        let inv_len = 1.0 / len_sq.sqrt();
        [v[0] * inv_len, v[1] * inv_len, v[2] * inv_len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// GPUコンピュートシェーダーを用いたボーンSDFベイクの実行
pub fn bake_bone_sdf_gpu(
    mesh_verts: &[[f32; 3]],
    mesh_tris: &[[i32; 3]],
    bones: &[BoneInput],
    resolution: usize,
    margin: f32,
    weight_threshold: f32,
    blend_k: f32,
    friction: f32,
    thickness: f32,
    restitution: f32,
) -> Result<GpuBoneSdfBakeResult, String> {
    // 1. 有効なボーンの選定
    let mut active_indices = Vec::new();
    for (i, b) in bones.iter().enumerate() {
        let max_w = b.weights.iter().cloned().fold(0.0f32, f32::max);
        if max_w > weight_threshold {
            active_indices.push(i);
        }
    }

    if active_indices.is_empty() {
        return Err("有効なスキンウェイトを持つボーンが存在しません".to_string());
    }

    let n_active = active_indices.len();
    let max_texture_dim = 2048;
    let (n_cols, n_rows, n_layers, safe_res, total_width, total_height, total_depth) =
        compute_3d_atlas_layout(n_active, resolution, max_texture_dim);

    let mut bake_params_list = Vec::with_capacity(n_active);
    let mut all_bake_triangles = Vec::new();
    let mut bone_infos = Vec::with_capacity(n_active);
    let mut active_bone_names = Vec::with_capacity(n_active);
    let mut bind_matrices = Vec::with_capacity(n_active);

    // 2. 各ボーンのローカルAABB算出およびボーンローカル三角形の収集
    for (out_b_idx, &src_b_idx) in active_indices.iter().enumerate() {
        let b = &bones[src_b_idx];
        active_bone_names.push(b.name.clone());
        bind_matrices.push(b.bind_matrix);

        let mat_inv = &b.bind_matrix;

        // ウェイトが閾値を超える頂点を抽出してボーンローカル座標系へ
        let mut min_pt = [f32::INFINITY; 3];
        let mut max_pt = [f32::NEG_INFINITY; 3];
        let mut valid_count = 0;

        for (v_idx, &w) in b.weights.iter().enumerate() {
            if w > weight_threshold && v_idx < mesh_verts.len() {
                let p_local = transform_point(mat_inv, &mesh_verts[v_idx]);
                for k in 0..3 {
                    min_pt[k] = min_pt[k].min(p_local[k]);
                    max_pt[k] = max_pt[k].max(p_local[k]);
                }
                valid_count += 1;
            }
        }

        if valid_count == 0 {
            min_pt = [-0.1, -0.1, -0.1];
            max_pt = [0.1, 0.1, 0.1];
        }

        // マージン付与と最小サイズ (5cm) 保証 (thickness を加算して大厚み時の境界外カリングを防止)
        let mut size = [
            (max_pt[0] - min_pt[0]).max(0.05),
            (max_pt[1] - min_pt[1]).max(0.05),
            (max_pt[2] - min_pt[2]).max(0.05),
        ];
        for k in 0..3 {
            let pad = size[k] * margin + thickness.max(0.0);
            min_pt[k] -= pad;
            max_pt[k] += pad;
            size[k] = max_pt[k] - min_pt[k];
        }

        // タイル配置
        let b_per_layer = n_cols * n_rows;
        let l = (out_b_idx / b_per_layer) as u32;
        let rem = out_b_idx % b_per_layer;
        let r = (rem / n_cols) as u32;
        let c = (rem % n_cols) as u32;

        // UVW スケール & オフセット
        let uvw_scale_local = [1.0 / size[0], 1.0 / size[1], 1.0 / size[2]];
        let uvw_offset_local = [
            -min_pt[0] / size[0],
            -min_pt[1] / size[1],
            -min_pt[2] / size[2],
        ];

        let atlas_uvw_scale = [
            uvw_scale_local[0] / n_cols as f32,
            uvw_scale_local[1] / n_rows as f32,
            uvw_scale_local[2] / n_layers as f32,
            blend_k,
        ];
        let atlas_uvw_offset = [
            (uvw_offset_local[0] + c as f32) / n_cols as f32,
            (uvw_offset_local[1] + r as f32) / n_rows as f32,
            (uvw_offset_local[2] + l as f32) / n_layers as f32,
            0.0,
        ];

        // 20要素の bone_info 行
        let mut info_row = [0.0f32; 20];
        info_row[0..3].copy_from_slice(&min_pt);
        info_row[3] = out_b_idx as f32;
        info_row[4..7].copy_from_slice(&max_pt);
        info_row[7] = safe_res as f32;
        info_row[8..12].copy_from_slice(&atlas_uvw_scale);
        info_row[12..16].copy_from_slice(&atlas_uvw_offset);
        info_row[16] = friction;
        info_row[17] = thickness;
        info_row[18] = restitution;
        info_row[19] = weight_threshold;
        bone_infos.push(info_row);

        // ボーン関係三角形の抽出
        let tri_start = all_bake_triangles.len() as u32;
        let aabb_min = min_pt;
        let aabb_max = max_pt;

        for tri in mesh_tris {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;
            if i0 >= mesh_verts.len() || i1 >= mesh_verts.len() || i2 >= mesh_verts.len() {
                continue;
            }

            let w0 = if i0 < b.weights.len() { b.weights[i0] } else { 0.0 };
            let w1 = if i1 < b.weights.len() { b.weights[i1] } else { 0.0 };
            let w2 = if i2 < b.weights.len() { b.weights[i2] } else { 0.0 };

            // 少なくとも1頂点がウェイトを持つか
            let has_weight = w0 > 0.0 || w1 > 0.0 || w2 > 0.0;
            if !has_weight {
                continue;
            }

            let p0_local = transform_point(mat_inv, &mesh_verts[i0]);
            let p1_local = transform_point(mat_inv, &mesh_verts[i1]);
            let p2_local = transform_point(mat_inv, &mesh_verts[i2]);

            // 三角形のAABBとボーンAABBの交差判定
            let t_min = [
                p0_local[0].min(p1_local[0]).min(p2_local[0]),
                p0_local[1].min(p1_local[1]).min(p2_local[1]),
                p0_local[2].min(p1_local[2]).min(p2_local[2]),
            ];
            let t_max = [
                p0_local[0].max(p1_local[0]).max(p2_local[0]),
                p0_local[1].max(p1_local[1]).max(p2_local[1]),
                p0_local[2].max(p1_local[2]).max(p2_local[2]),
            ];

            let overlap = t_min[0] <= aabb_max[0] && t_max[0] >= aabb_min[0]
                && t_min[1] <= aabb_max[1] && t_max[1] >= aabb_min[1]
                && t_min[2] <= aabb_max[2] && t_max[2] >= aabb_min[2];

            if overlap {
                let e01 = [p1_local[0] - p0_local[0], p1_local[1] - p0_local[1], p1_local[2] - p0_local[2]];
                let e02 = [p2_local[0] - p0_local[0], p2_local[1] - p0_local[1], p2_local[2] - p0_local[2]];
                let normal = normalize(&cross(&e01, &e02));

                all_bake_triangles.push(GpuBakeTriangle {
                    p0: p0_local,
                    w0,
                    p1: p1_local,
                    w1,
                    p2: p2_local,
                    w2,
                    normal,
                    _pad: 0.0,
                });
            }
        }

        let tri_count = all_bake_triangles.len() as u32 - tri_start;

        bake_params_list.push(GpuBakeParams {
            local_min: min_pt,
            tri_start,
            local_max: max_pt,
            tri_count,
            tile_col: c,
            tile_row: r,
            tile_layer: l,
            res: safe_res as u32,
            total_width: total_width as u32,
            total_height: total_height as u32,
            total_depth: total_depth as u32,
            row_pitch: 0,
        });
    }

    // 三角形が1つもない場合のダミー
    if all_bake_triangles.is_empty() {
        all_bake_triangles.push(GpuBakeTriangle {
            p0: [0.0; 3],
            w0: 0.0,
            p1: [0.0; 3],
            w1: 0.0,
            p2: [0.0; 3],
            w2: 0.0,
            normal: [0.0, 1.0, 0.0],
            _pad: 0.0,
        });
    }

    // 3. GPUパイプラインの構築と実行
    let ctx = GpuContext::get_or_init().map_err(|e| format!("GPU初期化失敗: {e}"))?;
    let device = &ctx.device;
    let queue = &ctx.queue;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("TareminCloth SDF Baker Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bake_sdf.wgsl").into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("SDF Baker BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("SDF Baker Pipeline Layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("SDF Baker Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // バッファ作成
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SDF Baker Params Buffer"),
        size: (std::mem::size_of::<GpuBakeParams>() * bake_params_list.len()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&bake_params_list));

    let triangles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SDF Baker Triangles Buffer"),
        size: (std::mem::size_of::<GpuBakeTriangle>() * all_bake_triangles.len()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&triangles_buffer, 0, bytemuck::cast_slice(&all_bake_triangles));

    let total_voxels = total_width * total_height * total_depth;
    let output_bytes = (total_voxels * 4) as u64; // 1 voxel = 4 bytes (Rg16Float = u32)

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SDF Baker Output Buffer"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SDF Baker Staging Buffer"),
        size: output_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SDF Baker BindGroup"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: triangles_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    // ディスパッチ記録
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("SDF Baker Encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SDF Baker Compute Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&compute_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let wg_x = ((safe_res as u32) + 3) / 4;
        let wg_y = ((safe_res as u32) + 3) / 4;
        let wg_z = ((safe_res as u32 * n_active as u32) + 3) / 4;
        pass.dispatch_workgroups(wg_x, wg_y, wg_z);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    // 4. ステージングバッファからCPUへ読み戻し
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    device.poll(wgpu::Maintain::Wait);

    receiver
        .recv()
        .map_err(|e| format!("マップ通知受信エラー: {e}"))?
        .map_err(|e| format!("ステージングバッファのマップ失敗: {e}"))?;

    let texture_bytes = {
        let view = buffer_slice.get_mapped_range();
        view.to_vec()
    };
    staging_buffer.unmap();

    Ok(GpuBoneSdfBakeResult {
        texture_bytes,
        width: total_width as u32,
        height: total_height as u32,
        depth: total_depth as u32,
        bone_infos,
        active_bones: active_bone_names,
        bind_matrices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_layout() {
        let (cols, rows, layers, res, w, h, d) = compute_3d_atlas_layout(8, 32, 2048);
        assert_eq!(res, 32);
        assert!(cols * rows * layers >= 8);
        assert_eq!(w, cols * 32);
        assert_eq!(h, rows * 32);
        assert_eq!(d, layers * 32);
    }

    #[test]
    fn test_gpu_sdf_bake_single_triangle() {
        // GPUが利用可能か判定
        if GpuContext::get_or_init().is_err() {
            println!("GPUが利用できない環境のためスキップします");
            return;
        }

        // XY平面上の三角形 (Z=0)
        let verts = vec![
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let tris = vec![[0, 1, 2]];
        let bone = BoneInput {
            name: "TestBone".to_string(),
            weights: vec![1.0, 1.0, 1.0],
            bind_matrix: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        };

        let res = bake_bone_sdf_gpu(
            &verts,
            &tris,
            &[bone],
            16,   // 解像度 16^3
            0.1,  // マージン
            0.02, // weight_th
            0.05,
            0.5,
            0.01,
            0.0,
        );

        assert!(res.is_ok(), "ベイクが成功すること: {:?}", res.err());
        let result = res.unwrap();
        assert_eq!(result.width, 16);
        assert_eq!(result.height, 16);
        assert_eq!(result.depth, 16);
        assert_eq!(result.texture_bytes.len(), 16 * 16 * 16 * 4);
    }
}
