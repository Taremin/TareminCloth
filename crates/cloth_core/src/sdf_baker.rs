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

/// シェーダー転送用単一メッシュベイクパラメータ（アライメント: 16バイト境界、サイズ: 48バイト）
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuBakeMeshParams {
    pub local_min: [f32; 3],
    pub tri_count: u32,
    pub local_max: [f32; 3],
    pub row_pitch: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub _pad0: u32,
}

/// シェーダー転送用メッシュ三角形データ（アライメント: 16バイト境界、サイズ: 64バイト）
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuBakeMeshTriangle {
    pub p0: [f32; 3],
    pub _pad0: f32,
    pub p1: [f32; 3],
    pub _pad1: f32,
    pub p2: [f32; 3],
    pub _pad2: f32,
    pub normal: [f32; 3],
    pub _pad3: f32,
}

/// メッシュSDFベイク結果
pub struct GpuMeshSdfBakeResult {
    /// Rg16Float (Z -> Y -> X -> [dist, 1.0]) の生バイト列
    pub texture_bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// 20要素パラメータ [local_min(3), 0.0, local_max(3), res(1), uvw_scale(4), uvw_offset(4), friction, thickness, restitution, 0.0]
    pub bone_info: [f32; 20],
}

/// 階層ベイクの粗密比 (H = RATIO * h)。整数比固定で親対応を一意にする。
/// 実メッシュ検証で粗セルより小さい突起（指先等）がtop-Kから脱落したため 8 -> 4。
pub const MESH_SDF_HIER_RATIO: u32 = 4;
/// 階層ベイク Pass2 の1回あたり最大レイヤー数 (Windows TDR回避用にsubmitを分割)
pub const MESH_SDF_HIER_SLAB_LAYERS: u32 = 64;

/// 階層ベイク Pass2 用パラメータ（アライメント: 16バイト境界、サイズ: 32バイト）
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuHierFineParams {
    pub coarse_w: u32,
    pub coarse_h: u32,
    pub coarse_d: u32,
    pub band: f32,
    pub z_offset: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// 階層ベイク Pass1 (粗グリッド) の結果
pub struct GpuMeshSdfCoarseResult {
    /// 粗セル毎の符号付き距離 (Rg16Floatパック u32、Z -> Y -> X)
    pub dist_packed: Vec<u32>,
    /// 粗セル毎の上位8近傍三角形インデックス (ストライド配置 [flat * 8 + k])
    pub nearest_tri: Vec<u32>,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// 粗セル幅 (= 要求ボクセル幅 x MESH_SDF_HIER_RATIO)
    pub coarse_voxel: f32,
    /// 20要素パラメータ (Pass2のドメイン共有用、legacyと同一式)
    pub bone_info: [f32; 20],
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

/// GPUコンピュートシェーダーを用いた単一メッシュ直方体SDFベイクの実行
/// メッシュSDF用ドメイン（パディング付きAABBと直方体解像度）の算出。
/// legacy単一passと階層passで共有する。戻り値: (min_pt, max_pt, size, w, h, d)
fn mesh_sdf_domain(
    mesh_verts: &[[f32; 3]],
    voxel_size: f32,
    margin: f32,
    thickness: f32,
) -> ([f32; 3], [f32; 3], [f32; 3], u32, u32, u32) {
    let mut min_pt = [f32::INFINITY; 3];
    let mut max_pt = [f32::NEG_INFINITY; 3];

    for v in mesh_verts {
        for k in 0..3 {
            min_pt[k] = min_pt[k].min(v[k]);
            max_pt[k] = max_pt[k].max(v[k]);
        }
    }

    // マージン付与と最小サイズ (5cm) 保証
    let mut size = [
        (max_pt[0] - min_pt[0]).max(0.05),
        (max_pt[1] - min_pt[1]).max(0.05),
        (max_pt[2] - min_pt[2]).max(0.05),
    ];
    for k in 0..3 {
        // 法線差分計算(5mm)や接触厚み判定がテクスチャ端で消失しないよう、十分なマージン(最低4cm + thickness + 2cm)を確保
        let pad = (size[k] * margin).max(0.04) + thickness.max(0.0) + 0.02;
        min_pt[k] -= pad;
        max_pt[k] += pad;
        size[k] = max_pt[k] - min_pt[k];
    }

    // 直方体解像度 (width, height, depth) の算出 (8〜2048クランプ)
    let v_size = voxel_size.max(0.001);
    let raw_w = (size[0] / v_size).ceil() as usize;
    let raw_h = (size[1] / v_size).ceil() as usize;
    let raw_d = (size[2] / v_size).ceil() as usize;

    let width = raw_w.clamp(8, 2048) as u32;
    let height = raw_h.clamp(8, 2048) as u32;
    let depth = raw_d.clamp(8, 2048) as u32;

    (min_pt, max_pt, size, width, height, depth)
}

/// メッシュSDF用20要素ボーンパラメータの構築 (legacyと同一式)
#[allow(clippy::too_many_arguments)]
fn mesh_sdf_bone_info(
    min_pt: [f32; 3],
    max_pt: [f32; 3],
    size: [f32; 3],
    width: u32,
    friction: f32,
    thickness: f32,
    restitution: f32,
) -> [f32; 20] {
    // UVW スケール & オフセット
    let uvw_scale = [1.0 / size[0], 1.0 / size[1], 1.0 / size[2], 0.0];
    let uvw_offset = [
        -min_pt[0] / size[0],
        -min_pt[1] / size[1],
        -min_pt[2] / size[2],
        0.0,
    ];

    // 20要素パラメータ [local_min(3), 0.0, local_max(3), width, uvw_scale(4), uvw_offset(4), friction, thickness, restitution, 0.0]
    let mut bone_info = [0.0f32; 20];
    bone_info[0..3].copy_from_slice(&min_pt);
    bone_info[3] = 0.0;
    bone_info[4..7].copy_from_slice(&max_pt);
    bone_info[7] = width as f32;
    bone_info[8..12].copy_from_slice(&uvw_scale);
    bone_info[12..16].copy_from_slice(&uvw_offset);
    bone_info[16] = friction;
    bone_info[17] = thickness;
    bone_info[18] = restitution;
    bone_info[19] = 0.0;
    bone_info
}

/// メッシュSDF用ベイク三角形列の収集 (空時はダミー1件。legacyと同一)
fn collect_mesh_bake_triangles(
    mesh_verts: &[[f32; 3]],
    mesh_tris: &[[i32; 3]],
) -> Vec<GpuBakeMeshTriangle> {
    let mut bake_triangles = Vec::with_capacity(mesh_tris.len());
    for tri in mesh_tris {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        if i0 >= mesh_verts.len() || i1 >= mesh_verts.len() || i2 >= mesh_verts.len() {
            continue;
        }

        let p0 = mesh_verts[i0];
        let p1 = mesh_verts[i1];
        let p2 = mesh_verts[i2];

        let e01 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e02 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let normal = normalize(&cross(&e01, &e02));

        bake_triangles.push(GpuBakeMeshTriangle {
            p0,
            _pad0: 0.0,
            p1,
            _pad1: 0.0,
            p2,
            _pad2: 0.0,
            normal,
            _pad3: 0.0,
        });
    }

    if bake_triangles.is_empty() {
        bake_triangles.push(GpuBakeMeshTriangle {
            p0: [0.0; 3],
            _pad0: 0.0,
            p1: [0.0; 3],
            _pad1: 0.0,
            p2: [0.0; 3],
            _pad2: 0.0,
            normal: [0.0, 1.0, 0.0],
            _pad3: 0.0,
        });
    }
    bake_triangles
}

/// ステージングバッファ (MAP_READ) の全内容をホストへ読み出す
fn read_staging_to_vec(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
) -> Result<Vec<u8>, String> {
    let buffer_slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    device.poll(wgpu::Maintain::Wait);

    receiver
        .recv()
        .map_err(|e| format!("マップ通知受信エラー: {e}"))?
        .map_err(|e| format!("ステージングバッファのマップ失敗: {e}"))?;

    let out = {
        let view = buffer_slice.get_mapped_range();
        view.to_vec()
    };
    staging.unmap();
    Ok(out)
}

pub fn bake_mesh_sdf_gpu(
    mesh_verts: &[[f32; 3]],
    mesh_tris: &[[i32; 3]],
    voxel_size: f32,
    margin: f32,
    friction: f32,
    thickness: f32,
    restitution: f32,
) -> Result<GpuMeshSdfBakeResult, String> {
    if mesh_verts.is_empty() {
        return Err("メッシュ頂点が空です".to_string());
    }

    // 1-2. ドメイン算出と20要素パラメータ構築 (共有ヘルパー。式は従来通り)
    let (min_pt, max_pt, size, width, height, depth) =
        mesh_sdf_domain(mesh_verts, voxel_size, margin, thickness);
    let bone_info =
        mesh_sdf_bone_info(min_pt, max_pt, size, width, friction, thickness, restitution);

    // 3. 三角形データの収集 (共有ヘルパー。式は従来通り)
    let bake_triangles = collect_mesh_bake_triangles(mesh_verts, mesh_tris);

    let bake_params = GpuBakeMeshParams {
        local_min: min_pt,
        tri_count: bake_triangles.len() as u32,
        local_max: max_pt,
        row_pitch: 0,
        width,
        height,
        depth,
        _pad0: 0,
    };

    // 4. GPUパイプラインの構築と実行
    let ctx = GpuContext::get_or_init().map_err(|e| format!("GPU初期化失敗: {e}"))?;
    let device = &ctx.device;
    let queue = &ctx.queue;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("TareminCloth Mesh SDF Baker Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bake_mesh_sdf.wgsl").into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Mesh SDF Baker BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
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
        label: Some("Mesh SDF Baker Pipeline Layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Mesh SDF Baker Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh SDF Baker Params Buffer"),
        size: std::mem::size_of::<GpuBakeMeshParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&bake_params));

    let triangles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh SDF Baker Triangles Buffer"),
        size: (std::mem::size_of::<GpuBakeMeshTriangle>() * bake_triangles.len()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&triangles_buffer, 0, bytemuck::cast_slice(&bake_triangles));

    let total_voxels = width * height * depth;
    let output_bytes = (total_voxels * 4) as u64;

    // GPU制限チェック: デバイスの最大バッファサイズおよびストレージバッファ制限を超えていないか事前検証（Panic防止）
    let max_buf_size = device.limits().max_buffer_size;
    let max_storage_size = device.limits().max_storage_buffer_binding_size as u64;
    if output_bytes > max_buf_size || output_bytes > max_storage_size {
        let req_mb = output_bytes / (1024 * 1024);
        let max_mb = max_buf_size.min(max_storage_size) / (1024 * 1024);
        return Err(format!(
            "Mesh SDF解像度 ({}x{}x{}, 必要容量: 約{}MB) がGPUのバッファ上限 ({}MB) を超えています。ボクセルサイズを大きくしてください。",
            width, height, depth, req_mb, max_mb
        ));
    }

    let max_3d_dim = device.limits().max_texture_dimension_3d;
    if width > max_3d_dim || height > max_3d_dim || depth > max_3d_dim {
        return Err(format!(
            "Mesh SDF寸法 ({}x{}x{}) がGPUの3Dテクスチャ最大解像度 ({}) を超えています。ボクセルサイズを大きくしてください。",
            width, height, depth, max_3d_dim
        ));
    }

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh SDF Baker Output Buffer"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh SDF Baker Staging Buffer"),
        size: output_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Mesh SDF Baker BindGroup"),
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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Mesh SDF Baker Encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Mesh SDF Baker Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&compute_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let dispatch_x = (width + 3) / 4;
        let dispatch_y = (height + 3) / 4;
        let dispatch_z = (depth + 3) / 4;
        pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_bytes);
    queue.submit(Some(encoder.finish()));

    let texture_bytes = read_staging_to_vec(device, &staging_buffer)?;

    Ok(GpuMeshSdfBakeResult {
        texture_bytes,
        width,
        height,
        depth,
        bone_info,
    })
}

/// 階層ベイク Pass1: 粗グリッドベイク。
/// 粗セル幅 H = 要求ボクセル幅 x MESH_SDF_HIER_RATIO（整数比固定）。
/// 各粗セルに符号付き距離と最近傍三角形インデックスを格納する。
pub fn bake_mesh_sdf_coarse(
    mesh_verts: &[[f32; 3]],
    mesh_tris: &[[i32; 3]],
    voxel_size: f32,
    margin: f32,
    friction: f32,
    thickness: f32,
    restitution: f32,
) -> Result<GpuMeshSdfCoarseResult, String> {
    if mesh_verts.is_empty() {
        return Err("メッシュ頂点が空です".to_string());
    }

    let coarse_voxel = (voxel_size * MESH_SDF_HIER_RATIO as f32).max(0.001);
    let (min_pt, max_pt, size, width, height, depth) =
        mesh_sdf_domain(mesh_verts, coarse_voxel, margin, thickness);
    let bone_info =
        mesh_sdf_bone_info(min_pt, max_pt, size, width, friction, thickness, restitution);
    let bake_triangles = collect_mesh_bake_triangles(mesh_verts, mesh_tris);

    let bake_params = GpuBakeMeshParams {
        local_min: min_pt,
        tri_count: bake_triangles.len() as u32,
        local_max: max_pt,
        row_pitch: 0,
        width,
        height,
        depth,
        _pad0: 0,
    };

    // GPUパイプラインの構築と実行
    let ctx = GpuContext::get_or_init().map_err(|e| format!("GPU初期化失敗: {e}"))?;
    let device = &ctx.device;
    let queue = &ctx.queue;

    let source = format!(
        "{}\n{}",
        include_str!("shaders/bake_mesh_sdf_hier_common.wgsl"),
        include_str!("shaders/bake_mesh_sdf_hier_coarse_main.wgsl"),
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("TareminCloth Mesh SDF Hierarchical Coarse Shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let storage_rw = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Mesh SDF Coarse BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
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
                ty: storage_rw,
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: storage_rw,
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Mesh SDF Coarse Pipeline Layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Mesh SDF Coarse Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh SDF Coarse Params Buffer"),
        size: std::mem::size_of::<GpuBakeMeshParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&bake_params));

    let triangles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh SDF Coarse Triangles Buffer"),
        size: (std::mem::size_of::<GpuBakeMeshTriangle>() * bake_triangles.len()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&triangles_buffer, 0, bytemuck::cast_slice(&bake_triangles));

    let total_voxels = width * height * depth;
    let output_bytes = (total_voxels * 4) as u64;

    // GPU制限チェック (legacyと同一式。tri出力は8倍)
    let max_buf_size = device.limits().max_buffer_size;
    let max_storage_size = device.limits().max_storage_buffer_binding_size as u64;
    if output_bytes * 8 > max_buf_size || output_bytes * 8 > max_storage_size {
        let req_mb = (output_bytes * 8) / (1024 * 1024);
        let max_mb = max_buf_size.min(max_storage_size) / (1024 * 1024);
        return Err(format!(
            "Mesh SDF粗グリッド ({}x{}x{}, 必要容量: 約{}MB) がGPUのバッファ上限 ({}MB) を超えています。ボクセルサイズを大きくしてください。",
            width, height, depth, req_mb, max_mb
        ));
    }

    let max_3d_dim = device.limits().max_texture_dimension_3d;
    if width > max_3d_dim || height > max_3d_dim || depth > max_3d_dim {
        return Err(format!(
            "Mesh SDF粗グリッド寸法 ({}x{}x{}) がGPUの3Dテクスチャ最大解像度 ({}) を超えています。ボクセルサイズを大きくしてください。",
            width, height, depth, max_3d_dim
        ));
    }

    let dist_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh SDF Coarse Dist Buffer"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let tri_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh SDF Coarse TriIdx Buffer"),
        size: output_bytes * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let dist_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh SDF Coarse Dist Staging"),
        size: output_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let tri_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh SDF Coarse TriIdx Staging"),
        size: output_bytes * 8,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Mesh SDF Coarse BindGroup"),
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
                resource: dist_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: tri_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Mesh SDF Coarse Encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Mesh SDF Coarse Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&compute_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((width + 3) / 4, (height + 3) / 4, (depth + 3) / 4);
    }

    encoder.copy_buffer_to_buffer(&dist_buffer, 0, &dist_staging, 0, output_bytes);
    encoder.copy_buffer_to_buffer(&tri_buffer, 0, &tri_staging, 0, output_bytes * 8);
    queue.submit(Some(encoder.finish()));

    let dist_bytes = read_staging_to_vec(device, &dist_staging)?;
    let tri_bytes = read_staging_to_vec(device, &tri_staging)?;

    let dist_packed: Vec<u32> = dist_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let nearest_tri: Vec<u32> = tri_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Ok(GpuMeshSdfCoarseResult {
        dist_packed,
        nearest_tri,
        width,
        height,
        depth,
        coarse_voxel,
        bone_info,
    })
}

/// 階層ベイク Pass1+Pass2 一括実行。戻り値形式はlegacyと同一のため差し替え可能。
/// - Pass1: 粗グリッド総当たり（距離＋最近傍三角形index）
/// - Pass2: 遠方セルは粗値転写、近傍セルのみ27→125候補exact計算＋AABB早期棄却
/// - Pass2のdispatchはZスラブ毎にsubmit＋pollし、Windows TDRを回避する
pub fn bake_mesh_sdf_hierarchical(
    mesh_verts: &[[f32; 3]],
    mesh_tris: &[[i32; 3]],
    voxel_size: f32,
    margin: f32,
    friction: f32,
    thickness: f32,
    restitution: f32,
) -> Result<GpuMeshSdfBakeResult, String> {
    if mesh_verts.is_empty() {
        return Err("メッシュ頂点が空です".to_string());
    }

    // 密ドメイン (legacyと同一式 → 同一dims・同一ドメイン)
    let (min_pt, max_pt, size, width, height, depth) =
        mesh_sdf_domain(mesh_verts, voxel_size, margin, thickness);
    let bone_info =
        mesh_sdf_bone_info(min_pt, max_pt, size, width, friction, thickness, restitution);
    let bake_triangles = collect_mesh_bake_triangles(mesh_verts, mesh_tris);

    // Pass1: 粗グリッド
    let coarse =
        bake_mesh_sdf_coarse(mesh_verts, mesh_tris, voxel_size, margin, friction, thickness, restitution)?;

    // 近傍band = 4 x 密ボクセル幅（最大軸）。ステンシル半径r=1ではband<=H_effが条件だが、
    // Pass2はr=2（125セル）のため余裕を持つ。粗実効幅の下限検証のみ行う。
    let h_fine = (size[0] / width as f32)
        .max(size[1] / height as f32)
        .max(size[2] / depth as f32);
    let band = 4.0 * h_fine;
    let h_coarse_min = (size[0] / coarse.width as f32)
        .min(size[1] / coarse.height as f32)
        .min(size[2] / coarse.depth as f32);
    if band > 2.0 * h_coarse_min {
        return Err(format!(
            "階層ベイクのband幅 ({:.4}m) が粗セル幅 ({:.4}m) に対して大きすぎます",
            band, h_coarse_min
        ));
    }

    let ctx = GpuContext::get_or_init().map_err(|e| format!("GPU初期化失敗: {e}"))?;
    let device = &ctx.device;
    let queue = &ctx.queue;

    let source = format!(
        "{}\n{}",
        include_str!("shaders/bake_mesh_sdf_hier_common.wgsl"),
        include_str!("shaders/bake_mesh_sdf_hier_fine_main.wgsl"),
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("TareminCloth Mesh SDF Hierarchical Fine Shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let uniform_ty = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let ro_ty = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let rw_ty = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Mesh SDF HierFine BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: uniform_ty, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: ro_ty, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: ro_ty, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: ro_ty, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: uniform_ty, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: rw_ty, count: None },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Mesh SDF HierFine Pipeline Layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Mesh SDF HierFine Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bake_params = GpuBakeMeshParams {
        local_min: min_pt,
        tri_count: bake_triangles.len() as u32,
        local_max: max_pt,
        row_pitch: 0,
        width,
        height,
        depth,
        _pad0: 0,
    };
    let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh SDF HierFine Params Buffer"),
        size: std::mem::size_of::<GpuBakeMeshParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&bake_params));

    let triangles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh SDF HierFine Triangles Buffer"),
        size: (std::mem::size_of::<GpuBakeMeshTriangle>() * bake_triangles.len()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&triangles_buffer, 0, bytemuck::cast_slice(&bake_triangles));

    let coarse_dist_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh SDF HierFine CoarseDist Buffer"),
        size: (coarse.dist_packed.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&coarse_dist_buffer, 0, bytemuck::cast_slice(&coarse.dist_packed));
    let coarse_tri_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh SDF HierFine CoarseTri Buffer"),
        size: (coarse.nearest_tri.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&coarse_tri_buffer, 0, bytemuck::cast_slice(&coarse.nearest_tri));

    let hier_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh SDF HierFine HierParams Buffer"),
        size: std::mem::size_of::<GpuHierFineParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let total_voxels = width * height * depth;
    let output_bytes = (total_voxels * 4) as u64;

    // GPU制限チェック (legacyと同一式)
    let max_buf_size = device.limits().max_buffer_size;
    let max_storage_size = device.limits().max_storage_buffer_binding_size as u64;
    if output_bytes > max_buf_size || output_bytes > max_storage_size {
        let req_mb = output_bytes / (1024 * 1024);
        let max_mb = max_buf_size.min(max_storage_size) / (1024 * 1024);
        return Err(format!(
            "Mesh SDF解像度 ({}x{}x{}, 必要容量: 約{}MB) がGPUのバッファ上限 ({}MB) を超えています。ボクセルサイズを大きくしてください。",
            width, height, depth, req_mb, max_mb
        ));
    }
    let max_3d_dim = device.limits().max_texture_dimension_3d;
    if width > max_3d_dim || height > max_3d_dim || depth > max_3d_dim {
        return Err(format!(
            "Mesh SDF寸法 ({}x{}x{}) がGPUの3Dテクスチャ最大解像度 ({}) を超えています。ボクセルサイズを大きくしてください。",
            width, height, depth, max_3d_dim
        ));
    }

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh SDF HierFine Output Buffer"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh SDF HierFine Staging Buffer"),
        size: output_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Mesh SDF HierFine BindGroup"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: triangles_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: coarse_dist_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: coarse_tri_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: hier_params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: output_buffer.as_entire_binding() },
        ],
    });

    // Zスラブ毎にsubmit＋pollし、長時間単一dispatchによるTDRを回避する
    let mut z0 = 0u32;
    while z0 < depth {
        let slab = (depth - z0).min(MESH_SDF_HIER_SLAB_LAYERS);
        let hp = GpuHierFineParams {
            coarse_w: coarse.width,
            coarse_h: coarse.height,
            coarse_d: coarse.depth,
            band,
            z_offset: z0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(&hier_params_buffer, 0, bytemuck::bytes_of(&hp));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Mesh SDF HierFine Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mesh SDF HierFine Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&compute_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((width + 3) / 4, (height + 3) / 4, (slab + 3) / 4);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        z0 += slab;
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Mesh SDF HierFine Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_bytes);
    queue.submit(Some(encoder.finish()));

    let texture_bytes = read_staging_to_vec(device, &staging_buffer)?;

    Ok(GpuMeshSdfBakeResult {
        texture_bytes,
        width,
        height,
        depth,
        bone_info,
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

    #[test]
    fn test_gpu_mesh_sdf_bake_cube() {
        if GpuContext::get_or_init().is_err() {
            println!("GPUが利用できない環境のためスキップします");
            return;
        }

        // 単位立方体メッシュ (8頂点, 12三角形)
        let verts = vec![
            [-0.5, -0.5, -0.5],
            [ 0.5, -0.5, -0.5],
            [ 0.5,  0.5, -0.5],
            [-0.5,  0.5, -0.5],
            [-0.5, -0.5,  0.5],
            [ 0.5, -0.5,  0.5],
            [ 0.5,  0.5,  0.5],
            [-0.5,  0.5,  0.5],
        ];

        let tris = vec![
            [0, 2, 1], [0, 3, 2], // -Z
            [4, 5, 6], [4, 6, 7], // +Z
            [0, 1, 5], [0, 5, 4], // -Y
            [2, 3, 7], [2, 7, 6], // +Y
            [0, 4, 7], [0, 7, 3], // -X
            [1, 2, 6], [1, 6, 5], // +X
        ];

        let res = bake_mesh_sdf_gpu(
            &verts,
            &tris,
            0.1,  // voxel_size (10cm)
            0.2,  // margin (20%)
            0.5,  // friction
            0.01, // thickness
            0.0,  // restitution
        );

        assert!(res.is_ok(), "メッシュSDFベイクが成功すること: {:?}", res.err());
        let result = res.unwrap();
        assert!(result.width >= 8);
        assert!(result.height >= 8);
        assert!(result.depth >= 8);
        assert_eq!(
            result.texture_bytes.len(),
            (result.width * result.height * result.depth * 4) as usize
        );

        // bone_info が正しく生成されていること
        assert_eq!(result.bone_info[3], 0.0);
        assert_eq!(result.bone_info[16], 0.5); // friction
        assert_eq!(result.bone_info[17], 0.01); // thickness
    }

    #[test]
    fn test_gpu_mesh_sdf_coarse_cube() {
        if GpuContext::get_or_init().is_err() {
            println!("GPUが利用できない環境のためスキップします");
            return;
        }

        // 単位立方体メッシュ (8頂点, 12三角形)
        let verts = vec![
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ];

        let tris = vec![
            [0, 2, 1], [0, 3, 2], // -Z
            [4, 5, 6], [4, 6, 7], // +Z
            [0, 1, 5], [0, 5, 4], // -Y
            [2, 3, 7], [2, 7, 6], // +Y
            [0, 4, 7], [0, 7, 3], // -X
            [1, 2, 6], [1, 6, 5], // +X
        ];

        let res = bake_mesh_sdf_coarse(
            &verts,
            &tris,
            0.05, // voxel_size (5cm) -> 粗セル幅 0.4m
            0.2,  // margin (20%)
            0.5,  // friction
            0.01, // thickness
            0.0,  // restitution
        );

        assert!(res.is_ok(), "粗グリッドベイクが成功すること: {:?}", res.err());
        let result = res.unwrap();
        // ドメイン 1.46m / 0.4m -> ceil 4 -> 最小クランプ 8
        assert_eq!((result.width, result.height, result.depth), (8, 8, 8));
        assert!((result.coarse_voxel - 0.2).abs() < 1e-6);
        assert_eq!(result.dist_packed.len(), 512);
        assert_eq!(result.nearest_tri.len(), 512 * 8);

        // 全インデックスが三角形数未満であること
        assert!(result.nearest_tri.iter().all(|&i| i < 12));
        // top-1列はソート済み（各セル先頭が最小距離の三角形）であること:
        // 各セル先頭4件のうち先頭が他3件以下の距離であることはシェーダ不変条件。
        // ここでは件数と範囲のみ検証し、距離順は階層等価テストで検証する。

        // 符号チェック (pack2x16float下位halfの符号ビット = u32 bit15)
        let flat = |x: u32, y: u32, z: u32| ((z * 8 + y) * 8 + x) as usize;
        let is_neg = |v: u32| (v & 0x8000) != 0;
        // 角セル (0,0,0): 立方体外 -> 正
        assert!(!is_neg(result.dist_packed[flat(0, 0, 0)]));
        // 中央セル (4,4,4): 立方体内 -> 負
        assert!(is_neg(result.dist_packed[flat(4, 4, 4)]));
    }

    #[test]
    fn test_gpu_mesh_sdf_hierarchical_matches_legacy() {
        if GpuContext::get_or_init().is_err() {
            println!("GPUが利用できない環境のためスキップします");
            return;
        }

        // 単位立方体メッシュ (8頂点, 12三角形)
        let verts = vec![
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ];

        let tris = vec![
            [0, 2, 1], [0, 3, 2], // -Z
            [4, 5, 6], [4, 6, 7], // +Z
            [0, 1, 5], [0, 5, 4], // -Y
            [2, 3, 7], [2, 7, 6], // +Y
            [0, 4, 7], [0, 7, 3], // -X
            [1, 2, 6], [1, 6, 5], // +X
        ];

        let legacy = bake_mesh_sdf_gpu(&verts, &tris, 0.05, 0.2, 0.5, 0.01, 0.0)
            .expect("legacyベイクが成功すること");
        let hier = bake_mesh_sdf_hierarchical(&verts, &tris, 0.05, 0.2, 0.5, 0.01, 0.0)
            .expect("階層ベイクが成功すること");

        assert_eq!((hier.width, hier.height, hier.depth), (legacy.width, legacy.height, legacy.depth));
        assert_eq!(hier.texture_bytes.len(), legacy.texture_bytes.len());

        let decode = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(4)
                .map(|c| {
                    let packed = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    half::f16::from_bits((packed & 0xFFFF) as u16).to_f32()
                })
                .collect()
        };
        let legacy_packed: Vec<u32> = legacy.texture_bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let hier_packed: Vec<u32> = hier.texture_bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let legacy_d = decode(&legacy.texture_bytes);
        let hier_d = decode(&hier.texture_bytes);

        // 近傍 (|d| <= band約0.20: 4 x 密ボクセル幅0.049＋余裕) はbit等価を要求
        // 遠方は符号一致＋差分上限（粗半対角約0.16＋f16量子化）を要求
        let mut near_mismatch = 0;
        let mut far_sign_mismatch = 0;
        let mut far_max_diff: f32 = 0.0;
        for i in 0..legacy_d.len() {
            if legacy_d[i].abs() <= 0.21 {
                if legacy_packed[i] != hier_packed[i] {
                    near_mismatch += 1;
                }
            } else {
                if (legacy_d[i] < 0.0) != (hier_d[i] < 0.0) {
                    far_sign_mismatch += 1;
                }
                far_max_diff = far_max_diff.max((legacy_d[i] - hier_d[i]).abs());
            }
        }
        println!(
            "hier-vs-legacy: voxels={} near_mismatch={} far_sign_mismatch={} far_max_diff={:.4}",
            legacy_d.len(), near_mismatch, far_sign_mismatch, far_max_diff
        );
        assert_eq!(near_mismatch, 0, "近傍場はbit等価であること");
        assert_eq!(far_sign_mismatch, 0, "遠方場の符号は一致すること");
        assert!(far_max_diff <= 0.18, "遠方場の差分は粗半対角以下であること");
    }
}
