use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::context::GpuContext;
use crate::debug_recorder::{SimulationDebugRecorder, SimulationMetadata};
use crate::mesh::{
    ClothMesh, GpuBendingConstraint, GpuCollider, GpuDistanceConstraint, GpuMeshTriangle,
    GpuPinConstraint, GpuSewingConstraint, GpuStarPair, GpuVertex, SelfCollisionParams, SimParams,
};
use crate::spatial_hash::GpuSpatialHash;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DispatchInfo {
    color_offset: u32,
    color_count: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PinParams {
    num_pins: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CollisionParams {
    num_vertices: u32,
    num_colliders: u32,
    num_mesh_triangles: u32,
    dt: f32,
    edge_margin_scale: f32,
    edge_margin_offset: f32,
    _pad0: f32,
    _pad1: f32,
}

pub struct GpuClothSimulator {
    pub context: Arc<GpuContext>,
    pub num_vertices: u32,
    pub num_distance_constraints: u32,
    pub num_bending_constraints: u32,
    pub num_sewing_constraints: u32,

    pub dist_color_offsets: Vec<u32>,
    pub dist_color_counts: Vec<u32>,
    pub bend_color_offsets: Vec<u32>,
    pub bend_color_counts: Vec<u32>,
    pub sew_color_offsets: Vec<u32>,
    pub sew_color_counts: Vec<u32>,

    initial_vertices: Vec<GpuVertex>,
    distance_constraints: Vec<GpuDistanceConstraint>,
    pub original_edge_to_constraint: Vec<usize>,
    pub initial_distance_rest_lengths: Vec<f32>,
    bending_constraints: Vec<GpuBendingConstraint>,

    vertex_buffer: wgpu::Buffer,
    dist_buffer: wgpu::Buffer,
    bend_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    // 非同期ダブルバッファリング用
    staging_buffers: [wgpu::Buffer; 2],
    staging_idx: usize,
    async_pending: bool,
    async_receiver: Option<futures_intrusive::channel::shared::OneshotReceiver<Result<(), wgpu::BufferAsyncError>>>,

    // チューニングオプション
    pub workgroup_size: u32,
    pub solver_mode: u32, // 0: Coloring (Gauss-Seidel), 1: Atomic (Jacobi)
    #[allow(dead_code)]
    accum_buffer: wgpu::Buffer,
    distance_atomic_solve_pipeline: wgpu::ComputePipeline,
    distance_atomic_apply_pipeline: wgpu::ComputePipeline,
    distance_atomic_bind_group: wgpu::BindGroup,

    // コンパクトリードバック用 (頂点座標 12B/頂点 のみ抽出・転送)
    pub enable_compact_readback: bool,
    compact_position_buffer: wgpu::Buffer,
    compact_staging_buffers: [wgpu::Buffer; 2],
    extract_positions_pipeline: wgpu::ComputePipeline,
    extract_positions_bind_group: wgpu::BindGroup,

    // フレームバッファリング用 (直近NフレームをGPU上で保持し一括転送)
    buffered_position_buffer: wgpu::Buffer,
    buffered_staging_buffer: wgpu::Buffer,
    pub buffered_frame_count: u32,
    pub max_buffered_frames: usize,

    // 動的ピン用
    dynamic_pins: HashMap<u32, GpuPinConstraint>,
    pin_buffer: wgpu::Buffer,
    pin_params_buffer: wgpu::Buffer,
    pin_bind_group: wgpu::BindGroup,
    pin_pipeline: wgpu::ComputePipeline,

    // プリミティブ & メッシュコライダー用
    colliders: Vec<GpuCollider>,
    mesh_triangles: Vec<GpuMeshTriangle>,
    collider_buffer: wgpu::Buffer,
    mesh_triangles_buffer: wgpu::Buffer,
    mesh_bounds_buffer: wgpu::Buffer,
    collider_params_buffer: wgpu::Buffer,
    collider_bind_group: wgpu::BindGroup,
    collision_pipeline: wgpu::ComputePipeline,

    // 自己・レイヤー衝突 & 貫通解消 (Untangling) 用
    spatial_hash: GpuSpatialHash,
    self_collision_pipeline: wgpu::ComputePipeline,
    self_collision_bind_group: wgpu::BindGroup,
    self_collision_params_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    normals_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    local_edge_lengths_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    adj_offsets_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    adj_indices_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    island_ids_buffer: wgpu::Buffer,
    compute_normals_pipeline: wgpu::ComputePipeline,
    compute_normals_bind_group: wgpu::BindGroup,

    predict_pipeline: wgpu::ComputePipeline,
    distance_pipeline: wgpu::ComputePipeline,
    bending_pipeline: wgpu::ComputePipeline,
    sewing_pipeline: wgpu::ComputePipeline,
    update_vel_pipeline: wgpu::ComputePipeline,

    predict_bind_group: wgpu::BindGroup,
    distance_bind_groups: Vec<wgpu::BindGroup>,
    bending_bind_groups: Vec<wgpu::BindGroup>,
    sewing_bind_groups: Vec<wgpu::BindGroup>,
    update_vel_bind_group: wgpu::BindGroup,

    pub solver_iterations: u32,
    pub gravity: [f32; 3],
    pub damping: f32,
    pub enable_self_collision: bool,
    pub self_collision_relief_factor: f32,
    pub self_collision_max_displacement_ratio: f32,
    pub self_collision_exclude_neighbors: bool,
    pub enable_normal_untangling: bool,
    pub enable_edge_collision: bool,
    pub edge_margin_scale: f32,
    pub edge_margin_offset: f32,
    edge_collision_pipeline: wgpu::ComputePipeline,
    edge_collision_bind_groups: Vec<wgpu::BindGroup>,

    // デバッグ記録用
    pub mesh_edges: Vec<[u32; 2]>,
    pub mesh_faces: Vec<[u32; 3]>,
    pub debug_recorder: SimulationDebugRecorder,
}

fn create_shader_with_wg_size(device: &wgpu::Device, label: &str, src: &str, wg_size: u32) -> wgpu::ShaderModule {
    let source = if wg_size != 64 {
        src.replace("@workgroup_size(64)", &format!("@workgroup_size({})", wg_size))
    } else {
        src.to_string()
    };
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

impl GpuClothSimulator {
    pub fn new(context: Arc<GpuContext>, mesh: ClothMesh) -> Self {
        Self::with_options(context, mesh, 32, 0)
    }

    pub fn with_options(
        context: Arc<GpuContext>,
        mesh: ClothMesh,
        workgroup_size: u32,
        solver_mode: u32,
    ) -> Self {
        let device = &context.device;
        let num_vertices = mesh.vertices.len() as u32;
        let num_distance_constraints = mesh.distance_constraints.len() as u32;
        let num_bending_constraints = mesh.bending_constraints.len() as u32;
        let num_sewing_constraints = mesh.sewing_constraints.len() as u32;

        let mut mesh = mesh;

        // 複数アイランドが存在し、全頂点のレイヤーが同一の場合、
        // 初期位置の平均Z座標に基づいて自動的にレイヤー階層（0, 1, ...）を付与する
        if !mesh.vertices.is_empty() && !mesh.island_ids.is_empty() {
            let first_layer = mesh.vertices[0].layer_id;
            let all_same_layer = mesh.vertices.iter().all(|v| v.layer_id == first_layer);
            if all_same_layer {
                let max_island = *mesh.island_ids.iter().max().unwrap_or(&0);
                if max_island > 0 {
                    // アイランドごとに平均Z座標を計算
                    let mut island_z_sum = vec![0.0f32; (max_island + 1) as usize];
                    let mut island_v_count = vec![0usize; (max_island + 1) as usize];
                    for (v_idx, &island) in mesh.island_ids.iter().enumerate() {
                        island_z_sum[island as usize] += mesh.vertices[v_idx].position[2];
                        island_v_count[island as usize] += 1;
                    }

                    let mut island_mean_z: Vec<(u32, f32)> = (0..=max_island)
                        .map(|id| {
                            let count = island_v_count[id as usize].max(1) as f32;
                            (id, island_z_sum[id as usize] / count)
                        })
                        .collect();

                    // 平均Zが低い順にソート (下にある布が layer 0、上にある布が layer 1, 2...)
                    island_mean_z.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                    let mut island_to_layer = vec![0u32; (max_island + 1) as usize];
                    for (layer_rank, &(island_id, _)) in island_mean_z.iter().enumerate() {
                        island_to_layer[island_id as usize] = layer_rank as u32;
                    }

                    for (v_idx, v) in mesh.vertices.iter_mut().enumerate() {
                        let island = mesh.island_ids[v_idx];
                        v.layer_id = island_to_layer[island as usize];
                    }
                }
            }
        }

        let initial_vertices = mesh.vertices.clone();

        // 1. GPU バッファの作成
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TareminCloth Vertex Buffer"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // 距離拘束バッファ
        let dummy_dist = GpuDistanceConstraint {
            v0: 0,
            v1: 0,
            rest_length: 0.0,
            tension_compliance: 0.0,
            compression_compliance: 0.0,
            constraint_type: 0,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        let dist_contents: &[u8] = if mesh.distance_constraints.is_empty() {
            bytemuck::bytes_of(&dummy_dist)
        } else {
            bytemuck::cast_slice(&mesh.distance_constraints)
        };

        let dist_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TareminCloth Distance Constraint Buffer"),
            contents: dist_contents,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // 曲げ拘束バッファ
        let dummy_bend = GpuBendingConstraint {
            v0: 0,
            v1: 0,
            v2: 0,
            v3: 0,
            rest_length: 0.0,
            compliance: 0.0,
            _pad: [0.0; 2],
        };
        let bend_contents: &[u8] = if mesh.bending_constraints.is_empty() {
            bytemuck::bytes_of(&dummy_bend)
        } else {
            bytemuck::cast_slice(&mesh.bending_constraints)
        };

        let bend_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TareminCloth Bending Constraint Buffer"),
            contents: bend_contents,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // 縫合拘束バッファ
        let dummy_sew = GpuSewingConstraint {
            v0: 0,
            v1: 0,
            current_rest_len: 0.0,
            target_rest_len: 0.0,
            shrink_speed: 0.0,
            compliance: 0.0,
            _pad: [0.0; 2],
        };
        let sew_contents: &[u8] = if mesh.sewing_constraints.is_empty() {
            bytemuck::bytes_of(&dummy_sew)
        } else {
            bytemuck::cast_slice(&mesh.sewing_constraints)
        };

        let sew_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TareminCloth Sewing Constraint Buffer"),
            contents: sew_contents,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // 動的ピンバッファ
        let max_pins = 1024;
        let pin_buffer_size = (max_pins * std::mem::size_of::<GpuPinConstraint>()) as u64;
        let pin_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TareminCloth Pin Buffer"),
            size: pin_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pin_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TareminCloth Pin Params Buffer"),
            contents: bytemuck::bytes_of(&PinParams {
                num_pins: 0,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // コライダーバッファ
        let max_colliders = 256;
        let collider_buffer_size = (max_colliders * std::mem::size_of::<GpuCollider>()) as u64;
        let collider_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TareminCloth Collider Buffer"),
            size: collider_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // メッシュ三角形コライダーバッファ (最大65536ポリゴン)
        let max_mesh_triangles = 65536;
        let mesh_triangles_buffer_size = (max_mesh_triangles * std::mem::size_of::<GpuMeshTriangle>()) as u64;
        let mesh_triangles_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TareminCloth Mesh Triangles Buffer"),
            size: mesh_triangles_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // メッシュ三角形の軽量境界球バッファ (vec4<f32>: xyz=中心, w=外接半径+厚み)
        let mesh_bounds_buffer_size = (max_mesh_triangles * std::mem::size_of::<[f32; 4]>()) as u64;
        let mesh_bounds_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TareminCloth Mesh Bounds Buffer"),
            size: mesh_bounds_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let collider_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TareminCloth Collider Params Buffer"),
            contents: bytemuck::bytes_of(&CollisionParams {
                num_vertices,
                num_colliders: 0,
                num_mesh_triangles: 0,
                dt: 0.016 / 20.0,
                edge_margin_scale: 1.0,
                edge_margin_offset: 0.0,
                _pad0: 0.0,
                _pad1: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let default_params = SimParams {
            gravity: [0.0, 0.0, -9.81, 0.016 / 20.0],
            damping: 0.05,
            substeps: 20,
            num_vertices,
            num_distance_constraints,
            num_bending_constraints,
            num_sewing_constraints,
            _pad: [0.0; 2],
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TareminCloth SimParams Buffer"),
            contents: bytemuck::bytes_of(&default_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let staging_buffer_size = (num_vertices as u64) * std::mem::size_of::<GpuVertex>() as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TareminCloth Staging Buffer"),
            size: staging_buffer_size.max(64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_buffer_0 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TareminCloth Staging Buffer 0"),
            size: staging_buffer_size.max(64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_buffer_1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TareminCloth Staging Buffer 1"),
            size: staging_buffer_size.max(64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // アトミック加算蓄積バッファ (各頂点: i32 dx, dy, dz, u32 count = 16 bytes)
        let accum_buffer_size = ((num_vertices as u64) * 16).max(64);
        let accum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TareminCloth Atomic Accum Buffer"),
            size: accum_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // コンパクトリードバック用バッファ (各頂点: f32 x, y, z = 12 bytes)
        let compact_buffer_size = ((num_vertices as u64) * 3 * 4).max(64);
        let compact_position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TareminCloth Compact Position Buffer"),
            size: compact_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let compact_staging_0 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TareminCloth Compact Staging Buffer 0"),
            size: compact_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let compact_staging_1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TareminCloth Compact Staging Buffer 1"),
            size: compact_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // フレームバッファリング用バッファ (最大8フレーム分)
        let max_buffered_frames = 8usize;
        let buffered_total_size = (compact_buffer_size * (max_buffered_frames as u64)).max(64);
        let buffered_position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TareminCloth Buffered Position Buffer"),
            size: buffered_total_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buffered_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TareminCloth Buffered Staging Buffer"),
            size: buffered_total_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 2. シェーダーモジュール
        let predict_shader = create_shader_with_wg_size(
            device,
            "Predict Shader",
            include_str!("shaders/predict.wgsl"),
            workgroup_size,
        );

        let distance_shader = create_shader_with_wg_size(
            device,
            "Distance Shader",
            include_str!("shaders/distance.wgsl"),
            workgroup_size,
        );

        let bending_shader = create_shader_with_wg_size(
            device,
            "Bending Shader",
            include_str!("shaders/bending.wgsl"),
            workgroup_size,
        );

        let sewing_shader = create_shader_with_wg_size(
            device,
            "Sewing Shader",
            include_str!("shaders/sewing.wgsl"),
            workgroup_size,
        );

        let collision_shader = create_shader_with_wg_size(
            device,
            "Collision Shader",
            include_str!("shaders/collision.wgsl"),
            workgroup_size,
        );

        let self_collision_shader = create_shader_with_wg_size(
            device,
            "Self Collision Shader",
            include_str!("shaders/self_collision.wgsl"),
            workgroup_size,
        );

        let compute_normals_shader = create_shader_with_wg_size(
            device,
            "Compute Normals Shader",
            include_str!("shaders/compute_normals.wgsl"),
            workgroup_size,
        );

        let pin_shader = create_shader_with_wg_size(
            device,
            "Pin Shader",
            include_str!("shaders/pin.wgsl"),
            workgroup_size,
        );

        let update_vel_shader = create_shader_with_wg_size(
            device,
            "Update Vel Shader",
            include_str!("shaders/update_vel.wgsl"),
            workgroup_size,
        );

        let edge_collision_shader = create_shader_with_wg_size(
            device,
            "Edge Collision Shader",
            include_str!("shaders/edge_collision.wgsl"),
            workgroup_size,
        );

        let distance_atomic_shader = create_shader_with_wg_size(
            device,
            "Distance Atomic Shader",
            include_str!("shaders/distance_atomic.wgsl"),
            workgroup_size,
        );

        let extract_positions_shader = create_shader_with_wg_size(
            device,
            "Extract Positions Shader",
            include_str!("shaders/extract_positions.wgsl"),
            workgroup_size,
        );

        // 3. Pipelines & Bind Groups
        let predict_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Predict BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let predict_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Predict Pipeline Layout"),
                bind_group_layouts: &[&predict_bgl],
                push_constant_ranges: &[],
            });

        let predict_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Predict Pipeline"),
            layout: Some(&predict_pipeline_layout),
            module: &predict_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let predict_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Predict BindGroup"),
            layout: &predict_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let constraint_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Constraint BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let mut_constraint_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mut Constraint BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Distance Pipeline
        let distance_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Distance Pipeline Layout"),
                bind_group_layouts: &[&constraint_bgl],
                push_constant_ranges: &[],
            });

        let distance_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Distance Pipeline"),
            layout: Some(&distance_pipeline_layout),
            module: &distance_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Distance Atomic Pipeline & Bind Group
        let distance_atomic_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Distance Atomic BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        let distance_atomic_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Distance Atomic Pipeline Layout"),
                bind_group_layouts: &[&distance_atomic_bgl],
                push_constant_ranges: &[],
            });

        let distance_atomic_solve_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Distance Atomic Solve Pipeline"),
                layout: Some(&distance_atomic_pipeline_layout),
                module: &distance_atomic_shader,
                entry_point: Some("solve_distance_atomic"),
                compilation_options: Default::default(),
                cache: None,
            });

        let distance_atomic_apply_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Distance Atomic Apply Pipeline"),
                layout: Some(&distance_atomic_pipeline_layout),
                module: &distance_atomic_shader,
                entry_point: Some("apply_atomic_accum"),
                compilation_options: Default::default(),
                cache: None,
            });

        let distance_atomic_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Distance Atomic BindGroup"),
            layout: &distance_atomic_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dist_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: accum_buffer.as_entire_binding(),
                },
            ],
        });

        // コンパクトリードバック用パイプライン & BindGroup
        let extract_positions_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Extract Positions BGL"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let extract_positions_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Extract Positions Pipeline Layout"),
                bind_group_layouts: &[&extract_positions_bgl],
                push_constant_ranges: &[],
            });

        let extract_positions_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Extract Positions Pipeline"),
                layout: Some(&extract_positions_pipeline_layout),
                module: &extract_positions_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let extract_positions_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Extract Positions BindGroup"),
            layout: &extract_positions_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: compact_position_buffer.as_entire_binding(),
                },
            ],
        });

        let mut distance_bind_groups = Vec::with_capacity(mesh.dist_color_counts.len());
        for (color_idx, &count) in mesh.dist_color_counts.iter().enumerate() {
            let offset = mesh.dist_color_offsets[color_idx];
            let dispatch_info = DispatchInfo {
                color_offset: offset,
                color_count: count,
                _pad0: 0,
                _pad1: 0,
            };
            let dispatch_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Dist DispatchInfo Buffer Color {}", color_idx)),
                contents: bytemuck::bytes_of(&dispatch_info),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Distance BindGroup Color {}", color_idx)),
                layout: &constraint_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: vertex_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dist_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dispatch_buf.as_entire_binding(),
                    },
                ],
            });
            distance_bind_groups.push(bg);
        }

        // Bending Pipeline
        let bending_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Bending Pipeline Layout"),
                bind_group_layouts: &[&constraint_bgl],
                push_constant_ranges: &[],
            });

        let bending_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bending Pipeline"),
            layout: Some(&bending_pipeline_layout),
            module: &bending_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut bending_bind_groups = Vec::with_capacity(mesh.bend_color_counts.len());
        for (color_idx, &count) in mesh.bend_color_counts.iter().enumerate() {
            let offset = mesh.bend_color_offsets[color_idx];
            let dispatch_info = DispatchInfo {
                color_offset: offset,
                color_count: count,
                _pad0: 0,
                _pad1: 0,
            };
            let dispatch_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Bend DispatchInfo Buffer Color {}", color_idx)),
                contents: bytemuck::bytes_of(&dispatch_info),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Bending BindGroup Color {}", color_idx)),
                layout: &constraint_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: vertex_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bend_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dispatch_buf.as_entire_binding(),
                    },
                ],
            });
            bending_bind_groups.push(bg);
        }

        // Sewing Pipeline
        let sewing_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Sewing Pipeline Layout"),
                bind_group_layouts: &[&mut_constraint_bgl],
                push_constant_ranges: &[],
            });

        let sewing_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sewing Pipeline"),
            layout: Some(&sewing_pipeline_layout),
            module: &sewing_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mut sewing_bind_groups = Vec::with_capacity(mesh.sew_color_counts.len());
        for (color_idx, &count) in mesh.sew_color_counts.iter().enumerate() {
            let offset = mesh.sew_color_offsets[color_idx];
            let dispatch_info = DispatchInfo {
                color_offset: offset,
                color_count: count,
                _pad0: 0,
                _pad1: 0,
            };
            let dispatch_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Sew DispatchInfo Buffer Color {}", color_idx)),
                contents: bytemuck::bytes_of(&dispatch_info),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Sewing BindGroup Color {}", color_idx)),
                layout: &mut_constraint_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: vertex_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: sew_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dispatch_buf.as_entire_binding(),
                    },
                ],
            });
            sewing_bind_groups.push(bg);
        }

        // Collision Pipeline (Primitive + Mesh Triangles)
        let collision_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Collision BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let collision_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Collision Pipeline Layout"),
                bind_group_layouts: &[&collision_bgl],
                push_constant_ranges: &[],
            });

        let collision_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Collision Pipeline"),
            layout: Some(&collision_pipeline_layout),
            module: &collision_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let collider_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Collision BindGroup"),
            layout: &collision_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: collider_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mesh_triangles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: collider_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: mesh_bounds_buffer.as_entire_binding(),
                },
            ],
        });

        // Edge Collision Pipeline & BindGroups (エッジ中点・コライダー頂点詳細接触用)
        let edge_collision_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Edge Collision BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let edge_collision_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Edge Collision Pipeline Layout"),
                bind_group_layouts: &[&edge_collision_bgl],
                push_constant_ranges: &[],
            });

        let edge_collision_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Edge Collision Pipeline"),
                layout: Some(&edge_collision_pipeline_layout),
                module: &edge_collision_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let mut edge_collision_bind_groups = Vec::with_capacity(mesh.dist_color_counts.len());
        for (color_idx, &count) in mesh.dist_color_counts.iter().enumerate() {
            let offset = mesh.dist_color_offsets[color_idx];
            let dispatch_info = DispatchInfo {
                color_offset: offset,
                color_count: count,
                _pad0: 0,
                _pad1: 0,
            };
            let dispatch_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("EdgeCol DispatchInfo Buffer Color {}", color_idx)),
                contents: bytemuck::bytes_of(&dispatch_info),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Edge Collision BindGroup Color {}", color_idx)),
                layout: &edge_collision_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: vertex_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dist_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: collider_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: mesh_triangles_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: mesh_bounds_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: collider_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: dispatch_buf.as_entire_binding(),
                    },
                ],
            });
            edge_collision_bind_groups.push(bg);
        }

        // Self & Layer Collision Pipeline (空間ハッシュ対応)
        let max_thick = mesh.vertices.iter().map(|v| v.thickness).fold(0.01f32, |a, b| a.max(b));
        let avg_local_len = if !mesh.local_edge_lengths.is_empty() {
            mesh.local_edge_lengths.iter().sum::<f32>() / mesh.local_edge_lengths.len() as f32
        } else {
            0.02
        };
        let cell_size = (max_thick * 2.0).max(avg_local_len).max(0.02);
        let spatial_hash = GpuSpatialHash::new(&context, &vertex_buffer, num_vertices, cell_size, 32768);

        // 法線バッファ (num_vertices * 16 bytes: vec4<f32>)
        let normals_init = vec![[0.0f32, 0.0, 1.0, 0.0]; num_vertices.max(1) as usize];
        let normals_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Normals Buffer"),
            contents: bytemuck::cast_slice(&normals_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // 局所エッジ長バッファ
        let default_local_edges;
        let local_edges_slice: &[f32] = if mesh.local_edge_lengths.is_empty() {
            default_local_edges = vec![0.01f32; num_vertices.max(1) as usize];
            &default_local_edges
        } else {
            &mesh.local_edge_lengths
        };
        let local_edge_lengths_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Local Edge Lengths Buffer"),
            contents: bytemuck::cast_slice(local_edges_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // 隣接頂点バッファ
        let default_adj_offsets;
        let adj_offsets_slice: &[u32] = if mesh.adj_offsets.is_empty() {
            default_adj_offsets = vec![0u32; (num_vertices + 1).max(2) as usize];
            &default_adj_offsets
        } else {
            &mesh.adj_offsets
        };
        let adj_offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Adj Offsets Buffer"),
            contents: bytemuck::cast_slice(adj_offsets_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let default_adj_indices;
        let adj_indices_slice: &[u32] = if mesh.adj_indices.is_empty() {
            default_adj_indices = vec![0u32; 1];
            &default_adj_indices
        } else {
            &mesh.adj_indices
        };
        let adj_indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Adj Indices Buffer"),
            contents: bytemuck::cast_slice(adj_indices_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // スターペアバッファ (法線計算用)
        let default_star_offsets;
        let star_offsets_slice: &[u32] = if mesh.star_offsets.is_empty() {
            default_star_offsets = vec![0u32; (num_vertices + 1).max(2) as usize];
            &default_star_offsets
        } else {
            &mesh.star_offsets
        };
        let star_offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Star Offsets Buffer"),
            contents: bytemuck::cast_slice(star_offsets_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let default_star_indices;
        let star_indices_slice: &[GpuStarPair] = if mesh.star_indices.is_empty() {
            default_star_indices = vec![GpuStarPair { v0: 0, v1: 0 }; 1];
            &default_star_indices
        } else {
            &mesh.star_indices
        };
        let star_indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Star Indices Buffer"),
            contents: bytemuck::cast_slice(star_indices_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // 連結成分 (アイランド) バッファ
        let default_island_ids;
        let island_ids_slice: &[u32] = if mesh.island_ids.is_empty() {
            default_island_ids = vec![0u32; num_vertices.max(1) as usize];
            &default_island_ids
        } else {
            &mesh.island_ids
        };
        let island_ids_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Island IDs Buffer"),
            contents: bytemuck::cast_slice(island_ids_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // 法線計算用パイプライン
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct NormalParams {
            num_vertices: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }
        let normal_params = NormalParams {
            num_vertices,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let normal_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Normal Params Buffer"),
            contents: bytemuck::bytes_of(&normal_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let compute_normals_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Normals BGL"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let compute_normals_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Normals Pipeline Layout"),
            bind_group_layouts: &[&compute_normals_bgl],
            push_constant_ranges: &[],
        });

        let compute_normals_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Normals Pipeline"),
            layout: Some(&compute_normals_pipeline_layout),
            module: &compute_normals_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let compute_normals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Normals BindGroup"),
            layout: &compute_normals_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: star_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: star_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: normals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: normal_params_buffer.as_entire_binding(),
                },
            ],
        });

        // 自己衝突パラメータ初期化
        let self_collision_relief_factor = 0.2f32;
        let self_collision_max_displacement_ratio = 0.2f32;
        let self_collision_exclude_neighbors = true;
        let enable_normal_untangling = true;

        let self_collision_params = SelfCollisionParams {
            cell_size: spatial_hash.cell_size,
            table_size: spatial_hash.table_size,
            num_vertices,
            _pad0: 0,
            relief_factor: self_collision_relief_factor,
            max_displacement_ratio: self_collision_max_displacement_ratio,
            enable_relief: 1,
            enable_normal_untangling: 1,
            exclude_neighbors: 1,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let self_collision_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Self Collision Params Buffer"),
            contents: bytemuck::bytes_of(&self_collision_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let self_collision_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Self Collision BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let self_collision_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Self Collision Pipeline Layout"),
                bind_group_layouts: &[&self_collision_bgl],
                push_constant_ranges: &[],
            });

        let self_collision_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Self Collision Pipeline"),
                layout: Some(&self_collision_pipeline_layout),
                module: &self_collision_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let self_collision_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Self Collision BindGroup"),
            layout: &self_collision_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spatial_hash.cell_heads_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: spatial_hash.vert_next_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self_collision_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: normals_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: local_edge_lengths_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: adj_offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: adj_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: island_ids_buffer.as_entire_binding(),
                },
            ],
        });

        // Pin Pipeline
        let pin_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Pin BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pin_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pin Pipeline Layout"),
                bind_group_layouts: &[&pin_bgl],
                push_constant_ranges: &[],
            });

        let pin_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pin Pipeline"),
            layout: Some(&pin_pipeline_layout),
            module: &pin_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pin_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pin BindGroup"),
            layout: &pin_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pin_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pin_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Update Vel Pipeline
        let update_vel_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Update Vel Pipeline Layout"),
                bind_group_layouts: &[&predict_bgl],
                push_constant_ranges: &[],
            });

        let update_vel_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Update Vel Pipeline"),
                layout: Some(&update_vel_pipeline_layout),
                module: &update_vel_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let update_vel_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Update Vel BindGroup"),
            layout: &predict_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mesh_edges: Vec<[u32; 2]> = mesh.distance_constraints.iter().map(|dc| [dc.v0, dc.v1]).collect();
        let mesh_faces: Vec<[u32; 3]> = mesh.triangles.iter().map(|tri| [tri.v0, tri.v1, tri.v2]).collect();

        Self {
            context,
            num_vertices,
            num_distance_constraints,
            num_bending_constraints,
            num_sewing_constraints,
            dist_color_offsets: mesh.dist_color_offsets,
            dist_color_counts: mesh.dist_color_counts,
            bend_color_offsets: mesh.bend_color_offsets,
            bend_color_counts: mesh.bend_color_counts,
            sew_color_offsets: mesh.sew_color_offsets,
            sew_color_counts: mesh.sew_color_counts,
            initial_vertices,
            distance_constraints: mesh.distance_constraints,
            original_edge_to_constraint: mesh.original_edge_to_constraint,
            initial_distance_rest_lengths: mesh.initial_distance_rest_lengths,
            bending_constraints: mesh.bending_constraints,
            vertex_buffer,
            dist_buffer,
            bend_buffer,
            params_buffer,
            staging_buffer,
            staging_buffers: [staging_buffer_0, staging_buffer_1],
            staging_idx: 0,
            async_pending: false,
            async_receiver: None,
            workgroup_size,
            solver_mode,
            accum_buffer,
            distance_atomic_solve_pipeline,
            distance_atomic_apply_pipeline,
            distance_atomic_bind_group,
            enable_compact_readback: true,
            compact_position_buffer,
            compact_staging_buffers: [compact_staging_0, compact_staging_1],
            extract_positions_pipeline,
            extract_positions_bind_group,
            buffered_position_buffer,
            buffered_staging_buffer,
            buffered_frame_count: 0,
            max_buffered_frames,
            dynamic_pins: HashMap::new(),
            pin_buffer,
            pin_params_buffer,
            pin_bind_group,
            pin_pipeline,
            colliders: Vec::new(),
            mesh_triangles: Vec::new(),
            collider_buffer,
            mesh_triangles_buffer,
            mesh_bounds_buffer,
            collider_params_buffer,
            collider_bind_group,
            collision_pipeline,
            spatial_hash,
            self_collision_pipeline,
            self_collision_bind_group,
            self_collision_params_buffer,
            normals_buffer,
            local_edge_lengths_buffer,
            adj_offsets_buffer,
            adj_indices_buffer,
            island_ids_buffer,
            compute_normals_pipeline,
            compute_normals_bind_group,
            predict_pipeline,
            distance_pipeline,
            bending_pipeline,
            sewing_pipeline,
            update_vel_pipeline,
            predict_bind_group,
            distance_bind_groups,
            bending_bind_groups,
            sewing_bind_groups,
            update_vel_bind_group,
            solver_iterations: 2,
            gravity: [0.0, 0.0, -9.81],
            damping: 1.0,
            enable_self_collision: true,
            self_collision_relief_factor,
            self_collision_max_displacement_ratio,
            self_collision_exclude_neighbors,
            enable_normal_untangling,
            enable_edge_collision: false,
            edge_margin_scale: 1.0,
            edge_margin_offset: 0.0,
            edge_collision_pipeline,
            edge_collision_bind_groups,
            mesh_edges,
            mesh_faces,
            debug_recorder: SimulationDebugRecorder::default(),
        }
    }

    /// 自己衝突および貫通解消オプションを設定する
    pub fn set_self_collision_options(
        &mut self,
        relief_factor: f32,
        max_displacement_ratio: f32,
        exclude_neighbors: bool,
        enable_normal_untangling: bool,
    ) {
        self.self_collision_relief_factor = relief_factor;
        self.self_collision_max_displacement_ratio = max_displacement_ratio;
        self.self_collision_exclude_neighbors = exclude_neighbors;
        self.enable_normal_untangling = enable_normal_untangling;

        let params = SelfCollisionParams {
            cell_size: self.spatial_hash.cell_size,
            table_size: self.spatial_hash.table_size,
            num_vertices: self.num_vertices,
            _pad0: 0,
            relief_factor,
            max_displacement_ratio,
            enable_relief: if relief_factor < 0.999 { 1 } else { 0 },
            enable_normal_untangling: if enable_normal_untangling { 1 } else { 0 },
            exclude_neighbors: if exclude_neighbors { 1 } else { 0 },
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        self.context.queue.write_buffer(
            &self.self_collision_params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );
    }

    /// エッジ詳細接触マージン倍率を設定する (1.0 = 標準, 1.2〜1.5 = 安全マージン付き)
    pub fn set_edge_margin_scale(&mut self, scale: f32) {
        self.edge_margin_scale = scale.max(1.0);
    }

    /// エッジ詳細接触マージン固定加算値を設定する (m単位, 0.0 = 加算なし, 0.005 = 5mm安全クリアランス)
    pub fn set_edge_margin_offset(&mut self, offset: f32) {
        self.edge_margin_offset = offset.max(0.0);
    }

    /// エッジ詳細接触判定の有効/無効を設定する
    pub fn set_enable_edge_collision(&mut self, enable: bool) {
        self.enable_edge_collision = enable;
    }

    /// 自己衝突・レイヤー衝突処理の有効/無効を設定する
    pub fn set_enable_self_collision(&mut self, enable: bool) {
        self.enable_self_collision = enable;
    }

    /// 拘束解決の反復回数を設定する (1 サブステップあたり)
    pub fn set_solver_iterations(&mut self, iterations: u32) {
        self.solver_iterations = iterations.max(1);
    }

    /// 剛性パラメータ（伸縮剛性・曲げ剛性）を動的に更新する (後方互換用)
    pub fn set_stiffness(&mut self, stretch_stiffness: f32, bending_stiffness: f32) {
        self.set_stiffness_all(stretch_stiffness, stretch_stiffness, stretch_stiffness * 0.5, bending_stiffness);
    }

    /// 剛性4種（引張・圧縮・せん断・曲げ）を動的に更新する
    pub fn set_stiffness_all(
        &mut self,
        tension_stiffness: f32,
        compression_stiffness: f32,
        shear_stiffness: f32,
        bending_stiffness: f32,
    ) {
        let tension_comp = if tension_stiffness >= 5000.0 {
            0.0
        } else if tension_stiffness > 0.0 {
            1.0 / (tension_stiffness * 1000.0)
        } else {
            1e10
        };

        let compression_comp = if compression_stiffness >= 5000.0 {
            0.0
        } else if compression_stiffness > 0.0 {
            1.0 / (compression_stiffness * 1000.0)
        } else {
            1e10
        };

        let shear_comp = if shear_stiffness >= 5000.0 {
            0.0
        } else if shear_stiffness > 0.0 {
            1.0 / (shear_stiffness * 1000.0)
        } else {
            1e10
        };

        for c in &mut self.distance_constraints {
            if c.constraint_type == 1 {
                c.tension_compliance = shear_comp;
                c.compression_compliance = shear_comp;
            } else {
                c.tension_compliance = tension_comp;
                c.compression_compliance = compression_comp;
            }
        }
        if !self.distance_constraints.is_empty() {
            self.context.queue.write_buffer(
                &self.dist_buffer,
                0,
                bytemuck::cast_slice(&self.distance_constraints),
            );
        }

        let bend_comp = if bending_stiffness > 0.0 {
            1.0 / (bending_stiffness * 100.0)
        } else {
            1e10
        };
        for c in &mut self.bending_constraints {
            c.compliance = bend_comp;
        }
        if !self.bending_constraints.is_empty() {
            self.context.queue.write_buffer(
                &self.bend_buffer,
                0,
                bytemuck::cast_slice(&self.bending_constraints),
            );
        }
    }

    /// 大域空気減衰率（Air Damping / Velocity Damping）を設定する
    pub fn set_damping(&mut self, damping: f32) {
        self.damping = damping.max(0.0);
    }

    /// 減衰4種（引張・圧縮・せん断・曲げ）を反映して減衰率を設定する
    pub fn set_damping_all(
        &mut self,
        tension_damp: f32,
        compression_damp: f32,
        shear_damp: f32,
        bending_damp: f32,
    ) {
        // 各拘束の減衰の寄与を合算
        let internal_damp = (tension_damp + compression_damp + shear_damp + bending_damp) * 0.05;
        self.damping = (self.damping + internal_damp).max(0.0);
    }

    /// 指定された元エッジインデックス群に対して自然長スケール（倍率）を設定し、GPUバッファに反映する
    pub fn set_edge_rest_length_scales(&mut self, edge_indices: &[u32], scales: &[f32]) {
        let count = edge_indices.len().min(scales.len());
        let mut changed = false;

        for i in 0..count {
            let edge_idx = edge_indices[i] as usize;
            if edge_idx < self.original_edge_to_constraint.len() {
                let constraint_idx = self.original_edge_to_constraint[edge_idx];
                if constraint_idx != usize::MAX && constraint_idx < self.distance_constraints.len() {
                    let base_len = self.initial_distance_rest_lengths[constraint_idx];
                    let scale = scales[i].max(0.01);
                    self.distance_constraints[constraint_idx].rest_length = base_len * scale;
                    changed = true;
                }
            }
        }

        if changed && !self.distance_constraints.is_empty() {
            self.context.queue.write_buffer(
                &self.dist_buffer,
                0,
                bytemuck::cast_slice(&self.distance_constraints),
            );
        }
    }

    /// 全エッジの自然長を初期長（スケール 1.0）にリセットし、GPUバッファに反映する
    pub fn reset_edge_rest_lengths(&mut self) {
        if self.distance_constraints.len() != self.initial_distance_rest_lengths.len() {
            return;
        }
        for (c, &base_len) in self.distance_constraints.iter_mut().zip(&self.initial_distance_rest_lengths) {
            c.rest_length = base_len;
        }
        if !self.distance_constraints.is_empty() {
            self.context.queue.write_buffer(
                &self.dist_buffer,
                0,
                bytemuck::cast_slice(&self.distance_constraints),
            );
        }
    }

    /// 指定された元エッジインデックスの現在の自然長を取得する
    pub fn get_edge_rest_length(&self, edge_idx: u32) -> f32 {
        let edge_idx = edge_idx as usize;
        if edge_idx < self.original_edge_to_constraint.len() {
            let constraint_idx = self.original_edge_to_constraint[edge_idx];
            if constraint_idx != usize::MAX && constraint_idx < self.distance_constraints.len() {
                return self.distance_constraints[constraint_idx].rest_length;
            }
        }
        0.0
    }

    /// 指定された元エッジインデックスの初期自然長を取得する
    pub fn get_edge_initial_rest_length(&self, edge_idx: u32) -> f32 {
        let edge_idx = edge_idx as usize;
        if edge_idx < self.original_edge_to_constraint.len() {
            let constraint_idx = self.original_edge_to_constraint[edge_idx];
            if constraint_idx != usize::MAX && constraint_idx < self.initial_distance_rest_lengths.len() {
                return self.initial_distance_rest_lengths[constraint_idx];
            }
        }
        0.0
    }

    /// 球コライダーを追加
    pub fn add_sphere_collider(&mut self, center: [f32; 3], radius: f32, friction: f32, restitution: f32) {
        self.colliders.push(GpuCollider {
            collider_type: 0,
            friction,
            restitution,
            _pad0: 0.0,
            point_a: center,
            radius,
            point_b: [0.0; 3],
            _pad1: 0.0,
        });
        self.upload_colliders();
    }

    /// カプセルコライダーを追加
    pub fn add_capsule_collider(&mut self, a: [f32; 3], b: [f32; 3], radius: f32, friction: f32, restitution: f32) {
        self.colliders.push(GpuCollider {
            collider_type: 1,
            friction,
            restitution,
            _pad0: 0.0,
            point_a: a,
            radius,
            point_b: b,
            _pad1: 0.0,
        });
        self.upload_colliders();
    }

    /// 平面コライダーを追加
    pub fn add_plane_collider(&mut self, point: [f32; 3], normal: [f32; 3], friction: f32, restitution: f32) {
        self.colliders.push(GpuCollider {
            collider_type: 2,
            friction,
            restitution,
            _pad0: 0.0,
            point_a: point,
            radius: 0.0,
            point_b: normal,
            _pad1: 0.0,
        });
        self.upload_colliders();
    }

    /// メッシュ三角形コライダーを設定
    pub fn set_mesh_triangles(&mut self, triangles: &[GpuMeshTriangle]) {
        self.mesh_triangles = triangles.to_vec();
        self.upload_colliders();
    }

    /// すべてのコライダーをクリア
    pub fn clear_colliders(&mut self) {
        self.colliders.clear();
        self.mesh_triangles.clear();
        self.upload_colliders();
    }

    fn upload_colliders(&self) {
        let num_colliders = self.colliders.len() as u32;
        if num_colliders > 0 {
            self.context.queue.write_buffer(
                &self.collider_buffer,
                0,
                bytemuck::cast_slice(&self.colliders),
            );
        }

        let num_mesh_triangles = self.mesh_triangles.len() as u32;
        if num_mesh_triangles > 0 {
            self.context.queue.write_buffer(
                &self.mesh_triangles_buffer,
                0,
                bytemuck::cast_slice(&self.mesh_triangles),
            );

            // 軽量境界球（中心vec3 + 外接半径f32）の事前計算とアップロード
            let mut bounds = Vec::with_capacity(self.mesh_triangles.len());
            for t in &self.mesh_triangles {
                let cx = (t.p0[0] + t.p1[0] + t.p2[0]) / 3.0;
                let cy = (t.p0[1] + t.p1[1] + t.p2[1]) / 3.0;
                let cz = (t.p0[2] + t.p1[2] + t.p2[2]) / 3.0;
                let d0 = (t.p0[0] - cx).powi(2) + (t.p0[1] - cy).powi(2) + (t.p0[2] - cz).powi(2);
                let d1 = (t.p1[0] - cx).powi(2) + (t.p1[1] - cy).powi(2) + (t.p1[2] - cz).powi(2);
                let d2 = (t.p2[0] - cx).powi(2) + (t.p2[1] - cy).powi(2) + (t.p2[2] - cz).powi(2);
                let r = d0.max(d1).max(d2).sqrt() + t.thickness;
                bounds.push([cx, cy, cz, r]);
            }
            self.context.queue.write_buffer(
                &self.mesh_bounds_buffer,
                0,
                bytemuck::cast_slice(&bounds),
            );
        }

        let params = CollisionParams {
            num_vertices: self.num_vertices,
            num_colliders,
            num_mesh_triangles,
            dt: 0.016 / 20.0,
            edge_margin_scale: self.edge_margin_scale,
            edge_margin_offset: self.edge_margin_offset,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        self.context.queue.write_buffer(
            &self.collider_params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );
    }

    /// 動的ピンを設定・更新する
    pub fn set_pin_target(&mut self, vertex_idx: u32, target_pos: [f32; 3], weight: f32) {
        if vertex_idx < self.num_vertices {
            self.dynamic_pins.insert(
                vertex_idx,
                GpuPinConstraint {
                    vertex_idx,
                    weight,
                    _pad: [0.0; 2],
                    target_pos,
                    _pad2: 0.0,
                },
            );
            self.upload_pins();
        }
    }

    /// 指定頂点の動的ピンを解除する
    pub fn release_pin(&mut self, vertex_idx: u32) {
        if self.dynamic_pins.remove(&vertex_idx).is_some() {
            self.upload_pins();
        }
    }

    /// すべての動的ピンを解除する
    pub fn clear_dynamic_pins(&mut self) {
        self.dynamic_pins.clear();
        self.upload_pins();
    }

    fn upload_pins(&self) {
        let pin_vec: Vec<GpuPinConstraint> = self.dynamic_pins.values().copied().collect();
        let num_pins = pin_vec.len() as u32;

        if num_pins > 0 {
            self.context.queue.write_buffer(
                &self.pin_buffer,
                0,
                bytemuck::cast_slice(&pin_vec),
            );
        }

        let pin_params = PinParams {
            num_pins,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        self.context.queue.write_buffer(
            &self.pin_params_buffer,
            0,
            bytemuck::bytes_of(&pin_params),
        );
    }

    /// シミュレーションの GPU コマンド（外力予測・衝突・拘束解決・速度更新）をエンコーダーに記録する
    fn encode_simulation_steps(&self, encoder: &mut wgpu::CommandEncoder, dt: f32, substeps: u32) {
        if self.num_vertices == 0 {
            return;
        }

        let substep_dt = dt / (substeps as f32);
        let params = SimParams {
            gravity: [self.gravity[0], self.gravity[1], self.gravity[2], substep_dt],
            damping: self.damping,
            substeps,
            num_vertices: self.num_vertices,
            num_distance_constraints: self.num_distance_constraints,
            num_bending_constraints: self.num_bending_constraints,
            num_sewing_constraints: self.num_sewing_constraints,
            _pad: [0.0; 2],
        };

        self.context.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );

        let wg_size = self.workgroup_size;
        let vert_workgroups = (self.num_vertices + wg_size - 1) / wg_size;
        let num_pins = self.dynamic_pins.len() as u32;
        let pin_workgroups = (num_pins + wg_size - 1) / wg_size;
        let has_colliders = !self.colliders.is_empty() || !self.mesh_triangles.is_empty();

        if has_colliders {
            let col_params = CollisionParams {
                num_vertices: self.num_vertices,
                num_colliders: self.colliders.len() as u32,
                num_mesh_triangles: self.mesh_triangles.len() as u32,
                dt: substep_dt,
                edge_margin_scale: self.edge_margin_scale,
                edge_margin_offset: self.edge_margin_offset,
                _pad0: 0.0,
                _pad1: 0.0,
            };
            self.context.queue.write_buffer(
                &self.collider_params_buffer,
                0,
                bytemuck::bytes_of(&col_params),
            );
        }

        for _ in 0..substeps {
            // 1. Predict Pass
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Predict Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.predict_pipeline);
                cpass.set_bind_group(0, &self.predict_bind_group, &[]);
                cpass.dispatch_workgroups(vert_workgroups, 1, 1);
            }

            // 1.5 Compute Normals Pass (自己衝突 & 法線Untangling用)
            if self.enable_self_collision {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Normals Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.compute_normals_pipeline);
                cpass.set_bind_group(0, &self.compute_normals_bind_group, &[]);
                cpass.dispatch_workgroups(vert_workgroups, 1, 1);
            }

            // 1.6 Self & Layer Collision Pass (GPU空間ハッシュ: 有効時のみ実行)
            // 拘束解決（Distance/Bending）の前にめり込みを解消することで、
            // 自己衝突で生じた変位をバネ拘束が滑らかに吸収・調停し、自励振動を防止する
            if self.enable_self_collision {
                let hash_clear_workgroups = (self.spatial_hash.table_size + wg_size - 1) / wg_size;
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SpatialHash Clear Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.spatial_hash.clear_pipeline);
                cpass.set_bind_group(0, &self.spatial_hash.build_bind_group, &[]);
                cpass.dispatch_workgroups(hash_clear_workgroups, 1, 1);
            }
            if self.enable_self_collision {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SpatialHash Build Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.spatial_hash.build_pipeline);
                cpass.set_bind_group(0, &self.spatial_hash.build_bind_group, &[]);
                cpass.dispatch_workgroups(vert_workgroups, 1, 1);
            }
            if self.enable_self_collision {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Self Collision Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.self_collision_pipeline);
                cpass.set_bind_group(0, &self.self_collision_bind_group, &[]);
                cpass.dispatch_workgroups(vert_workgroups, 1, 1);
            }

            // 2. Collision Constraints Pass (1サブステップあたり1回実行)
            if has_colliders {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Collision Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.collision_pipeline);
                cpass.set_bind_group(0, &self.collider_bind_group, &[]);
                cpass.dispatch_workgroups(vert_workgroups, 1, 1);
            }

            // 2.5. Edge Collision Constraints Pass (オプション有効時のみ実行)
            if has_colliders && self.enable_edge_collision {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Edge Collision Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.edge_collision_pipeline);
                for (color_idx, &count) in self.dist_color_counts.iter().enumerate() {
                    if count > 0 {
                        cpass.set_bind_group(0, &self.edge_collision_bind_groups[color_idx], &[]);
                        cpass.dispatch_workgroups((count + wg_size - 1) / wg_size, 1, 1);
                    }
                }
            }

            // 拘束解決反復ループ (Solver Iterations)
            for _ in 0..self.solver_iterations {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Solver Iteration Pass"),
                    timestamp_writes: None,
                });

                // 1. Pin Constraints
                if num_pins > 0 {
                    cpass.set_pipeline(&self.pin_pipeline);
                    cpass.set_bind_group(0, &self.pin_bind_group, &[]);
                    cpass.dispatch_workgroups(pin_workgroups, 1, 1);
                }

                // 2. Distance Constraints Projection
                if self.solver_mode == 1 {
                    // Atomic Jacobi モード (全エッジを単一ディスパッチで一斉評価 + 頂点変位平均適用)
                    if self.num_distance_constraints > 0 {
                        let edge_workgroups = (self.num_distance_constraints + wg_size - 1) / wg_size;
                        cpass.set_pipeline(&self.distance_atomic_solve_pipeline);
                        cpass.set_bind_group(0, &self.distance_atomic_bind_group, &[]);
                        cpass.dispatch_workgroups(edge_workgroups, 1, 1);
                    }
                    cpass.set_pipeline(&self.distance_atomic_apply_pipeline);
                    cpass.set_bind_group(0, &self.distance_atomic_bind_group, &[]);
                    cpass.dispatch_workgroups(vert_workgroups, 1, 1);
                } else {
                    // Coloring モード (従来どおり全色グループを同一Pass内で連続ディスパッチ)
                    cpass.set_pipeline(&self.distance_pipeline);
                    for (color_idx, &count) in self.dist_color_counts.iter().enumerate() {
                        if count > 0 {
                            cpass.set_bind_group(0, &self.distance_bind_groups[color_idx], &[]);
                            cpass.dispatch_workgroups((count + wg_size - 1) / wg_size, 1, 1);
                        }
                    }
                }

                // 3. Sewing Constraints Projection
                cpass.set_pipeline(&self.sewing_pipeline);
                for (color_idx, &count) in self.sew_color_counts.iter().enumerate() {
                    if count > 0 {
                        cpass.set_bind_group(0, &self.sewing_bind_groups[color_idx], &[]);
                        cpass.dispatch_workgroups((count + wg_size - 1) / wg_size, 1, 1);
                    }
                }

                // 4. Bending Constraints Projection (反復ループ内でDistance等と同調解決)
                if !self.bend_color_counts.is_empty() {
                    cpass.set_pipeline(&self.bending_pipeline);
                    for (color_idx, &count) in self.bend_color_counts.iter().enumerate() {
                        if count > 0 {
                            cpass.set_bind_group(0, &self.bending_bind_groups[color_idx], &[]);
                            cpass.dispatch_workgroups((count + wg_size - 1) / wg_size, 1, 1);
                        }
                    }
                }
            }

            // 5. 反復終了後にピン位置を厳密に固定
            if num_pins > 0 {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Final Pin Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pin_pipeline);
                cpass.set_bind_group(0, &self.pin_bind_group, &[]);
                cpass.dispatch_workgroups(pin_workgroups, 1, 1);
            }

            // 6. Update Vel & Commit Positions Pass
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Update Vel Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.update_vel_pipeline);
                cpass.set_bind_group(0, &self.update_vel_bind_group, &[]);
                cpass.dispatch_workgroups(vert_workgroups, 1, 1);
            }
        }
    }

    /// シミュレーションを同期的に 1 フレーム進行する
    pub fn step(&mut self, dt: f32, substeps: u32) {
        if self.num_vertices == 0 {
            return;
        }

        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Simulation Frame Encoder"),
            },
        );

        self.encode_simulation_steps(&mut encoder, dt, substeps);
        self.context.queue.submit(Some(encoder.finish()));

        if self.debug_recorder.is_recording() {
            self.record_current_frame(dt, substeps);
        }
    }

    /// 頂点座標抽出パスをエンコード (GpuVertex から連続 f32 座標配列へ抽出)
    fn encode_extract_positions(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg_size = self.workgroup_size as u64;
        let vert_workgroups = ((self.num_vertices as u64) + wg_size - 1) / wg_size;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Extract Positions Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.extract_positions_pipeline);
        cpass.set_bind_group(0, &self.extract_positions_bind_group, &[]);
        cpass.dispatch_workgroups(vert_workgroups as u32, 1, 1);
    }

    /// シミュレーションを非同期に 1 フレーム進行する（ステージングバッファへコピーし map_async 発行、CPUブロックなし）
    pub fn step_async(&mut self, dt: f32, substeps: u32) {
        if self.num_vertices == 0 {
            return;
        }

        // 前の非同期マッピングが未回収の場合、自動的に待機・回収してクリーンアップ
        if self.async_pending {
            let mut dummy = vec![0.0f32; (self.num_vertices as usize) * 3];
            self.fetch_positions_flat(&mut dummy);
        }

        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Simulation Frame Async Encoder"),
            },
        );

        self.encode_simulation_steps(&mut encoder, dt, substeps);

        if self.enable_compact_readback {
            self.encode_extract_positions(&mut encoder);
            let compact_size = (self.num_vertices as u64) * 3 * 4;
            let active_staging = &self.compact_staging_buffers[self.staging_idx];
            encoder.copy_buffer_to_buffer(
                &self.compact_position_buffer,
                0,
                active_staging,
                0,
                compact_size,
            );
            self.context.queue.submit(Some(encoder.finish()));

            let slice = active_staging.slice(..compact_size);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |v| {
                let _ = sender.send(v);
            });
            self.async_receiver = Some(receiver);
        } else {
            let buffer_size = (self.num_vertices as u64) * std::mem::size_of::<GpuVertex>() as u64;
            let active_staging = &self.staging_buffers[self.staging_idx];
            encoder.copy_buffer_to_buffer(
                &self.vertex_buffer,
                0,
                active_staging,
                0,
                buffer_size,
            );
            self.context.queue.submit(Some(encoder.finish()));

            let slice = active_staging.slice(..buffer_size);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |v| {
                let _ = sender.send(v);
            });
            self.async_receiver = Some(receiver);
        }

        self.async_pending = true;
        self.staging_idx = 1 - self.staging_idx;
    }

    /// 非同期実行された前フレームの頂点座標を回収する (成功時 true, 未実行/失敗時 false)
    pub fn fetch_positions_flat(&mut self, out: &mut [f32]) -> bool {
        if !self.async_pending || self.num_vertices == 0 {
            return false;
        }

        let read_idx = 1 - self.staging_idx;

        if let Some(receiver) = self.async_receiver.take() {
            self.context.device.poll(wgpu::Maintain::Wait);
            if let Some(Ok(())) = pollster::block_on(receiver.receive()) {
                if self.enable_compact_readback {
                    let compact_size = (self.num_vertices as u64) * 3 * 4;
                    let slice = self.compact_staging_buffers[read_idx].slice(..compact_size);
                    let data = slice.get_mapped_range();
                    let floats: &[f32] = bytemuck::cast_slice(&data);
                    let copy_len = out.len().min(floats.len());
                    out[..copy_len].copy_from_slice(&floats[..copy_len]);

                    drop(data);
                    self.compact_staging_buffers[read_idx].unmap();
                } else {
                    let buffer_size = (self.num_vertices as u64) * std::mem::size_of::<GpuVertex>() as u64;
                    let slice = self.staging_buffers[read_idx].slice(..buffer_size);
                    let data = slice.get_mapped_range();
                    let verts: &[GpuVertex] = bytemuck::cast_slice(&data);

                    for (i, v) in verts.iter().enumerate() {
                        let base = i * 3;
                        if base + 2 < out.len() {
                            out[base] = v.position[0];
                            out[base + 1] = v.position[1];
                            out[base + 2] = v.position[2];
                        }
                    }

                    drop(data);
                    self.staging_buffers[read_idx].unmap();
                }

                self.async_pending = false;
                return true;
            }
        }

        self.async_pending = false;
        false
    }

    /// コンパクトリードバックの有効/無効を設定する (true: 12B/頂点, false: 48B/頂点)
    pub fn set_enable_compact_readback(&mut self, enable: bool) {
        self.enable_compact_readback = enable;
    }

    /// ソルバーモードを設定する (0: Coloring, 1: Atomic Jacobi)
    pub fn set_solver_mode(&mut self, mode: u32) {
        self.solver_mode = mode;
    }

    /// ワークグループサイズを設定する
    pub fn set_workgroup_size(&mut self, wg_size: u32) {
        self.workgroup_size = wg_size;
    }

    /// GPU から頂点データを読み出す
    pub fn read_vertices(&self) -> Vec<GpuVertex> {
        if self.num_vertices == 0 {
            return Vec::new();
        }

        let buffer_size = (self.num_vertices as u64) * std::mem::size_of::<GpuVertex>() as u64;

        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Read Vertices Encoder"),
            },
        );
        encoder.copy_buffer_to_buffer(
            &self.vertex_buffer,
            0,
            &self.staging_buffer,
            0,
            buffer_size,
        );
        self.context.queue.submit(Some(encoder.finish()));

        let slice = self.staging_buffer.slice(..buffer_size);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.context.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver.receive()).unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<GpuVertex> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging_buffer.unmap();

        result
    }

    /// 頂点座標を flat array に直接書き込む（ヒープアロケーションなし）
    pub fn get_positions_flat(&self, out: &mut [f32]) {
        if self.num_vertices == 0 {
            return;
        }

        if self.enable_compact_readback {
            let compact_size = (self.num_vertices as u64) * 3 * 4;
            let mut encoder = self.context.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("Compact Read Positions Encoder"),
                },
            );
            self.encode_extract_positions(&mut encoder);
            encoder.copy_buffer_to_buffer(
                &self.compact_position_buffer,
                0,
                &self.compact_staging_buffers[0],
                0,
                compact_size,
            );
            self.context.queue.submit(Some(encoder.finish()));

            let slice = self.compact_staging_buffers[0].slice(..compact_size);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            self.context.device.poll(wgpu::Maintain::Wait);
            pollster::block_on(receiver.receive()).unwrap().unwrap();

            let data = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&data);
            let copy_len = out.len().min(floats.len());
            out[..copy_len].copy_from_slice(&floats[..copy_len]);

            drop(data);
            self.compact_staging_buffers[0].unmap();
        } else {
            let buffer_size = (self.num_vertices as u64) * std::mem::size_of::<GpuVertex>() as u64;

            let mut encoder = self.context.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("Read Positions Encoder"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.vertex_buffer,
                0,
                &self.staging_buffer,
                0,
                buffer_size,
            );
            self.context.queue.submit(Some(encoder.finish()));

            let slice = self.staging_buffer.slice(..buffer_size);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            self.context.device.poll(wgpu::Maintain::Wait);
            pollster::block_on(receiver.receive()).unwrap().unwrap();

            let data = slice.get_mapped_range();
            let verts: &[GpuVertex] = bytemuck::cast_slice(&data);

            for (i, v) in verts.iter().enumerate() {
                let base = i * 3;
                if base + 2 < out.len() {
                    out[base] = v.position[0];
                    out[base + 1] = v.position[1];
                    out[base + 2] = v.position[2];
                }
            }

            drop(data);
            self.staging_buffer.unmap();
        }
    }

    /// シミュレーションを 1 フレーム進め、計算結果座標をGPU内部のリングバッファに保存する。
    /// （GPU同期待機やCPUへの転送は行わず、即座に復帰します）
    /// 戻り値: 現在バッファリングされているフレーム数 (1 ..= max_buffered_frames)
    pub fn step_buffered(&mut self, dt: f32, substeps: u32) -> u32 {
        if self.num_vertices == 0 {
            return 0;
        }

        let slot = (self.buffered_frame_count as usize).min(self.max_buffered_frames - 1);

        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Step Buffered Frame Encoder"),
            },
        );

        // 1. シミュレーション計算
        self.encode_simulation_steps(&mut encoder, dt, substeps);

        // 2. 頂点座標の抽出 (GpuVertex -> compact_position_buffer)
        self.encode_extract_positions(&mut encoder);

        // 3. GPU内部でリングバッファの該当スロットへコピー (所要時間 ~0.005ms)
        let slot_size = (self.num_vertices as u64) * 3 * 4;
        let dst_offset = (slot as u64) * slot_size;
        encoder.copy_buffer_to_buffer(
            &self.compact_position_buffer,
            0,
            &self.buffered_position_buffer,
            dst_offset,
            slot_size,
        );

        self.context.queue.submit(Some(encoder.finish()));

        if (self.buffered_frame_count as usize) < self.max_buffered_frames {
            self.buffered_frame_count += 1;
        }
        self.buffered_frame_count
    }

    /// 現在リングバッファにたまっているフレーム数を取得
    pub fn get_buffered_frame_count(&self) -> u32 {
        self.buffered_frame_count
    }

    /// リングバッファにたまっている全フレームの頂点座標を一度のPCIe転送でまとめて取得する。
    /// out の長さは (buffered_frame_count * num_vertices * 3) 以上である必要があります。
    /// 戻り値: 取得したフレーム数
    pub fn fetch_buffered_positions(&mut self, out: &mut [f32]) -> u32 {
        let count = self.buffered_frame_count;
        if count == 0 || self.num_vertices == 0 {
            return 0;
        }

        let total_size = (count as u64) * (self.num_vertices as u64) * 3 * 4;

        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Fetch Buffered Positions Encoder"),
            },
        );
        encoder.copy_buffer_to_buffer(
            &self.buffered_position_buffer,
            0,
            &self.buffered_staging_buffer,
            0,
            total_size,
        );
        self.context.queue.submit(Some(encoder.finish()));

        let slice = self.buffered_staging_buffer.slice(..total_size);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.context.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver.receive()).unwrap().unwrap();

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let copy_len = out.len().min(floats.len());
        out[..copy_len].copy_from_slice(&floats[..copy_len]);

        drop(data);
        self.buffered_staging_buffer.unmap();

        self.buffered_frame_count = 0;
        count
    }

    /// リングバッファの蓄積カウントをクリアする
    pub fn clear_frame_buffer(&mut self) {
        self.buffered_frame_count = 0;
    }

    /// 初期状態にリセットする
    pub fn reset(&mut self) {
        self.dynamic_pins.clear();
        self.upload_pins();
        self.context.queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(&self.initial_vertices),
        );
        self.reset_edge_rest_lengths();
        self.buffered_frame_count = 0;
        self.debug_recorder.clear();
    }

    // ==========================================
    // デバッグ状態記録 (Simulation Debug Recording)
    // ==========================================

    /// シミュレーション状態のデバッグ記録を開始する
    pub fn start_debug_recording(&mut self, object_name: &str, max_frames: Option<usize>) {
        let stiffness = self
            .distance_constraints
            .first()
            .map(|dc| if dc.tension_compliance > 1e-9 { 1.0 / dc.tension_compliance } else { 0.0 })
            .unwrap_or(0.0);
        let bending_stiffness = self
            .bending_constraints
            .first()
            .map(|bc| if bc.compliance > 1e-9 { 1.0 / bc.compliance } else { 0.0 })
            .unwrap_or(0.0);
        let thickness = self
            .initial_vertices
            .first()
            .map(|v| v.thickness)
            .unwrap_or(0.005);

        let metadata = SimulationMetadata {
            object_name: object_name.to_string(),
            num_vertices: self.num_vertices,
            num_edges: self.mesh_edges.len() as u32,
            num_faces: self.mesh_faces.len() as u32,
            edges: self.mesh_edges.clone(),
            faces: self.mesh_faces.clone(),
            stiffness,
            bending_stiffness,
            thickness,
            solver_mode: self.solver_mode,
            workgroup_size: self.workgroup_size,
        };

        self.debug_recorder.start_recording(metadata, max_frames);
    }

    /// 現在デバッグ記録中かどうか
    pub fn is_debug_recording(&self) -> bool {
        self.debug_recorder.is_recording()
    }

    /// 現在記録されているデバッグフレーム数
    pub fn get_debug_frame_count(&self) -> usize {
        self.debug_recorder.frame_count()
    }

    /// 現在のフレーム状態をデバッグ記録バッファに追記する
    pub fn record_current_frame(&mut self, dt: f32, substeps: u32) {
        if !self.debug_recorder.is_recording() || self.num_vertices == 0 {
            return;
        }

        let verts = self.read_vertices();
        let mut positions = Vec::with_capacity(verts.len());
        let mut velocities = Vec::with_capacity(verts.len());
        for v in &verts {
            positions.push(v.position);
            velocities.push(v.velocity);
        }

        let pinned: Vec<u32> = self.dynamic_pins.keys().copied().collect();
        self.debug_recorder.record_frame(
            dt,
            substeps,
            self.solver_iterations,
            &positions,
            &velocities,
            &pinned,
        );
    }

    /// 記録されたデバッグトレースをgzip圧縮ファイルとして保存する
    pub fn save_debug_recording(&self, file_path: &str) -> Result<String, String> {
        let path = std::path::Path::new(file_path);
        self.debug_recorder.save_to_file(path)
    }

    /// デバッグ記録を停止しメモリバッファを解放する
    pub fn stop_debug_recording(&mut self) {
        self.debug_recorder.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_triangle_collision() {
        let ctx = Arc::new(GpuContext::new().expect("GPU Context creation"));

        // 頂点 (0, 0, 0.5) を自由落下させる
        let positions = vec![[0.0, 0.0, 0.5]];
        let edges = vec![];

        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            None,
            None,
            None,
            None,
            None,
            0,
            0.02,
            10000.0,
            10000.0,
            5000.0,
            0.0,
            1.0,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);

        // z=0 に水平な三角形メッシュコライダーを配置
        let tri = GpuMeshTriangle {
            p0: [-1.0, -1.0, 0.0],
            friction: 0.2,
            p1: [1.0, -1.0, 0.0],
            thickness: 0.02,
            p2: [0.0, 1.0, 0.0],
            restitution: 0.0,
            flags: 0,
            _pad: [0.0; 3],
        };
        sim.set_mesh_triangles(&[tri]);

        for _ in 0..60 {
            sim.step(1.0 / 60.0, 20);
        }

        let verts = sim.read_vertices();
        let z = verts[0].position[2];
        println!("[Test Mesh Triangle Collision] Final Vertex Z: {}", z);

        // 三角形の厚み (0.02) + 布の厚み (0.02) = 0.04 以上で止まること
        assert!(z >= 0.04 - 1e-3, "頂点がメッシュ三角形を貫通してはならない (z={})", z);
    }

    #[test]
    fn test_mesh_triangle_single_sided_recovery() {
        let ctx = Arc::new(GpuContext::new().expect("GPU Context creation"));

        // 頂点を意図的に三角形の裏側（内側 z = -0.1m）にめり込んだ状態で初期化
        let positions = vec![[0.0, 0.0, -0.1]];
        let edges = vec![];

        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            None,
            None,
            None,
            None,
            None,
            0,
            0.02,
            10000.0,
            10000.0,
            5000.0,
            0.0,
            1.0,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);

        // z=0 に法線上向きの片面三角形コライダーを配置 (flags=1: single_sided)
        let tri = GpuMeshTriangle {
            p0: [-1.0, -1.0, 0.0],
            friction: 0.2,
            p1: [1.0, -1.0, 0.0],
            thickness: 0.02,
            p2: [0.0, 1.0, 0.0],
            restitution: 0.0,
            flags: 1, // is_single_sided = true
            _pad: [0.0; 3],
        };
        sim.set_mesh_triangles(&[tri]);

        // シミュレーションを実行（裏側から表側へ押し出されるか検証）
        for _ in 0..10 {
            sim.step(1.0 / 60.0, 20);
        }

        let verts = sim.read_vertices();
        let z = verts[0].position[2];
        println!("[Test Single Sided Recovery] Recovered Vertex Z: {}", z);

        // 片面判定により、裏側から表側 (z >= 0.04 - 1e-3) へ押し戻されていること
        assert!(
            z >= 0.04 - 1e-3,
            "裏側に侵入した頂点が片面リカバリーによって表側へ押し戻されなければならない (z={})",
            z
        );
    }

    #[test]
    fn test_edge_rest_length_scaling() {
        let ctx = Arc::new(GpuContext::new().expect("GPU Context creation"));

        // 2頂点 (0, 0, 0) と (1.0, 0, 0) を結ぶエッジ（初期長 1.0）
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let edges = vec![[0, 1]];
        // 頂点0を固定ピン (inv_mass=0.0)、頂点1を自由頂点 (inv_mass=1.0)
        let inv_masses = vec![0.0, 1.0];

        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            None,
            Some(&inv_masses),
            None,
            None,
            None,
            0,
            0.005,
            10000.0,
            10000.0,
            5000.0,
            0.0,
            1.0,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.gravity = [0.0, 0.0, 0.0]; // 無重力

        // エッジ0のスケールを 0.5 (目標長 0.5) に縮小
        sim.set_edge_rest_length_scales(&[0], &[0.5]);

        // シミュレーションを進める
        for _ in 0..30 {
            sim.step(1.0 / 60.0, 20);
        }

        let verts = sim.read_vertices();
        let p0 = verts[0].position;
        let p1 = verts[1].position;
        let dist = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2) + (p1[2] - p0[2]).powi(2)).sqrt();
        println!("[Test Edge Scaling] Initial: 1.0, Target: 0.5, Result: {}", dist);

        // 許容誤差 5% 以内で 0.5 に収縮していること
        assert!((dist - 0.5).abs() < 0.03, "エッジが目標長 0.5 に収縮していない: dist={}", dist);

        // リセットすると元の長さに戻ること
        sim.reset_edge_rest_lengths();
        assert_eq!(sim.get_edge_rest_length(0), 1.0, "エッジの自然長が 1.0 にリセットされること");
    }

    #[test]
    fn test_self_collision_options_and_untangling() {
        let ctx = Arc::new(GpuContext::new().expect("GPU Context creation"));

        // 4頂点のクアッド
        let positions = vec![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0],
        ];
        let edges = vec![
            [0, 1],
            [1, 2],
            [2, 0],
            [1, 3],
            [3, 2],
        ];
        let faces = vec![
            [0, 1, 2],
            [2, 1, 3],
        ];

        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            Some(&faces),
            None,
            None,
            None,
            None,
            0,
            0.02, // 厚み2cm
            1000.0,
            1000.0,
            1000.0,
            10.0,
            1.0,
        );

        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.gravity = [0.0, 0.0, 0.0];
        sim.set_enable_self_collision(true);

        // 新オプションの設定
        sim.set_self_collision_options(0.1, 0.2, true, true);
        assert_eq!(sim.self_collision_relief_factor, 0.1);
        assert_eq!(sim.self_collision_max_displacement_ratio, 0.2);
        assert!(sim.self_collision_exclude_neighbors);
        assert!(sim.enable_normal_untangling);

        // シミュレーションを実行してクラッシュやNaNが生じないこと
        for _ in 0..10 {
            sim.step(1.0 / 60.0, 10);
        }

        let verts = sim.read_vertices();
        for v in &verts {
            for c in v.position {
                assert!(!c.is_nan() && !c.is_infinite(), "頂点座標にNaNまたはInfが含まれてはいけない: {:?}", v.position);
            }
        }
    }
}
