use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::context::GpuContext;
use crate::mesh::{
    ClothMesh, GpuBendingConstraint, GpuCollider, GpuDistanceConstraint, GpuMeshTriangle,
    GpuPinConstraint, GpuSewingConstraint, GpuStarPair, GpuVertex, SelfCollisionParams, SimParams,
};
use crate::spatial_hash::GpuSpatialHash;
use super::types::{
    CollisionParams, DispatchInfo, GpuBoneInfo, GpuBoneTransform, NormalParams, PinParams,
};

pub fn create_shader_with_wg_size(device: &wgpu::Device, label: &str, src: &str, wg_size: u32) -> wgpu::ShaderModule {
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

pub struct SimulationResources {
    pub vertex_buffer: wgpu::Buffer,
    pub dist_buffer: wgpu::Buffer,
    pub bend_buffer: wgpu::Buffer,
    pub sew_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,

    pub staging_buffer: wgpu::Buffer,
    pub staging_buffers: [wgpu::Buffer; 2],
    pub accum_buffer: wgpu::Buffer,
    pub distance_atomic_solve_pipeline: wgpu::ComputePipeline,
    pub distance_atomic_apply_pipeline: wgpu::ComputePipeline,
    pub distance_atomic_bind_group: wgpu::BindGroup,
    pub compact_position_buffer: wgpu::Buffer,
    pub compact_staging_buffers: [wgpu::Buffer; 2],
    pub extract_positions_pipeline: wgpu::ComputePipeline,
    pub extract_positions_bind_group: wgpu::BindGroup,
    pub buffered_position_buffer: wgpu::Buffer,
    pub buffered_staging_buffer: wgpu::Buffer,
    pub max_buffered_frames: usize,
    pub pin_buffer: wgpu::Buffer,
    pub pin_params_buffer: wgpu::Buffer,
    pub pin_bind_group: wgpu::BindGroup,
    pub pin_pipeline: wgpu::ComputePipeline,
    pub collider_buffer: wgpu::Buffer,
    pub mesh_triangles_buffer: wgpu::Buffer,
    pub mesh_bounds_buffer: wgpu::Buffer,
    pub collider_group_bounds_buffer: wgpu::Buffer,
    pub collider_params_buffer: wgpu::Buffer,
    pub collider_bind_group: wgpu::BindGroup,
    pub collision_pipeline: wgpu::ComputePipeline,
    pub collision_bgl: wgpu::BindGroupLayout,
    pub bone_sdf_texture: wgpu::Texture,
    pub bone_sdf_texture_view: wgpu::TextureView,
    pub bone_sdf_sampler: wgpu::Sampler,
    pub bone_info_buffer: wgpu::Buffer,
    pub bone_transform_buffer: wgpu::Buffer,
    pub spatial_hash: GpuSpatialHash,
    pub self_collision_pipeline: wgpu::ComputePipeline,
    pub self_collision_bind_group: wgpu::BindGroup,
    pub self_collision_params_buffer: wgpu::Buffer,
    pub self_collision_accum_buffer: wgpu::Buffer,
    pub self_collision_apply_pipeline: wgpu::ComputePipeline,
    pub self_collision_apply_bind_group: wgpu::BindGroup,
    pub normals_buffer: wgpu::Buffer,
    pub local_edge_lengths_buffer: wgpu::Buffer,
    pub adj_offsets_buffer: wgpu::Buffer,
    pub adj_indices_buffer: wgpu::Buffer,
    pub island_ids_buffer: wgpu::Buffer,
    pub compute_normals_pipeline: wgpu::ComputePipeline,
    pub compute_normals_bind_group: wgpu::BindGroup,
    pub predict_pipeline: wgpu::ComputePipeline,
    pub distance_pipeline: wgpu::ComputePipeline,
    pub bending_pipeline: wgpu::ComputePipeline,
    pub sewing_pipeline: wgpu::ComputePipeline,
    pub update_vel_pipeline: wgpu::ComputePipeline,
    pub predict_bind_group: wgpu::BindGroup,
    pub distance_bind_groups: Vec<wgpu::BindGroup>,
    pub bending_bind_groups: Vec<wgpu::BindGroup>,
    pub sewing_bind_groups: Vec<wgpu::BindGroup>,
    pub update_vel_bind_group: wgpu::BindGroup,
    pub edge_collision_pipeline: wgpu::ComputePipeline,
    pub edge_collision_bind_groups: Vec<wgpu::BindGroup>,
    pub self_collision_relief_factor: f32,
    pub self_collision_max_displacement_ratio: f32,
    pub self_collision_exclude_neighbors: bool,
    pub self_collision_max_iterations: u32,
    pub enable_normal_untangling: bool,
}

pub fn build_simulation_resources(
    context: &Arc<GpuContext>,
    mesh: &mut ClothMesh,
    workgroup_size: u32,
) -> SimulationResources {
    let device = &context.device;
    let num_vertices = mesh.vertices.len() as u32;
    let num_distance_constraints = mesh.distance_constraints.len() as u32;
    let num_bending_constraints = mesh.bending_constraints.len() as u32;
    let num_sewing_constraints = mesh.sewing_constraints.len() as u32;

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

    // 1. GPU バッファの作成
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("TareminCloth Vertex Buffer"),
        contents: bytemuck::cast_slice(&mesh.vertices),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::VERTEX,
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
        lock_on_close: 0.0,
        _pad1: 0.0,
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

    // 16面グループごとの階層境界球バッファ (vec4<f32>: xyz=中心, w=半径)
    let max_groups = ((max_mesh_triangles + 15) / 16).max(1);
    let collider_group_bounds_size = (max_groups * std::mem::size_of::<[f32; 4]>()) as u64;
    let collider_group_bounds_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TareminCloth Collider Group Bounds Buffer"),
        size: collider_group_bounds_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let collider_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("TareminCloth Collider Params Buffer"),
        contents: bytemuck::bytes_of(&CollisionParams {
            num_vertices,
            num_colliders: 0,
            num_mesh_triangles: 0,
            num_clusters: 0,
            dt: 0.016 / 20.0,
            edge_margin_scale: 1.0,
            edge_margin_offset: 0.0,
            enable_cluster_culling: 0,
            enable_single_sided_recovery: 1,
            sweep_margin_offset: 0.05,
            num_bones: 0,
            enable_bone_sdf: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // ボーンSDF用バッファ & 初期ダミー3Dテクスチャ
    let max_bones = 1024;
    let bone_info_buffer_size = (max_bones * std::mem::size_of::<GpuBoneInfo>()) as u64;
    let bone_info_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TareminCloth Bone Info Buffer"),
        size: bone_info_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bone_transform_buffer_size = (max_bones * std::mem::size_of::<GpuBoneTransform>()) as u64;
    let bone_transform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TareminCloth Bone Transform Buffer"),
        size: bone_transform_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bone_sdf_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("TareminCloth Dummy Bone SDF Texture"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rg16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let bone_sdf_texture_view = bone_sdf_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bone_sdf_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("TareminCloth Bone SDF Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let default_params = SimParams {
        gravity: [0.0, 0.0, -9.81, 0.016 / 20.0],
        damping: 0.05,
        substeps: 20,
        num_vertices,
        num_distance_constraints,
        num_bending_constraints,
        num_sewing_constraints,
        sewing_compliance: 0.0,
        enable_sewing_lock: 1.0,
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

    let self_collision_accum_buffer_size = ((num_vertices as u64) * 16).max(64);
    let self_collision_accum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TareminCloth Self Collision Accum Buffer"),
        size: self_collision_accum_buffer_size,
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
        include_str!("../shaders/predict.wgsl"),
        workgroup_size,
    );

    let distance_shader = create_shader_with_wg_size(
        device,
        "Distance Shader",
        include_str!("../shaders/distance.wgsl"),
        workgroup_size,
    );

    let bending_shader = create_shader_with_wg_size(
        device,
        "Bending Shader",
        include_str!("../shaders/bending.wgsl"),
        workgroup_size,
    );

    let sewing_shader = create_shader_with_wg_size(
        device,
        "Sewing Shader",
        include_str!("../shaders/sewing.wgsl"),
        workgroup_size,
    );

    let collision_shader = create_shader_with_wg_size(
        device,
        "Collision Shader",
        include_str!("../shaders/collision.wgsl"),
        workgroup_size,
    );

    let self_collision_shader = create_shader_with_wg_size(
        device,
        "Self Collision Shader",
        include_str!("../shaders/self_collision.wgsl"),
        workgroup_size,
    );

    let self_collision_apply_shader = create_shader_with_wg_size(
        device,
        "Self Collision Apply Shader",
        include_str!("../shaders/self_collision_apply.wgsl"),
        workgroup_size,
    );

    let edge_collision_shader = create_shader_with_wg_size(
        device,
        "Edge Collision Shader",
        include_str!("../shaders/edge_collision.wgsl"),
        workgroup_size,
    );

    let compute_normals_shader = create_shader_with_wg_size(
        device,
        "Compute Normals Shader",
        include_str!("../shaders/compute_normals.wgsl"),
        workgroup_size,
    );

    let pin_shader = create_shader_with_wg_size(
        device,
        "Pin Shader",
        include_str!("../shaders/pin.wgsl"),
        workgroup_size,
    );

    let update_vel_shader = create_shader_with_wg_size(
        device,
        "Update Velocity Shader",
        include_str!("../shaders/update_vel.wgsl"),
        workgroup_size,
    );

    let distance_atomic_shader = create_shader_with_wg_size(
        device,
        "Distance Atomic Shader",
        include_str!("../shaders/distance_atomic.wgsl"),
        workgroup_size,
    );

    let extract_positions_shader = create_shader_with_wg_size(
        device,
        "Extract Positions Shader",
        include_str!("../shaders/extract_positions.wgsl"),
        workgroup_size,
    );

    // 3. バインドグループレイアウト
    let storage_rw = wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let storage_rw_at = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };

    let predict_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Predict Bind Group Layout"),
        entries: &[storage_rw, uniform_entry(1)],
    });

    let constraint_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Constraint Bind Group Layout"),
        entries: &[storage_rw, storage_ro(1), uniform_entry(2), uniform_entry(3)],
    });

    let sewing_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Sewing Bind Group Layout"),
        entries: &[storage_rw, storage_rw_at(1), uniform_entry(2), uniform_entry(3)],
    });

    let pin_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Pin Bind Group Layout"),
        entries: &[storage_rw, storage_ro(1), uniform_entry(2)],
    });

    let collision_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Collision Bind Group Layout"),
        entries: &[
            storage_rw,
            storage_ro(1),
            storage_ro(2),
            uniform_entry(3),
            storage_ro(4),
            storage_ro(5),
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            storage_ro(8),
            storage_ro(9),
        ],
    });

    let edge_collision_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Edge Collision Bind Group Layout"),
        entries: &[storage_rw, storage_ro(1), storage_ro(2), storage_ro(3), storage_ro(4), uniform_entry(5), uniform_entry(6)],
    });

    let update_vel_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Update Vel Bind Group Layout"),
        entries: &[storage_rw, uniform_entry(1)],
    });

    let distance_atomic_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Distance Atomic Bind Group Layout"),
        entries: &[
            storage_rw,
            storage_ro(1),
            uniform_entry(2),
            storage_rw_at(3),
        ],
    });

    let extract_positions_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Extract Positions Bind Group Layout"),
        entries: &[
            storage_ro(0),
            storage_rw_at(1),
        ],
    });

    // 4. パイプライン作成
    let predict_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Predict Pipeline Layout"),
        bind_group_layouts: &[&predict_bgl],
        push_constant_ranges: &[],
    });
    let predict_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Predict Pipeline"),
        layout: Some(&predict_pl),
        module: &predict_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let distance_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Distance Pipeline Layout"),
        bind_group_layouts: &[&constraint_bgl],
        push_constant_ranges: &[],
    });
    let distance_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Distance Pipeline"),
        layout: Some(&distance_pl),
        module: &distance_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bending_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Bending Pipeline Layout"),
        bind_group_layouts: &[&constraint_bgl],
        push_constant_ranges: &[],
    });
    let bending_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Bending Pipeline"),
        layout: Some(&bending_pl),
        module: &bending_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let sewing_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Sewing Pipeline Layout"),
        bind_group_layouts: &[&sewing_bgl],
        push_constant_ranges: &[],
    });
    let sewing_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Sewing Pipeline"),
        layout: Some(&sewing_pl),
        module: &sewing_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let pin_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pin Pipeline Layout"),
        bind_group_layouts: &[&pin_bgl],
        push_constant_ranges: &[],
    });
    let pin_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Pin Pipeline"),
        layout: Some(&pin_pl),
        module: &pin_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let collision_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Collision Pipeline Layout"),
        bind_group_layouts: &[&collision_bgl],
        push_constant_ranges: &[],
    });
    let collision_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Collision Pipeline"),
        layout: Some(&collision_pl),
        module: &collision_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let edge_collision_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Edge Collision Pipeline Layout"),
        bind_group_layouts: &[&edge_collision_bgl],
        push_constant_ranges: &[],
    });
    let edge_collision_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Edge Collision Pipeline"),
        layout: Some(&edge_collision_pl),
        module: &edge_collision_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let update_vel_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Update Vel Pipeline Layout"),
        bind_group_layouts: &[&update_vel_bgl],
        push_constant_ranges: &[],
    });
    let update_vel_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Update Vel Pipeline"),
        layout: Some(&update_vel_pl),
        module: &update_vel_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let distance_atomic_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Distance Atomic Pipeline Layout"),
        bind_group_layouts: &[&distance_atomic_bgl],
        push_constant_ranges: &[],
    });
    let distance_atomic_solve_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Distance Atomic Solve Pipeline"),
        layout: Some(&distance_atomic_pl),
        module: &distance_atomic_shader,
        entry_point: Some("solve_distance_atomic"),
        compilation_options: Default::default(),
        cache: None,
    });
    let distance_atomic_apply_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Distance Atomic Apply Pipeline"),
        layout: Some(&distance_atomic_pl),
        module: &distance_atomic_shader,
        entry_point: Some("apply_atomic_accum"),
        compilation_options: Default::default(),
        cache: None,
    });

    let extract_positions_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Extract Positions Pipeline Layout"),
        bind_group_layouts: &[&extract_positions_bgl],
        push_constant_ranges: &[],
    });
    let extract_positions_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Extract Positions Pipeline"),
        layout: Some(&extract_positions_pl),
        module: &extract_positions_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // 5. バインドグループ作成
    let predict_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Predict Bind Group"),
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

    let pin_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Pin Bind Group"),
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

    let collider_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Collider Bind Group"),
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
            wgpu::BindGroupEntry {
                binding: 5,
                resource: collider_group_bounds_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&bone_sdf_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::Sampler(&bone_sdf_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: bone_info_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: bone_transform_buffer.as_entire_binding(),
            },
        ],
    });

    let update_vel_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Update Vel Bind Group"),
        layout: &update_vel_bgl,
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

    let distance_atomic_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Distance Atomic Bind Group"),
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

    let extract_positions_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Extract Positions Bind Group"),
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

    // 距離拘束バインドグループ (彩色グループごと)
    let mut distance_bind_groups = Vec::new();
    for (offset, &count) in mesh.dist_color_offsets.iter().zip(&mesh.dist_color_counts) {
        let info = DispatchInfo {
            color_offset: *offset,
            color_count: count,
            _pad0: 0,
            _pad1: 0,
        };
        let info_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Distance DispatchInfo Buffer"),
            contents: bytemuck::bytes_of(&info),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        distance_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Distance Bind Group"),
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
                    resource: info_buf.as_entire_binding(),
                },
            ],
        }));
    }

    // エッジコライダー拘束バインドグループ
    let mut edge_collision_bind_groups = Vec::new();
    for (offset, &count) in mesh.dist_color_offsets.iter().zip(&mesh.dist_color_counts) {
        let info = DispatchInfo {
            color_offset: *offset,
            color_count: count,
            _pad0: 0,
            _pad1: 0,
        };
        let info_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Edge Collision DispatchInfo Buffer"),
            contents: bytemuck::bytes_of(&info),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        edge_collision_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Edge Collision Bind Group"),
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
                    resource: info_buf.as_entire_binding(),
                },
            ],
        }));
    }

    // 曲げ拘束バインドグループ (彩色グループごと)
    let mut bending_bind_groups = Vec::new();
    for (offset, &count) in mesh.bend_color_offsets.iter().zip(&mesh.bend_color_counts) {
        let info = DispatchInfo {
            color_offset: *offset,
            color_count: count,
            _pad0: 0,
            _pad1: 0,
        };
        let info_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bending DispatchInfo Buffer"),
            contents: bytemuck::bytes_of(&info),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        bending_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bending Bind Group"),
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
                    resource: info_buf.as_entire_binding(),
                },
            ],
        }));
    }

    // 縫合拘束バインドグループ (彩色グループごと)
    let mut sewing_bind_groups = Vec::new();
    for (offset, &count) in mesh.sew_color_offsets.iter().zip(&mesh.sew_color_counts) {
        let info = DispatchInfo {
            color_offset: *offset,
            color_count: count,
            _pad0: 0,
            _pad1: 0,
        };
        let info_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sewing DispatchInfo Buffer"),
            contents: bytemuck::bytes_of(&info),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        sewing_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sewing Bind Group"),
            layout: &sewing_bgl,
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
                    resource: info_buf.as_entire_binding(),
                },
            ],
        }));
    }

    // 自己衝突用リソース
    let self_collision_relief_factor = 0.2f32;
    let self_collision_max_displacement_ratio = 0.2f32;
    let self_collision_exclude_neighbors = true;
    let enable_normal_untangling = true;
    let self_collision_max_iterations = 256u32;

    let thickness = mesh.vertices.first().map(|v| v.thickness).unwrap_or(0.005);
    let cell_size = (thickness * 4.0).max(0.01);
    let table_size = (num_vertices * 4).next_power_of_two().max(1024);
    let spatial_hash = GpuSpatialHash::new(context, &vertex_buffer, num_vertices, cell_size, table_size);

    let self_col_params = SelfCollisionParams {
        cell_size,
        table_size,
        num_vertices,
        _pad0: 0,
        relief_factor: self_collision_relief_factor,
        max_displacement_ratio: self_collision_max_displacement_ratio,
        enable_relief: 1,
        enable_normal_untangling: 1,
        exclude_neighbors: 1,
        max_search_iterations: self_collision_max_iterations,
        _pad2: 0,
        _pad3: 0,
    };
    let self_collision_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Self Collision Params Buffer"),
        contents: bytemuck::bytes_of(&self_col_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let normals_buffer_size = ((num_vertices as u64) * 16).max(64);
    let normals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Normals Buffer"),
        size: normals_buffer_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let dummy_f32 = [0.0f32];
    let edge_lens_contents: &[u8] = if mesh.local_edge_lengths.is_empty() {
        bytemuck::cast_slice(&dummy_f32)
    } else {
        bytemuck::cast_slice(&mesh.local_edge_lengths)
    };
    let local_edge_lengths_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Local Edge Lengths Buffer"),
        contents: edge_lens_contents,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let dummy_u32 = [0u32];
    let adj_off_contents: &[u8] = if mesh.adj_offsets.is_empty() {
        bytemuck::cast_slice(&dummy_u32)
    } else {
        bytemuck::cast_slice(&mesh.adj_offsets)
    };
    let adj_offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Adj Offsets Buffer"),
        contents: adj_off_contents,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let adj_ind_contents: &[u8] = if mesh.adj_indices.is_empty() {
        bytemuck::cast_slice(&dummy_u32)
    } else {
        bytemuck::cast_slice(&mesh.adj_indices)
    };
    let adj_indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Adj Indices Buffer"),
        contents: adj_ind_contents,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let island_contents: &[u8] = if mesh.island_ids.is_empty() {
        bytemuck::cast_slice(&dummy_u32)
    } else {
        bytemuck::cast_slice(&mesh.island_ids)
    };
    let island_ids_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Island IDs Buffer"),
        contents: island_contents,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let dummy_star_pair = [GpuStarPair { v0: 0, v1: 0 }];
    let star_indices_contents: &[u8] = if mesh.star_indices.is_empty() {
        bytemuck::cast_slice(&dummy_star_pair)
    } else {
        bytemuck::cast_slice(&mesh.star_indices)
    };
    let star_indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Star Indices Buffer"),
        contents: star_indices_contents,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let star_offsets_contents: &[u8] = if mesh.star_offsets.is_empty() {
        bytemuck::cast_slice(&dummy_u32)
    } else {
        bytemuck::cast_slice(&mesh.star_offsets)
    };
    let star_offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Star Offsets Buffer"),
        contents: star_offsets_contents,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let self_collision_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Self Collision Bind Group Layout"),
        entries: &[
            storage_rw,
            storage_ro(1),
            storage_ro(2),
            uniform_entry(3),
            storage_ro(4),
            storage_ro(5),
            storage_ro(6),
            storage_ro(7),
            storage_ro(8),
            storage_ro(9),
            storage_ro(10),
            storage_rw_at(11),
        ],
    });

    let self_collision_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Self Collision Pipeline Layout"),
        bind_group_layouts: &[&self_collision_bgl],
        push_constant_ranges: &[],
    });

    let self_collision_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Self Collision Pipeline"),
        layout: Some(&self_collision_pl),
        module: &self_collision_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let self_collision_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Self Collision Bind Group"),
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
            wgpu::BindGroupEntry {
                binding: 9,
                resource: star_offsets_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: star_indices_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: self_collision_accum_buffer.as_entire_binding(),
            },
        ],
    });

    let self_collision_apply_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Self Collision Apply Bind Group Layout"),
        entries: &[
            storage_rw,
            storage_rw_at(1),
            uniform_entry(2),
            storage_ro(3),
        ],
    });

    let self_collision_apply_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Self Collision Apply Pipeline Layout"),
        bind_group_layouts: &[&self_collision_apply_bgl],
        push_constant_ranges: &[],
    });

    let self_collision_apply_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Self Collision Apply Pipeline"),
        layout: Some(&self_collision_apply_pl),
        module: &self_collision_apply_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let self_collision_apply_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Self Collision Apply Bind Group"),
        layout: &self_collision_apply_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: vertex_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: self_collision_accum_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: self_collision_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: local_edge_lengths_buffer.as_entire_binding(),
            },
        ],
    });

    let compute_normals_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Normals Bind Group Layout"),
        entries: &[
            storage_ro(0),
            storage_ro(1),
            storage_ro(2),
            storage_rw_at(3),
            uniform_entry(4),
        ],
    });

    let compute_normals_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Normals Pipeline Layout"),
        bind_group_layouts: &[&compute_normals_bgl],
        push_constant_ranges: &[],
    });

    let compute_normals_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Normals Pipeline"),
        layout: Some(&compute_normals_pl),
        module: &compute_normals_shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let normal_params = NormalParams {
        num_vertices,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let normal_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Normal Params Buffer"),
        contents: bytemuck::bytes_of(&normal_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let compute_normals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Compute Normals Bind Group"),
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

    SimulationResources {
        vertex_buffer,
        dist_buffer,
        bend_buffer,
        sew_buffer,
        params_buffer,
        staging_buffer,
        staging_buffers: [staging_buffer_0, staging_buffer_1],
        accum_buffer,
        distance_atomic_solve_pipeline,
        distance_atomic_apply_pipeline,
        distance_atomic_bind_group,
        compact_position_buffer,
        compact_staging_buffers: [compact_staging_0, compact_staging_1],
        extract_positions_pipeline,
        extract_positions_bind_group,
        buffered_position_buffer,
        buffered_staging_buffer,
        max_buffered_frames,
        pin_buffer,
        pin_params_buffer,
        pin_bind_group,
        pin_pipeline,
        collider_buffer,
        mesh_triangles_buffer,
        mesh_bounds_buffer,
        collider_group_bounds_buffer,
        collider_params_buffer,
        collider_bind_group,
        collision_pipeline,
        collision_bgl,
        bone_sdf_texture,
        bone_sdf_texture_view,
        bone_sdf_sampler,
        bone_info_buffer,
        bone_transform_buffer,
        spatial_hash,
        self_collision_pipeline,
        self_collision_bind_group,
        self_collision_params_buffer,
        self_collision_accum_buffer,
        self_collision_apply_pipeline,
        self_collision_apply_bind_group,
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
        edge_collision_pipeline,
        edge_collision_bind_groups,
        self_collision_relief_factor,
        self_collision_max_displacement_ratio,
        self_collision_exclude_neighbors,
        self_collision_max_iterations,
        enable_normal_untangling,
    }
}
