use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;

use crate::context::GpuContext;
use crate::mesh::{
    ClothMesh, EdgeCollisionParams, GpuBendingConstraint, GpuCollider, GpuDistanceConstraint, GpuEdge,
    GpuMeshTriangle, GpuPinConstraint, GpuSewingConstraint, GpuStarPair, GpuVertex, SelfCollisionParams,
    SimParams, GpuVtPair, GpuEePair, PairCollectParams, PairCounters, PairSolveParams,
};
use crate::spatial_hash::{GpuSpatialHash, GpuEdgeSpatialHash, GpuColliderEdgeSpatialHash};
use super::pipeline_cache::{get_or_create_shared, SharedPipelines};
use super::types::{
    CollisionParams, DispatchInfo, GpuBoneInfo, GpuBoneTransform, NormalParams, PinParams,
};

pub struct SimulationResources {
    /// 共有パイプライン群 (プロセス全体でキャッシュ・メッシュ非依存)
    pub shared: Arc<SharedPipelines>,
    /// 今回の呼び出しで共有キャッシュを新規生成したか (計測用)
    pub shared_built_now: bool,
    /// per-sim 構築のフェーズ別所要時間 (名前, ミリ秒)
    pub build_timings: Vec<(String, f32)>,
    pub vertex_buffer: wgpu::Buffer,
    pub dist_buffer: wgpu::Buffer,
    pub bend_buffer: wgpu::Buffer,
    pub sew_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,

    pub staging_buffer: wgpu::Buffer,
    pub staging_buffers: [wgpu::Buffer; 2],
    pub accum_buffer: wgpu::Buffer,
    pub dist_lambda_buffer: wgpu::Buffer,
    pub bend_lambda_buffer: wgpu::Buffer,
    pub sew_lambda_buffer: wgpu::Buffer,
    pub distance_atomic_bind_group: wgpu::BindGroup,
    pub compact_position_buffer: wgpu::Buffer,
    pub compact_staging_buffers: [wgpu::Buffer; 2],
    pub extract_positions_bind_group: wgpu::BindGroup,
    pub buffered_position_buffer: wgpu::Buffer,
    pub buffered_staging_buffer: wgpu::Buffer,
    pub max_buffered_frames: usize,
    pub pin_buffer: wgpu::Buffer,
    pub pin_params_buffer: wgpu::Buffer,
    pub pin_bind_group: wgpu::BindGroup,
    pub collider_buffer: wgpu::Buffer,
    pub mesh_triangles_buffer: wgpu::Buffer,
    pub mesh_bounds_buffer: wgpu::Buffer,
    pub collider_group_bounds_buffer: wgpu::Buffer,
    pub collider_params_buffer: wgpu::Buffer,
    pub collider_bind_group: wgpu::BindGroup,
    pub bone_sdf_texture: wgpu::Texture,
    pub bone_sdf_texture_view: wgpu::TextureView,
    pub bone_sdf_sampler: wgpu::Sampler,
    pub bone_info_buffer: wgpu::Buffer,
    pub bone_transform_buffer: wgpu::Buffer,
    pub spatial_hash: GpuSpatialHash,
    pub edge_spatial_hash: GpuEdgeSpatialHash,
    pub self_collision_bind_group: wgpu::BindGroup,
    pub self_collision_bind_group_ee_off: wgpu::BindGroup,
    pub self_collision_ee_bind_group: wgpu::BindGroup,
    pub self_collision_params_buffer: wgpu::Buffer,
    pub self_collision_params_buffer_ee_off: wgpu::Buffer,
    pub self_collision_accum_buffer: wgpu::Buffer,
    pub self_collision_apply_bind_group: wgpu::BindGroup,
    pub num_edges: u32,
    pub edge_buffer: wgpu::Buffer,
    pub active_vt_pairs_buffer: wgpu::Buffer,
    pub active_ee_pairs_buffer: wgpu::Buffer,
    pub pair_counters_buffer: wgpu::Buffer,
    pub pair_collect_params_buffer: wgpu::Buffer,
    pub pair_solve_params_buffer: wgpu::Buffer,
    pub pair_collect_bind_group: wgpu::BindGroup,
    pub pair_solve_vt_bind_group: wgpu::BindGroup,
    pub pair_solve_ee_bind_group: wgpu::BindGroup,
    pub normals_buffer: wgpu::Buffer,
    pub local_edge_lengths_buffer: wgpu::Buffer,
    pub adj_offsets_buffer: wgpu::Buffer,
    pub adj_indices_buffer: wgpu::Buffer,
    pub two_hop_offsets_buffer: wgpu::Buffer,
    pub two_hop_indices_buffer: wgpu::Buffer,
    pub two_hop_rest_lengths_buffer: wgpu::Buffer,
    pub island_ids_buffer: wgpu::Buffer,
    pub compute_normals_bind_group: wgpu::BindGroup,
    pub predict_bind_group: wgpu::BindGroup,
    pub distance_bind_groups: Vec<wgpu::BindGroup>,
    pub bending_bind_groups: Vec<wgpu::BindGroup>,
    pub sewing_bind_groups: Vec<wgpu::BindGroup>,
    pub sew_shrink_bind_group: wgpu::BindGroup,
    pub update_vel_bind_group: wgpu::BindGroup,
    pub collider_edge_hash: GpuColliderEdgeSpatialHash,
    pub edge_collision_params_buffer: wgpu::Buffer,
    pub edge_collision_bind_group: wgpu::BindGroup,
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
    // 共有パイプライン群を取得 (初回のみ eager 8 本をコンパイル・約 2 秒)。
    // per-sim 側はバッファと BindGroup のみを構築する。
    let t_shared = Instant::now();
    let (shared, shared_built_now) = get_or_create_shared(context, workgroup_size);
    let shared_ms = t_shared.elapsed().as_secs_f32() * 1000.0;
    let t_buffers = Instant::now();
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
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    });

    // 動的ピンバッファ (範囲グラブ等の多頂点ピンに備えて余裕を持たせる。8192×32B = 256KB)
    let max_pins = 8192;
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
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
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

    // XPBDラグランジュ乗数バッファ (拘束毎f32、サブステップ先頭で零化)。
    // 空時はダミー1要素。clear_buffer 用に COPY_DST を付与する。
    let dist_lambda_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TareminCloth Distance Lambda Buffer"),
        size: ((num_distance_constraints as u64) * 4).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bend_lambda_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TareminCloth Bending Lambda Buffer"),
        size: ((num_bending_constraints as u64) * 4).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let sew_lambda_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TareminCloth Sewing Lambda Buffer"),
        size: ((num_sewing_constraints as u64) * 4).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let self_collision_accum_buffer_size = ((num_vertices as u64) * 32).max(64);
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

    // 2. シェーダーモジュール・BGL・パイプラインは共有キャッシュ由来 (SharedPipelines)。
    //    per-sim 側はバッファと BindGroup のみを構築する。
    let buffers_ms = t_buffers.elapsed().as_secs_f32() * 1000.0;
    let t_bindgroups = Instant::now();
    // 5. バインドグループ作成
    let predict_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Predict Bind Group"),
        layout: &shared.predict_bgl,
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
        layout: &shared.pin_bgl,
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let collider_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Collider Bind Group"),
        layout: &shared.collision_bgl,
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
        layout: &shared.update_vel_bgl,
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
        layout: &shared.distance_atomic_bgl,
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
            wgpu::BindGroupEntry {
                binding: 4,
                resource: dist_lambda_buffer.as_entire_binding(),
            },
        ],
    });

    let extract_positions_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Extract Positions Bind Group"),
        layout: &shared.extract_positions_bgl,
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
            layout: &shared.constraint_bgl,
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dist_lambda_buffer.as_entire_binding(),
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
            layout: &shared.constraint_bgl,
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bend_lambda_buffer.as_entire_binding(),
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
            layout: &shared.sewing_bgl,
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sew_lambda_buffer.as_entire_binding(),
                },
            ],
        }));
    }

    // 縫合自然長の時間進行パス用バインドグループ (サブステップ毎に1回)
    let sew_shrink_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Sew Shrink Bind Group"),
        layout: &shared.sew_shrink_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: sew_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // 自己衝突用リソース
    let self_collision_relief_factor = 0.2f32;
    let self_collision_max_displacement_ratio = 0.2f32;
    let self_collision_exclude_neighbors = true;
    let enable_normal_untangling = true;
    let self_collision_max_iterations = 256u32;

    let thickness = mesh.vertices.first().map(|v| v.thickness).unwrap_or(0.005);
    // 幾何学的見逃し防止のためのセルサイズ自動決定:
    // 1. 厚みベース: 衝突境界 (thickness * 4.0) をカバー
    // 2. メッシュ解像度ベース: エッジ端点接触時の中点間最大距離 (L + d) を 27 セル探索で確実に捕捉するため、
    //    エッジ幾何半径 (mean_edge_length * 0.5) を下限として保証する (粗いメッシュや極薄の布での抜け落ち防止)
    // 3. 最小値クランプ: 0.01 (1cm)
    let cell_size = (thickness * 4.0)
        .max(mesh.mean_edge_length * 0.5)
        .max(0.01);
    let table_size = (num_vertices * 4).next_power_of_two().max(1024);
    let spatial_hash = GpuSpatialHash::new(context, &shared.hash_bgl, &vertex_buffer, num_vertices, cell_size, table_size);

    let self_col_params = SelfCollisionParams {
        cell_size,
        table_size,
        num_vertices,
        num_edges: mesh.edges.len() as u32,
        relief_factor: self_collision_relief_factor,
        max_displacement_ratio: self_collision_max_displacement_ratio,
        enable_relief: 1,
        enable_normal_untangling: 1,
        exclude_neighbors: 1,
        max_search_iterations: self_collision_max_iterations,
        enable_ee: 1,
        _pad3: 0,
    };
    let self_collision_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Self Collision Params Buffer"),
        contents: bytemuck::bytes_of(&self_col_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let mut self_col_params_ee_off = self_col_params;
    self_col_params_ee_off.enable_ee = 0;
    let self_collision_params_buffer_ee_off = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Self Collision Params Buffer (EE Off)"),
        contents: bytemuck::bytes_of(&self_col_params_ee_off),
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

    let two_hop_off_contents: &[u8] = if mesh.two_hop_offsets.is_empty() {
        bytemuck::cast_slice(&dummy_u32)
    } else {
        bytemuck::cast_slice(&mesh.two_hop_offsets)
    };
    let two_hop_offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Two Hop Offsets Buffer"),
        contents: two_hop_off_contents,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let two_hop_idx_contents: &[u8] = if mesh.two_hop_indices.is_empty() {
        bytemuck::cast_slice(&dummy_u32)
    } else {
        bytemuck::cast_slice(&mesh.two_hop_indices)
    };
    let two_hop_indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Two Hop Indices Buffer"),
        contents: two_hop_idx_contents,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let dummy_f32 = [0.0f32];
    let two_hop_rest_contents: &[u8] = if mesh.two_hop_rest_lengths.is_empty() {
        bytemuck::cast_slice(&dummy_f32)
    } else {
        bytemuck::cast_slice(&mesh.two_hop_rest_lengths)
    };
    let two_hop_rest_lengths_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Two Hop Rest Lengths Buffer"),
        contents: two_hop_rest_contents,
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

    let dummy_gpu_edge = [GpuEdge { v0: 0, v1: 0 }];
    let edge_contents: &[u8] = if mesh.edges.is_empty() {
        bytemuck::cast_slice(&dummy_gpu_edge)
    } else {
        bytemuck::cast_slice(&mesh.edges)
    };
    let edge_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Edge Buffer"),
        contents: edge_contents,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let num_edges = mesh.edges.len() as u32;
    let edge_table_size = (num_edges * 4).next_power_of_two().max(1024);
    let edge_spatial_hash = GpuEdgeSpatialHash::new(
        context,
        &shared.edge_hash_bgl,
        &edge_buffer,
        &vertex_buffer,
        num_edges,
        cell_size,
        edge_table_size,
    );

    let self_collision_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Self Collision Bind Group"),
        layout: &shared.self_collision_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: vertex_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: spatial_hash.cell_starts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: spatial_hash.sorted_indices_buffer.as_entire_binding(),
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
            wgpu::BindGroupEntry {
                binding: 12,
                resource: two_hop_offsets_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: two_hop_indices_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: two_hop_rest_lengths_buffer.as_entire_binding(),
            },
        ],
    });

    let self_collision_bind_group_ee_off = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Self Collision Bind Group (EE Off)"),
        layout: &shared.self_collision_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: vertex_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: spatial_hash.cell_starts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: spatial_hash.sorted_indices_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: self_collision_params_buffer_ee_off.as_entire_binding(),
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
            wgpu::BindGroupEntry {
                binding: 12,
                resource: two_hop_offsets_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: two_hop_indices_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: two_hop_rest_lengths_buffer.as_entire_binding(),
            },
        ],
    });

    let self_collision_apply_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Self Collision Apply Bind Group"),
        layout: &shared.self_collision_apply_bgl,
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

    let self_collision_ee_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Self Collision EE Bind Group"),
        layout: &shared.self_collision_ee_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: edge_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vertex_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: edge_spatial_hash.cell_starts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: edge_spatial_hash.sorted_indices_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: self_collision_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: normals_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: two_hop_offsets_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: two_hop_indices_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: self_collision_accum_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: two_hop_rest_lengths_buffer.as_entire_binding(),
            },
        ],
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
        layout: &shared.compute_normals_bgl,
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

    // 接触候補ペアキャッシュ用バッファ & リソース (Active Pair Caching)
    // 頂点メジャー配置: 論理 N*quota をコンパクト配置、物理確保は N*QUOTA_MAX(8)。
    // 収集時に毎フレーム全論理範囲を書き直すため quota 変更時も整合する。
    use super::PAIR_QUOTA_MAX;
    let phys_slots = ((num_vertices * PAIR_QUOTA_MAX).max(64)) as usize;
    // 初期予算 (with_options の既定値と一致。初回 write_* で上書きされる)
    let max_vt_pairs = (num_vertices * 8).clamp(8192, 262144);
    let max_ee_pairs = (num_vertices * 8).clamp(8192, 262144);
    let vt_buffer_size = (phys_slots * std::mem::size_of::<GpuVtPair>()) as u64;
    let ee_buffer_size = (phys_slots * std::mem::size_of::<GpuEePair>()) as u64;

    let active_vt_pairs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Active VT Pairs Buffer"),
        size: vt_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let active_ee_pairs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Active EE Pairs Buffer"),
        size: ee_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let initial_counters = PairCounters {
        vt_count: 0,
        ee_count: 0,
        vt_dropped: 0,
        ee_dropped: 0,
    };
    let pair_counters_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Pair Counters Buffer"),
        contents: bytemuck::bytes_of(&initial_counters),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    });

    let pair_collect_params = PairCollectParams {
        cell_size,
        table_size,
        num_vertices,
        max_vt_pairs,
        max_ee_pairs,
        safety_margin: 0.005,
        exclude_neighbors: if self_collision_exclude_neighbors { 1 } else { 0 },
        margin_mode: 1,
        dt_frame: 1.0 / 60.0,
        velocity_horizon_scale: 1.3,
        max_horizon: 0.02,
        quota_vt: PAIR_QUOTA_MAX,
        quota_ee: PAIR_QUOTA_MAX,
    };
    let pair_collect_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Pair Collect Params Buffer"),
        contents: bytemuck::bytes_of(&pair_collect_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let pair_solve_params = PairSolveParams {
        num_vertices,
        max_vt_pairs: num_vertices * PAIR_QUOTA_MAX,
        max_ee_pairs: num_vertices * PAIR_QUOTA_MAX,
        enable_normal_untangling: if enable_normal_untangling { 1 } else { 0 },
    };
    let pair_solve_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Pair Solve Params Buffer"),
        contents: bytemuck::bytes_of(&pair_solve_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let pair_collect_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Pair Collect Bind Group"),
        layout: &shared.pair_collect_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: vertex_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: spatial_hash.cell_starts_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: spatial_hash.sorted_indices_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: pair_collect_params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: local_edge_lengths_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: adj_offsets_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: adj_indices_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: island_ids_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: star_offsets_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: star_indices_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: two_hop_offsets_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 11, resource: two_hop_indices_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 12, resource: pair_counters_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 13, resource: active_vt_pairs_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 14, resource: active_ee_pairs_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 15, resource: two_hop_rest_lengths_buffer.as_entire_binding() },
        ],
    });



    let pair_solve_vt_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Pair Solve VT Bind Group"),
        layout: &shared.pair_solve_vt_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: vertex_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: active_vt_pairs_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pair_counters_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: self_collision_accum_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pair_solve_params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: normals_buffer.as_entire_binding() },
        ],
    });

    let pair_solve_ee_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Pair Solve EE Bind Group"),
        layout: &shared.pair_solve_ee_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: vertex_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: active_ee_pairs_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: pair_counters_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: self_collision_accum_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: pair_solve_params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: normals_buffer.as_entire_binding() },
        ],
    });

    // コライダーエッジ空間ハッシュおよびエッジコライダー拘束バインドグループ
    let collider_edge_hash = GpuColliderEdgeSpatialHash::new_empty(context, cell_size, crate::spatial_hash::DEFAULT_HASH_TABLE_SIZE);

    let edge_collision_params = EdgeCollisionParams {
        cell_size,
        table_size: crate::spatial_hash::DEFAULT_HASH_TABLE_SIZE,
        num_cloth_edges: mesh.edges.len() as u32,
        num_collider_edges: 0,
        edge_margin_scale: 1.0,
        edge_margin_offset: 0.0,
        _pad0: 0,
        _pad1: 0,
    };
    let edge_collision_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Edge Collision Params Buffer"),
        contents: bytemuck::bytes_of(&edge_collision_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let edge_collision_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Edge Collision Bind Group"),
        layout: &shared.edge_collision_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: edge_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vertex_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: collider_edge_hash.collider_edges_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: collider_edge_hash.cell_starts_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: collider_edge_hash.sorted_indices_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: edge_collision_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: self_collision_accum_buffer.as_entire_binding(),
            },
        ],
    });

    let bind_groups_ms = t_bindgroups.elapsed().as_secs_f32() * 1000.0;
    let build_timings = vec![
        ("shared_cache_ms".to_string(), shared_ms),
        ("buffers_ms".to_string(), buffers_ms),
        ("bind_groups_ms".to_string(), bind_groups_ms),
    ];

    SimulationResources {
        shared,
        shared_built_now,
        build_timings,
        vertex_buffer,
        dist_buffer,
        bend_buffer,
        sew_buffer,
        params_buffer,
        staging_buffer,
        staging_buffers: [staging_buffer_0, staging_buffer_1],
        accum_buffer,
        dist_lambda_buffer,
        bend_lambda_buffer,
        sew_lambda_buffer,
        distance_atomic_bind_group,
        compact_position_buffer,
        compact_staging_buffers: [compact_staging_0, compact_staging_1],
        extract_positions_bind_group,
        buffered_position_buffer,
        buffered_staging_buffer,
        max_buffered_frames,
        pin_buffer,
        pin_params_buffer,
        pin_bind_group,
        collider_buffer,
        mesh_triangles_buffer,
        mesh_bounds_buffer,
        collider_group_bounds_buffer,
        collider_params_buffer,
        collider_bind_group,
        bone_sdf_texture,
        bone_sdf_texture_view,
        bone_sdf_sampler,
        bone_info_buffer,
        bone_transform_buffer,
        spatial_hash,
        edge_spatial_hash,
        self_collision_bind_group,
        self_collision_bind_group_ee_off,
        self_collision_ee_bind_group,
        self_collision_params_buffer,
        self_collision_params_buffer_ee_off,
        self_collision_accum_buffer,
        self_collision_apply_bind_group,
        num_edges: mesh.edges.len() as u32,
        edge_buffer,
        active_vt_pairs_buffer,
        active_ee_pairs_buffer,
        pair_counters_buffer,
        pair_collect_params_buffer,
        pair_solve_params_buffer,
        pair_collect_bind_group,
        pair_solve_vt_bind_group,
        pair_solve_ee_bind_group,
        normals_buffer,

        local_edge_lengths_buffer,
        adj_offsets_buffer,
        adj_indices_buffer,
        two_hop_offsets_buffer,
        two_hop_indices_buffer,
        two_hop_rest_lengths_buffer,
        island_ids_buffer,
        compute_normals_bind_group,
        predict_bind_group,
        distance_bind_groups,
        bending_bind_groups,
        sewing_bind_groups,
        sew_shrink_bind_group,
        update_vel_bind_group,
        collider_edge_hash,
        edge_collision_params_buffer,
        edge_collision_bind_group,
        self_collision_relief_factor,
        self_collision_max_displacement_ratio,
        self_collision_exclude_neighbors,
        self_collision_max_iterations,
        enable_normal_untangling,
    }
}
