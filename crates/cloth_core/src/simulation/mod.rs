pub mod types;
pub mod pipelines;
pub mod dispatch;
pub mod recording;
#[cfg(test)]
pub mod tests;

use std::collections::HashMap;
use std::sync::Arc;

use crate::context::GpuContext;
use crate::debug_recorder::SimulationDebugRecorder;
use crate::mesh::{
    ClothMesh, GpuBendingConstraint, GpuCollider, GpuDistanceConstraint, GpuMeshTriangle,
    GpuPinConstraint, GpuSewingConstraint, GpuVertex, SelfCollisionParams,
};
use crate::spatial_hash::GpuSpatialHash;
use crate::sdf_baker::{GpuBakeParams, GpuBakeTriangle};
use self::types::{CollisionParams, GpuBoneInfo, GpuBoneTransform, GpuBoneTriangleSource, GpuSkinningVertex, PinParams};
use self::pipelines::build_simulation_resources;

/// 動的ボーンSDF（GPU LBS + インメモリSDF更新）の初期設定データ
#[derive(Clone, Debug)]
pub struct DynamicBoneSdfSetup {
    pub rest_verts: Vec<GpuSkinningVertex>,
    pub tri_sources: Vec<GpuBoneTriangleSource>,
    pub bone_bind_inv_matrices: Vec<[[f32; 4]; 4]>,
    pub bake_params: Vec<GpuBakeParams>,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub res: u32,
    pub bone_infos: Vec<GpuBoneInfo>,
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

    pub(crate) initial_vertices: Vec<GpuVertex>,
    pub(crate) distance_constraints: Vec<GpuDistanceConstraint>,
    pub original_edge_to_constraint: Vec<usize>,
    pub initial_distance_rest_lengths: Vec<f32>,
    pub(crate) bending_constraints: Vec<GpuBendingConstraint>,
    pub(crate) sewing_constraints: Vec<GpuSewingConstraint>,

    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) dist_buffer: wgpu::Buffer,
    pub(crate) bend_buffer: wgpu::Buffer,
    pub(crate) params_buffer: wgpu::Buffer,
    pub(crate) staging_buffer: wgpu::Buffer,
    // 非同期ダブルバッファリング用
    pub(crate) staging_buffers: [wgpu::Buffer; 2],
    pub(crate) staging_idx: usize,
    pub(crate) async_pending: bool,
    pub(crate) async_receiver: Option<futures_intrusive::channel::shared::OneshotReceiver<Result<(), wgpu::BufferAsyncError>>>,

    // チューニングオプション
    pub workgroup_size: u32,
    pub solver_mode: u32, // 0: Coloring (Gauss-Seidel), 1: Atomic (Jacobi)
    #[allow(dead_code)]
    pub(crate) accum_buffer: wgpu::Buffer,
    pub(crate) distance_atomic_solve_pipeline: wgpu::ComputePipeline,
    pub(crate) distance_atomic_apply_pipeline: wgpu::ComputePipeline,
    pub(crate) distance_atomic_bind_group: wgpu::BindGroup,

    // コンパクトリードバック用 (頂点座標 12B/頂点 のみ抽出・転送)
    pub enable_compact_readback: bool,
    pub(crate) compact_position_buffer: wgpu::Buffer,
    pub(crate) compact_staging_buffers: [wgpu::Buffer; 2],
    pub(crate) extract_positions_pipeline: wgpu::ComputePipeline,
    pub(crate) extract_positions_bind_group: wgpu::BindGroup,

    // フレームバッファリング用 (直近NフレームをGPU上で保持し一括転送)
    pub(crate) buffered_position_buffer: wgpu::Buffer,
    pub(crate) buffered_staging_buffer: wgpu::Buffer,
    pub buffered_frame_count: u32,
    pub max_buffered_frames: usize,

    // 動的ピン用
    pub(crate) dynamic_pins: HashMap<u32, GpuPinConstraint>,
    pub(crate) pin_buffer: wgpu::Buffer,
    pub(crate) pin_params_buffer: wgpu::Buffer,
    pub(crate) pin_bind_group: wgpu::BindGroup,
    pub(crate) pin_pipeline: wgpu::ComputePipeline,

    // プリミティブ & メッシュコライダー用
    pub(crate) colliders: Vec<GpuCollider>,
    pub(crate) mesh_triangles: Vec<GpuMeshTriangle>,
    pub(crate) collider_buffer: wgpu::Buffer,
    pub(crate) mesh_triangles_buffer: wgpu::Buffer,
    pub(crate) mesh_bounds_buffer: wgpu::Buffer,
    pub(crate) collider_group_bounds_buffer: wgpu::Buffer,
    pub(crate) collider_params_buffer: wgpu::Buffer,
    pub(crate) collider_bind_group: wgpu::BindGroup,
    pub(crate) collision_pipeline: wgpu::ComputePipeline,
    pub(crate) collision_bgl: wgpu::BindGroupLayout,

    // ボーンSDFコライダー用
    pub(crate) bone_infos: Vec<GpuBoneInfo>,
    pub(crate) bone_transforms: Vec<GpuBoneTransform>,
    pub(crate) enable_bone_sdf: bool,
    pub(crate) bone_sdf_texture: wgpu::Texture,
    pub(crate) bone_sdf_texture_view: wgpu::TextureView,
    pub(crate) bone_sdf_sampler: wgpu::Sampler,
    pub(crate) bone_info_buffer: wgpu::Buffer,
    pub(crate) bone_transform_buffer: wgpu::Buffer,

    // 動的ボーンSDF (GPU LBS + インメモリSDF更新) 用
    pub enable_dynamic_bone_sdf: bool,
    pub dynamic_sdf_update_interval: u32,
    pub(crate) dynamic_sdf_frame_counter: u32,
    pub(crate) dynamic_sdf_setup: Option<DynamicBoneSdfSetup>,
    pub active_dirty_bone_indices: Option<Vec<u32>>,
    pub dynamic_sdf_initial_baked: bool,

    pub(crate) rest_vertices_buffer: Option<wgpu::Buffer>,
    pub(crate) bone_skin_matrices_buffer: Option<wgpu::Buffer>,
    pub(crate) skinned_positions_buffer: Option<wgpu::Buffer>,
    pub(crate) tri_sources_buffer: Option<wgpu::Buffer>,
    pub(crate) dynamic_bake_triangles_buffer: Option<wgpu::Buffer>,
    pub(crate) dynamic_bake_params_buffer: Option<wgpu::Buffer>,
    pub(crate) dynamic_sdf_output_buffer: Option<wgpu::Buffer>,
    pub(crate) bone_inv_matrices_buffer: Option<wgpu::Buffer>,

    pub(crate) skinning_pipeline: Option<wgpu::ComputePipeline>,
    pub(crate) skinning_bind_group: Option<wgpu::BindGroup>,
    pub(crate) prep_triangles_pipeline: Option<wgpu::ComputePipeline>,
    pub(crate) prep_triangles_bind_group: Option<wgpu::BindGroup>,
    pub(crate) dynamic_bake_pipeline: Option<wgpu::ComputePipeline>,
    pub(crate) dynamic_bake_bind_group: Option<wgpu::BindGroup>,

    // 自己・レイヤー衝突 & 貫通解消 (Untangling) 用
    pub(crate) spatial_hash: GpuSpatialHash,
    pub(crate) self_collision_pipeline: wgpu::ComputePipeline,
    pub(crate) self_collision_bind_group: wgpu::BindGroup,
    pub(crate) self_collision_params_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    pub(crate) self_collision_accum_buffer: wgpu::Buffer,
    pub(crate) self_collision_apply_pipeline: wgpu::ComputePipeline,
    pub(crate) self_collision_apply_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    pub(crate) normals_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    pub(crate) local_edge_lengths_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    pub(crate) adj_offsets_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    pub(crate) adj_indices_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    pub(crate) island_ids_buffer: wgpu::Buffer,
    pub(crate) compute_normals_pipeline: wgpu::ComputePipeline,
    pub(crate) compute_normals_bind_group: wgpu::BindGroup,

    pub(crate) predict_pipeline: wgpu::ComputePipeline,
    pub(crate) distance_pipeline: wgpu::ComputePipeline,
    pub(crate) bending_pipeline: wgpu::ComputePipeline,
    pub(crate) sewing_pipeline: wgpu::ComputePipeline,
    pub(crate) update_vel_pipeline: wgpu::ComputePipeline,

    pub(crate) predict_bind_group: wgpu::BindGroup,
    pub(crate) distance_bind_groups: Vec<wgpu::BindGroup>,
    pub(crate) bending_bind_groups: Vec<wgpu::BindGroup>,
    pub(crate) sewing_bind_groups: Vec<wgpu::BindGroup>,
    pub(crate) update_vel_bind_group: wgpu::BindGroup,

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
    pub enable_collider_cluster_culling: bool,
    pub enable_single_sided_recovery: bool,
    pub collider_sweep_margin_offset: f32,
    pub self_collision_max_iterations: u32,
    pub(crate) edge_collision_pipeline: wgpu::ComputePipeline,
    pub(crate) edge_collision_bind_groups: Vec<wgpu::BindGroup>,

    // デバッグ記録用
    pub mesh_edges: Vec<[u32; 2]>,
    pub mesh_faces: Vec<[u32; 3]>,
    pub debug_recorder: SimulationDebugRecorder,
    pub original_inv_masses: Vec<f32>,
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
        let num_vertices = mesh.vertices.len() as u32;
        let num_distance_constraints = mesh.distance_constraints.len() as u32;
        let num_bending_constraints = mesh.bending_constraints.len() as u32;
        let num_sewing_constraints = mesh.sewing_constraints.len() as u32;

        let mut mesh = mesh;
        let res = build_simulation_resources(&context, &mut mesh, workgroup_size);

        let initial_vertices = mesh.vertices.clone();
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
            sewing_constraints: mesh.sewing_constraints,
            vertex_buffer: res.vertex_buffer,
            dist_buffer: res.dist_buffer,
            bend_buffer: res.bend_buffer,
            params_buffer: res.params_buffer,
            staging_buffer: res.staging_buffer,
            staging_buffers: res.staging_buffers,
            staging_idx: 0,
            async_pending: false,
            async_receiver: None,
            workgroup_size,
            solver_mode,
            accum_buffer: res.accum_buffer,
            distance_atomic_solve_pipeline: res.distance_atomic_solve_pipeline,
            distance_atomic_apply_pipeline: res.distance_atomic_apply_pipeline,
            distance_atomic_bind_group: res.distance_atomic_bind_group,
            enable_compact_readback: true,
            compact_position_buffer: res.compact_position_buffer,
            compact_staging_buffers: res.compact_staging_buffers,
            extract_positions_pipeline: res.extract_positions_pipeline,
            extract_positions_bind_group: res.extract_positions_bind_group,
            buffered_position_buffer: res.buffered_position_buffer,
            buffered_staging_buffer: res.buffered_staging_buffer,
            buffered_frame_count: 0,
            max_buffered_frames: res.max_buffered_frames,
            dynamic_pins: HashMap::new(),
            pin_buffer: res.pin_buffer,
            pin_params_buffer: res.pin_params_buffer,
            pin_bind_group: res.pin_bind_group,
            pin_pipeline: res.pin_pipeline,
            colliders: Vec::new(),
            mesh_triangles: Vec::new(),
            collider_buffer: res.collider_buffer,
            mesh_triangles_buffer: res.mesh_triangles_buffer,
            mesh_bounds_buffer: res.mesh_bounds_buffer,
            collider_group_bounds_buffer: res.collider_group_bounds_buffer,
            collider_params_buffer: res.collider_params_buffer,
            collider_bind_group: res.collider_bind_group,
            collision_pipeline: res.collision_pipeline,
            collision_bgl: res.collision_bgl,
            bone_infos: Vec::new(),
            bone_transforms: Vec::new(),
            enable_bone_sdf: false,
            bone_sdf_texture: res.bone_sdf_texture,
            bone_sdf_texture_view: res.bone_sdf_texture_view,
            bone_sdf_sampler: res.bone_sdf_sampler,
            bone_info_buffer: res.bone_info_buffer,
            bone_transform_buffer: res.bone_transform_buffer,
            enable_dynamic_bone_sdf: false,
            dynamic_sdf_update_interval: 1,
            dynamic_sdf_frame_counter: 0,
            dynamic_sdf_setup: None,
            rest_vertices_buffer: None,
            bone_skin_matrices_buffer: None,
            skinned_positions_buffer: None,
            tri_sources_buffer: None,
            dynamic_bake_triangles_buffer: None,
            dynamic_bake_params_buffer: None,
            dynamic_sdf_output_buffer: None,
            bone_inv_matrices_buffer: None,
            skinning_pipeline: None,
            skinning_bind_group: None,
            prep_triangles_pipeline: None,
            prep_triangles_bind_group: None,
            dynamic_bake_pipeline: None,
            dynamic_bake_bind_group: None,
            active_dirty_bone_indices: None,
            dynamic_sdf_initial_baked: false,
            spatial_hash: res.spatial_hash,
            self_collision_pipeline: res.self_collision_pipeline,
            self_collision_bind_group: res.self_collision_bind_group,
            self_collision_params_buffer: res.self_collision_params_buffer,
            self_collision_accum_buffer: res.self_collision_accum_buffer,
            self_collision_apply_pipeline: res.self_collision_apply_pipeline,
            self_collision_apply_bind_group: res.self_collision_apply_bind_group,
            normals_buffer: res.normals_buffer,
            local_edge_lengths_buffer: res.local_edge_lengths_buffer,
            adj_offsets_buffer: res.adj_offsets_buffer,
            adj_indices_buffer: res.adj_indices_buffer,
            island_ids_buffer: res.island_ids_buffer,
            compute_normals_pipeline: res.compute_normals_pipeline,
            compute_normals_bind_group: res.compute_normals_bind_group,
            predict_pipeline: res.predict_pipeline,
            distance_pipeline: res.distance_pipeline,
            bending_pipeline: res.bending_pipeline,
            sewing_pipeline: res.sewing_pipeline,
            update_vel_pipeline: res.update_vel_pipeline,
            predict_bind_group: res.predict_bind_group,
            distance_bind_groups: res.distance_bind_groups,
            bending_bind_groups: res.bending_bind_groups,
            sewing_bind_groups: res.sewing_bind_groups,
            update_vel_bind_group: res.update_vel_bind_group,
            solver_iterations: 2,
            gravity: [0.0, 0.0, -9.81],
            damping: 1.0,
            enable_self_collision: true,
            self_collision_relief_factor: res.self_collision_relief_factor,
            self_collision_max_displacement_ratio: res.self_collision_max_displacement_ratio,
            self_collision_exclude_neighbors: res.self_collision_exclude_neighbors,
            self_collision_max_iterations: res.self_collision_max_iterations,
            enable_normal_untangling: res.enable_normal_untangling,
            enable_edge_collision: false,
            edge_margin_scale: 1.0,
            edge_margin_offset: 0.0,
            enable_collider_cluster_culling: false,
            enable_single_sided_recovery: true,
            collider_sweep_margin_offset: 0.05,
            edge_collision_pipeline: res.edge_collision_pipeline,
            edge_collision_bind_groups: res.edge_collision_bind_groups,
            mesh_edges,
            mesh_faces,
            debug_recorder: SimulationDebugRecorder::default(),
            original_inv_masses: mesh.vertices.iter().map(|v| v.inv_mass).collect(),
        }
    }

    /// 自己衝突および貫通解消オプションを設定する
    pub fn set_self_collision_options(
        &mut self,
        relief_factor: f32,
        max_displacement_ratio: f32,
        exclude_neighbors: bool,
        enable_normal_untangling: bool,
        max_iterations: u32,
    ) {
        self.self_collision_relief_factor = relief_factor;
        self.self_collision_max_displacement_ratio = max_displacement_ratio;
        self.self_collision_exclude_neighbors = exclude_neighbors;
        self.enable_normal_untangling = enable_normal_untangling;
        self.self_collision_max_iterations = max_iterations;

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
            max_search_iterations: max_iterations,
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

    /// すべてのコライダー（球、カプセル、平面、動的メッシュ三角形）をクリア
    /// 注: ボーンSDFコライダーは静的アセットのためクリアしません（clear_bone_sdf_collidersを使用）
    pub fn clear_colliders(&mut self) {
        self.colliders.clear();
        self.mesh_triangles.clear();
        self.upload_colliders();
    }

    /// ボーンSDFコライダーの3Dテクスチャアトラスおよび静的情報を設定
    pub fn set_bone_sdf_colliders(
        &mut self,
        width: u32,
        height: u32,
        depth: u32,
        texture_data: &[u8],
        bone_infos: &[GpuBoneInfo],
    ) {
        let device = &self.context.device;
        let queue = &self.context.queue;

        // 1. 新しい 3D テクスチャ (Rg16Float) を作成
        self.bone_sdf_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("TareminCloth Bone SDF 3D Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.bone_sdf_texture_view = self.bone_sdf_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 2. テクスチャデータを書き込み (Rg16Float は 1 ピクセル 4 バイト)
        let bytes_per_pixel = 4u32;
        let bytes_per_row = width * bytes_per_pixel;
        let rows_per_image = height;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.bone_sdf_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            texture_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(rows_per_image),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: depth,
            },
        );

        // 3. ボーン静的情報をアップロード
        self.bone_infos = bone_infos.to_vec();
        if !self.bone_infos.is_empty() {
            queue.write_buffer(
                &self.bone_info_buffer,
                0,
                bytemuck::cast_slice(&self.bone_infos),
            );
        }

        self.enable_bone_sdf = !self.bone_infos.is_empty();

        // 4. BindGroup を再作成
        self.recreate_collider_bind_group();
        self.upload_colliders();
    }

    /// 各ボーンのワールド変換行列を更新（毎フレーム実行）
    pub fn update_bone_transforms(&mut self, transforms: &[GpuBoneTransform]) {
        self.bone_transforms = transforms.to_vec();
        if !self.bone_transforms.is_empty() {
            self.context.queue.write_buffer(
                &self.bone_transform_buffer,
                0,
                bytemuck::cast_slice(&self.bone_transforms),
            );
        }
    }

    /// ボーンSDFコライダーをクリア・無効化
    pub fn clear_bone_sdf_colliders(&mut self) {
        self.bone_infos.clear();
        self.bone_transforms.clear();
        self.enable_bone_sdf = false;
        self.enable_dynamic_bone_sdf = false;
        self.dynamic_sdf_setup = None;
        self.upload_colliders();
    }

    /// 動的ボーンSDFの有効/無効を切り替え
    pub fn set_dynamic_bone_sdf_enabled(&mut self, enabled: bool) {
        self.enable_dynamic_bone_sdf = enabled && self.dynamic_sdf_setup.is_some();
    }

    /// 動的ボーンSDFの更新間隔（フレーム数）を設定
    pub fn set_dynamic_bone_sdf_update_interval(&mut self, interval: u32) {
        self.dynamic_sdf_update_interval = interval.max(1);
    }

    /// 動的ボーンSDF（GPU LBS + インメモリSDF更新）パイプラインのセットアップ
    pub fn setup_dynamic_bone_sdf(&mut self, setup: DynamicBoneSdfSetup) {
        let num_verts = setup.rest_verts.len();
        let num_tris = setup.tri_sources.len();
        let num_bones = setup.bone_bind_inv_matrices.len();

        if num_verts == 0 || num_tris == 0 || num_bones == 0 {
            return;
        }

        // 1. 3Dテクスチャを setup.width, setup.height, setup.depth で作成
        self.bone_sdf_texture = self.context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("TareminCloth Dynamic Bone SDF 3D Texture"),
            size: wgpu::Extent3d {
                width: setup.width,
                height: setup.height,
                depth_or_array_layers: setup.depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.bone_sdf_texture_view = self.bone_sdf_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 2. ボーン静的情報をセット
        self.bone_infos = setup.bone_infos.clone();
        if !self.bone_infos.is_empty() {
            self.context.queue.write_buffer(&self.bone_info_buffer, 0, bytemuck::cast_slice(&self.bone_infos));
        }
        self.enable_bone_sdf = !self.bone_infos.is_empty();
        self.recreate_collider_bind_group();
        self.upload_colliders();

        let device = &self.context.device;
        let queue = &self.context.queue;

        // 3. スキニング用バッファ
        let rest_verts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dynamic SDF Rest Vertices"),
            size: (std::mem::size_of::<GpuSkinningVertex>() * num_verts) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&rest_verts_buf, 0, bytemuck::cast_slice(&setup.rest_verts));

        let bone_skin_mats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dynamic SDF Bone Skin Matrices"),
            size: (std::mem::size_of::<[[f32; 4]; 4]>() * num_bones) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let skinned_pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dynamic SDF Skinned Positions"),
            size: (std::mem::size_of::<[f32; 4]>() * num_verts) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // 4. 三角形準備用バッファ
        let bone_inv_mats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dynamic SDF Bone Inv Matrices"),
            size: (std::mem::size_of::<[[f32; 4]; 4]>() * num_bones) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tri_sources_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dynamic SDF Triangle Sources"),
            size: (std::mem::size_of::<GpuBoneTriangleSource>() * num_tris) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&tri_sources_buf, 0, bytemuck::cast_slice(&setup.tri_sources));

        let dynamic_bake_tris_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dynamic SDF Bake Triangles"),
            size: (std::mem::size_of::<GpuBakeTriangle>() * num_tris) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // 5. ベイク用バッファ
        let bytes_per_pixel = 4u32;
        let bytes_per_row = setup.width * bytes_per_pixel;
        let aligned_bytes_per_row = (bytes_per_row + 255) & !255;
        let row_pitch = aligned_bytes_per_row / bytes_per_pixel;

        let mut setup = setup;
        for p in &mut setup.bake_params {
            p.row_pitch = row_pitch;
        }

        let bake_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dynamic SDF Bake Params"),
            size: (std::mem::size_of::<GpuBakeParams>() * setup.bake_params.len()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&bake_params_buf, 0, bytemuck::cast_slice(&setup.bake_params));

        let total_output_bytes = (aligned_bytes_per_row * setup.height * setup.depth) as u64;
        let sdf_output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dynamic SDF Output Voxels Buffer"),
            size: total_output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 6. シェーダーモジュール作成
        let skinning_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dynamic Skinning Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/skinning.wgsl").into()),
        });
        let prep_triangles_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dynamic Prep Triangles Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/prep_bone_triangles.wgsl").into()),
        });
        let dynamic_bake_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dynamic SDF Bake Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/bake_sdf.wgsl").into()),
        });

        // 7. パイプライン作成
        let storage_bgl_entry = |binding: u32, read_only: bool| -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        };

        // (A) スキニング
        let skinning_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Dynamic Skinning BGL"),
            entries: &[
                storage_bgl_entry(0, true),
                storage_bgl_entry(1, true),
                storage_bgl_entry(2, false),
            ],
        });
        let skinning_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Dynamic Skinning Pipeline Layout"),
            bind_group_layouts: &[&skinning_bgl],
            push_constant_ranges: &[],
        });
        let skinning_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dynamic Skinning Pipeline"),
            layout: Some(&skinning_pl),
            module: &skinning_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let skinning_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dynamic Skinning BindGroup"),
            layout: &skinning_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: rest_verts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bone_skin_mats_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: skinned_pos_buf.as_entire_binding() },
            ],
        });

        // (B) 三角形準備
        let prep_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Dynamic Prep Triangles BGL"),
            entries: &[
                storage_bgl_entry(0, true),
                storage_bgl_entry(1, true),
                storage_bgl_entry(2, true),
                storage_bgl_entry(3, false),
            ],
        });
        let prep_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Dynamic Prep Triangles Pipeline Layout"),
            bind_group_layouts: &[&prep_bgl],
            push_constant_ranges: &[],
        });
        let prep_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dynamic Prep Triangles Pipeline"),
            layout: Some(&prep_pl),
            module: &prep_triangles_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let prep_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dynamic Prep Triangles BindGroup"),
            layout: &prep_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: skinned_pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bone_inv_mats_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: tri_sources_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dynamic_bake_tris_buf.as_entire_binding() },
            ],
        });

        // (C) ベイク
        let bake_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Dynamic SDF Bake BGL"),
            entries: &[
                storage_bgl_entry(0, true),
                storage_bgl_entry(1, true),
                storage_bgl_entry(2, false),
            ],
        });
        let bake_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Dynamic SDF Bake Pipeline Layout"),
            bind_group_layouts: &[&bake_bgl],
            push_constant_ranges: &[],
        });
        let bake_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dynamic SDF Bake Pipeline"),
            layout: Some(&bake_pl),
            module: &dynamic_bake_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bake_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dynamic SDF Bake BindGroup"),
            layout: &bake_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bake_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: dynamic_bake_tris_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: sdf_output_buf.as_entire_binding() },
            ],
        });

        // 8. 内部状態の保存
        self.rest_vertices_buffer = Some(rest_verts_buf);
        self.bone_skin_matrices_buffer = Some(bone_skin_mats_buf);
        self.skinned_positions_buffer = Some(skinned_pos_buf);
        self.tri_sources_buffer = Some(tri_sources_buf);
        self.dynamic_bake_triangles_buffer = Some(dynamic_bake_tris_buf);
        self.dynamic_bake_params_buffer = Some(bake_params_buf);
        self.dynamic_sdf_output_buffer = Some(sdf_output_buf);
        self.bone_inv_matrices_buffer = Some(bone_inv_mats_buf);

        self.skinning_pipeline = Some(skinning_pipeline);
        self.skinning_bind_group = Some(skinning_bg);
        self.prep_triangles_pipeline = Some(prep_pipeline);
        self.prep_triangles_bind_group = Some(prep_bg);
        self.dynamic_bake_pipeline = Some(bake_pipeline);
        self.dynamic_bake_bind_group = Some(bake_bg);

        self.dynamic_sdf_setup = Some(setup);
        self.active_dirty_bone_indices = None;
        self.dynamic_sdf_initial_baked = false;
        self.enable_dynamic_bone_sdf = true;
    }

    /// 動的ボーンSDFのDirtyボーンインデックス（再ベイク対象）を設定
    pub fn set_dynamic_bone_sdf_dirty_bones(&mut self, dirty_bones: Option<Vec<u32>>) {
        self.active_dirty_bone_indices = dirty_bones;
    }

    /// 毎フレームの動的SDF更新ディスパッチ（GPUインメモリ完結）
    pub fn dispatch_dynamic_bone_sdf_update(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.enable_dynamic_bone_sdf {
            return;
        }

        let setup = match self.dynamic_sdf_setup.as_ref() {
            Some(s) => s,
            None => return,
        };

        self.dynamic_sdf_frame_counter = self.dynamic_sdf_frame_counter.wrapping_add(1);
        if (self.dynamic_sdf_frame_counter - 1) % self.dynamic_sdf_update_interval != 0 {
            return;
        }

        let num_bones = setup.bone_bind_inv_matrices.len();
        if self.bone_transforms.len() < num_bones {
            return;
        }

        // 1. 各ボーンの変形行列 S = M_curr * B_bind を計算してアップロード
        // self.bone_transforms[b].world_matrix は WGSL 用に Column-Major で格納されているため、
        // Row-Major に転置してから B_bind (Row-Major) と積を取り、再度 Column-Major に転置して GPU へ渡す
        let mut skin_mats = Vec::with_capacity(num_bones);
        let mut inv_mats = Vec::with_capacity(num_bones);
        for b in 0..num_bones {
            let w_row = mat4_transpose(&self.bone_transforms[b].world_matrix);
            let b_bind = &setup.bone_bind_inv_matrices[b];
            let s_row = mat4_mul(&w_row, b_bind);
            skin_mats.push(mat4_transpose(&s_row));
            inv_mats.push(self.bone_transforms[b].inv_world_matrix);
        }

        if let Some(ref buf) = self.bone_skin_matrices_buffer {
            self.context.queue.write_buffer(buf, 0, bytemuck::cast_slice(&skin_mats));
        }
        if let Some(ref buf) = self.bone_inv_matrices_buffer {
            self.context.queue.write_buffer(buf, 0, bytemuck::cast_slice(&inv_mats));
        }

        // 2. Pass 1: LBSスキニング
        let num_verts = setup.rest_verts.len() as u32;
        if let (Some(ref pipeline), Some(ref bg)) = (&self.skinning_pipeline, &self.skinning_bind_group) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dynamic SDF Skinning Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups((num_verts + 63) / 64, 1, 1);
        }

        // 3. Pass 2: ボーン局所三角形準備
        let num_tris = setup.tri_sources.len() as u32;
        if let (Some(ref pipeline), Some(ref bg)) = (&self.prep_triangles_pipeline, &self.prep_triangles_bind_group) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dynamic SDF Prep Triangles Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups((num_tris + 63) / 64, 1, 1);
        }

        // Dirty ボーン判定
        // 初期ベイクが未実行の場合は、Python側の指定にかかわらず強制的に全ボーンベイクを実行
        let (dirty_count, is_dirty_mode) = if !self.dynamic_sdf_initial_baked {
            (num_bones as u32, false)
        } else {
            match self.active_dirty_bone_indices {
                Some(ref list) => {
                    if list.is_empty() {
                        // ポーズ変化なし -> GPUディスパッチ・テクスチャコピーを完全スキップ (0ms)
                        return;
                    }
                    (list.len() as u32, true)
                }
                None => (num_bones as u32, false),
            }
        };

        // 4. Pass 3: SDFベイク用パラメータの準備
        if is_dirty_mode {
            let list = self.active_dirty_bone_indices.as_ref().unwrap();
            let mut dirty_params = Vec::with_capacity(list.len());
            for &b in list {
                if (b as usize) < setup.bake_params.len() {
                    dirty_params.push(setup.bake_params[b as usize]);
                }
            }
            if let Some(ref buf) = self.dynamic_bake_params_buffer {
                self.context.queue.write_buffer(buf, 0, bytemuck::cast_slice(&dirty_params));
            }
        } else {
            if let Some(ref buf) = self.dynamic_bake_params_buffer {
                self.context.queue.write_buffer(buf, 0, bytemuck::cast_slice(&setup.bake_params));
            }
        }

        // 4. Pass 3: SDFベイク
        if let (Some(ref pipeline), Some(ref bg)) = (&self.dynamic_bake_pipeline, &self.dynamic_bake_bind_group) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dynamic SDF Bake Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            let wg_x = (setup.res + 3) / 4;
            let wg_y = (setup.res + 3) / 4;
            let wg_z = (setup.res * dirty_count + 3) / 4;
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }

        // 5. Pass 4: 3D テクスチャへの直接GPU内コピー
        let bytes_per_pixel = 4u32;
        let bytes_per_row = setup.width * bytes_per_pixel;
        let aligned_bytes_per_row = (bytes_per_row + 255) & !255;
        if let Some(ref out_buf) = self.dynamic_sdf_output_buffer {
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: out_buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(aligned_bytes_per_row),
                        rows_per_image: Some(setup.height),
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.bone_sdf_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: setup.width,
                    height: setup.height,
                    depth_or_array_layers: setup.depth,
                },
            );
        }

        self.dynamic_sdf_initial_baked = true;
    }

    fn recreate_collider_bind_group(&mut self) {
        self.collider_bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Collider Bind Group (Recreated with Bone SDF)"),
            layout: &self.collision_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.collider_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.mesh_triangles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.collider_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.mesh_bounds_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.collider_group_bounds_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.bone_sdf_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&self.bone_sdf_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.bone_info_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.bone_transform_buffer.as_entire_binding(),
                },
            ],
        });
    }
    /// コライダー最適化およびリカバリーのオプションを設定する
    pub fn set_collider_options(
        &mut self,
        enable_cluster_culling: bool,
        enable_single_sided_recovery: bool,
        sweep_margin: f32,
    ) {
        self.enable_collider_cluster_culling = enable_cluster_culling;
        self.enable_single_sided_recovery = enable_single_sided_recovery;
        self.collider_sweep_margin_offset = sweep_margin;
        self.upload_colliders();
    }

    pub(crate) fn upload_colliders(&self) {
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

            // 16面クラスタごとの階層境界球の事前計算とアップロード
            let num_clusters = (num_mesh_triangles + 15) / 16;
            let mut group_bounds = Vec::with_capacity(num_clusters as usize);
            for g in 0..num_clusters {
                let start = (g * 16) as usize;
                let end = ((g * 16 + 16) as usize).min(bounds.len());
                let chunk = &bounds[start..end];
                let (mut min_x, mut min_y, mut min_z) = (f32::MAX, f32::MAX, f32::MAX);
                let (mut max_x, mut max_y, mut max_z) = (f32::MIN, f32::MIN, f32::MIN);
                for b in chunk {
                    min_x = min_x.min(b[0] - b[3]);
                    min_y = min_y.min(b[1] - b[3]);
                    min_z = min_z.min(b[2] - b[3]);
                    max_x = max_x.max(b[0] + b[3]);
                    max_y = max_y.max(b[1] + b[3]);
                    max_z = max_z.max(b[2] + b[3]);
                }
                let g_cx = (min_x + max_x) * 0.5;
                let g_cy = (min_y + max_y) * 0.5;
                let g_cz = (min_z + max_z) * 0.5;
                let g_r = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2) + (max_z - min_z).powi(2)).sqrt() * 0.5;
                group_bounds.push([g_cx, g_cy, g_cz, g_r]);
            }
            if !group_bounds.is_empty() {
                self.context.queue.write_buffer(
                    &self.collider_group_bounds_buffer,
                    0,
                    bytemuck::cast_slice(&group_bounds),
                );
            }
        }

        let num_clusters = if num_mesh_triangles > 0 { (num_mesh_triangles + 15) / 16 } else { 0 };
        let params = CollisionParams {
            num_vertices: self.num_vertices,
            num_colliders,
            num_mesh_triangles,
            num_clusters,
            dt: 0.016 / 20.0,
            edge_margin_scale: self.edge_margin_scale,
            edge_margin_offset: self.edge_margin_offset,
            enable_cluster_culling: if self.enable_collider_cluster_culling { 1 } else { 0 },
            enable_single_sided_recovery: if self.enable_single_sided_recovery { 1 } else { 0 },
            sweep_margin_offset: self.collider_sweep_margin_offset,
            num_bones: self.bone_infos.len() as u32,
            enable_bone_sdf: if self.enable_bone_sdf { 1 } else { 0 },
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

            let new_inv_m = if weight > 0.5 { 0.0f32 } else { self.original_inv_masses[vertex_idx as usize] };
            let offset = (vertex_idx as usize * std::mem::size_of::<crate::mesh::GpuVertex>() + 12) as u64;
            self.context.queue.write_buffer(&self.vertex_buffer, offset, bytemuck::bytes_of(&new_inv_m));
        }
    }

    /// 指定頂点の動的ピンを解除する
    pub fn release_pin(&mut self, vertex_idx: u32) {
        if self.dynamic_pins.remove(&vertex_idx).is_some() {
            self.upload_pins();
            if (vertex_idx as usize) < self.original_inv_masses.len() {
                let orig_m = self.original_inv_masses[vertex_idx as usize];
                let offset = (vertex_idx as usize * std::mem::size_of::<crate::mesh::GpuVertex>() + 12) as u64;
                self.context.queue.write_buffer(&self.vertex_buffer, offset, bytemuck::bytes_of(&orig_m));
            }
        }
    }

    /// すべての動的ピンを解除する
    pub fn clear_dynamic_pins(&mut self) {
        for &v_idx in self.dynamic_pins.keys() {
            if (v_idx as usize) < self.original_inv_masses.len() {
                let orig_m = self.original_inv_masses[v_idx as usize];
                let offset = (v_idx as usize * std::mem::size_of::<crate::mesh::GpuVertex>() + 12) as u64;
                self.context.queue.write_buffer(&self.vertex_buffer, offset, bytemuck::bytes_of(&orig_m));
            }
        }
        self.dynamic_pins.clear();
        self.upload_pins();
    }

    pub(crate) fn upload_pins(&self) {
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
}

#[inline]
fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = a[i][0] * b[0][j]
                + a[i][1] * b[1][j]
                + a[i][2] * b[2][j]
                + a[i][3] * b[3][j];
        }
    }
    out
}

#[inline]
fn mat4_transpose(m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = m[j][i];
        }
    }
    out
}
