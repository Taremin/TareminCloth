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
use self::types::{CollisionParams, PinParams};
use self::pipelines::build_simulation_resources;

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

    // 自己・レイヤー衝突 & 貫通解消 (Untangling) 用
    pub(crate) spatial_hash: GpuSpatialHash,
    pub(crate) self_collision_pipeline: wgpu::ComputePipeline,
    pub(crate) self_collision_bind_group: wgpu::BindGroup,
    pub(crate) self_collision_params_buffer: wgpu::Buffer,
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
            spatial_hash: res.spatial_hash,
            self_collision_pipeline: res.self_collision_pipeline,
            self_collision_bind_group: res.self_collision_bind_group,
            self_collision_params_buffer: res.self_collision_params_buffer,
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

    /// すべてのコライダーをクリア
    pub fn clear_colliders(&mut self) {
        self.colliders.clear();
        self.mesh_triangles.clear();
        self.upload_colliders();
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
            _pad0: 0,
            _pad1: 0,
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
