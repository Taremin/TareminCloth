use crate::debug_recorder::{ColliderRecord, PinRecord, SimulationMetadata};
use super::GpuClothSimulator;

impl GpuClothSimulator {
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

        let sewing_springs = if !self.sewing_constraints.is_empty() {
            Some(self.sewing_constraints.iter().map(|sc| [sc.v0, sc.v1]).collect())
        } else {
            None
        };
        let compression_stiffness = self
            .distance_constraints
            .first()
            .map(|dc| if dc.compression_compliance > 1e-9 { 1.0 / dc.compression_compliance } else { 0.0 });
        let shear_stiffness = self
            .distance_constraints
            .iter()
            .find(|dc| dc.constraint_type == 1)
            .map(|dc| if dc.tension_compliance > 1e-9 { 1.0 / dc.tension_compliance } else { 0.0 });

        let metadata = SimulationMetadata {
            object_name: object_name.to_string(),
            num_vertices: self.num_vertices,
            num_edges: self.mesh_edges.len() as u32,
            num_faces: self.mesh_faces.len() as u32,
            edges: self.mesh_edges.clone(),
            faces: self.mesh_faces.clone(),
            inv_masses: self.original_inv_masses.clone(),
            initial_rest_lengths: self.initial_distance_rest_lengths.clone(),
            sewing_springs,
            stiffness,
            compression_stiffness,
            shear_stiffness,
            bending_stiffness,
            thickness,
            gravity: self.gravity,
            damping: self.damping,
            solver_mode: self.solver_mode,
            solver_iterations: self.solver_iterations,
            workgroup_size: self.workgroup_size,
            enable_self_collision: self.enable_self_collision,
            self_collision_relief_factor: self.self_collision_relief_factor,
            self_collision_max_displacement_ratio: self.self_collision_max_displacement_ratio,
            self_collision_exclude_neighbors: self.self_collision_exclude_neighbors,
            enable_normal_untangling: self.enable_normal_untangling,
            enable_edge_collision: self.enable_edge_collision,
            edge_margin_scale: self.edge_margin_scale,
            edge_margin_offset: self.edge_margin_offset,
            self_collision_max_iterations: self.self_collision_max_iterations,
            enable_pair_cache: self.enable_pair_cache,
            pair_cache_margin_mode: self.pair_cache_margin_mode,
            pair_cache_safety_margin: self.pair_cache_safety_margin,
            pair_cache_horizon_scale: self.pair_cache_horizon_scale,
            pair_cache_max_horizon: self.pair_cache_max_horizon,
            pair_cache_max_pairs: self.pair_cache_max_vt_pairs.max(self.pair_cache_max_ee_pairs),
            enable_pair_cache_final_fallback: self.enable_pair_cache_final_fallback,
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

        let pins: Vec<PinRecord> = self
            .dynamic_pins
            .values()
            .map(|p| PinRecord {
                vertex_idx: p.vertex_idx,
                target_pos: p.target_pos,
                weight: p.weight,
            })
            .collect();

        let mut colliders: Vec<ColliderRecord> = self
            .colliders
            .iter()
            .map(|c| match c.collider_type {
                0 => ColliderRecord::Sphere {
                    center: c.point_a,
                    radius: c.radius,
                    friction: c.friction,
                    restitution: c.restitution,
                },
                1 => ColliderRecord::Capsule {
                    point_a: c.point_a,
                    point_b: c.point_b,
                    radius: c.radius,
                    friction: c.friction,
                    restitution: c.restitution,
                },
                2 => ColliderRecord::Plane {
                    point: c.point_a,
                    normal: c.point_b,
                    friction: c.friction,
                    restitution: c.restitution,
                },
                _ => ColliderRecord::Sphere {
                    center: c.point_a,
                    radius: c.radius,
                    friction: c.friction,
                    restitution: c.restitution,
                },
            })
            .collect();

        if !self.mesh_triangles.is_empty() {
            let mut tris: Vec<[f32; 9]> = Vec::with_capacity(self.mesh_triangles.len());
            for tri in &self.mesh_triangles {
                tris.push([
                    tri.p0[0], tri.p0[1], tri.p0[2],
                    tri.p1[0], tri.p1[1], tri.p1[2],
                    tri.p2[0], tri.p2[1], tri.p2[2],
                ]);
            }
            let first = &self.mesh_triangles[0];
            colliders.push(ColliderRecord::Mesh {
                triangles: tris,
                friction: first.friction,
                thickness: first.thickness,
                restitution: first.restitution,
                single_sided: (first.flags & 1) != 0,
            });
        }

        self.debug_recorder.record_frame(
            dt,
            substeps,
            self.solver_iterations,
            &positions,
            &velocities,
            &pins,
            &colliders,
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
