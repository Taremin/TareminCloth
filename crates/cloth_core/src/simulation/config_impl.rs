use super::GpuClothSimulator;
use crate::config::SimConfig;

pub(crate) fn compliance_to_stiffness(comp: f32, scale: f32) -> f32 {
    if comp <= 1e-9 {
        5000.0
    } else if comp >= 1e9 {
        0.0
    } else {
        1.0 / (comp * scale)
    }
}

impl GpuClothSimulator {
    /// 現在の可変パラメータを SimConfig として書き出す (記録・差分検出用)
    pub fn export_config(&self) -> SimConfig {
        let (tension, compression, shear, bending) = self.current_stiffness_quad();
        SimConfig {
            version: 1,
            gravity: self.gravity,
            damping: self.damping,
            tension_damping: 0.0,
            compression_damping: 0.0,
            shear_damping: 0.0,
            bending_damping: 0.0,
            tension_stiffness: tension,
            compression_stiffness: compression,
            shear_stiffness: shear,
            bending_stiffness: bending,
            solver_iterations: self.solver_iterations,
            solver_mode: self.solver_mode,
            workgroup_size: self.workgroup_size,
            enable_self_collision: self.enable_self_collision,
            relief_factor: self.self_collision_relief_factor,
            max_displacement_ratio: self.self_collision_max_displacement_ratio,
            exclude_neighbors: self.self_collision_exclude_neighbors,
            enable_normal_untangling: self.enable_normal_untangling,
            self_collision_max_iterations: self.self_collision_max_iterations,
            coupled_mode: self.coupled_self_collision_mode,
            coupled_collider: self.coupled_collider,
            post_relaxation_iters: self.post_collision_relaxation_iters,
            substep_interval: self.self_collision_substep_interval,
            ee_substep_interval: self.self_collision_ee_substep_interval,
            self_collision_algorithm: self.self_collision_algorithm,
            enable_pair_cache: self.enable_pair_cache,
            pair_margin_mode: self.pair_cache_margin_mode,
            pair_safety_margin: self.pair_cache_safety_margin,
            pair_horizon_scale: self.pair_cache_horizon_scale,
            pair_max_horizon: self.pair_cache_max_horizon,
            pair_max_pairs: self
                .pair_cache_max_vt_pairs
                .max(self.pair_cache_max_ee_pairs),
            enable_pair_final_fallback: self.enable_pair_cache_final_fallback,
            enable_edge_collision: self.enable_edge_collision,
            edge_margin_scale: self.edge_margin_scale,
            edge_margin_offset: self.edge_margin_offset,
            sewing_stiffness: self.sewing_stiffness,
            enable_sewing_lock: self.enable_sewing_lock,
            sewing_lock_distance: self.sewing_lock_distance,
            areal_density: self.areal_density,
            sewing_priority_enabled: self.sewing_priority_enabled,
            sewing_priority_threshold: self.sewing_priority_threshold,
            sewing_priority_merge_dist: self.sewing_priority_merge_dist,
            sewing_priority_ramp_frames: self.sewing_priority_ramp_frames,
            sewing_priority_max_frames: self.sewing_priority_max_frames,
            enable_adaptive_substep: self.enable_adaptive_substep,
            min_substeps: self.min_substeps,
            max_substeps: self.max_substeps,
            auto_coupled_on_low_substeps: self.auto_coupled_on_low_substeps,
            auto_compensate_iterations: self.auto_compensate_iterations,
            enable_strain_adaptive: self.enable_strain_adaptive,
            strain_tolerance: self.strain_tolerance,
        }
    }

    fn current_stiffness_quad(&self) -> (f32, f32, f32, f32) {
        let mut tension_c = None;
        let mut shear_c = None;
        for dc in &self.distance_constraints {
            if dc.constraint_type == 1 {
                if shear_c.is_none() {
                    shear_c = Some((dc.tension_compliance, dc.compression_compliance));
                }
            } else if tension_c.is_none() {
                tension_c = Some((dc.tension_compliance, dc.compression_compliance));
            }
        }
        let (t_t, t_c) = tension_c.unwrap_or((1.0 / 500.0, 1.0 / 500.0));
        let (s_t, _) = shear_c.unwrap_or((1.0 / 250.0, 1.0 / 250.0));
        let tension = compliance_to_stiffness(t_t, 1.0);
        let compression = compliance_to_stiffness(t_c, 1.0);
        let shear = compliance_to_stiffness(s_t, 1.0);
        let bending = if let Some(bc) = self.bending_constraints.first() {
            compliance_to_stiffness(bc.compliance, 1.0)
        } else {
            5.0
        };
        (tension, compression, shear, bending)
    }

    /// SimConfig をシミュレータに一括適用する (再生・初期化用).
    /// 既存 set_* の薄い集約であり、GPUバッファ更新も各 setter 経由で行う.
    /// workgroup_size は生成後変更してもパイプライン再構築されないため記録のみ.
    pub fn apply_config(&mut self, c: &SimConfig) {
        self.set_gravity(c.gravity[0], c.gravity[1], c.gravity[2]);
        // damping_all は加算型のため直接代入で再現性を保つ
        let detail = (c.tension_damping + c.compression_damping + c.shear_damping + c.bending_damping) * 0.05;
        self.damping = (c.damping + detail).max(0.0);
        self.set_stiffness_all(
            c.tension_stiffness,
            c.compression_stiffness,
            c.shear_stiffness,
            c.bending_stiffness,
        );
        self.set_solver_iterations(c.solver_iterations);
        self.set_solver_mode(c.solver_mode);
        self.workgroup_size = c.workgroup_size;
        self.set_enable_self_collision(c.enable_self_collision);
        self.set_self_collision_options(
            c.relief_factor,
            c.max_displacement_ratio,
            c.exclude_neighbors,
            c.enable_normal_untangling,
            c.self_collision_max_iterations,
        );
        self.set_coupled_self_collision_options(c.coupled_mode, c.post_relaxation_iters);
        self.set_coupled_collider(c.coupled_collider);
        self.set_auto_coupled_on_low_substeps(c.auto_coupled_on_low_substeps);
        self.set_self_collision_substep_interval(c.substep_interval.max(1));
        self.set_self_collision_ee_substep_interval(c.ee_substep_interval);
        self.set_self_collision_algorithm(c.self_collision_algorithm);
        self.set_enable_pair_cache(c.enable_pair_cache);
        self.set_pair_cache_options(
            c.pair_max_pairs,
            c.pair_max_pairs,
            c.pair_margin_mode,
            c.pair_safety_margin,
            c.pair_horizon_scale,
            c.pair_max_horizon,
        );
        self.set_enable_pair_cache_final_fallback(c.enable_pair_final_fallback);
        self.set_enable_edge_collision(c.enable_edge_collision);
        self.set_edge_margin_scale(c.edge_margin_scale);
        self.set_edge_margin_offset(c.edge_margin_offset);
        self.set_sewing_stiffness(c.sewing_stiffness);
        self.set_enable_sewing_lock(c.enable_sewing_lock);
        self.set_sewing_lock_distance(c.sewing_lock_distance);
        self.set_areal_density(c.areal_density);
        self.set_sewing_priority_options(
            c.sewing_priority_enabled,
            c.sewing_priority_threshold,
            c.sewing_priority_merge_dist,
            c.sewing_priority_ramp_frames,
            c.sewing_priority_max_frames,
        );
        self.set_adaptive_substep_options(
            c.enable_adaptive_substep,
            c.min_substeps,
            c.max_substeps,
        );
        self.set_auto_coupled_on_low_substeps(c.auto_coupled_on_low_substeps);
        self.set_auto_compensate_iterations(c.auto_compensate_iterations);
        self.set_strain_adaptive_options(c.enable_strain_adaptive, c.strain_tolerance);
    }

    /// 現在設定のハッシュ (フレーム途中変更検出用)
    pub fn config_hash_u64(&self) -> u64 {
        self.export_config().hash_u64()
    }
}
