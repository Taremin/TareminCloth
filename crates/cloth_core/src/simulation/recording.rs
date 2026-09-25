use crate::debug_recorder::{ColliderRecord, PinRecord, SimulationMetadata};
use super::GpuClothSimulator;
use super::config_impl::compliance_to_stiffness;

impl GpuClothSimulator {
    /// シミュレーション状態のデバッグ記録を開始する
    pub fn start_debug_recording(&mut self, object_name: &str, max_frames: Option<usize>) {
        let stiffness = self
            .distance_constraints
            .first()
            .map(|dc| compliance_to_stiffness(dc.tension_compliance, 1.0))
            .unwrap_or(0.0);
        let bending_stiffness = self
            .bending_constraints
            .first()
            .map(|bc| compliance_to_stiffness(bc.compliance, 1.0))
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
            .map(|dc| compliance_to_stiffness(dc.compression_compliance, 1.0));
        let shear_stiffness = self
            .distance_constraints
            .iter()
            .find(|dc| dc.constraint_type == 1)
            .map(|dc| compliance_to_stiffness(dc.tension_compliance, 1.0));

        let metadata = SimulationMetadata {
            object_name: object_name.to_string(),
            num_vertices: self.num_vertices,
            num_edges: self.mesh_edges.len() as u32,
            num_faces: self.mesh_faces.len() as u32,
            edges: self.mesh_edges.clone(),
            faces: self.mesh_faces.clone(),
            inv_masses: self.original_inv_masses.clone(),
            initial_rest_lengths: self.initial_distance_rest_lengths.clone(),
            initial_positions: Some(self.initial_vertices.iter().map(|v| v.position).collect()),
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
            sewing_priority_enabled: self.sewing_priority_enabled,
            sewing_priority_threshold: self.sewing_priority_threshold,
            sewing_priority_merge_dist: self.sewing_priority_merge_dist,
            sewing_priority_ramp_frames: self.sewing_priority_ramp_frames,
            sewing_priority_max_frames: self.sewing_priority_max_frames,
            thicknesses: Some(self.initial_vertices.iter().map(|v| v.thickness).collect()),
            layer_ids: Some(self.initial_vertices.iter().map(|v| v.layer_id).collect()),
            original_edges: Some(self.original_mesh_edges()),
            original_rest_lengths: Some(self.original_mesh_rest_lengths()),
            bone_sdf: self.snapshot_bone_sdf(),
            sewing_shrink_speed: self
                .sewing_constraints
                .first()
                .map(|c| c.shrink_speed),
            areal_density: self.areal_density,
            enable_coarse_constraints: self.coarse_constraint_count > 0,
            config: Some(self.export_config()),
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

        // 拡張Statsの事前計算 (単一フェーズで記録するため borrow 前に済ませる)
        let live_config = self.export_config();
        let mut max_strain = 0.0f32;
        for (i, e) in self.mesh_edges.iter().enumerate() {
            if let Some(&rest) = self.initial_distance_rest_lengths.get(i) {
                if rest > 1e-9 {
                    if let (Some(&pa), Some(&pb)) =
                        (positions.get(e[0] as usize), positions.get(e[1] as usize))
                    {
                        let dx = pa[0] - pb[0];
                        let dy = pa[1] - pb[1];
                        let dz = pa[2] - pb[2];
                        let len = (dx * dx + dy * dy + dz * dz).sqrt();
                        let s = ((len - rest) / rest).abs();
                        if s.is_finite() && s > max_strain {
                            max_strain = s;
                        }
                    }
                }
            }
        }
        // Pair統計は間引き可能 (既定毎フレーム)。飽和トリガーの粒度に影響する。
        let pair_stride = self.debug_recorder.record_options().pair_stats_stride.max(1);
        let seq = self.debug_recorder.next_frame_seq();
        let pair = if self.enable_pair_cache && seq % pair_stride == 0 {
            let (vt, ee, max_vt, max_ee) = self.get_pair_cache_stats();
            Some((vt, ee, vt >= max_vt || ee >= max_ee))
        } else {
            None
        };

        // 縫合の現在自然長 (GPU進行値。縫合なしでは記録しない)
        let sewing_rests = if self.num_sewing_constraints > 0 {
            let rests = self.read_sewing_current_rest_lengths();
            if rests.len() == self.num_sewing_constraints as usize {
                Some(rests)
            } else {
                None
            }
        } else {
            None
        };
        // 縫合優先モードのラッチ内部状態 (有効時のみ)
        let sewing_priority = if self.sewing_priority_enabled && !self.sewing_constraints.is_empty() {
            let (latched, frame, ramp_t, ratio) = self.sewing_priority_state();
            Some([latched as u8 as f32, frame as f32, ramp_t, ratio])
        } else {
            None
        };

        self.debug_recorder.record_frame_ext(
            dt,
            substeps,
            self.solver_iterations,
            &positions,
            &velocities,
            &pins,
            &colliders,
            pair,
            max_strain,
            Some(live_config),
            self.current_elastic_scales(),
            self.current_bone_state(),
            sewing_rests,
            sewing_priority,
            false,
        );
    }

    /// BONE_SDFコライダーの静的スナップショットを取得する (記録開始時に1回)。
    /// テクスチャ読戻しに失敗した場合は None (その旨で記録されリプレイ時に警告される)。
    fn snapshot_bone_sdf(&self) -> Option<crate::debug_recorder::BoneSdfRecord> {
        use crate::debug_recorder::BoneSdfRecord;
        let (w, h, d) = self.bone_sdf_dims()?;
        let bytes = self.read_bone_sdf_texture()?;
        Some(BoneSdfRecord {
            width: w,
            height: h,
            depth: d,
            texture_base64: base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                &bytes,
            ),
            bone_infos: self.bone_infos_flat(),
            dynamic_enabled: self.enable_dynamic_bone_sdf,
        })
    }

    /// 現在のボーン姿勢を毎フレーム記録用に取得する (CPUミラーのみ、GPU転送なし)。
    fn current_bone_state(&self) -> Option<crate::debug_recorder::BoneFrameState> {
        use crate::debug_recorder::BoneFrameState;
        if !self.enable_bone_sdf || self.bone_transforms.is_empty() {
            return None;
        }
        let (world, inv_world) = self.bone_transforms_snapshot();
        Some(BoneFrameState { world, inv_world })
    }

    /// 現在の辺自然長スケールと初期値の差分を辺インデックス付きで返す。
    /// 変更がなければ None。リプレイ側は set_edge_rest_length_scales で復元する。
    fn current_elastic_scales(&self) -> Option<Vec<crate::debug_recorder::ElasticScaleRecord>> {
        use crate::debug_recorder::ElasticScaleRecord;
        if self.distance_constraints.len() != self.initial_distance_rest_lengths.len() {
            return None;
        }
        // 拘束 -> 元辺の逆引き (せん断対角など辺を持たない拘束は対象外)
        let mut constraint_to_edge = vec![u32::MAX; self.distance_constraints.len()];
        for (edge_idx, &c) in self.original_edge_to_constraint.iter().enumerate() {
            if c < constraint_to_edge.len() && constraint_to_edge[c] == u32::MAX {
                constraint_to_edge[c] = edge_idx as u32;
            }
        }
        let mut changed = Vec::new();
        for (i, (c, &base)) in self
            .distance_constraints
            .iter()
            .zip(&self.initial_distance_rest_lengths)
            .enumerate()
        {
            if base > 1e-9 {
                let scale = c.rest_length / base;
                if (scale - 1.0).abs() > 1e-4 && constraint_to_edge[i] != u32::MAX {
                    changed.push(ElasticScaleRecord {
                        edge_idx: constraint_to_edge[i],
                        scale,
                    });
                }
            }
        }
        if changed.is_empty() {
            None
        } else {
            Some(changed)
        }
    }

    /// 元メッシュ辺のみを返す (自動生成せん断対角を除外)。
    /// mesh_edges / distance_constraints は同順のため constraint_type で判定する。
    fn original_mesh_edges(&self) -> Vec<[u32; 2]> {
        self.mesh_edges
            .iter()
            .zip(&self.distance_constraints)
            .filter(|(_, c)| c.constraint_type == 0)
            .map(|(e, _)| *e)
            .collect()
    }

    /// 元メッシュ辺に対応する初期自然長 (original_mesh_edges と同順)。
    fn original_mesh_rest_lengths(&self) -> Vec<f32> {
        self.initial_distance_rest_lengths
            .iter()
            .zip(&self.distance_constraints)
            .filter(|(_, c)| c.constraint_type == 0)
            .map(|(r, _)| *r)
            .collect()
    }

    /// 2階層記録オプションを設定する
    /// full_stride: フル保存間隔 (1=毎フレーム、従来動作)。ring_size: トリガー時の遡及フル数。
    /// lookahead: トリガー後にフル保存を続けるフレーム数。
    #[allow(clippy::too_many_arguments)]
    pub fn set_debug_recording_options(
        &mut self,
        full_stride: u32,
        ring_size: usize,
        lookahead: u32,
        enable_triggers: bool,
        disp_trigger_mm: f32,
        vel_trigger: f32,
        strain_trigger: f32,
        pair_stats_stride: u32,
    ) {
        self.debug_recorder.set_record_options(crate::debug_recorder::DebugRecordOptions {
            full_stride,
            ring_size,
            lookahead,
            enable_triggers,
            disp_trigger_mm,
            vel_trigger,
            strain_trigger,
            saturate_trigger: true,
            nan_trigger: true,
            config_change_trigger: true,
            pair_stats_stride,
        });
    }

    /// 現在の2階層記録オプションを返す
    pub fn debug_recording_options(&self) -> crate::debug_recorder::DebugRecordOptions {
        self.debug_recorder.record_options().clone()
    }

    /// (保存数, フル数, スタブ数) を返す
    pub fn debug_recording_sparse_info(&self) -> (usize, usize, usize) {
        let total = self.debug_recorder.frame_count();
        let full = self.debug_recorder.full_frame_count();
        (total, full, total.saturating_sub(full))
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
