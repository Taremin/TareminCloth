use crate::mesh::{GpuVertex, SimParams};
use super::types::CollisionParams;
use super::GpuClothSimulator;

impl GpuClothSimulator {
    /// シミュレーションの GPU コマンド（外力予測・衝突・拘束解決・速度更新）をエンコーダーに記録する
    pub(crate) fn encode_simulation_steps(&self, encoder: &mut wgpu::CommandEncoder, dt: f32, substeps: u32) {
        if self.num_vertices == 0 {
            return;
        }

        let substep_dt = dt / (substeps as f32);
        // 縫合優先モード: 工程フェーズ中は重力ベクトルをスケール (0=無重力〜1=通常)。
        // WGSL側の変更なし。ラッチ・ランプ状態はホストが update_sewing_priority_from_positions で更新する。
        let priority_scale = self.sewing_priority_scale();
        let params = SimParams {
            gravity: [
                self.gravity[0] * priority_scale,
                self.gravity[1] * priority_scale,
                self.gravity[2] * priority_scale,
                substep_dt,
            ],
            damping: self.damping,
            substeps,
            num_vertices: self.num_vertices,
            num_distance_constraints: self.num_distance_constraints,
            num_bending_constraints: self.num_bending_constraints,
            num_sewing_constraints: self.num_sewing_constraints,
            sewing_compliance: if self.sewing_stiffness >= 5000.0 {
                0.0
            } else {
                1.0 / (self.sewing_stiffness.max(1.0))
            },
            enable_sewing_lock: if self.enable_sewing_lock { 1.0 } else { 0.0 },
        };

        self.context.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );

        // ペア収集の速度ホライゾン用にフレームdtを反映 (queue書き込みのみ、エンコーダ順序に影響なし)
        if self.enable_pair_cache {
            self.write_pair_collect_params(dt);
        }

        let wg_size = self.workgroup_size;
        let vert_workgroups = (self.num_vertices + wg_size - 1) / wg_size;
        let num_pins = self.dynamic_pins.len() as u32;
        let pin_workgroups = (num_pins + wg_size - 1) / wg_size;
        let has_colliders = !self.colliders.is_empty() || !self.mesh_triangles.is_empty() || self.enable_bone_sdf;

        if has_colliders {
            let num_mesh_triangles = self.mesh_triangles.len() as u32;
            let num_clusters = if num_mesh_triangles > 0 { (num_mesh_triangles + 15) / 16 } else { 0 };
            let col_params = CollisionParams {
                num_vertices: self.num_vertices,
                num_colliders: self.colliders.len() as u32,
                num_mesh_triangles,
                num_clusters,
                dt: substep_dt,
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
                bytemuck::bytes_of(&col_params),
            );
        }

        for sub_idx in 0..substeps {
            let is_last_substep = sub_idx + 1 == substeps;
            let should_solve_self_collision = self.enable_self_collision
                && (self.self_collision_substep_interval <= 1
                    || sub_idx % self.self_collision_substep_interval == 0
                    || is_last_substep);

            // XPBDラムダの零化 (サブステップ毎。反復ループ内では蓄積する)
            encoder.clear_buffer(&self.dist_lambda_buffer, 0, None);
            encoder.clear_buffer(&self.bend_lambda_buffer, 0, None);
            encoder.clear_buffer(&self.sew_lambda_buffer, 0, None);

            // 0. 縫合自然長の時間進行 (サブステップ毎に1回。反復数・衝突モード非依存)
            if self.num_sewing_constraints > 0 {
                let sew_workgroups =
                    (self.num_sewing_constraints + wg_size - 1) / wg_size;
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Sew Shrink Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(self.shared.ensure_sew_shrink());
                cpass.set_bind_group(0, &self.sew_shrink_bind_group, &[]);
                cpass.dispatch_workgroups(sew_workgroups, 1, 1);
            }

            // 1. Predict Pass
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Predict Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.shared.predict_pipeline);
                cpass.set_bind_group(0, &self.predict_bind_group, &[]);
                cpass.dispatch_workgroups(vert_workgroups, 1, 1);
            }

            // 拘束解決反復ループ (Solver Iterations)
            for _ in 0..self.solver_iterations {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Solver Iteration Pass"),
                    timestamp_writes: None,
                });

                // 1. Pin Constraints
                if num_pins > 0 {
                    cpass.set_pipeline(&self.shared.pin_pipeline);
                    cpass.set_bind_group(0, &self.pin_bind_group, &[]);
                    cpass.dispatch_workgroups(pin_workgroups, 1, 1);
                }

                // 2. Distance Constraints Projection
                self.dispatch_distance_constraints(&mut cpass, vert_workgroups, wg_size);

                // 3. Sewing Constraints Projection
                cpass.set_pipeline(&self.shared.sewing_pipeline);
                for (color_idx, &count) in self.sew_color_counts.iter().enumerate() {
                    if count > 0 {
                        cpass.set_bind_group(0, &self.sewing_bind_groups[color_idx], &[]);
                        cpass.dispatch_workgroups((count + wg_size - 1) / wg_size, 1, 1);
                    }
                }

                // 4. Bending Constraints Projection (反復ループ内でDistance等と同調解決)
                if !self.bend_color_counts.is_empty() {
                    cpass.set_pipeline(&self.shared.bending_pipeline);
                    for (color_idx, &count) in self.bend_color_counts.iter().enumerate() {
                        if count > 0 {
                            cpass.set_bind_group(0, &self.bending_bind_groups[color_idx], &[]);
                            cpass.dispatch_workgroups((count + wg_size - 1) / wg_size, 1, 1);
                        }
                    }
                }

                // 5. コライダー衝突拘束 (Coupled XPBD: 押し出しと距離拘束の協調収束)
                //    反復ループ内で距離拘束等と同調して解くことで、押し出しによるエッジの過剰伸長を防止
                if has_colliders {
                    cpass.set_pipeline(&self.shared.collision_pipeline);
                    cpass.set_bind_group(0, &self.collider_bind_group, &[]);
                    cpass.dispatch_workgroups(vert_workgroups, 1, 1);
                }

                drop(cpass);

                // mode 2 または 3: 反復ループの各回で自己衝突を実行 (Coupled 同調解決)
                if should_solve_self_collision && (self.coupled_self_collision_mode == 2 || self.coupled_self_collision_mode == 3) {
                    let need_rebuild = sub_idx == 0;
                    self.dispatch_self_collision_passes(encoder, vert_workgroups, wg_size, "In-Loop", need_rebuild, is_last_substep);
                }
            }

        // 5. 反復終了後にピン位置を適用 (Grab等)
        if num_pins > 0 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Final Pin Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.shared.pin_pipeline);
            cpass.set_bind_group(0, &self.pin_bind_group, &[]);
            cpass.dispatch_workgroups(pin_workgroups, 1, 1);
        }

        // =========================================================================
        // 6. エッジコライダー詳細衝突 & 自己衝突パス
        // =========================================================================
        // 6.1 Edge Collision Constraints Pass (エッジコライダー衝突)
        if has_colliders && self.enable_edge_collision {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Edge Collision Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(self.shared.ensure_edge());
            for (color_idx, &count) in self.dist_color_counts.iter().enumerate() {
                if count > 0 {
                    cpass.set_bind_group(0, &self.edge_collision_bind_groups[color_idx], &[]);
                    cpass.dispatch_workgroups((count + wg_size - 1) / wg_size, 1, 1);
                }
            }
        }

        // 6.2 自己衝突パス (mode 0 または 1 の場合: 反復ループ外で1回実行)
        if should_solve_self_collision && (self.coupled_self_collision_mode == 0 || self.coupled_self_collision_mode == 1) {
            let need_rebuild = sub_idx == 0;
            self.dispatch_self_collision_passes(encoder, vert_workgroups, wg_size, "Outer", need_rebuild, is_last_substep);
        }

        // 6.3 Post-Self-Collision Relaxation (mode 1 または 3 の場合: 距離拘束・縫合拘束を再適用してエッジ伸びと隙間を抑制)
        if should_solve_self_collision
            && (self.coupled_self_collision_mode == 1 || self.coupled_self_collision_mode == 3)
            && self.post_collision_relaxation_iters > 0
        {
            for _ in 0..self.post_collision_relaxation_iters {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Post-Collision Relaxation Pass"),
                    timestamp_writes: None,
                });
                self.dispatch_distance_constraints(&mut cpass, vert_workgroups, wg_size);

                // 縫合拘束も同時に適用して、距離拘束による縫合ペアの再開口を防止
                if self.num_sewing_constraints > 0 {
                    cpass.set_pipeline(&self.shared.sewing_pipeline);
                    for (color_idx, &count) in self.sew_color_counts.iter().enumerate() {
                        if count > 0 {
                            cpass.set_bind_group(0, &self.sewing_bind_groups[color_idx], &[]);
                            cpass.dispatch_workgroups((count + wg_size - 1) / wg_size, 1, 1);
                        }
                    }
                }
            }
        }

        // 6.4 Final Sewing Pass (全モード共通: 速度確定の直前にもう一度だけ縫合解決を実行する)。
        // 反復ループ内ではコライダー押し出しが縫合解決の後に実行されるため、
        // 身体上に載った縫合線は閉じた後に押し戻されて開いたままになる。
        // 直前でもう一度閉じ直してから速度を確定することで、自己衝突の有無に
        // かかわらず同じ閉鎖結果になる (Final Pin Pass と同じ配置理由)。
        if self.num_sewing_constraints > 0 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Final Sewing Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.shared.sewing_pipeline);
            for (color_idx, &count) in self.sew_color_counts.iter().enumerate() {
                if count > 0 {
                    cpass.set_bind_group(0, &self.sewing_bind_groups[color_idx], &[]);
                    cpass.dispatch_workgroups((count + wg_size - 1) / wg_size, 1, 1);
                }
            }
        }

            // 7. Update Vel & Commit Positions Pass
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Update Vel Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.shared.update_vel_pipeline);
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

        self.dispatch_dynamic_bone_sdf_update(&mut encoder);
        self.encode_simulation_steps(&mut encoder, dt, substeps);
        self.context.queue.submit(Some(encoder.finish()));

        if self.debug_recorder.is_recording() {
            self.record_current_frame(dt, substeps);
        }
    }

    /// 単一サブステップのみ計算を進める（オンデマンド・サブステップ顕微鏡解析用）
    pub fn step_single_substep(&mut self, dt_sub: f32) {
        if self.num_vertices == 0 {
            return;
        }

        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Single Substep Encoder"),
            },
        );

        self.encode_simulation_steps(&mut encoder, dt_sub, 1);
        self.context.queue.submit(Some(encoder.finish()));
        self.context.device.poll(wgpu::Maintain::Wait);
    }

    /// 頂点座標抽出パスをエンコード (GpuVertex から連続 f32 座標配列へ抽出)
    pub(crate) fn encode_extract_positions(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg_size = self.workgroup_size as u64;
        let vert_workgroups = ((self.num_vertices as u64) + wg_size - 1) / wg_size;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Extract Positions Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.shared.extract_positions_pipeline);
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

    /// 縫合拘束の現在自然長 (GPU側の進行値) を読戻す。空の場合は空ベクトル。
    /// デバッグ記録専用のブロッキング読戻し。順序はGPUバッファ順 (ソート済み)。
    pub fn read_sewing_current_rest_lengths(&self) -> Vec<f32> {
        use crate::mesh::GpuSewingConstraint;
        if self.num_sewing_constraints == 0 {
            return Vec::new();
        }
        let size =
            (self.num_sewing_constraints as u64) * std::mem::size_of::<GpuSewingConstraint>() as u64;
        let data = self.read_storage_buffer(&self.sew_buffer, size);
        bytemuck::cast_slice::<u8, GpuSewingConstraint>(&data)
            .iter()
            .map(|c| c.current_rest_len)
            .collect()
    }

    /// 縫合拘束の現在自然長を復元する (GPUバッファ順の配列)。
    /// 要素数が一致しない場合は何もしない。
    pub fn set_sewing_current_rest_lengths(&mut self, values: &[f32]) {
        use crate::mesh::GpuSewingConstraint;
        if values.len() != self.num_sewing_constraints as usize || values.is_empty() {
            return;
        }
        let size = (values.len() * std::mem::size_of::<GpuSewingConstraint>()) as u64;
        let data = self.read_storage_buffer(&self.sew_buffer, size);
        let mut constraints: Vec<GpuSewingConstraint> =
            bytemuck::cast_slice::<u8, GpuSewingConstraint>(&data).to_vec();
        if constraints.len() != values.len() {
            return;
        }
        for (c, &v) in constraints.iter_mut().zip(values) {
            if v.is_finite() && v >= 0.0 {
                c.current_rest_len = v;
            }
        }
        self.context.queue.write_buffer(
            &self.sew_buffer,
            0,
            bytemuck::cast_slice(&constraints),
        );
    }

    /// 頂点座標および速度ベクトルをGPUバッファに直接書き込み、シミュレーション状態を任意フレームの状態へ復元する
    pub fn set_positions_and_velocities(
        &mut self,
        positions: &[[f32; 3]],
        velocities: Option<&[[f32; 3]]>,
    ) {
        if self.num_vertices == 0 || positions.len() != self.num_vertices as usize {
            return;
        }

        let mut verts = self.initial_vertices.clone();
        for (i, v) in verts.iter_mut().enumerate() {
            v.position = positions[i];
            v.prev_pos = positions[i];
            if let Some(vels) = velocities {
                if i < vels.len() {
                    v.velocity = vels[i];
                }
            } else {
                v.velocity = [0.0, 0.0, 0.0];
            }
        }

        self.context.queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(&verts),
        );
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

    /// ペアキャッシュ統計を取得する (vt_count, ee_count, max_vt, max_ee)。
    /// デバッグ・飽和検出専用のブロッキング読戻し。毎フレーム呼び出しは避けること。
    pub fn get_pair_cache_stats(&self) -> (u32, u32, u32, u32) {
        let (vt_count, ee_count) = self.read_pair_counters();
        (
            vt_count,
            ee_count,
            self.pair_cache_max_vt_pairs,
            self.pair_cache_max_ee_pairs,
        )
    }

    /// ペアカウンタのみを読戻す (内部用)
    fn read_pair_counters(&self) -> (u32, u32) {
        use crate::mesh::PairCounters;
        let size = std::mem::size_of::<PairCounters>() as u64;
        let data = self.read_storage_buffer(&self.pair_counters_buffer, size);
        let counters: &[PairCounters] = bytemuck::cast_slice(&data);
        if counters.is_empty() {
            (0, 0)
        } else {
            (counters[0].vt_count, counters[0].ee_count)
        }
    }

    /// ストレージバッファの先頭 size バイトをブロッキング読戻す (内部用)
    fn read_storage_buffer(&self, src: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let staging = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Readback Staging"),
            size: size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Debug Readback Encoder"),
            },
        );
        if size > 0 {
            encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        }
        self.context.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..size.max(4));
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.context.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver.receive()).unwrap().unwrap();

        let data = slice.get_mapped_range();
        let out = data[..size.min(data.len() as u64) as usize].to_vec();
        drop(data);
        staging.unmap();
        out
    }

    /// 収集中のアクティブペア一覧を読戻す (vt_count, ee_count, vt_pairs, ee_pairs)。
    ///
    /// vt/ee_count は受理候補総数 (枠超過分を含む)。pairs は頂点メジャー配置の
    /// 有効ペア (センチネル除外) のみ。各ペアは4頂点ID。
    /// デバッグ専用のブロッキング読戻し。毎フレーム呼び出しは避け、
    /// watchヒット時や監査サンプリングなどに限定すること。
    pub fn get_active_pairs(&self) -> (u32, u32, Vec<[u32; 4]>, Vec<[u32; 4]>) {
        use crate::mesh::{GpuEePair, GpuVtPair};
        const SENTINEL: u32 = 0xFFFFFFFF;
        let (vt_count, ee_count) = self.read_pair_counters();
        let qvt = self.pair_cache_quota_vt.clamp(1, super::PAIR_QUOTA_MAX);
        let qee = self.pair_cache_quota_ee.clamp(1, super::PAIR_QUOTA_MAX);
        let vt_slots = (self.num_vertices * qvt) as usize;
        let ee_slots = (self.num_vertices * qee) as usize;

        let vt_pairs = if vt_slots == 0 {
            Vec::new()
        } else {
            let raw = self.read_storage_buffer(
                &self.active_vt_pairs_buffer,
                (vt_slots * std::mem::size_of::<GpuVtPair>()) as u64,
            );
            bytemuck::cast_slice::<u8, GpuVtPair>(&raw)
                .iter()
                .filter(|p| p.vert_i != SENTINEL)
                .map(|p| [p.vert_i, p.vert_j, p.v0, p.v1])
                .collect()
        };
        let ee_pairs = if ee_slots == 0 {
            Vec::new()
        } else {
            let raw = self.read_storage_buffer(
                &self.active_ee_pairs_buffer,
                (ee_slots * std::mem::size_of::<GpuEePair>()) as u64,
            );
            bytemuck::cast_slice::<u8, GpuEePair>(&raw)
                .iter()
                .filter(|p| p.v_i != SENTINEL)
                .map(|p| [p.v_i, p.v_ui, p.v_j, p.v_vj])
                .collect()
        };
        (vt_count, ee_count, vt_pairs, ee_pairs)
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
        self.sewing_priority_latched = false;
        self.sewing_priority_frame = 0;
        self.sewing_priority_ramp_t = 0.0;
        self.sewing_priority_ratio = 0.0;
        self.context.queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(&self.initial_vertices),
        );
        self.reset_edge_rest_lengths();
        self.buffered_frame_count = 0;
        self.debug_recorder.clear();
    }

    /// 距離拘束（Atomic Jacobi または Coloring）を単一ComputePass内でディスパッチする
    pub(crate) fn dispatch_distance_constraints(
        &self,
        cpass: &mut wgpu::ComputePass,
        vert_workgroups: u32,
        wg_size: u32,
    ) {
        if self.solver_mode == 1 {
            // Atomic Jacobi モード (全エッジを単一ディスパッチで一斉評価 + 頂点変位平均適用)
            if self.num_distance_constraints > 0 {
                let edge_workgroups = (self.num_distance_constraints + wg_size - 1) / wg_size;
                cpass.set_pipeline(&self.shared.ensure_atomic().solve);
                cpass.set_bind_group(0, &self.distance_atomic_bind_group, &[]);
                cpass.dispatch_workgroups(edge_workgroups, 1, 1);
            }
            cpass.set_pipeline(&self.shared.ensure_atomic().apply);
            cpass.set_bind_group(0, &self.distance_atomic_bind_group, &[]);
            cpass.dispatch_workgroups(vert_workgroups, 1, 1);
        } else {
            // Coloring モード (全色グループを連続ディスパッチ)
            cpass.set_pipeline(&self.shared.distance_pipeline);
            for (color_idx, &count) in self.dist_color_counts.iter().enumerate() {
                if count > 0 {
                    cpass.set_bind_group(0, &self.distance_bind_groups[color_idx], &[]);
                    cpass.dispatch_workgroups((count + wg_size - 1) / wg_size, 1, 1);
                }
            }
        }
    }

    /// 自己衝突パス一式（法線計算・空間ハッシュクリア＆構築・自己衝突Solve＆Apply）をディスパッチする
    pub(crate) fn dispatch_self_collision_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        vert_workgroups: u32,
        _wg_size: u32,
        prefix: &str,
        need_rebuild: bool,
        is_last_substep: bool,
    ) {
        // 1. Compute Normals Pass
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{prefix} Compute Normals Pass")),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.shared.ensure_self_collision().normals);
            cpass.set_bind_group(0, &self.compute_normals_bind_group, &[]);
            cpass.dispatch_workgroups(vert_workgroups, 1, 1);
        }

        // 最終サブステップ直進フォールバック: 速度確定前のトンネリングを新鮮なBroadphaseで遮断。
        // Narrow置換方式のためディスパッチ増は +1構築+1フルSolve/frame に限定される。
        let force_direct = self.enable_pair_cache
            && self.enable_pair_cache_final_fallback
            && is_last_substep;

        if !self.enable_pair_cache || need_rebuild || force_direct {
            // 2. SpatialGrid GPU Counting Sort (Clear -> Count -> ScanBlocks -> ScanTop -> AddOffsets -> Scatter)
            self.spatial_hash.dispatch_build(encoder, self.num_vertices, self.shared.ensure_hash());
        }

        if self.enable_pair_cache && !force_direct {
            // =========================================================================
            // 接触候補ペアキャッシュ方式 (Active Pair Caching / I-Cloth 方式)
            // =========================================================================
            if need_rebuild {
                // Step A: カウンターバッファのクリア (vt_count = 0, ee_count = 0)
                encoder.clear_buffer(&self.pair_counters_buffer, 0, None);

                // Step B: ブロードフェーズ (接近ペアの検出・収集)
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&format!("{prefix} Pair Collect Pass")),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.shared.ensure_pair().collect);
                    cpass.set_bind_group(0, &self.pair_collect_bind_group, &[]);
                    cpass.dispatch_workgroups(vert_workgroups, 1, 1);
                }
            }

            // Step C: ナローフェーズ (収集されたペアのみピンポイント解決)
            // 頂点メジャー固定スロット配置 (N*quota) を直接ディスパッチする。
            // 未使用枠はセンチネルで早期リターンするため、間接ディスパッチ
            // (DispatchIndirect) を避けた直接ディスパッチで足りる。
            let vt_slots = self.num_vertices * self.pair_cache_quota_vt.clamp(1, super::PAIR_QUOTA_MAX);
            let ee_slots = self.num_vertices * self.pair_cache_quota_ee.clamp(1, super::PAIR_QUOTA_MAX);
            let vt_workgroups = ((vt_slots + 63) / 64).max(1);
            let ee_workgroups = ((ee_slots + 63) / 64).max(1);

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("{prefix} Pair Solve Narrowphase Pass")),
                    timestamp_writes: None,
                });

                // V-T ペア解決
                cpass.set_pipeline(&self.shared.ensure_pair().solve_vt);
                cpass.set_bind_group(0, &self.pair_solve_vt_bind_group, &[]);
                cpass.dispatch_workgroups(vt_workgroups, 1, 1);

                // E-E ペア解決
                cpass.set_pipeline(&self.shared.ensure_pair().solve_ee);
                cpass.set_bind_group(0, &self.pair_solve_ee_bind_group, &[]);
                cpass.dispatch_workgroups(ee_workgroups, 1, 1);
            }

            // Step D: 変位の適用 (既存の self_collision_apply_pipeline を共用)
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("{prefix} Pair Self Collision Apply Pass")),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.shared.ensure_self_collision().apply);
                cpass.set_bind_group(0, &self.self_collision_apply_bind_group, &[]);
                cpass.dispatch_workgroups(vert_workgroups, 1, 1);
            }
        } else {
            // 従来のダイレクト走査方式
            // 4. Self Collision Pass (Solve)
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("{prefix} Self Collision Pass")),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.shared.ensure_self_collision().solve);
                cpass.set_bind_group(0, &self.self_collision_bind_group, &[]);
                cpass.dispatch_workgroups(vert_workgroups, 1, 1);
            }

            // 5. Self Collision Apply Pass
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("{prefix} Self Collision Apply Pass")),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.shared.ensure_self_collision().apply);
                cpass.set_bind_group(0, &self.self_collision_apply_bind_group, &[]);
                cpass.dispatch_workgroups(vert_workgroups, 1, 1);
            }
        }
    }
}

