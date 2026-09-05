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
                _pad0: 0,
                _pad1: 0,
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

            // 5. 反復終了後にピン位置を適用 (Grab等)
            if num_pins > 0 {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Final Pin Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pin_pipeline);
                cpass.set_bind_group(0, &self.pin_bind_group, &[]);
                cpass.dispatch_workgroups(pin_workgroups, 1, 1);
            }

            // =========================================================================
            // 6. コライダー衝突 & 自己衝突パス (パイプライン順序適正化)
            //    拘束解決（Distance/Sewing/Pin）によって動かされた頂点の貫通を最終調停
            // =========================================================================
            // 6.1 Collision Constraints Pass (コライダー衝突)
            if has_colliders {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Collision Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.collision_pipeline);
                cpass.set_bind_group(0, &self.collider_bind_group, &[]);
                cpass.dispatch_workgroups(vert_workgroups, 1, 1);
            }

            // 6.2 Edge Collision Constraints Pass (エッジコライダー衝突)
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

            // 6.3 法線計算パス (自己衝突用)
            if self.enable_self_collision {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Normals Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.compute_normals_pipeline);
                cpass.set_bind_group(0, &self.compute_normals_bind_group, &[]);
                cpass.dispatch_workgroups(vert_workgroups, 1, 1);
            }

            // 6.4 GPU空間ハッシュ構築 & 自己衝突パス (トポロジーCCD & 幾何反発)
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

            // 7. Update Vel & Commit Positions Pass
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
}
