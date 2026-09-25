use std::sync::Arc;
use wgpu::util::DeviceExt;
use crate::context::GpuContext;
use crate::mesh::GpuColliderEdge;
use crate::simulation::pipeline_cache::HashPipelines;

pub const DEFAULT_HASH_TABLE_SIZE: u32 = 32768;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpatialHashParams {
    pub cell_size: f32,
    pub table_size: u32,
    pub num_vertices: u32,
    pub num_blocks: u32,
}

pub struct GpuSpatialHash {
    pub cell_starts_buffer: wgpu::Buffer,
    pub sorted_indices_buffer: wgpu::Buffer,
    pub cell_counts_buffer: wgpu::Buffer,
    pub cell_currents_buffer: wgpu::Buffer,
    pub block_sums_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,

    pub grid_bind_group: wgpu::BindGroup,
    pub table_size: u32,
    pub cell_size: f32,
}

impl GpuSpatialHash {
    /// 空間ハッシュの per-sim リソース (バッファ + BindGroup) を構築する。
    /// 6 本のコンピュートパイプラインは共有キャッシュ由来の `hash` を参照する
    /// (コンパイルは `SharedPipelines::ensure_hash` で遅延・共有化済み)。
    pub fn new(
        context: &Arc<GpuContext>,
        hash_bgl: &wgpu::BindGroupLayout,
        vertex_buffer: &wgpu::Buffer,
        num_vertices: u32,
        cell_size: f32,
        table_size: u32,
    ) -> Self {
        let device = &context.device;
        let num_blocks = (table_size + 255) / 256;

        let cell_counts_init = vec![0u32; table_size as usize];
        let cell_counts_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SpatialGrid Cell Counts Buffer"),
            contents: bytemuck::cast_slice(&cell_counts_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // cell_starts は末尾に総数を保持するため table_size + 1 要素
        let cell_starts_init = vec![0u32; (table_size + 1) as usize];
        let cell_starts_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SpatialGrid Cell Starts Buffer"),
            contents: bytemuck::cast_slice(&cell_starts_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let cell_currents_init = vec![0u32; table_size as usize];
        let cell_currents_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SpatialGrid Cell Currents Buffer"),
            contents: bytemuck::cast_slice(&cell_currents_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sorted_indices_init = vec![0u32; num_vertices.max(1) as usize];
        let sorted_indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SpatialGrid Sorted Indices Buffer"),
            contents: bytemuck::cast_slice(&sorted_indices_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let block_sums_init = vec![0u32; num_blocks.max(1) as usize];
        let block_sums_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SpatialGrid Block Sums Buffer"),
            contents: bytemuck::cast_slice(&block_sums_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let params = SpatialHashParams {
            cell_size,
            table_size,
            num_vertices,
            num_blocks,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SpatialGrid Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SpatialGrid Sort BG"),
            layout: hash_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_starts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cell_currents_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sorted_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: block_sums_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            cell_starts_buffer,
            sorted_indices_buffer,
            cell_counts_buffer,
            cell_currents_buffer,
            block_sums_buffer,
            params_buffer,
            grid_bind_group,
            table_size,
            cell_size,
        }
    }

    /// GPU Counting Sort による空間グリッドの構築（6コンピュートパスを一括ディスパッチ）。
    /// パイプラインは共有キャッシュ由来の `hash` を使用する。
    pub fn dispatch_build(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        num_vertices: u32,
        hash: &HashPipelines,
        prof: &crate::simulation::profile::GpuProfiler,
    ) {
        let num_blocks = (self.table_size + 255) / 256;
        let vert_workgroups = (num_vertices + 255) / 256;

        // 1. カウンタクリア
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Clear Pass"),
                timestamp_writes: prof.writes(prof.enter("hash_clear")),
            });
            cpass.set_pipeline(&hash.clear);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(num_blocks, 1, 1);
        }

        // 2. 頂点セルカウント
        if num_vertices > 0 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Count Pass"),
                timestamp_writes: prof.writes(prof.enter("hash_count")),
            });
            cpass.set_pipeline(&hash.count);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(vert_workgroups, 1, 1);
        }

        // 3. ブロック内 Prefix Sum
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Scan Blocks Pass"),
                timestamp_writes: prof.writes(prof.enter("hash_scan")),
            });
            cpass.set_pipeline(&hash.scan_blocks);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(num_blocks, 1, 1);
        }

        // 4. トップレベル Prefix Sum (block_sums)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Scan Top Pass"),
                timestamp_writes: prof.writes(prof.enter("hash_top")),
            });
            cpass.set_pipeline(&hash.scan_top);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        // 5. ブロックオフセット加算 (最終 cell_starts 作成)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Add Offsets Pass"),
                timestamp_writes: prof.writes(prof.enter("hash_add")),
            });
            cpass.set_pipeline(&hash.add_offsets);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(num_blocks, 1, 1);
        }

        // 6. 頂点インデックスのスキャッター配置
        if num_vertices > 0 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Scatter Pass"),
                timestamp_writes: prof.writes(prof.enter("hash_scatter")),
            });
            cpass.set_pipeline(&hash.scatter);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(vert_workgroups, 1, 1);
        }
    }
}

pub struct GpuEdgeSpatialHash {
    pub cell_starts_buffer: wgpu::Buffer,
    pub sorted_indices_buffer: wgpu::Buffer,
    pub cell_counts_buffer: wgpu::Buffer,
    pub cell_currents_buffer: wgpu::Buffer,
    pub block_sums_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,

    pub grid_bind_group: wgpu::BindGroup,
    pub table_size: u32,
    pub cell_size: f32,
}

impl GpuEdgeSpatialHash {
    pub fn new(
        context: &Arc<GpuContext>,
        edge_hash_bgl: &wgpu::BindGroupLayout,
        edge_buffer: &wgpu::Buffer,
        vertex_buffer: &wgpu::Buffer,
        num_edges: u32,
        cell_size: f32,
        table_size: u32,
    ) -> Self {
        let device = &context.device;
        let num_blocks = (table_size + 255) / 256;

        let cell_counts_init = vec![0u32; table_size as usize];
        let cell_counts_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("EdgeSpatialGrid Cell Counts Buffer"),
            contents: bytemuck::cast_slice(&cell_counts_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let cell_starts_init = vec![0u32; (table_size + 1) as usize];
        let cell_starts_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("EdgeSpatialGrid Cell Starts Buffer"),
            contents: bytemuck::cast_slice(&cell_starts_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let cell_currents_init = vec![0u32; table_size as usize];
        let cell_currents_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("EdgeSpatialGrid Cell Currents Buffer"),
            contents: bytemuck::cast_slice(&cell_currents_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sorted_indices_init = vec![0u32; num_edges.max(1) as usize];
        let sorted_indices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("EdgeSpatialGrid Sorted Indices Buffer"),
            contents: bytemuck::cast_slice(&sorted_indices_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let block_sums_init = vec![0u32; num_blocks.max(1) as usize];
        let block_sums_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("EdgeSpatialGrid Block Sums Buffer"),
            contents: bytemuck::cast_slice(&block_sums_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let params = SpatialHashParams {
            cell_size,
            table_size,
            num_vertices: num_edges,
            num_blocks,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("EdgeSpatialGrid Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("EdgeSpatialGrid Sort BG"),
            layout: edge_hash_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: edge_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_starts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cell_currents_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sorted_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: block_sums_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: vertex_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            cell_starts_buffer,
            sorted_indices_buffer,
            cell_counts_buffer,
            cell_currents_buffer,
            block_sums_buffer,
            params_buffer,
            grid_bind_group,
            table_size,
            cell_size,
        }
    }

    pub fn dispatch_build(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        num_edges: u32,
        hash: &HashPipelines,
        prof: &crate::simulation::profile::GpuProfiler,
    ) {
        let num_blocks = (self.table_size + 255) / 256;
        let edge_workgroups = (num_edges + 255) / 256;

        // 1. カウンタクリア
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("EdgeSpatialGrid Clear Pass"),
                timestamp_writes: prof.writes(prof.enter("edge_hash_clear")),
            });
            cpass.set_pipeline(&hash.clear);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(num_blocks, 1, 1);
        }

        // 2. エッジセルカウント
        if num_edges > 0 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("EdgeSpatialGrid Count Pass"),
                timestamp_writes: prof.writes(prof.enter("edge_hash_count")),
            });
            cpass.set_pipeline(&hash.count);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(edge_workgroups, 1, 1);
        }

        // 3. ブロック内 Prefix Sum
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("EdgeSpatialGrid Scan Blocks Pass"),
                timestamp_writes: prof.writes(prof.enter("edge_hash_scan")),
            });
            cpass.set_pipeline(&hash.scan_blocks);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(num_blocks, 1, 1);
        }

        // 4. トップレベル Prefix Sum (block_sums)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("EdgeSpatialGrid Scan Top Pass"),
                timestamp_writes: prof.writes(prof.enter("edge_hash_top")),
            });
            cpass.set_pipeline(&hash.scan_top);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        // 5. ブロックオフセット加算 (最終 cell_starts 作成)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("EdgeSpatialGrid Add Offsets Pass"),
                timestamp_writes: prof.writes(prof.enter("edge_hash_add")),
            });
            cpass.set_pipeline(&hash.add_offsets);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(num_blocks, 1, 1);
        }

        // 6. エッジインデックスのスキャッター配置
        if num_edges > 0 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("EdgeSpatialGrid Scatter Pass"),
                timestamp_writes: prof.writes(prof.enter("edge_hash_scatter")),
            });
            cpass.set_pipeline(&hash.scatter);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(edge_workgroups, 1, 1);
        }
    }
}

fn hash_coords_collider(coord: [i32; 3], table_size: u32) -> u32 {
    let p1 = 73856093u32;
    let p2 = 19349663u32;
    let p3 = 83492791u32;
    let n = (coord[0] as u32).wrapping_mul(p1)
        ^ (coord[1] as u32).wrapping_mul(p2)
        ^ (coord[2] as u32).wrapping_mul(p3);
    n % table_size
}

pub fn build_collider_edge_grid(
    edges: &[GpuColliderEdge],
    cell_size: f32,
    table_size: u32,
) -> (Vec<u32>, Vec<u32>) {
    let mut cell_counts = vec![0u32; table_size as usize];
    let mut cell_coords = Vec::with_capacity(edges.len());
    for edge in edges {
        let mid = [
            (edge.p0[0] + edge.p1[0]) * 0.5,
            (edge.p0[1] + edge.p1[1]) * 0.5,
            (edge.p0[2] + edge.p1[2]) * 0.5,
        ];
        let cx = (mid[0] / cell_size).floor() as i32;
        let cy = (mid[1] / cell_size).floor() as i32;
        let cz = (mid[2] / cell_size).floor() as i32;
        let h = hash_coords_collider([cx, cy, cz], table_size);
        cell_counts[h as usize] += 1;
        cell_coords.push(h);
    }

    let mut cell_starts = vec![0u32; (table_size + 1) as usize];
    let mut sum = 0u32;
    for i in 0..table_size as usize {
        cell_starts[i] = sum;
        sum += cell_counts[i];
    }
    cell_starts[table_size as usize] = sum;

    let mut cell_currents = cell_starts[0..table_size as usize].to_vec();
    let mut sorted_indices = vec![0u32; edges.len()];
    for (edge_idx, &h) in cell_coords.iter().enumerate() {
        let pos = cell_currents[h as usize];
        cell_currents[h as usize] += 1;
        sorted_indices[pos as usize] = edge_idx as u32;
    }

    (cell_starts, sorted_indices)
}

pub struct GpuColliderEdgeSpatialHash {
    pub collider_edges_buffer: wgpu::Buffer,
    pub cell_starts_buffer: wgpu::Buffer,
    pub sorted_indices_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,
    pub num_edges: u32,
    pub table_size: u32,
    pub cell_size: f32,
}

impl GpuColliderEdgeSpatialHash {
    pub fn new_empty(
        context: &Arc<GpuContext>,
        cell_size: f32,
        table_size: u32,
    ) -> Self {
        let device = &context.device;
        let collider_edges_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Collider Edge Buffer (Empty)"),
            size: std::mem::size_of::<GpuColliderEdge>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cell_starts_init = vec![0u32; (table_size + 1) as usize];
        let cell_starts_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Collider Edge Cell Starts Buffer"),
            contents: bytemuck::cast_slice(&cell_starts_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let sorted_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Collider Edge Sorted Indices Buffer (Empty)"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = SpatialHashParams {
            cell_size,
            table_size,
            num_vertices: 0,
            num_blocks: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Collider Edge SpatialGrid Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            collider_edges_buffer,
            cell_starts_buffer,
            sorted_indices_buffer,
            params_buffer,
            num_edges: 0,
            table_size,
            cell_size,
        }
    }

    /// コライダーエッジリストから空間ハッシュを構築・更新する
    pub fn upload(&mut self, context: &Arc<GpuContext>, edges: &[GpuColliderEdge]) {
        self.num_edges = edges.len() as u32;
        let device = &context.device;
        let queue = &context.queue;

        if edges.is_empty() {
            let cell_starts_init = vec![0u32; (self.table_size + 1) as usize];
            queue.write_buffer(&self.cell_starts_buffer, 0, bytemuck::cast_slice(&cell_starts_init));
            let params = SpatialHashParams {
                cell_size: self.cell_size,
                table_size: self.table_size,
                num_vertices: 0,
                num_blocks: 0,
            };
            queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
            return;
        }

        // 1. エッジバッファの再確保・書き込み
        let req_edge_size = (edges.len() * std::mem::size_of::<GpuColliderEdge>()) as u64;
        if self.collider_edges_buffer.size() < req_edge_size {
            self.collider_edges_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Collider Edge Buffer"),
                size: req_edge_size.max(65536),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(&self.collider_edges_buffer, 0, bytemuck::cast_slice(edges));

        // 2. CPU側で Counting Sort
        let (cell_starts, sorted_indices) = build_collider_edge_grid(edges, self.cell_size, self.table_size);

        // 3. cell_starts の書き込み
        queue.write_buffer(&self.cell_starts_buffer, 0, bytemuck::cast_slice(&cell_starts));

        // 4. sorted_indices バッファの再確保・書き込み
        let req_indices_size = (sorted_indices.len() * 4) as u64;
        if self.sorted_indices_buffer.size() < req_indices_size {
            self.sorted_indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Collider Edge Sorted Indices Buffer"),
                size: req_indices_size.max(65536),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(&self.sorted_indices_buffer, 0, bytemuck::cast_slice(&sorted_indices));

        // 5. params 更新
        let params = SpatialHashParams {
            cell_size: self.cell_size,
            table_size: self.table_size,
            num_vertices: self.num_edges,
            num_blocks: (self.table_size + 255) / 256,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }
}


