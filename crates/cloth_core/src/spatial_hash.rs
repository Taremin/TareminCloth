use std::sync::Arc;
use wgpu::util::DeviceExt;
use crate::context::GpuContext;
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
    ) {
        let num_blocks = (self.table_size + 255) / 256;
        let vert_workgroups = (num_vertices + 255) / 256;

        // 1. カウンタクリア
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Clear Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&hash.clear);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(num_blocks, 1, 1);
        }

        // 2. 頂点セルカウント
        if num_vertices > 0 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Count Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&hash.count);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(vert_workgroups, 1, 1);
        }

        // 3. ブロック内 Prefix Sum
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Scan Blocks Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&hash.scan_blocks);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(num_blocks, 1, 1);
        }

        // 4. トップレベル Prefix Sum (block_sums)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Scan Top Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&hash.scan_top);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        // 5. ブロックオフセット加算 (最終 cell_starts 作成)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Add Offsets Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&hash.add_offsets);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(num_blocks, 1, 1);
        }

        // 6. 頂点インデックスのスキャッター配置
        if num_vertices > 0 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Scatter Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&hash.scatter);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(vert_workgroups, 1, 1);
        }
    }
}
