use std::sync::Arc;
use wgpu::util::DeviceExt;
use crate::context::GpuContext;

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

    pub clear_pipeline: wgpu::ComputePipeline,
    pub count_pipeline: wgpu::ComputePipeline,
    pub scan_blocks_pipeline: wgpu::ComputePipeline,
    pub scan_top_pipeline: wgpu::ComputePipeline,
    pub add_offsets_pipeline: wgpu::ComputePipeline,
    pub scatter_pipeline: wgpu::ComputePipeline,

    pub grid_bind_group: wgpu::BindGroup,
    pub table_size: u32,
    pub cell_size: f32,
}

impl GpuSpatialHash {
    pub fn new(
        context: &Arc<GpuContext>,
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SpatialGrid Sort Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/spatial_grid_sort.wgsl").into(),
            ),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SpatialGrid Sort BGL"),
            entries: &[
                // 0: vertices (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: cell_counts (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: cell_starts (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: cell_currents (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: sorted_indices (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: block_sums (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SpatialGrid Sort BG"),
            layout: &bgl,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SpatialGrid Pipeline Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SpatialGrid Clear Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("clear_counts"),
            compilation_options: Default::default(),
            cache: None,
        });

        let count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SpatialGrid Count Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("count_vertices"),
            compilation_options: Default::default(),
            cache: None,
        });

        let scan_blocks_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SpatialGrid Scan Blocks Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scan_blocks"),
            compilation_options: Default::default(),
            cache: None,
        });

        let scan_top_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SpatialGrid Scan Top Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scan_top"),
            compilation_options: Default::default(),
            cache: None,
        });

        let add_offsets_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SpatialGrid Add Offsets Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("add_offsets"),
            compilation_options: Default::default(),
            cache: None,
        });

        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SpatialGrid Scatter Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scatter_indices"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            cell_starts_buffer,
            sorted_indices_buffer,
            cell_counts_buffer,
            cell_currents_buffer,
            block_sums_buffer,
            params_buffer,
            clear_pipeline,
            count_pipeline,
            scan_blocks_pipeline,
            scan_top_pipeline,
            add_offsets_pipeline,
            scatter_pipeline,
            grid_bind_group,
            table_size,
            cell_size,
        }
    }

    /// GPU Counting Sort による空間グリッドの構築（6コンピュートパスを一括ディスパッチ）
    pub fn dispatch_build(&self, encoder: &mut wgpu::CommandEncoder, num_vertices: u32) {
        let num_blocks = (self.table_size + 255) / 256;
        let vert_workgroups = (num_vertices + 255) / 256;

        // 1. カウンタクリア
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Clear Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.clear_pipeline);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(num_blocks, 1, 1);
        }

        // 2. 頂点セルカウント
        if num_vertices > 0 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Count Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.count_pipeline);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(vert_workgroups, 1, 1);
        }

        // 3. ブロック内 Prefix Sum
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Scan Blocks Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.scan_blocks_pipeline);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(num_blocks, 1, 1);
        }

        // 4. トップレベル Prefix Sum (block_sums)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Scan Top Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.scan_top_pipeline);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        // 5. ブロックオフセット加算 (最終 cell_starts 作成)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Add Offsets Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.add_offsets_pipeline);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(num_blocks, 1, 1);
        }

        // 6. 頂点インデックスのスキャッター配置
        if num_vertices > 0 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpatialGrid Scatter Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.scatter_pipeline);
            cpass.set_bind_group(0, &self.grid_bind_group, &[]);
            cpass.dispatch_workgroups(vert_workgroups, 1, 1);
        }
    }
}
