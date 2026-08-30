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
    pub _pad: u32,
}

pub struct GpuSpatialHash {
    pub cell_heads_buffer: wgpu::Buffer,
    pub vert_next_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,
    pub clear_pipeline: wgpu::ComputePipeline,
    pub build_pipeline: wgpu::ComputePipeline,
    pub build_bind_group: wgpu::BindGroup,
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

        let cell_heads_init = vec![-1i32; table_size as usize];
        let cell_heads_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SpatialHash Cell Heads Buffer"),
            contents: bytemuck::cast_slice(&cell_heads_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let vert_next_init = vec![-1i32; num_vertices.max(1) as usize];
        let vert_next_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SpatialHash Vert Next Buffer"),
            contents: bytemuck::cast_slice(&vert_next_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let params = SpatialHashParams {
            cell_size,
            table_size,
            num_vertices,
            _pad: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SpatialHash Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SpatialHash Build Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/spatial_hash_build.wgsl").into(),
            ),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SpatialHash Build BGL"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        let build_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SpatialHash Build BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_heads_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vert_next_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SpatialHash Pipeline Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SpatialHash Clear Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("clear_heads"),
            compilation_options: Default::default(),
            cache: None,
        });

        let build_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SpatialHash Build Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("build_grid"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            cell_heads_buffer,
            vert_next_buffer,
            params_buffer,
            clear_pipeline,
            build_pipeline,
            build_bind_group,
            table_size,
            cell_size,
        }
    }
}
