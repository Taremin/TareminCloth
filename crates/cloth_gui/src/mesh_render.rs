use wgpu::util::DeviceExt;
use glam::{Mat4, Vec3};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RenderUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub _pad0: f32,
    pub options: [u32; 4], // [0]: highlight_backface (0/1), [1]: shading_mode, [2]: wireframe_mode, [3]: pad
}

pub struct MeshRenderer {
    pub surface_pipeline: wgpu::RenderPipeline,
    pub wire_pipeline: wgpu::RenderPipeline,
    pub collider_pipeline: wgpu::RenderPipeline,
    pub line_pipeline: wgpu::RenderPipeline,
    pub voxel_pipeline: wgpu::RenderPipeline,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
    pub face_index_buffer: Option<wgpu::Buffer>,
    pub num_face_indices: u32,
    pub wire_index_buffer: Option<wgpu::Buffer>,
    pub num_wire_indices: u32,
    pub collider_vertex_buffer: Option<wgpu::Buffer>,
    pub collider_index_buffer: Option<wgpu::Buffer>,
    pub num_collider_indices: u32,
    pub line_vertex_buffer: Option<wgpu::Buffer>,
    pub num_line_vertices: u32,
    pub voxel_cube_buffer: wgpu::Buffer,
    pub voxel_instance_buffer: Option<wgpu::Buffer>,
    pub num_voxel_instances: u32,
}


impl MeshRenderer {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader_source = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _pad0: f32,
    options: vec4<u32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) velocity: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) velocity: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.world_pos = input.position;
    out.velocity = input.velocity;
    out.clip_pos = uniforms.view_proj * vec4<f32>(input.position, 1.0);
    return out;
}

@fragment
fn fs_surface(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> @location(0) vec4<f32> {
    // スクリーン微分から幾何法線を算出
    let dx = dpdx(in.world_pos);
    let dy = dpdy(in.world_pos);
    var geom_normal = normalize(cross(dx, dy));
    if (!is_front) {
        geom_normal = -geom_normal;
    }

    let light_dir1 = normalize(vec3<f32>(0.5, 0.8, 1.0));
    let light_dir2 = normalize(vec3<f32>(-0.6, -0.4, 0.5));

    let diff1 = max(dot(geom_normal, light_dir1), 0.0);
    let diff2 = max(dot(geom_normal, light_dir2), 0.0) * 0.4;
    let ambient = 0.25;
    let lighting = clamp(diff1 + diff2 + ambient, 0.0, 1.0);

    let highlight_back = uniforms.options.x != 0u;

    var base_color: vec3<f32>;
    if (is_front) {
        // 表側: アイボリー・ライトグレー
        base_color = vec3<f32>(0.92, 0.94, 0.96);
    } else {
        if (highlight_back) {
            // 裏側: 鮮やかなレッド（裏返り・自己交差の強調）
            base_color = vec3<f32>(1.0, 0.15, 0.15);
        } else {
            base_color = vec3<f32>(0.75, 0.78, 0.82);
        }
    }

    return vec4<f32>(base_color * lighting, 1.0);
}

@fragment
fn fs_wire() -> @location(0) vec4<f32> {
    // ワイヤーフレーム色: 濃いダークスレート
    return vec4<f32>(0.15, 0.18, 0.22, 0.9);
}

struct ColliderVertexInput {
    @location(0) position: vec3<f32>,
};

struct ColliderVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn vs_collider(input: ColliderVertexInput) -> ColliderVertexOutput {
    var out: ColliderVertexOutput;
    out.world_pos = input.position;
    out.clip_pos = uniforms.view_proj * vec4<f32>(input.position, 1.0);
    return out;
}

@fragment
fn fs_collider(
    in: ColliderVertexOutput,
    @builtin(front_facing) is_front: bool,
) -> @location(0) vec4<f32> {
    let dx = dpdx(in.world_pos);
    let dy = dpdy(in.world_pos);
    var geom_normal = normalize(cross(dx, dy));
    if (!is_front) {
        geom_normal = -geom_normal;
    }
    let light_dir = normalize(vec3<f32>(0.5, 0.8, 1.0));
    let diff = abs(dot(geom_normal, light_dir));
    let lighting = clamp(diff * 0.7 + 0.3, 0.0, 1.0);

    // コライダー色: スレートブルー・グレー
    let col = vec3<f32>(0.55, 0.63, 0.71);
    return vec4<f32>(col * lighting, 1.0);
}

struct LineVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct LineVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_line(input: LineVertexInput) -> LineVertexOutput {
    var out: LineVertexOutput;
    out.color = input.color;
    out.clip_pos = uniforms.view_proj * vec4<f32>(input.position, 1.0);
    return out;
}

@fragment
fn fs_line(in: LineVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}

struct VoxelVertexInput {
    @location(0) position: vec3<f32>,
};

struct VoxelInstanceInput {
    @location(1) center: vec3<f32>,
    @location(2) size: f32,
    @location(3) color: vec3<f32>,
};

struct VoxelVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
};

@vertex
fn vs_voxel(vert: VoxelVertexInput, inst: VoxelInstanceInput) -> VoxelVertexOutput {
    var out: VoxelVertexOutput;
    let world_p = inst.center + vert.position * inst.size;
    out.world_pos = world_p;
    out.color = inst.color;
    out.clip_pos = uniforms.view_proj * vec4<f32>(world_p, 1.0);
    return out;
}

@fragment
fn fs_voxel(in: VoxelVertexOutput) -> @location(0) vec4<f32> {
    let dx = dpdx(in.world_pos);
    let dy = dpdy(in.world_pos);
    let normal = normalize(cross(dx, dy));
    let light_dir = normalize(vec3<f32>(0.5, 0.8, 1.0));
    let diff = abs(dot(normal, light_dir));
    let lighting = clamp(diff * 0.6 + 0.4, 0.0, 1.0);
    return vec4<f32>(in.color * lighting, 0.9);
}
"#;


        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mesh Render Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Render Uniform Buffer"),
            size: std::mem::size_of::<RenderUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Uniform BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Uniform BindGroup"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // 頂点バッファレイアウト (GpuVertex: stride 48)
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 48,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3, // position
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3, // velocity
                },
            ],
        };

        // 面パイプライン (両面描画, CullMode::None)
        let surface_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Surface Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout.clone()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_surface"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // 両面描画
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ワイヤーフレームパイプライン (LineList)
        let wire_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Wire Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_wire"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: -1,
                    slope_scale: -1.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // コライダー専用パイプライン (位置のみ stride 12, 両面描画)
        let collider_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        };

        let collider_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Collider Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_collider"),
                buffers: &[collider_vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_collider"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // 3Dラインパイプライン (LineList, stride 24: pos 12B + col 12B)
        let line_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 24,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        };

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_line"),
                buffers: &[line_vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_line"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // ラインは深度書き込みオフ
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ボクセルインスタンス描画パイプライン (キューブ 36頂点 + インスタンス stride 28)
        let cube_vertices: [[f32; 3]; 36] = [
            [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5],
            [-0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
            [-0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
            [-0.5,  0.5, -0.5], [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5],
            [-0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5],
            [-0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5],
            [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5],
            [ 0.5, -0.5, -0.5], [ 0.5,  0.5,  0.5], [ 0.5, -0.5,  0.5],
            [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5],
            [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5],
            [-0.5, -0.5, -0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5],
        ];
        let voxel_cube_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Voxel Unit Cube Buffer"),
            contents: bytemuck::cast_slice(&cube_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let voxel_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        };

        let voxel_instance_layout = wgpu::VertexBufferLayout {
            array_stride: 28, // center: 12B, size: 4B, color: 12B
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        };

        let voxel_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Voxel Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_voxel"),
                buffers: &[voxel_vertex_layout, voxel_instance_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_voxel"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            surface_pipeline,
            wire_pipeline,
            collider_pipeline,
            line_pipeline,
            voxel_pipeline,
            uniform_buffer,
            uniform_bind_group,
            face_index_buffer: None,
            num_face_indices: 0,
            wire_index_buffer: None,
            num_wire_indices: 0,
            collider_vertex_buffer: None,
            collider_index_buffer: None,
            num_collider_indices: 0,
            line_vertex_buffer: None,
            num_line_vertices: 0,
            voxel_cube_buffer,
            voxel_instance_buffer: None,
            num_voxel_instances: 0,
        }
    }


    /// メッシュトポロジー（面とエッジ）からインデックスバッファを構築
    pub fn update_mesh_topology(
        &mut self,
        device: &wgpu::Device,
        faces: &[[u32; 3]],
        edges: &[[u32; 2]],
    ) {
        if !faces.is_empty() {
            let mut face_indices: Vec<u32> = Vec::with_capacity(faces.len() * 3);
            for f in faces {
                face_indices.push(f[0]);
                face_indices.push(f[1]);
                face_indices.push(f[2]);
            }
            self.face_index_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mesh Face Index Buffer"),
                contents: bytemuck::cast_slice(&face_indices),
                usage: wgpu::BufferUsages::INDEX,
            }));
            self.num_face_indices = face_indices.len() as u32;
        } else {
            self.face_index_buffer = None;
            self.num_face_indices = 0;
        }

        if !edges.is_empty() {
            let mut wire_indices: Vec<u32> = Vec::with_capacity(edges.len() * 2);
            for e in edges {
                wire_indices.push(e[0]);
                wire_indices.push(e[1]);
            }
            self.wire_index_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mesh Wire Index Buffer"),
                contents: bytemuck::cast_slice(&wire_indices),
                usage: wgpu::BufferUsages::INDEX,
            }));
            self.num_wire_indices = wire_indices.len() as u32;
        } else {
            self.wire_index_buffer = None;
            self.num_wire_indices = 0;
        }
    }

    /// カメラと描画オプションの更新
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        view_proj: Mat4,
        camera_pos: Vec3,
        highlight_backface: bool,
    ) {
        let uniforms = RenderUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _pad0: 0.0,
            options: [
                if highlight_backface { 1 } else { 0 },
                0,
                0,
                0,
            ],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// メッシュ描画パスの実行（vertex_buffer を直接バインド）
    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        vertex_buffer: &'a wgpu::Buffer,
        draw_surface: bool,
        draw_wireframe: bool,
    ) {
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));

        if draw_surface && self.num_face_indices > 0 {
            if let Some(ref ib) = self.face_index_buffer {
                render_pass.set_pipeline(&self.surface_pipeline);
                render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.num_face_indices, 0, 0..1);
            }
        }

        if draw_wireframe && self.num_wire_indices > 0 {
            if let Some(ref ib) = self.wire_index_buffer {
                render_pass.set_pipeline(&self.wire_pipeline);
                render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.num_wire_indices, 0, 0..1);
            }
        }
    }

    /// コライダー形状から頂点・インデックスバッファを構築
    pub fn update_collider_topology(
        &mut self,
        device: &wgpu::Device,
        colliders: &[crate::protocol::GuiColliderData],
        mesh_triangles: &[crate::protocol::GuiMeshTriangleData],
    ) {
        let mut vertices: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        // 1. メッシュコライダー三角形の追加
        for tri in mesh_triangles {
            let base = vertices.len() as u32;
            vertices.push(tri.p0);
            vertices.push(tri.p1);
            vertices.push(tri.p2);
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
        }

        // 2. 解析コライダーの追加
        for col in colliders {
            match col.collider_type {
                0 => {
                    // 球コライダー
                    generate_sphere_mesh(col.point_a, col.radius, 12, 16, &mut vertices, &mut indices);
                }
                1 => {
                    // カプセルコライダー
                    generate_capsule_mesh(col.point_a, col.point_b, col.radius, &mut vertices, &mut indices);
                }
                2 => {
                    // 平面コライダー
                    generate_plane_mesh(col.point_a, col.point_b, 10.0, &mut vertices, &mut indices);
                }
                _ => {}
            }
        }

        if !indices.is_empty() {
            self.collider_vertex_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Collider Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }));
            self.collider_index_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Collider Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            }));
            self.num_collider_indices = indices.len() as u32;
        } else {
            self.collider_vertex_buffer = None;
            self.collider_index_buffer = None;
            self.num_collider_indices = 0;
        }
    }

    /// コライダー描画パスの実行
    pub fn render_colliders<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.num_collider_indices > 0 {
            if let (Some(ref vb), Some(ref ib)) = (&self.collider_vertex_buffer, &self.collider_index_buffer) {
                render_pass.set_pipeline(&self.collider_pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                render_pass.set_vertex_buffer(0, vb.slice(..));
                render_pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..self.num_collider_indices, 0, 0..1);
            }
        }
    }

    /// 3Dライン（SDF BBOX枠線等）の頂点バッファを構築 [p0x..z, p1x..z, r, g, b]
    pub fn update_lines(&mut self, device: &wgpu::Device, extra_lines: &[[f32; 9]]) {
        if !extra_lines.is_empty() {
            let mut line_verts: Vec<f32> = Vec::with_capacity(extra_lines.len() * 12);
            for l in extra_lines {
                let p0 = [l[0], l[1], l[2]];
                let p1 = [l[3], l[4], l[5]];
                let col = [l[6], l[7], l[8]];
                // 頂点0
                line_verts.extend_from_slice(&p0);
                line_verts.extend_from_slice(&col);
                // 頂点1
                line_verts.extend_from_slice(&p1);
                line_verts.extend_from_slice(&col);
            }
            self.line_vertex_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Lines Vertex Buffer"),
                contents: bytemuck::cast_slice(&line_verts),
                usage: wgpu::BufferUsages::VERTEX,
            }));
            self.num_line_vertices = (extra_lines.len() * 2) as u32;
        } else {
            self.line_vertex_buffer = None;
            self.num_line_vertices = 0;
        }
    }

    /// 3Dライン描画パスの実行
    pub fn render_lines<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.num_line_vertices > 0 {
            if let Some(ref vb) = self.line_vertex_buffer {
                render_pass.set_pipeline(&self.line_pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                render_pass.set_vertex_buffer(0, vb.slice(..));
                render_pass.draw(0..self.num_line_vertices, 0..1);
            }
        }
    }

    /// SDF表面ボクセル群のインスタンスバッファを構築 [x, y, z, size, r, g, b]
    pub fn update_voxels(&mut self, device: &wgpu::Device, voxels: &[[f32; 7]]) {
        if !voxels.is_empty() {
            let mut inst_data: Vec<f32> = Vec::with_capacity(voxels.len() * 7);
            for v in voxels {
                inst_data.extend_from_slice(v);
            }
            self.voxel_instance_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Voxel Instances Buffer"),
                contents: bytemuck::cast_slice(&inst_data),
                usage: wgpu::BufferUsages::VERTEX,
            }));
            self.num_voxel_instances = voxels.len() as u32;
        } else {
            self.voxel_instance_buffer = None;
            self.num_voxel_instances = 0;
        }
    }

    /// ボクセルキューブ描画パスの実行
    pub fn render_voxels<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.num_voxel_instances > 0 {
            if let Some(ref inst_buf) = self.voxel_instance_buffer {
                render_pass.set_pipeline(&self.voxel_pipeline);
                render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.voxel_cube_buffer.slice(..));
                render_pass.set_vertex_buffer(1, inst_buf.slice(..));
                render_pass.draw(0..36, 0..self.num_voxel_instances);
            }
        }
    }
}


fn generate_sphere_mesh(
    center: [f32; 3],
    radius: f32,
    rings: u32,
    sectors: u32,
    vertices: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let base_idx = vertices.len() as u32;
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let phi = v * std::f32::consts::PI;
        for s in 0..=sectors {
            let u = s as f32 / sectors as f32;
            let theta = u * std::f32::consts::TAU;
            let x = center[0] + radius * phi.sin() * theta.cos();
            let y = center[1] + radius * phi.sin() * theta.sin();
            let z = center[2] + radius * phi.cos();
            vertices.push([x, y, z]);
        }
    }
    for r in 0..rings {
        for s in 0..sectors {
            let cur = base_idx + r * (sectors + 1) + s;
            let next = cur + sectors + 1;
            indices.push(cur);
            indices.push(next);
            indices.push(cur + 1);

            indices.push(cur + 1);
            indices.push(next);
            indices.push(next + 1);
        }
    }
}

fn generate_plane_mesh(
    point: [f32; 3],
    normal: [f32; 3],
    size: f32,
    vertices: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let base_idx = vertices.len() as u32;
    let n = glam::Vec3::from_array(normal).normalize_or_zero();
    let n = if n.length_squared() < 0.1 { glam::Vec3::Z } else { n };
    let up = if n.z.abs() < 0.99 { glam::Vec3::Z } else { glam::Vec3::X };
    let tangent = n.cross(up).normalize();
    let bitangent = n.cross(tangent).normalize();
    let p = glam::Vec3::from_array(point);

    let half = size * 0.5;
    let v0 = p - tangent * half - bitangent * half;
    let v1 = p + tangent * half - bitangent * half;
    let v2 = p + tangent * half + bitangent * half;
    let v3 = p - tangent * half + bitangent * half;

    vertices.push(v0.to_array());
    vertices.push(v1.to_array());
    vertices.push(v2.to_array());
    vertices.push(v3.to_array());

    indices.push(base_idx);
    indices.push(base_idx + 1);
    indices.push(base_idx + 2);

    indices.push(base_idx);
    indices.push(base_idx + 2);
    indices.push(base_idx + 3);
}

fn generate_capsule_mesh(
    a: [f32; 3],
    b: [f32; 3],
    radius: f32,
    vertices: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    generate_sphere_mesh(a, radius, 8, 12, vertices, indices);
    generate_sphere_mesh(b, radius, 8, 12, vertices, indices);
}
