// 仮想コライダー粒子（Virtual Collision Particles）の前向き座標更新パス
// 親三角形の最新座標から重心座標補間により仮想頂点座標を更新する

struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct GpuVirtualVertexDef {
    parent_indices: vec4<u32>, // x, y, z: 親三角形の3頂点, w: 親face_id
    bary_weights: vec4<f32>,   // x, y, z: 重心座標 (u, v, w), w: thickness
};

struct VirtualForwardParams {
    num_real_vertices: u32,
    num_virtual_vertices: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read> virt_defs: array<GpuVirtualVertexDef>;
@group(0) @binding(2) var<uniform> params: VirtualForwardParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let virt_idx = global_id.x;
    if (virt_idx >= params.num_virtual_vertices) {
        return;
    }

    let def = virt_defs[virt_idx];
    let v0 = def.parent_indices.x;
    let v1 = def.parent_indices.y;
    let v2 = def.parent_indices.z;

    let vert0 = vertices[v0];
    let vert1 = vertices[v1];
    let vert2 = vertices[v2];

    let w0 = def.bary_weights.x;
    let w1 = def.bary_weights.y;
    let w2 = def.bary_weights.z;

    let p_prev = vert0.prev_pos * w0 + vert1.prev_pos * w1 + vert2.prev_pos * w2;
    let p_curr = vert0.position * w0 + vert1.position * w1 + vert2.position * w2;
    let vel = vert0.velocity * w0 + vert1.velocity * w1 + vert2.velocity * w2;
    let thick = def.bary_weights.w;

    let dst_idx = params.num_real_vertices + virt_idx;
    vertices[dst_idx].position = p_curr;
    vertices[dst_idx].prev_pos = p_prev;
    vertices[dst_idx].velocity = vel;
    vertices[dst_idx].inv_mass = 0.0; // 仮想頂点は物理ソルバー反復に影響させない
    vertices[dst_idx].layer_id = vert0.layer_id;
    vertices[dst_idx].thickness = thick;
}
