struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct NormalParams {
    num_vertices: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read> star_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> star_indices: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> normals: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: NormalParams;

const EPSILON: f32 = 1e-7;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_vertices) {
        return;
    }

    let p_i = vertices[index].position;
    let start = star_offsets[index];
    let end = star_offsets[index + 1u];

    var n_sum = vec3<f32>(0.0, 0.0, 0.0);

    for (var k = start; k < end; k = k + 1u) {
        let pair = star_indices[k];
        let p_a = vertices[pair.x].position;
        let p_b = vertices[pair.y].position;

        let e1 = p_a - p_i;
        let e2 = p_b - p_i;
        let face_n = cross(e1, e2);
        n_sum = n_sum + face_n;
    }

    let len = length(n_sum);
    if (len > EPSILON) {
        normals[index] = vec4<f32>(n_sum / len, 0.0);
    } else {
        // 三角形が存在しない頂点（ワイヤー等）はデフォルト上向き
        normals[index] = vec4<f32>(0.0, 0.0, 1.0, 0.0);
    }
}
