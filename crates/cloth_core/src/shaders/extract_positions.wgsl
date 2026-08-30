// 頂点座標抽出シェーダー (GpuVertex から position のみを取り出して連続 f32 配列に書き込む)
struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

@group(0) @binding(0) var<storage, read> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read_write> out_positions: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&vertices)) {
        return;
    }

    let p = vertices[index].position;
    let base = index * 3u;
    out_positions[base] = p.x;
    out_positions[base + 1u] = p.y;
    out_positions[base + 2u] = p.z;
}
