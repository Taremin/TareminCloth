struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct SpatialHashParams {
    cell_size: f32,
    table_size: u32,
    num_vertices: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read_write> cell_heads: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> vert_next: array<i32>;
@group(0) @binding(3) var<uniform> params: SpatialHashParams;

fn hash_coords(coord: vec3<i32>, table_size: u32) -> u32 {
    let p1 = 73856093u;
    let p2 = 19349663u;
    let p3 = 83492791u;
    let n = (u32(coord.x) * p1) ^ (u32(coord.y) * p2) ^ (u32(coord.z) * p3);
    return n % table_size;
}

@compute @workgroup_size(64)
fn clear_heads(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.table_size) {
        return;
    }
    atomicStore(&cell_heads[index], -1);
}

@compute @workgroup_size(64)
fn build_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_vertices) {
        return;
    }

    let p = vertices[index].prev_pos;
    let cell = vec3<i32>(floor(p / params.cell_size));
    let h = hash_coords(cell, params.table_size);

    let old_head = atomicExchange(&cell_heads[h], i32(index));
    vert_next[index] = old_head;
}
