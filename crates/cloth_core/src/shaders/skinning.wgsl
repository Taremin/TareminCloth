// taremin_cloth GPU Linear Blend Skinning (skinning.wgsl)

struct RestVertex {
    pos: vec3<f32>,
    _pad0: f32,
    normal: vec3<f32>,
    _pad1: f32,
    bone_indices: vec4<u32>,
    bone_weights: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> rest_vertices: array<RestVertex>;
@group(0) @binding(1) var<storage, read> bone_skin_matrices: array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read_write> skinned_positions: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&rest_vertices)) {
        return;
    }

    let v = rest_vertices[idx];
    var p = vec3<f32>(0.0);

    for (var i = 0u; i < 4u; i = i + 1u) {
        let w = v.bone_weights[i];
        if (w > 0.0) {
            let b_idx = v.bone_indices[i];
            let m = bone_skin_matrices[b_idx];
            p += w * (m * vec4<f32>(v.pos, 1.0)).xyz;
        }
    }

    skinned_positions[idx] = vec4<f32>(p, 1.0);
}
