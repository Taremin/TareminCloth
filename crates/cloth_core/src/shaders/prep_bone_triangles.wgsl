// taremin_cloth ボーン局所三角形準備シェーダー (prep_bone_triangles.wgsl)

struct BoneTriangleSource {
    i0: u32,
    i1: u32,
    i2: u32,
    bone_idx: u32,
    w0: f32,
    w1: f32,
    w2: f32,
    _pad: f32,
}

struct BakeTriangle {
    p0: vec3<f32>,
    w0: f32,
    p1: vec3<f32>,
    w1: f32,
    p2: vec3<f32>,
    w2: f32,
    normal: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> skinned_positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bone_inv_matrices: array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read> tri_sources: array<BoneTriangleSource>;
@group(0) @binding(3) var<storage, read_write> out_triangles: array<BakeTriangle>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let t_idx = id.x;
    if (t_idx >= arrayLength(&tri_sources)) {
        return;
    }

    let src = tri_sources[t_idx];
    let v0_world = skinned_positions[src.i0].xyz;
    let v1_world = skinned_positions[src.i1].xyz;
    let v2_world = skinned_positions[src.i2].xyz;

    let inv_m = bone_inv_matrices[src.bone_idx];
    let p0 = (inv_m * vec4<f32>(v0_world, 1.0)).xyz;
    let p1 = (inv_m * vec4<f32>(v1_world, 1.0)).xyz;
    let p2 = (inv_m * vec4<f32>(v2_world, 1.0)).xyz;

    let e01 = p1 - p0;
    let e02 = p2 - p0;
    let cr = cross(e01, e02);
    let len_sq = dot(cr, cr);
    var norm = vec3<f32>(0.0, 1.0, 0.0);
    if (len_sq > 1e-12) {
        norm = cr * inverseSqrt(len_sq);
    }

    var tri: BakeTriangle;
    tri.p0 = p0;
    tri.w0 = src.w0;
    tri.p1 = p1;
    tri.w1 = src.w1;
    tri.p2 = p2;
    tri.w2 = src.w2;
    tri.normal = norm;
    tri._pad = 0.0;

    out_triangles[t_idx] = tri;
}
