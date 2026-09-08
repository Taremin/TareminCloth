// taremin_cloth 単一メッシュSDF高速ベイクコンピュートシェーダー (bake_mesh_sdf.wgsl)

struct BakeMeshParams {
    local_min: vec3<f32>,
    tri_count: u32,
    local_max: vec3<f32>,
    row_pitch: u32,
    width: u32,
    height: u32,
    depth: u32,
    _pad0: u32,
}

struct BakeMeshTriangle {
    p0: vec3<f32>,
    _pad0: f32,
    p1: vec3<f32>,
    _pad1: f32,
    p2: vec3<f32>,
    _pad2: f32,
    normal: vec3<f32>,
    _pad3: f32,
}

@group(0) @binding(0) var<uniform> params: BakeMeshParams;
@group(0) @binding(1) var<storage, read> triangles: array<BakeMeshTriangle>;
@group(0) @binding(2) var<storage, read_write> output_voxels: array<u32>;

// Christer Ericson「Real-Time Collision Detection」に基づく点-三角形最短距離計算
fn closest_point_on_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return a; // A頂点領域
    }

    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return b; // B頂点領域
    }

    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return a + v * ab; // AB辺領域
    }

    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return c; // C頂点領域
    }

    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return a + w * ac; // AC辺領域
    }

    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b); // BC辺領域
    }

    // 三角形面領域（内部）
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return a + ab * v + ac * w;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vx = global_id.x;
    let vy = global_id.y;
    let vz = global_id.z;

    if (vx >= params.width || vy >= params.height || vz >= params.depth) {
        return;
    }

    // ボクセル中心座標（メッシュローカル座標系）の算出
    let inv_dims = vec3<f32>(1.0 / f32(params.width), 1.0 / f32(params.height), 1.0 / f32(params.depth));
    let uvw = (vec3<f32>(f32(vx), f32(vy), f32(vz)) + 0.5) * inv_dims;
    let p_local = params.local_min + uvw * (params.local_max - params.local_min);

    var min_dist_sq = 1e10;
    var best_signed_dist = 10.0;

    // 三角形群との最短距離探索
    for (var i = 0u; i < params.tri_count; i = i + 1u) {
        let tri = triangles[i];
        let closest_pt = closest_point_on_triangle(p_local, tri.p0, tri.p1, tri.p2);
        let diff = p_local - closest_pt;
        let d_sq = dot(diff, diff);

        if (d_sq < min_dist_sq) {
            min_dist_sq = d_sq;
            let d = sqrt(d_sq);

            // 面法線との内積で内外判定（外側: 正, 内側: 負）
            let sign = select(-1.0, 1.0, dot(diff, tri.normal) >= 0.0);
            best_signed_dist = d * sign;
        }
    }

    // Z -> Y -> X 順（C-contiguous: [depth, height, width]）
    let pitch = select(params.width, params.row_pitch, params.row_pitch > 0u);
    let flat_idx = (vz * params.height + vy) * pitch + vx;

    // Rg16Float (vec2<f32> -> 2 x f16) を 32bit u32 にパックして格納 (Alpha=1.0固定)
    output_voxels[flat_idx] = pack2x16float(vec2<f32>(best_signed_dist, 1.0));
}
