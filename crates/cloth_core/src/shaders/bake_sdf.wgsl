// taremin_cloth ボーンSDF高速ベイクコンピュートシェーダー (bake_sdf.wgsl)

struct BakeParams {
    local_min: vec3<f32>,
    tri_start: u32,
    local_max: vec3<f32>,
    tri_count: u32,
    tile_col: u32,
    tile_row: u32,
    tile_layer: u32,
    res: u32,
    total_width: u32,
    total_height: u32,
    total_depth: u32,
    row_pitch: u32,
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

struct BakeClosestResult {
    closest_pt: vec3<f32>,
    barycentric: vec3<f32>,
}

@group(0) @binding(0) var<storage, read> params_array: array<BakeParams>;
@group(0) @binding(1) var<storage, read> triangles: array<BakeTriangle>;
@group(0) @binding(2) var<storage, read_write> output_voxels: array<u32>;

// Christer Ericson「Real-Time Collision Detection」に基づく点-三角形最短距離および重心座標計算
fn closest_point_on_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> BakeClosestResult {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return BakeClosestResult(a, vec3<f32>(1.0, 0.0, 0.0)); // A頂点領域
    }

    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return BakeClosestResult(b, vec3<f32>(0.0, 1.0, 0.0)); // B頂点領域
    }

    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return BakeClosestResult(a + v * ab, vec3<f32>(1.0 - v, v, 0.0)); // AB辺領域
    }

    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return BakeClosestResult(c, vec3<f32>(0.0, 0.0, 1.0)); // C頂点領域
    }

    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return BakeClosestResult(a + w * ac, vec3<f32>(1.0 - w, 0.0, w)); // AC辺領域
    }

    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return BakeClosestResult(b + w * (c - b), vec3<f32>(0.0, 1.0 - w, w)); // BC辺領域
    }

    // 三角形面領域（内部）
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let u = 1.0 - v - w;
    return BakeClosestResult(a + ab * v + ac * w, vec3<f32>(u, v, w));
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let num_bones = arrayLength(&params_array);
    if (num_bones == 0u) {
        return;
    }

    let res = params_array[0].res;
    let vx = global_id.x;
    let vy = global_id.y;
    let bone_idx = global_id.z / res;
    let vz = global_id.z % res;

    if (bone_idx >= num_bones || vx >= res || vy >= res) {
        return;
    }

    let params = params_array[bone_idx];

    // 1. ボクセル中心座標（ボーンローカル座標系）の算出
    let uvw = (vec3<f32>(f32(vx), f32(vy), f32(vz)) + 0.5) / f32(res);
    let p_local = params.local_min + uvw * (params.local_max - params.local_min);

    var min_dist_sq = 1e10;
    var best_signed_dist = 10.0;
    var best_alpha = 0.0;

    // 2. 三角形群との最短距離探索
    for (var i = 0u; i < params.tri_count; i = i + 1u) {
        let tri = triangles[params.tri_start + i];
        let res_closest = closest_point_on_triangle(p_local, tri.p0, tri.p1, tri.p2);
        let diff = p_local - res_closest.closest_pt;
        let d_sq = dot(diff, diff);

        if (d_sq < min_dist_sq) {
            min_dist_sq = d_sq;
            let d = sqrt(d_sq);

            // 面法線との内積で内外判定（外側: 正, 内側: 負）
            let sign = select(-1.0, 1.0, dot(diff, tri.normal) >= 0.0);
            best_signed_dist = d * sign;

            // 重心座標でボーンウェイトを補間
            best_alpha = res_closest.barycentric.x * tri.w0 + res_closest.barycentric.y * tri.w1 + res_closest.barycentric.z * tri.w2;
        }
    }

    // 3. アトラス 3D テクスチャ内インデックスの計算
    let gx = params.tile_col * res + vx;
    let gy = params.tile_row * res + vy;
    let gz = params.tile_layer * res + vz;

    // Z -> Y -> X 順（C-contiguous: [depth, height, width]）
    let pitch = select(params.total_width, params.row_pitch, params.row_pitch > 0u);
    let flat_idx = (gz * params.total_height + gy) * pitch + gx;

    // Rg16Float (vec2<f32> -> 2 x f16) を 32bit u32 にパックして格納
    output_voxels[flat_idx] = pack2x16float(vec2<f32>(best_signed_dist, best_alpha));
}
