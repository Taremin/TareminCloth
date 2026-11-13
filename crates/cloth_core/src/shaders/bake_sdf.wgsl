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

@group(0) @binding(0) var<storage, read> params_array: array<BakeParams>;
@group(0) @binding(1) var<storage, read> triangles: array<BakeTriangle>;
@group(0) @binding(2) var<storage, read_write> output_voxels: array<u32>;

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
    var best_abs_proj = -1.0;
    var best_signed_dist = 10.0;
    var best_alpha = 0.0;

    // 2. 三角形群との最短距離探索
    for (var i = 0u; i < params.tri_count; i = i + 1u) {
        let tri = triangles[params.tri_start + i];

        // Christer Ericson「Real-Time Collision Detection」に基づく点-三角形最短距離および重心座標計算。
        // DX12 WARP + FXC 環境で関数境界および複数早期 return を含む構造体返却が Device Removed (0x887A0005)
        // を誘発するため、単一出口の平坦な分岐構造としてループ内に直接展開する。
        let a = tri.p0;
        let b = tri.p1;
        let c = tri.p2;
        let ab = b - a;
        let ac = c - a;
        let ap = p_local - a;
        let d1 = dot(ab, ap);
        let d2 = dot(ac, ap);

        var closest_pt = a;
        var bary = vec3<f32>(1.0, 0.0, 0.0);

        if (d1 <= 0.0 && d2 <= 0.0) {
            // A頂点領域 (既定値のまま)
        } else {
            let bp = p_local - b;
            let d3 = dot(ab, bp);
            let d4 = dot(ac, bp);
            if (d3 >= 0.0 && d4 <= d3) {
                // B頂点領域
                closest_pt = b;
                bary = vec3<f32>(0.0, 1.0, 0.0);
            } else {
                let vc = d1 * d4 - d3 * d2;
                if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
                    // AB辺領域
                    let v = d1 / (d1 - d3);
                    closest_pt = a + v * ab;
                    bary = vec3<f32>(1.0 - v, v, 0.0);
                } else {
                    let cp = p_local - c;
                    let d5 = dot(ab, cp);
                    let d6 = dot(ac, cp);
                    if (d6 >= 0.0 && d5 <= d6) {
                        // C頂点領域
                        closest_pt = c;
                        bary = vec3<f32>(0.0, 0.0, 1.0);
                    } else {
                        let vb = d5 * d2 - d1 * d6;
                        if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
                            // AC辺領域
                            let w = d2 / (d2 - d6);
                            closest_pt = a + w * ac;
                            bary = vec3<f32>(1.0 - w, 0.0, w);
                        } else {
                            let va = d3 * d6 - d5 * d4;
                            if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
                                // BC辺領域
                                let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                                closest_pt = b + w * (c - b);
                                bary = vec3<f32>(0.0, 1.0 - w, w);
                            } else {
                                // 三角形面領域（内部）
                                let denom = 1.0 / (va + vb + vc);
                                let v = vb * denom;
                                let w = vc * denom;
                                let u = 1.0 - v - w;
                                closest_pt = a + ab * v + ac * w;
                                bary = vec3<f32>(u, v, w);
                            }
                        }
                    }
                }
            }
        }

        let diff = p_local - closest_pt;
        let d_sq = dot(diff, diff);
        let normal_proj = dot(diff, tri.normal);
        let abs_proj = abs(normal_proj);
        let eps_sq = 1e-7 * max(1.0, min_dist_sq);

        // 共有エッジ・共有頂点での同着時は法線射影の絶対値が大きい面を優先 (Baerentzen & Aanaes 2005)
        let is_closer = (d_sq + eps_sq < min_dist_sq);
        let is_tie_better = (abs(d_sq - min_dist_sq) <= eps_sq && abs_proj > best_abs_proj);

        if (is_closer || is_tie_better) {
            if (d_sq < min_dist_sq) {
                min_dist_sq = d_sq;
            }
            best_abs_proj = abs_proj;
            let d = sqrt(d_sq);

            // 面法線との内積で内外判定（外側: 正, 内側: 負）
            var sign = select(-1.0, 1.0, normal_proj >= 0.0);

            // 開放境界エッジ（穴の縁）の外側領域における誤った負符号（内部判定）の防止
            let bmask = u32(tri._pad + 0.5);
            if (bmask != 0u && sign < 0.0) {
                var out_proj = -1.0;
                // bit 0: AB辺 (bary.z == 0)
                if ((bmask & 1u) != 0u && bary.z <= 1e-5) {
                    let edge_out = normalize(cross(tri.p1 - tri.p0, tri.normal));
                    out_proj = max(out_proj, dot(diff, edge_out));
                }
                // bit 1: BC辺 (bary.x == 0)
                if ((bmask & 2u) != 0u && bary.x <= 1e-5) {
                    let edge_out = normalize(cross(tri.p2 - tri.p1, tri.normal));
                    out_proj = max(out_proj, dot(diff, edge_out));
                }
                // bit 2: CA辺 (bary.y == 0)
                if ((bmask & 4u) != 0u && bary.y <= 1e-5) {
                    let edge_out = normalize(cross(tri.p0 - tri.p2, tri.normal));
                    out_proj = max(out_proj, dot(diff, edge_out));
                }
                if (out_proj > 0.0 && out_proj >= abs_proj * 0.25) {
                    sign = 1.0;
                }
            }

            best_signed_dist = d * sign;

            // 重心座標でボーンウェイトを補間
            best_alpha = bary.x * tri.w0 + bary.y * tri.w1 + bary.z * tri.w2;
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
