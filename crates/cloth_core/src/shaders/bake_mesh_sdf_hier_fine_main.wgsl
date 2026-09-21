// taremin_cloth 階層メッシュSDFベイク Pass2: 密グリッド (bake_mesh_sdf_hier_fine_main.wgsl)
//
// bake_mesh_sdf_hier_common.wgsl と連結してコンパイルすること。
// Pass1の粗セル（距離＋上位8近傍三角形インデックス、ストライド配置）を参照し、
//   遠: d_coarse - halfdiag >= band の細ボクセルは粗値をそのまま転写（親単独で保証）
//   近: 親±2ステンシル（125セル×top-4 = 最大500候補）をexact計算（AABB早期棄却付き）
// 候補集合が真の最近傍を含まない病理に備え、真のtop-2がeps圏内 (2e-4) で符号が
// 不一致の場合のみ全三角形フォールバック（legacyと同一順序・同一式）する。
// LB証明方式は採用しない（証明の緩さがフォールバック嵐を招き、密メッシュで
// legacyより遅くなるため）。残差リスクはテストでゲートする。

struct HierFineParams {
    coarse_w: u32,
    coarse_h: u32,
    coarse_d: u32,
    band: f32,
    z_offset: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: BakeMeshParams;
@group(0) @binding(1) var<storage, read> triangles: array<BakeMeshTriangle>;
@group(0) @binding(2) var<storage, read> coarse_dist: array<u32>;
@group(0) @binding(3) var<storage, read> coarse_tri: array<u32>;
@group(0) @binding(4) var<uniform> hparams: HierFineParams;
@group(0) @binding(5) var<storage, read_write> output_voxels: array<u32>;

// 1三角形分のexact距離・符号計算（legacy本体と同一式）
fn tri_signed_dist(p: vec3<f32>, tri: BakeMeshTriangle) -> vec3<f32> {
    // 戻り値: (d_sq, signed_dist, valid=1.0)
    let closest_pt = closest_point_on_triangle(p, tri.p0, tri.p1, tri.p2);
    let diff = p - closest_pt;
    let d_sq = dot(diff, diff);
    let sign = select(-1.0, 1.0, dot(diff, tri.normal) >= 0.0);
    return vec3<f32>(d_sq, sqrt(d_sq) * sign, 1.0);
}

// 候補1件の評価と真のtop-2更新
fn eval_candidate(p: vec3<f32>, tri: BakeMeshTriangle, best_sq: ptr<function, f32>, best_signed: ptr<function, f32>, second_sq: ptr<function, f32>, second_signed: ptr<function, f32>) {
    // 案2: AABB早期棄却（真の距離の下界。改善不能なら厳密スキップ）
    let tmin = min(tri.p0, min(tri.p1, tri.p2));
    let tmax = max(tri.p0, max(tri.p1, tri.p2));
    let dvec = max(tmin - p, vec3<f32>(0.0)) + max(p - tmax, vec3<f32>(0.0));
    if (dot(dvec, dvec) >= *best_sq) {
        return;
    }
    let r = tri_signed_dist(p, tri);
    if (r.x < *best_sq) {
        *second_sq = *best_sq;
        *second_signed = *best_signed;
        *best_sq = r.x;
        *best_signed = r.y;
    } else if (r.x < *second_sq) {
        *second_sq = r.x;
        *second_signed = r.y;
    }
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vx = global_id.x;
    let vy = global_id.y;
    let vz_g = global_id.z + hparams.z_offset;

    if (vx >= params.width || vy >= params.height || vz_g >= params.depth) {
        return;
    }

    let size = params.local_max - params.local_min;
    let inv_dims = vec3<f32>(1.0 / f32(params.width), 1.0 / f32(params.height), 1.0 / f32(params.depth));
    let uvw = (vec3<f32>(f32(vx), f32(vy), f32(vz_g)) + 0.5) * inv_dims;
    let p = params.local_min + uvw * size;

    // 粗グリッドの軸毎セル幅（clamp後の実効値。公称H=8hと一致しない場合あり）
    let coarse_dims_f = vec3<f32>(f32(hparams.coarse_w), f32(hparams.coarse_h), f32(hparams.coarse_d));
    let spacing = size / coarse_dims_f;
    let halfdiag = 0.5 * length(spacing);

    // 親粗セル（位置ベース。整数比でなくても一意）
    let parent = clamp(
        vec3<i32>(floor((p - params.local_min) / spacing)),
        vec3<i32>(0),
        vec3<i32>(coarse_dims_f) - vec3<i32>(1),
    );
    let pidx = (u32(parent.z) * hparams.coarse_h + u32(parent.y)) * hparams.coarse_w + u32(parent.x);
    let dc_packed = coarse_dist[pidx];
    let dc = unpack2x16float(dc_packed).x;

    // Z -> Y -> X 順（C-contiguous: [depth, height, width]）
    let pitch = select(params.width, params.row_pitch, params.row_pitch > 0u);
    let flat_idx = (vz_g * params.height + vy) * pitch + vx;

    // 遠方確定: セル内全細ボクセルがbandより遠いことが保証される
    if (dc - halfdiag >= hparams.band) {
        output_voxels[flat_idx] = dc_packed;
        return;
    }

    // 近傍: 親±2ステンシル（125セル×top-8）の候補をexact計算し真のtop-2を追跡
    let lo = vec3<i32>(0);
    let hi = vec3<i32>(coarse_dims_f) - vec3<i32>(1);
    var best_sq = 1e10;
    var best_signed = 10.0;
    var second_sq = 1e10;
    var second_signed = 10.0;
    for (var dz: i32 = -2; dz <= 2; dz++) {
        for (var dy: i32 = -2; dy <= 2; dy++) {
            for (var dx: i32 = -2; dx <= 2; dx++) {
                let c = clamp(parent + vec3<i32>(dx, dy, dz), lo, hi);
                let base = ((u32(c.z) * hparams.coarse_h + u32(c.y)) * hparams.coarse_w + u32(c.x)) * 8u;
                for (var k = 0u; k < 8u; k++) {
                    eval_candidate(p, triangles[coarse_tri[base + k]], &best_sq, &best_signed, &second_sq, &second_signed);
                }
            }
        }
    }

    // 真のtop-2がeps圏内 (2e-4) で符号不一致の場合のみ全三角形フォールバック
    // （エッジ角でのargmin同一性の曖昧さ。legacyと同一順序・同一式で解決）
    var ambiguous = (sqrt(second_sq) - sqrt(best_sq) <= 0.0002);
    if (ambiguous) {
        let s_best = select(-1.0, 1.0, best_signed >= 0.0);
        let s_second = select(-1.0, 1.0, second_signed >= 0.0);
        ambiguous = (s_best != s_second);
    }
    if (ambiguous) {
        best_sq = 1e10;
        best_signed = 10.0;
        for (var i = 0u; i < params.tri_count; i = i + 1u) {
            let r = tri_signed_dist(p, triangles[i]);
            if (r.x < best_sq) {
                best_sq = r.x;
                best_signed = r.y;
            }
        }
    }

    output_voxels[flat_idx] = pack2x16float(vec2<f32>(best_signed, 1.0));
}
