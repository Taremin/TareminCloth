// taremin_cloth 階層メッシュSDFベイク Pass1: 粗グリッド (bake_mesh_sdf_hier_coarse_main.wgsl)
//
// bake_mesh_sdf_hier_common.wgsl と連結してコンパイルすること。
// 各粗セルに符号付き距離と最近傍三角形インデックスを格納する。
// Pass2 (密グリッド) が遠近分類と候補三角形継承に使用する。

@group(0) @binding(0) var<uniform> params: BakeMeshParams;
@group(0) @binding(1) var<storage, read> triangles: array<BakeMeshTriangle>;
@group(0) @binding(2) var<storage, read_write> coarse_dist: array<u32>;
@group(0) @binding(3) var<storage, read_write> coarse_tri: array<u32>;

const TOP_K: u32 = 8u;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vx = global_id.x;
    let vy = global_id.y;
    let vz = global_id.z;

    if (vx >= params.width || vy >= params.height || vz >= params.depth) {
        return;
    }

    // 粗ボクセル中心座標（メッシュローカル座標系）の算出
    let inv_dims = vec3<f32>(1.0 / f32(params.width), 1.0 / f32(params.height), 1.0 / f32(params.depth));
    let uvw = (vec3<f32>(f32(vx), f32(vy), f32(vz)) + 0.5) * inv_dims;
    let p_local = params.local_min + uvw * (params.local_max - params.local_min);

    // 上位K近傍三角形を追跡（挿入ソート。K=8固定）
    var best_sq = array<f32, 8>(1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10);
    var best_signed = array<f32, 8>(10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0);
    var best_idx = array<u32, 8>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

    // 三角形群との最短距離探索（legacyと同一式＋top-K追跡）
    for (var i = 0u; i < params.tri_count; i = i + 1u) {
        let tri = triangles[i];
        let closest_pt = closest_point_on_triangle(p_local, tri.p0, tri.p1, tri.p2);
        let diff = p_local - closest_pt;
        let d_sq = dot(diff, diff);

        if (d_sq < best_sq[7]) {
            let d = sqrt(d_sq);
            // 面法線との内積で内外判定（外側: 正, 内側: 負）
            let sign = select(-1.0, 1.0, dot(diff, tri.normal) >= 0.0);
            var ins_d = d * sign;
            var ins_sq = d_sq;
            var ins_idx = i;
            for (var k = 0u; k < 8u; k++) {
                if (ins_sq < best_sq[k]) {
                    let tmp_sq = best_sq[k];
                    let tmp_s = best_signed[k];
                    let tmp_i = best_idx[k];
                    best_sq[k] = ins_sq;
                    best_signed[k] = ins_d;
                    best_idx[k] = ins_idx;
                    ins_sq = tmp_sq;
                    ins_d = tmp_s;
                    ins_idx = tmp_i;
                }
            }
        }
    }

    // Z -> Y -> X 順（C-contiguous: [depth, height, width]）
    let pitch = select(params.width, params.row_pitch, params.row_pitch > 0u);
    let flat_idx = (vz * params.height + vy) * pitch + vx;

    // Rg16Float (vec2<f32> -> 2 x f16) を 32bit u32 にパックして格納 (Alpha=1.0固定)
    coarse_dist[flat_idx] = pack2x16float(vec2<f32>(best_signed[0], 1.0));
    // top-K indexをストライド配置 [flat * K + k]
    for (var k = 0u; k < 8u; k++) {
        coarse_tri[flat_idx * 8u + k] = best_idx[k];
    }
}
