// taremin_cloth 階層メッシュSDFベイク共通定義 (bake_mesh_sdf_hier_common.wgsl)
//
// bake_mesh_sdf.wgsl (legacy単一pass) と同一の構造体・最近傍計算を共有する。
// legacyファイル自体は回帰基準として凍結するため、こちらに複写して保持する。
// 変更時は両ファイルの等価性を維持すること。

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
