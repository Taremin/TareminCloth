// ドレープガイド（Wrinkle Field / Drape Guide）拘束シェーダー
// ボーン円柱座標系をUV展開した2D-SDFテクスチャに基づき、
// 山領域への外向き押し出し（+r方向）と谷領域への内向き引き込み（-r方向）を適用します。
// （U軸のRepeatサンプラーにより円周全周が完全シームレス、2D平面上で"Y"字分岐や複数段シワも自由自在）

struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct WrinkleFieldParams {
    bone_origin: vec4<f32>,     // xyz: origin, w: influence_radius
    bone_axis: vec4<f32>,       // xyz: axis, w: bone_radius
    bone_normal: vec4<f32>,     // xyz: normal, w: stiffness
    bone_binormal: vec4<f32>,   // xyz: binormal, w: blend_weight
    z_range: vec2<f32>,         // x: z_min, y: z_max (ボーン長手方向のUVマッピング範囲)
    r_range: vec2<f32>,         // x: r_min, y: r_max (径方向のターゲット正規化範囲)
    enabled: u32,               // 有効フラグ (0 or 1)
    use_texture: u32,           // 2Dテクスチャモードフラグ (0 or 1)
    valley_window: f32,         // 谷の径方向グラデーション窓 (m)
    crest_window: f32,          // 山の径方向グラデーション窓 (m)
};

struct SimParams {
    gravity: vec4<f32>, // xyz: gravity vector, w: dt
    damping: f32,
    substeps: u32,
    num_vertices: u32,
    num_distance_constraints: u32,
    num_bending_constraints: u32,
    num_sewing_constraints: u32,
    sewing_compliance: f32,
    enable_sewing_lock: f32,
    sewing_lock_distance: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(1) var wrinkle_tex: texture_2d<f32>;
@group(0) @binding(2) var wrinkle_sampler: sampler;
@group(0) @binding(3) var<uniform> wrinkle_params: WrinkleFieldParams;
@group(0) @binding(4) var<uniform> params: SimParams;

const TWO_PI: f32 = 6.283185307179586;
const EPSILON: f32 = 1e-7;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_vertices || wrinkle_params.enabled == 0u) {
        return;
    }

    let inv_m = vertices[idx].inv_mass;
    if (inv_m <= 0.0) {
        return; // 固定ピンは変更しない
    }

    let p = vertices[idx].prev_pos;

    // 1. ボーン円柱座標系へ射影 (theta, z, r)
    let diff = p - wrinkle_params.bone_origin.xyz;
    let axis = wrinkle_params.bone_axis.xyz;
    let z = dot(diff, axis);
    let r_vec = diff - axis * z;
    let r = length(r_vec);

    let x_cross = dot(r_vec, wrinkle_params.bone_normal.xyz);
    let y_cross = dot(r_vec, wrinkle_params.bone_binormal.xyz);
    var theta = atan2(y_cross, x_cross);
    if (theta < 0.0) {
        theta += TWO_PI;
    }

    // 2. 円柱UV展開座標 (u, v) の計算
    // u: 円周方向 (0.0 〜 1.0, Repeatサンプラーで左右境界は完全シームレス)
    let u = (theta / TWO_PI) % 1.0;
    // v: ボーン長手方向 (z_min 〜 z_max を 0.0 〜 1.0 に正規化)
    let z_len = max(wrinkle_params.z_range.y - wrinkle_params.z_range.x, EPSILON);
    let v = clamp((z - wrinkle_params.z_range.x) / z_len, 0.0, 1.0);

    // 3. 2Dテクスチャサンプリング (バイリニア補間)
    let tex_sample = textureSampleLevel(wrinkle_tex, wrinkle_sampler, vec2<f32>(u, v), 0.0);
    // R: 谷のポテンシャル強度 (0.0〜1.0)
    // G: 山のポテンシャル強度 (0.0〜1.0)
    // B: 谷の目標半径 (0.0〜1.0 -> r_min〜r_max)
    // A: 山の目標半径 (0.0〜1.0 -> r_min〜r_max)
    let valley_w = tex_sample.r;
    let crest_w = tex_sample.g;
    let r_len = max(wrinkle_params.r_range.y - wrinkle_params.r_range.x, EPSILON);
    let target_valley_r = wrinkle_params.r_range.x + tex_sample.b * r_len;
    let target_crest_r = wrinkle_params.r_range.x + tex_sample.a * r_len;

    let bone_r = wrinkle_params.bone_axis.w;
    let stiffness = wrinkle_params.bone_normal.w;
    let blend = wrinkle_params.bone_binormal.w;

    let dt = params.gravity.w;

    // 4. 谷の内向き外力加速度 (-r 方向の風・引力)
    var a_pull = 0.0;
    if (valley_w > 0.0) {
        let v_radius = max(target_valley_r, bone_r + 0.002);
        let r_margin = max(0.0, r - v_radius);
        let v_win = max(wrinkle_params.valley_window, 0.001);
        let dist_factor = clamp(r_margin / v_win, 0.0, 1.0);
        a_pull = valley_w * blend * stiffness * dist_factor;
    }

    // 5. 山の外向き外力加速度 (+r 方向の風)
    var a_push = 0.0;
    if (crest_w > 0.0) {
        let r_excess = max(0.0, target_crest_r - r);
        let c_win = max(wrinkle_params.crest_window, 0.001);
        let dist_factor = clamp(r_excess / c_win, 0.0, 1.0);
        a_push = crest_w * blend * stiffness * dist_factor;
    }

    // 6. 合成外力加速度の積分 (速度 v と予測位置 prev_pos への反映)
    let a_net = a_push - a_pull;
    if (abs(a_net) > EPSILON) {
        var r_dir = vec3<f32>(0.0, 0.0, 0.0);
        if (r > EPSILON) {
            r_dir = r_vec / r;
        } else {
            r_dir = wrinkle_params.bone_normal.xyz;
        }
        let dv = r_dir * (a_net * dt);
        vertices[idx].velocity += dv;
        vertices[idx].prev_pos += dv * dt;
    }
}
