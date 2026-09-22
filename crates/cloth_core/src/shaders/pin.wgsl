struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct GpuPinConstraint {
    vertex_idx: u32,
    weight: f32,
    _pad0: f32,
    _pad1: f32,
    target_pos: vec3<f32>,
    _pad2: f32,
};

struct PinParams {
    num_pins: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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
};

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read> pin_constraints: array<GpuPinConstraint>;
@group(0) @binding(2) var<uniform> pin_params: PinParams;
@group(0) @binding(3) var<uniform> params: SimParams;

const EPSILON: f32 = 1e-7;
// ピン基準コンプライアンス: 構造剛性 1000 相当 (mesh.rs の tension 換算と同オーダー)。
// w=0.5 で布の構造拘束と同等に競合するよう基準化している。
const PIN_COMPLIANCE_REF: f32 = 1e-6;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= pin_params.num_pins) {
        return;
    }

    let pin = pin_constraints[index];
    let v_idx = pin.vertex_idx;
    let w = clamp(pin.weight, 0.0, 1.0);

    if (w <= 0.0) {
        return;
    }

    let inv_m = vertices[v_idx].inv_mass;

    if (inv_m <= 0.0) {
        // 完全固定 (w=1 または元質量0): 従来と同一のスナップ
        vertices[v_idx].prev_pos = pin.target_pos;
        return;
    }

    // XPBD コンプライアント拘束 C(x) = |x - target| (距離拘束と同一形式)。
    // 質量・コンプライアンスの両方が w で連続変化し、布拘束との力比べになる。
    let d = vertices[v_idx].prev_pos - pin.target_pos;
    let dist = length(d);
    if (!(dist > EPSILON)) {
        return;
    }
    let n = d / dist;
    let dt = params.gravity.w;
    let compliance = PIN_COMPLIANCE_REF * (1.0 - w) / max(w, 1e-6);
    let alpha = compliance / (dt * dt);
    let corr = -dist / (inv_m + alpha);
    vertices[v_idx].prev_pos += inv_m * corr * n;
}
