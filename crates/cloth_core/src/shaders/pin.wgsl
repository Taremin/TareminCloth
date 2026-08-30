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

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read> pin_constraints: array<GpuPinConstraint>;
@group(0) @binding(2) var<uniform> pin_params: PinParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= pin_params.num_pins) {
        return;
    }

    let pin = pin_constraints[index];
    let v_idx = pin.vertex_idx;
    let w = clamp(pin.weight, 0.0, 1.0);

    vertices[v_idx].prev_pos = mix(vertices[v_idx].prev_pos, pin.target_pos, w);
}
