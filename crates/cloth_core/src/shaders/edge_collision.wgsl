struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct GpuDistanceConstraint {
    v0: u32,
    v1: u32,
    rest_length: f32,
    tension_compliance: f32,
    compression_compliance: f32,
    constraint_type: u32,
    _pad0: f32,
    _pad1: f32,
};

struct GpuCollider {
    collider_type: u32, // 0: Sphere, 1: Capsule, 2: Plane
    friction: f32,
    restitution: f32,
    _pad0: f32,
    point_a: vec3<f32>,
    radius: f32,
    point_b: vec3<f32>,
    _pad1: f32,
};

struct GpuMeshTriangle {
    p0: vec3<f32>,
    friction: f32,
    p1: vec3<f32>,
    thickness: f32,
    p2: vec3<f32>,
    restitution: f32,
    flags: u32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

struct CollisionParams {
    num_vertices: u32,
    num_colliders: u32,
    num_mesh_triangles: u32,
    dt: f32,
    edge_margin_scale: f32,
    edge_margin_offset: f32,
    _pad0: f32,
    _pad1: f32,
};

struct DispatchInfo {
    color_offset: u32,
    color_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read> constraints: array<GpuDistanceConstraint>;
@group(0) @binding(2) var<storage, read> colliders: array<GpuCollider>;
@group(0) @binding(3) var<storage, read> mesh_triangles: array<GpuMeshTriangle>;
@group(0) @binding(4) var<storage, read> mesh_bounds: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> params: CollisionParams;
@group(0) @binding(6) var<uniform> dispatch_info: DispatchInfo;

const EPSILON: f32 = 1e-7;

// 点から三角形への最近傍点
fn closest_point_on_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return a;
    }

    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return a + v * ab;
    }

    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return a + w * ac;
    }

    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return a + ab * v + ac * w;
}

// 点 pt から線分 p0 -> p1 上の最近傍点パラメータ t in [0, 1]
fn closest_t_on_segment(pt: vec3<f32>, p0: vec3<f32>, p1: vec3<f32>) -> f32 {
    let u = p1 - p0;
    let u2 = dot(u, u);
    if (u2 < EPSILON) {
        return 0.5;
    }
    return clamp(dot(pt - p0, u) / u2, 0.0, 1.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let local_idx = global_id.x;
    if (local_idx >= dispatch_info.color_count) {
        return;
    }

    let constraint_idx = dispatch_info.color_offset + local_idx;
    let c = constraints[constraint_idx];

    let w0 = vertices[c.v0].inv_mass;
    let w1 = vertices[c.v1].inv_mass;
    let w_sum = w0 + w1;
    if (w_sum <= EPSILON) {
        return;
    }

    var prev0 = vertices[c.v0].prev_pos;
    var prev1 = vertices[c.v1].prev_pos;

    let edge_scale = max(params.edge_margin_scale, 1.0);
    let edge_offset = max(params.edge_margin_offset, 0.0);
    let thickness = max(vertices[c.v0].thickness, vertices[c.v1].thickness) * edge_scale + edge_offset;
    let inv_w_sum = 1.0 / w_sum;

    // 1. プリミティブコライダー判定 (中点判定)
    let mid_point = (prev0 + prev1) * 0.5;
    for (var i = 0u; i < params.num_colliders; i = i + 1u) {
        let col = colliders[i];
        if (col.collider_type == 0u) {
            let center = col.point_a;
            let target_r = col.radius + thickness;
            let delta = mid_point - center;
            let dist = length(delta);
            if (dist < target_r) {
                var normal = vec3<f32>(0.0, 0.0, 1.0);
                if (dist > EPSILON) {
                    normal = delta / dist;
                }
                let push = normal * (target_r - dist);
                let factor0 = 2.0 * w0 * inv_w_sum;
                let factor1 = 2.0 * w1 * inv_w_sum;
                prev0 += push * factor0;
                prev1 += push * factor1;
            }
        } else if (col.collider_type == 1u) {
            let a = col.point_a;
            let b = col.point_b;
            let target_r = col.radius + thickness;
            let ab = b - a;
            let l2 = dot(ab, ab);
            var t = 0.0;
            if (l2 > EPSILON) {
                t = clamp(dot(mid_point - a, ab) / l2, 0.0, 1.0);
            }
            let closest = a + ab * t;
            let delta = mid_point - closest;
            let dist = length(delta);
            if (dist < target_r) {
                var normal = vec3<f32>(0.0, 0.0, 1.0);
                if (dist > EPSILON) {
                    normal = delta / dist;
                }
                let push = normal * (target_r - dist);
                let factor0 = 2.0 * w0 * inv_w_sum;
                let factor1 = 2.0 * w1 * inv_w_sum;
                prev0 += push * factor0;
                prev1 += push * factor1;
            }
        } else if (col.collider_type == 2u) {
            let plane_pt = col.point_a;
            let plane_n = normalize(col.point_b);
            let dist = dot(mid_point - plane_pt, plane_n);
            if (dist < thickness) {
                let push = plane_n * (thickness - dist);
                let factor0 = 2.0 * w0 * inv_w_sum;
                let factor1 = 2.0 * w1 * inv_w_sum;
                prev0 += push * factor0;
                prev1 += push * factor1;
            }
        }
    }

    // 2. メッシュコライダー判定 (中点判定 + コライダー頂点最近傍点判定)
    for (var i = 0u; i < params.num_mesh_triangles; i = i + 1u) {
        let b = mesh_bounds[i];
        let mid = (prev0 + prev1) * 0.5;
        let diff = mid - b.xyz;
        let edge_len = length(prev1 - prev0);
        let bound_r = b.w + thickness + edge_len * 0.5;
        if (dot(diff, diff) > bound_r * bound_r) {
            continue; // エッジが三角形の近傍球外にある場合はスキップ
        }

        let tri = mesh_triangles[i];
        let target_dist = thickness + tri.thickness * edge_scale;

        let cross_prod = cross(tri.p1 - tri.p0, tri.p2 - tri.p0);
        let cross_len = length(cross_prod);
        if (cross_len < EPSILON) {
            continue;
        }
        let face_normal = cross_prod / cross_len;

        // (A) エッジ中点からコライダー面への最近傍押し戻し (弦の沈み込み防止)
        let q_mid = closest_point_on_triangle(mid, tri.p0, tri.p1, tri.p2);
        let delta_mid = mid - q_mid;
        let dist_mid = length(delta_mid);
        let signed_dist_mid = dot(delta_mid, face_normal);

        let is_single_sided = (tri.flags & 1u) != 0u;
        if (is_single_sided) {
            if (signed_dist_mid >= 0.0) {
                if (dist_mid < target_dist) {
                    var normal = face_normal;
                    if (dist_mid > EPSILON) {
                        normal = delta_mid / dist_mid;
                    }
                    let push = normal * (target_dist - dist_mid);
                    let factor0 = 2.0 * w0 * inv_w_sum;
                    let factor1 = 2.0 * w1 * inv_w_sum;
                    prev0 += push * factor0;
                    prev1 += push * factor1;
                }
            } else {
                let max_recovery_depth = bound_r;
                if (dist_mid < max_recovery_depth) {
                    let normal = face_normal;
                    let push = normal * (target_dist - signed_dist_mid);
                    let factor0 = 2.0 * w0 * inv_w_sum;
                    let factor1 = 2.0 * w1 * inv_w_sum;
                    prev0 += push * factor0;
                    prev1 += push * factor1;
                }
            }
        } else {
            if (dist_mid < target_dist && signed_dist_mid >= -target_dist) {
                var normal = face_normal;
                if (dist_mid > EPSILON && signed_dist_mid > 0.0) {
                    normal = delta_mid / dist_mid;
                }
                let push = normal * (target_dist - dist_mid);
                let factor0 = 2.0 * w0 * inv_w_sum;
                let factor1 = 2.0 * w1 * inv_w_sum;
                prev0 += push * factor0;
                prev1 += push * factor1;
            }
        }

        // (B) コライダーの各頂点 (p0, p1, p2) からエッジへの最近傍押し戻し (尖った角の貫通防止)
        // p0 に対して
        {
            let t0 = closest_t_on_segment(tri.p0, prev0, prev1);
            let pt_on_edge = prev0 + t0 * (prev1 - prev0);
            let delta_p0 = pt_on_edge - tri.p0;
            let dist_p0 = length(delta_p0);
            let signed_p0 = dot(delta_p0, face_normal);
            if (dist_p0 < target_dist && signed_p0 >= -target_dist) {
                var normal = face_normal;
                if (dist_p0 > EPSILON && signed_p0 > 0.0) {
                    normal = delta_p0 / dist_p0;
                }
                let push = normal * (target_dist - dist_p0);
                let a = 1.0 - t0;
                let b = t0;
                let denom = a * a * w0 + b * b * w1;
                if (denom > EPSILON) {
                    let d0 = (a * w0 / denom) * push;
                    let d1 = (b * w1 / denom) * push;
                    prev0 += d0;
                    prev1 += d1;
                }
            }
        }
        // p1 に対して
        {
            let t1 = closest_t_on_segment(tri.p1, prev0, prev1);
            let pt_on_edge = prev0 + t1 * (prev1 - prev0);
            let delta_p1 = pt_on_edge - tri.p1;
            let dist_p1 = length(delta_p1);
            let signed_p1 = dot(delta_p1, face_normal);
            if (dist_p1 < target_dist && signed_p1 >= -target_dist) {
                var normal = face_normal;
                if (dist_p1 > EPSILON && signed_p1 > 0.0) {
                    normal = delta_p1 / dist_p1;
                }
                let push = normal * (target_dist - dist_p1);
                let a = 1.0 - t1;
                let b = t1;
                let denom = a * a * w0 + b * b * w1;
                if (denom > EPSILON) {
                    let d0 = (a * w0 / denom) * push;
                    let d1 = (b * w1 / denom) * push;
                    prev0 += d0;
                    prev1 += d1;
                }
            }
        }
        // p2 に対して
        {
            let t2 = closest_t_on_segment(tri.p2, prev0, prev1);
            let pt_on_edge = prev0 + t2 * (prev1 - prev0);
            let delta_p2 = pt_on_edge - tri.p2;
            let dist_p2 = length(delta_p2);
            let signed_p2 = dot(delta_p2, face_normal);
            if (dist_p2 < target_dist && signed_p2 >= -target_dist) {
                var normal = face_normal;
                if (dist_p2 > EPSILON && signed_p2 > 0.0) {
                    normal = delta_p2 / dist_p2;
                }
                let push = normal * (target_dist - dist_p2);
                let a = 1.0 - t2;
                let b = t2;
                let denom = a * a * w0 + b * b * w1;
                if (denom > EPSILON) {
                    let d0 = (a * w0 / denom) * push;
                    let d1 = (b * w1 / denom) * push;
                    prev0 += d0;
                    prev1 += d1;
                }
            }
        }
    }

    // XPBD原則: position (x_n) は不変。予測位置 prev_pos のみを更新
    vertices[c.v0].prev_pos = prev0;
    vertices[c.v1].prev_pos = prev1;
}
