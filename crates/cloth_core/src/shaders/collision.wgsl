struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
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

@group(0) @binding(0) var<storage, read_write> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read> colliders: array<GpuCollider>;
@group(0) @binding(2) var<storage, read> mesh_triangles: array<GpuMeshTriangle>;
@group(0) @binding(3) var<uniform> params: CollisionParams;
@group(0) @binding(4) var<storage, read> mesh_bounds: array<vec4<f32>>;

const EPSILON: f32 = 1e-7;

// 接触応答（クーロン摩擦と非弾性反発係数の適用）
// PBD/XPBDの接触定式化 (Velocity Projection via Normal Position Adjustment):
// PBDでは速度が v = (p - x_n) / dt で暗黙更新されるため、
// 位置 p をコライダー表面に押し出しても、x_n が衝突前の上空位置のままだと
// 下向きの突入速度が速度として引き継がれ、次ステップでコライダーを貫通してしまう。
// したがって、法線方向の相対速度を希望の反発速度 desired_v_out_n (-e * v_in) に
// 調停するため、x_n の法線成分を target_pos - normal * (desired_v_out_n * dt) に投影する。
fn apply_contact_response(
    target_pos: vec3<f32>,
    normal: vec3<f32>,
    friction: f32,
    restitution: f32,
    dt: f32,
    p_initial: vec3<f32>,
    x_n: ptr<function, vec3<f32>>,
    p: ptr<function, vec3<f32>>
) {
    // 1. 接線方向すべり変位の抽出とクーロン摩擦の適用
    let total_disp = target_pos - *x_n;
    let tangent_disp = total_disp - dot(total_disp, normal) * normal;
    *p = target_pos - tangent_disp * clamp(friction, 0.0, 1.0);

    // 2. 法線突入速度の吸収と反発 (Velocity Projection)
    let effective_dt = max(dt, 1e-6);
    let v_in_n = dot((p_initial - *x_n) / effective_dt, normal);
    var desired_v_out_n = 0.0;
    if (v_in_n < 0.0) {
        desired_v_out_n = -clamp(restitution, 0.0, 1.0) * v_in_n;
    }
    let delta_x_n = dot(target_pos - *x_n, normal) - desired_v_out_n * effective_dt;
    *x_n = *x_n + normal * delta_x_n;

    // 3. 接触押し出しに伴う異常初速（局所スパイク）の物理クランプ (Velocity Clamping)
    // コライダー裏面リカバリー等の急激な位置補正が時速数百km/hの暴走速度に変換されるのを防ぐ
    let max_safe_vel = 25.0; // 最大許容速度 25 m/s (時速 90km/h 相当)
    let v_post = (*p - *x_n) / effective_dt;
    let v_speed = length(v_post);
    if (v_speed > max_safe_vel) {
        *x_n = *p - (v_post / v_speed) * (max_safe_vel * effective_dt);
    }
}

struct ClosestResult {
    point: vec3<f32>,
    is_face: bool,
};

// 空間上の点 p から三角形 (a, b, c) への最近傍点および面内部フラグ
fn closest_point_on_triangle_ext(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> ClosestResult {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return ClosestResult(a, false);
    }

    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return ClosestResult(b, false);
    }

    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return ClosestResult(a + v * ab, false);
    }

    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return ClosestResult(c, false);
    }

    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return ClosestResult(a + w * ac, false);
    }

    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return ClosestResult(b + w * (c - b), false);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return ClosestResult(a + ab * v + ac * w, true);
}

// 線分 (p_old -> p_new) と 三角形 (a, b, c) の連続衝突判定 (CCD: Möller–Trumbore)
fn intersect_segment_triangle(
    p_old: vec3<f32>, p_new: vec3<f32>,
    a: vec3<f32>, b: vec3<f32>, c: vec3<f32>
) -> vec4<f32> {
    let d = p_new - p_old;
    let len_d = length(d);
    if (len_d < EPSILON) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let e1 = b - a;
    let e2 = c - a;
    let pvec = cross(d, e2);
    let det = dot(e1, pvec);

    if (abs(det) < EPSILON) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let inv_det = 1.0 / det;
    let tvec = p_old - a;
    let u = dot(tvec, pvec) * inv_det;
    if (u < -0.05 || u > 1.05) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let qvec = cross(tvec, e1);
    let v = dot(d, qvec) * inv_det;
    if (v < -0.05 || (u + v) > 1.05) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let t = dot(e2, qvec) * inv_det;
    if (t >= 0.0 && t <= 1.0) {
        return vec4<f32>(1.0, t, u, v);
    }

    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_vertices) {
        return;
    }

    var v = vertices[index];
    if (v.inv_mass <= 0.0) {
        return;
    }

    var p = v.prev_pos;
    let p_initial = p;
    var x_n = v.position;
    let thickness = v.thickness;
    let dt = params.dt;

    // 1. プリミティブコライダー (Sphere, Capsule, Plane)
    for (var i = 0u; i < params.num_colliders; i = i + 1u) {
        let col = colliders[i];

        if (col.collider_type == 0u) {
            let center = col.point_a;
            let target_r = col.radius + thickness;
            let delta = p - center;
            let dist = length(delta);

            if (dist < target_r) {
                var normal = vec3<f32>(0.0, 0.0, 1.0);
                if (dist > EPSILON) {
                    normal = delta / dist;
                }
                let target_pos = center + normal * target_r;
                apply_contact_response(target_pos, normal, col.friction, col.restitution, dt, p_initial, &x_n, &p);
            }
        } else if (col.collider_type == 1u) {
            let a = col.point_a;
            let b = col.point_b;
            let target_r = col.radius + thickness;

            let ab = b - a;
            let l2 = dot(ab, ab);
            var t = 0.0;
            if (l2 > EPSILON) {
                t = clamp(dot(p - a, ab) / l2, 0.0, 1.0);
            }
            let closest = a + ab * t;
            let delta = p - closest;
            let dist = length(delta);

            if (dist < target_r) {
                var normal = vec3<f32>(0.0, 0.0, 1.0);
                if (dist > EPSILON) {
                    normal = delta / dist;
                }
                let target_pos = closest + normal * target_r;
                apply_contact_response(target_pos, normal, col.friction, col.restitution, dt, p_initial, &x_n, &p);
            }
        } else if (col.collider_type == 2u) {
            let plane_pt = col.point_a;
            let plane_n = normalize(col.point_b);
            let dist = dot(p - plane_pt, plane_n);

            if (dist < thickness) {
                let target_pos = p + plane_n * (thickness - dist);
                apply_contact_response(target_pos, plane_n, col.friction, col.restitution, dt, p_initial, &x_n, &p);
            }
        }
    }

    // 2. メッシュコライダー (Triangle Mesh: 2段階フェッチ + CCD & 表側優先・裏面抑制リカバリー)
    var has_front_contact = false;
    var best_recovery_depth = 999999.0;
    var best_recovery_pos = p;
    var best_recovery_normal = vec3<f32>(0.0);
    var best_recovery_fric = 0.0;
    var best_recovery_rest = 0.0;
    var has_valid_recovery = false;

    for (var i = 0u; i < params.num_mesh_triangles; i = i + 1u) {
        // 第1段階: 16バイトの軽量境界球 (vec4: xyz=中心, w=外接半径+厚み) のみをフェッチして判定
        // Sweep 移動を考慮し、前位置 v.position と 現在位置 p の双方を境界球判定
        let b = mesh_bounds[i];
        let diff = p - b.xyz;
        let diff_old = v.position - b.xyz;
        let r = b.w * 1.5 + thickness;
        if (dot(diff, diff) > r * r && dot(diff_old, diff_old) > r * r) {
            continue; // 近傍外の三角形はここで即座にスキップ
        }

        // 第2段階: 境界球と交差した三角形のみ、48バイトの詳細頂点データをフェッチ
        let tri = mesh_triangles[i];
        let target_dist = thickness + tri.thickness;

        let cross_prod = cross(tri.p1 - tri.p0, tri.p2 - tri.p0);
        let cross_len = length(cross_prod);
        if (cross_len < EPSILON) {
            continue;
        }
        let face_normal = cross_prod / cross_len;
        let is_single_sided = (tri.flags & 1u) != 0u;

        // (A) 連続衝突判定 (CCD: Möller–Trumbore)
        // サブステップ開始位置 v.position から現在予測位置 p への移動線分が三角形を通過したかを判定
        let isect = intersect_segment_triangle(v.position, p, tri.p0, tri.p1, tri.p2);
        if (isect.x > 0.5) {
            let t_hit = isect.y;
            let hit_pt = v.position + t_hit * (p - v.position);
            var normal = face_normal;
            let move_dir = p - v.position;
            if (dot(move_dir, face_normal) > 0.0 && !is_single_sided) {
                normal = -face_normal;
            }
            let target_pos = hit_pt + normal * target_dist;
            apply_contact_response(target_pos, normal, tri.friction, tri.restitution, dt, p_initial, &x_n, &p);
            has_front_contact = true;
        }

        // (B) 近傍厚み接触 & 静的貫通リカバリー候補の収集
        let res = closest_point_on_triangle_ext(p, tri.p0, tri.p1, tri.p2);
        let q = res.point;
        let delta = p - q;
        let dist = length(delta);
        let signed_dist = dot(delta, face_normal);

        if (is_single_sided) {
            if (signed_dist >= 0.0) {
                // 表側: 通常の厚み判定 (多面接触を完全維持)
                if (dist < target_dist) {
                    var normal = face_normal;
                    if (dist > EPSILON) {
                        normal = delta / dist;
                    }
                    let target_pos = p + normal * (target_dist - dist);
                    apply_contact_response(target_pos, normal, tri.friction, tri.restitution, dt, p_initial, &x_n, &p);
                    has_front_contact = true;
                }
            } else {
                // 裏側 (メッシュ内部侵入):
                // 表側接触が一切ない孤立頂点のみを対象とし、安全ガードを満たす最近傍1面を記録
                let max_recovery_depth = target_dist * 3.0;
                if (res.is_face && dist < max_recovery_depth && dist <= abs(signed_dist) * 1.1 + EPSILON) {
                    if (dist < best_recovery_depth) {
                        best_recovery_depth = dist;
                        best_recovery_normal = face_normal;
                        best_recovery_pos = p + face_normal * (target_dist - signed_dist);
                        best_recovery_fric = tri.friction;
                        best_recovery_rest = tri.restitution;
                        has_valid_recovery = true;
                    }
                }
            }
        } else {
            // 両面コライダー: 厚み target_dist の両面薄皮判定
            if (dist < target_dist && signed_dist >= -target_dist) {
                var normal = face_normal;
                if (dist > EPSILON && signed_dist > 0.0) {
                    normal = delta / dist;
                }
                let target_pos = p + normal * (target_dist - dist);
                apply_contact_response(target_pos, normal, tri.friction, tri.restitution, dt, p_initial, &x_n, &p);
                has_front_contact = true;
            }
        }
    }

    // フェーズ2: 表側接触が一切ない完全なめり込み頂点のみ、最適な最近傍1面から安全に脱出リカバリー
    if (!has_front_contact && has_valid_recovery) {
        apply_contact_response(best_recovery_pos, best_recovery_normal, best_recovery_fric, best_recovery_rest, dt, p_initial, &x_n, &p);
    }

    v.prev_pos = p;
    v.position = x_n; // 法線突入速度を吸収した基準位置
    vertices[index] = v;
}

