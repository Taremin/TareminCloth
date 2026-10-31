use std::collections::HashMap;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 3],
    pub inv_mass: f32,
    pub prev_pos: [f32; 3],
    pub layer_id: u32,
    pub velocity: [f32; 3],
    pub thickness: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDistanceConstraint {
    pub v0: u32,
    pub v1: u32,
    pub rest_length: f32,
    pub tension_compliance: f32,
    pub compression_compliance: f32,
    pub constraint_type: u32, // 0: Stretch, 1: Shear
    pub _pad0: f32,
    pub _pad1: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBendingConstraint {
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
    pub v3: u32,
    pub rest_length: f32,
    pub compliance: f32,
    pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPinConstraint {
    pub vertex_idx: u32,
    pub weight: f32,
    pub _pad: [f32; 2],
    pub target_pos: [f32; 3],
    pub _pad2: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSewingConstraint {
    pub v0: u32,
    pub v1: u32,
    pub current_rest_len: f32,
    pub target_rest_len: f32,
    pub shrink_speed: f32,
    pub compliance: f32,
    pub lock_on_close: f32,
    pub _pad1: f32,
}


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCollider {
    pub collider_type: u32, // 0: Sphere, 1: Capsule, 2: Plane
    pub friction: f32,
    pub restitution: f32,
    pub _pad0: f32,
    pub point_a: [f32; 3],
    pub radius: f32,
    pub point_b: [f32; 3],
    pub _pad1: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMeshTriangle {
    pub p0: [f32; 3],
    pub friction: f32,
    pub p1: [f32; 3],
    pub thickness: f32,
    pub p2: [f32; 3],
    pub restitution: f32,
    pub flags: u32,
    pub _pad: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuColliderEdge {
    pub p0: [f32; 3],
    pub thickness: f32,
    pub p1: [f32; 3],
    pub friction: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EdgeCollisionParams {
    pub cell_size: f32,
    pub table_size: u32,
    pub num_cloth_edges: u32,
    pub num_collider_edges: u32,
    pub edge_margin_scale: f32,
    pub edge_margin_offset: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimParams {
    pub gravity: [f32; 4], // [gx, gy, gz, dt]
    pub damping: f32,
    pub substeps: u32,
    pub num_vertices: u32,
    pub num_distance_constraints: u32,
    pub num_bending_constraints: u32,
    pub num_sewing_constraints: u32,
    pub sewing_compliance: f32,
    pub enable_sewing_lock: f32,
    pub sewing_lock_distance: f32,
    pub _pad0: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuStarPair {
    pub v0: u32,
    pub v1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SelfCollisionParams {
    pub cell_size: f32,
    pub table_size: u32,
    pub num_vertices: u32,
    pub num_edges: u32,
    pub relief_factor: f32,
    pub max_displacement_ratio: f32,
    pub enable_relief: u32,
    pub enable_normal_untangling: u32,
    pub exclude_neighbors: u32,
    pub max_search_iterations: u32,
    pub enable_ee: u32,
    pub _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVtPair {
    pub vert_i: u32,
    pub vert_j: u32,
    pub v0: u32,
    pub v1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuEePair {
    pub v_i: u32,
    pub v_ui: u32,
    pub v_j: u32,
    pub v_vj: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PairCollectParams {
    pub cell_size: f32,
    pub table_size: u32,
    pub num_vertices: u32,
    pub max_vt_pairs: u32,
    pub max_ee_pairs: u32,
    pub safety_margin: f32,
    pub exclude_neighbors: u32,
    pub margin_mode: u32,
    pub dt_frame: f32,
    pub velocity_horizon_scale: f32,
    pub max_horizon: f32,
    /// 頂点あたり保持枠数 (V-T用、1..=8)。配置は index*quota+k のコンパクト配置。
    pub quota_vt: u32,
    /// 頂点あたり保持枠数 (E-E用、1..=8)。
    pub quota_ee: u32,
}


#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PairCounters {
    /// 受理された候補数 (check通過・重複排除後。枠上限を超え得る)
    pub vt_count: u32,
    pub ee_count: u32,
    /// 枠不足で破棄された候補数
    pub vt_dropped: u32,
    pub ee_dropped: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PairSolveParams {
    pub num_vertices: u32,
    pub max_vt_pairs: u32,
    pub max_ee_pairs: u32,
    pub enable_normal_untangling: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuEdge {
    pub v0: u32,
    pub v1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTriangle {
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
    pub layer_id: u32,
}

/// 面密度 (kg/m2) および線密度 (kg/m) とピンウェイトから物理単位の逆質量配列を計算する。
/// 各三角形の面積×面密度を3頂点へ等分配し、
/// さらに三角形に属さない1Dエッジ長×線密度を2頂点へ等分配する。
/// 面も1Dエッジも持たない孤立頂点は微小質量 (MIN_MASS) で支え、ゼロ除算を防ぐ。
/// ピンウェイト1.0の頂点は 0.0 (完全固定) となる。
pub fn compute_combined_inv_masses(
    positions: &[[f32; 3]],
    faces: Option<&[[u32; 3]]>,
    strand_edges: Option<&[[u32; 2]]>,
    pin_weights: Option<&[f32]>,
    areal_density: f32,
    linear_density: f32,
) -> Vec<f32> {
    const MIN_MASS: f32 = 1e-9;
    let n = positions.len();
    let mut masses = vec![0.0f32; n];

    // 1. 面（2D布）の質量分配
    if areal_density > 0.0 {
        if let Some(tris) = faces {
            for &[a, b, c] in tris {
                let (ia, ib, ic) = (a as usize, b as usize, c as usize);
                if ia >= n || ib >= n || ic >= n || (a == b || b == c || c == a) {
                    continue;
                }
                let pa = positions[ia];
                let pb = positions[ib];
                let pc = positions[ic];
                let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
                let cross = [
                    ab[1] * ac[2] - ab[2] * ac[1],
                    ab[2] * ac[0] - ab[0] * ac[2],
                    ab[0] * ac[1] - ab[1] * ac[0],
                ];
                let area = 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
                let share = area * areal_density / 3.0;
                masses[ia] += share;
                masses[ib] += share;
                masses[ic] += share;
            }
        }
    }

    // 2. 1Dストランド（紐）の質量分配
    if linear_density > 0.0 {
        if let Some(edges) = strand_edges {
            for &[a, b] in edges {
                let (ia, ib) = (a as usize, b as usize);
                if ia >= n || ib >= n || a == b {
                    continue;
                }
                let pa = positions[ia];
                let pb = positions[ib];
                let dx = pa[0] - pb[0];
                let dy = pa[1] - pb[1];
                let dz = pa[2] - pb[2];
                let len = (dx * dx + dy * dy + dz * dz).sqrt();
                let share = len * linear_density / 2.0;
                masses[ia] += share;
                masses[ib] += share;
            }
        }
    }

    masses
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            let w = pin_weights
                .and_then(|ws| ws.get(i).copied())
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            if w >= 1.0 {
                0.0
            } else {
                (1.0 - w) / m.max(MIN_MASS)
            }
        })
        .collect()
}

/// 面密度 (kg/m2) とピンウェイトから物理単位の逆質量配列を計算する。
/// 後方互換性用ラッパー。
pub fn areal_inv_masses(
    positions: &[[f32; 3]],
    faces: Option<&[[u32; 3]]>,
    pin_weights: Option<&[f32]>,
    areal_density: f32,
) -> Vec<f32> {
    compute_combined_inv_masses(positions, faces, None, pin_weights, areal_density, 0.0)
}

fn expand_bits_10(mut v: u32) -> u32 {
    v = (v.wrapping_mul(0x00010001)) & 0xFF0000FF;
    v = (v.wrapping_mul(0x00000101)) & 0x0F00F00F;
    v = (v.wrapping_mul(0x00000011)) & 0xC30C30C3;
    v = (v.wrapping_mul(0x00000005)) & 0x49249249;
    v
}

pub fn morton_code_3d(p: [f32; 3], bmin: [f32; 3], inv_extent: [f32; 3]) -> u32 {
    let x = (((p[0] - bmin[0]) * inv_extent[0]).clamp(0.0, 1.0) * 1023.0) as u32;
    let y = (((p[1] - bmin[1]) * inv_extent[1]).clamp(0.0, 1.0) * 1023.0) as u32;
    let z = (((p[2] - bmin[2]) * inv_extent[2]).clamp(0.0, 1.0) * 1023.0) as u32;
    (expand_bits_10(x) << 2) | (expand_bits_10(y) << 1) | expand_bits_10(z)
}


pub struct ClothMesh {
    pub vertices: Vec<GpuVertex>,
    pub distance_constraints: Vec<GpuDistanceConstraint>,
    pub dist_color_offsets: Vec<u32>,
    pub dist_color_counts: Vec<u32>,
    /// 長距離拘束 (constraint_type=2) の件数。無効時は 0。
    pub coarse_constraint_count: u32,
    pub original_edge_to_constraint: Vec<usize>,
    pub initial_distance_rest_lengths: Vec<f32>,

    pub bending_constraints: Vec<GpuBendingConstraint>,
    pub bend_color_offsets: Vec<u32>,
    pub bend_color_counts: Vec<u32>,

    pub sewing_constraints: Vec<GpuSewingConstraint>,
    pub sew_color_offsets: Vec<u32>,
    pub sew_color_counts: Vec<u32>,

    pub adj_offsets: Vec<u32>,
    pub adj_indices: Vec<u32>,
    pub two_hop_offsets: Vec<u32>,
    pub two_hop_indices: Vec<u32>,
    pub two_hop_rest_lengths: Vec<f32>,
    pub local_edge_lengths: Vec<f32>,
    pub star_offsets: Vec<u32>,
    pub star_indices: Vec<GpuStarPair>,
    pub triangles: Vec<GpuTriangle>,
    pub edges: Vec<GpuEdge>,
    pub island_ids: Vec<u32>,
    pub mean_edge_length: f32,
    pub max_edge_length: f32,
}

impl ClothMesh {
    /// 後方互換コンストラクタ (長距離拘束なし)。新規コードは from_raw_with_opts を使うこと。
    #[allow(clippy::too_many_arguments)]
    pub fn from_raw(
        positions: &[[f32; 3]],
        edges: &[[u32; 2]],
        faces: Option<&[[u32; 3]]>,
        inv_masses: Option<&[f32]>,
        sewing_springs: Option<&[[u32; 2]]>,
        layer_ids: Option<&[u32]>,
        thicknesses: Option<&[f32]>,
        default_layer_id: u32,
        default_thickness: f32,
        tension_stiffness: f32,
        compression_stiffness: f32,
        shear_stiffness: f32,
        bending_stiffness: f32,
        sewing_shrink_speed: f32,
        sewing_stiffness: Option<f32>,
        enable_sewing_lock: Option<bool>,
    ) -> Self {
        Self::from_raw_with_opts(
            positions, edges, faces, inv_masses, sewing_springs, layer_ids, thicknesses,
            default_layer_id, default_thickness, tension_stiffness, compression_stiffness,
            shear_stiffness, bending_stiffness, sewing_shrink_speed, sewing_stiffness,
            enable_sewing_lock, false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_raw_with_opts(
        positions: &[[f32; 3]],
        edges: &[[u32; 2]],
        faces: Option<&[[u32; 3]]>,
        inv_masses: Option<&[f32]>,
        sewing_springs: Option<&[[u32; 2]]>,
        layer_ids: Option<&[u32]>,
        thicknesses: Option<&[f32]>,
        default_layer_id: u32,
        default_thickness: f32,
        tension_stiffness: f32,
        compression_stiffness: f32,
        shear_stiffness: f32,
        bending_stiffness: f32,
        sewing_shrink_speed: f32,
        sewing_stiffness: Option<f32>,
        enable_sewing_lock: Option<bool>,
        enable_coarse_constraints: bool,
    ) -> Self {
        let n_verts = positions.len();
        let mut vertices = Vec::with_capacity(n_verts);

        for i in 0..n_verts {
            let inv_m = inv_masses.map_or(1.0, |m| m[i]);
            let layer_id = layer_ids.map_or(default_layer_id, |l| l[i]);
            let thick = thicknesses.map_or(default_thickness, |t| t[i]);

            vertices.push(GpuVertex {
                position: positions[i],
                inv_mass: inv_m,
                prev_pos: positions[i],
                layer_id,
                velocity: [0.0, 0.0, 0.0],
                thickness: thick,
            });
        }

        // 1. 距離拘束 (Stretch: 引張 & 圧縮, Shear: せん断)
        let mut dist_constraints = Vec::with_capacity(edges.len() * 2);

        // 剛性は物理単位 (N/m) のバネ定数 k として扱い、XPBDコンプライアンス
        // alpha = 1/k とする。頂点質量も物理単位 (kg) のため、静止釣り合い伸びは
        // 荷重/k で決まる物理的に正しい値に収束する。5000 N/m以上は完全非伸縮。
        let tension_compliance = if tension_stiffness >= 5000.0 {
            0.0 // 完全非伸縮 (Inextensible PBD)
        } else if tension_stiffness > 0.0 {
            1.0 / tension_stiffness
        } else {
            1e10
        };

        let compression_compliance = if compression_stiffness >= 5000.0 {
            0.0
        } else if compression_stiffness > 0.0 {
            1.0 / compression_stiffness
        } else {
            1e10
        };

        let shear_compliance = if shear_stiffness >= 5000.0 {
            0.0
        } else if shear_stiffness > 0.0 {
            1.0 / shear_stiffness
        } else {
            1e10
        };

        // 辺エッジ (Tension / Compression)
        let mut raw_edge_to_dist: Vec<usize> = Vec::with_capacity(edges.len());
        for &[v0, v1] in edges {
            if (v0 as usize) < n_verts && (v1 as usize) < n_verts && v0 != v1 {
                let p0 = positions[v0 as usize];
                let p1 = positions[v1 as usize];
                let dx = p0[0] - p1[0];
                let dy = p0[1] - p1[1];
                let dz = p0[2] - p1[2];
                let rest_length = (dx * dx + dy * dy + dz * dz).sqrt();

                raw_edge_to_dist.push(dist_constraints.len());
                dist_constraints.push(GpuDistanceConstraint {
                    v0,
                    v1,
                    rest_length,
                    tension_compliance,
                    compression_compliance,
                    constraint_type: 0,
                    _pad0: 0.0,
                    _pad1: 0.0,
                });
            } else {
                raw_edge_to_dist.push(usize::MAX);
            }
        }

        // せん断拘束 (Shear: 対向頂点ペアまたはQuadクロスエッジ)
        if let Some(triangles) = faces {
            let mut edge_to_opp: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
            for &[f0, f1, f2] in triangles {
                let es = [
                    (f0.min(f1), f0.max(f1), f2),
                    (f1.min(f2), f1.max(f2), f0),
                    (f2.min(f0), f2.max(f0), f1),
                ];
                for (ea, eb, opp) in es {
                    edge_to_opp.entry((ea, eb)).or_default().push(opp);
                }
            }

            let mut existing_edges: std::collections::HashSet<(u32, u32)> = edges
                .iter()
                .map(|&[a, b]| (a.min(b), a.max(b)))
                .collect();

            let mut shear_edge_keys: Vec<(u32, u32)> = edge_to_opp.keys().copied().collect();
            shear_edge_keys.sort_unstable();
            for key in shear_edge_keys {
                let opps = &edge_to_opp[&key];
                if opps.len() == 2 {
                    let v0 = opps[0].min(opps[1]);
                    let v1 = opps[0].max(opps[1]);
                    if v0 != v1 && existing_edges.insert((v0, v1)) {
                        let p0 = positions[v0 as usize];
                        let p1 = positions[v1 as usize];
                        let dx = p0[0] - p1[0];
                        let dy = p0[1] - p1[1];
                        let dz = p0[2] - p1[2];
                        let rest_length = (dx * dx + dy * dy + dz * dz).sqrt();

                        dist_constraints.push(GpuDistanceConstraint {
                            v0,
                            v1,
                            rest_length,
                            tension_compliance: shear_compliance,
                            compression_compliance: shear_compliance,
                            constraint_type: 1,
                            _pad0: 0.0,
                            _pad1: 0.0,
                        });
                    }
                }
            }
        }

        // 長距離拘束 (Coarse: 2ホップ先頂点間の距離拘束、オプション)。
        // 反復1回あたりの拘束伝播距離を約2倍にし、長尺布の静止伸び残留を低減する
        // (2階層法の粗層に相当。細層と同一ループで解くため prolongation 不要)。
        // 既存辺・せん断対角と重複せず、頂点あたり最大4件 (最短優先)。
        // constraint_type=2 (引張コンプライアンスを使用)。
        let mut coarse_constraint_count: u32 = 0;
        if enable_coarse_constraints && n_verts > 0 {
            let mut existing: std::collections::HashSet<(u32, u32)> = edges
                .iter()
                .filter_map(|&[a, b]| {
                    if (a as usize) < n_verts && (b as usize) < n_verts && a != b {
                        Some((a.min(b), a.max(b)))
                    } else {
                        None
                    }
                })
                .collect();
            for dc in dist_constraints.iter() {
                existing.insert((dc.v0.min(dc.v1), dc.v0.max(dc.v1)));
            }
            let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n_verts];
            for &[a, b] in edges {
                if (a as usize) < n_verts && (b as usize) < n_verts && a != b {
                    adj[a as usize].push(b);
                    adj[b as usize].push(a);
                }
            }
            const MAX_COARSE_PER_VERT: usize = 4;
            for i in 0..n_verts {
                let mut cands: Vec<(f32, u32)> = Vec::new();
                for &j in &adj[i] {
                    for &k in &adj[j as usize] {
                        if k as usize == i {
                            continue;
                        }
                        let key = ((i as u32).min(k), (i as u32).max(k));
                        if existing.contains(&key) {
                            continue;
                        }
                        let p0 = positions[i];
                        let p1 = positions[k as usize];
                        let dx = p0[0] - p1[0];
                        let dy = p0[1] - p1[1];
                        let dz = p0[2] - p1[2];
                        cands.push(((dx * dx + dy * dy + dz * dz).sqrt(), k));
                    }
                }
                cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                cands.dedup_by(|a, b| a.1 == b.1);
                for (rest_length, k) in cands.into_iter().take(MAX_COARSE_PER_VERT) {
                    let key = ((i as u32).min(k), (i as u32).max(k));
                    if !existing.insert(key) {
                        continue;
                    }
                    dist_constraints.push(GpuDistanceConstraint {
                        v0: i as u32,
                        v1: k,
                        rest_length,
                        tension_compliance,
                        compression_compliance,
                        constraint_type: 2,
                        _pad0: 0.0,
                        _pad1: 0.0,
                    });
                    coarse_constraint_count += 1;
                }
            }
        }

        let (sorted_dist, dist_color_offsets, dist_color_counts, dist_remap) =
            crate::coloring::color_distance_constraints(n_verts, &dist_constraints);

        let original_edge_to_constraint: Vec<usize> = raw_edge_to_dist
            .iter()
            .map(|&raw_idx| {
                if raw_idx != usize::MAX && raw_idx < dist_remap.len() {
                    dist_remap[raw_idx]
                } else {
                    usize::MAX
                }
            })
            .collect();

        let initial_distance_rest_lengths: Vec<f32> =
            sorted_dist.iter().map(|c| c.rest_length).collect();

        // 2. 曲げ拘束 (Distance Bending: 共有エッジを持つ2三角形の対向頂点間距離拘束)
        let mut bending_constraints = Vec::new();
        let bend_compliance = if bending_stiffness > 0.0 {
            1.0 / bending_stiffness
        } else {
            1e10
        };

        if let Some(triangles) = faces {
            let mut edge_to_opp_verts: HashMap<(u32, u32), Vec<u32>> = HashMap::new();

            for &[f0, f1, f2] in triangles {
                let edges = [
                    (f0.min(f1), f0.max(f1), f2),
                    (f1.min(f2), f1.max(f2), f0),
                    (f2.min(f0), f2.max(f0), f1),
                ];
                for (ea, eb, opp) in edges {
                    edge_to_opp_verts.entry((ea, eb)).or_default().push(opp);
                }
            }

            let mut edge_keys: Vec<(u32, u32)> = edge_to_opp_verts.keys().copied().collect();
            edge_keys.sort_unstable();

            for (v0, v1) in edge_keys {
                let opps = &edge_to_opp_verts[&(v0, v1)];
                if opps.len() == 2 {
                    let v2 = opps[0];
                    let v3 = opps[1];

                    let p2 = positions[v2 as usize];
                    let p3 = positions[v3 as usize];
                    let dx = p2[0] - p3[0];
                    let dy = p2[1] - p3[1];
                    let dz = p2[2] - p3[2];
                    let rest_length = (dx * dx + dy * dy + dz * dz).sqrt();

                    bending_constraints.push(GpuBendingConstraint {
                        v0,
                        v1,
                        v2,
                        v3,
                        rest_length,
                        compliance: bend_compliance,
                        _pad: [0.0; 2],
                    });
                }
            }

            // 2b. 縫合シーム（sewing_springs）を跨ぐ隣接境界三角形間の曲げ拘束 (Seam-Gated Bending)
            if let Some(sew_pairs) = sewing_springs {
                let mut parent: Vec<usize> = (0..n_verts).collect();
                let mut has_sew = vec![false; n_verts];
                fn find_sew_root(p: &mut [usize], mut i: usize) -> usize {
                    let mut root = i;
                    while root != p[root] {
                        root = p[root];
                    }
                    while i != root {
                        let next = p[i];
                        p[i] = root;
                        i = next;
                    }
                    root
                }
                for &[u, v] in sew_pairs {
                    let ui = u as usize;
                    let vi = v as usize;
                    if ui < n_verts && vi < n_verts && ui != vi {
                        has_sew[ui] = true;
                        has_sew[vi] = true;
                        let ru = find_sew_root(&mut parent, ui);
                        let rv = find_sew_root(&mut parent, vi);
                        if ru != rv {
                            let min_r = ru.min(rv);
                            let max_r = ru.max(rv);
                            parent[max_r] = min_r;
                        }
                    }
                }
                let sew_rep: Vec<u32> = (0..n_verts)
                    .map(|i| find_sew_root(&mut parent, i) as u32)
                    .collect();

                let mut seam_edge_map: HashMap<(u32, u32), Vec<(u32, u32, u32)>> = HashMap::new();
                for (&(ea, eb), opps) in &edge_to_opp_verts {
                    if opps.len() != 1 {
                        continue;
                    }
                    if (ea as usize) >= n_verts || (eb as usize) >= n_verts {
                        continue;
                    }
                    // 縫合シーム辺は両端点とも縫合スプリングに接続されている必要がある
                    if !has_sew[ea as usize] || !has_sew[eb as usize] {
                        continue;
                    }
                    let ra = sew_rep[ea as usize];
                    let rb = sew_rep[eb as usize];
                    if ra == rb {
                        continue;
                    }
                    let (canon_key, u_aligned, v_aligned) = if ra < rb {
                        ((ra, rb), ea, eb)
                    } else {
                        ((rb, ra), eb, ea)
                    };
                    seam_edge_map
                        .entry(canon_key)
                        .or_default()
                        .push((u_aligned, v_aligned, opps[0]));
                }

                let mut seam_keys: Vec<(u32, u32)> = seam_edge_map.keys().copied().collect();
                seam_keys.sort_unstable();

                for key in seam_keys {
                    let entries = &seam_edge_map[&key];
                    if entries.len() != 2 {
                        continue;
                    }
                    let (a0, a1, v2) = entries[0];
                    let (b0, b1, v3) = entries[1];
                    // 端点を共有する開放断面エッジや、対向頂点が同一（または同一縫合グループ）の極薄チューブは除外
                    if a0 == b0 || a1 == b1 || v2 == v3 || sew_rep[v2 as usize] == sew_rep[v3 as usize] {
                        continue;
                    }
                    if (v2 as usize) >= n_verts || (v3 as usize) >= n_verts {
                        continue;
                    }

                    let pa0 = positions[a0 as usize];
                    let pa1 = positions[a1 as usize];
                    let pv2 = positions[v2 as usize];
                    let ea_vec = [pa1[0] - pa0[0], pa1[1] - pa0[1], pa1[2] - pa0[2]];
                    let len_a = (ea_vec[0] * ea_vec[0] + ea_vec[1] * ea_vec[1] + ea_vec[2] * ea_vec[2]).sqrt();

                    let pb0 = positions[b0 as usize];
                    let pb1 = positions[b1 as usize];
                    let pv3 = positions[v3 as usize];
                    let eb_vec = [pb1[0] - pb0[0], pb1[1] - pb0[1], pb1[2] - pb0[2]];
                    let len_b = (eb_vec[0] * eb_vec[0] + eb_vec[1] * eb_vec[1] + eb_vec[2] * eb_vec[2]).sqrt();

                    if len_a <= 1e-6 || len_b <= 1e-6 {
                        continue;
                    }

                    let ua = [ea_vec[0] / len_a, ea_vec[1] / len_a, ea_vec[2] / len_a];
                    let da = [pv2[0] - pa0[0], pv2[1] - pa0[1], pv2[2] - pa0[2]];
                    let sa = da[0] * ua[0] + da[1] * ua[1] + da[2] * ua[2];
                    let perp_a = [da[0] - sa * ua[0], da[1] - sa * ua[1], da[2] - sa * ua[2]];
                    let ha = (perp_a[0] * perp_a[0] + perp_a[1] * perp_a[1] + perp_a[2] * perp_a[2]).sqrt();

                    let ub = [eb_vec[0] / len_b, eb_vec[1] / len_b, eb_vec[2] / len_b];
                    let db = [pv3[0] - pb0[0], pv3[1] - pb0[1], pv3[2] - pb0[2]];
                    let sb = db[0] * ub[0] + db[1] * ub[1] + db[2] * ub[2];
                    let perp_b = [db[0] - sb * ub[0], db[1] - sb * ub[1], db[2] - sb * ub[2]];
                    let hb = (perp_b[0] * perp_b[0] + perp_b[1] * perp_b[1] + perp_b[2] * perp_b[2]).sqrt();

                    // シーム軸周りに2三角形を180度平坦展開した際の真の対向頂点間距離
                    let ds = sa - sb;
                    let dh = ha + hb;
                    let flat_rest_len = (ds * ds + dh * dh).sqrt();
                    if flat_rest_len <= 1e-5 {
                        continue;
                    }

                    let gate_dist = (flat_rest_len * 0.5).max(0.01);
                    let gate_v0 = if a0 != b0 { a0 } else { a1 };
                    let gate_v1 = if a0 != b0 { b0 } else { b1 };

                    bending_constraints.push(GpuBendingConstraint {
                        v0: gate_v0,
                        v1: gate_v1,
                        v2,
                        v3,
                        rest_length: flat_rest_len,
                        compliance: bend_compliance,
                        _pad: [1.0, gate_dist],
                    });
                }
            }
        }

        // 1Dストランド（紐）の曲げ拘束:
        // 三角形に属さないエッジ列において、連続する頂点 a - u - b の 2ホップペア (a, b) に曲げ拘束を生成
        {
            let mut face_edges: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
            if let Some(triangles) = faces {
                for &[f0, f1, f2] in triangles {
                    face_edges.insert((f0.min(f1), f0.max(f1)));
                    face_edges.insert((f1.min(f2), f1.max(f2)));
                    face_edges.insert((f2.min(f0), f2.max(f0)));
                }
            }

            let mut strand_adj: HashMap<u32, Vec<u32>> = HashMap::new();
            for &[v0, v1] in edges {
                if (v0 as usize) < n_verts && (v1 as usize) < n_verts && v0 != v1 {
                    let key = (v0.min(v1), v0.max(v1));
                    if !face_edges.contains(&key) {
                        strand_adj.entry(v0).or_default().push(v1);
                        strand_adj.entry(v1).or_default().push(v0);
                    }
                }
            }

            let mut existing_bend_pairs: std::collections::HashSet<(u32, u32)> = bending_constraints
                .iter()
                .map(|bc| (bc.v2.min(bc.v3), bc.v2.max(bc.v3)))
                .collect();

            let mut strand_keys: Vec<u32> = strand_adj.keys().copied().collect();
            strand_keys.sort_unstable();
            for u in strand_keys {
                let neighbors = &strand_adj[&u];
                if neighbors.len() >= 2 {
                    for i in 0..neighbors.len() {
                        for j in (i + 1)..neighbors.len() {
                            let a = neighbors[i];
                            let b = neighbors[j];
                            if a != b {
                                let pair = (a.min(b), a.max(b));
                                if existing_bend_pairs.insert(pair) {
                                    let pa = positions[a as usize];
                                    let pb = positions[b as usize];
                                    let dx = pa[0] - pb[0];
                                    let dy = pa[1] - pb[1];
                                    let dz = pa[2] - pb[2];
                                    let rest_length = (dx * dx + dy * dy + dz * dz).sqrt();

                                    // 1Dストランドは引き絞り（距離拘束の動的短縮）を阻害しないよう適度に柔軟なコンプライアンスを持たせる
                                    let strand_bend_compliance = (bend_compliance * 10.0).max(0.5);
                                    bending_constraints.push(GpuBendingConstraint {
                                        v0: u,
                                        v1: u,
                                        v2: a,
                                        v3: b,
                                        rest_length,
                                        compliance: strand_bend_compliance,
                                        _pad: [0.0; 2],
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        let (sorted_bend, bend_color_offsets, bend_color_counts) =
            crate::coloring::color_bending_constraints(n_verts, &bending_constraints);

        // 3. 縫合拘束
        let mut sewing_constraints = Vec::new();
        let sew_stiff = sewing_stiffness.unwrap_or(10000.0);
        let sew_compliance = if sew_stiff >= 5000.0 {
            0.0 // 完全非伸縮 (Inextensible PBD)
        } else if sew_stiff > 0.0 {
            1.0 / sew_stiff
        } else {
            1e10
        };
        let lock_val = if enable_sewing_lock.unwrap_or(true) { 1.0 } else { 0.0 };

        if let Some(sew_pairs) = sewing_springs {
            for &[v0, v1] in sew_pairs {
                if (v0 as usize) < n_verts && (v1 as usize) < n_verts && v0 != v1 {
                    let p0 = positions[v0 as usize];
                    let p1 = positions[v1 as usize];
                    let dx = p0[0] - p1[0];
                    let dy = p0[1] - p1[1];
                    let dz = p0[2] - p1[2];
                    let initial_len = (dx * dx + dy * dy + dz * dz).sqrt();

                    sewing_constraints.push(GpuSewingConstraint {
                        v0,
                        v1,
                        current_rest_len: initial_len,
                        target_rest_len: 0.0,
                        shrink_speed: sewing_shrink_speed.max(0.1),
                        compliance: sew_compliance,
                        lock_on_close: lock_val,
                        _pad1: 0.0,
                    });
                }
            }
        }

        let (sorted_sew, sew_color_offsets, sew_color_counts) =
            crate::coloring::color_sewing_constraints(n_verts, &sewing_constraints);

        // 4. 隣接頂点情報 (Topology Adjacent Neighbors) & 局所エッジ長 (Local Edge Lengths)
        let mut adj_lists = vec![Vec::new(); n_verts];
        let mut edge_len_sums = vec![0.0f32; n_verts];
        let mut edge_counts = vec![0u32; n_verts];
        let mut total_edge_len = 0.0f32;
        let mut max_edge_len = 0.0f32;
        let mut valid_edge_count = 0u32;

        for &[v0, v1] in edges {
            if (v0 as usize) < n_verts && (v1 as usize) < n_verts && v0 != v1 {
                let p0 = positions[v0 as usize];
                let p1 = positions[v1 as usize];
                let dx = p0[0] - p1[0];
                let dy = p0[1] - p1[1];
                let dz = p0[2] - p1[2];
                let len = (dx * dx + dy * dy + dz * dz).sqrt();

                total_edge_len += len;
                if len > max_edge_len {
                    max_edge_len = len;
                }
                valid_edge_count += 1;

                adj_lists[v0 as usize].push(v1);
                adj_lists[v1 as usize].push(v0);
                edge_len_sums[v0 as usize] += len;
                edge_counts[v0 as usize] += 1;
                edge_len_sums[v1 as usize] += len;
                edge_counts[v1 as usize] += 1;
            }
        }

        let mean_edge_length = if valid_edge_count > 0 {
            total_edge_len / valid_edge_count as f32
        } else {
            0.0
        };
        let max_edge_length = max_edge_len;

        // 元のメッシュ（縫合考慮前）の隣接リストを保持（自己衝突レスト長判定用）
        let orig_adj_lists = adj_lists.clone();

        // 縫合エッジ（sewing_springs）で結ばれた頂点ペアのトポロジー同一視（ホップ数0換算・完全縮約グラフ）
        // 縫合ペア (v0, v1) を同一結節点（縮約頂点）とみなし、完成形メッシュと同一の
        // 1ホップ・2ホップトポロジー隣接距離を自己衝突除外グラフ（adj_lists）に対称構築する。
        if let Some(sewing_pairs) = sewing_springs {
            // 1. Union-Find で縫合ペアを同一グループに統合
            let mut parent: Vec<usize> = (0..n_verts).collect();
            fn find_root(p: &mut [usize], mut i: usize) -> usize {
                let mut root = i;
                while root != p[root] {
                    root = p[root];
                }
                while i != root {
                    let next = p[i];
                    p[i] = root;
                    i = next;
                }
                root
            }
            fn union_roots(p: &mut [usize], i: usize, j: usize) {
                let root_i = find_root(p, i);
                let root_j = find_root(p, j);
                if root_i != root_j {
                    p[root_j] = root_i;
                }
            }

            for &[v0, v1] in sewing_pairs {
                if (v0 as usize) < n_verts && (v1 as usize) < n_verts && v0 != v1 {
                    union_roots(&mut parent, v0 as usize, v1 as usize);
                }
            }

            // 2. 代表頂点ごとの実頂点グループを構築
            let mut group_verts: Vec<Vec<u32>> = vec![Vec::new(); n_verts];
            for i in 0..n_verts {
                let rep = find_root(&mut parent, i);
                group_verts[rep].push(i as u32);
            }

            // 3. 同一結節点グループ内の全頂点同士を 1ホップ隣接登録
            for g in &group_verts {
                if g.len() > 1 {
                    for &u in g {
                        for &v in g {
                            if u != v {
                                adj_lists[u as usize].push(v);
                            }
                        }
                    }
                }
            }

            // 4. 同一結節点グループに接続する全隣接頂点を相互に共有（完全対称登録）
            let base_adj = adj_lists.clone();
            for g in &group_verts {
                if g.len() > 1 {
                    let mut shared_neighbors = std::collections::HashSet::new();
                    for &u in g {
                        for &n in &base_adj[u as usize] {
                            if !g.contains(&n) {
                                shared_neighbors.insert(n);
                            }
                        }
                    }
                    for &u in g {
                        for &n in &shared_neighbors {
                            adj_lists[u as usize].push(n);
                            adj_lists[n as usize].push(u); // 対称性保証
                        }
                    }
                }
            }
        }

        let mut local_edge_lengths = Vec::with_capacity(n_verts);
        for i in 0..n_verts {
            if edge_counts[i] > 0 {
                local_edge_lengths.push(edge_len_sums[i] / edge_counts[i] as f32);
            } else {
                local_edge_lengths.push(default_thickness.max(0.01));
            }
        }

        let mut adj_offsets = Vec::with_capacity(n_verts + 1);
        let mut adj_indices = Vec::new();
        let mut current_offset = 0u32;
        for list in &mut adj_lists {
            list.sort_unstable();
            list.dedup();
            adj_offsets.push(current_offset);
            adj_indices.extend_from_slice(list);
            current_offset += list.len() as u32;
        }
        adj_offsets.push(current_offset);

        // 4b. トポロジー2ホップ近傍リスト (Two-Hop Neighbor CSR for Self-Collision)
        let mut two_hop_offsets = Vec::with_capacity(n_verts + 1);
        let mut two_hop_indices = Vec::new();
        let mut two_hop_rest_lengths = Vec::new();
        let mut current_two_hop_offset = 0u32;
        for i in 0..n_verts {
            let mut neighbors = Vec::new();
            neighbors.push(i as u32); // 0ホップ（自己参照）

            let start_i = adj_offsets[i] as usize;
            let end_i = adj_offsets[i + 1] as usize;
            for k in start_i..end_i {
                let u = adj_indices[k];
                neighbors.push(u); // 1ホップ

                let start_u = adj_offsets[u as usize] as usize;
                let end_u = adj_offsets[u as usize + 1] as usize;
                for ku in start_u..end_u {
                    neighbors.push(adj_indices[ku]); // 2ホップ
                }
            }

            neighbors.sort_unstable();
            neighbors.dedup();

            // 元のメッシュトポロジー（縫合拡張前）における2ホップ近傍を収集
            let mut orig_neighbors = Vec::new();
            orig_neighbors.push(i as u32);
            for &u in &orig_adj_lists[i] {
                orig_neighbors.push(u);
                for &v in &orig_adj_lists[u as usize] {
                    orig_neighbors.push(v);
                }
            }
            orig_neighbors.sort_unstable();
            orig_neighbors.dedup();

            two_hop_offsets.push(current_two_hop_offset);
            for &nbr in &neighbors {
                two_hop_indices.push(nbr);
                // 元のメッシュ内の2ホップ近傍であれば初期レスト距離を記録（Bridson動的折り畳み検出・V-V反発用）。
                // 縫合トポロジーによって追加されたペアや自身（0ホップ）は 0.0 を設定し、常時除外する。
                let is_orig = orig_neighbors.binary_search(&nbr).is_ok();
                let dist = if is_orig && nbr != i as u32 {
                    let p_i = positions[i];
                    let p_nbr = positions[nbr as usize];
                    let dx = p_i[0] - p_nbr[0];
                    let dy = p_i[1] - p_nbr[1];
                    let dz = p_i[2] - p_nbr[2];
                    (dx * dx + dy * dy + dz * dz).sqrt()
                } else {
                    0.0
                };
                two_hop_rest_lengths.push(dist);
            }
            current_two_hop_offset += neighbors.len() as u32;
        }
        two_hop_offsets.push(current_two_hop_offset);

        // 5. 頂点法線計算用スター情報 (Star Pairs for Vertex Normals)
        let mut star_lists = vec![Vec::new(); n_verts];
        if let Some(triangles) = faces {
            for &[f0, f1, f2] in triangles {
                if (f0 as usize) < n_verts && (f1 as usize) < n_verts && (f2 as usize) < n_verts {
                    star_lists[f0 as usize].push(GpuStarPair { v0: f1, v1: f2 });
                    star_lists[f1 as usize].push(GpuStarPair { v0: f2, v1: f0 });
                    star_lists[f2 as usize].push(GpuStarPair { v0: f0, v1: f1 });
                }
            }
        }

        let mut star_offsets = Vec::with_capacity(n_verts + 1);
        let mut star_indices = Vec::new();
        let mut current_star_offset = 0u32;
        for list in &star_lists {
            star_offsets.push(current_star_offset);
            star_indices.extend_from_slice(list);
            current_star_offset += list.len() as u32;
        }
        star_offsets.push(current_star_offset);
        // 6. 三角形面情報 (Faces for Face Untangling)
        let mut triangles_vec = Vec::new();
        if let Some(tris) = faces {
            for &[f0, f1, f2] in tris {
                if (f0 as usize) < n_verts && (f1 as usize) < n_verts && (f2 as usize) < n_verts {
                    let l_id = vertices[f0 as usize].layer_id;
                    triangles_vec.push(GpuTriangle { v0: f0, v1: f1, v2: f2, layer_id: l_id });
                }
            }
        }

        // 7. 連結成分 (Connected Components / Islands) の算出
        let mut island_ids = vec![u32::MAX; n_verts];
        let mut current_island = 0u32;
        let mut queue = std::collections::VecDeque::new();

        for i in 0..n_verts {
            if island_ids[i] == u32::MAX {
                island_ids[i] = current_island;
                queue.push_back(i);

                while let Some(u) = queue.pop_front() {
                    let start = adj_offsets[u] as usize;
                    let end = adj_offsets[u + 1] as usize;
                    for k in start..end {
                        let v = adj_indices[k] as usize;
                        if v < n_verts && island_ids[v] == u32::MAX {
                            island_ids[v] = current_island;
                            queue.push_back(v);
                        }
                    }
                }
                current_island += 1;
            }
        }

        let mut gpu_edges = Vec::new();
        let mut edge_set = std::collections::HashSet::new();
        for &[v0, v1] in edges {
            if (v0 as usize) < n_verts && (v1 as usize) < n_verts && v0 != v1 {
                let (e0, e1) = (v0.min(v1), v0.max(v1));
                if edge_set.insert((e0, e1)) {
                    gpu_edges.push(GpuEdge { v0: e0, v1: e1 });
                }
            }
        }

        if !gpu_edges.is_empty() && !positions.is_empty() {
            let mut bmin = [f32::MAX; 3];
            let mut bmax = [f32::MIN; 3];
            for p in positions {
                for axis in 0..3 {
                    bmin[axis] = bmin[axis].min(p[axis]);
                    bmax[axis] = bmax[axis].max(p[axis]);
                }
            }
            let inv_ext = [
                1.0 / (bmax[0] - bmin[0]).max(1e-5),
                1.0 / (bmax[1] - bmin[1]).max(1e-5),
                1.0 / (bmax[2] - bmin[2]).max(1e-5),
            ];
            gpu_edges.sort_by_cached_key(|e| {
                let p0 = positions[e.v0 as usize];
                let p1 = positions[e.v1 as usize];
                let mid = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5, (p0[2] + p1[2]) * 0.5];
                morton_code_3d(mid, bmin, inv_ext)
            });
        }

        Self {
            vertices,
            distance_constraints: sorted_dist,
            dist_color_offsets,
            dist_color_counts,
            coarse_constraint_count,
            original_edge_to_constraint,
            initial_distance_rest_lengths,
            bending_constraints: sorted_bend,
            bend_color_offsets,
            bend_color_counts,
            sewing_constraints: sorted_sew,
            sew_color_offsets,
            sew_color_counts,
            adj_offsets,
            adj_indices,
            two_hop_offsets,
            two_hop_indices,
            two_hop_rest_lengths,
            local_edge_lengths,
            star_offsets,
            star_indices,
            triangles: triangles_vec,
            edges: gpu_edges,
            island_ids,
            mean_edge_length,
            max_edge_length,
        }
    }
}

fn canonical_edge_key(a: [f32; 3], b: [f32; 3]) -> ([u32; 3], [u32; 3]) {
    let a_bits = [a[0].to_bits(), a[1].to_bits(), a[2].to_bits()];
    let b_bits = [b[0].to_bits(), b[1].to_bits(), b[2].to_bits()];
    if a_bits < b_bits {
        (a_bits, b_bits)
    } else {
        (b_bits, a_bits)
    }
}

/// メッシュコライダーの三角形群から重複のない稜線（エッジ）リストを一意に抽出する
pub fn extract_collider_edges(triangles: &[GpuMeshTriangle]) -> Vec<GpuColliderEdge> {
    let mut edge_map = HashMap::new();
    for t in triangles {
        let edges = [
            (t.p0, t.p1),
            (t.p1, t.p2),
            (t.p2, t.p0),
        ];
        for (a, b) in edges {
            let key = canonical_edge_key(a, b);
            edge_map
                .entry(key)
                .and_modify(|e: &mut GpuColliderEdge| {
                    e.thickness = e.thickness.max(t.thickness);
                    e.friction = (e.friction + t.friction) * 0.5;
                })
                .or_insert(GpuColliderEdge {
                    p0: a,
                    thickness: t.thickness,
                    p1: b,
                    friction: t.friction,
                });
        }
    }
    edge_map.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_topology_and_star_pairs() {
        // 2つの三角形からなるクアッド (0,1,2) と (2,1,3)
        // 0 -- 1
        // |  / |
        // 2 -- 3
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let edges = [
            [0, 1],
            [1, 2],
            [2, 0],
            [1, 3],
            [3, 2],
        ];
        let faces = [
            [0, 1, 2],
            [2, 1, 3],
        ];

        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            Some(&faces),
            None,
            None,
            None,
            None,
            0,
            0.005,
            1000.0,
            1000.0,
            1000.0,
            10.0,
            1.0,
            None,
            None,
        );

        // 頂点数4
        assert_eq!(mesh.adj_offsets.len(), 5);
        assert_eq!(mesh.local_edge_lengths.len(), 4);

        // 頂点0の隣接頂点: 1, 2
        let v0_adj_start = mesh.adj_offsets[0] as usize;
        let v0_adj_end = mesh.adj_offsets[1] as usize;
        let v0_adj = &mesh.adj_indices[v0_adj_start..v0_adj_end];
        assert_eq!(v0_adj, &[1, 2]);

        // 頂点1の隣接頂点: 0, 2, 3
        let v1_adj_start = mesh.adj_offsets[1] as usize;
        let v1_adj_end = mesh.adj_offsets[2] as usize;
        let v1_adj = &mesh.adj_indices[v1_adj_start..v1_adj_end];
        assert_eq!(v1_adj, &[0, 2, 3]);

        // スター情報の検証
        assert_eq!(mesh.star_offsets.len(), 5);
        // 頂点0は面 (0,1,2) の1つに属するためスターペアは (1,2)
        let v0_star_start = mesh.star_offsets[0] as usize;
        let v0_star_end = mesh.star_offsets[1] as usize;
        assert_eq!(v0_star_end - v0_star_start, 1);
        assert_eq!(mesh.star_indices[v0_star_start].v0, 1);
        assert_eq!(mesh.star_indices[v0_star_start].v1, 2);

        // 頂点1は2つの面に属する: (0,1,2)->(2,0), (2,1,3)->(3,2)
        let v1_star_start = mesh.star_offsets[1] as usize;
        let v1_star_end = mesh.star_offsets[2] as usize;
        assert_eq!(v1_star_end - v1_star_start, 2);

        // 連結成分の検証 (4頂点すべて連結)
        assert_eq!(mesh.island_ids, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_multiple_disconnected_islands() {
        // 2つの独立した三角形 (0,1,2) と (3,4,5)
        let positions = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [10.0, 1.0, 0.0],
        ];
        let edges = [
            [0, 1], [1, 2], [2, 0],
            [3, 4], [4, 5], [5, 3],
        ];
        let faces = [[0, 1, 2], [3, 4, 5]];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, Some(&faces), None, None, None, None,
            0, 0.005, 1000.0, 1000.0, 1000.0, 10.0, 1.0,
            None, None,
        );
        assert_eq!(mesh.island_ids, vec![0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn test_sewing_topology_zero_hop() {
        // 2つの独立したエッジ: (0-1) と (2-3)
        // 縫合エッジ: (1-2)
        // 縫合により 1 と 2 が同一結節点化され、0 から 3 はメッシュ2ホップ (0->1(=2)->3) と同一視される
        let positions = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
        ];
        let edges = [[0, 1], [2, 3]];
        let sew = [[1, 2]];

        let mesh = ClothMesh::from_raw(
            &positions, &edges, None, None, Some(&sew), None, None,
            0, 0.005, 1000.0, 1000.0, 1000.0, 10.0, 1.0,
            Some(10000.0), Some(true),
        );

        // 頂点1の隣接に 0 だけでなく 3 (頂点2の隣接) も含まれていること
        let v1_start = mesh.adj_offsets[1] as usize;
        let v1_end = mesh.adj_offsets[2] as usize;
        let v1_adjs = &mesh.adj_indices[v1_start..v1_end];
        assert!(v1_adjs.contains(&0));
        assert!(v1_adjs.contains(&2));
        assert!(v1_adjs.contains(&3), "縫合ペアの隣接頂点3が頂点1の近傍に共有されていること");

        // 頂点2の隣接にも 0 (頂点1の隣接) が含まれていること
        let v2_start = mesh.adj_offsets[2] as usize;
        let v2_end = mesh.adj_offsets[3] as usize;
        let v2_adjs = &mesh.adj_indices[v2_start..v2_end];
        assert!(v2_adjs.contains(&0), "縫合ペアの隣接頂点0が頂点2の近傍に共有されていること");
        assert!(v2_adjs.contains(&1));
        assert!(v2_adjs.contains(&3));
    }

    #[test]
    fn test_two_hop_topology_csr() {
        // 直線メッシュ: 0 -- 1 -- 2 -- 3 -- 4
        let positions = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0], [4.0, 0.0, 0.0],
        ];
        let edges = [[0, 1], [1, 2], [2, 3], [3, 4]];

        let mesh = ClothMesh::from_raw(
            &positions, &edges, None, None, None, None, None,
            0, 0.005, 1000.0, 1000.0, 1000.0, 10.0, 1.0,
            None, None,
        );

        assert_eq!(mesh.two_hop_offsets.len(), 6);

        // 各頂点の2ホップ近傍を取得
        let get_two_hop = |v: usize| -> &[u32] {
            let start = mesh.two_hop_offsets[v] as usize;
            let end = mesh.two_hop_offsets[v + 1] as usize;
            &mesh.two_hop_indices[start..end]
        };

        // 頂点0の2ホップ: 0自身, 1(1ホップ), 2(2ホップ)
        assert_eq!(get_two_hop(0), &[0, 1, 2]);
        // 頂点0自身のレスト距離は0.0
        assert!((mesh.two_hop_rest_lengths[0] - 0.0).abs() < 1e-6);

        // 頂点1の2ホップ: 1自身, 0,2(1ホップ), 3(2ホップ)
        assert_eq!(get_two_hop(1), &[0, 1, 2, 3]);

        // 頂点2の2ホップ: 全頂点 [0, 1, 2, 3, 4]
        assert_eq!(get_two_hop(2), &[0, 1, 2, 3, 4]);

        // 頂点3の2ホップ: 1, 2, 3, 4
        assert_eq!(get_two_hop(3), &[1, 2, 3, 4]);

        // 頂点4の2ホップ: 2, 3, 4
        assert_eq!(get_two_hop(4), &[2, 3, 4]);

        // 全頂点で昇順ソート & 重複なしを確認
        for v in 0..5 {
            let list = get_two_hop(v);
            for w in list.windows(2) {
                assert!(w[0] < w[1], "頂点 {} の2ホップリストが厳密昇順であること", v);
            }
        }
    }

    #[test]
    fn test_coarse_long_range_constraints() {
        // 直線メッシュ: 0 -- 1 -- 2 -- 3 -- 4
        let positions = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0], [4.0, 0.0, 0.0],
        ];
        let edges = [[0, 1], [1, 2], [2, 3], [3, 4]];

        let plain = ClothMesh::from_raw(
            &positions, &edges, None, None, None, None, None,
            0, 0.005, 1000.0, 1000.0, 1000.0, 10.0, 1.0,
            None, None,
        );
        assert_eq!(plain.coarse_constraint_count, 0);

        let coarse = ClothMesh::from_raw_with_opts(
            &positions, &edges, None, None, None, None, None,
            0, 0.005, 1000.0, 1000.0, 1000.0, 10.0, 1.0,
            None, None, true,
        );
        assert!(coarse.coarse_constraint_count > 0);
        // 2ホップ対 (0,2) 等が type=2 で含まれること
        let has_02 = coarse.distance_constraints.iter().any(|c| {
            c.constraint_type == 2 && ((c.v0 == 0 && c.v1 == 2) || (c.v0 == 2 && c.v1 == 0))
        });
        assert!(has_02, "2ホップ対 (0,2) が長距離拘束として含まれること");
        // 長距離拘束の自然長が初期幾何と一致すること
        for c in coarse.distance_constraints.iter().filter(|c| c.constraint_type == 2) {
            let p0 = positions[c.v0 as usize];
            let p1 = positions[c.v1 as usize];
            let dx = p0[0] - p1[0];
            let dy = p0[1] - p1[1];
            let dz = p0[2] - p1[2];
            let expect = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!((c.rest_length - expect).abs() < 1e-6);
        }
    }

    #[test]
    fn test_areal_inv_masses_physical_units() {
        // 1m x 1m の正方形 (2三角形)。面密度 0.15 kg/m2 なら総質量 0.15 kg。
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces = [[0, 1, 2], [0, 2, 3]];
        let inv = super::areal_inv_masses(&positions, Some(&faces), None, 0.15);
        assert_eq!(inv.len(), 4);
        let masses: Vec<f32> = inv.iter().map(|&w| 1.0 / w).collect();
        let total: f32 = masses.iter().sum();
        assert!((total - 0.15).abs() < 1e-5, "総質量が面積×面密度に一致すること: {}", total);
        // 対角頂点 (0, 2) は2面分、他は1面分のため重い
        assert!(masses[0] > masses[1]);
        assert!((masses[1] - masses[3]).abs() < 1e-7);

        // ピンウェイト1.0は完全固定 (0.0)
        let pinned = super::areal_inv_masses(&positions, Some(&faces), Some(&[1.0, 0.0, 0.0, 0.0]), 0.15);
        assert_eq!(pinned[0], 0.0);
        assert!(pinned[1] > 0.0);

        // 面なし・密度ゼロは有限値を返す (ゼロ除算なし)
        let bare = super::areal_inv_masses(&positions, None, None, 0.15);
        assert!(bare.iter().all(|&w| w.is_finite() && w > 0.0));
    }

    #[test]
    fn test_extract_collider_edges() {
        // 2つの三角形 (0,1,2) と (2,1,3) が辺 (1,2) を共有
        let tri1 = GpuMeshTriangle {
            p0: [0.0, 0.0, 0.0],
            friction: 0.3,
            p1: [1.0, 0.0, 0.0],
            thickness: 0.01,
            p2: [0.0, 1.0, 0.0],
            restitution: 0.0,
            flags: 0,
            _pad: [0.0; 3],
        };
        let tri2 = GpuMeshTriangle {
            p0: [0.0, 1.0, 0.0],
            friction: 0.5,
            p1: [1.0, 0.0, 0.0],
            thickness: 0.02,
            p2: [1.0, 1.0, 0.0],
            restitution: 0.0,
            flags: 0,
            _pad: [0.0; 3],
        };
        let edges = super::extract_collider_edges(&[tri1, tri2]);
        // 6辺のうち共有辺が1本あるため、一意エッジは 5 本
        assert_eq!(edges.len(), 5);
        // 共有辺の厚みは max(0.01, 0.02) = 0.02 であること
        let shared = edges.iter().find(|e| {
            (e.p0 == [1.0, 0.0, 0.0] && e.p1 == [0.0, 1.0, 0.0])
                || (e.p0 == [0.0, 1.0, 0.0] && e.p1 == [1.0, 0.0, 0.0])
        });
        assert!(shared.is_some());
        assert_eq!(shared.unwrap().thickness, 0.02);
    }

    #[test]
    fn test_two_hop_rest_lengths_sewing() {
        // 2つの離れたラインセグメント: パネルA (0 - 1) と パネルB (2 - 3)
        // パネルA: x=0, y=0 と x=0.1, y=0 (長さ 0.1)
        // パネルB: x=1.0, y=0 と x=1.1, y=0 (長さ 0.1、パネルAから約0.9m離れている)
        let positions = [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
        ];
        let edges = [[0, 1], [2, 3]];
        let sewing_springs = [[1, 2]]; // 頂点1と2を縫合

        let mesh = ClothMesh::from_raw(
            &positions, &edges, None, None, Some(&sewing_springs), None, None,
            0, 0.005, 1000.0, 1000.0, 1000.0, 10.0, 1.0,
            None, None,
        );

        let get_two_hop = |v: usize| -> Vec<(u32, f32)> {
            let start = mesh.two_hop_offsets[v] as usize;
            let end = mesh.two_hop_offsets[v + 1] as usize;
            let indices = &mesh.two_hop_indices[start..end];
            let lens = &mesh.two_hop_rest_lengths[start..end];
            indices.iter().copied().zip(lens.iter().copied()).collect()
        };

        // 頂点0の2ホップ:
        // - 0自身: レスト長 0.0
        // - 1 (同一パネルの元のエッジ): レスト長 0.1
        // - 2 (縫合によって追加されたトポロジー近傍): レスト長 0.0 (自己衝突除外解除防止)
        let v0_pairs = get_two_hop(0);
        let nbr_1 = v0_pairs.iter().find(|(idx, _)| *idx == 1).unwrap();
        assert!((nbr_1.1 - 0.1).abs() < 1e-5, "同一パネル内エッジは幾何長を保持: {}", nbr_1.1);

        let nbr_2 = v0_pairs.iter().find(|(idx, _)| *idx == 2);
        if let Some(p) = nbr_2 {
            assert_eq!(p.1, 0.0, "縫合トポロジーで追加された1ホップペアはレスト長0.0: {}", p.1);
        }

        let nbr_3 = v0_pairs.iter().find(|(idx, _)| *idx == 3);
        if let Some(p) = nbr_3 {
            assert_eq!(p.1, 0.0, "縫合トポロジーで追加された2ホップペアはレスト長0.0: {}", p.1);
        }

        // 頂点1の2ホップ:
        // - 0 (元のエッジ): レスト長 0.1
        // - 2 (縫合相手): レスト長 0.0
        let v1_pairs = get_two_hop(1);
        let nbr_0_from_1 = v1_pairs.iter().find(|(idx, _)| *idx == 0).unwrap();
        assert!((nbr_0_from_1.1 - 0.1).abs() < 1e-5, "同一パネル内エッジは幾何長を保持: {}", nbr_0_from_1.1);

        let nbr_2_from_1 = v1_pairs.iter().find(|(idx, _)| *idx == 2).unwrap();
        assert_eq!(nbr_2_from_1.1, 0.0, "縫合相手ペアはレスト長0.0: {}", nbr_2_from_1.1);
    }

    #[test]
    fn test_combined_inv_masses_areal_and_linear() {
        // 0-1-2 の直角三角形 (面積 0.5 m^2) + 2-3-4 の1Dストランド (各1m, 計2m)
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
        ];
        let faces = [[0, 1, 2]];
        let strand_edges = [[2, 3], [3, 4]];
        let areal_density = 0.30; // 三角形質量 = 0.5 * 0.30 = 0.15 kg -> 各頂点 0.05 kg
        let linear_density = 0.02; // 各エッジ質量 = 1.0 * 0.02 = 0.02 kg -> 端点 0.01 kg, 中間 0.02 kg

        let inv = super::compute_combined_inv_masses(
            &positions,
            Some(&faces),
            Some(&strand_edges),
            None,
            areal_density,
            linear_density,
        );
        assert_eq!(inv.len(), 5);
        // 頂点0, 1: 面のみ 0.05 kg -> inv = 20.0
        assert!((inv[0] - 20.0).abs() < 1e-4);
        assert!((inv[1] - 20.0).abs() < 1e-4);
        // 頂点2: 面 0.05 kg + ストランド端 0.01 kg = 0.06 kg -> inv = 1 / 0.06 = 16.6667
        assert!((inv[2] - (1.0 / 0.06)).abs() < 1e-3);
        // 頂点3: ストランド中間 0.02 kg -> inv = 50.0
        assert!((inv[3] - 50.0).abs() < 1e-4);
        // 頂点4: ストランド端 0.01 kg -> inv = 100.0
        assert!((inv[4] - 100.0).abs() < 1e-3);
    }

    #[test]
    fn test_strand_1d_bending_constraints() {
        // 1Dストランド: 0 -- 1 -- 2 -- 3
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ];
        let edges = [[0, 1], [1, 2], [2, 3]];

        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            None,
            None,
            None,
            None,
            None,
            0,
            0.005,
            1000.0,
            1000.0,
            1000.0,
            10.0,
            1.0,
            None,
            None,
        );

        // 2ホップペア (0, 2) と (1, 3) の2件の曲げ拘束が生成されていること
        assert_eq!(mesh.bending_constraints.len(), 2);
        let pairs: Vec<(u32, u32)> = mesh
            .bending_constraints
            .iter()
            .map(|bc| (bc.v2.min(bc.v3), bc.v2.max(bc.v3)))
            .collect();
        assert!(pairs.contains(&(0, 2)));
        assert!(pairs.contains(&(1, 3)));
        for bc in &mesh.bending_constraints {
            assert!((bc.rest_length - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_seam_gated_bending_constraints_across_sewn_panels() {
        // 2枚の離れた直角二等辺三角形パネル:
        // パネルA: 0=(0,0,0), 1=(0,1,0), 2=(-1,0.5,0) (シーム辺 0-1, 高さ 1.0, s_a = 0.5)
        // パネルB: 3=(5,0,0), 4=(5,1,0), 5=( 6,0.5,2) (5m離れかつ傾いているが、シーム辺 3-4 長さ1.0, 高さ sqrt(1+4)=sqrt(5), s_b = 0.5)
        // 縫合ペア: (0, 3), (1, 4)
        let positions = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.5, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
            [6.0, 0.5, 0.0],
        ];
        let edges = [
            [0, 1], [1, 2], [2, 0],
            [3, 4], [4, 5], [5, 3],
        ];
        let faces = [
            [0, 1, 2],
            [4, 3, 5],
        ];
        let sewing = [[0, 3], [1, 4]];

        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            Some(&faces),
            None,
            Some(&sewing),
            None,
            None,
            0,
            0.005,
            1000.0,
            1000.0,
            1000.0,
            10.0,
            1.0,
            Some(10000.0),
            Some(true),
        );

        // 同一パネル内には共有エッジがないため、生成される曲げ拘束はシーム跨ぎの1件のみ
        assert_eq!(mesh.bending_constraints.len(), 1);
        let bc = &mesh.bending_constraints[0];
        assert_eq!((bc.v2.min(bc.v3), bc.v2.max(bc.v3)), (2, 5));
        // 初期3D距離は ||(-1,0.5,0) - (6,0.5,0)|| = 7.0 だが、平坦展開時の自然長は h_a(1.0) + h_b(1.0) = 2.0 であること
        assert!(
            (bc.rest_length - 2.0).abs() < 1e-5,
            "シーム跨ぎ曲げ拘束の自然長は初期配置距離(7.0)ではなく平坦展開距離(2.0)であること: {}",
            bc.rest_length
        );
        assert!(bc._pad[0] > 0.5, "シームゲートフラグが有効であること");
        assert!((bc._pad[1] - 1.0).abs() < 1e-5, "ゲート距離が rest_length * 0.5 であること");
    }
}

