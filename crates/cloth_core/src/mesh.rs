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
    pub _pad0: u32,
    pub relief_factor: f32,
    pub max_displacement_ratio: f32,
    pub enable_relief: u32,
    pub enable_normal_untangling: u32,
    pub exclude_neighbors: u32,
    pub max_search_iterations: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuTriangle {
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
    pub layer_id: u32,
}

pub struct ClothMesh {
    pub vertices: Vec<GpuVertex>,
    pub distance_constraints: Vec<GpuDistanceConstraint>,
    pub dist_color_offsets: Vec<u32>,
    pub dist_color_counts: Vec<u32>,
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
    pub local_edge_lengths: Vec<f32>,
    pub star_offsets: Vec<u32>,
    pub star_indices: Vec<GpuStarPair>,
    pub triangles: Vec<GpuTriangle>,
    pub island_ids: Vec<u32>,
}

impl ClothMesh {
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

        let tension_compliance = if tension_stiffness >= 5000.0 {
            0.0 // 完全非伸縮 (Inextensible PBD)
        } else if tension_stiffness > 0.0 {
            1.0 / (tension_stiffness * 1000.0)
        } else {
            1e10
        };

        let compression_compliance = if compression_stiffness >= 5000.0 {
            0.0
        } else if compression_stiffness > 0.0 {
            1.0 / (compression_stiffness * 1000.0)
        } else {
            1e10
        };

        let shear_compliance = if shear_stiffness >= 5000.0 {
            0.0
        } else if shear_stiffness > 0.0 {
            1.0 / (shear_stiffness * 1000.0)
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

            for opps in edge_to_opp.values() {
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
            1.0 / (bending_stiffness * 100.0)
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

            for ((v0, v1), opps) in edge_to_opp_verts {
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
        }

        let (sorted_bend, bend_color_offsets, bend_color_counts) =
            crate::coloring::color_bending_constraints(n_verts, &bending_constraints);

        // 3. 縫合拘束
        let mut sewing_constraints = Vec::new();
        let sew_stiff = sewing_stiffness.unwrap_or(10000.0);
        let sew_compliance = if sew_stiff >= 5000.0 {
            0.0 // 完全非伸縮 (Inextensible PBD)
        } else if sew_stiff > 0.0 {
            1.0 / (sew_stiff * 1000.0)
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

        for &[v0, v1] in edges {
            if (v0 as usize) < n_verts && (v1 as usize) < n_verts && v0 != v1 {
                let p0 = positions[v0 as usize];
                let p1 = positions[v1 as usize];
                let dx = p0[0] - p1[0];
                let dy = p0[1] - p1[1];
                let dz = p0[2] - p1[2];
                let len = (dx * dx + dy * dy + dz * dz).sqrt();

                adj_lists[v0 as usize].push(v1);
                adj_lists[v1 as usize].push(v0);
                edge_len_sums[v0 as usize] += len;
                edge_counts[v0 as usize] += 1;
                edge_len_sums[v1 as usize] += len;
                edge_counts[v1 as usize] += 1;
            }
        }

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

        Self {
            vertices,
            distance_constraints: sorted_dist,
            dist_color_offsets,
            dist_color_counts,
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
            local_edge_lengths,
            star_offsets,
            star_indices,
            triangles: triangles_vec,
            island_ids,
        }
    }
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

}
