//! 仮想コライダー粒子（Virtual Collision Particles）の生成と管理
//!
//! 自己衝突において V-T / E-E 幾何計算を排し、規則的サンプリングによる
//! 仮想頂点（スレーブ粒子）を配置して高速な V-V（球対球）判定のみで自己交差を遮断します。

use std::collections::HashMap;

/// GPU側で参照する仮想頂点（サンプリング粒子）定義
/// WGSL側の `struct GpuVirtualVertexDef` と 100% バイナリ互換（32 bytes）
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVirtualVertexDef {
    /// 親三角形の3頂点インデックス
    pub parent_indices: [u32; 3],
    /// 親三角形のID（トポロジー除外・自爆防止用）
    pub parent_face_id: u32,
    /// 重心座標 (u, v, w)（u + v + w = 1.0）
    pub bary_weights: [f32; 3],
    /// 衝突球半径（厚み）
    pub thickness: f32,
}

/// 仮想頂点 Forward パス用 Uniform パラメータ (16 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VirtualForwardParams {
    pub num_real_vertices: u32,
    pub num_virtual_vertices: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// V-V 自己衝突パス用 Uniform パラメータ (32 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SelfCollisionVvParams {
    pub cell_size: f32,
    pub table_size: u32,
    pub num_real_vertices: u32,
    pub num_total_particles: u32,
    pub relief_factor: f32,
    pub max_displacement_ratio: f32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// 仮想頂点サンプリング結果
#[derive(Clone, Debug, Default)]
pub struct VirtualMeshSampling {
    /// 仮想頂点定義リスト
    pub virtual_defs: Vec<GpuVirtualVertexDef>,
    /// エッジ上に生成された仮想頂点数
    pub edge_particle_count: usize,
    /// 面内部に生成された仮想頂点数
    pub interior_particle_count: usize,
}

/// メッシュの各エッジ・面に対して規則的なサンプリングを行い、仮想頂点群を生成する
///
/// # 引数
/// - `positions`: メッシュ頂点の初期座標
/// - `faces`: 三角面インデックス配列（3頂点インデックスのタプル）
/// - `edges`: エッジインデックス配列（2頂点インデックスのタプル）
/// - `thickness`: 衝突厚み（球半径）
/// - `target_spacing`: サンプリング目標間隔 (m)（通常 2.0 * thickness 〜 2.8 * thickness）
pub fn generate_virtual_vertices(
    positions: &[[f32; 3]],
    faces: &[[u32; 3]],
    edges: &[[u32; 2]],
    thickness: f32,
    target_spacing: f32,
) -> VirtualMeshSampling {
    if target_spacing <= 1e-6 || positions.is_empty() || faces.is_empty() {
        return VirtualMeshSampling::default();
    }

    let mut virtual_defs = Vec::new();

    // 1. 各エッジを含む代表三角形を検索するためのマップを構築
    // キー: 正規化エッジ (min(u, v), max(u, v)) -> (親face_idx, [a, b, c], 局所インデックス)
    let mut edge_to_face: HashMap<(u32, u32), (u32, [u32; 3])> = HashMap::with_capacity(edges.len());
    for (f_idx, &[a, b, c]) in faces.iter().enumerate() {
        let tri = [a, b, c];
        let e0 = if a < b { (a, b) } else { (b, a) };
        let e1 = if b < c { (b, c) } else { (c, b) };
        let e2 = if c < a { (c, a) } else { (a, c) };
        edge_to_face.entry(e0).or_insert((f_idx as u32, tri));
        edge_to_face.entry(e1).or_insert((f_idx as u32, tri));
        edge_to_face.entry(e2).or_insert((f_idx as u32, tri));
    }

    // 2. エッジサンプリング（一意性保証）
    let mut edge_count = 0;
    for &[u, v] in edges {
        let key = if u < v { (u, v) } else { (v, u) };
        let (parent_face_id, parent_tri) = match edge_to_face.get(&key) {
            Some(&res) => res,
            None => continue, // 面に属さない孤立エッジはスキップ
        };

        let pu = positions[u as usize];
        let pv = positions[v as usize];
        let dx = pv[0] - pu[0];
        let dy = pv[1] - pu[1];
        let dz = pv[2] - pu[2];
        let length = (dx * dx + dy * dy + dz * dz).sqrt();

        let ke = (length / target_spacing).ceil() as u32;
        if ke > 1 {
            // 親三角形内での u と v の位置を特定
            let u_slot = parent_tri.iter().position(|&x| x == u).unwrap();
            let v_slot = parent_tri.iter().position(|&x| x == v).unwrap();

            for m in 1..ke {
                let t = m as f32 / ke as f32; // u -> v への内分比
                let mut bary = [0.0f32; 3];
                bary[u_slot] = 1.0 - t;
                bary[v_slot] = t;

                virtual_defs.push(GpuVirtualVertexDef {
                    parent_indices: parent_tri,
                    parent_face_id,
                    bary_weights: bary,
                    thickness,
                });
                edge_count += 1;
            }
        }
    }

    // 3. 面内部サンプリング（規則的重心グリッド）
    let mut interior_count = 0;
    for (f_idx, &tri) in faces.iter().enumerate() {
        let [a, b, c] = tri;
        let pa = positions[a as usize];
        let pb = positions[b as usize];
        let pc = positions[c as usize];

        let l_ab = dist(pa, pb);
        let l_bc = dist(pb, pc);
        let l_ca = dist(pc, pa);
        let l_max = l_ab.max(l_bc).max(l_ca);

        let k = (l_max / target_spacing).ceil() as u32;
        if k >= 3 {
            // 面内部の点: i >= 1, j >= 1, i + j < k
            for i in 1..k {
                for j in 1..(k - i) {
                    let u = i as f32 / k as f32;
                    let v = j as f32 / k as f32;
                    let w = 1.0 - u - v;

                    virtual_defs.push(GpuVirtualVertexDef {
                        parent_indices: tri,
                        parent_face_id: f_idx as u32,
                        bary_weights: [u, v, w],
                        thickness,
                    });
                    interior_count += 1;
                }
            }
        }
    }

    VirtualMeshSampling {
        virtual_defs,
        edge_particle_count: edge_count,
        interior_particle_count: interior_count,
    }
}

#[inline(always)]
fn dist(p0: [f32; 3], p1: [f32; 3]) -> f32 {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let dz = p1[2] - p0[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virtual_vertex_def_layout() {
        assert_eq!(std::mem::size_of::<GpuVirtualVertexDef>(), 32);
        assert_eq!(std::mem::align_of::<GpuVirtualVertexDef>(), 4);
        assert_eq!(std::mem::size_of::<VirtualForwardParams>(), 16);
        assert_eq!(std::mem::align_of::<VirtualForwardParams>(), 4);
        assert_eq!(std::mem::size_of::<SelfCollisionVvParams>(), 32);
        assert_eq!(std::mem::align_of::<SelfCollisionVvParams>(), 4);
    }

    #[test]
    fn test_generate_virtual_vertices_simple_triangle() {
        // 一辺 10mm の正三角形
        let positions = vec![
            [0.0, 0.0, 0.0],
            [0.010, 0.0, 0.0],
            [0.005, 0.00866, 0.0],
        ];
        let faces = vec![[0, 1, 2]];
        let edges = vec![[0, 1], [1, 2], [2, 0]];
        let thickness = 0.002; // 2mm
        let target_spacing = 0.004; // 4mm -> 各エッジ 10mm / 4mm = ceil(2.5) = 3 分割 (各2点追加)

        let sampling = generate_virtual_vertices(&positions, &faces, &edges, thickness, target_spacing);
        
        // 各エッジで 2 点ずつ追加 -> 3 * 2 = 6 点
        assert_eq!(sampling.edge_particle_count, 6);
        // k = 3 のとき面内部は i=1, j=1 で 1点 (1/3, 1/3, 1/3)
        assert_eq!(sampling.interior_particle_count, 1);
        assert_eq!(sampling.virtual_defs.len(), 7);

        // 重心座標の和が常に 1.0 かつ親三角形が正しいことを検証
        for def in &sampling.virtual_defs {
            assert_eq!(def.parent_indices, [0, 1, 2]);
            assert_eq!(def.parent_face_id, 0);
            assert_eq!(def.thickness, thickness);
            let sum: f32 = def.bary_weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
