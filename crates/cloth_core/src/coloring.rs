use crate::mesh::{GpuBendingConstraint, GpuDistanceConstraint, GpuSewingConstraint};

/// 距離拘束をグラフ彩色し、頂点競合が起きない色グループ順に拘束を再配置する
/// 戻り値: (sorted_constraints, color_offsets, color_counts, remap)
/// remap[orig_idx] = sorted_idx
pub fn color_distance_constraints(
    num_vertices: usize,
    constraints: &[GpuDistanceConstraint],
) -> (Vec<GpuDistanceConstraint>, Vec<u32>, Vec<u32>, Vec<usize>) {
    if constraints.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    let num_constraints = constraints.len();
    let mut constraint_colors = vec![u32::MAX; num_constraints];

    let mut vertex_to_constraints: Vec<Vec<usize>> = vec![Vec::new(); num_vertices];
    for (i, c) in constraints.iter().enumerate() {
        vertex_to_constraints[c.v0 as usize].push(i);
        vertex_to_constraints[c.v1 as usize].push(i);
    }

    // Welsh-Powell法: 拘束の接続次数（隣接する拘束数）が大きい順に彩色順序をソート (タイブレークとしてインデックスを使用し完全決定論化)
    let mut order: Vec<usize> = (0..num_constraints).collect();
    order.sort_by_key(|&i| {
        let c = &constraints[i];
        let deg = vertex_to_constraints[c.v0 as usize].len() + vertex_to_constraints[c.v1 as usize].len();
        (std::cmp::Reverse(deg), i)
    });

    let mut num_colors = 0;

    for &i in &order {
        let c = &constraints[i];
        let mut used_colors = std::collections::HashSet::new();

        for &adj_i in &vertex_to_constraints[c.v0 as usize] {
            if adj_i != i && constraint_colors[adj_i] != u32::MAX {
                used_colors.insert(constraint_colors[adj_i]);
            }
        }
        for &adj_i in &vertex_to_constraints[c.v1 as usize] {
            if adj_i != i && constraint_colors[adj_i] != u32::MAX {
                used_colors.insert(constraint_colors[adj_i]);
            }
        }

        let mut color = 0;
        while used_colors.contains(&color) {
            color += 1;
        }

        constraint_colors[i] = color;
        if color >= num_colors {
            num_colors = color + 1;
        }
    }

    let mut color_buckets: Vec<Vec<(usize, GpuDistanceConstraint)>> = vec![Vec::new(); num_colors as usize];
    for (i, &color) in constraint_colors.iter().enumerate() {
        color_buckets[color as usize].push((i, constraints[i]));
    }

    let mut sorted_constraints = Vec::with_capacity(num_constraints);
    let mut color_offsets = Vec::with_capacity(num_colors as usize);
    let mut color_counts = Vec::with_capacity(num_colors as usize);
    let mut remap = vec![0; num_constraints];

    for bucket in color_buckets {
        color_offsets.push(sorted_constraints.len() as u32);
        color_counts.push(bucket.len() as u32);
        for (orig_idx, constraint) in bucket {
            remap[orig_idx] = sorted_constraints.len();
            sorted_constraints.push(constraint);
        }
    }

    (sorted_constraints, color_offsets, color_counts, remap)
}

/// 曲げ拘束（4頂点）をグラフ彩色し、頂点競合が起きない色グループ順に拘束を再配置する
pub fn color_bending_constraints(
    num_vertices: usize,
    constraints: &[GpuBendingConstraint],
) -> (Vec<GpuBendingConstraint>, Vec<u32>, Vec<u32>) {
    if constraints.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let num_constraints = constraints.len();
    let mut constraint_colors = vec![u32::MAX; num_constraints];

    let mut vertex_to_constraints: Vec<Vec<usize>> = vec![Vec::new(); num_vertices];
    for (i, c) in constraints.iter().enumerate() {
        vertex_to_constraints[c.v2 as usize].push(i);
        vertex_to_constraints[c.v3 as usize].push(i);
    }

    // Welsh-Powell法: 拘束の接続次数が大きい順にソート (タイブレークとしてインデックスを使用)
    let mut order: Vec<usize> = (0..num_constraints).collect();
    order.sort_by_key(|&i| {
        let c = &constraints[i];
        let deg = vertex_to_constraints[c.v2 as usize].len() + vertex_to_constraints[c.v3 as usize].len();
        (std::cmp::Reverse(deg), i)
    });

    let mut num_colors = 0;

    for &i in &order {
        let c = &constraints[i];
        let mut used_colors = std::collections::HashSet::new();

        let verts = [c.v2, c.v3];
        for &v in &verts {
            for &adj_i in &vertex_to_constraints[v as usize] {
                if adj_i != i && constraint_colors[adj_i] != u32::MAX {
                    used_colors.insert(constraint_colors[adj_i]);
                }
            }
        }

        let mut color = 0;
        while used_colors.contains(&color) {
            color += 1;
        }

        constraint_colors[i] = color;
        if color >= num_colors {
            num_colors = color + 1;
        }
    }

    let mut color_buckets: Vec<Vec<GpuBendingConstraint>> = vec![Vec::new(); num_colors as usize];
    for (i, &color) in constraint_colors.iter().enumerate() {
        color_buckets[color as usize].push(constraints[i]);
    }

    let mut sorted_constraints = Vec::with_capacity(num_constraints);
    let mut color_offsets = Vec::with_capacity(num_colors as usize);
    let mut color_counts = Vec::with_capacity(num_colors as usize);

    for bucket in color_buckets {
        color_offsets.push(sorted_constraints.len() as u32);
        color_counts.push(bucket.len() as u32);
        sorted_constraints.extend(bucket);
    }

    (sorted_constraints, color_offsets, color_counts)
}

/// 汎用エッジペア (v0, v1) に対するWelsh-Powell彩色。
/// 距離拘束・縫合拘束いずれの可視化・検証用途でも使えるよう、
/// 拘束ペイロードに依存せず頂点競合のみで色番号を付与する。
/// 戻り値: 拘束ごとの色番号 (長さ = pairs.len())。
/// 範囲外頂点を含むペアは隣接登録をスキップし、色0として扱う。
pub fn color_edge_pairs(num_vertices: usize, pairs: &[[u32; 2]]) -> Vec<u32> {
    let n = pairs.len();
    if n == 0 || num_vertices == 0 {
        return vec![0; n];
    }
    let mut vertex_to_constraints: Vec<Vec<usize>> = vec![Vec::new(); num_vertices];
    for (i, &[v0, v1]) in pairs.iter().enumerate() {
        let (a, b) = (v0 as usize, v1 as usize);
        if a < num_vertices {
            vertex_to_constraints[a].push(i);
        }
        if b < num_vertices {
            vertex_to_constraints[b].push(i);
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| {
        let [v0, v1] = pairs[i];
        let (a, b) = (v0 as usize, v1 as usize);
        let deg = vertex_to_constraints.get(a).map_or(0, |v| v.len())
            + vertex_to_constraints.get(b).map_or(0, |v| v.len());
        (std::cmp::Reverse(deg), i)
    });
    let mut colors = vec![u32::MAX; n];
    let mut num_colors = 0u32;
    for &i in &order {
        let [v0, v1] = pairs[i];
        let (a, b) = (v0 as usize, v1 as usize);
        let mut used = std::collections::HashSet::new();
        if let Some(list) = vertex_to_constraints.get(a) {
            for &adj in list {
                if adj != i && colors[adj] != u32::MAX {
                    used.insert(colors[adj]);
                }
            }
        }
        if let Some(list) = vertex_to_constraints.get(b) {
            for &adj in list {
                if adj != i && colors[adj] != u32::MAX {
                    used.insert(colors[adj]);
                }
            }
        }
        let mut color = 0u32;
        while used.contains(&color) {
            color += 1;
        }
        colors[i] = color;
        if color >= num_colors {
            num_colors = color + 1;
        }
    }
    colors
}

/// 縫合拘束をグラフ彩色し、頂点競合が起きない色グループ順に拘束を再配置する
pub fn color_sewing_constraints(
    num_vertices: usize,
    constraints: &[GpuSewingConstraint],
) -> (Vec<GpuSewingConstraint>, Vec<u32>, Vec<u32>) {
    if constraints.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let num_constraints = constraints.len();
    let mut constraint_colors = vec![u32::MAX; num_constraints];

    let mut vertex_to_constraints: Vec<Vec<usize>> = vec![Vec::new(); num_vertices];
    for (i, c) in constraints.iter().enumerate() {
        vertex_to_constraints[c.v0 as usize].push(i);
        vertex_to_constraints[c.v1 as usize].push(i);
    }

    // Welsh-Powell法: 拘束の接続次数が大きい順にソート (タイブレークとしてインデックスを使用)
    let mut order: Vec<usize> = (0..num_constraints).collect();
    order.sort_by_key(|&i| {
        let c = &constraints[i];
        let deg = vertex_to_constraints[c.v0 as usize].len() + vertex_to_constraints[c.v1 as usize].len();
        (std::cmp::Reverse(deg), i)
    });

    let mut num_colors = 0;

    for &i in &order {
        let c = &constraints[i];
        let mut used_colors = std::collections::HashSet::new();

        for &adj_i in &vertex_to_constraints[c.v0 as usize] {
            if adj_i != i && constraint_colors[adj_i] != u32::MAX {
                used_colors.insert(constraint_colors[adj_i]);
            }
        }
        for &adj_i in &vertex_to_constraints[c.v1 as usize] {
            if adj_i != i && constraint_colors[adj_i] != u32::MAX {
                used_colors.insert(constraint_colors[adj_i]);
            }
        }

        let mut color = 0;
        while used_colors.contains(&color) {
            color += 1;
        }

        constraint_colors[i] = color;
        if color >= num_colors {
            num_colors = color + 1;
        }
    }

    let mut color_buckets: Vec<Vec<GpuSewingConstraint>> = vec![Vec::new(); num_colors as usize];
    for (i, &color) in constraint_colors.iter().enumerate() {
        color_buckets[color as usize].push(constraints[i]);
    }

    let mut sorted_constraints = Vec::with_capacity(num_constraints);
    let mut color_offsets = Vec::with_capacity(num_colors as usize);
    let mut color_counts = Vec::with_capacity(num_colors as usize);

    for bucket in color_buckets {
        color_offsets.push(sorted_constraints.len() as u32);
        color_counts.push(bucket.len() as u32);
        sorted_constraints.extend(bucket);
    }

    (sorted_constraints, color_offsets, color_counts)
}
