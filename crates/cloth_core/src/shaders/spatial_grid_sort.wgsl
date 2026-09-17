struct GpuVertex {
    position: vec3<f32>,
    inv_mass: f32,
    prev_pos: vec3<f32>,
    layer_id: u32,
    velocity: vec3<f32>,
    thickness: f32,
};

struct SpatialHashParams {
    cell_size: f32,
    table_size: u32,
    num_vertices: u32,
    num_blocks: u32,
};

@group(0) @binding(0) var<storage, read> vertices: array<GpuVertex>;
@group(0) @binding(1) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> cell_starts: array<u32>;
@group(0) @binding(3) var<storage, read_write> cell_currents: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> sorted_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(6) var<uniform> params: SpatialHashParams;

fn hash_coords(coord: vec3<i32>, table_size: u32) -> u32 {
    let p1 = 73856093u;
    let p2 = 19349663u;
    let p3 = 83492791u;
    let n = (u32(coord.x) * p1) ^ (u32(coord.y) * p2) ^ (u32(coord.z) * p3);
    return n % table_size;
}

// 1. カウンタおよび作業用バッファのゼロクリア
@compute @workgroup_size(256)
fn clear_counts(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.table_size) {
        atomicStore(&cell_counts[idx], 0u);
        atomicStore(&cell_currents[idx], 0u);
    }
}

// 2. 各セルハッシュ内の頂点数をカウント
@compute @workgroup_size(256)
fn count_vertices(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_vertices) {
        return;
    }
    let p = vertices[idx].prev_pos;
    let cell = vec3<i32>(floor(p / params.cell_size));
    let h = hash_coords(cell, params.table_size);
    atomicAdd(&cell_counts[h], 1u);
}

// 3. ブロック内 Prefix Sum (Blelloch Scan, 256要素)
var<workgroup> temp_block: array<u32, 256>;

@compute @workgroup_size(256)
fn scan_blocks(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let g_idx = global_id.x;
    let l_idx = local_id.x;

    // グローバルメモリから読み込み
    if (g_idx < params.table_size) {
        temp_block[l_idx] = atomicLoad(&cell_counts[g_idx]);
    } else {
        temp_block[l_idx] = 0u;
    }
    workgroupBarrier();

    // Up-Sweep (Reduce)
    var stride = 1u;
    while (stride < 256u) {
        let index = (l_idx + 1u) * stride * 2u - 1u;
        if (index < 256u) {
            temp_block[index] += temp_block[index - stride];
        }
        stride = stride * 2u;
        workgroupBarrier();
    }

    // ブロック総和を保存し、ルートを0に初期化
    if (l_idx == 0u) {
        block_sums[wg_id.x] = temp_block[255];
        temp_block[255] = 0u;
    }
    workgroupBarrier();

    // Down-Sweep
    stride = 128u;
    while (stride > 0u) {
        let index = (l_idx + 1u) * stride * 2u - 1u;
        if (index < 256u) {
            let t = temp_block[index - stride];
            temp_block[index - stride] = temp_block[index];
            temp_block[index] += t;
        }
        stride = stride / 2u;
        workgroupBarrier();
    }

    // ブロック内排他プレフィックスサムを書き込み
    if (g_idx < params.table_size) {
        cell_starts[g_idx] = temp_block[l_idx];
    }
}

// 4. block_sums（最大256要素）の Top Scan
var<workgroup> temp_top: array<u32, 256>;

@compute @workgroup_size(256)
fn scan_top(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let l_idx = local_id.x;
    if (l_idx < params.num_blocks) {
        temp_top[l_idx] = block_sums[l_idx];
    } else {
        temp_top[l_idx] = 0u;
    }
    workgroupBarrier();

    // Up-Sweep
    var stride = 1u;
    while (stride < 256u) {
        let index = (l_idx + 1u) * stride * 2u - 1u;
        if (index < 256u) {
            temp_top[index] += temp_top[index - stride];
        }
        stride = stride * 2u;
        workgroupBarrier();
    }

    if (l_idx == 0u) {
        temp_top[255] = 0u;
    }
    workgroupBarrier();

    // Down-Sweep
    stride = 128u;
    while (stride > 0u) {
        let index = (l_idx + 1u) * stride * 2u - 1u;
        if (index < 256u) {
            let t = temp_top[index - stride];
            temp_top[index - stride] = temp_top[index];
            temp_top[index] += t;
        }
        stride = stride / 2u;
        workgroupBarrier();
    }

    if (l_idx < params.num_blocks) {
        block_sums[l_idx] = temp_top[l_idx];
    }
}

// 5. 各ブロックに block_sums オフセットを加算
@compute @workgroup_size(256)
fn add_offsets(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let g_idx = global_id.x;
    if (g_idx < params.table_size) {
        let base = block_sums[wg_id.x];
        cell_starts[g_idx] += base;
    }

    // 配列末尾 (table_size 番目) に総頂点数を格納
    if (g_idx == params.table_size - 1u) {
        let count_last = atomicLoad(&cell_counts[g_idx]);
        cell_starts[params.table_size] = cell_starts[g_idx] + count_last;
    }
}

// 6. 各頂点をソート済みインデックス配列にスキャッター配置
@compute @workgroup_size(256)
fn scatter_indices(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_vertices) {
        return;
    }
    let p = vertices[idx].prev_pos;
    let cell = vec3<i32>(floor(p / params.cell_size));
    let h = hash_coords(cell, params.table_size);

    let base = cell_starts[h];
    let offset = atomicAdd(&cell_currents[h], 1u);
    sorted_indices[base + offset] = idx;
}
