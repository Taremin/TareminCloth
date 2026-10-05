//! プロセス全体で共有する GPU コンピュートパイプラインキャッシュ。
//!
//! 背景: `ClothSimulator` の生成ごとに WGSL シェーダ 16 本・コンピュート
//! パイプライン 17 本 (+空間ハッシュ 6 本) をコンパイルしていたため、
//! インタラクティブモード起動の stage 3「GPUを初期化中...」が約 5.5 秒を
//! 占めていた (実測: RX 9070 XT / DX12)。
//!
//! 本モジュールはメッシュ非依存部分 (シェーダ・BGL・パイプライン) を
//! `(デバイス世代, workgroup_size)` キーで共有し、未使用機能のパイプライン
//! (Atomic ソルバー / PairCache / Edge衝突 / 自己衝突系 / 空間ハッシュ) を
//! 初回有効化時まで遅延生成する。per-sim 側はバッファと BindGroup のみを
//! 保持し、ディスパッチ時は共有パイプラインを参照する。
//!
//! 共有 BGL オブジェクトを使い回すことで、per-sim BindGroup と共有
//! パイプライン間のレイアウト互換性を構造的に保証している。

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use crate::context::GpuContext;

/// `workgroup_size` の正規化 (32 / 64 以外は 32 にフォールバック)。
/// キャッシュキーの発散防止と無効値の素通し防止を兼ねる。
pub fn normalize_workgroup_size(wg_size: u32) -> u32 {
    if wg_size == 64 {
        64
    } else {
        32
    }
}

pub fn create_shader_with_wg_size(
    device: &wgpu::Device,
    label: &str,
    src: &str,
    wg_size: u32,
) -> wgpu::ShaderModule {
    let source = if wg_size != 64 {
        src.replace("@workgroup_size(64)", &format!("@workgroup_size({})", wg_size))
    } else {
        src.to_string()
    };
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

/// 遅延生成: Atomic Jacobi ソルバー用パイプライン対
pub struct AtomicPipelines {
    pub solve: wgpu::ComputePipeline,
    pub apply: wgpu::ComputePipeline,
}

/// 遅延生成: Active Pair Caching 用パイプライン群
pub struct PairPipelines {
    pub collect: wgpu::ComputePipeline,
    pub solve_vt: wgpu::ComputePipeline,
    pub solve_ee: wgpu::ComputePipeline,
}

/// 遅延生成: 自己衝突ダイレクト走査用パイプライン群
pub struct SelfCollisionPipelines {
    pub solve: wgpu::ComputePipeline,
    pub solve_ee: wgpu::ComputePipeline,
    pub apply: wgpu::ComputePipeline,
    pub normals: wgpu::ComputePipeline,
}

/// 遅延生成: 空間ハッシュ構築用パイプライン群 (`workgroup_size` 非依存)
pub struct HashPipelines {
    pub clear: wgpu::ComputePipeline,
    pub count: wgpu::ComputePipeline,
    pub scan_blocks: wgpu::ComputePipeline,
    pub scan_top: wgpu::ComputePipeline,
    pub add_offsets: wgpu::ComputePipeline,
    pub scatter: wgpu::ComputePipeline,
}

/// メッシュ非依存の共有 GPU リソース。
/// キーは `(デバイス世代, workgroup_size, コンテキスト同一性)`。
/// 同一プロセス内で複数デバイスが存在し得る (並列テスト等) ため、
/// 世代だけでなく `Arc` ポインタによる同一性検証が必須。
pub struct SharedPipelines {
    pub epoch: u64,
    pub workgroup_size: u32,
    context: Arc<GpuContext>,

    // --- BindGroupLayout (作成は µs オーダーのため全て eager) ---
    pub predict_bgl: wgpu::BindGroupLayout,
    pub constraint_bgl: wgpu::BindGroupLayout,
    pub sewing_bgl: wgpu::BindGroupLayout,
    pub pin_bgl: wgpu::BindGroupLayout,
    pub collision_bgl: wgpu::BindGroupLayout,
    pub edge_collision_bgl: wgpu::BindGroupLayout,
    pub update_vel_bgl: wgpu::BindGroupLayout,
    pub distance_atomic_bgl: wgpu::BindGroupLayout,
    pub extract_positions_bgl: wgpu::BindGroupLayout,
    pub self_collision_bgl: wgpu::BindGroupLayout,
    pub self_collision_ee_bgl: wgpu::BindGroupLayout,
    pub self_collision_apply_bgl: wgpu::BindGroupLayout,
    pub compute_normals_bgl: wgpu::BindGroupLayout,
    pub pair_collect_bgl: wgpu::BindGroupLayout,
    pub pair_solve_vt_bgl: wgpu::BindGroupLayout,
    pub pair_solve_ee_bgl: wgpu::BindGroupLayout,
    pub hash_bgl: wgpu::BindGroupLayout,
    pub edge_hash_bgl: wgpu::BindGroupLayout,
    pub sew_shrink_bgl: wgpu::BindGroupLayout,
    pub wrinkle_field_bgl: wgpu::BindGroupLayout,
    pub virtual_forward_bgl: wgpu::BindGroupLayout,
    pub self_collision_vv_bgl: wgpu::BindGroupLayout,

    // --- 常時パイプライン (eager・8 本) ---
    pub predict_pipeline: wgpu::ComputePipeline,
    pub distance_pipeline: wgpu::ComputePipeline,
    pub bending_pipeline: wgpu::ComputePipeline,
    pub sewing_pipeline: wgpu::ComputePipeline,
    pub pin_pipeline: wgpu::ComputePipeline,
    pub collision_pipeline: wgpu::ComputePipeline,
    pub update_vel_pipeline: wgpu::ComputePipeline,
    pub extract_positions_pipeline: wgpu::ComputePipeline,

    // --- 遅延パイプライン ---
    atomic: OnceLock<AtomicPipelines>,
    pair: OnceLock<PairPipelines>,
    edge: OnceLock<wgpu::ComputePipeline>,
    self_collision: OnceLock<SelfCollisionPipelines>,
    hash: OnceLock<HashPipelines>,
    edge_hash: OnceLock<HashPipelines>,
    sew_shrink: OnceLock<wgpu::ComputePipeline>,
    wrinkle_field: OnceLock<wgpu::ComputePipeline>,
    virtual_forward: OnceLock<wgpu::ComputePipeline>,
    self_collision_vv: OnceLock<wgpu::ComputePipeline>,

    /// ビルド計測ログ (パイプライン名, ミリ秒)。eager + 遅延分を追記する。
    timings: Mutex<Vec<(String, f32)>>,
    /// eager 構築に要した合計ミリ秒 (スナップショット用)
    eager_ms: f32,
}

fn lock_timings(timings: &Mutex<Vec<(String, f32)>>) -> MutexGuard<'_, Vec<(String, f32)>> {
    timings.lock().unwrap_or_else(|e| e.into_inner())
}

fn timed<T>(
    timings: &Mutex<Vec<(String, f32)>>,
    name: &str,
    f: impl FnOnce() -> T,
) -> T {
    let t0 = Instant::now();
    let v = f();
    lock_timings(timings).push((name.to_string(), t0.elapsed().as_secs_f32() * 1000.0));
    v
}

fn compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn pipeline_layout(
    device: &wgpu::Device,
    label: &str,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    })
}

impl SharedPipelines {
    fn build(context: &Arc<GpuContext>, workgroup_size: u32) -> Self {
        let device = &context.device;
        let timings: Mutex<Vec<(String, f32)>> = Mutex::new(Vec::new());
        let t_all = Instant::now();

        // --- BGL (安価・eager) ---
        let storage_rw = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_rw_at = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let predict_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Predict Bind Group Layout"),
            entries: &[storage_rw, uniform_entry(1)],
        });
        let constraint_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Constraint Bind Group Layout"),
            entries: &[storage_rw, storage_ro(1), uniform_entry(2), uniform_entry(3), storage_rw_at(4)],
        });
        let sewing_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sewing Bind Group Layout"),
            entries: &[storage_rw, storage_rw_at(1), uniform_entry(2), uniform_entry(3), storage_rw_at(4)],
        });
        let pin_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Pin Bind Group Layout"),
            entries: &[storage_rw, storage_ro(1), uniform_entry(2), uniform_entry(3)],
        });
        let collision_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Collision Bind Group Layout"),
            entries: &[
                storage_rw,
                storage_ro(1),
                storage_ro(2),
                uniform_entry(3),
                storage_ro(4),
                storage_ro(5),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                storage_ro(8),
                storage_ro(9),
            ],
        });
        let edge_collision_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Edge Collision Bind Group Layout"),
                entries: &[
                    storage_ro(0),
                    storage_ro(1),
                    storage_ro(2),
                    storage_ro(3),
                    storage_ro(4),
                    uniform_entry(5),
                    storage_rw_at(6),
                ],
            });
        let update_vel_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Update Vel Bind Group Layout"),
            entries: &[storage_rw, uniform_entry(1)],
        });
        let distance_atomic_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Distance Atomic Bind Group Layout"),
                entries: &[storage_rw, storage_ro(1), uniform_entry(2), storage_rw_at(3), storage_rw_at(4)],
            });
        let extract_positions_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Extract Positions Bind Group Layout"),
                entries: &[storage_ro(0), storage_rw_at(1)],
            });
        let self_collision_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Self Collision Bind Group Layout"),
                entries: &[
                    storage_rw,
                    storage_ro(1),
                    storage_ro(2),
                    uniform_entry(3),
                    storage_ro(4),
                    storage_ro(5),
                    storage_ro(6),
                    storage_ro(7),
                    storage_ro(8),
                    storage_ro(9),
                    storage_ro(10),
                    storage_rw_at(11),
                    storage_ro(12),
                    storage_ro(13),
                    storage_ro(14), // two_hop_rest_lengths
                ],
            });
        let self_collision_ee_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Self Collision EE Bind Group Layout"),
                entries: &[
                    storage_ro(0),     // edges
                    storage_ro(1),     // vertices
                    storage_ro(2),     // edge_cell_starts
                    storage_ro(3),     // edge_sorted_indices
                    uniform_entry(4),  // params
                    storage_ro(5),     // normals
                    storage_ro(6),     // two_hop_offsets
                    storage_ro(7),     // two_hop_indices
                    storage_rw_at(8),  // accum
                    storage_ro(9),     // two_hop_rest_lengths
                ],
            });
        let self_collision_apply_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Self Collision Apply Bind Group Layout"),
                entries: &[storage_rw, storage_rw_at(1), uniform_entry(2), storage_ro(3)],
            });
        let compute_normals_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Normals Bind Group Layout"),
                entries: &[
                    storage_ro(0),
                    storage_ro(1),
                    storage_ro(2),
                    storage_rw_at(3),
                    uniform_entry(4),
                ],
            });
        let pair_collect_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Pair Collect Bind Group Layout"),
                entries: &[
                    storage_ro(0),
                    storage_ro(1),
                    storage_ro(2),
                    uniform_entry(3),
                    storage_ro(4),
                    storage_ro(5),
                    storage_ro(6),
                    storage_ro(7),
                    storage_ro(8),
                    storage_ro(9),
                    storage_ro(10),
                    storage_ro(11),
                    storage_rw_at(12),
                    storage_rw_at(13),
                    storage_rw_at(14),
                    storage_ro(15), // two_hop_rest_lengths
                ],
            });
        let pair_solve_vt_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Pair Solve VT Bind Group Layout"),
                entries: &[
                    storage_ro(0),
                    storage_ro(1),
                    storage_ro(2),
                    storage_rw_at(3),
                    uniform_entry(4),
                    storage_ro(5),
                ],
            });
        let pair_solve_ee_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Pair Solve EE Bind Group Layout"),
                entries: &[
                    storage_ro(0),
                    storage_ro(1),
                    storage_ro(2),
                    storage_rw_at(3),
                    uniform_entry(4),
                    storage_ro(5),
                ],
            });
        let hash_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SpatialGrid Sort BGL"),
            entries: &[
                storage_ro(0),
                storage_rw_at(1),
                storage_rw_at(2),
                storage_rw_at(3),
                storage_rw_at(4),
                storage_rw_at(5),
                uniform_entry(6),
            ],
        });
        let edge_hash_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Edge SpatialGrid Sort BGL"),
            entries: &[
                storage_ro(0),    // edges
                storage_rw_at(1), // cell_counts
                storage_rw_at(2), // cell_starts
                storage_rw_at(3), // cell_currents
                storage_rw_at(4), // sorted_indices
                storage_rw_at(5), // block_sums
                uniform_entry(6), // params
                storage_ro(7),    // vertices
            ],
        });
        let sew_shrink_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sew Shrink BGL"),
            entries: &[storage_rw, uniform_entry(1)],
        });
        let wrinkle_field_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Wrinkle Field BGL"),
            entries: &[
                storage_rw,
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                uniform_entry(3),
                uniform_entry(4),
            ],
        });
        let virtual_forward_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Virtual Forward Bind Group Layout"),
            entries: &[storage_rw, storage_ro(1), uniform_entry(2)],
        });
        let self_collision_vv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Self Collision VV Bind Group Layout"),
            entries: &[
                storage_rw,
                storage_ro(1),
                storage_ro(2),
                uniform_entry(3),
                storage_rw_at(4),
                storage_ro(5),
                storage_ro(6),
            ],
        });

        // --- 常時シェーダ (eager・8 本) ---
        let predict_shader = create_shader_with_wg_size(
            device,
            "Predict Shader",
            include_str!("../shaders/predict.wgsl"),
            workgroup_size,
        );
        let distance_shader = create_shader_with_wg_size(
            device,
            "Distance Shader",
            include_str!("../shaders/distance.wgsl"),
            workgroup_size,
        );
        let bending_shader = create_shader_with_wg_size(
            device,
            "Bending Shader",
            include_str!("../shaders/bending.wgsl"),
            workgroup_size,
        );
        let sewing_shader = create_shader_with_wg_size(
            device,
            "Sewing Shader",
            include_str!("../shaders/sewing.wgsl"),
            workgroup_size,
        );
        let collision_shader = create_shader_with_wg_size(
            device,
            "Collision Shader",
            include_str!("../shaders/collision.wgsl"),
            workgroup_size,
        );
        let pin_shader = create_shader_with_wg_size(
            device,
            "Pin Shader",
            include_str!("../shaders/pin.wgsl"),
            workgroup_size,
        );
        let update_vel_shader = create_shader_with_wg_size(
            device,
            "Update Velocity Shader",
            include_str!("../shaders/update_vel.wgsl"),
            workgroup_size,
        );
        let extract_positions_shader = create_shader_with_wg_size(
            device,
            "Extract Positions Shader",
            include_str!("../shaders/extract_positions.wgsl"),
            workgroup_size,
        );

        // --- 常時パイプライン (eager・8 本) ---
        let predict_pl = pipeline_layout(device, "Predict Pipeline Layout", &predict_bgl);
        let predict_pipeline = timed(&timings, "predict", || {
            compute_pipeline(device, "Predict Pipeline", &predict_pl, &predict_shader, "main")
        });
        let distance_pl = pipeline_layout(device, "Distance Pipeline Layout", &constraint_bgl);
        let distance_pipeline = timed(&timings, "distance", || {
            compute_pipeline(device, "Distance Pipeline", &distance_pl, &distance_shader, "main")
        });
        let bending_pl = pipeline_layout(device, "Bending Pipeline Layout", &constraint_bgl);
        let bending_pipeline = timed(&timings, "bending", || {
            compute_pipeline(device, "Bending Pipeline", &bending_pl, &bending_shader, "main")
        });
        let sewing_pl = pipeline_layout(device, "Sewing Pipeline Layout", &sewing_bgl);
        let sewing_pipeline = timed(&timings, "sewing", || {
            compute_pipeline(device, "Sewing Pipeline", &sewing_pl, &sewing_shader, "main")
        });
        let pin_pl = pipeline_layout(device, "Pin Pipeline Layout", &pin_bgl);
        let pin_pipeline = timed(&timings, "pin", || {
            compute_pipeline(device, "Pin Pipeline", &pin_pl, &pin_shader, "main")
        });
        let collision_pl = pipeline_layout(device, "Collision Pipeline Layout", &collision_bgl);
        let collision_pipeline = timed(&timings, "collision", || {
            compute_pipeline(
                device,
                "Collision Pipeline",
                &collision_pl,
                &collision_shader,
                "main",
            )
        });
        let update_vel_pl = pipeline_layout(device, "Update Vel Pipeline Layout", &update_vel_bgl);
        let update_vel_pipeline = timed(&timings, "update_vel", || {
            compute_pipeline(
                device,
                "Update Vel Pipeline",
                &update_vel_pl,
                &update_vel_shader,
                "main",
            )
        });
        let extract_positions_pl = pipeline_layout(
            device,
            "Extract Positions Pipeline Layout",
            &extract_positions_bgl,
        );
        let extract_positions_pipeline = timed(&timings, "extract_positions", || {
            compute_pipeline(
                device,
                "Extract Positions Pipeline",
                &extract_positions_pl,
                &extract_positions_shader,
                "main",
            )
        });

        let eager_ms = t_all.elapsed().as_secs_f32() * 1000.0;
        Self {
            epoch: context.epoch,
            workgroup_size,
            context: Arc::clone(context),
            predict_bgl,
            constraint_bgl,
            sewing_bgl,
            pin_bgl,
            collision_bgl,
            edge_collision_bgl,
            update_vel_bgl,
            distance_atomic_bgl,
            extract_positions_bgl,
            self_collision_bgl,
            self_collision_ee_bgl,
            self_collision_apply_bgl,
            compute_normals_bgl,
            pair_collect_bgl,
            pair_solve_vt_bgl,
            pair_solve_ee_bgl,
            hash_bgl,
            edge_hash_bgl,
            sew_shrink_bgl,
            wrinkle_field_bgl,
            virtual_forward_bgl,
            self_collision_vv_bgl,
            predict_pipeline,
            distance_pipeline,
            bending_pipeline,
            sewing_pipeline,
            pin_pipeline,
            collision_pipeline,
            update_vel_pipeline,
            extract_positions_pipeline,
            atomic: OnceLock::new(),
            pair: OnceLock::new(),
            edge: OnceLock::new(),
            self_collision: OnceLock::new(),
            hash: OnceLock::new(),
            edge_hash: OnceLock::new(),
            sew_shrink: OnceLock::new(),
            wrinkle_field: OnceLock::new(),
            virtual_forward: OnceLock::new(),
            self_collision_vv: OnceLock::new(),
            timings,
            eager_ms,
        }
    }

    /// Atomic Jacobi ソルバー用パイプライン対を遅延生成・取得する。
    /// 初回呼び出し時に約 0.5 秒のコンパイルヒッチが発生する。
    pub fn ensure_atomic(&self) -> &AtomicPipelines {
        self.atomic.get_or_init(|| {
            let device = &self.context.device;
            let wg = self.workgroup_size;
            let shader = create_shader_with_wg_size(
                device,
                "Distance Atomic Shader",
                include_str!("../shaders/distance_atomic.wgsl"),
                wg,
            );
            let pl = pipeline_layout(device, "Distance Atomic Pipeline Layout", &self.distance_atomic_bgl);
            let solve = timed(&self.timings, "distance_atomic_solve", || {
                compute_pipeline(device, "Distance Atomic Solve Pipeline", &pl, &shader, "solve_distance_atomic")
            });
            let apply = timed(&self.timings, "distance_atomic_apply", || {
                compute_pipeline(device, "Distance Atomic Apply Pipeline", &pl, &shader, "apply_atomic_accum")
            });
            AtomicPipelines { solve, apply }
        })
    }

    /// Active Pair Caching 用パイプライン群を遅延生成・取得する。
    pub fn ensure_pair(&self) -> &PairPipelines {
        self.pair.get_or_init(|| {
            let device = &self.context.device;
            let wg = self.workgroup_size;
            let collect_shader = create_shader_with_wg_size(
                device,
                "Pair Collect Shader",
                include_str!("../shaders/self_collision_collect_pairs.wgsl"),
                wg,
            );
            let solve_vt_shader = create_shader_with_wg_size(
                device,
                "Pair Solve VT Shader",
                include_str!("../shaders/self_collision_solve_vt.wgsl"),
                wg,
            );
            let solve_ee_shader = create_shader_with_wg_size(
                device,
                "Pair Solve EE Shader",
                include_str!("../shaders/self_collision_solve_ee.wgsl"),
                wg,
            );
            let collect_pl =
                pipeline_layout(device, "Pair Collect Pipeline Layout", &self.pair_collect_bgl);
            let collect = timed(&self.timings, "pair_collect", || {
                compute_pipeline(device, "Pair Collect Pipeline", &collect_pl, &collect_shader, "main")
            });
            let solve_vt_pl =
                pipeline_layout(device, "Pair Solve VT Pipeline Layout", &self.pair_solve_vt_bgl);
            let solve_vt = timed(&self.timings, "pair_solve_vt", || {
                compute_pipeline(device, "Pair Solve VT Pipeline", &solve_vt_pl, &solve_vt_shader, "main")
            });
            let solve_ee_pl =
                pipeline_layout(device, "Pair Solve EE Pipeline Layout", &self.pair_solve_ee_bgl);
            let solve_ee = timed(&self.timings, "pair_solve_ee", || {
                compute_pipeline(device, "Pair Solve EE Pipeline", &solve_ee_pl, &solve_ee_shader, "main")
            });
            PairPipelines { collect, solve_vt, solve_ee }
        })
    }

    /// 縫合自然長の時間進行パスを遅延生成・取得する (縫合ありモデルのみ初回ヒッチ)。
    pub fn ensure_sew_shrink(&self) -> &wgpu::ComputePipeline {
        self.sew_shrink.get_or_init(|| {
            let device = &self.context.device;
            let shader = create_shader_with_wg_size(
                device,
                "Sew Shrink Shader",
                include_str!("../shaders/sewing_shrink.wgsl"),
                self.workgroup_size,
            );
            let pl = pipeline_layout(device, "Sew Shrink Pipeline Layout", &self.sew_shrink_bgl);
            timed(&self.timings, "sew_shrink", || {
                compute_pipeline(device, "Sew Shrink Pipeline", &pl, &shader, "main")
            })
        })
    }

    /// ドレープガイド拘束パイプラインを遅延生成・取得する。
    pub fn ensure_wrinkle_field(&self) -> &wgpu::ComputePipeline {
        self.wrinkle_field.get_or_init(|| {
            let device = &self.context.device;
            let wg = self.workgroup_size;
            let shader = create_shader_with_wg_size(
                device,
                "Wrinkle Field Shader",
                include_str!("../shaders/wrinkle_field.wgsl"),
                wg,
            );
            let pl = pipeline_layout(device, "Wrinkle Field Pipeline Layout", &self.wrinkle_field_bgl);
            timed(&self.timings, "wrinkle_field", || {
                compute_pipeline(device, "Wrinkle Field Pipeline", &pl, &shader, "main")
            })
        })
    }

    /// エッジコライダー詳細衝突パイプラインを遅延生成・取得する。
    pub fn ensure_edge(&self) -> &wgpu::ComputePipeline {
        self.edge.get_or_init(|| {
            let device = &self.context.device;
            let shader = create_shader_with_wg_size(
                device,
                "Edge Collision Shader",
                include_str!("../shaders/edge_collision.wgsl"),
                self.workgroup_size,
            );
            let pl = pipeline_layout(device, "Edge Collision Pipeline Layout", &self.edge_collision_bgl);
            timed(&self.timings, "edge_collision", || {
                compute_pipeline(device, "Edge Collision Pipeline", &pl, &shader, "main")
            })
        })
    }

    /// 自己衝突ダイレクト走査用パイプライン群を遅延生成・取得する。
    pub fn ensure_self_collision(&self) -> &SelfCollisionPipelines {
        self.self_collision.get_or_init(|| {
            let device = &self.context.device;
            let wg = self.workgroup_size;
            let solve_shader = create_shader_with_wg_size(
                device,
                "Self Collision Shader",
                include_str!("../shaders/self_collision.wgsl"),
                wg,
            );
            let apply_shader = create_shader_with_wg_size(
                device,
                "Self Collision Apply Shader",
                include_str!("../shaders/self_collision_apply.wgsl"),
                wg,
            );
            let normals_shader = create_shader_with_wg_size(
                device,
                "Compute Normals Shader",
                include_str!("../shaders/compute_normals.wgsl"),
                wg,
            );
            let solve_ee_shader = create_shader_with_wg_size(
                device,
                "Self Collision EE Shader",
                include_str!("../shaders/self_collision_ee_direct.wgsl"),
                wg,
            );
            let solve_pl =
                pipeline_layout(device, "Self Collision Pipeline Layout", &self.self_collision_bgl);
            let solve = timed(&self.timings, "self_collision", || {
                compute_pipeline(device, "Self Collision Pipeline", &solve_pl, &solve_shader, "main")
            });
            let solve_ee_pl = pipeline_layout(
                device,
                "Self Collision EE Pipeline Layout",
                &self.self_collision_ee_bgl,
            );
            let solve_ee = timed(&self.timings, "self_collision_ee", || {
                compute_pipeline(device, "Self Collision EE Pipeline", &solve_ee_pl, &solve_ee_shader, "main")
            });
            let apply_pl = pipeline_layout(
                device,
                "Self Collision Apply Pipeline Layout",
                &self.self_collision_apply_bgl,
            );
            let apply = timed(&self.timings, "self_collision_apply", || {
                compute_pipeline(device, "Self Collision Apply Pipeline", &apply_pl, &apply_shader, "main")
            });
            let normals_pl = pipeline_layout(
                device,
                "Compute Normals Pipeline Layout",
                &self.compute_normals_bgl,
            );
            let normals = timed(&self.timings, "compute_normals", || {
                compute_pipeline(device, "Compute Normals Pipeline", &normals_pl, &normals_shader, "main")
            });
            SelfCollisionPipelines { solve, solve_ee, apply, normals }
        })
    }

    /// 空間ハッシュ構築用パイプライン群を遅延生成・取得する
    /// (`workgroup_size` 非依存だが単一キャッシュのため同一キーで保持)。
    pub fn ensure_hash(&self) -> &HashPipelines {
        self.hash.get_or_init(|| {
            let device = &self.context.device;
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("SpatialGrid Sort Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/spatial_grid_sort.wgsl").into(),
                ),
            });
            let pl = pipeline_layout(device, "SpatialGrid Pipeline Layout", &self.hash_bgl);
            let entry = |name: &'static str, timing: &'static str| {
                timed(&self.timings, timing, || {
                    compute_pipeline(device, name, &pl, &shader, timing_entry(timing))
                })
            };
            HashPipelines {
                clear: entry("SpatialGrid Clear Pipeline", "hash_clear"),
                count: entry("SpatialGrid Count Pipeline", "hash_count"),
                scan_blocks: entry("SpatialGrid Scan Blocks Pipeline", "hash_scan_blocks"),
                scan_top: entry("SpatialGrid Scan Top Pipeline", "hash_scan_top"),
                add_offsets: entry("SpatialGrid Add Offsets Pipeline", "hash_add_offsets"),
                scatter: entry("SpatialGrid Scatter Pipeline", "hash_scatter"),
            }
        })
    }

    /// エッジ空間ハッシュ構築用パイプライン群を遅延生成・取得する。
    pub fn ensure_edge_hash(&self) -> &HashPipelines {
        self.edge_hash.get_or_init(|| {
            let device = &self.context.device;
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Edge SpatialGrid Sort Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/spatial_grid_sort_edges.wgsl").into(),
                ),
            });
            let pl = pipeline_layout(device, "Edge SpatialGrid Pipeline Layout", &self.edge_hash_bgl);
            let entry = |name: &'static str, timing: &'static str, ep: &'static str| {
                timed(&self.timings, timing, || {
                    compute_pipeline(device, name, &pl, &shader, ep)
                })
            };
            HashPipelines {
                clear: entry("Edge SpatialGrid Clear Pipeline", "edge_hash_clear", "clear_counts"),
                count: entry("Edge SpatialGrid Count Pipeline", "edge_hash_count", "count_edges"),
                scan_blocks: entry("Edge SpatialGrid Scan Blocks Pipeline", "edge_hash_scan_blocks", "scan_blocks"),
                scan_top: entry("Edge SpatialGrid Scan Top Pipeline", "edge_hash_scan_top", "scan_top"),
                add_offsets: entry("Edge SpatialGrid Add Offsets Pipeline", "edge_hash_add_offsets", "add_offsets"),
                scatter: entry("Edge SpatialGrid Scatter Pipeline", "edge_hash_scatter", "scatter_indices"),
            }
        })
    }

    /// 仮想頂点座標更新パイプラインを遅延生成・取得する。
    pub fn ensure_virtual_forward(&self) -> &wgpu::ComputePipeline {
        self.virtual_forward.get_or_init(|| {
            let device = &self.context.device;
            let shader = create_shader_with_wg_size(
                device,
                "Virtual Vertices Forward Shader",
                include_str!("../shaders/virtual_vertices_forward.wgsl"),
                self.workgroup_size,
            );
            let pl = pipeline_layout(device, "Virtual Forward Pipeline Layout", &self.virtual_forward_bgl);
            timed(&self.timings, "virtual_forward", || {
                compute_pipeline(device, "Virtual Forward Pipeline", &pl, &shader, "main")
            })
        })
    }

    /// 純粋球対球（V-V）自己衝突パイプラインを遅延生成・取得する。
    pub fn ensure_self_collision_vv(&self) -> &wgpu::ComputePipeline {
        self.self_collision_vv.get_or_init(|| {
            let device = &self.context.device;
            let shader = create_shader_with_wg_size(
                device,
                "Self Collision VV Shader",
                include_str!("../shaders/self_collision_vv.wgsl"),
                self.workgroup_size,
            );
            let pl = pipeline_layout(device, "Self Collision VV Pipeline Layout", &self.self_collision_vv_bgl);
            timed(&self.timings, "self_collision_vv", || {
                compute_pipeline(device, "Self Collision VV Pipeline", &pl, &shader, "main")
            })
        })
    }

    /// 遅延パイプラインのビルド済み名一覧 (診断用)
    pub fn lazy_built_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        if self.atomic.get().is_some() {
            names.push("atomic".to_string());
        }
        if self.pair.get().is_some() {
            names.push("pair".to_string());
        }
        if self.edge.get().is_some() {
            names.push("edge".to_string());
        }
        if self.self_collision.get().is_some() {
            names.push("self_collision".to_string());
        }
        if self.hash.get().is_some() {
            names.push("hash".to_string());
        }
        if self.edge_hash.get().is_some() {
            names.push("edge_hash".to_string());
        }
        if self.sew_shrink.get().is_some() {
            names.push("sew_shrink".to_string());
        }
        if self.wrinkle_field.get().is_some() {
            names.push("wrinkle_field".to_string());
        }
        if self.virtual_forward.get().is_some() {
            names.push("virtual_forward".to_string());
        }
        if self.self_collision_vv.get().is_some() {
            names.push("self_collision_vv".to_string());
        }
        names
    }

    /// ビルド計測ログのスナップショット (診断用)
    pub fn timings_snapshot(&self) -> Vec<(String, f32)> {
        lock_timings(&self.timings).clone()
    }

    pub fn eager_ms(&self) -> f32 {
        self.eager_ms
    }
}

/// 遅延ハッシュパイプラインの計測名から WGSL エントリーポイント名への対応
fn timing_entry(timing: &'static str) -> &'static str {
    match timing {
        "hash_clear" => "clear_counts",
        "hash_count" => "count_vertices",
        "hash_scan_blocks" => "scan_blocks",
        "hash_scan_top" => "scan_top",
        "hash_add_offsets" => "add_offsets",
        "hash_scatter" => "scatter_indices",
        _ => "main",
    }
}

/// キャッシュキー: (デバイス世代, workgroup_size, コンテキスト識別)。
/// `SharedPipelines` が `Arc<GpuContext>` を保持するため、ポインタの
/// 再利用は起きない。
type CacheKey = (u64, u32, usize);

static PIPELINE_CACHE: LazyLock<Mutex<HashMap<CacheKey, Arc<SharedPipelines>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn lock_cache() -> MutexGuard<'static, HashMap<CacheKey, Arc<SharedPipelines>>> {
    PIPELINE_CACHE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// 共有パイプライン群を取得または生成する。
/// 戻り値の bool は今回新規生成したかどうか (計測用)。
pub fn get_or_create_shared(
    context: &Arc<GpuContext>,
    workgroup_size: u32,
) -> (Arc<SharedPipelines>, bool) {
    let wg = normalize_workgroup_size(workgroup_size);
    let key = (context.epoch, wg, Arc::as_ptr(context) as usize);
    {
        let map = lock_cache();
        if let Some(shared) = map.get(&key) {
            return (Arc::clone(shared), false);
        }
    }
    // 生成中はロックを保持して直列化する (Blender/Python は単一スレッド想定)。
    let mut map = lock_cache();
    if let Some(shared) = map.get(&key) {
        return (Arc::clone(shared), false);
    }
    let shared = Arc::new(SharedPipelines::build(context, wg));
    // 旧世代エントリは無効 (旧デバイス紐付け) のため除去する
    map.retain(|(epoch, _, _), _| *epoch == context.epoch);
    map.insert(key, Arc::clone(&shared));
    (shared, true)
}

/// 共有キャッシュの診断スナップショット
#[derive(Debug, Clone)]
pub struct SharedCacheEntry {
    pub epoch: u64,
    pub workgroup_size: u32,
    pub eager_ms: f32,
    pub lazy_built: Vec<String>,
    pub lazy_ms: f32,
}

pub fn cache_info() -> Vec<SharedCacheEntry> {
    let map = lock_cache();
    map.iter()
        .map(|((epoch, wg, _), shared)| {
            let timings = shared.timings_snapshot();
            // eager 本数は将来変わり得るため、合計−eager で遅延分を算出する
            let total_ms: f32 = timings.iter().map(|(_, ms)| *ms).sum();
            let lazy_ms = (total_ms - shared.eager_ms()).max(0.0);
            SharedCacheEntry {
                epoch: *epoch,
                workgroup_size: *wg,
                eager_ms: shared.eager_ms(),
                lazy_built: shared.lazy_built_names(),
                lazy_ms,
            }
        })
        .collect()
}
