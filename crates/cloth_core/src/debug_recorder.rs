use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};

use crate::config::SimConfig;

/// 伸縮グループによる辺自然長スケール変更の記録 (edge_idx -> scale)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ElasticScaleRecord {
    pub edge_idx: u32,
    pub scale: f32,
}

/// BONE_SDFコライダーの静的記録 (メタデータに1回のみ)。
/// テクスチャは base64 の Rg16Float 密パックバイト列。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BoneSdfRecord {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub texture_base64: String,
    pub bone_infos: Vec<[f32; 20]>,
    /// 記録時に動的再ベイクが有効だった場合 true (テクスチャが途中で
    /// 変化し得るため完全再現できない可能性がある旨の目印)。
    #[serde(default)]
    pub dynamic_enabled: bool,
}

/// BONE_SDFの毎フレーム状態 (ボーン姿勢行列)。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BoneFrameState {
    pub world: Vec<[[f32; 4]; 4]>,
    pub inv_world: Vec<[[f32; 4]; 4]>,
}

/// 動的ピン留め（Grab操作など）の記録
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PinRecord {
    pub vertex_idx: u32,
    pub target_pos: [f32; 3],
    pub weight: f32,
}

/// コライダー（球・カプセル・平面・メッシュ）の記録
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ColliderRecord {
    #[serde(rename = "sphere")]
    Sphere {
        center: [f32; 3],
        radius: f32,
        friction: f32,
        restitution: f32,
    },
    #[serde(rename = "capsule")]
    Capsule {
        point_a: [f32; 3],
        point_b: [f32; 3],
        radius: f32,
        friction: f32,
        restitution: f32,
    },
    #[serde(rename = "plane")]
    Plane {
        point: [f32; 3],
        normal: [f32; 3],
        friction: f32,
        restitution: f32,
    },
    #[serde(rename = "mesh")]
    Mesh {
        triangles: Vec<[f32; 9]>, // [p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z]
        friction: f32,
        thickness: f32,
        restitution: f32,
        single_sided: bool,
    },
}

/// シミュレーション初期設定およびメッシュトポロジーのメタデータ
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SimulationMetadata {
    pub object_name: String,
    pub num_vertices: u32,
    pub num_edges: u32,
    pub num_faces: u32,
    pub edges: Vec<[u32; 2]>,
    pub faces: Vec<[u32; 3]>,
    pub inv_masses: Vec<f32>,
    pub initial_rest_lengths: Vec<f32>,
    /// 構築時レストポーズ座標 (全rest長の算出基準)。リプレイはここから
    /// 構築し、フレーム状態を上書きする。旧ログでは None→フレーム座標。
    #[serde(default)]
    pub initial_positions: Option<Vec<[f32; 3]>>,
    /// 元メッシュ辺のみ (せん断対角を含まない)。リプレイ時の再構築用。
    /// 旧ログでは None のため従来の edges にフォールバックする。
    #[serde(default)]
    pub original_edges: Option<Vec<[u32; 2]>>,
    /// 元メッシュ辺に対応する初期自然長 (original_edges と同順)。
    #[serde(default)]
    pub original_rest_lengths: Option<Vec<f32>>,
    /// 頂点毎の厚み・レイヤー (均一時は単一値の繰り返し)。旧ログでは None。
    #[serde(default)]
    pub thicknesses: Option<Vec<f32>>,
    #[serde(default)]
    pub layer_ids: Option<Vec<u32>>,
    // 縫合スプリング (v0, v1)
    #[serde(default)]
    pub sewing_springs: Option<Vec<[u32; 2]>>,
    // 物理パラメータ
    pub stiffness: f32,
    #[serde(default)]
    pub compression_stiffness: Option<f32>,
    #[serde(default)]
    pub shear_stiffness: Option<f32>,
    pub bending_stiffness: f32,
    pub thickness: f32,
    pub gravity: [f32; 3],
    pub damping: f32,
    pub solver_mode: u32,
    pub solver_iterations: u32,
    pub workgroup_size: u32,
    // セルフコリジョン & Untangling パラメータ
    pub enable_self_collision: bool,
    pub self_collision_relief_factor: f32,
    pub self_collision_max_displacement_ratio: f32,
    pub self_collision_exclude_neighbors: bool,
    pub enable_normal_untangling: bool,
    pub enable_edge_collision: bool,
    pub edge_margin_scale: f32,
    pub edge_margin_offset: f32,
    #[serde(default = "default_max_iterations")]
    pub self_collision_max_iterations: u32,
    #[serde(default)]
    pub enable_pair_cache: bool,
    #[serde(default = "default_pair_margin_mode")]
    pub pair_cache_margin_mode: u32,
    #[serde(default = "default_pair_margin")]
    pub pair_cache_safety_margin: f32,
    #[serde(default = "default_horizon_scale")]
    pub pair_cache_horizon_scale: f32,
    #[serde(default = "default_max_horizon")]
    pub pair_cache_max_horizon: f32,
    #[serde(default = "default_pair_max")]
    pub pair_cache_max_pairs: u32,
    #[serde(default = "default_true")]
    pub enable_pair_cache_final_fallback: bool,
    #[serde(default)]
    pub sewing_priority_enabled: bool,
    #[serde(default = "default_sewing_priority_threshold")]
    pub sewing_priority_threshold: f32,
    #[serde(default = "default_sewing_priority_merge_dist")]
    pub sewing_priority_merge_dist: f32,
    #[serde(default = "default_sewing_priority_ramp_frames")]
    pub sewing_priority_ramp_frames: u32,
    #[serde(default = "default_sewing_priority_max_frames")]
    pub sewing_priority_max_frames: u32,
    /// BONE_SDFコライダーの静的記録 (存在時のみ)。旧ログでは None。
    #[serde(default)]
    pub bone_sdf: Option<BoneSdfRecord>,
    /// 縫合収縮速度 (コンストラクタ再現用)。旧ログでは None→既定値。
    #[serde(default)]
    pub sewing_shrink_speed: Option<f32>,
    /// SimConfig一本化後の正本。旧個別フィールドは読み専用fallbackとして維持。
    #[serde(default)]
    pub config: Option<crate::config::SimConfig>,
}

fn default_pair_margin_mode() -> u32 {
    1
}

fn default_pair_margin() -> f32 {
    0.005
}

fn default_horizon_scale() -> f32 {
    1.3
}

fn default_max_horizon() -> f32 {
    0.02
}

fn default_pair_max() -> u32 {
    32768
}

fn default_true() -> bool {
    true
}

fn default_sewing_priority_threshold() -> f32 {
    0.9
}

fn default_sewing_priority_merge_dist() -> f32 {
    0.005
}

fn default_sewing_priority_ramp_frames() -> u32 {
    3
}

fn default_sewing_priority_max_frames() -> u32 {
    600
}

fn default_max_iterations() -> u32 {
    128
}

/// デバッグ記録の2階層化オプション (間引きフル保存 + トリガー時フラッシュ)
///
/// full_stride=1, ring_size=0, enable_triggers=false が従来動作 (毎フレームフル保存)。
/// full_stride>1 の場合、間欠フレームは positions/velocities を空にした
/// スタブ (stats + pins/colliders は保持) として保存し、ファイル肥大を抑える。
/// トリガー発火時は ring 内の直近フルをスタブと置換して文脈を復元する。
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DebugRecordOptions {
    #[serde(default = "default_full_stride")]
    pub full_stride: u32,
    #[serde(default)]
    pub ring_size: usize,
    #[serde(default)]
    pub lookahead: u32,
    #[serde(default)]
    pub enable_triggers: bool,
    #[serde(default = "default_disp_trigger")]
    pub disp_trigger_mm: f32,
    #[serde(default = "default_vel_trigger")]
    pub vel_trigger: f32,
    #[serde(default = "default_strain_trigger")]
    pub strain_trigger: f32,
    #[serde(default = "default_true")]
    pub saturate_trigger: bool,
    #[serde(default = "default_true")]
    pub nan_trigger: bool,
    #[serde(default = "default_true")]
    pub config_change_trigger: bool,
    #[serde(default = "default_pair_stride")]
    pub pair_stats_stride: u32,
}

fn default_full_stride() -> u32 { 1 }
fn default_disp_trigger() -> f32 { 5.0 }
fn default_vel_trigger() -> f32 { 10.0 }
fn default_strain_trigger() -> f32 { 0.5 }
fn default_pair_stride() -> u32 { 1 }

impl Default for DebugRecordOptions {
    fn default() -> Self {
        Self {
            full_stride: 1,
            ring_size: 0,
            lookahead: 0,
            enable_triggers: false,
            disp_trigger_mm: 5.0,
            vel_trigger: 10.0,
            strain_trigger: 0.5,
            saturate_trigger: true,
            nan_trigger: true,
            config_change_trigger: true,
            pair_stats_stride: 1,
        }
    }
}

/// 単一フレーム内の統計情報（異常値検知・境界箱など）
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FrameStats {
    pub max_velocity: f32,
    pub max_displacement: f32,
    pub has_nan_or_inf: bool,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    #[serde(default)]
    pub vt_count: u32,
    #[serde(default)]
    pub ee_count: u32,
    #[serde(default)]
    pub vt_saturated: bool,
    #[serde(default)]
    pub max_strain: f32,
    #[serde(default)]
    pub config_hash: u64,
}

/// 単一フレームの状態記録（入出力を含む）
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FrameRecord {
    pub frame_index: u32,
    pub dt: f32,
    pub substeps: u32,
    pub solver_iterations: u32,
    pub positions: Vec<[f32; 3]>,
    pub velocities: Vec<[f32; 3]>,
    pub pinned_indices: Vec<u32>, // 互換性用
    pub pins: Vec<PinRecord>,
    pub colliders: Vec<ColliderRecord>,
    pub stats: FrameStats,
    /// 設定変更フレームのみに保存される当時のフルSimConfig。
    /// 初期値は metadata.config が担うため初回フレームでは None。
    #[serde(default)]
    pub param_deltas: Option<SimConfig>,
    /// 伸縮グループ等による辺自然長スケール変更 (変更フレームのみ)。
    /// None は「変更なし」(前値を維持) を意味する。
    #[serde(default)]
    pub elastic_scales: Option<Vec<ElasticScaleRecord>>,
    /// BONE_SDFの毎フレーム姿勢。静止時は同一内容が繰り返される。
    #[serde(default)]
    pub bone_transforms: Option<BoneFrameState>,
    /// 縫合拘束の現在自然長 (GPU進行値、ソート済みバッファ順)。
    /// 縫合なし・変化なしの場合は None。
    #[serde(default)]
    pub sewing_rest_lengths: Option<Vec<f32>>,
    /// 縫合優先モードのラッチ内部状態 [latched01, frame, ramp_t, ratio]。
    /// 優先モード有効時のみ記録される。
    #[serde(default)]
    pub sewing_priority: Option<[f32; 4]>,
}

/// JSONL 形式の 1 行を表すレコード
#[derive(Serialize, Deserialize)]
#[serde(tag = "record_type")]
pub enum JsonlRecord {
    #[serde(rename = "metadata")]
    Metadata {
        version: u32,
        created_at: String,
        metadata: SimulationMetadata,
    },
    #[serde(rename = "frame")]
    Frame(FrameRecord),
}

/// シミュレーションデバッグ記録マネージャー
pub struct SimulationDebugRecorder {
    pub enabled: bool,
    pub max_frames: usize,
    pub metadata: Option<SimulationMetadata>,
    pub frames: Vec<FrameRecord>,
    pub options: DebugRecordOptions,
    ring: std::collections::VecDeque<FrameRecord>,
    seq: u32,
    lookahead_left: u32,
    prev_positions: Vec<[f32; 3]>,
    prev_hash: u64,
}

impl Default for SimulationDebugRecorder {
    fn default() -> Self {
        Self::new(3600)
    }
}

impl SimulationDebugRecorder {
    /// 新規レコーダーを作成する（デフォルト最大フレーム数: 3600 = 60FPSで1分）
    pub fn new(max_frames: usize) -> Self {
        Self {
            enabled: false,
            max_frames,
            metadata: None,
            frames: Vec::new(),
            options: DebugRecordOptions::default(),
            ring: std::collections::VecDeque::new(),
            seq: 0,
            lookahead_left: 0,
            prev_positions: Vec::new(),
            prev_hash: 0,
        }
    }

    /// 2階層記録オプションを設定する
    pub fn set_record_options(&mut self, options: DebugRecordOptions) {
        self.options.full_stride = options.full_stride.max(1);
        self.options.ring_size = options.ring_size;
        self.options.lookahead = options.lookahead;
        self.options.enable_triggers = options.enable_triggers;
        self.options.disp_trigger_mm = options.disp_trigger_mm;
        self.options.vel_trigger = options.vel_trigger;
        self.options.strain_trigger = options.strain_trigger;
        self.options.saturate_trigger = options.saturate_trigger;
        self.options.nan_trigger = options.nan_trigger;
        self.options.config_change_trigger = options.config_change_trigger;
        self.options.pair_stats_stride = options.pair_stats_stride.max(1);
    }

    /// 現在の記録オプションを取得する
    pub fn record_options(&self) -> &DebugRecordOptions {
        &self.options
    }

    /// 次に記録されるフレームの通番 (スパース時も単調増加するタイムライン index)
    pub fn next_frame_seq(&self) -> u32 {
        self.seq
    }

    /// フル座標を持つフレーム数
    pub fn full_frame_count(&self) -> usize {
        self.frames.iter().filter(|f| !f.positions.is_empty()).count()
    }

    /// フレームがフル座標を持つか
    pub fn frame_has_positions(frame: &FrameRecord) -> bool {
        !frame.positions.is_empty()
    }

    /// 記録を開始する
    pub fn start_recording(&mut self, metadata: SimulationMetadata, max_frames: Option<usize>) {
        // options は保持する (start前に set_record_options した設定を消さない)
        let opts = self.options.clone();
        self.clear();
        self.options = opts;
        if let Some(mf) = max_frames {
            self.max_frames = mf;
        }
        self.metadata = Some(metadata);
        self.enabled = true;
    }

    /// 記録中かどうか
    pub fn is_recording(&self) -> bool {
        self.enabled && self.metadata.is_some()
    }

    /// 現在記録されているフレーム数
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// 1フレーム分の状態を記録する (従来互換: 常にフル保存扱いではなく options に従う)
    pub fn record_frame(
        &mut self,
        dt: f32,
        substeps: u32,
        solver_iterations: u32,
        positions: &[[f32; 3]],
        velocities: &[[f32; 3]],
        pins: &[PinRecord],
        colliders: &[ColliderRecord],
    ) {
        self.record_frame_ext(
            dt, substeps, solver_iterations, positions, velocities, pins, colliders,
            None, 0.0, None, None, None, None, None, false,
        );
    }

    /// - config: Some の場合ハッシュ化して stats.config_hash に格納し、前回と
    ///   異なれば当該フレームの param_deltas にフル保存する (初回は metadata.config
    ///   が初期値を担うため保存しない)
    /// - elastic: Some(空でない場合のみ保存、変更検出は呼出側の責務)
    /// - bone: BONE_SDF有効時は毎フレームの姿勢を保存 (CPUミラー、GPU転送なし)
    /// - sewing_rests: 縫合ありの場合は現在自然長の全配列 (進行値そのまま)
    /// - スタブ化されても pins/colliders/dt/substeps/solver_iterations/stats は保持し、
    ///   positions/velocities のみ空にする (リプレイの入力連続性を保つため)
    #[allow(clippy::too_many_arguments)]
    pub fn record_frame_ext(
        &mut self,
        dt: f32,
        substeps: u32,
        solver_iterations: u32,
        positions: &[[f32; 3]],
        velocities: &[[f32; 3]],
        pins: &[PinRecord],
        colliders: &[ColliderRecord],
        pair: Option<(u32, u32, bool)>,
        max_strain: f32,
        config: Option<SimConfig>,
        elastic: Option<Vec<ElasticScaleRecord>>,
        bone: Option<BoneFrameState>,
        sewing_rests: Option<Vec<f32>>,
        sewing_priority: Option<[f32; 4]>,
        force_full: bool,
    ) {
        if !self.enabled || self.metadata.is_none() {
            return;
        }

        // 最大フレーム数に達した場合は記録をスキップ（メモリ保護）
        if self.frames.len() >= self.max_frames {
            return;
        }

        let frame_index = self.seq;

        // 統計量の算出 (変位はスタブ化されても正確なよう前回フル座標基準)
        let mut max_vel_sq: f32 = 0.0;
        let mut max_disp_sq: f32 = 0.0;
        let mut has_nan_or_inf = false;
        let mut aabb_min = [f32::INFINITY; 3];
        let mut aabb_max = [f32::NEG_INFINITY; 3];

        for (i, p) in positions.iter().enumerate() {
            for c in 0..3 {
                let v = p[c];
                if !v.is_finite() {
                    has_nan_or_inf = true;
                }
                if v < aabb_min[c] {
                    aabb_min[c] = v;
                }
                if v > aabb_max[c] {
                    aabb_max[c] = v;
                }
            }

            if let Some(vel) = velocities.get(i) {
                let vsq = vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2];
                if !vel[0].is_finite() || !vel[1].is_finite() || !vel[2].is_finite() {
                    has_nan_or_inf = true;
                }
                if vsq > max_vel_sq {
                    max_vel_sq = vsq;
                }
            }

            if let Some(prev_p) = self.prev_positions.get(i) {
                let dx = p[0] - prev_p[0];
                let dy = p[1] - prev_p[1];
                let dz = p[2] - prev_p[2];
                let dsq = dx * dx + dy * dy + dz * dz;
                if dsq > max_disp_sq {
                    max_disp_sq = dsq;
                }
            }
        }

        if aabb_min[0] == f32::INFINITY {
            aabb_min = [0.0; 3];
            aabb_max = [0.0; 3];
        }

        let (vt_count, ee_count, vt_saturated) = pair.unwrap_or((0, 0, false));
        let config_hash = config.as_ref().map(|c| c.hash_u64()).unwrap_or(0);
        // 設定変更の有無 (初回は metadata.config が初期値を担うため差分なし扱い)
        let config_changed = config_hash != 0 && self.prev_hash != 0 && config_hash != self.prev_hash;
        let stats = FrameStats {
            max_velocity: max_vel_sq.sqrt(),
            max_displacement: max_disp_sq.sqrt(),
            has_nan_or_inf,
            aabb_min,
            aabb_max,
            vt_count,
            ee_count,
            vt_saturated,
            max_strain,
            config_hash,
        };

        // トリガー評価
        let mut trigger = force_full;
        if self.options.enable_triggers && !trigger {
            let disp_mm = stats.max_displacement * 1000.0;
            if self.options.nan_trigger && has_nan_or_inf {
                trigger = true;
            } else if disp_mm > self.options.disp_trigger_mm {
                trigger = true;
            } else if stats.max_velocity > self.options.vel_trigger {
                trigger = true;
            } else if max_strain > self.options.strain_trigger {
                trigger = true;
            } else if self.options.saturate_trigger && vt_saturated {
                trigger = true;
            } else if self.options.config_change_trigger && config_changed {
                trigger = true;
            }
        }

        let pinned_indices: Vec<u32> = pins.iter().map(|p| p.vertex_idx).collect();
        let elastic_stored = match elastic {
            Some(v) if !v.is_empty() => Some(v),
            _ => None,
        };
        let full = FrameRecord {
            frame_index,
            dt,
            substeps,
            solver_iterations,
            positions: positions.to_vec(),
            velocities: velocities.to_vec(),
            pinned_indices,
            pins: pins.to_vec(),
            colliders: colliders.to_vec(),
            stats,
            param_deltas: if config_changed { config.clone() } else { None },
            elastic_scales: elastic_stored.clone(),
            bone_transforms: bone.clone(),
            sewing_rest_lengths: sewing_rests.clone(),
            sewing_priority,
        };

        let stride = self.options.full_stride.max(1);
        let use_lookahead = self.lookahead_left > 0;
        if use_lookahead {
            self.lookahead_left -= 1;
        }
        let keep_full = (frame_index % stride == 0) || trigger || use_lookahead || force_full;
        if trigger && self.options.lookahead > 0 {
            self.lookahead_left = self.options.lookahead;
        }

        if keep_full {
            if trigger {
                self.flush_ring_into_frames();
            }
            self.frames.push(full);
        } else {
            let stub = FrameRecord {
                frame_index: full.frame_index,
                dt: full.dt,
                substeps: full.substeps,
                solver_iterations: full.solver_iterations,
                positions: Vec::new(),
                velocities: Vec::new(),
                pinned_indices: full.pinned_indices.clone(),
                pins: full.pins.clone(),
                colliders: full.colliders.clone(),
                stats: full.stats.clone(),
                param_deltas: full.param_deltas.clone(),
                elastic_scales: full.elastic_scales.clone(),
                bone_transforms: full.bone_transforms.clone(),
                sewing_rest_lengths: full.sewing_rest_lengths.clone(),
                sewing_priority: full.sewing_priority,
            };
            self.frames.push(stub);
            if self.options.ring_size > 0 {
                self.ring.push_back(full);
                while self.ring.len() > self.options.ring_size {
                    self.ring.pop_front();
                }
            }
        }

        self.prev_positions = positions.to_vec();
        if config_hash != 0 {
            self.prev_hash = config_hash;
        }
        self.seq += 1;
    }

    /// ring 内の直近フルで対応するスタブを置換する (トリガー時の文脈復元)
    fn flush_ring_into_frames(&mut self) {
        if self.ring.is_empty() {
            return;
        }
        for full in self.ring.drain(..) {
            if let Some(pos) = self.frames.iter().position(|f| f.frame_index == full.frame_index) {
                self.frames[pos] = full;
            } else {
                self.frames.push(full);
            }
        }
        self.frames.sort_by_key(|f| f.frame_index);
    }

    /// 記録されたトレースデータをgzip圧縮JSON Linesファイル (.jsonl.gz) として保存する
    pub fn save_to_file(&self, file_path: &Path) -> Result<String, String> {
        let metadata = match &self.metadata {
            Some(m) => m.clone(),
            None => return Err("デバッグ記録のメタデータが存在しません".to_string()),
        };

        // 親ディレクトリの自動作成
        if let Some(parent) = file_path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    format!("保存先ディレクトリの作成に失敗しました ({}): {e}", parent.display())
                })?;
            }
        }

        let file = File::create(file_path).map_err(|e| {
            format!("保存先ファイルの作成に失敗しました ({}): {e}", file_path.display())
        })?;
        let buf_writer = BufWriter::new(file);
        let mut gz_encoder = GzEncoder::new(buf_writer, Compression::default());

        // 1行目: メタデータレコード (スタブ含み=スパース形式は version 3)
        let version = if self.frames.iter().any(|f| f.positions.is_empty()) { 3 } else { 2 };
        let meta_record = JsonlRecord::Metadata {
            version,
            created_at: chrono_timestamp(),
            metadata,
        };
        serde_json::to_writer(&mut gz_encoder, &meta_record).map_err(|e| {
            format!("メタデータのJSONシリアライズに失敗しました: {e}")
        })?;
        gz_encoder.write_all(b"\n").map_err(|e| format!("改行の書き込みに失敗しました: {e}"))?;

        // 2行目以降: 各フレームレコード
        for frame in &self.frames {
            let frame_record = JsonlRecord::Frame(frame.clone());
            serde_json::to_writer(&mut gz_encoder, &frame_record).map_err(|e| {
                format!("フレーム {} のJSONシリアライズに失敗しました: {e}", frame.frame_index)
            })?;
            gz_encoder.write_all(b"\n").map_err(|e| format!("改行の書き込みに失敗しました: {e}"))?;
        }

        gz_encoder.finish().map_err(|e| {
            format!("gzip圧縮ストリームの書き出しフラッシュに失敗しました: {e}")
        })?;

        Ok(file_path.to_string_lossy().to_string())
    }

    /// 記録を停止し、バッファをクリアする (options は保持する)
    pub fn clear(&mut self) {
        self.enabled = false;
        self.metadata = None;
        self.frames.clear();
        self.frames.shrink_to_fit();
        self.ring.clear();
        self.seq = 0;
        self.lookahead_left = 0;
        self.prev_positions.clear();
        self.prev_hash = 0;
    }
}

/// 外部依存クレートなしでISO8601形式のUTCタイムスタンプ文字列を生成する
fn chrono_timestamp() -> String {
    use std::time::SystemTime;
    match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
        Ok(dur) => {
            let secs = dur.as_secs();
            let millis = dur.subsec_millis();
            format!("{secs}.{millis:03}Z")
        }
        Err(_) => "1970-01-01T00:00:00Z".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::GzDecoder;
    use std::io::BufRead;
    use std::io::BufReader;

    #[test]
    fn test_debug_recorder_lifecycle() {
        let mut recorder = SimulationDebugRecorder::new(10);
        assert!(!recorder.is_recording());
        assert_eq!(recorder.frame_count(), 0);

        let meta = SimulationMetadata {
            object_name: "TestCloth".to_string(),
            num_vertices: 2,
            num_edges: 1,
            num_faces: 0,
            edges: vec![[0, 1]],
            faces: vec![],
            inv_masses: vec![1.0, 1.0],
            initial_rest_lengths: vec![1.0],
            initial_positions: None,
            original_edges: None,
            original_rest_lengths: None,
            thicknesses: None,
            layer_ids: None,
            bone_sdf: None,
            sewing_shrink_speed: None,
            sewing_springs: None,
            stiffness: 500.0,
            compression_stiffness: None,
            shear_stiffness: None,
            bending_stiffness: 5.0,
            thickness: 0.01,
            gravity: [0.0, 0.0, -9.81],
            damping: 0.01,
            solver_mode: 0,
            solver_iterations: 2,
            workgroup_size: 32,
            enable_self_collision: true,
            self_collision_relief_factor: 0.2,
            self_collision_max_displacement_ratio: 0.2,
            self_collision_exclude_neighbors: true,
            enable_normal_untangling: true,
            enable_edge_collision: false,
            edge_margin_scale: 1.0,
            edge_margin_offset: 0.0,
            self_collision_max_iterations: 128,
            enable_pair_cache: false,
            pair_cache_margin_mode: 1,
            pair_cache_safety_margin: 0.005,
            pair_cache_horizon_scale: 1.3,
            pair_cache_max_horizon: 0.02,
            pair_cache_max_pairs: 32768,
            enable_pair_cache_final_fallback: true,
            sewing_priority_enabled: false,
            sewing_priority_threshold: 0.9,
            sewing_priority_merge_dist: 0.005,
            sewing_priority_ramp_frames: 3,
            sewing_priority_max_frames: 600,
            config: None,
        };

        recorder.start_recording(meta, Some(5));
        assert!(recorder.is_recording());

        // 2フレーム記録
        let pos1 = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let vel1 = vec![[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]];
        let pins = vec![PinRecord {
            vertex_idx: 0,
            target_pos: [0.0, 0.0, 0.0],
            weight: 1.0,
        }];
        let cols = vec![ColliderRecord::Sphere {
            center: [0.0, -1.0, 0.0],
            radius: 0.5,
            friction: 0.2,
            restitution: 0.0,
        }];

        recorder.record_frame(0.016, 10, 2, &pos1, &vel1, &pins, &cols);

        let pos2 = vec![[0.0, 0.0, 0.0], [1.0, 0.1, 0.0]];
        let vel2 = vec![[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]];
        recorder.record_frame(0.016, 10, 2, &pos2, &vel2, &pins, &cols);

        assert_eq!(recorder.frame_count(), 2);
        assert_eq!(recorder.frames[1].stats.max_displacement, 0.1);
        assert!(!recorder.frames[1].stats.has_nan_or_inf);

        // 一時ファイルに保存して読み戻し検証 (.jsonl.gz)
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_cloth_debug_trace.jsonl.gz");

        let _ = recorder.save_to_file(&test_file).expect("save_to_file should succeed");
        assert!(test_file.exists());

        // 解凍して各行をJSONLパース
        let f = File::open(&test_file).unwrap();
        let gz = GzDecoder::new(f);
        let reader = BufReader::new(gz);
        let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();

        assert_eq!(lines.len(), 3); // 1行目: metadata, 2行目: frame 0, 3行目: frame 1

        // 1行目のパース
        let meta_rec: JsonlRecord = serde_json::from_str(&lines[0]).unwrap();
        match meta_rec {
            JsonlRecord::Metadata { version, metadata, .. } => {
                assert_eq!(version, 2);
                assert_eq!(metadata.object_name, "TestCloth");
                assert_eq!(metadata.gravity, [0.0, 0.0, -9.81]);
            }
            _ => panic!("Expected metadata record on line 1"),
        }

        // 2行目のパース
        let f0_rec: JsonlRecord = serde_json::from_str(&lines[1]).unwrap();
        match f0_rec {
            JsonlRecord::Frame(f) => {
                assert_eq!(f.frame_index, 0);
                assert_eq!(f.positions.len(), 2);
                assert_eq!(f.pins.len(), 1);
                assert_eq!(f.colliders.len(), 1);
            }
            _ => panic!("Expected frame record on line 2"),
        }

        // クリーンアップ
        let _ = std::fs::remove_file(&test_file);
        recorder.clear();
        assert!(!recorder.is_recording());
        assert_eq!(recorder.frame_count(), 0);
    }

    fn sparse_test_meta() -> SimulationMetadata {
        SimulationMetadata {
            object_name: "SparseTest".to_string(),
            num_vertices: 2,
            num_edges: 1,
            num_faces: 0,
            edges: vec![[0, 1]],
            faces: vec![],
            inv_masses: vec![1.0, 1.0],
            initial_rest_lengths: vec![1.0],
            initial_positions: None,
            original_edges: None,
            original_rest_lengths: None,
            thicknesses: None,
            layer_ids: None,
            bone_sdf: None,
            sewing_shrink_speed: None,
            sewing_springs: None,
            stiffness: 500.0,
            compression_stiffness: None,
            shear_stiffness: None,
            bending_stiffness: 5.0,
            thickness: 0.01,
            gravity: [0.0, 0.0, -9.81],
            damping: 0.01,
            solver_mode: 0,
            solver_iterations: 2,
            workgroup_size: 32,
            enable_self_collision: false,
            self_collision_relief_factor: 0.2,
            self_collision_max_displacement_ratio: 0.2,
            self_collision_exclude_neighbors: true,
            enable_normal_untangling: false,
            enable_edge_collision: false,
            edge_margin_scale: 1.0,
            edge_margin_offset: 0.0,
            self_collision_max_iterations: 128,
            enable_pair_cache: false,
            pair_cache_margin_mode: 1,
            pair_cache_safety_margin: 0.005,
            pair_cache_horizon_scale: 1.3,
            pair_cache_max_horizon: 0.02,
            pair_cache_max_pairs: 32768,
            enable_pair_cache_final_fallback: true,
            sewing_priority_enabled: false,
            sewing_priority_threshold: 0.9,
            sewing_priority_merge_dist: 0.005,
            sewing_priority_ramp_frames: 3,
            sewing_priority_max_frames: 600,
            config: None,
        }
    }

    #[test]
    fn test_sparse_stride_and_stubs() {
        let mut recorder = SimulationDebugRecorder::new(100);
        recorder.set_record_options(DebugRecordOptions {
            full_stride: 3,
            ring_size: 2,
            lookahead: 0,
            enable_triggers: false,
            ..Default::default()
        });
        recorder.start_recording(sparse_test_meta(), None);
        let pins: Vec<PinRecord> = vec![];
        let cols: Vec<ColliderRecord> = vec![];
        let vel = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        for i in 0..7u32 {
            let pos = vec![[0.0, 0.0, 0.0], [1.0 + i as f32 * 0.001, 0.0, 0.0]];
            recorder.record_frame(0.016, 10, 2, &pos, &vel, &pins, &cols);
        }
        // フルは 0,3,6 の3件、残りはスタブ
        assert_eq!(recorder.frame_count(), 7);
        assert_eq!(recorder.full_frame_count(), 3);
        for f in &recorder.frames {
            assert_eq!(f.dt, 0.016);
            if f.frame_index % 3 == 0 {
                assert!(!f.positions.is_empty(), "frame {} should be full", f.frame_index);
            } else {
                assert!(f.positions.is_empty(), "frame {} should be stub", f.frame_index);
                // スタブでも stats は有効 (変位は前回フル基準で算出)
                assert!(f.stats.max_displacement >= 0.0);
            }
        }
        // frame_index は通番を維持する
        let indices: Vec<u32> = recorder.frames.iter().map(|f| f.frame_index).collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_sparse_trigger_flush_restores_ring() {
        let mut recorder = SimulationDebugRecorder::new(100);
        recorder.set_record_options(DebugRecordOptions {
            full_stride: 100,
            ring_size: 2,
            lookahead: 1,
            enable_triggers: true,
            disp_trigger_mm: 5.0,
            ..Default::default()
        });
        recorder.start_recording(sparse_test_meta(), None);
        let pins: Vec<PinRecord> = vec![];
        let cols: Vec<ColliderRecord> = vec![];
        let vel = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        // frame 0 は stride でフル、1,2 は微動でスタブ化
        recorder.record_frame(0.016, 10, 2, &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], &vel, &pins, &cols);
        recorder.record_frame(0.016, 10, 2, &[[0.0, 0.0, 0.0], [1.0005, 0.0, 0.0]], &vel, &pins, &cols);
        recorder.record_frame(0.016, 10, 2, &[[0.0, 0.0, 0.0], [1.001, 0.0, 0.0]], &vel, &pins, &cols);
        assert!(recorder.frames[1].positions.is_empty());
        assert!(recorder.frames[2].positions.is_empty());
        // frame 3 で大変位トリガー → ring(1,2)が復元され、lookaheadで frame 4 もフル
        recorder.record_frame(0.016, 10, 2, &[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], &vel, &pins, &cols);
        recorder.record_frame(0.016, 10, 2, &[[0.0, 0.0, 0.0], [1.5005, 0.0, 0.0]], &vel, &pins, &cols);
        assert!(!recorder.frames[1].positions.is_empty(), "trigger should restore frame 1");
        assert!(!recorder.frames[2].positions.is_empty(), "trigger should restore frame 2");
        assert!(!recorder.frames[3].positions.is_empty());
        assert!(!recorder.frames[4].positions.is_empty(), "lookahead should keep frame 4 full");
    }

    #[test]
    fn test_sparse_file_version_bump() {
        let mut recorder = SimulationDebugRecorder::new(100);
        recorder.set_record_options(DebugRecordOptions {
            full_stride: 2,
            ..Default::default()
        });
        recorder.start_recording(sparse_test_meta(), None);
        let pins: Vec<PinRecord> = vec![];
        let cols: Vec<ColliderRecord> = vec![];
        let vel = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        for i in 0..3u32 {
            let pos = vec![[0.0, 0.0, 0.0], [1.0 + i as f32 * 0.0001, 0.0, 0.0]];
            recorder.record_frame(0.016, 10, 2, &pos, &vel, &pins, &cols);
        }
        let test_file = std::env::temp_dir().join("test_sparse_version.jsonl.gz");
        recorder.save_to_file(&test_file).unwrap();
        let f = File::open(&test_file).unwrap();
        let gz = GzDecoder::new(f);
        let reader = BufReader::new(gz);
        let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();
        let meta_rec: JsonlRecord = serde_json::from_str(&lines[0]).unwrap();
        match meta_rec {
            JsonlRecord::Metadata { version, .. } => assert_eq!(version, 3),
            _ => panic!("expected metadata"),
        }
        let _ = std::fs::remove_file(&test_file);
    }

    #[test]
    fn test_param_deltas_only_on_change() {
        use crate::config::SimConfig;
        let mut recorder = SimulationDebugRecorder::new(100);
        recorder.start_recording(sparse_test_meta(), None);
        let pins: Vec<PinRecord> = vec![];
        let cols: Vec<ColliderRecord> = vec![];
        let vel = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut cfg = SimConfig::default();
        // frame 0: 初回は差分なし (metadata.config が初期値)
        recorder.record_frame_ext(0.016, 10, 2, &pos, &vel, &pins, &cols, None, 0.0, Some(cfg.clone()), None, None, None, None, false);
        // frame 1: 同一設定は差分なし
        recorder.record_frame_ext(0.016, 10, 2, &pos, &vel, &pins, &cols, None, 0.0, Some(cfg.clone()), None, None, None, None, false);
        // frame 2: 変更時のみ差分あり
        cfg.solver_iterations = 9;
        recorder.record_frame_ext(0.016, 10, 2, &pos, &vel, &pins, &cols, None, 0.0, Some(cfg.clone()), None, None, None, None, false);
        // frame 3: 設定なし呼び出し (旧経路) は差分なし
        recorder.record_frame(0.016, 10, 2, &pos, &vel, &pins, &cols);

        assert!(recorder.frames[0].param_deltas.is_none());
        assert!(recorder.frames[1].param_deltas.is_none());
        assert_eq!(
            recorder.frames[2].param_deltas.as_ref().unwrap().solver_iterations, 9
        );
        assert!(recorder.frames[3].param_deltas.is_none());
        // ハッシュは設定あり全フレームで記録される
        assert_ne!(recorder.frames[0].stats.config_hash, 0);
        assert_eq!(recorder.frames[0].stats.config_hash, recorder.frames[1].stats.config_hash);
        assert_ne!(recorder.frames[2].stats.config_hash, recorder.frames[1].stats.config_hash);
    }

    #[test]
    fn test_elastic_scales_stored_when_nonempty() {
        let mut recorder = SimulationDebugRecorder::new(100);
        recorder.start_recording(sparse_test_meta(), None);
        let pins: Vec<PinRecord> = vec![];
        let cols: Vec<ColliderRecord> = vec![];
        let vel = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let elastic = Some(vec![ElasticScaleRecord { edge_idx: 3, scale: 0.8 }]);
        recorder.record_frame_ext(0.016, 10, 2, &pos, &vel, &pins, &cols, None, 0.0, None, elastic, None, None, None, false);
        recorder.record_frame(0.016, 10, 2, &pos, &vel, &pins, &cols);
        let stored = recorder.frames[0].elastic_scales.as_ref().unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].edge_idx, 3);
        assert!((stored[0].scale - 0.8).abs() < 1e-6);
        assert!(recorder.frames[1].elastic_scales.is_none());
    }
}
