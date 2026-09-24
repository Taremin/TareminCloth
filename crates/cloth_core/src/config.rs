use serde::{Deserialize, Serialize};

fn d_gravity() -> [f32; 3] { [0.0, 0.0, -9.81] }
fn d_damping() -> f32 { 0.01 }
fn d_iterations() -> u32 { 2 }
fn d_solver_mode() -> u32 { 0 }
fn d_workgroup() -> u32 { 32 }
fn d_relief() -> f32 { 0.2 }
fn d_max_disp() -> f32 { 0.2 }
fn d_max_iters() -> u32 { 128 }
fn d_true() -> bool { true }
fn d_horizon_scale() -> f32 { 1.3 }
fn d_max_horizon() -> f32 { 0.02 }
fn d_safety_margin() -> f32 { 0.005 }
fn d_pair_max() -> u32 { 65536 }
fn d_margin_mode() -> u32 { 1 }
fn d_sewing_stiffness() -> f32 { 10000.0 }
fn d_merge_dist() -> f32 { 0.005 }
fn d_threshold() -> f32 { 0.9 }
fn d_ramp() -> u32 { 3 }
fn d_max_frames() -> u32 { 600 }
fn d_version() -> u32 { 1 }
fn d_relax_iters() -> u32 { 1 }
fn d_substep_interval() -> u32 { 1 }
fn d_edge_scale() -> f32 { 1.0 }
fn d_stiffness() -> f32 { 500.0 }
fn d_bending() -> f32 { 5.0 }

/// シミュレーション可変パラメータの Single Source of Truth.
///
/// 新規パラメータ追加時の手順:
/// 1. ここに `#[serde(default = "...")]` 付きでフィールド追加
/// 2. `export_config` / `apply_config` に写像追加
/// 3. `tests/core/test_config_parity.py` が自動で欠落を検出する
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SimConfig {
    #[serde(default = "d_version")]
    pub version: u32,
    #[serde(default = "d_gravity")]
    pub gravity: [f32; 3],
    #[serde(default = "d_damping")]
    pub damping: f32,
    #[serde(default)]
    pub tension_damping: f32,
    #[serde(default)]
    pub compression_damping: f32,
    #[serde(default)]
    pub shear_damping: f32,
    #[serde(default)]
    pub bending_damping: f32,
    #[serde(default = "d_stiffness")]
    pub tension_stiffness: f32,
    #[serde(default = "d_stiffness")]
    pub compression_stiffness: f32,
    #[serde(default = "d_stiffness")]
    pub shear_stiffness: f32,
    #[serde(default = "d_bending")]
    pub bending_stiffness: f32,
    #[serde(default = "d_iterations")]
    pub solver_iterations: u32,
    #[serde(default = "d_solver_mode")]
    pub solver_mode: u32,
    #[serde(default = "d_workgroup")]
    pub workgroup_size: u32,
    #[serde(default = "d_true")]
    pub enable_self_collision: bool,
    #[serde(default = "d_relief")]
    pub relief_factor: f32,
    #[serde(default = "d_max_disp")]
    pub max_displacement_ratio: f32,
    #[serde(default = "d_true")]
    pub exclude_neighbors: bool,
    #[serde(default = "d_true")]
    pub enable_normal_untangling: bool,
    #[serde(default = "d_max_iters")]
    pub self_collision_max_iterations: u32,
    #[serde(default)]
    pub coupled_mode: u32,
    #[serde(default = "d_relax_iters")]
    pub post_relaxation_iters: u32,
    #[serde(default = "d_substep_interval")]
    pub substep_interval: u32,
    #[serde(default)]
    pub enable_pair_cache: bool,
    #[serde(default = "d_margin_mode")]
    pub pair_margin_mode: u32,
    #[serde(default = "d_safety_margin")]
    pub pair_safety_margin: f32,
    #[serde(default = "d_horizon_scale")]
    pub pair_horizon_scale: f32,
    #[serde(default = "d_max_horizon")]
    pub pair_max_horizon: f32,
    #[serde(default = "d_pair_max")]
    pub pair_max_pairs: u32,
    #[serde(default = "d_true")]
    pub enable_pair_final_fallback: bool,
    #[serde(default)]
    pub enable_edge_collision: bool,
    #[serde(default = "d_edge_scale")]
    pub edge_margin_scale: f32,
    #[serde(default)]
    pub edge_margin_offset: f32,
    #[serde(default = "d_sewing_stiffness")]
    pub sewing_stiffness: f32,
    #[serde(default = "d_true")]
    pub enable_sewing_lock: bool,
    #[serde(default)]
    pub sewing_priority_enabled: bool,
    #[serde(default = "d_threshold")]
    pub sewing_priority_threshold: f32,
    #[serde(default = "d_merge_dist")]
    pub sewing_priority_merge_dist: f32,
    #[serde(default = "d_ramp")]
    pub sewing_priority_ramp_frames: u32,
    #[serde(default = "d_max_frames")]
    pub sewing_priority_max_frames: u32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            version: 1,
            gravity: d_gravity(),
            damping: d_damping(),
            tension_damping: 0.0,
            compression_damping: 0.0,
            shear_damping: 0.0,
            bending_damping: 0.0,
            tension_stiffness: d_stiffness(),
            compression_stiffness: d_stiffness(),
            shear_stiffness: 250.0,
            bending_stiffness: d_bending(),
            solver_iterations: d_iterations(),
            solver_mode: d_solver_mode(),
            workgroup_size: d_workgroup(),
            enable_self_collision: true,
            relief_factor: d_relief(),
            max_displacement_ratio: d_max_disp(),
            exclude_neighbors: true,
            enable_normal_untangling: true,
            self_collision_max_iterations: d_max_iters(),
            coupled_mode: 0,
            post_relaxation_iters: 1,
            substep_interval: 1,
            enable_pair_cache: false,
            pair_margin_mode: d_margin_mode(),
            pair_safety_margin: d_safety_margin(),
            pair_horizon_scale: d_horizon_scale(),
            pair_max_horizon: d_max_horizon(),
            pair_max_pairs: d_pair_max(),
            enable_pair_final_fallback: true,
            enable_edge_collision: false,
            edge_margin_scale: d_edge_scale(),
            edge_margin_offset: 0.0,
            sewing_stiffness: d_sewing_stiffness(),
            enable_sewing_lock: true,
            sewing_priority_enabled: false,
            sewing_priority_threshold: d_threshold(),
            sewing_priority_merge_dist: d_merge_dist(),
            sewing_priority_ramp_frames: d_ramp(),
            sewing_priority_max_frames: d_max_frames(),
        }
    }
}

impl SimConfig {
    /// 設定内容の簡易ハッシュ (フレーム途中変更検出用、FNV-1a 64bit)
    pub fn hash_u64(&self) -> u64 {
        let s = serde_json::to_string(self).unwrap_or_default();
        let mut h: u64 = 0xcbf29ce484222325;
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_json_roundtrip() {
        let c = SimConfig::default();
        let s = serde_json::to_string(&c).unwrap();
        let d: SimConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn config_old_json_with_missing_fields() {
        let d: SimConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(d.version, 1);
    }
}
