use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};

/// シミュレーション初期設定およびメッシュトポロジーのメタデータ
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationMetadata {
    pub object_name: String,
    pub num_vertices: u32,
    pub num_edges: u32,
    pub num_faces: u32,
    pub edges: Vec<[u32; 2]>,
    pub faces: Vec<[u32; 3]>,
    pub stiffness: f32,
    pub bending_stiffness: f32,
    pub thickness: f32,
    pub solver_mode: u32,
    pub workgroup_size: u32,
}

/// 単一フレーム内の統計情報（異常値検知・境界箱など）
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameStats {
    pub max_velocity: f32,
    pub max_displacement: f32,
    pub has_nan_or_inf: bool,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
}

/// 単一フレームの状態記録
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameRecord {
    pub frame_index: u32,
    pub dt: f32,
    pub substeps: u32,
    pub solver_iterations: u32,
    pub positions: Vec<[f32; 3]>,
    pub velocities: Vec<[f32; 3]>,
    pub pinned_indices: Vec<u32>,
    pub stats: FrameStats,
}

/// 全体のトレースデータ（ファイル保存用ルート構造体）
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationTrace {
    pub version: u32,
    pub created_at: String,
    pub metadata: SimulationMetadata,
    pub frames: Vec<FrameRecord>,
}

/// シミュレーションデバッグ記録マネージャー
pub struct SimulationDebugRecorder {
    pub enabled: bool,
    pub max_frames: usize,
    pub metadata: Option<SimulationMetadata>,
    pub frames: Vec<FrameRecord>,
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
        }
    }

    /// 記録を開始する
    pub fn start_recording(&mut self, metadata: SimulationMetadata, max_frames: Option<usize>) {
        self.clear();
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

    /// 1フレーム分の状態を記録する
    pub fn record_frame(
        &mut self,
        dt: f32,
        substeps: u32,
        solver_iterations: u32,
        positions: &[[f32; 3]],
        velocities: &[[f32; 3]],
        pinned_indices: &[u32],
    ) {
        if !self.enabled || self.metadata.is_none() {
            return;
        }

        // 最大フレーム数に達した場合は記録をスキップ（メモリ保護）
        if self.frames.len() >= self.max_frames {
            return;
        }

        let frame_index = self.frames.len() as u32;

        // 統計量の算出
        let mut max_vel_sq: f32 = 0.0;
        let mut max_disp_sq: f32 = 0.0;
        let mut has_nan_or_inf = false;
        let mut aabb_min = [f32::INFINITY; 3];
        let mut aabb_max = [f32::NEG_INFINITY; 3];

        let prev_frame = self.frames.last();

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

            if let Some(prev) = prev_frame {
                if let Some(prev_p) = prev.positions.get(i) {
                    let dx = p[0] - prev_p[0];
                    let dy = p[1] - prev_p[1];
                    let dz = p[2] - prev_p[2];
                    let dsq = dx * dx + dy * dy + dz * dz;
                    if dsq > max_disp_sq {
                        max_disp_sq = dsq;
                    }
                }
            }
        }

        if aabb_min[0] == f32::INFINITY {
            aabb_min = [0.0; 3];
            aabb_max = [0.0; 3];
        }

        let stats = FrameStats {
            max_velocity: max_vel_sq.sqrt(),
            max_displacement: max_disp_sq.sqrt(),
            has_nan_or_inf,
            aabb_min,
            aabb_max,
        };

        self.frames.push(FrameRecord {
            frame_index,
            dt,
            substeps,
            solver_iterations,
            positions: positions.to_vec(),
            velocities: velocities.to_vec(),
            pinned_indices: pinned_indices.to_vec(),
            stats,
        });
    }

    /// 記録されたトレースデータをgzip圧縮JSONファイルとして保存する
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

        let trace = SimulationTrace {
            version: 1,
            created_at: chrono_timestamp(),
            metadata,
            frames: self.frames.clone(),
        };

        let file = File::create(file_path).map_err(|e| {
            format!("保存先ファイルの作成に失敗しました ({}): {e}", file_path.display())
        })?;
        let buf_writer = BufWriter::new(file);
        let mut gz_encoder = GzEncoder::new(buf_writer, Compression::default());

        serde_json::to_writer(&mut gz_encoder, &trace).map_err(|e| {
            format!("JSONシリアライズに失敗しました: {e}")
        })?;

        gz_encoder.finish().map_err(|e| {
            format!("gzip圧縮ストリームの書き出しフラッシュに失敗しました: {e}")
        })?;

        Ok(file_path.to_string_lossy().to_string())
    }

    /// 記録を停止し、バッファをクリアする
    pub fn clear(&mut self) {
        self.enabled = false;
        self.metadata = None;
        self.frames.clear();
        self.frames.shrink_to_fit();
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
    use std::io::Read;

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
            stiffness: 500.0,
            bending_stiffness: 5.0,
            thickness: 0.01,
            solver_mode: 0,
            workgroup_size: 32,
        };

        recorder.start_recording(meta, Some(5));
        assert!(recorder.is_recording());

        // 2フレーム記録
        let pos1 = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let vel1 = vec![[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]];
        recorder.record_frame(0.016, 10, 2, &pos1, &vel1, &[0]);

        let pos2 = vec![[0.0, 0.0, 0.0], [1.0, 0.1, 0.0]];
        let vel2 = vec![[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]];
        recorder.record_frame(0.016, 10, 2, &pos2, &vel2, &[0]);

        assert_eq!(recorder.frame_count(), 2);
        assert_eq!(recorder.frames[1].stats.max_displacement, 0.1);
        assert!(!recorder.frames[1].stats.has_nan_or_inf);

        // 一時ファイルに保存して読み戻し検証
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_cloth_debug_trace.json.gz");

        let _ = recorder.save_to_file(&test_file).expect("save_to_file should succeed");
        assert!(test_file.exists());

        // 解凍してJSONパース
        let f = File::open(&test_file).unwrap();
        let mut gz = GzDecoder::new(f);
        let mut json_str = String::new();
        gz.read_to_string(&mut json_str).unwrap();

        let trace: SimulationTrace = serde_json::from_str(&json_str).unwrap();
        assert_eq!(trace.version, 1);
        assert_eq!(trace.metadata.object_name, "TestCloth");
        assert_eq!(trace.frames.len(), 2);
        assert_eq!(trace.frames[0].positions.len(), 2);

        // クリーンアップ
        let _ = std::fs::remove_file(&test_file);
        recorder.clear();
        assert!(!recorder.is_recording());
        assert_eq!(recorder.frame_count(), 0);
    }
}
