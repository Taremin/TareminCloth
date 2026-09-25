//! 診断専用のGPU時刻問い合わせ計装。
//!
//! 自己衝突のパス別内訳 (法線・ハッシュ6相・パッチ構築・解決・適用) を
//! 特定するための一時的な計測手段である。既定で無効であり、無効時は
//! `timestamp_writes: None` と同一経路のため挙動・性能は変わらない。
//! 有効化にはアダプタの TIMESTAMP_QUERY 対応が必要であり、
//! 非対応環境では有効化要求が偽を返して何もしない。

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::context::GpuContext;

/// 1フレームあたりの最大クエリ数 (2スタンプ/パス)。
const MAX_QUERIES: u32 = 2048;

fn lock_labels(labels: &Mutex<Vec<String>>) -> MutexGuard<'_, Vec<String>> {
    labels.lock().unwrap_or_else(|e| e.into_inner())
}

pub struct GpuProfiler {
    context: Arc<GpuContext>,
    supported: bool,
    enabled: AtomicBool,
    query_set: Option<wgpu::QuerySet>,
    resolve_buffer: Option<wgpu::Buffer>,
    staging_buffer: Option<wgpu::Buffer>,
    period: f32,
    next: AtomicU32,
    used: AtomicU32,
    labels: Mutex<Vec<String>>,
}

impl GpuProfiler {
    pub fn new(context: &Arc<GpuContext>) -> Self {
        let supported = context.timestamp_query_supported();
        let (query_set, resolve_buffer, staging_buffer, period) = if supported {
            let device = &context.device;
            let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Profiler Query Set"),
                ty: wgpu::QueryType::Timestamp,
                count: MAX_QUERIES,
            });
            let size = (MAX_QUERIES as u64) * 8;
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Profiler Resolve Buffer"),
                size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Profiler Staging Buffer"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let period = context.queue.get_timestamp_period();
            (Some(qs), Some(resolve), Some(staging), period)
        } else {
            (None, None, None, 0.0)
        };
        Self {
            context: Arc::clone(context),
            supported,
            enabled: AtomicBool::new(false),
            query_set,
            resolve_buffer,
            staging_buffer,
            period,
            next: AtomicU32::new(0),
            used: AtomicU32::new(0),
            labels: Mutex::new(Vec::new()),
        }
    }

    pub fn supported(&self) -> bool {
        self.supported
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed) && self.supported
    }

    /// 計測の有効/無効を切り替える。非対応環境では偽を返して無効のままにする。
    pub fn set_enabled(&self, enable: bool) -> bool {
        let on = enable && self.supported;
        self.enabled.store(on, Ordering::Relaxed);
        self.next.store(0, Ordering::Relaxed);
        self.used.store(0, Ordering::Relaxed);
        lock_labels(&self.labels).clear();
        on
    }

    /// フレーム記録開始 (エンコード直前に呼ぶ)。
    pub(crate) fn frame_begin(&self) {
        if !self.is_enabled() {
            return;
        }
        self.next.store(0, Ordering::Relaxed);
        self.used.store(0, Ordering::Relaxed);
        lock_labels(&self.labels).clear();
    }

    /// パス記録開始。ラベルを登録し、開始スタンプ番号を返す。
    /// 無効時や枯渇時は u32::MAX を返し、呼び出し側は素通りする。
    pub(crate) fn enter(&self, label: &'static str) -> u32 {
        if !self.is_enabled() {
            return u32::MAX;
        }
        let begin = self.next.load(Ordering::Relaxed);
        if begin + 2 > MAX_QUERIES {
            return u32::MAX;
        }
        lock_labels(&self.labels).push(label.to_string());
        self.next.store(begin + 2, Ordering::Relaxed);
        self.used.store(begin + 2, Ordering::Relaxed);
        begin
    }

    /// ComputePassDescriptor に渡す時刻書き込み。無効時は None を返す。
    pub(crate) fn writes(&self, begin: u32) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        if begin == u32::MAX {
            return None;
        }
        let qs = self.query_set.as_ref()?;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: qs,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(begin + 1),
        })
    }

    /// フレーム記録終了 (提出直前に呼ぶ)。使用分を解決バッファへ回収する。
    pub(crate) fn frame_end(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.is_enabled() {
            return;
        }
        let used = self.used.load(Ordering::Relaxed);
        if used == 0 {
            return;
        }
        if let (Some(qs), Some(resolve)) = (self.query_set.as_ref(), self.resolve_buffer.as_ref()) {
            encoder.resolve_query_set(qs, 0..used, resolve, 0);
        }
    }

    /// 直近フレームの (ラベル, ミリ秒) を回収する。診断専用の同期待機を伴う。
    pub fn take_ms(&self) -> Vec<(String, f32)> {
        if !self.is_enabled() {
            return Vec::new();
        }
        let used = self.used.load(Ordering::Relaxed);
        if used == 0 {
            return Vec::new();
        }
        let (resolve, staging) = match (self.resolve_buffer.as_ref(), self.staging_buffer.as_ref()) {
            (Some(r), Some(s)) => (r, s),
            _ => return Vec::new(),
        };
        let size = (used as u64) * 8;
        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Profiler Readback Encoder"),
            },
        );
        encoder.copy_buffer_to_buffer(resolve, 0, staging, 0, size);
        self.context.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..size);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });
        self.context.device.poll(wgpu::Maintain::Wait);
        if pollster::block_on(receiver.receive()).is_none() {
            return Vec::new();
        }
        let data = slice.get_mapped_range();
        let ticks: &[u64] = bytemuck::cast_slice(&data);
        let labels = lock_labels(&self.labels);
        let mut out = Vec::new();
        for (i, label) in labels.iter().enumerate() {
            let b = (i * 2) as usize;
            if b + 1 < ticks.len() {
                let dt = ticks[b + 1].wrapping_sub(ticks[b]) as f32;
                out.push((label.clone(), dt * self.period / 1_000_000.0));
            }
        }
        drop(data);
        staging.unmap();
        out
    }
}
