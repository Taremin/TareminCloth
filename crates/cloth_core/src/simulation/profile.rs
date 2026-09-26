//! 診断専用のGPU時刻問い合わせ計装（`wgpu-profiler 0.20系` 委譲）。
//!
//! 自己衝突のパス別内訳 (法線・ハッシュ6相・パッチ構築・解決・適用) を
//! 特定するための計測手段である。既定で無効であり、無効時は
//! `timestamp_writes: None` と同一経路のため挙動・性能は変わらない。
//! 有効化にはアダプタの TIMESTAMP_QUERY 対応が必要であり、
//! 非対応環境では有効化要求が偽を返して何もしない。
//!
//! 親スコープ（`begin_scope`）と子パス（`begin_pass_with_parent`）による
//! ネスト集計に対応する。INSIDE系非対応環境では親が時刻なしになるが、
//! 子の時刻と木構造は維持される。
//!
//! `take_ms` は最大 `max_num_pending_frames` 件遅延の非同期回収である
//! （直前フレームの同期値ではない）。

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::context::GpuContext;

fn lock_inner(
    inner: &Mutex<Option<wgpu_profiler::GpuProfiler>>,
) -> MutexGuard<'_, Option<wgpu_profiler::GpuProfiler>> {
    inner.lock().unwrap_or_else(|e| e.into_inner())
}

fn lock_flat(flat: &Mutex<Vec<(String, f32)>>) -> MutexGuard<'_, Vec<(String, f32)>> {
    flat.lock().unwrap_or_else(|e| e.into_inner())
}

fn lock_tree(
    tree: &Mutex<Vec<wgpu_profiler::GpuTimerQueryResult>>,
) -> MutexGuard<'_, Vec<wgpu_profiler::GpuTimerQueryResult>> {
    tree.lock().unwrap_or_else(|e| e.into_inner())
}

pub struct GpuProfiler {
    context: Arc<GpuContext>,
    supported: bool,
    enabled: AtomicBool,
    inner: Mutex<Option<wgpu_profiler::GpuProfiler>>,
    last_flat: Mutex<Vec<(String, f32)>>,
    last_tree: Mutex<Vec<wgpu_profiler::GpuTimerQueryResult>>,
}

impl GpuProfiler {
    pub fn new(context: &Arc<GpuContext>) -> Self {
        let supported = context.timestamp_query_supported();
        let inner = if supported {
            let settings = wgpu_profiler::GpuProfilerSettings {
                enable_timer_queries: true,
                enable_debug_groups: false,
                max_num_pending_frames: 3,
            };
            wgpu_profiler::GpuProfiler::new(settings).ok()
        } else {
            None
        };
        let supported = supported && inner.is_some();
        Self {
            context: Arc::clone(context),
            supported,
            enabled: AtomicBool::new(false),
            inner: Mutex::new(inner),
            last_flat: Mutex::new(Vec::new()),
            last_tree: Mutex::new(Vec::new()),
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
        lock_flat(&self.last_flat).clear();
        lock_tree(&self.last_tree).clear();
        on
    }

    /// フレーム記録開始 (エンコード直前に呼ぶ)。互換用の no-op。
    pub(crate) fn frame_begin(&self) {
        // wgpu-profiler はスコープ単位で管理するため開始処理は不要。
    }

    /// 旧 `enter` 互換のパス計測開始。無効時は `None` を返す。
    /// 戻り値は所有権付きクエリであり、パス終了後に `end_pass` へ渡すこと。
    pub(crate) fn begin_pass(
        &self,
        label: &'static str,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Option<wgpu_profiler::GpuProfilerQuery> {
        self.begin_pass_with_parent(label, encoder, None)
    }

    /// 親付きのパス計測開始。ネスト集計用。親が `None` の場合は頂層になる。
    pub(crate) fn begin_pass_with_parent(
        &self,
        label: impl Into<String>,
        encoder: &mut wgpu::CommandEncoder,
        parent: Option<&wgpu_profiler::GpuProfilerQuery>,
    ) -> Option<wgpu_profiler::GpuProfilerQuery> {
        if !self.is_enabled() {
            return None;
        }
        let guard = lock_inner(&self.inner);
        let inner = guard.as_ref()?;
        Some(
            inner
                .begin_pass_query(label, encoder, &self.context.device)
                .with_parent(parent),
        )
    }

    /// エンコーダ区間の親スコープ開始。子パスを束ねる grouping 用。
    /// `TIMESTAMP_QUERY_INSIDE_ENCODERS` 非対応環境では時刻なし（n/a）になるが、
    /// 子の集計構造は維持される。
    pub(crate) fn begin_scope(
        &self,
        label: impl Into<String>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Option<wgpu_profiler::GpuProfilerQuery> {
        if !self.is_enabled() {
            return None;
        }
        let guard = lock_inner(&self.inner);
        let inner = guard.as_ref()?;
        Some(inner.begin_query(label, encoder, &self.context.device))
    }

    /// 親スコープ終了。`begin_scope` と対で呼ぶこと。
    pub(crate) fn end_scope(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: Option<wgpu_profiler::GpuProfilerQuery>,
    ) {
        let Some(q) = query else { return };
        if !self.is_enabled() {
            return;
        }
        let guard = lock_inner(&self.inner);
        if let Some(inner) = guard.as_ref() {
            inner.end_query(encoder, q);
        }
    }

    /// `begin_pass` で得たクエリから `ComputePassDescriptor` 用の書き込みを得る。
    pub(crate) fn pass_writes<'a>(
        &self,
        query: &'a Option<wgpu_profiler::GpuProfilerQuery>,
    ) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
        query
            .as_ref()
            .and_then(|q| q.compute_pass_timestamp_writes())
    }

    /// パス計測終了。`begin_pass` と対で呼ぶこと。
    pub(crate) fn end_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: Option<wgpu_profiler::GpuProfilerQuery>,
    ) {
        let Some(q) = query else { return };
        if !self.is_enabled() {
            return;
        }
        let guard = lock_inner(&self.inner);
        if let Some(inner) = guard.as_ref() {
            inner.end_query(encoder, q);
        }
    }

    /// フレーム記録終了 (提出直前に呼ぶ)。使用分を解決バッファへ回収する。
    pub(crate) fn frame_end(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.is_enabled() {
            return;
        }
        let mut guard = lock_inner(&self.inner);
        if let Some(inner) = guard.as_mut() {
            inner.resolve_queries(encoder);
        }
    }

    /// 提出直後に呼ぶ。フレーム境界を確定する。
    pub(crate) fn frame_submitted(&self) {
        if !self.is_enabled() {
            return;
        }
        let mut guard = lock_inner(&self.inner);
        if let Some(inner) = guard.as_mut() {
            if let Err(e) = inner.end_frame() {
                log::warn!("profiler end_frame failed: {:?}", e);
            }
        }
    }

    /// 直近の完成フレームを (ラベル, ミリ秒) の平坦列で回収する。
    /// 非同期回収のため最大数フレーム遅延する。未完成時は前回値を返す。
    pub fn take_ms(&self) -> Vec<(String, f32)> {
        if !self.is_enabled() {
            return Vec::new();
        }
        let period = self.context.queue.get_timestamp_period();
        let mut guard = lock_inner(&self.inner);
        let inner = match guard.as_mut() {
            Some(v) => v,
            None => return Vec::new(),
        };
        if let Some(results) = inner.process_finished_frame(period) {
            let mut flat = Vec::new();
            flatten_results(&results, &mut flat);
            *lock_flat(&self.last_flat) = flat.clone();
            *lock_tree(&self.last_tree) = results;
            flat
        } else {
            lock_flat(&self.last_flat).clone()
        }
    }

    /// 直近の完成フレームツリーを取得する（chrometrace書出し用）。
    pub fn last_tree(&self) -> Vec<wgpu_profiler::GpuTimerQueryResult> {
        lock_tree(&self.last_tree).clone()
    }

    /// 直近の完成フレームを chrometrace JSON へ保存する。
    pub fn save_chrometrace(&self, path: &Path) -> std::io::Result<()> {
        let tree = lock_tree(&self.last_tree);
        wgpu_profiler::chrometrace::write_chrometrace(path, &tree)
    }
}

fn flatten_results(
    results: &[wgpu_profiler::GpuTimerQueryResult],
    out: &mut Vec<(String, f32)>,
) {
    for r in results {
        if let Some(time) = &r.time {
            let ms = ((time.end - time.start) * 1000.0) as f32;
            out.push((r.label.clone(), ms));
        }
        if !r.nested_queries.is_empty() {
            flatten_results(&r.nested_queries, out);
        }
    }
}
