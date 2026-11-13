use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use thiserror::Error;

static GLOBAL_CONTEXT: RwLock<Option<Arc<GpuContext>>> = RwLock::new(None);

/// 最後に明示指定されたデバイス設定（Device Lost 後の同一設定での自動復旧に使用）
static CURRENT_DEVICE_CONFIG: RwLock<(Option<String>, Option<usize>)> = RwLock::new((None, None));

/// GPUデバイス世代カウンタ。`init_or_reset` による差し替えのたびに加算され、
/// プロセス全体の共有パイプラインキャッシュの無効化キーとして使用する。
static DEVICE_EPOCH: AtomicU64 = AtomicU64::new(0);

#[derive(Error, Debug)]
pub enum GpuContextError {
    #[error("適合するGPUアダプタが見つかりませんでした")]
    AdapterNotFound,
    #[error("GPUデバイスの要求に失敗しました: {0}")]
    DeviceRequestFailed(#[from] wgpu::RequestDeviceError),
}

/// 利用可能なGPUデバイス（アダプタ）の情報
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDeviceInfo {
    pub index: usize,
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub driver: String,
}

/// デバイスのバッファ関連上限値（SDFベイク可否判定・VRAM予算解決用）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuBufferLimits {
    /// 単一バッファの最大サイズ (bytes)。`device.limits().max_buffer_size`
    pub max_buffer_size: u64,
    /// ストレージバッファ binding の最大サイズ (bytes)。
    /// `device.limits().max_storage_buffer_binding_size`
    pub max_storage_buffer_binding_size: u64,
    /// 3Dテクスチャの最大一辺 (texel)。`device.limits().max_texture_dimension_3d`
    pub max_texture_dimension_3d: u32,
}

pub struct GpuContext {
    // RustのDrop順序はフィールド定義順（先頭から末尾）。
    // 子オブジェクト（Queue, Device）を親オブジェクト（Adapter, Instance）より先にDropさせるため、
    // 依存順序の逆順（Queue -> Device -> Adapter -> Instance）で宣言する。
    pub queue: wgpu::Queue,
    pub device: wgpu::Device,
    pub adapter: wgpu::Adapter,
    pub instance: wgpu::Instance,
    /// 生成時点のデバイス世代 (`DEVICE_EPOCH` のスナップショット)。
    /// 共有パイプラインキャッシュのキーに使用する。
    pub epoch: u64,
    /// Device Lost / 回復不能なGPUエラー発生フラグ（インスタンス固有）
    device_lost: Arc<AtomicBool>,
}

fn format_backend(b: wgpu::Backend) -> String {
    match b {
        wgpu::Backend::Vulkan => "Vulkan".to_string(),
        wgpu::Backend::Metal => "Metal".to_string(),
        wgpu::Backend::Dx12 => "DirectX 12".to_string(),
        wgpu::Backend::Gl => "OpenGL".to_string(),
        wgpu::Backend::BrowserWebGpu => "WebGPU".to_string(),
        _ => format!("{:?}", b),
    }
}

fn format_device_type(d: wgpu::DeviceType) -> String {
    match d {
        wgpu::DeviceType::DiscreteGpu => "DiscreteGPU".to_string(),
        wgpu::DeviceType::IntegratedGpu => "IntegratedGPU".to_string(),
        wgpu::DeviceType::Cpu => "CPU".to_string(),
        wgpu::DeviceType::VirtualGpu => "VirtualGPU".to_string(),
        wgpu::DeviceType::Other => "Other".to_string(),
    }
}

fn parse_backends(backend_str: Option<&str>) -> wgpu::Backends {
    match backend_str.map(|s| s.to_lowercase()).as_deref() {
        Some("dx12") => wgpu::Backends::DX12,
        Some("vulkan") => wgpu::Backends::VULKAN,
        Some("metal") => wgpu::Backends::METAL,
        Some("gl") => wgpu::Backends::GL,
        Some("all") => wgpu::Backends::all(),
        _ => {
            // Windows の場合: Blender本体描画との競合を防止するため、DirectX 12 を最優先
            // その他 (Linux/macOS): PRIMARY (Vulkan / Metal)
            if cfg!(target_os = "windows") {
                wgpu::Backends::DX12
            } else {
                wgpu::Backends::PRIMARY
            }
        }
    }
}

impl GpuContext {
    /// プロセス内で共有されるグローバルな GPU Context を取得または初期化する。
    /// Device Lost 済みのコンテキストが保持されている場合は同一設定で自動再初期化する。
    pub fn get_or_init() -> Result<Arc<Self>, GpuContextError> {
        {
            let read_lock = GLOBAL_CONTEXT.read().unwrap();
            if let Some(ctx) = read_lock.as_ref() {
                if !ctx.is_device_lost() {
                    return Ok(Arc::clone(ctx));
                }
            }
        }
        let (backend, device_index) = {
            let cfg = CURRENT_DEVICE_CONFIG.read().unwrap();
            (cfg.0.clone(), cfg.1)
        };
        Self::init_or_reset(backend.as_deref(), device_index)
    }

    /// このGPUコンテキストで Device Lost が検出済みかを返す
    pub fn is_device_lost(&self) -> bool {
        self.device_lost.load(Ordering::SeqCst)
    }

    /// このGPUコンテキストを Device Lost 状態として明示マークする（異常検知・テスト用）
    pub fn mark_device_lost(&self) {
        self.device_lost.store(true, Ordering::SeqCst);
    }

    /// 利用可能なGPUデバイスの一覧を取得する（指定バックエンド対応デバイスのみを走査）
    pub fn enumerate_available_devices(backend_str: Option<&str>) -> Vec<GpuDeviceInfo> {
        let backends = parse_backends(backend_str);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        instance
            .enumerate_adapters(backends)
            .into_iter()
            .enumerate()
            .map(|(idx, adapter)| {
                let info = adapter.get_info();
                GpuDeviceInfo {
                    index: idx,
                    name: info.name,
                    backend: format_backend(info.backend),
                    device_type: format_device_type(info.device_type),
                    driver: info.driver,
                }
            })
            .collect()
    }

    /// 新規 GPU Context を同期的に作成する
    pub fn new() -> Result<Self, GpuContextError> {
        pollster::block_on(Self::new_async())
    }

    /// 新規 GPU Context を非同期に作成する
    pub async fn new_async() -> Result<Self, GpuContextError> {
        Self::create_device_async(None, None).await
    }

    /// 指定されたバックエンドおよびデバイスで新規 GPU Context を作成する
    pub async fn create_device_async(
        backend_str: Option<&str>,
        device_index: Option<usize>,
    ) -> Result<Self, GpuContextError> {
        let backends = parse_backends(backend_str);
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapter = if let Some(idx) = device_index {
            let adapters = instance.enumerate_adapters(backends);
            adapters
                .into_iter()
                .nth(idx)
                .ok_or(GpuContextError::AdapterNotFound)?
        } else {
            // 1. 指定バックエンドで高パフォーマンスアダプタを検索
            let adapter_opt = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await;

            let adapter_opt = match adapter_opt {
                Some(a) => Some(a),
                None => {
                    // 2. ソフトウェアアダプタ (Lavapipe, WARP 等) へのフォールバック
                    instance
                        .request_adapter(&wgpu::RequestAdapterOptions {
                            power_preference: wgpu::PowerPreference::LowPower,
                            compatible_surface: None,
                            force_fallback_adapter: true,
                        })
                        .await
                }
            };

            match adapter_opt {
                Some(a) => a,
                None if cfg!(target_os = "windows") && backends == wgpu::Backends::DX12 => {
                    // WindowsでDX12が取得できなかった場合はPRIMARYにフォールバック
                    let fallback_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                        backends: wgpu::Backends::PRIMARY,
                        ..Default::default()
                    });
                    let fallback_adapter = fallback_instance
                        .request_adapter(&wgpu::RequestAdapterOptions {
                            power_preference: wgpu::PowerPreference::HighPerformance,
                            compatible_surface: None,
                            force_fallback_adapter: false,
                        })
                        .await;
                    match fallback_adapter {
                        Some(a) => a,
                        None => {
                            fallback_instance
                                .request_adapter(&wgpu::RequestAdapterOptions {
                                    power_preference: wgpu::PowerPreference::LowPower,
                                    compatible_surface: None,
                                    force_fallback_adapter: true,
                                })
                                .await
                                .ok_or(GpuContextError::AdapterNotFound)?
                        }
                    }
                }
                None => return Err(GpuContextError::AdapterNotFound),
            }
        };

        let adapter_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::default();
        // アダプタがサポートする限界値までバッファサイズとストレージバインディングサイズを引き上げる
        required_limits.max_buffer_size = adapter_limits.max_buffer_size;
        required_limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        required_limits.max_storage_buffers_per_shader_stage = adapter_limits.max_storage_buffers_per_shader_stage.min(16).max(8);
        required_limits.max_compute_workgroup_storage_size = adapter_limits.max_compute_workgroup_storage_size;
        required_limits.max_compute_invocations_per_workgroup = adapter_limits.max_compute_invocations_per_workgroup;
        required_limits.max_compute_workgroup_size_x = adapter_limits.max_compute_workgroup_size_x;
        required_limits.max_compute_workgroup_size_y = adapter_limits.max_compute_workgroup_size_y;
        required_limits.max_compute_workgroup_size_z = adapter_limits.max_compute_workgroup_size_z;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("TareminCloth Device"),
                    // 診断用の時刻問い合わせは対応時のみ要求する (非対応でも起動できる)。
                    // 親子ネストのエンコーダ計測を有効にするため INSIDE 系も要求する。
                    required_features: adapter.features()
                        & (wgpu::Features::TIMESTAMP_QUERY
                            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES),
                    required_limits,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await?;

        let device_lost = Arc::new(AtomicBool::new(false));
        let lost_flag_cb = Arc::clone(&device_lost);
        device.set_device_lost_callback(move |reason, message| {
            lost_flag_cb.store(true, Ordering::SeqCst);
            if reason != wgpu::DeviceLostReason::Destroyed {
                eprintln!("[taremin_cloth] GPU Device Lost ({reason:?}): {message}");
            }
        });

        let lost_flag_err = Arc::clone(&device_lost);
        device.on_uncaptured_error(Box::new(move |error| {
            let msg = error.to_string();
            let is_lost_err = matches!(
                error,
                wgpu::Error::OutOfMemory { .. } | wgpu::Error::Internal { .. }
            ) || msg.contains("Device is lost")
                || msg.contains("Parent device is lost");

            if is_lost_err || lost_flag_err.load(Ordering::SeqCst) {
                let already_lost = lost_flag_err.swap(true, Ordering::SeqCst);
                if !already_lost {
                    eprintln!("[taremin_cloth] Uncaptured GPU error (marking device lost): {msg}");
                }
                return;
            }
            // 通常の Validation エラー（バグ）は従来通り即時検知させる
            panic!("wgpu validation error: {error}");
        }));

        Ok(Self {
            queue,
            device,
            adapter,
            instance,
            epoch: DEVICE_EPOCH.load(Ordering::SeqCst),
            device_lost,
        })
    }

    /// 同期的に指定バックエンド・デバイスで GPU Context を初期化・再設定する
    pub fn init_or_reset(
        backend_str: Option<&str>,
        device_index: Option<usize>,
    ) -> Result<Arc<Self>, GpuContextError> {
        pollster::block_on(Self::init_or_reset_async(backend_str, device_index))
    }

    /// 非同期に指定バックエンド・デバイスで GPU Context を初期化・再設定する
    pub async fn init_or_reset_async(
        backend_str: Option<&str>,
        device_index: Option<usize>,
    ) -> Result<Arc<Self>, GpuContextError> {
        {
            let mut cfg = CURRENT_DEVICE_CONFIG.write().unwrap();
            *cfg = (backend_str.map(str::to_string), device_index);
        }
        // Device Lost 状態の旧コンテキストが残っていれば先に破棄して資源を解放する
        {
            let mut write_lock = GLOBAL_CONTEXT.write().unwrap();
            if write_lock.as_ref().is_some_and(|ctx| ctx.is_device_lost()) {
                *write_lock = None;
            }
        }
        // 世代を先に進めて旧デバイス由来の共有パイプライン再利用を防止する
        DEVICE_EPOCH.fetch_add(1, Ordering::SeqCst);
        let ctx = Arc::new(Self::create_device_async(backend_str, device_index).await?);
        {
            let mut write_lock = GLOBAL_CONTEXT.write().unwrap();
            *write_lock = Some(Arc::clone(&ctx));
        }
        Ok(ctx)
    }

    /// 現在のGPUデバイス情報を取得する
    pub fn get_current_device_info() -> Result<GpuDeviceInfo, GpuContextError> {
        let ctx = Self::get_or_init()?;
        let info = ctx.get_adapter_info();
        Ok(GpuDeviceInfo {
            index: 0,
            name: info.name,
            backend: format_backend(info.backend),
            device_type: format_device_type(info.device_type),
            driver: info.driver,
        })
    }

    /// 現在のデバイスのバッファ関連上限値を取得する。
    ///
    /// wgpu のデフォルト下限（`max_buffer_size = 256MiB`）ではなく、
    /// 初期化済みデバイスの実効値（アダプタ上限まで引き上げ済み）を返す。
    /// SDFベイク前の事前検証および Python 側の VRAM 予算解決に使用する。
    pub fn get_device_buffer_limits() -> Result<GpuBufferLimits, GpuContextError> {
        let ctx = Self::get_or_init()?;
        let limits = ctx.device.limits();
        Ok(GpuBufferLimits {
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size as u64,
            max_texture_dimension_3d: limits.max_texture_dimension_3d,
        })
    }

    /// アダプタの情報を取得する
    pub fn get_adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// 診断用の時刻問い合わせに対応しているかを取得する
    pub fn timestamp_query_supported(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
    }

    /// グローバルに保持されている GPU Context を解放する（プロセス終了時・明示的クリーンアップ用）
    pub fn clear_global_context() {
        let mut write_lock = GLOBAL_CONTEXT.write().unwrap();
        *write_lock = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_creation() {
        let context_result = GpuContext::get_or_init();
        match context_result {
            Ok(ctx) => {
                let info = ctx.get_adapter_info();
                println!("GPU Adapter Name: {}", info.name);
                println!("GPU Backend: {:?}", info.backend);
                let dev_limits = ctx.device.limits();
                println!("Device max_buffer_size: {} bytes ({} MB)", dev_limits.max_buffer_size, dev_limits.max_buffer_size / (1024 * 1024));
                assert!(!info.name.is_empty(), "アダプタ名が空であってはならない");
                assert!(dev_limits.max_buffer_size >= 128 * 1024 * 1024, "max_buffer_size は128MB以上");
            }
            Err(GpuContextError::AdapterNotFound) => {
                eprintln!("警告: GPU アダプタが検出されませんでした (GPUドライバのないCI環境の可能性があります)");
            }
            Err(e) => {
                eprintln!("GPU Context 初期化エラー: {:?}", e);
                panic!("GPU Context の作成に失敗しました: {}", e);
            }
        }
    }

    #[test]
    fn test_enumerate_devices() {
        let devices = GpuContext::enumerate_available_devices(None);
        println!("検出されたGPUデバイス一覧: {:?}", devices);
        if devices.is_empty() {
            eprintln!("注意: 利用可能なGPUデバイスが検出されませんでした (CI環境の可能性があります)");
        }
    }

    #[test]
    fn test_reinit_gpu_context() {
        // Windows は DX12、それ以外 (Linux / macOS) は PRIMARY (Vulkan / Metal)
        let backend = if cfg!(target_os = "windows") {
            Some("dx12")
        } else {
            Some("vulkan")
        };
        let ctx_res = GpuContext::init_or_reset(backend, None);
        match ctx_res {
            Ok(_ctx) => {
                let dev_info = GpuContext::get_current_device_info().unwrap();
                println!("現在のデバイス情報: {:?}", dev_info);
                assert!(!dev_info.name.is_empty());
            }
            Err(GpuContextError::AdapterNotFound) => {
                eprintln!("警告: 指定バックエンドのアダプタが検出されませんでした (CI環境の可能性があります)");
            }
            Err(e) => {
                panic!("GPU初期化テスト失敗: {}", e);
            }
        }
    }

    #[test]
    fn test_device_lost_marks_and_recovers() {
        let first = match GpuContext::get_or_init() {
            Ok(ctx) => ctx,
            Err(GpuContextError::AdapterNotFound) => return,
            Err(e) => panic!("GPU Context 取得失敗: {e}"),
        };
        assert!(!first.is_device_lost());
        let old_epoch = first.epoch;

        first.mark_device_lost();
        assert!(first.is_device_lost());

        let recovered = GpuContext::get_or_init().expect("Device Lost 後の自動再初期化に成功すべき");
        assert!(!recovered.is_device_lost());
        assert!(recovered.epoch > old_epoch, "再初期化後は DEVICE_EPOCH が更新されるべき");
    }
}
