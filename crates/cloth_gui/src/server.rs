use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{mpsc, Arc, RwLock};
use std::thread;
use std::time::Duration;

use crate::protocol::{GuiCommand, GuiResponse};

/// GUIメインスレッドとサーバー間で共有されるシミュレーション状態
#[derive(Clone, Debug, Default)]
pub struct SharedSimState {
    pub is_connected: bool,
    pub is_running: bool,
    pub frame_seq: u64,
    pub num_vertices: u32,
    pub physics_fps: f32,
    pub render_fps: f32,
    pub step_time_ms: f32,
    pub latest_coords: Vec<[f32; 3]>,
    pub applied_pose: Option<Vec<[f32; 3]>>,
    pub needs_coords: bool,
}

/// メインスレッドへ委譲するタスク
pub enum MainThreadTask {
    InitScene {
        data: crate::protocol::SceneInitData,
        respond_to: mpsc::Sender<GuiResponse>,
    },
    Step {
        dt: f32,
        substeps: u32,
        solver_iterations: u32,
        respond_to: mpsc::Sender<GuiResponse>,
    },
    Command(GuiCommand),
}

pub struct TcpServerHandle {
    pub shared_state: Arc<RwLock<SharedSimState>>,
    pub task_rx: mpsc::Receiver<MainThreadTask>,
    _thread: thread::JoinHandle<()>,
}

impl TcpServerHandle {
    pub fn start(bind_addr: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let listener = TcpListener::bind(bind_addr)?;
        listener.set_nonblocking(true)?;
        log::info!("[Server] Listening on {}", bind_addr);

        let shared_state = Arc::new(RwLock::new(SharedSimState::default()));
        let (task_tx, task_rx) = mpsc::channel();

        let state_clone = Arc::clone(&shared_state);
        let thread = thread::spawn(move || {
            server_loop(listener, state_clone, task_tx);
        });

        Ok(Self {
            shared_state,
            task_rx,
            _thread: thread,
        })
    }
}

fn server_loop(
    listener: TcpListener,
    shared_state: Arc<RwLock<SharedSimState>>,
    task_tx: mpsc::Sender<MainThreadTask>,
) {
    let mut active_stream: Option<TcpStream> = None;
    let mut read_buf = Vec::with_capacity(65536);

    loop {
        // 新規接続の受け入れ
        if active_stream.is_none() {
            match listener.accept() {
                Ok((stream, addr)) => {
                    log::info!("[Server] Client connected from: {}", addr);
                    let _ = stream.set_nodelay(true);
                    let _ = stream.set_nonblocking(true);
                    active_stream = Some(stream);
                    if let Ok(mut s) = shared_state.write() {
                        s.is_connected = true;
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(10));
                }
                Err(e) => {
                    log::error!("[Server] Accept error: {}", e);
                    thread::sleep(Duration::from_millis(100));
                }
            }
        }

        // 接続中のクライアントからメッセージを受信
        if let Some(ref mut stream) = active_stream {
            let mut temp_buf = [0u8; 131072];
            match stream.read(&mut temp_buf) {
                Ok(0) => {
                    // 切断検知
                    log::info!("[Server] Client disconnected");
                    active_stream = None;
                    read_buf.clear();
                    if let Ok(mut s) = shared_state.write() {
                        s.is_connected = false;
                    }
                    continue;
                }
                Ok(n) => {
                    read_buf.extend_from_slice(&temp_buf[..n]);
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // データ待機
                }
                Err(e) => {
                    log::warn!("[Server] Read error: {}", e);
                    active_stream = None;
                    read_buf.clear();
                    if let Ok(mut s) = shared_state.write() {
                        s.is_connected = false;
                    }
                    continue;
                }
            }

            // パケット解析 (長さプレフィックス方式: 4バイト Big-Endian + JSONペイロード)
            while read_buf.len() >= 4 {
                let length = u32::from_be_bytes([read_buf[0], read_buf[1], read_buf[2], read_buf[3]]) as usize;
                if read_buf.len() < 4 + length {
                    break; // パケット全体が届くまで待機
                }

                let payload = &read_buf[4..4 + length];
                let cmd_result: Result<GuiCommand, _> = serde_json::from_slice(payload);
                read_buf.drain(..4 + length);

                match cmd_result {
                    Ok(cmd) => {
                        handle_command(cmd, stream, &shared_state, &task_tx);
                    }
                    Err(e) => {
                        log::error!("[Server] JSON parse error: {}", e);
                        let resp = GuiResponse::Error { message: format!("JSON error: {}", e) };
                        let _ = send_response(stream, &resp);
                    }
                }
            }
        }

        thread::sleep(Duration::from_millis(1));
    }
}

fn handle_command(
    cmd: GuiCommand,
    stream: &mut TcpStream,
    shared_state: &Arc<RwLock<SharedSimState>>,
    task_tx: &mpsc::Sender<MainThreadTask>,
) {
    match cmd {
        GuiCommand::GetStatus => {
            let state = shared_state.read().unwrap();
            let resp = GuiResponse::Status {
                running: state.is_running,
                current_frame: state.frame_seq,
                num_vertices: state.num_vertices,
                physics_fps: state.physics_fps,
                render_fps: state.render_fps,
                step_time_ms: state.step_time_ms,
            };
            let _ = send_response(stream, &resp);
        }
        GuiCommand::GetLatestCoords => {
            let mut state = shared_state.write().unwrap();
            state.needs_coords = true;
            let resp = GuiResponse::Coords {
                frame_seq: state.frame_seq,
                num_vertices: state.num_vertices,
                physics_fps: state.physics_fps,
                render_fps: state.render_fps,
                step_time_ms: state.step_time_ms,
                positions: state.latest_coords.clone(),
            };
            let _ = send_response(stream, &resp);
        }
        GuiCommand::ApplyPose => {
            let mut state = shared_state.write().unwrap();
            let pose = state.latest_coords.clone();
            state.applied_pose = Some(pose.clone());
            let resp = GuiResponse::AppliedPose {
                num_vertices: pose.len() as u32,
                positions: pose,
            };
            let _ = send_response(stream, &resp);
        }
        GuiCommand::InitScene(data) => {
            let (tx, rx) = mpsc::channel();
            if let Err(e) = task_tx.send(MainThreadTask::InitScene { data, respond_to: tx }) {
                let resp = GuiResponse::Error { message: format!("Task send error: {}", e) };
                let _ = send_response(stream, &resp);
            } else {
                match rx.recv_timeout(Duration::from_secs(15)) {
                    Ok(resp) => {
                        let _ = send_response(stream, &resp);
                    }
                    Err(e) => {
                        let resp = GuiResponse::Error { message: format!("InitScene timeout: {}", e) };
                        let _ = send_response(stream, &resp);
                    }
                }
            }
        }
        GuiCommand::Step { dt, substeps, solver_iterations } => {
            let (tx, rx) = mpsc::channel();
            if let Err(e) = task_tx.send(MainThreadTask::Step { dt, substeps, solver_iterations, respond_to: tx }) {
                let resp = GuiResponse::Error { message: format!("Task send error: {}", e) };
                let _ = send_response(stream, &resp);
            } else {
                match rx.recv_timeout(Duration::from_secs(5)) {
                    Ok(resp) => {
                        let _ = send_response(stream, &resp);
                    }
                    Err(e) => {
                        let resp = GuiResponse::Error { message: format!("Step timeout: {}", e) };
                        let _ = send_response(stream, &resp);
                    }
                }
            }
        }
        other => {
            // メインスレッドへコマンドを転送
            if let Err(e) = task_tx.send(MainThreadTask::Command(other)) {
                log::error!("[Server] Failed to forward command: {}", e);
                let resp = GuiResponse::Error { message: "Internal server error".to_string() };
                let _ = send_response(stream, &resp);
            } else {
                let resp = GuiResponse::Ack { message: "OK".to_string() };
                let _ = send_response(stream, &resp);
            }
        }
    }
}

pub fn send_response(stream: &mut TcpStream, resp: &GuiResponse) -> std::io::Result<()> {
    let json_bytes = serde_json::to_vec(resp)?;
    let length = (json_bytes.len() as u32).to_be_bytes();
    stream.write_all(&length)?;
    stream.write_all(&json_bytes)?;
    stream.flush()?;
    Ok(())
}
