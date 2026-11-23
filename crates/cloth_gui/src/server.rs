use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::protocol::{GuiCommand, GuiResponse};

const MAX_IPC_PAYLOAD_BYTES: usize = 128 * 1024 * 1024;

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
    ApplyPose {
        respond_to: mpsc::Sender<GuiResponse>,
    },
    Command(GuiCommand),
}

pub struct TcpServerHandle {
    pub shared_state: Arc<RwLock<SharedSimState>>,
    pub task_rx: mpsc::Receiver<MainThreadTask>,
    shutdown: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl TcpServerHandle {
    #[cfg(test)]
    pub fn new_dummy_for_test() -> Self {
        let (_tx, task_rx) = mpsc::channel();
        Self {
            shared_state: Arc::new(RwLock::new(SharedSimState::default())),
            task_rx,
            shutdown: Arc::new(AtomicBool::new(false)),
            thread: None,
        }
    }
    pub fn start(bind_addr: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let listener = TcpListener::bind(bind_addr)?;
        listener.set_nonblocking(true)?;
        log::info!("[Server] Listening on {}", bind_addr);

        let shared_state = Arc::new(RwLock::new(SharedSimState::default()));
        let (task_tx, task_rx) = mpsc::channel();
        let shutdown = Arc::new(AtomicBool::new(false));

        let state_clone = Arc::clone(&shared_state);
        let shutdown_clone = Arc::clone(&shutdown);
        let thread = thread::spawn(move || {
            server_loop(listener, state_clone, task_tx, shutdown_clone);
        });

        Ok(Self {
            shared_state,
            task_rx,
            shutdown,
            thread: Some(thread),
        })
    }

    pub fn stop(&mut self) {
        log::info!("[Server] Stopping TCP server thread...");
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn server_loop(
    listener: TcpListener,
    shared_state: Arc<RwLock<SharedSimState>>,
    task_tx: mpsc::Sender<MainThreadTask>,
    shutdown: Arc<AtomicBool>,
) {
    let mut active_stream: Option<TcpStream> = None;
    let mut read_buf = Vec::with_capacity(65536);

    while !shutdown.load(Ordering::Relaxed) {
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
                    thread::sleep(Duration::from_millis(5));
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
                if length > MAX_IPC_PAYLOAD_BYTES {
                    log::error!("[Server] Rejecting oversized IPC payload: {} bytes", length);
                    let resp = GuiResponse::Error {
                        message: format!("Payload too large: {} bytes", length),
                    };
                    let _ = send_response(stream, &resp);
                    active_stream = None;
                    read_buf.clear();
                    if let Ok(mut s) = shared_state.write() {
                        s.is_connected = false;
                    }
                    break;
                }
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
            let (tx, rx) = mpsc::channel();
            if let Err(e) = task_tx.send(MainThreadTask::ApplyPose { respond_to: tx }) {
                let resp = GuiResponse::Error { message: format!("Task send error: {}", e) };
                let _ = send_response(stream, &resp);
            } else {
                match rx.recv_timeout(Duration::from_secs(10)) {
                    Ok(resp) => {
                        let _ = send_response(stream, &resp);
                    }
                    Err(e) => {
                        let resp = GuiResponse::Error { message: format!("ApplyPose timeout: {}", e) };
                        let _ = send_response(stream, &resp);
                    }
                }
            }
        }
        GuiCommand::InitScene(data) => {
            let (tx, rx) = mpsc::channel();
            if let Err(e) = task_tx.send(MainThreadTask::InitScene { data, respond_to: tx }) {
                let resp = GuiResponse::Error { message: format!("Task send error: {}", e) };
                let _ = send_response(stream, &resp);
            } else {
                match rx.recv_timeout(Duration::from_secs(30)) {
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

fn write_all_nonblocking(
    stream: &mut TcpStream,
    data: &[u8],
    timeout: Duration,
) -> std::io::Result<()> {
    let deadline = Instant::now() + timeout;
    let mut written = 0usize;
    while written < data.len() {
        match stream.write(&data[written..]) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "failed to write whole buffer",
                ));
            }
            Ok(n) => {
                written += n;
            }
            Err(ref e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::Interrupted =>
            {
                if Instant::now() >= deadline {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "non-blocking socket write timed out",
                    ));
                }
                thread::sleep(Duration::from_millis(1));
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

pub fn send_response(stream: &mut TcpStream, resp: &GuiResponse) -> std::io::Result<()> {
    let json_bytes = serde_json::to_vec(resp)?;
    let length = (json_bytes.len() as u32).to_be_bytes();
    let mut packet = Vec::with_capacity(4 + json_bytes.len());
    packet.extend_from_slice(&length);
    packet.extend_from_slice(&json_bytes);
    write_all_nonblocking(stream, &packet, Duration::from_secs(10))?;
    match stream.flush() {
        Ok(()) => Ok(()),
        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(()),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn send_response_nonblocking_handles_large_payload() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind failed");
        let addr = listener.local_addr().unwrap();

        let client_handle = thread::spawn(move || {
            let mut client = TcpStream::connect(addr).expect("connect failed");
            client
                .set_read_timeout(Some(Duration::from_secs(10)))
                .unwrap();
            let mut len_buf = [0u8; 4];
            client.read_exact(&mut len_buf).expect("read header failed");
            let len = u32::from_be_bytes(len_buf) as usize;
            let mut body = vec![0u8; len];
            client.read_exact(&mut body).expect("read body failed");
            let resp: GuiResponse = serde_json::from_slice(&body).expect("json decode failed");
            resp
        });

        let (mut server_stream, _) = listener.accept().expect("accept failed");
        server_stream.set_nonblocking(true).unwrap();

        let n_verts = 120_000;
        let large_resp = GuiResponse::Coords {
            frame_seq: 42,
            num_vertices: n_verts as u32,
            physics_fps: 60.0,
            render_fps: 60.0,
            step_time_ms: 1.5,
            positions: vec![[1.25, -2.5, 3.75]; n_verts],
        };
        send_response(&mut server_stream, &large_resp).expect("nonblocking send_response failed");
        drop(server_stream);

        match client_handle.join().expect("client thread panicked") {
            GuiResponse::Coords {
                frame_seq,
                num_vertices,
                positions,
                ..
            } => {
                assert_eq!(frame_seq, 42);
                assert_eq!(num_vertices, n_verts as u32);
                assert_eq!(positions.len(), n_verts);
                assert_eq!(positions[n_verts - 1], [1.25, -2.5, 3.75]);
            }
            other => panic!("Unexpected response variant: {:?}", other),
        }
    }
}
