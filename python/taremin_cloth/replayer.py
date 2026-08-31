"""
Blender非依存の完全再現リプレイヤー (ClothReplayer)
ログファイル (.jsonl.gz) からシミュレーションを100%同一条件で再実行し、
挙動の検証・比較やサブステップ顕微鏡解析を行います。
"""

import gzip
import json
from typing import Any, Dict, Generator, List, Optional, Tuple
import numpy as np

import taremin_cloth_core


def read_metadata(log_path: str) -> Dict[str, Any]:
    """ログファイル (.jsonl.gz) の1行目からメタデータを読み取ります。"""
    with gzip.open(log_path, "rt", encoding="utf-8") as f:
        first_line = f.readline()
        if not first_line:
            raise ValueError(f"ログファイルが空です: {log_path}")
        data = json.loads(first_line)
        if data.get("record_type") == "metadata":
            return data["metadata"]
        raise ValueError(f"先頭行がメタデータレコードではありません: {first_line[:100]}")


def iter_frames(log_path: str) -> Generator[Dict[str, Any], None, None]:
    """
    ログファイルを行単位でストリーミング走査し、各フレームのデータを生成します。
    ファイル全体を一括パースしないため、メモリを消費しません。
    """
    with gzip.open(log_path, "rt", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # クラッシュ時の不完全な末尾行は安全に無視
                break
            if data.get("record_type") == "frame":
                yield data


def get_frame(log_path: str, target_frame_idx: int) -> Optional[Dict[str, Any]]:
    """指定したフレーム番号のデータをピンポイントで読み取ります（未存在時はNone）。"""
    for frame in iter_frames(log_path):
        if frame.get("frame_index") == target_frame_idx:
            return frame
    return None


class ClothReplayer:
    """
    ログデータからシミュレータを完全初期化し、決定論的に再実行するリプレイヤークラス。
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.metadata = read_metadata(log_path)
        self.sim: Optional[taremin_cloth_core.ClothSimulator] = None
        self.edges = np.array(self.metadata.get("edges", []), dtype=np.uint32)
        self.faces = np.array(self.metadata.get("faces", []), dtype=np.uint32)
        self.num_vertices = int(self.metadata.get("num_vertices", 0))

    def create_simulator(self, initial_positions: np.ndarray) -> taremin_cloth_core.ClothSimulator:
        """メタデータに基づいてシミュレータインスタンスを完全パラメータで生成"""
        inv_masses = np.array(self.metadata.get("inv_masses", []), dtype=np.float32)
        if len(inv_masses) != len(initial_positions):
            inv_masses = np.ones(len(initial_positions), dtype=np.float32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=initial_positions.astype(np.float32),
            edges=self.edges,
            faces=self.faces,
            inv_masses=inv_masses,
            thickness=float(self.metadata.get("thickness", 0.005)),
            stiffness=float(self.metadata.get("stiffness", 500.0)),
            bending_stiffness=float(self.metadata.get("bending_stiffness", 5.0)),
        )

        # 物理パラメータの反映
        gravity = self.metadata.get("gravity", [0.0, 0.0, -9.81])
        sim.set_gravity(float(gravity[0]), float(gravity[1]), float(gravity[2]))
        sim.set_damping(float(self.metadata.get("damping", 0.01)))

        # ソルバー設定
        sim.set_solver_mode(int(self.metadata.get("solver_mode", 0)))
        sim.set_solver_iterations(int(self.metadata.get("solver_iterations", 2)))

        # セルフコリジョン設定
        enable_sc = bool(self.metadata.get("enable_self_collision", True))
        sim.set_enable_self_collision(enable_sc)
        if enable_sc:
            sim.set_self_collision_options(
                relief_factor=float(self.metadata.get("self_collision_relief_factor", 0.2)),
                max_displacement_ratio=float(self.metadata.get("self_collision_max_displacement_ratio", 0.2)),
                exclude_neighbors=bool(self.metadata.get("self_collision_exclude_neighbors", True)),
                enable_normal_untangling=bool(self.metadata.get("enable_normal_untangling", True)),
            )

        # エッジコリジョン設定
        enable_ec = bool(self.metadata.get("enable_edge_collision", False))
        sim.set_enable_edge_collision(enable_ec)
        if enable_ec:
            scale = float(self.metadata.get("edge_margin_scale", 1.0))
            offset = float(self.metadata.get("edge_margin_offset", 0.0))
            sim.set_edge_margin_scale(scale)
            sim.set_edge_margin_offset(offset)

        self.sim = sim
        return sim

    def apply_frame_inputs(self, frame_data: Dict[str, Any]) -> None:
        """フレームに記録された外部入力（ピン・コライダー）をシミュレータに適用"""
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")

        # 1. 動的ピンの適用
        self.sim.clear_pins()
        for pin in frame_data.get("pins", []):
            v_idx = int(pin["vertex_idx"])
            t_pos = [float(c) for c in pin["target_pos"]]
            w = float(pin.get("weight", 1.0))
            self.sim.set_pin(v_idx, t_pos, w)

        # 2. コライダーの適用
        self.sim.clear_colliders()
        for col in frame_data.get("colliders", []):
            c_type = col.get("type")
            fric = float(col.get("friction", 0.2))
            rest = float(col.get("restitution", 0.0))

            if c_type == "sphere":
                c_pos = [float(v) for v in col["center"]]
                r = float(col["radius"])
                self.sim.add_sphere_collider(c_pos, r, fric, rest)
            elif c_type == "capsule":
                pt_a = [float(v) for v in col["point_a"]]
                pt_b = [float(v) for v in col["point_b"]]
                r = float(col["radius"])
                self.sim.add_capsule_collider(pt_a, pt_b, r, fric, rest)
            elif c_type == "plane":
                pt = [float(v) for v in col["point"]]
                norm = [float(v) for v in col["normal"]]
                self.sim.add_plane_collider(pt, norm, fric, rest)
            elif c_type == "mesh":
                raw_tris = col.get("triangles", [])
                if raw_tris:
                    tris = np.array(raw_tris, dtype=np.float32).reshape((-1, 3, 3))
                    thick = float(col.get("thickness", 0.005))
                    single = bool(col.get("single_sided", True))
                    self.sim.set_mesh_collider_triangles(
                        tris, friction=fric, thickness=thick, restitution=rest, single_sided=single
                    )

    def replay_range(
        self,
        start_frame_idx: int,
        end_frame_idx: int,
        compare: bool = True
    ) -> List[Dict[str, Any]]:
        """
        指定されたフレーム区間を完全再現し、実測ログと再計算値の差分を検証します。

        Returns:
            各フレームの再現結果リスト
            [{"frame_index": int, "max_diff": float, "positions": np.ndarray}, ...]
        """
        frames = {}
        for f in iter_frames(self.log_path):
            idx = f["frame_index"]
            if start_frame_idx <= idx <= end_frame_idx:
                frames[idx] = f

        if start_frame_idx not in frames:
            raise ValueError(f"開始フレーム {start_frame_idx} がログに存在しません")

        # 開始フレームの頂点座標および速度でシミュレータを完全初期化
        init_pos = np.array(frames[start_frame_idx]["positions"], dtype=np.float32).reshape((-1, 3))
        init_vel = np.array(frames[start_frame_idx]["velocities"], dtype=np.float32).reshape((-1, 3)) if "velocities" in frames[start_frame_idx] else None
        self.create_simulator(init_pos)
        self.sim.set_positions_and_velocities(init_pos, init_vel)

        results = []
        out_pos = np.zeros(self.num_vertices * 3, dtype=np.float32)

        for f_idx in range(start_frame_idx + 1, end_frame_idx + 1):
            if f_idx not in frames:
                break
            fdata = frames[f_idx]
            dt = float(fdata.get("dt", 1.0 / 60.0))
            substeps = int(fdata.get("substeps", 10))

            # 入力の適用とステップ実行
            self.apply_frame_inputs(fdata)
            self.sim.step(dt=dt, substeps=substeps)

            # 座標読み出し
            self.sim.get_positions(out_pos)
            sim_pos = out_pos.reshape((-1, 3)).copy()

            max_diff = 0.0
            if compare and "positions" in fdata:
                log_pos = np.array(fdata["positions"], dtype=np.float32).reshape((-1, 3))
                max_diff = float(np.max(np.abs(sim_pos - log_pos)))

            results.append({
                "frame_index": f_idx,
                "max_diff": max_diff,
                "positions": sim_pos,
            })

        return results

    def trace_substeps(
        self,
        target_frame_idx: int
    ) -> List[Dict[str, Any]]:
        """
        問題フレーム (target_frame_idx) の直前状態から、1サブステップ刻みでステップ実行し、
        サブステップごとの最大変位・最大速度推移を詳細トレースします。
        """
        prev_idx = target_frame_idx - 1
        f_prev = get_frame(self.log_path, prev_idx)
        f_curr = get_frame(self.log_path, target_frame_idx)
        if f_prev is None or f_curr is None:
            raise ValueError(f"Frame {prev_idx} または {target_frame_idx} がログに存在しません")

        init_pos = np.array(f_prev["positions"], dtype=np.float32).reshape((-1, 3))
        init_vel = np.array(f_prev["velocities"], dtype=np.float32).reshape((-1, 3)) if "velocities" in f_prev else None
        self.create_simulator(init_pos)
        self.sim.set_positions_and_velocities(init_pos, init_vel)
        self.apply_frame_inputs(f_curr)

        dt = float(f_curr.get("dt", 1.0 / 60.0))
        substeps = int(f_curr.get("substeps", 10))
        dt_sub = dt / float(substeps)

        substep_traces = []
        out_pos = np.zeros(self.num_vertices * 3, dtype=np.float32)
        prev_sub_pos = init_pos.copy()

        for s in range(substeps):
            self.sim.step_single_substep(dt_sub=dt_sub)
            self.sim.get_positions(out_pos)
            curr_sub_pos = out_pos.reshape((-1, 3)).copy()

            disps = np.linalg.norm(curr_sub_pos - prev_sub_pos, axis=1) * 1000.0  # mm
            max_d = float(np.max(disps))
            max_v = int(np.argmax(disps))

            substep_traces.append({
                "substep": s + 1,
                "dt_sub": dt_sub,
                "max_disp_mm": max_d,
                "max_disp_vert": max_v,
                "positions": curr_sub_pos,
            })
            prev_sub_pos = curr_sub_pos

        return substep_traces
