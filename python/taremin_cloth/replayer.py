"""
Blender非依存の完全再現リプレイヤー (ClothReplayer)
ログファイル (.jsonl.gz) からシミュレーションを100%同一条件で再実行し、
挙動の検証・比較やサブステップ顕微鏡解析を行います。
"""

import gzip
import json
from collections import defaultdict
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
import numpy as np

try:
    import taremin_cloth_core
except ImportError:
    from . import taremin_cloth_core

try:
    from .utils import allow_positional
except ImportError:
    try:
        from taremin_cloth.utils import allow_positional
    except ImportError:
        def allow_positional(fn):
            return fn


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


def frame_has_positions(frame: Optional[Dict[str, Any]]) -> bool:
    """フレームがフル座標を持つか (スパース記録のスタブ判定)。"""
    if frame is None:
        return False
    pos = frame.get("positions")
    return isinstance(pos, list) and len(pos) > 0


def count_full_frames(log_path: str) -> Tuple[int, int]:
    """(全保存数, フル数) を返す。スタブ数 = 全 - フル。"""
    total = 0
    full = 0
    for f in iter_frames(log_path):
        total += 1
        if frame_has_positions(f):
            full += 1
    return total, full


def nearest_full_frame(
    log_path: str,
    target_frame_idx: int,
    *,
    frames: Optional[Iterable[Dict[str, Any]]] = None,
) -> Optional[int]:
    """指定番号に最も近いフル座標フレーム番号を返す (前方向優先)。"""
    best = None
    best_dist = None
    source = frames if frames is not None else iter_frames(log_path)
    for f in source:
        if not frame_has_positions(f):
            continue
        idx = int(f.get("frame_index", -1))
        d = abs(idx - target_frame_idx)
        # 同距離なら前方向を優先
        if best is None or d < best_dist or (d == best_dist and idx < best):
            best = idx
            best_dist = d
    return best


@allow_positional
def config_at_frame(
    log_path: str,
    metadata: Dict[str, Any],
    target_frame_idx: int,
    *,
    frames: Optional[Iterable[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """指定フレーム時点の有効設定を返す (metadata.config + それ以前の最新param_deltas)。

    フレーム途中のNパネル変更を再現するための折り畳み。差分がなければ
    metadata.config (旧ログではNone) をそのまま返す。
    """
    base = metadata.get("config")
    if not isinstance(base, dict):
        return None
    current: Dict[str, Any] = dict(base)
    found = False
    source = frames if frames is not None else iter_frames(log_path)
    for f in source:
        idx = int(f.get("frame_index", -1))
        if idx > target_frame_idx:
            break
        d = f.get("param_deltas")
        if isinstance(d, dict) and d:
            current = d
            found = True
    return current if (found or isinstance(base, dict)) else None


def apply_config_dict(sim: Any, cfg: Optional[Dict[str, Any]]) -> bool:
    """設定辞書をシミュレータに適用する。適用したらTrue。"""
    if not isinstance(cfg, dict) or not cfg:
        return False
    applier = getattr(sim, "apply_config_json", None)
    if not callable(applier):
        return False
    try:
        import json as _json
        applier(_json.dumps(cfg))
        return True
    except Exception:
        return False


def apply_elastic_list(sim: Any, esc: Optional[List[Dict[str, Any]]]) -> bool:
    """辺スケール差分リストをシミュレータに適用する。適用したらTrue。"""
    if not esc:
        return False
    setter = getattr(sim, "set_edge_rest_length_scales", None)
    if not callable(setter):
        return False
    try:
        idx = np.ascontiguousarray(
            np.array([int(e["edge_idx"]) for e in esc], dtype=np.uint32))
        scales = np.ascontiguousarray(
            np.array([float(e["scale"]) for e in esc], dtype=np.float32))
        setter(idx, scales)
        return True
    except Exception:
        return False


def elastic_at_frame(
    log_path: str,
    target_frame_idx: int,
    *,
    frames: Optional[Iterable[Dict[str, Any]]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """指定フレーム時点で有効な辺スケール差分 (それ以前の最新) を返す。

    途中開始リプレイでも変更履歴を取りこぼさないための折り畳み。
    """
    latest: Optional[List[Dict[str, Any]]] = None
    source = frames if frames is not None else iter_frames(log_path)
    for f in source:
        idx = int(f.get("frame_index", -1))
        if idx > target_frame_idx:
            break
        esc = f.get("elastic_scales")
        if isinstance(esc, list) and esc:
            latest = esc
    return latest


def sewing_at_frame(
    log_path: str,
    target_frame_idx: int,
    *,
    frames: Optional[Iterable[Dict[str, Any]]] = None,
) -> Optional[List[float]]:
    """指定フレーム時点で有効な縫合現在自然長 (それ以前の最新) を返す。"""
    latest: Optional[List[float]] = None
    source = frames if frames is not None else iter_frames(log_path)
    for f in source:
        idx = int(f.get("frame_index", -1))
        if idx > target_frame_idx:
            break
        rests = f.get("sewing_rest_lengths")
        if isinstance(rests, list) and rests:
            latest = [float(v) for v in rests]
    return latest


def apply_inv_mass_list(sim: Any, deltas: Optional[List[Dict[str, Any]]]) -> bool:
    """頂点逆質量 (original_inv_masses) 差分リストをシミュレータに適用する。適用したらTrue。"""
    if not deltas:
        return False
    setter = getattr(sim, "set_vertex_inv_masses", None)
    if not callable(setter):
        return False
    try:
        idx = np.ascontiguousarray(
            np.array([int(d["vertex_idx"]) for d in deltas], dtype=np.uint32))
        masses = np.ascontiguousarray(
            np.array([float(d["inv_mass"]) for d in deltas], dtype=np.float32))
        setter(idx, masses)
        return True
    except Exception:
        return False


@allow_positional
def inv_masses_at_frame(
    log_path: str,
    metadata: Dict[str, Any],
    target_frame_idx: int,
    *,
    frames: Optional[Iterable[Dict[str, Any]]] = None,
) -> Optional[np.ndarray]:
    """指定フレーム時点で有効な基準逆質量 (metadata.inv_masses + inv_mass_deltas折り畳み) を返す。"""
    raw_base = metadata.get("inv_masses")
    if not isinstance(raw_base, list) or len(raw_base) == 0:
        return None
    curr = np.array(raw_base, dtype=np.float32).copy()
    source = frames if frames is not None else iter_frames(log_path)
    for f in source:
        idx = int(f.get("frame_index", -1))
        if idx > target_frame_idx:
            break
        deltas = f.get("inv_mass_deltas")
        if isinstance(deltas, list):
            for d in deltas:
                v_idx = int(d["vertex_idx"])
                if 0 <= v_idx < len(curr):
                    curr[v_idx] = float(d["inv_mass"])
    return curr


class ClothReplayer:
    """
    ログデータからシミュレータを完全初期化し、決定論的に再実行するリプレイヤークラス。
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.metadata = read_metadata(log_path)
        self.sim: Optional[taremin_cloth_core.ClothSimulator] = None
        # 辺は original_edges (せん断対角なし) を優先。旧ログは edges に
        # フォールバックする (対角がstretch化される既知の劣化あり)。
        raw_edges = self.metadata.get("original_edges") or self.metadata.get("edges", [])
        self.edges = np.array(raw_edges, dtype=np.uint32)
        self.faces = np.array(self.metadata.get("faces", []), dtype=np.uint32)
        self.num_vertices = int(self.metadata.get("num_vertices", 0))
        # 全フレーム1パスロード用キャッシュ (O(N^2)の再スキャンを回避)
        self._frames_by_idx: Optional[Dict[int, Dict[str, Any]]] = None
        self._ordered_indices: Optional[List[int]] = None
        self._ordered_frames: Optional[List[Dict[str, Any]]] = None
        self._has_inv_mass_deltas_field: Optional[bool] = None
        # 旧ログ互換: inv_mass_deltas未記録時のWaypoint解放コントローラー
        self._legacy_wp_seams: Optional[list] = None
        self._legacy_wp_checked: bool = False
        self._legacy_wp_mgr: Optional[Any] = None
        self._pair_cache_override: Optional[bool] = None

    def create_simulator(self, initial_positions: np.ndarray) -> taremin_cloth_core.ClothSimulator:
        """メタデータに基づいてシミュレータインスタンスを完全パラメータで生成

        構築はレストポーズ (metadata.initial_positions) から行い、呼び出し側が
        フレーム状態を上書きする。これにより距離・曲げ・縫合の自然長や
        local_edge_lengths が順方向と一致する。旧ログではフレーム座標で構築する。
        """
        init_arg = np.asarray(initial_positions, dtype=np.float32).reshape((-1, 3))
        raw_init = self.metadata.get("initial_positions")
        if isinstance(raw_init, list) and len(raw_init) == len(init_arg):
            build_pos = np.ascontiguousarray(np.array(raw_init, dtype=np.float32))
        else:
            build_pos = np.ascontiguousarray(init_arg)
        inv_masses = np.array(self.metadata.get("inv_masses", []), dtype=np.float32)
        if len(inv_masses) != len(build_pos):
            inv_masses = np.ones(len(build_pos), dtype=np.float32)

        # SimConfig正本があれば初期コンストラクタにも反映 (剛性・厚み)
        cfg = self.metadata.get("config") if isinstance(self.metadata.get("config"), dict) else None
        thickness = float(self.metadata.get("thickness", 0.005))
        if cfg is not None:
            # SimConfig由来の剛性を優先 (旧フィールドはfallback)
            stiffness = float(cfg.get("tension_stiffness", self.metadata.get("stiffness", 500.0)))
            bending = float(cfg.get("bending_stiffness", self.metadata.get("bending_stiffness", 5.0)))
            comp_stiff = cfg.get("compression_stiffness", self.metadata.get("compression_stiffness"))
            shear_stiff = cfg.get("shear_stiffness", self.metadata.get("shear_stiffness"))
        else:
            stiffness = float(self.metadata.get("stiffness", 500.0))
            bending = float(self.metadata.get("bending_stiffness", 5.0))
            comp_stiff = self.metadata.get("compression_stiffness")
            shear_stiff = self.metadata.get("shear_stiffness")
        raw_sew = self.metadata.get("sewing_springs")
        if raw_sew is not None and len(raw_sew) > 0:
            sew_arr = np.ascontiguousarray(np.array(raw_sew, dtype=np.uint32))
        else:
            sew_arr = None

        # 縫合収縮速度 (旧ログではNone→既定値1.0)
        shrink_speed = self.metadata.get("sewing_shrink_speed")
        try:
            shrink_speed_f = float(shrink_speed) if shrink_speed is not None else 1.0
        except (TypeError, ValueError):
            shrink_speed_f = 1.0

        # 頂点毎厚み・レイヤー (旧ログではNone→既定値)
        thick_arr = None
        raw_thick = self.metadata.get("thicknesses")
        if isinstance(raw_thick, list) and len(raw_thick) == len(build_pos):
            thick_arr = np.ascontiguousarray(np.array(raw_thick, dtype=np.float32))
        layer_arr = None
        raw_layer = self.metadata.get("layer_ids")
        if isinstance(raw_layer, list) and len(raw_layer) == len(build_pos):
            layer_arr = np.ascontiguousarray(np.array(raw_layer, dtype=np.uint32))
        faces_arg = self.faces if self.faces.size > 0 else None
        try:
            areal_density_f = float(self.metadata.get("areal_density", 0.15))
        except (TypeError, ValueError):
            areal_density_f = 0.15
        coarse_f = bool(self.metadata.get("enable_coarse_constraints", False))

        sim = taremin_cloth_core.ClothSimulator(
            positions=build_pos,
            edges=self.edges,
            faces=faces_arg,
            inv_masses=inv_masses,
            sewing_springs=sew_arr,
            layer_ids=layer_arr,
            thicknesses=thick_arr,
            thickness=thickness,
            stiffness=stiffness,
            bending_stiffness=bending,
            compression_stiffness=float(comp_stiff) if comp_stiff is not None else None,
            shear_stiffness=float(shear_stiff) if shear_stiff is not None else None,
            sewing_shrink_speed=shrink_speed_f,
            areal_density=areal_density_f,
            enable_coarse_constraints=coarse_f,
        )

        # SimConfig一本化: 正本があれば一括適用 (新パラメータ自動追従)
        if cfg is not None and hasattr(sim, "apply_config_json"):
            try:
                import json as _json
                sim.apply_config_json(_json.dumps(cfg))
                self.setup_bone_sdf(sim)
                self.sim = sim
                return sim
            except Exception:
                pass  # 失敗時は旧個別復元にフォールバック

        # 旧ログ互換: 個別復元 (configなしログ用)
        gravity = self.metadata.get("gravity", [0.0, 0.0, -9.81])
        sim.set_gravity(float(gravity[0]), float(gravity[1]), float(gravity[2]))
        sim.set_damping(float(self.metadata.get("damping", 0.01)))
        if hasattr(sim, "set_areal_density"):
            try:
                sim.set_areal_density(float(self.metadata.get("areal_density", 0.15)))
            except (TypeError, ValueError):
                pass

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
                max_iterations=int(self.metadata.get("self_collision_max_iterations", 128)),
            )
            # 欠落していたパラメータも旧ログでは既定値で復元
            if hasattr(sim, "set_coupled_self_collision_options"):
                sim.set_coupled_self_collision_options(0, 1)
            if hasattr(sim, "set_self_collision_substep_interval"):
                sim.set_self_collision_substep_interval(1)
            if hasattr(sim, "set_pair_cache_options"):
                max_pairs = int(self.metadata.get("pair_cache_max_pairs", 65536))
                margin_mode = 0 if self.metadata.get("pair_cache_margin_mode", "AUTO") == "FIXED" else 1
                if isinstance(self.metadata.get("pair_cache_margin_mode"), int):
                    margin_mode = int(self.metadata.get("pair_cache_margin_mode", 1))
                sim.set_pair_cache_options(
                    max_pairs,
                    max_pairs,
                    margin_mode,
                    float(self.metadata.get("pair_cache_safety_margin", 0.005)),
                    float(self.metadata.get("pair_cache_horizon_scale", 1.3)),
                    float(self.metadata.get("pair_cache_max_horizon", 0.02)),
                )
            if hasattr(sim, "set_enable_pair_cache_final_fallback"):
                sim.set_enable_pair_cache_final_fallback(
                    bool(self.metadata.get("enable_pair_cache_final_fallback", True))
                )

        # エッジコリジョン設定
        enable_ec = bool(self.metadata.get("enable_edge_collision", False))
        sim.set_enable_edge_collision(enable_ec)
        if enable_ec:
            scale = float(self.metadata.get("edge_margin_scale", 1.0))
            offset = float(self.metadata.get("edge_margin_offset", 0.0))
            sim.set_edge_margin_scale(scale)
            sim.set_edge_margin_offset(offset)

        # 縫合・優先モード (旧ログ互換)
        if hasattr(sim, "set_sewing_priority_options"):
            try:
                sim.set_sewing_priority_options(
                    bool(self.metadata.get("sewing_priority_enabled", False)),
                    float(self.metadata.get("sewing_priority_threshold", 0.9)),
                    float(self.metadata.get("sewing_priority_merge_dist", 0.005)),
                    int(self.metadata.get("sewing_priority_ramp_frames", 3)),
                    int(self.metadata.get("sewing_priority_max_frames", 600)),
                )
            except Exception:
                pass

        self.setup_bone_sdf(sim)

        self.sim = sim
        return sim

    def setup_bone_sdf(self, sim) -> bool:
        """メタデータのBONE_SDFをシミュレータに復元する。復元したらTrue。

        旧ログ (bone_sdfなし) や対応外バイナリでは何もせずFalse。
        動的再ベイク有効で記録された場合はテクスチャが古い可能性があり、
        呼び出し側で metadata['bone_sdf']['dynamic_enabled'] を確認すること。
        """
        rec = self.metadata.get("bone_sdf")
        if not isinstance(rec, dict):
            return False
        try:
            import base64 as _b64
            tex = _b64.b64decode(rec.get("texture_base64", ""))
            infos = np.ascontiguousarray(
                np.array(rec.get("bone_infos", []), dtype=np.float32))
            if infos.size == 0:
                return False
            sim.set_bone_sdf_colliders(
                int(rec.get("width", 0)), int(rec.get("height", 0)),
                int(rec.get("depth", 0)), tex, infos)
            return True
        except Exception:
            return False

    def apply_frame_bone(self, frame_data: Dict[str, Any]) -> bool:
        """フレームに記録されたボーン姿勢を適用する。適用したらTrue。"""
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        st = frame_data.get("bone_transforms")
        if not isinstance(st, dict):
            return False
        setter = getattr(self.sim, "update_bone_transforms", None)
        if not callable(setter):
            return False
        try:
            world = np.ascontiguousarray(
                np.array(st.get("world", []), dtype=np.float32))
            invm = np.ascontiguousarray(
                np.array(st.get("inv_world", []), dtype=np.float32))
            if world.size == 0 or world.shape != invm.shape:
                return False
            setter(world, invm)
            return True
        except Exception:
            return False

    def apply_frame_config(self, frame_data: Dict[str, Any]) -> bool:
        """フレームに記録された設定差分 (param_deltas) を適用する。適用したらTrue。"""
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        return apply_config_dict(self.sim, frame_data.get("param_deltas"))

    def apply_frame_elastic(self, frame_data: Dict[str, Any]) -> bool:
        """フレームに記録された辺自然長スケール差分を適用する。適用したらTrue。"""
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        return apply_elastic_list(self.sim, frame_data.get("elastic_scales"))

    def _sewing_priority_enabled(self) -> bool:
        """縫合優先モードが有効かをメタデータから判定する。"""
        cfg = self.metadata.get("config")
        if isinstance(cfg, dict) and "sewing_priority_enabled" in cfg:
            return bool(cfg["sewing_priority_enabled"])
        return bool(self.metadata.get("sewing_priority_enabled", False))

    def _update_sewing_priority(self, positions_2d) -> None:
        """直近座標で縫合結合率を測定しラッチ状態を進める (runnerと同順序)。"""
        if self.sim is None or not self._sewing_priority_enabled():
            return
        updater = getattr(self.sim, "update_sewing_priority", None)
        if not callable(updater):
            return
        try:
            updater(np.ascontiguousarray(positions_2d, dtype=np.float32))
        except Exception:
            pass

    def apply_frame_sewing(self, frame_data: Dict[str, Any]) -> bool:
        """フレームに記録された縫合現在自然長を適用する。適用したらTrue。"""
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        rests = frame_data.get("sewing_rest_lengths")
        if not rests:
            return False
        setter = getattr(self.sim, "set_sewing_current_rest_lengths", None)
        if not callable(setter):
            return False
        try:
            setter([float(v) for v in rests])
            return True
        except Exception:
            return False

    def apply_frame_priority(self, frame_data: Dict[str, Any]) -> bool:
        """フレームに記録された縫合優先ラッチ状態を復元する。復元したらTrue。

        ラッチ状態はステップ入力状態 (ホストがステップ前に進める値) のため、
        当該フレーム自身の記録値を使う。記録がない旧ログではFalseを返し、
        呼び出し側はupdater駆動 (近似) にフォールバックすること。
        """
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        st = frame_data.get("sewing_priority")
        if not isinstance(st, list) or len(st) < 4:
            return False
        setter = getattr(self.sim, "set_sewing_priority_state", None)
        if not callable(setter):
            return False
        try:
            setter(bool(float(st[0]) > 0.5), int(st[1]), float(st[2]), float(st[3]))
            return True
        except Exception:
            return False

    def reload_frames(self) -> None:
        """ログの全フレームを1パス走査してインスタンス内キャッシュを再構築する。"""
        by_idx: Dict[int, Dict[str, Any]] = {}
        ordered_frames: List[Dict[str, Any]] = []
        has_inv_field = False
        for f in iter_frames(self.log_path):
            idx = int(f.get("frame_index", -1))
            if "inv_mass_deltas" in f:
                has_inv_field = True
            by_idx[idx] = f
            ordered_frames.append(f)
        ordered_frames.sort(key=lambda item: int(item.get("frame_index", -1)))
        self._frames_by_idx = by_idx
        self._ordered_frames = ordered_frames
        self._ordered_indices = [int(f.get("frame_index", -1)) for f in ordered_frames]
        self._has_inv_mass_deltas_field = has_inv_field

    def _ensure_frames_loaded(self) -> None:
        """初回のみログの全フレームを読み込んでキャッシュする。"""
        if self._frames_by_idx is None:
            self.reload_frames()

    def get_frames_dict(self) -> Dict[int, Dict[str, Any]]:
        """frame_index -> frame_data のキャッシュ済み辞書を返す。"""
        self._ensure_frames_loaded()
        return self._frames_by_idx or {}

    def get_ordered_frames(self) -> List[Dict[str, Any]]:
        """frame_index 昇順のキャッシュ済みフレームリストを返す。"""
        self._ensure_frames_loaded()
        return self._ordered_frames or []

    def get_ordered_frame_indices(self) -> List[int]:
        """昇順のフレーム番号リストを返す。"""
        self._ensure_frames_loaded()
        return self._ordered_indices or []

    def apply_elastic_at(self, frame_idx: int) -> bool:
        """指定フレーム時点で有効な辺スケール (変更履歴の折り畳み) を適用する。"""
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        self._ensure_frames_loaded()
        return apply_elastic_list(
            self.sim,
            elastic_at_frame(self.log_path, frame_idx, frames=self._ordered_frames),
        )

    def apply_sewing_at(self, frame_idx: int) -> bool:
        """指定フレーム時点で有効な縫合現在自然長を適用する。"""
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        self._ensure_frames_loaded()
        rests = sewing_at_frame(self.log_path, frame_idx, frames=self._ordered_frames)
        if not rests:
            return False
        setter = getattr(self.sim, "set_sewing_current_rest_lengths", None)
        if not callable(setter):
            return False
        try:
            setter([float(v) for v in rests])
            return True
        except Exception:
            return False

    def apply_config_at(self, frame_idx: int) -> bool:
        """指定フレーム時点の有効設定 (初期値+差分折り畳み) を適用する。"""
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        self._ensure_frames_loaded()
        return apply_config_dict(
            self.sim,
            config_at_frame(
                self.log_path, self.metadata, frame_idx, frames=self._ordered_frames
            ),
        )

    def apply_frame_inv_masses(self, frame_data: Dict[str, Any]) -> bool:
        """フレームに記録された基準逆質量差分 (inv_mass_deltas) を適用する。"""
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        return apply_inv_mass_list(self.sim, frame_data.get("inv_mass_deltas"))

    def apply_inv_masses_at(self, frame_idx: int) -> bool:
        """指定フレーム時点で有効な基準逆質量 (変更履歴の折り畳み) を適用する。"""
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        self._ensure_frames_loaded()
        target = inv_masses_at_frame(
            self.log_path, self.metadata, frame_idx, frames=self._ordered_frames
        )
        if target is None:
            return False
        setter = getattr(self.sim, "set_vertex_inv_masses", None)
        getter = getattr(self.sim, "get_vertex_inv_masses", None)
        if not callable(setter):
            return False
        try:
            if callable(getter):
                current = np.asarray(getter(), dtype=np.float32)
                if len(current) == len(target):
                    diff_idx = np.flatnonzero(current != target)
                    if len(diff_idx) == 0:
                        return False
                    setter(
                        np.ascontiguousarray(diff_idx, dtype=np.uint32),
                        np.ascontiguousarray(target[diff_idx], dtype=np.float32),
                    )
                    return True
            all_idx = np.arange(len(target), dtype=np.uint32)
            setter(all_idx, np.ascontiguousarray(target, dtype=np.float32))
            return True
        except Exception:
            return False

    def _apply_pair_cache_override(self) -> None:
        """指定がある場合、ペアキャッシュ設定の上書きを再適用する。"""
        if (
            self._pair_cache_override is not None
            and self.sim is not None
            and hasattr(self.sim, "set_enable_pair_cache")
        ):
            self.sim.set_enable_pair_cache(bool(self._pair_cache_override))

    def _extract_legacy_waypoint_seams(self, *, fallback_pos: np.ndarray) -> list:
        """旧ログ(inv_mass_deltas未記録)向けにメタデータからWaypointSeamを再抽出する。"""
        if self._legacy_wp_checked:
            return self._legacy_wp_seams or []
        self._legacy_wp_checked = True
        self._legacy_wp_seams = []

        raw_sew = self.metadata.get("sewing_springs")
        raw_inv = self.metadata.get("inv_masses")
        if not raw_sew or not isinstance(raw_inv, list) or len(raw_inv) == 0:
            return []

        inv0 = np.asarray(raw_inv, dtype=np.float32)
        if not np.any(inv0 == 0.0):
            return []

        raw_init = self.metadata.get("initial_positions")
        if isinstance(raw_init, list) and len(raw_init) == len(inv0):
            pos_ref = np.asarray(raw_init, dtype=np.float32).reshape((-1, 3))
        else:
            pos_ref = np.asarray(fallback_pos, dtype=np.float32).reshape((-1, 3))

        try:
            from .utils.mesh_extract import WaypointSeam
        except ImportError:
            try:
                from taremin_cloth.utils.mesh_extract import WaypointSeam  # type: ignore
            except ImportError:
                return []

        # 面頂点と通常辺(紐等)頂点を合わせた実体頂点集合を端点候補とする
        body_verts: set[int] = set()
        if self.faces.size > 0:
            body_verts.update(int(v) for v in self.faces.ravel())
        if self.edges.size > 0:
            body_verts.update(int(v) for v in self.edges.ravel())

        adj: Dict[int, set[int]] = defaultdict(set)
        for pair in raw_sew:
            a, b = int(pair[0]), int(pair[1])
            if a != b:
                adj[a].add(b)
                adj[b].add(a)

        visited: set[int] = set()
        wp_seams = []
        for start_v in adj:
            if start_v in visited:
                continue
            comp = []
            queue = [start_v]
            visited.add(start_v)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for nb in adj[curr]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)

            endpoints = [x for x in comp if x in body_verts]
            internal = [x for x in comp if x not in body_verts]
            if (
                internal
                and len(endpoints) == 2
                and all(0 <= m < len(inv0) and inv0[m] == 0.0 for m in internal)
                and len(adj[endpoints[0]]) == 1
                and len(adj[endpoints[1]]) == 1
                and all(len(adj[m]) == 2 for m in internal)
            ):
                path = [endpoints[0]]
                prev = None
                curr = endpoints[0]
                while curr != endpoints[1]:
                    nxt = [n for n in adj[curr] if n != prev]
                    if not nxt:
                        break
                    path.append(nxt[0])
                    prev = curr
                    curr = nxt[0]
                if path[-1] == endpoints[1] and len(path) == len(comp) and len(path) >= 3:
                    wp_indices = path[1:-1]
                    wp_seams.append(
                        WaypointSeam(
                            vert_a=endpoints[0],
                            vert_b=endpoints[1],
                            waypoint_indices=wp_indices,
                            waypoint_coords=pos_ref[wp_indices].astype(np.float32),
                        )
                    )

        self._legacy_wp_seams = wp_seams
        return wp_seams

    def _reset_legacy_waypoint_manager(self, *, fallback_pos: np.ndarray) -> Optional[Any]:
        """旧ログ向けの WaypointManager インスタンスを生成して返す。"""
        wp_seams = self._extract_legacy_waypoint_seams(fallback_pos=fallback_pos)
        if not wp_seams:
            self._legacy_wp_mgr = None
            return None
        try:
            from .engine.waypoint import WaypointManager
        except ImportError:
            try:
                from taremin_cloth.engine.waypoint import WaypointManager  # type: ignore
            except ImportError:
                self._legacy_wp_mgr = None
                return None

        inv0 = np.asarray(self.metadata.get("inv_masses", []), dtype=np.float32)
        thickness = float(self.metadata.get("thickness", 0.005))
        mgr = WaypointManager(
            waypoint_seams=wp_seams,
            initial_inv_masses=inv0,
            merge_dist=max(0.01, thickness * 2.0),
            pass_dist=max(0.015, thickness * 3.0),
        )
        self._legacy_wp_mgr = mgr
        return mgr

    def _init_legacy_waypoint_at(self, frame_idx: int, *, fallback_pos: np.ndarray) -> None:
        """旧ログで開始フレーム以前に解放済みのWaypoint状態をシミュレータへ反映する。"""
        wp_mgr = self._reset_legacy_waypoint_manager(fallback_pos=fallback_pos)
        if wp_mgr is None or not wp_mgr.has_active_seams:
            return
        frames = self._frames_by_idx or {}
        for idx in (self._ordered_indices or []):
            if idx >= frame_idx:
                break
            fdata = frames.get(idx)
            if frame_has_positions(fdata):
                pos = np.asarray(fdata["positions"], dtype=np.float32).reshape((-1, 3))
                wp_mgr.update(pos, sim=self.sim)

    def initialize_at_frame(
        self,
        frame_idx: int,
        *,
        force_enable_pair_cache: bool = False,
        force_pair_cache: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """指定フルフレームの状態からシミュレータを構築・初期化し、(init_pos, frame_data) を返す。"""
        self._ensure_frames_loaded()
        frames = self._frames_by_idx or {}
        if frame_idx not in frames:
            raise ValueError(f"開始フレーム {frame_idx} がログに存在しません")

        start_frame = frames[frame_idx]
        if not frame_has_positions(start_frame):
            near = nearest_full_frame(
                self.log_path, frame_idx, frames=self._ordered_frames
            )
            hint = f" (近傍フル: Frame {near})" if near is not None else ""
            raise ValueError(
                f"開始フレーム {frame_idx} はスタブ (座標なし) のため再現できません{hint}。"
                "フル保存フレームを開始点にするか、full_stride=1で記録してください"
            )

        init_pos = np.array(start_frame["positions"], dtype=np.float32).reshape((-1, 3))
        raw_vel = start_frame.get("velocities")
        init_vel = (
            np.array(raw_vel, dtype=np.float32).reshape((-1, 3))
            if isinstance(raw_vel, list) and len(raw_vel) > 0
            else None
        )

        self.create_simulator(init_pos)
        self.sim.set_positions_and_velocities(init_pos, init_vel)
        # 開始時点までの変更を折り畳んで適用 (途中開始でも正確に復元)
        self.apply_config_at(frame_idx)
        self.apply_elastic_at(frame_idx)
        self.apply_sewing_at(frame_idx)
        if self._has_inv_mass_deltas_field:
            self._legacy_wp_mgr = None
            self.apply_inv_masses_at(frame_idx)
        else:
            self._init_legacy_waypoint_at(frame_idx, fallback_pos=init_pos)

        if force_pair_cache is not None:
            self._pair_cache_override = bool(force_pair_cache)
        elif force_enable_pair_cache:
            self._pair_cache_override = True
        else:
            self._pair_cache_override = None
        self._apply_pair_cache_override()

        try:
            self.apply_frame_bone(start_frame)
        except Exception:
            pass
        return init_pos, start_frame

    def prepare_step(
        self,
        frame_data: Dict[str, Any],
        *,
        prev_frame_data: Optional[Dict[str, Any]] = None,
        prev_positions: Optional[np.ndarray] = None,
    ) -> None:
        """次フレームへの1ステップ実行前に必要な全入力・状態差分をシミュレータへ適用する。"""
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")

        # ステップ前に設定変更・上書き・ソルバ反復回数を反映
        self.apply_frame_config(frame_data)
        self._apply_pair_cache_override()
        try:
            iters = int(frame_data.get("solver_iterations", 0))
            if iters > 0 and hasattr(self.sim, "set_solver_iterations"):
                self.sim.set_solver_iterations(iters)
        except Exception:
            pass

        # 外部入力(ピン・コライダー)・辺スケール・動的inv_massesを反映
        self.apply_frame_inputs(frame_data)
        self.apply_frame_elastic(frame_data)
        if self._has_inv_mass_deltas_field:
            self.apply_frame_inv_masses(frame_data)
        elif self._legacy_wp_mgr is not None and self._legacy_wp_mgr.has_active_seams:
            wp_pos = None
            if frame_has_positions(prev_frame_data):
                wp_pos = np.asarray(
                    prev_frame_data["positions"], dtype=np.float32
                ).reshape((-1, 3))
            elif prev_positions is not None:
                wp_pos = np.asarray(prev_positions, dtype=np.float32).reshape((-1, 3))
            if wp_pos is not None:
                self._legacy_wp_mgr.update(wp_pos, sim=self.sim)

        self.apply_frame_bone(frame_data)
        if prev_frame_data is not None:
            self.apply_frame_sewing(prev_frame_data)

        # 縫合優先モード: 記録状態を復元し、なければ直近座標で測定 (旧ログ近似)
        if not self.apply_frame_priority(frame_data) and prev_positions is not None:
            self._update_sewing_priority(prev_positions)

    def get_active_pairs(self) -> Dict[str, Any]:
        """現在のシミュレータ状態の収集ペアを読戻す (デバッグ専用)。

        `replay_range` 等で目的フレームまで進めた後に呼ぶこと。
        戻り値: {"vt_count", "ee_count", "vt_pairs": (N,4), "ee_pairs": (M,4)}。
        """
        if self.sim is None:
            raise RuntimeError("Simulator is not initialized")
        if not hasattr(self.sim, "get_active_pairs"):
            raise RuntimeError("このバイナリは get_active_pairs に対応していません")
        vt_count, ee_count, vt_pairs, ee_pairs = self.sim.get_active_pairs()
        return {
            "vt_count": int(vt_count),
            "ee_count": int(ee_count),
            "vt_pairs": np.asarray(vt_pairs, dtype=np.uint32).reshape((-1, 4)),
            "ee_pairs": np.asarray(ee_pairs, dtype=np.uint32).reshape((-1, 4)),
        }

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
        *,
        compare: bool = True,
        force_enable_pair_cache: bool = False,
        force_pair_cache: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        指定されたフレーム区間を完全再現し、実測ログと再計算値の差分を検証します。

        force_enable_pair_cache=True の場合、ログの設定によらずペアキャッシュを
        有効化して再現する (ペア監査用)。force_pair_cache=True/False では
        有効/無効を明示指定する (ペアキャッシュ起因判定用。None時は記録通り)。
        ペア内容自体は結果に含まず、呼び出し後に get_active_pairs() で読戻すこと。

        Returns:
            各フレームの再現結果リスト
            [{"frame_index": int, "max_diff": float, "compared": bool,
              "positions": np.ndarray}, ...]
        """
        frames = self.get_frames_dict()
        init_pos, start_frame = self.initialize_at_frame(
            start_frame_idx,
            force_enable_pair_cache=force_enable_pair_cache,
            force_pair_cache=force_pair_cache,
        )

        results = []
        out_pos = np.zeros(self.num_vertices * 3, dtype=np.float32)
        prev_fdata = start_frame
        prev_replay_pos = init_pos.copy()

        for f_idx in range(start_frame_idx + 1, end_frame_idx + 1):
            if f_idx not in frames:
                break
            fdata = frames[f_idx]
            dt = float(fdata.get("dt", 1.0 / 60.0))
            substeps = int(fdata.get("substeps", 10))

            self.prepare_step(
                fdata, prev_frame_data=prev_fdata, prev_positions=prev_replay_pos
            )
            self.sim.step(dt=dt, substeps=substeps)

            self.sim.get_positions(out_pos)
            sim_pos = out_pos.reshape((-1, 3)).copy()

            max_diff = 0.0
            compared = False
            if compare and frame_has_positions(fdata):
                log_pos = np.array(fdata["positions"], dtype=np.float32).reshape((-1, 3))
                max_diff = float(np.max(np.abs(sim_pos - log_pos)))
                compared = True

            results.append({
                "frame_index": f_idx,
                "max_diff": max_diff,
                "compared": compared,
                "positions": sim_pos,
            })
            prev_fdata = fdata
            prev_replay_pos = sim_pos.copy()

        return results

    def trace_substeps(
        self,
        target_frame_idx: int
    ) -> List[Dict[str, Any]]:
        """
        問題フレーム (target_frame_idx) の直前状態から、1サブステップ刻みでステップ実行し、
        サブステップごとの最大変位・最大速度推移を詳細トレースします。
        PairCache有効時はサブステップ毎の vt/ee カウントも取得します。
        """
        prev_idx = target_frame_idx - 1
        frames = self.get_frames_dict()
        f_prev = frames.get(prev_idx)
        f_curr = frames.get(target_frame_idx)
        if f_prev is None or f_curr is None:
            raise ValueError(f"Frame {prev_idx} または {target_frame_idx} がログに存在しません")
        if not frame_has_positions(f_prev) or not frame_has_positions(f_curr):
            near = nearest_full_frame(
                self.log_path, target_frame_idx, frames=self._ordered_frames
            )
            hint = f" (近傍フル: Frame {near})" if near is not None else ""
            raise ValueError(
                f"trace_substepsには前後フレームのフル座標が必要です{hint}。"
                "スタブ区間は full_stride=1 で再記録するかフル区間で実行してください"
            )

        init_pos, _ = self.initialize_at_frame(prev_idx)
        self.prepare_step(f_curr, prev_frame_data=f_prev, prev_positions=init_pos)

        dt = float(f_curr.get("dt", 1.0 / 60.0))
        substeps = int(f_curr.get("substeps", 10))
        # Rust側 `dt_f32 / (substeps as f32)` とf32丸めを揃え、JSONパース由来の1ULP差を防ぐ
        dt_sub = float(np.float32(dt) / np.float32(substeps)) if substeps > 0 else 0.0

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
            entry: Dict[str, Any] = {
                "substep": s + 1,
                "dt_sub": dt_sub,
                "max_disp_mm": max_d,
                "max_disp_vert": max_v,
                "positions": curr_sub_pos,
            }
            # Pair統計 (取得失敗時は欠番、記録を止めない)
            try:
                if hasattr(self.sim, "get_pair_cache_stats"):
                    vt, ee, max_vt, max_ee = self.sim.get_pair_cache_stats()
                    entry["vt_count"] = int(vt)
                    entry["ee_count"] = int(ee)
                    entry["vt_saturated"] = bool(vt >= max_vt or ee >= max_ee)
            except Exception:
                pass

            substep_traces.append(entry)
            prev_sub_pos = curr_sub_pos

        return substep_traces

    def replay_until(
        self,
        start_frame_idx: int,
        end_frame_idx: int,
        *,
        stop_on=None,
        stride: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """条件付きブレークポイント付きリプレイ.

        startからendまで逐次リプレイし、stop_on(ctx)が真(または非空文字)を返した
        最初のフレームで停止する。ctxは下記キーを含む:
          frame_index, pos, prev_pos, max_disp_mm, max_diff,
          vt_count, ee_count, vt_saturated, config_hash, stats
        戻り値: Hit時の {"frame_index": int, "reason": str, "positions": ndarray,
          "max_disp_mm": float, "max_diff": float, ...}、未Hit時はNone.
        """
        frames = self.get_frames_dict()
        init_pos, prev_fdata = self.initialize_at_frame(start_frame_idx)

        out_pos = np.zeros(self.num_vertices * 3, dtype=np.float32)
        prev_pos = init_pos.copy()
        ordered = self.get_ordered_frame_indices()

        for f_idx in ordered:
            if f_idx <= start_frame_idx or f_idx > end_frame_idx:
                continue
            if stride > 1 and ((f_idx - start_frame_idx) % stride != 0) and f_idx != end_frame_idx:
                # stride間引き時は入力適用なしでスキップできないため、
                # 計算自体は進めるが述語評価のみ省略する
                evaluate = False
            else:
                evaluate = True
            fdata = frames[f_idx]
            dt = float(fdata.get("dt", 1.0 / 60.0))
            substeps = int(fdata.get("substeps", 10))
            self.prepare_step(
                fdata, prev_frame_data=prev_fdata, prev_positions=prev_pos
            )
            self.sim.step(dt=dt, substeps=substeps)
            self.sim.get_positions(out_pos)
            sim_pos = out_pos.reshape((-1, 3)).copy()

            max_diff = 0.0
            compared = False
            if frame_has_positions(fdata):
                log_pos = np.array(fdata["positions"], dtype=np.float32).reshape((-1, 3))
                max_diff = float(np.max(np.abs(sim_pos - log_pos)))
                compared = True

            disps = np.linalg.norm(sim_pos - prev_pos, axis=1) * 1000.0
            max_disp_mm = float(np.max(disps)) if len(disps) else 0.0

            vt_count = 0
            ee_count = 0
            vt_saturated = False
            try:
                if hasattr(self.sim, "get_pair_cache_stats"):
                    vt, ee, max_vt, max_ee = self.sim.get_pair_cache_stats()
                    vt_count, ee_count = int(vt), int(ee)
                    vt_saturated = bool(vt >= max_vt or ee >= max_ee)
            except Exception:
                pass
            try:
                config_hash = int(self.sim.get_config_hash()) if hasattr(self.sim, "get_config_hash") else 0
            except Exception:
                config_hash = 0

            ctx: Dict[str, Any] = {
                "frame_index": f_idx,
                "pos": sim_pos,
                "prev_pos": prev_pos.copy(),
                "max_disp_mm": max_disp_mm,
                "max_diff": max_diff,
                "compared": compared,
                "config_changed": bool(isinstance(fdata.get("param_deltas"), dict) and fdata.get("param_deltas")),
                "vt_count": vt_count,
                "ee_count": ee_count,
                "vt_saturated": vt_saturated,
                "config_hash": config_hash,
                "stats": fdata.get("stats", {}),
                "log_frame": fdata,
            }
            prev_pos = sim_pos.copy()
            prev_fdata = fdata

            if evaluate and stop_on is not None:
                try:
                    hit = stop_on(ctx)
                except Exception:
                    hit = False
                if hit:
                    reason = hit if isinstance(hit, str) else "predicate matched"
                    ctx["reason"] = reason
                    ctx["positions"] = sim_pos
                    return ctx
        return None


def build_stop_predicate(
    *,
    disp_mm: Optional[float] = None,
    intersections: bool = False,
    faces=None,
    vt_saturated: bool = False,
    watch_verts: Optional[List[int]] = None,
    watch_vert_disp_mm: float = 2.0,
    max_diff_m: Optional[float] = None,
    config_changed: bool = False,
):
    """log_tools watch用の簡易述語ビルダー。真時は理由文字列を返す。"""
    watch_set = set(int(v) for v in (watch_verts or []))

    def _pred(ctx: Dict[str, Any]):
        if disp_mm is not None and float(ctx.get("max_disp_mm", 0.0)) > float(disp_mm):
            return f"disp {ctx['max_disp_mm']:.2f}mm > {disp_mm}mm @F{ctx['frame_index']}"
        if max_diff_m is not None and bool(ctx.get("compared", True)) and float(ctx.get("max_diff", 0.0)) > float(max_diff_m):
            return f"replay divergence {ctx['max_diff']:.6f}m @F{ctx['frame_index']}"
        if config_changed and bool(ctx.get("config_changed", False)):
            return f"config changed @F{ctx['frame_index']}"
        if vt_saturated and bool(ctx.get("vt_saturated", False)):
            return f"pair saturated vt={ctx.get('vt_count')} ee={ctx.get('ee_count')} @F{ctx['frame_index']}"
        # ログ側Statsの飽和・NaNも検出 (リプレイ前でも分かる異常)
        stats = ctx.get("stats", {}) or {}
        if vt_saturated and bool(stats.get("vt_saturated", False)):
            return f"log vt_saturated @F{ctx['frame_index']}"
        if stats.get("has_nan_or_inf", False):
            return f"log NaN/Inf @F{ctx['frame_index']}"
        if watch_set:
            try:
                pos = ctx["pos"]
                prev = ctx["prev_pos"]
                for vid in watch_set:
                    if 0 <= vid < len(pos):
                        d = float(np.linalg.norm(pos[vid] - prev[vid]) * 1000.0)
                        if d > float(watch_vert_disp_mm):
                            return f"vert {vid} disp {d:.2f}mm @F{ctx['frame_index']}"
            except Exception:
                pass
        if intersections:
            try:
                if faces is not None and len(faces) > 0:
                    try:
                        from . import analysis as _ana
                    except ImportError:
                        from taremin_cloth import analysis as _ana  # type: ignore
                    pairs = _ana.find_triangle_intersections(
                        np.asarray(ctx.get("pos")), np.asarray(faces)
                    )
                    if len(pairs) > 0:
                        return f"intersections={len(pairs)} @F{ctx['frame_index']}"
            except Exception:
                pass
        return False

    return _pred


def build_release_persist_predicate(
    base_pred=None,
    *,
    persist: int = 1,
    require_release: bool = False,
):
    """リリースゲート＋持続条件付きの状態付き述語を構築する。

    - require_release=True: ピン数減少 (grab解放) を検出して初めてアームする。
      それ以前の base_pred ヒット (grab中の瞬間貫通など) は無視される。
    - persist=N: アーム後に base_pred がNフレーム連続で真のとき発火する。
    - 戻り述語は `.state` 属性 (armed/prev_pins/run/release_frame) を持つ。
    """
    persist = max(1, int(persist))
    state: Dict[str, Any] = {
        "armed": not require_release,
        "prev_pins": None,
        "run": 0,
        "release_frame": None,
    }

    def _pred(ctx: Dict[str, Any]):
        try:
            pins = len((ctx.get("log_frame") or {}).get("pins", []))
        except Exception:
            pins = 0
        prev = state["prev_pins"]
        if prev is not None and pins < prev:
            state["armed"] = True
            state["run"] = 0
            state["release_frame"] = ctx.get("frame_index")
        state["prev_pins"] = pins

        if not state["armed"]:
            return False
        try:
            hit = base_pred(ctx) if base_pred is not None else True
        except Exception:
            hit = False
        if hit:
            state["run"] += 1
            if state["run"] >= persist:
                reason = hit if isinstance(hit, str) else "predicate matched"
                if state["release_frame"] is not None:
                    return f"{reason} (released at F{state['release_frame']})"
                return reason
            return False
        state["run"] = 0
        return False

    _pred.state = state  # type: ignore[attr-defined]
    return _pred
