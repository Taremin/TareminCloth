"""
中継点縫合（Waypoint Seam）ライフサイクル管理モジュール (taremin_cloth.engine.waypoint)

縫合進行中のコライダー迂回ガイド機能から、中継点通過時の順次ターゲットシフト、
および両端点合流時の固定ピン解除（自由ドレープ化）を制御する。
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from ..utils.logger import logger
from ..utils.mesh_extract import WaypointSeam


@dataclass
class WaypointSeamState:
    """単一の中継点縫合パスの進行状態"""
    seam: WaypointSeam
    target_inv_mass: float
    released_indices: set[int] = field(default_factory=set)
    curr_a_ptr: int = 0
    curr_b_ptr: int = 0
    is_fully_closed: bool = False

    def __post_init__(self):
        self.curr_b_ptr = len(self.seam.waypoint_indices) - 1


class WaypointManager:
    """中継点縫合の動的通過・合流・ピン解放を統括するコントローラー"""

    def __init__(
        self,
        waypoint_seams: Optional[list[WaypointSeam]],
        initial_inv_masses: np.ndarray,
        merge_dist: float = 0.015,
        pass_dist: float = 0.02,
    ):
        """
        Args:
            waypoint_seams: 中継点縫合のリスト
            initial_inv_masses: 初期逆質量配列 (len = N)
            merge_dist: 最終合流判定距離 (m)。これ以下で最後の中継点ピンを解除
            pass_dist: 中継点通過判定距離 (m)。これ以下で中間中継点ピンを順次解除
        """
        self.merge_dist = float(merge_dist)
        self.pass_dist = float(pass_dist)
        self.states: list[WaypointSeamState] = []
        self._all_released: set[int] = set()

        if not waypoint_seams:
            return

        inv_m = np.asarray(initial_inv_masses, dtype=np.float32)

        for ws in waypoint_seams:
            # 端点AとBの基準逆質量を取得
            ma = float(inv_m[ws.vert_a]) if ws.vert_a < len(inv_m) else 1.0
            mb = float(inv_m[ws.vert_b]) if ws.vert_b < len(inv_m) else 1.0

            # どちらかが固定ピン（0.0）の場合、非固定側の質量を採用
            if ma > 0.0 and mb > 0.0:
                target_m = (ma + mb) * 0.5
            elif ma > 0.0:
                target_m = ma
            elif mb > 0.0:
                target_m = mb
            else:
                target_m = 1.0

            self.states.append(WaypointSeamState(seam=ws, target_inv_mass=target_m))

    @property
    def has_active_seams(self) -> bool:
        """未完了の中継点縫合が存在するか"""
        return any(not s.is_fully_closed for s in self.states)

    @property
    def released_indices(self) -> set[int]:
        """現在までにピン解除された全中継点頂点インデックスの集合"""
        return set(self._all_released)

    def update(self, coords: np.ndarray, sim=None) -> list[int]:
        """
        現在の頂点座標から通過・合流状態を判定し、ピン解除すべき中継点頂点を更新する。

        Args:
            coords: shape [N, 3] または [N * 3] の頂点座標
            sim: ClothSimulator インスタンス (渡された場合は set_vertex_inv_masses を即時呼び出し)

        Returns:
            list[int]: 今回新しくピン解放された中継点頂点インデックスのリスト
        """
        if not self.states:
            return []

        pos = coords.reshape(-1, 3)
        newly_released = []
        release_masses = []

        for s in self.states:
            if s.is_fully_closed:
                continue

            ws = s.seam
            va = ws.vert_a
            vb = ws.vert_b
            wp_list = ws.waypoint_indices
            k = len(wp_list)

            pa = pos[va]
            pb = pos[vb]

            # ケース 1: 単一中継点 (k == 1)
            if k == 1:
                wp_idx = wp_list[0]
                dist_ab = float(np.linalg.norm(pa - pb))
                p_wp = pos[wp_idx]
                dist_a_wp = float(np.linalg.norm(pa - p_wp))
                dist_b_wp = float(np.linalg.norm(pb - p_wp))

                # AとBの直接距離が merge_dist 以下、または両方が中継点に近接
                if dist_ab <= self.merge_dist or (dist_a_wp <= self.merge_dist and dist_b_wp <= self.merge_dist):
                    if wp_idx not in s.released_indices:
                        s.released_indices.add(wp_idx)
                        self._all_released.add(wp_idx)
                        newly_released.append(wp_idx)
                        release_masses.append(s.target_inv_mass)
                    s.is_fully_closed = True
                    logger.info(
                        f"[Waypoint] Seam ({va}, {vb}) merged (dist_ab={dist_ab * 1000:.1f}mm). "
                        f"Released waypoint pin v{wp_idx} (inv_mass={s.target_inv_mass:.2f})."
                    )

            # ケース 2: 複数中継点 (k > 1)
            else:
                # まだA側とB側の中継点が交差していない場合
                while s.curr_a_ptr < s.curr_b_ptr:
                    wp_a = wp_list[s.curr_a_ptr]
                    dist_a = float(np.linalg.norm(pa - pos[wp_a]))
                    if dist_a <= self.pass_dist:
                        if wp_a not in s.released_indices:
                            s.released_indices.add(wp_a)
                            self._all_released.add(wp_a)
                            newly_released.append(wp_a)
                            release_masses.append(s.target_inv_mass)
                            logger.info(
                                f"[Waypoint] Seam ({va}, {vb}) vertex A reached waypoint v{wp_a} "
                                f"(dist={dist_a * 1000:.1f}mm). Shifted target to next waypoint."
                            )
                        s.curr_a_ptr += 1
                    else:
                        break

                while s.curr_b_ptr > s.curr_a_ptr:
                    wp_b = wp_list[s.curr_b_ptr]
                    dist_b = float(np.linalg.norm(pb - pos[wp_b]))
                    if dist_b <= self.pass_dist:
                        if wp_b not in s.released_indices:
                            s.released_indices.add(wp_b)
                            self._all_released.add(wp_b)
                            newly_released.append(wp_b)
                            release_masses.append(s.target_inv_mass)
                            logger.info(
                                f"[Waypoint] Seam ({va}, {vb}) vertex B reached waypoint v{wp_b} "
                                f"(dist={dist_b * 1000:.1f}mm). Shifted target to next waypoint."
                            )
                        s.curr_b_ptr -= 1
                    else:
                        break

                # A側とB側が同一の最後の中継点に到達した場合
                if s.curr_a_ptr == s.curr_b_ptr:
                    wp_mid = wp_list[s.curr_a_ptr]
                    dist_ab = float(np.linalg.norm(pa - pb))
                    p_mid = pos[wp_mid]
                    dist_a_mid = float(np.linalg.norm(pa - p_mid))
                    dist_b_mid = float(np.linalg.norm(pb - p_mid))

                    if dist_ab <= self.merge_dist or (dist_a_mid <= self.merge_dist and dist_b_mid <= self.merge_dist):
                        if wp_mid not in s.released_indices:
                            s.released_indices.add(wp_mid)
                            self._all_released.add(wp_mid)
                            newly_released.append(wp_mid)
                            release_masses.append(s.target_inv_mass)
                        s.is_fully_closed = True
                        logger.info(
                            f"[Waypoint] Seam ({va}, {vb}) fully merged at waypoint v{wp_mid} "
                            f"(dist_ab={dist_ab * 1000:.1f}mm). All waypoints released."
                        )

        # GPU側への inv_mass 反映
        if newly_released and sim is not None:
            setter = getattr(sim, "set_vertex_inv_masses", None)
            if callable(setter):
                idx_arr = np.array(newly_released, dtype=np.uint32)
                mass_arr = np.array(release_masses, dtype=np.float32)
                setter(idx_arr, mass_arr)

        return newly_released
