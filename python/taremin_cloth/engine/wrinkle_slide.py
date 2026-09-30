"""
ボーンチェーン補間およびシワカーブスライド幾何計算モジュール (wrinkle_slide.py)

親Tail〜子Headの非連続区間（空欄）を仮想リンクとして自動補間し、
ボーンチェーン全体を1本の滑らかな連続ポリラインとしてパラメータ化します。
3Dビューポート上での直感的なスライド操作、半径拡縮、回転追従を支える幾何エンジンです。
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple
import numpy as np

from .wrinkle_field import build_orthonormal_basis


@dataclass
class BoneSegment:
    """ボーンまたは仮想リンクの単一区間"""
    head: np.ndarray          # 始点 (3,)
    tail: np.ndarray          # 終点 (3,)
    radius_head: float        # 始点半径 (m)
    radius_tail: float        # 終点半径 (m)
    length: float             # 区間長 (m)
    cum_start: float          # チェーン全体における開始累積距離 (m)
    cum_end: float            # チェーン全体における終了累積距離 (m)
    is_virtual: bool = False  # 非連続ボーン間の空欄補間リンクか
    name: str = ""            # ボーン名


@dataclass
class ChainEvalResult:
    """ボーンチェーン上の任意位置における評価結果"""
    position: np.ndarray      # 3D中心座標 (3,)
    axis: np.ndarray          # 単位接線ベクトル (進行方向) (3,)
    normal: np.ndarray        # 単位法線ベクトル (3,)
    binormal: np.ndarray      # 単位従法線ベクトル (3,)
    radius: float             # 補間された基準半径 (m)
    segment_idx: int          # 所属セグメント番号
    is_virtual: bool          # 仮想リンク上か


class BoneChain:
    """
    連続・非連続のボーン群を1つの滑らかな経路として統合管理するクラス
    """
    def __init__(self, segments: List[BoneSegment], total_length: float):
        self.segments = segments
        self.total_length = max(total_length, 1e-6)

    @classmethod
    def from_bones(
        cls,
        bones_data: Sequence[Tuple[str, Sequence[float], Sequence[float], float]],
        gap_threshold: float = 0.001,
    ) -> "BoneChain":
        """
        ボーン定義リスト [(name, head, tail, radius), ...] からボーンチェーンを構築。
        親Tailと子Headが gap_threshold (既定1mm) 以上離れている場合は仮想リンクを自動補間。
        """
        segments: List[BoneSegment] = []
        cum_dist = 0.0

        for i, (b_name, b_head, b_tail, b_rad) in enumerate(bones_data):
            h = np.array(b_head, dtype=np.float32)
            t = np.array(b_tail, dtype=np.float32)
            seg_len = float(np.linalg.norm(t - h))
            if seg_len < 1e-6:
                continue

            # 半径の取得 (単一値または (head_radius, tail_radius))
            if isinstance(b_rad, (tuple, list)):
                r_head = float(b_rad[0])
                r_tail = float(b_rad[1])
            else:
                r_head = float(b_rad)
                r_tail = float(b_rad)

            # 直前ボーンのTailとの隙間（空欄）を検査し、必要なら仮想リンクを生成
            if segments:
                prev_tail = segments[-1].tail
                gap_vec = h - prev_tail
                gap_len = float(np.linalg.norm(gap_vec))
                if gap_len > gap_threshold:
                    v_link = BoneSegment(
                        head=prev_tail,
                        tail=h,
                        radius_head=segments[-1].radius_tail,
                        radius_tail=r_head,
                        length=gap_len,
                        cum_start=cum_dist,
                        cum_end=cum_dist + gap_len,
                        is_virtual=True,
                        name=f"Link_{segments[-1].name}_to_{b_name}",
                    )
                    segments.append(v_link)
                    cum_dist += gap_len

            # 実ボーンセグメントの追加
            b_seg = BoneSegment(
                head=h,
                tail=t,
                radius_head=r_head,
                radius_tail=r_tail,
                length=seg_len,
                cum_start=cum_dist,
                cum_end=cum_dist + seg_len,
                is_virtual=False,
                name=b_name,
            )
            segments.append(b_seg)
            cum_dist += seg_len

        return cls(segments, cum_dist)

    def transported_segment_frames(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """各セグメントの回転最小輸送フレーム (axis, normal, binormal) を返す。

        `evaluate()` の区間毎任意基底とは異なり、関節を跨いでもねじれが
        最小になるよう前区間の法線を次区間平面へ射影伝搬する。スライド時の
        面追従再構成は本フレーム系で統一するため、屈曲関節跨ぎでもカーブが
        関節内側へ slingshot せず表面に沿って移動する。
        """
        frames: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        n_ref: Optional[np.ndarray] = None
        for seg in self.segments:
            d = seg.tail - seg.head
            ln = float(np.linalg.norm(d))
            if ln < 1e-9:
                axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                axis = d / ln
            if n_ref is None:
                _, n0, _ = build_orthonormal_basis(axis)
                n_ref = n0.astype(np.float64)
            # 前区間の法線を現区間平面へ射影（最小回転伝搬）
            n_proj = n_ref - float(np.dot(n_ref, axis)) * axis
            nl = float(np.linalg.norm(n_proj))
            if nl < 1e-6:
                # 180度反転の縮退時のみ任意直交を再取得
                _, n_proj, _ = build_orthonormal_basis(axis)
                nl = float(np.linalg.norm(n_proj))
                n_proj = n_proj / max(nl, 1e-9)
            else:
                n_proj = n_proj / nl
            bn = np.cross(axis, n_proj)
            bl = float(np.linalg.norm(bn))
            if bl < 1e-9:
                _, _, bn = build_orthonormal_basis(axis)
            else:
                bn = bn / bl
            frames.append((axis.astype(np.float64), n_proj.astype(np.float64),
                           bn.astype(np.float64)))
            n_ref = n_proj
        return frames

    def station_point(self, s: float) -> Tuple[np.ndarray, float, int]:
        """弧長 s における中心座標・半径・所属セグメント番号を返す。"""
        if not self.segments:
            return np.zeros(3, dtype=np.float64), 0.05, -1
        sc = max(0.0, min(float(self.total_length), float(s)))
        seg_idx = len(self.segments) - 1
        for idx, seg in enumerate(self.segments):
            if sc <= seg.cum_end or idx == len(self.segments) - 1:
                seg_idx = idx
                break
        seg = self.segments[seg_idx]
        u = 0.0
        if seg.length > 1e-9:
            u = max(0.0, min(1.0, (sc - seg.cum_start) / seg.length))
        pos = seg.head.astype(np.float64) + (seg.tail - seg.head).astype(np.float64) * u
        rad = float(seg.radius_head + (seg.radius_tail - seg.radius_head) * u)
        return pos, rad, seg_idx

    def blended_frame_at(
        self,
        s: float,
        frames: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """弧長 s における連続ブレンドフレーム (axis, normal, binormal) を返す。

        関節から ε 以内では前後区間の輸送フレームを nlerp 補間し、離散切替に
        よるスナップ・せん断を除去する。区間中央では輸送フレームと一致する
        ため直進チェーンの既存動作は不変。
        """
        if not self.segments:
            return (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]))
        sc = max(0.0, min(float(self.total_length), float(s)))
        if frames is None:
            frames = self.transported_segment_frames()
        n_seg = len(self.segments)
        seg_idx = n_seg - 1
        for idx, seg in enumerate(self.segments):
            if sc <= seg.cum_end or idx == n_seg - 1:
                seg_idx = idx
                break
        # 高速パス: 関節近傍でなければ輸送フレームをそのまま返す (numpy演算なし)
        near_joint = False
        for j in range(n_seg - 1):
            l0 = self.segments[j].length
            l1 = self.segments[j + 1].length
            if l0 < 1e-9 or l1 < 1e-9:
                continue
            eps = min(0.02, 0.15 * min(l0, l1))
            if eps > 1e-9 and abs(sc - float(self.segments[j].cum_end)) < eps:
                near_joint = True
                break
        if not near_joint:
            return frames[seg_idx]
        seg = self.segments[seg_idx]
        axis, normal, _ = frames[seg_idx]
        if seg.length < 1e-9:
            bn = np.cross(axis, normal)
            return axis, normal, bn / max(float(np.linalg.norm(bn)), 1e-9)

        def _nlerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
            v = a * (1.0 - t) + b * t
            n = float(np.linalg.norm(v))
            if n < 1e-9:
                return a
            return v / n

        def _finish(ax: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            n = n - float(np.dot(n, ax)) * ax
            nl = float(np.linalg.norm(n))
            if nl < 1e-6:
                _, n, _ = build_orthonormal_basis(ax)
            else:
                n = n / nl
            bn = np.cross(ax, n)
            bl = float(np.linalg.norm(bn))
            return ax, n, bn / max(bl, 1e-9)

        # 内部関節の符号付き距離による対称ブレンド（関節点上でも連続）
        for j in range(len(self.segments) - 1):
            joint = float(self.segments[j].cum_end)
            l0 = self.segments[j].length
            l1 = self.segments[j + 1].length
            if l0 < 1e-9 or l1 < 1e-9:
                continue
            eps = min(0.02, 0.15 * min(l0, l1))
            if eps > 1e-9 and abs(sc - joint) < eps:
                t = (sc - (joint - eps)) / (2.0 * eps)  # 0:手前区間 〜 1:先区間
                ax0, n0, _ = frames[j]
                ax1, n1, _ = frames[j + 1]
                return _finish(_nlerp(ax0, ax1, t), _nlerp(n0, n1, t))
        bn = np.cross(axis, normal)
        bl = float(np.linalg.norm(bn))
        return axis, normal, bn / max(bl, 1e-9)

    def project_to_station(self, p: np.ndarray) -> Tuple[float, int]:
        """点に最も近い中心線上の弧長 s と所属セグメント番号を返す。"""
        pf = np.asarray(p, dtype=np.float64).reshape(3)
        best_s = 0.0
        best_idx = 0
        best_d2 = float("inf")
        for idx, seg in enumerate(self.segments):
            h = seg.head.astype(np.float64)
            v = seg.tail.astype(np.float64) - h
            ll = float(np.dot(v, v))
            if ll < 1e-18:
                u = 0.0
            else:
                u = float(np.dot(pf - h, v) / ll)
            uc = max(0.0, min(1.0, u))
            q = h + v * uc
            d2 = float(np.sum((pf - q) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best_idx = idx
                best_s = float(seg.cum_start + uc * seg.length)
        return best_s, best_idx

    def evaluate(self, t_param: float) -> ChainEvalResult:
        """
        正規化パラメータ t_param in [0.0, 1.0] における位置・姿勢・半径を連続評価。
        """
        if not self.segments:
            zero = np.zeros(3, dtype=np.float32)
            return ChainEvalResult(zero, np.array([0, 0, 1], dtype=np.float32),
                                   np.array([1, 0, 0], dtype=np.float32),
                                   np.array([0, 1, 0], dtype=np.float32),
                                   0.05, -1, False)

        t_clamped = max(0.0, min(1.0, float(t_param)))
        target_dist = t_clamped * self.total_length

        # 該当セグメントを探索
        target_seg = self.segments[-1]
        seg_idx = len(self.segments) - 1
        for idx, seg in enumerate(self.segments):
            if target_dist <= seg.cum_end or idx == len(self.segments) - 1:
                target_seg = seg
                seg_idx = idx
                break

        # セグメント内の局所割合 u in [0, 1]
        seg_dist = target_dist - target_seg.cum_start
        u = max(0.0, min(1.0, seg_dist / max(target_seg.length, 1e-6)))

        pos = target_seg.head + (target_seg.tail - target_seg.head) * u
        rad = target_seg.radius_head + (target_seg.radius_tail - target_seg.radius_head) * u

        axis = target_seg.tail - target_seg.head
        axis_norm, norm, binorm = build_orthonormal_basis(axis)

        # 関節境界での接線急変を緩和するため、前後のセグメントと滑らかにブレンド
        # (セグメント端の 15% 区間で球面線形補間に近い接線ブレンド)
        blend_margin = 0.15
        if u < blend_margin and seg_idx > 0:
            prev_axis = self.segments[seg_idx - 1].tail - self.segments[seg_idx - 1].head
            prev_norm = prev_axis / (np.linalg.norm(prev_axis) + 1e-8)
            blend_w = 0.5 * (1.0 - u / blend_margin)
            axis_blended = axis_norm * (1.0 - blend_w) + prev_norm * blend_w
            axis_norm, norm, binorm = build_orthonormal_basis(axis_blended)
        elif u > (1.0 - blend_margin) and seg_idx < len(self.segments) - 1:
            next_axis = self.segments[seg_idx + 1].tail - self.segments[seg_idx + 1].head
            next_norm = next_axis / (np.linalg.norm(next_axis) + 1e-8)
            blend_w = 0.5 * ((u - (1.0 - blend_margin)) / blend_margin)
            axis_blended = axis_norm * (1.0 - blend_w) + next_norm * blend_w
            axis_norm, norm, binorm = build_orthonormal_basis(axis_blended)

        return ChainEvalResult(
            position=pos,
            axis=axis_norm,
            normal=norm,
            binormal=binorm,
            radius=rad,
            segment_idx=seg_idx,
            is_virtual=target_seg.is_virtual,
        )


def slide_curves_along_chain(
    curve_points_list: Sequence[np.ndarray],
    chain: BoneChain,
    source_t: float,
    target_t: float,
    scale_radius: float = 1.0,
    base_scale_radius: float = 1.0,
) -> List[np.ndarray]:
    """
    シワカーブ群を、ボーンチェーンに沿って弧長Δsだけ面追従スライド変形。

    - Δs = (target_t - source_t) * total_length。各制御点は自身の最寄り
      弧長 s_p から s_p + Δs へ移動する（共有source断面への依存なし）。
    - 角度θ・動径高さhは回転最小輸送フレームで保持し、テーパー追従は
      各点の実位置基準 (R(s_new)/R(s_p)) で行う。屈曲関節跨ぎでも
      slingshot・形状崩壊を起こさない。
    - scale_radius は base_scale_radius からの相対倍率として扱う。
      モーダル開始時点の scale を base に渡すことで、初動での二重拡縮
      （配置時scaleの再乗算）を防ぐ。既定 base=1.0 は従来動作と同一。
    - 直進チェーンでは従来動作（断面再配置＋拡縮）と一致する。
    """
    slid_curves = []
    if not chain.segments or chain.total_length < 1e-9:
        for pts in curve_points_list:
            slid_curves.append(np.array(pts, dtype=np.float32).copy())
        return slid_curves

    delta_s = (float(target_t) - float(source_t)) * float(chain.total_length)
    user_scale = float(scale_radius) / max(float(base_scale_radius), 1e-4)
    frames = chain.transported_segment_frames()
    total_len = float(chain.total_length)
    # セグメント配列の事前展開（点ループ内の astype 排除）
    seg_heads = np.stack([s.head.astype(np.float64) for s in chain.segments])
    seg_vecs = np.stack([s.tail.astype(np.float64) - s.head.astype(np.float64)
                         for s in chain.segments])
    seg_ll = np.sum(seg_vecs * seg_vecs, axis=1)
    seg_ll = np.where(seg_ll < 1e-18, 1.0, seg_ll)
    seg_cum = np.array([s.cum_start for s in chain.segments], dtype=np.float64)
    seg_len = np.array([s.length for s in chain.segments], dtype=np.float64)
    seg_rh = np.array([s.radius_head for s in chain.segments], dtype=np.float64)
    seg_rt = np.array([s.radius_tail for s in chain.segments], dtype=np.float64)
    n_seg = len(chain.segments)

    for pts in curve_points_list:
        arr = np.asarray(pts, dtype=np.float64)
        if len(arr) == 0:
            slid_curves.append(np.array(pts, dtype=np.float32).copy())
            continue
        n_pts = len(arr)
        # 1. 点毎の断面候補（全セグメントへの射影をベクトル化）
        u_all = np.sum((arr[:, None, :] - seg_heads[None, :, :]) * seg_vecs[None, :, :],
                       axis=2) / seg_ll[None, :]
        uc_all = np.clip(u_all, 0.0, 1.0)
        q_all = seg_heads[None, :, :] + seg_vecs[None, :, :] * uc_all[:, :, None]
        s_all = seg_cum[None, :] + uc_all * seg_len[None, :]
        d_all = np.linalg.norm(arr[:, None, :] - q_all, axis=2)
        # 2. 曲線順序に沿った連続割当て（関節部での破砕防止）
        #    コスト = |s - s_prev| + W*dist。重心駅をアンカーとする。
        centroid = np.mean(arr, axis=0)
        s_anchor, _ = chain.project_to_station(centroid)
        assign = [0] * n_pts
        s_prev = float(s_anchor)
        w_dist = 2.0
        for k in range(n_pts):
            best_j = 0
            best_c = float("inf")
            sk = s_all[k]
            dk = d_all[k]
            for j in range(n_seg):
                c = abs(float(sk[j]) - s_prev) + w_dist * float(dk[j])
                if c < best_c:
                    best_c = c
                    best_j = j
            assign[k] = best_j
            s_prev = float(sk[best_j])
        # 3. 分解→輸送→再構成（分解・再構成とも連続ブレンドフレームで統一）
        new_pts = np.empty_like(arr)
        for k in range(n_pts):
            j = assign[k]
            s_p = float(s_all[k, j])
            axis_p, n_p, b_p = chain.blended_frame_at(s_p, frames)
            q = q_all[k, j]
            uc = float(uc_all[k, j])
            r_p = float(seg_rh[j] + (seg_rt[j] - seg_rh[j]) * uc)
            p = arr[k]
            rel = p - q
            axial = float(np.dot(rel, axis_p))
            r_vec = rel - axis_p * axial
            h = float(np.linalg.norm(r_vec))
            if h < 1e-9:
                theta = 0.0
            else:
                theta = math.atan2(float(np.dot(r_vec, b_p)), float(np.dot(r_vec, n_p)))
            s_new = max(0.0, min(total_len, s_p + axial + delta_s))
            # station_point と等価のインライン評価（astype排除）
            jn = n_seg - 1
            for jj in range(n_seg):
                if s_new <= float(seg_cum[jj] + seg_len[jj]) or jj == n_seg - 1:
                    jn = jj
                    break
            denom = seg_len[jn]
            un = 0.0 if denom < 1e-9 else max(0.0, min(1.0, (s_new - float(seg_cum[jn])) / denom))
            q_new = seg_heads[jn] + seg_vecs[jn] * un
            r_new = float(seg_rh[jn] + (seg_rt[jn] - seg_rh[jn]) * un)
            axis_new, n_new, b_new = chain.blended_frame_at(s_new, frames)
            kk = (r_new / max(r_p, 1e-4)) * user_scale
            new_pts[k] = (q_new
                          + (math.cos(theta) * n_new + math.sin(theta) * b_new) * (h * kk))
        slid_curves.append(np.asarray(new_pts, dtype=np.float32))

    return slid_curves
