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
) -> List[np.ndarray]:
    """
    シワカーブ群を、ボーンチェーン上の source_t から target_t へ連続スライド変形。
    
    1. 各制御点の source 断面ローカル座標 (θ, r, Δz) を抽出
    2. target 断面の (position, axis, normal, binormal, radius) へ再配置
    """
    src_eval = chain.evaluate(source_t)
    tgt_eval = chain.evaluate(target_t)

    slid_curves = []
    rad_ratio = (tgt_eval.radius / max(src_eval.radius, 1e-4)) * scale_radius

    for pts in curve_points_list:
        if len(pts) == 0:
            slid_curves.append(pts.copy())
            continue

        new_pts = []
        for p in pts:
            # 1. source_t からの相対オフセット
            diff = p - src_eval.position
            dz = float(np.dot(diff, src_eval.axis))
            r_vec = diff - src_eval.axis * dz

            x_comp = float(np.dot(r_vec, src_eval.normal))
            y_comp = float(np.dot(r_vec, src_eval.binormal))

            # 2. target_t への再投影
            # 半径拡縮比率 rad_ratio を断面座標に乗算
            new_r_vec = (tgt_eval.normal * x_comp + tgt_eval.binormal * y_comp) * rad_ratio
            new_p = tgt_eval.position + tgt_eval.axis * dz + new_r_vec
            new_pts.append(new_p)

        slid_curves.append(np.array(new_pts, dtype=np.float32))

    return slid_curves
