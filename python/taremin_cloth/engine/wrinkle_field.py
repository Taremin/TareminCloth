"""
ドレープガイド（Wrinkle Field / Drape Guide）の幾何プロファイル抽出・管理モジュール

BlenderのCurveオブジェクト（山カーブ Crest / 谷カーブ Root）から、
ボーン円柱座標系における1D角度プロファイルをサンプリング・正規化し、
GPU XPBDシワ拘束カーネルへ渡すバイナリ構造体を生成します。
"""

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    import bpy
    _HAS_BPY = True
except ImportError:
    _HAS_BPY = False


@dataclass
class WrinkleProfile:
    """旧1D角度プロファイルデータ (現行2D-SDF方式では未使用・後方互換保持)。

    注意: GPU `GpuWrinkleProfileSample` に対応する旧1Dパスであり、
    ランタイム (`params.py` / PyO3 / WGSL) からは参照されていない。
    `interpolate_curve_to_profile` / `build_wrinkle_profile_from_curves` と共に
    テスト (`test_wrinkle_profile.py`) の回帰検出用に保持する。新規コードからは使用しないこと。
    """
    valley_z: np.ndarray                      # 谷の軸方向ターゲット位置 (N, float32)
    crest_radius: np.ndarray                  # 山の径方向ターゲット半径 (N, float32)
    valley_weight: np.ndarray                 # 谷の引き寄せ強度 (0.0〜1.0) (N, float32)
    crest_weight: np.ndarray                  # 山の押し出し強度 (0.0〜1.0) (N, float32)
    valley_radius: Optional[np.ndarray] = None # 谷の径方向ターゲット半径 (N, float32)
    crest_z: Optional[np.ndarray] = None       # 山の軸方向ターゲット位置 (N, float32)

    def __post_init__(self):
        self.valley_z = np.ascontiguousarray(self.valley_z, dtype=np.float32)
        self.crest_radius = np.ascontiguousarray(self.crest_radius, dtype=np.float32)
        self.valley_weight = np.ascontiguousarray(self.valley_weight, dtype=np.float32)
        self.crest_weight = np.ascontiguousarray(self.crest_weight, dtype=np.float32)
        n = len(self.valley_z)
        if not (len(self.crest_radius) == n and len(self.valley_weight) == n and len(self.crest_weight) == n):
            raise ValueError("All profile arrays must have the same length")

        if self.valley_radius is None:
            self.valley_radius = np.zeros(n, dtype=np.float32)
        else:
            self.valley_radius = np.ascontiguousarray(self.valley_radius, dtype=np.float32)
            if len(self.valley_radius) != n:
                raise ValueError("valley_radius must have the same length as valley_z")

        if self.crest_z is None:
            self.crest_z = np.copy(self.valley_z)
        else:
            self.crest_z = np.ascontiguousarray(self.crest_z, dtype=np.float32)
            if len(self.crest_z) != n:
                raise ValueError("crest_z must have the same length as valley_z")

    @property
    def num_samples(self) -> int:
        return len(self.valley_z)

    def to_numpy(self) -> np.ndarray:
        """[N, 8] (valley_z, valley_radius, crest_z, crest_radius, valley_weight, crest_weight, pad0, pad1) の NumPy 配列を生成します。"""
        n = len(self.valley_z)
        pads = np.zeros(n, dtype=np.float32)
        return np.stack(
            [
                self.valley_z,
                self.valley_radius,
                self.crest_z,
                self.crest_radius,
                self.valley_weight,
                self.crest_weight,
                pads,
                pads,
            ],
            axis=-1
        ).astype(np.float32)

    def to_gpu_bytes(self) -> bytes:
        """
        GPUの struct GpuWrinkleProfileSample {
            float valley_z; float valley_radius; float crest_z; float crest_radius;
            float valley_weight; float crest_weight; float _pad0; float _pad1;
        }
        配列（1サンプルあたり32バイト）と完全に一致するバイナリを生成します。
        """
        return self.to_numpy().tobytes()


@dataclass
class WrinkleFieldParams:
    """旧1Dメタパラメータ (現行2D-SDF方式では未使用・後方互換保持)。

    注意: ランタイム (`params.py` は `sim.set_wrinkle_field_params` を直接呼ぶ) からは
    参照されていない。新規コードからは使用しないこと。
    """
    bone_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bone_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)      # ボーン軸方向 (単位ベクトル)
    bone_normal: Tuple[float, float, float] = (1.0, 0.0, 0.0)    # 断面基準軸X (単位ベクトル)
    influence_radius: float = 0.05                               # 軸方向フォールオフ半径 (m)
    bone_radius: float = 0.05                                    # 基準円柱半径 (m)
    stiffness: float = 1.0                                       # シワ拘束剛性 (0.0〜1.0)
    num_samples: int = 64                                        # プロファイル角度解像度


def build_orthonormal_basis(axis: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """軸ベクトルから正規直交基底 (axis, normal, binormal) を構築します。"""
    a = np.array(axis, dtype=np.float32)
    len_a = np.linalg.norm(a)
    if len_a < 1e-7:
        a = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        a = a / len_a

    # 軸と平行でないベクトルを探す
    if abs(a[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    n = np.cross(a, ref)
    n = n / np.linalg.norm(n)
    bn = np.cross(a, n)
    bn = bn / np.linalg.norm(bn)
    return a, n, bn


def project_points_to_cylindrical(
    points: np.ndarray,
    origin: Sequence[float],
    axis: Sequence[float],
    normal: Optional[Sequence[float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    3D空間の点群をボーン円柱座標系 (theta, z, r) に射影します。
    theta: [0, 2*pi)
    z: ボーン軸に沿った符号付き距離 (m)
    r: ボーン中心軸からの最短距離 (m)
    """
    pts = np.ascontiguousarray(points, dtype=np.float32)
    orig = np.array(origin, dtype=np.float32)
    a, n, bn = build_orthonormal_basis(axis)
    if normal is not None:
        n_custom = np.array(normal, dtype=np.float32)
        n_proj = n_custom - np.dot(n_custom, a) * a
        if np.linalg.norm(n_proj) > 1e-6:
            n = n_proj / np.linalg.norm(n_proj)
            bn = np.cross(a, n)
            bn = bn / np.linalg.norm(bn)

    diff = pts - orig
    z = np.dot(diff, a)
    r_vec = diff - np.outer(z, a)
    r = np.linalg.norm(r_vec, axis=1)

    x_cross = np.dot(r_vec, n)
    y_cross = np.dot(r_vec, bn)
    theta = np.arctan2(y_cross, x_cross)
    # [-pi, pi] -> [0, 2*pi)
    theta = np.where(theta < 0.0, theta + 2.0 * math.pi, theta)

    return theta, z, r


def interpolate_curve_to_profile(
    curve_points: np.ndarray,
    origin: Sequence[float],
    axis: Sequence[float],
    normal: Optional[Sequence[float]] = None,
    num_samples: int = 64,
    is_closed: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """旧1D用: 単一3Dカーブ点群を円柱座標系グリッドのプロファイルに補間する (現行2D-SDF方式では未使用)。

    `build_wrinkle_profile_from_curves` および `test_wrinkle_profile.py` の回帰検出用に保持する。
    新規コードからは使用しないこと。
    """
    theta_pts, z_pts, r_pts = project_points_to_cylindrical(curve_points, origin, axis, normal)
    num_pts = len(theta_pts)
    if num_pts == 0:
        return np.zeros(num_samples, dtype=np.float32), np.zeros(num_samples, dtype=np.float32), np.zeros(num_samples, dtype=np.float32)

    t_unwrapped = np.unwrap(theta_pts)
    span = float(np.max(t_unwrapped) - np.min(t_unwrapped)) if num_pts > 0 else 0.0

    # 閉じたカーブかどうかの自動判定
    # 1. 始点と終点の3D距離が近い (閉じたポリライン)
    # 2. または角度スパンがほぼ全周 (>= 1.85π) をカバーしている
    if is_closed is None:
        if num_pts >= 3 and (np.linalg.norm(curve_points[0] - curve_points[-1]) < 1e-3 or span >= 1.85 * math.pi):
            is_closed = True
        else:
            is_closed = False

    grid_theta = np.linspace(0.0, 2.0 * math.pi, num_samples, endpoint=False, dtype=np.float32)

    if is_closed and num_pts >= 3:
        # 閉じたループの場合、角度順にソートして周期境界（0 と 2*pi）を拡張
        order = np.argsort(theta_pts)
        t_sorted = theta_pts[order]
        z_sorted = z_pts[order]
        r_sorted = r_pts[order]

        t_ext = np.concatenate([t_sorted - 2.0 * math.pi, t_sorted, t_sorted + 2.0 * math.pi])
        z_ext = np.concatenate([z_sorted, z_sorted, z_sorted])
        r_ext = np.concatenate([r_sorted, r_sorted, r_sorted])

        z_prof = np.interp(grid_theta, t_ext, z_ext).astype(np.float32)
        r_prof = np.interp(grid_theta, t_ext, r_ext).astype(np.float32)
        w_prof = np.ones(num_samples, dtype=np.float32)
    else:
        # 開いたカーブの場合: カーブの点列順（折れ線順）に沿って角度アンラップを行い、
        # 0 (2*pi) を跨ぐ円弧でも正しく局所的な角度スパン [t_min, t_max] を特定
        t_unwrapped = np.unwrap(theta_pts)
        order = np.argsort(t_unwrapped)
        t_sorted = t_unwrapped[order]
        z_sorted = z_pts[order]
        r_sorted = r_pts[order]

        t_min = float(t_sorted[0])
        t_max = float(t_sorted[-1])
        span = max(t_max - t_min, 1e-6)
        margin = 0.15 * span  # 端部の滑らかなフォールオフ区間

        z_prof = np.zeros(num_samples, dtype=np.float32)
        r_prof = np.zeros(num_samples, dtype=np.float32)
        w_prof = np.zeros(num_samples, dtype=np.float32)

        for i, th in enumerate(grid_theta):
            # th を [t_min - pi, t_min + pi] にシフトしてスパン内判定
            th_shifted = ((th - t_min + math.pi) % (2.0 * math.pi)) - math.pi + t_min
            if t_min <= th_shifted <= t_max:
                d_edge = min(th_shifted - t_min, t_max - th_shifted)
                if margin > 1e-6:
                    w = min(1.0, d_edge / margin)
                else:
                    w = 1.0
                w_prof[i] = w
                # 線形補間
                z_prof[i] = np.interp(th_shifted, t_sorted, z_sorted)
                r_prof[i] = np.interp(th_shifted, t_sorted, r_sorted)

    return z_prof, r_prof, w_prof


def build_wrinkle_profile_from_curves(
    crest_curves: Sequence[np.ndarray],
    root_curves: Sequence[np.ndarray],
    origin: Sequence[float],
    axis: Sequence[float],
    normal: Optional[Sequence[float]] = None,
    num_samples: int = 64,
    base_radius: float = 0.05,
    is_closed: Optional[bool] = None,
) -> WrinkleProfile:
    """旧1D用: 山/谷カーブ群から WrinkleProfile を生成する (現行2D-SDF方式では未使用)。

    複数カーブが重なる箇所では、ポテンシャル密度を加算し、目標位置（Z, R）を加重平均で滑らかにブレンドします。
    `test_wrinkle_profile.py` の回帰検出用に保持する。新規コードからは使用しないこと。
    """
    # 谷（Root）のメタボール合成バッファ
    v_z_weighted = np.zeros(num_samples, dtype=np.float64)
    v_r_weighted = np.zeros(num_samples, dtype=np.float64)
    v_w_sum = np.zeros(num_samples, dtype=np.float64)

    if root_curves:
        for pts in root_curves:
            if len(pts) >= 2:
                z_p, r_p, w_p = interpolate_curve_to_profile(
                    pts, origin, axis, normal, num_samples, is_closed=is_closed
                )
                v_w_sum += w_p
                v_z_weighted += w_p * z_p
                v_r_weighted += w_p * r_p

    v_has_weight = v_w_sum > 1e-6
    v_z = np.zeros(num_samples, dtype=np.float32)
    v_r = np.full(num_samples, base_radius * 0.9, dtype=np.float32)
    v_w = np.clip(v_w_sum, 0.0, 1.0).astype(np.float32)
    v_z[v_has_weight] = (v_z_weighted[v_has_weight] / v_w_sum[v_has_weight]).astype(np.float32)
    v_r[v_has_weight] = (v_r_weighted[v_has_weight] / v_w_sum[v_has_weight]).astype(np.float32)

    # 山（Crest）のメタボール合成バッファ
    c_z_weighted = np.zeros(num_samples, dtype=np.float64)
    c_r_weighted = np.zeros(num_samples, dtype=np.float64)
    c_w_sum = np.zeros(num_samples, dtype=np.float64)

    if crest_curves:
        for pts in crest_curves:
            if len(pts) >= 2:
                z_p, r_p, w_p = interpolate_curve_to_profile(
                    pts, origin, axis, normal, num_samples, is_closed=is_closed
                )
                c_w_sum += w_p
                c_z_weighted += w_p * z_p
                c_r_weighted += w_p * r_p

    c_has_weight = c_w_sum > 1e-6
    c_z = np.zeros(num_samples, dtype=np.float32)
    c_r = np.full(num_samples, base_radius * 1.1, dtype=np.float32)
    c_w = np.clip(c_w_sum, 0.0, 1.0).astype(np.float32)
    c_z[c_has_weight] = (c_z_weighted[c_has_weight] / c_w_sum[c_has_weight]).astype(np.float32)
    c_r[c_has_weight] = (c_r_weighted[c_has_weight] / c_w_sum[c_has_weight]).astype(np.float32)

    return WrinkleProfile(
        valley_z=v_z,
        crest_radius=c_r,
        valley_weight=v_w,
        crest_weight=c_w,
        valley_radius=v_r,
        crest_z=c_z,
    )


def extract_curves_from_blender_collection(
    collection_name: str,
    default_influence_radius: float = 0.03,
) -> Tuple[List["WrinkleCurveItem"], List["WrinkleCurveItem"]]:
    """
    Blenderシーンから指定コレクション内のCurveオブジェクトを走査し、
    山カーブ (Crest_* / Ridge_*) と谷カーブ (Root_* / Valley_*) の3Dワールド座標点列を抽出します。
    オブジェクトのカスタムプロパティ (wrinkle_strength, wrinkle_influence_radius, wrinkle_target_radius) を読み取ります。
    """
    if not _HAS_BPY:
        raise RuntimeError("Blender environment (bpy) is not available")

    col = bpy.data.collections.get(collection_name)
    if not col:
        return [], []

    crest_curves = []
    root_curves = []

    for obj in col.objects:
        if obj.type != 'CURVE':
            continue

        name_lower = obj.name.lower()
        is_crest = "crest" in name_lower or "ridge" in name_lower
        is_root = "root" in name_lower or "valley" in name_lower

        if not (is_crest or is_root):
            # 命名指定がない場合はカスタムプロパティを検査
            w_type = obj.get("wrinkle_type", "")
            if w_type == "crest":
                is_crest = True
            elif w_type == "root":
                is_root = True
            else:
                continue

        # 個別パラメータの取得
        strength = float(obj.get("wrinkle_strength", 1.0))
        inf_radius = float(obj.get("wrinkle_influence_radius", default_influence_radius))
        target_radius = obj.get("wrinkle_target_radius", None)
        if target_radius is not None:
            target_radius = float(target_radius)
        try:
            target_bone = str(obj.get("wrinkle_target_bone", "") or "")
        except Exception:
            target_bone = ""

        # スプライン頂点のワールド座標を取得
        curve_data = obj.data
        world_matrix = obj.matrix_world
        for spline in curve_data.splines:
            points = []
            if spline.type == 'BEZIER':
                for bp in spline.bezier_points:
                    points.append(list(world_matrix @ bp.co))
            else:
                for p in spline.points:
                    # NURBS/POLYは4要素 (x, y, z, w)
                    co = p.co.xyz if hasattr(p.co, "xyz") else p.co[:3]
                    points.append(list(world_matrix @ co))

            if len(points) >= 2:
                pts_arr = np.array(points, dtype=np.float32)
                item = WrinkleCurveItem(
                    points=pts_arr,
                    strength=strength,
                    influence_radius=inf_radius,
                    target_radius=target_radius,
                    target_bone=target_bone,
                )
                if is_crest:
                    crest_curves.append(item)
                elif is_root:
                    root_curves.append(item)

    return crest_curves, root_curves


@dataclass
class WrinkleCurveItem:
    """
    シワカーブの個別パラメータコンテナ
    - points: 3Dワールド座標点列 (N, 3), float32
    - strength: 個別強度倍率 (全体強度に乗算される倍率, 既定値: 1.0)
    - influence_radius: 個別影響半径 (m, グラデーション幅, 既定値: 0.03)
    - target_radius: 目標半径 (m, Noneの場合はカーブ自身の半径を使用)
    - target_bone: 帰属ボーン名 ("": 未指定。コレクション抽出時に設定される)
    """
    points: np.ndarray
    strength: float = 1.0
    influence_radius: float = 0.03
    target_radius: Optional[float] = None
    target_bone: str = ""


def _normalize_curve_items(
    curves: Sequence[Any],
    default_influence_radius: float,
) -> List[WrinkleCurveItem]:
    """入力カーブ列（np.ndarray, dict, tuple, WrinkleCurveItem）を WrinkleCurveItem のリストへ正規化"""
    items = []
    for c in curves:
        if isinstance(c, WrinkleCurveItem):
            items.append(c)
        elif isinstance(c, dict):
            pts = np.asarray(c.get("points", []), dtype=np.float32)
            s = float(c.get("strength", 1.0))
            inf = float(c.get("influence_radius", default_influence_radius))
            tr = c.get("target_radius", None)
            if tr is not None:
                tr = float(tr)
            items.append(WrinkleCurveItem(pts, s, inf, tr))
        elif isinstance(c, (list, tuple)) and len(c) >= 2 and isinstance(c[0], (np.ndarray, list)):
            pts = np.asarray(c[0], dtype=np.float32)
            s = float(c[1])
            inf = float(c[2]) if len(c) >= 3 else default_influence_radius
            tr = float(c[3]) if len(c) >= 4 else None
            items.append(WrinkleCurveItem(pts, s, inf, tr))
        else:
            pts = np.asarray(c, dtype=np.float32)
            items.append(WrinkleCurveItem(pts, 1.0, default_influence_radius))
    return items


@dataclass
class WrinkleCurveDesc:
    """単一カーブの骨ローカル形状記述子（ベイクキャッシュ・スライド照合用）。

    theta/z/r は `project_points_to_cylindrical` による骨ローカル座標。
    r は target_radius 上書き適用済みの実効半径。ワールド原点を含まないため、
    骨とカーブが剛体追従する限り姿勢変化でも不変となる。
    """
    kind: str                    # "crest" or "root"
    theta: np.ndarray            # (N,) [0, 2pi), float32
    z: np.ndarray                # (N,) m, float32
    r: np.ndarray                # (N,) m, float32 (実効半径)
    strength: float
    influence_radius: float


def describe_wrinkle_curves(
    crest_curves: Sequence[Any],
    root_curves: Sequence[Any],
    origin: Sequence[float],
    axis: Sequence[float],
    normal: Optional[Sequence[float]],
    default_influence_radius: float,
) -> List[WrinkleCurveDesc]:
    """カーブ群を骨ローカル記述子列へ変換する（ベイク本体より2桁軽い射影のみ）。

    順序は入力順を保持する（抽出順が安定している前提で照合は index-wise）。
    点数1以下のカーブは除外する（ベイク時と同様に無視されるため）。
    """
    norm_crest = _normalize_curve_items(crest_curves, default_influence_radius)
    norm_root = _normalize_curve_items(root_curves, default_influence_radius)
    axis_arr = np.asarray(axis, dtype=np.float32)
    descs: List[WrinkleCurveDesc] = []
    for kind, items in (("crest", norm_crest), ("root", norm_root)):
        for item in items:
            pts = np.asarray(item.points, dtype=np.float32)
            if len(pts) < 2:
                continue
            ths, zs, rs = project_points_to_cylindrical(pts, origin, axis_arr, normal)
            if item.target_radius is not None:
                rs = np.full_like(rs, float(item.target_radius))
            descs.append(WrinkleCurveDesc(
                kind=kind,
                theta=np.ascontiguousarray(ths, dtype=np.float32),
                z=np.ascontiguousarray(zs, dtype=np.float32),
                r=np.ascontiguousarray(rs, dtype=np.float32),
                strength=float(item.strength),
                influence_radius=float(item.influence_radius),
            ))
    return descs


def wrinkles_descs_match(
    a: Sequence[WrinkleCurveDesc],
    b: Sequence[WrinkleCurveDesc],
    theta_tol: float = 2e-4,
    zr_tol: float = 2e-6,
) -> bool:
    """記述子列の等価判定（許容値付き）。

    形状署名ハッシュの代わりにキャッシュ照合に用いる。float32の1ULP
    ノイズ（BLAS差・剛体移動の丸め）では不一致にしないため、量子化境界の
    非決定性を持たない。thetaは周期的距離で評価する。
    """
    if len(a) != len(b):
        return False
    for da, db in zip(a, b):
        if da.kind != db.kind:
            return False
        if len(da.theta) != len(db.theta):
            return False
        if len(da.theta) == 0:
            continue
        dth = np.abs((db.theta - da.theta + math.pi) % (2.0 * math.pi) - math.pi)
        if float(np.max(dth)) > theta_tol:
            return False
        if float(np.max(np.abs(db.z - da.z))) > zr_tol:
            return False
        if float(np.max(np.abs(db.r - da.r))) > zr_tol:
            return False
        if abs(db.strength - da.strength) > 1e-9:
            return False
        if abs(db.influence_radius - da.influence_radius) > 1e-9:
            return False
    return True


def wrinkle_shape_signature(
    descs: Sequence[WrinkleCurveDesc],
    width: int,
    height: int,
    bone_radius: float,
    cloth_radius: float,
    resample_step: float,
) -> str:
    """形状署名を返す（プロセス安定な sha256 hex）。

    デバッグ・ログ表示用の識別子。キャッシュ照合の等価判定には
    `wrinkles_descs_match`（許容値付き）を用いること。ハッシュは量子化
    境界で1ULP非決定性を持ち得るため、等価性テストには使わないこと。
    """
    h = hashlib.sha256()
    h.update(f"{width}x{height}|{bone_radius:.6f}|{cloth_radius:.6f}|{resample_step:.6f}|".encode())
    for d in descs:
        h.update(d.kind.encode())
        h.update(np.round(d.theta.astype(np.float64), 4).tobytes())
        h.update(np.round(d.z.astype(np.float64), 5).tobytes())
        h.update(np.round(d.r.astype(np.float64), 5).tobytes())
        h.update(f"|{d.strength:.6f}|{d.influence_radius:.6f};".encode())
    return h.hexdigest()


def match_slide_transform(
    cached: Sequence[WrinkleCurveDesc],
    current: Sequence[WrinkleCurveDesc],
    theta_tol: float = 2e-3,
    z_std_tol: float = 5e-4,
    r_resid_tol: float = 1e-3,
) -> Optional[Dict[str, float]]:
    """正準形状からの剛体スライド (dz, r_scale) 照合。

    スライド操作は theta 保存・z 定数シフト・r アフィン変換のため、
    その条件を満たせばフルベイク不要で z_range/r_range の uniform 更新のみで
    厳密に等価になる（target' = a*target + b より range' = a*range + b）。
    強度の一様スケールも検出し stiffness 側へ折り畳めるよう返す。
    戻りは {"dz", "r_scale", "r_offset", "strength_scale"}、条件不一致時は None。
    """
    if len(cached) != len(current):
        return None
    if not cached:
        return None
    dz_list = []
    strength_ratios = []
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    total_n = 0
    for dc, dn in zip(cached, current):
        if dc.kind != dn.kind:
            return None
        if len(dc.theta) != len(dn.theta):
            return None
        if len(dc.theta) == 0:
            continue
        dth = np.abs((dn.theta - dc.theta + math.pi) % (2.0 * math.pi) - math.pi)
        if float(np.max(dth)) > theta_tol:
            return None
        dz = dn.z - dc.z
        dz_list.append(float(np.mean(dz)))
        if float(np.std(dz)) > z_std_tol:
            return None
        # r のアフィン適合: r_cur = a * r_cached + b（全カーブでプール）
        xc = dc.r.astype(np.float64)
        xn = dn.r.astype(np.float64)
        sx += float(np.sum(xc))
        sy += float(np.sum(xn))
        sxx += float(np.sum(xc * xc))
        sxy += float(np.sum(xc * xn))
        total_n += len(xc)
        if dn.influence_radius != dc.influence_radius:
            return None
        if dc.strength <= 1e-9 or dn.strength <= 1e-9:
            if abs(dn.strength - dc.strength) > 1e-9:
                return None
            strength_ratios.append(1.0)
        else:
            strength_ratios.append(float(dn.strength) / float(dc.strength))
    dz_mean = float(sum(dz_list) / len(dz_list)) if dz_list else 0.0
    if any(abs(d - dz_mean) > z_std_tol for d in dz_list):
        return None
    if total_n == 0:
        return {"dz": dz_mean, "r_scale": 1.0, "r_offset": 0.0,
                "strength_scale": float(strength_ratios[0]) if strength_ratios else 1.0}
    denom = sxx - sx * sx / total_n
    if abs(denom) < 1e-12:
        # 縮退（全カーブが定数半径）: 軸中心スケールとして解釈する。
        # スライド操作の rad_ratio は軸中心スケール (b=0) であり、定数リングの
        # 重み>0領域では B/A デコードも厳密に一致する。平均半径ゼロ時は
        # オフセットとして扱い、許容以上の差は不一致とする。
        mean_c = sx / total_n
        mean_n = sy / total_n
        if abs(mean_c) > 1e-6:
            r_scale, r_offset = float(mean_n / mean_c), 0.0
        else:
            r_scale, r_offset = 1.0, float(mean_n - mean_c)
            if abs(r_offset) > r_resid_tol:
                return None
    else:
        r_scale = float((sxy - sx * sy / total_n) / denom)
        r_offset = float((sy - r_scale * sx) / total_n)
    # 残差の厳密検査
    for dc, dn in zip(cached, current):
        if len(dc.theta) == 0:
            continue
        resid = dn.r.astype(np.float64) - (r_scale * dc.r.astype(np.float64) + r_offset)
        if float(np.max(np.abs(resid))) > r_resid_tol:
            return None
    s0 = strength_ratios[0] if strength_ratios else 1.0
    if any(abs(s - s0) > 1e-6 for s in strength_ratios):
        return None
    return {"dz": dz_mean, "r_scale": float(r_scale), "r_offset": float(r_offset),
            "strength_scale": float(s0)}


@dataclass
class WrinkleTexture2D:
    """円柱UV展開されたドレープガイド2D-SDFテクスチャコンテナ"""
    width: int
    height: int
    texture_bytes: bytes       # RGBA8Unorm (width * height * 4 バイト)
    z_min: float
    z_max: float
    r_min: float
    r_max: float
    rgba_array: np.ndarray     # shape: [height, width, 4], uint8


def bake_wrinkle_2d_sdf_texture(
    crest_curves: Sequence[Union[np.ndarray, WrinkleCurveItem]],
    root_curves: Sequence[Union[np.ndarray, WrinkleCurveItem]],
    origin: Sequence[float],
    axis: Sequence[float],
    normal: Optional[Sequence[float]] = None,
    width: int = 512,
    height: int = 256,
    influence_radius: float = 0.03,
    bone_radius: float = 0.05,
    cloth_radius: float = 0.06,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    resample_step: float = 0.002,
) -> WrinkleTexture2D:
    """
    山カーブ（Crest）群と谷カーブ（Root）群から、円柱UV展開された2D-SDFテクスチャ（RGBA8Unorm）を高速生成します。
    各カーブの個別強度 (strength) および個別影響半径 (influence_radius) を反映します。
    
    - U軸: 円周方向 (θ / 2π: 0.0〜1.0, 左右境界は自動ループ)
    - V軸: ボーン長手方向 (z_min〜z_max: 0.0〜1.0)
    - R: 谷のポテンシャル強度 (0〜255: 個別強度 × フォールオフ)
    - G: 山のポテンシャル強度 (0〜255: 個別強度 × フォールオフ)
    - B: 谷の正規化目標半径 (0〜255)
    - A: 山の正規化目標半径 (0〜255)
    """

    axis_norm, norm, binorm = build_orthonormal_basis(axis)
    if normal is not None:
        norm = np.array(normal, dtype=np.float32)
        norm = norm - np.dot(norm, axis_norm) * axis_norm
        norm = norm / (np.linalg.norm(norm) + 1e-8)
        binorm = np.cross(axis_norm, norm)

    norm_crest = _normalize_curve_items(crest_curves, influence_radius)
    norm_root = _normalize_curve_items(root_curves, influence_radius)

    all_curves = [item.points for item in norm_crest + norm_root]
    all_zs = []
    all_rs = []

    for pts in all_curves:
        if len(pts) > 0:
            _, zs, rs = project_points_to_cylindrical(pts, origin, axis_norm, norm)
            all_zs.extend(zs)
            all_rs.extend(rs)

    all_inf_radii = [item.influence_radius for item in norm_crest + norm_root]
    max_influence = max(all_inf_radii) if all_inf_radii else influence_radius

    if z_min is None:
        if all_zs:
            z_min = float(np.min(all_zs) - max_influence * 1.5)
        else:
            z_min = -0.15
    if z_max is None:
        if all_zs:
            z_max = float(np.max(all_zs) + max_influence * 1.5)
        else:
            z_max = 0.15

    if z_max <= z_min:
        z_max = z_min + 0.1

    if r_min is None:
        r_min = float(bone_radius * 0.85)
    if r_max is None:
        if all_rs:
            r_max = float(max(cloth_radius + 0.02, np.max(all_rs) + 0.01))
        else:
            r_max = float(cloth_radius + 0.02)

    if r_max <= r_min:
        r_max = r_min + 0.05

    # グリッド定義（従来と同一: Uはendpoint=False、Vはendpoint=True）
    # x_col[i] = i * circ/width, z_row[j] = z_min + j*(z_max-z_min)/(height-1)
    if width <= 0 or height <= 0:
        empty = np.zeros((max(height, 0), max(width, 0), 4), dtype=np.uint8)
        return WrinkleTexture2D(
            width=max(width, 0),
            height=max(height, 0),
            texture_bytes=b"",
            z_min=float(z_min),
            z_max=float(z_max),
            r_min=float(r_min),
            r_max=float(r_max),
            rgba_array=empty,
        )
    circ_phys = max(2.0 * math.pi * float(bone_radius), 1e-6)
    dx_tex = circ_phys / float(width)
    dz_tex = float(z_max - z_min) / float(max(height - 1, 1))
    xs_centers = np.arange(width, dtype=np.float64) * dx_tex
    zs_centers = float(z_min) + np.arange(height, dtype=np.float64) * dz_tex

    def sample_single_curve_canonical(pts: np.ndarray, target_r: Optional[float] = None):
        """単一カーブを稠密サンプリングして正準点群 (x in [0, circ), z, r) を生成。

        従来の `sample_single_curve_phys` と同一の補間点列だが、3コピー展開は
        行わない。周期境界は参照側のmod距離で扱うため結果は等価。
        """
        if len(pts) < 2:
            return None, None, None
        ths, zs, rs = project_points_to_cylindrical(pts, origin, axis_norm, norm)
        if target_r is not None:
            rs = np.full_like(rs, target_r)
        n_pts = len(pts)
        total = 0
        segs = []
        for i in range(n_pts - 1):
            p0_th, p0_z, p0_r = float(ths[i]), float(zs[i]), float(rs[i])
            p1_th, p1_z, p1_r = float(ths[i + 1]), float(zs[i + 1]), float(rs[i + 1])
            dth = ((p1_th - p0_th + math.pi) % (2.0 * math.pi)) - math.pi
            dz = p1_z - p0_z
            seg_len = math.sqrt((dth * bone_radius) ** 2 + dz ** 2)
            num_sub = max(1, int(math.ceil(seg_len / resample_step)))
            segs.append((p0_th, p0_z, p0_r, dth, dz, p1_r - p0_r, num_sub))
            total += num_sub + 1
        xs = np.empty(total, dtype=np.float32)
        ys = np.empty(total, dtype=np.float32)
        rr = np.empty(total, dtype=np.float32)
        k = 0
        for (p0_th, p0_z, p0_r, dth, dz, dr, num_sub) in segs:
            for s in range(num_sub + 1):
                frac = s / float(num_sub)
                th = (p0_th + frac * dth) % (2.0 * math.pi)
                xs[k] = th * bone_radius
                ys[k] = p0_z + frac * dz
                rr[k] = p0_r + frac * dr
                k += 1
        return xs, ys, rr

    def splat_channel(items):
        """カーブ群の距離場をsplat反転で求める（画素->全点の総当たりを廃止）。

        各参照点の影響BOX内画素だけを更新するため、計算量は
        O(Nref * (inf/dx) * (inf/dz))。U周期はmod距離で厳密に扱う。
        戻りは (potential[H*W], near_r[H*W]) で従来の max合成と同一意味。
        """
        n = height * width
        pot = np.zeros(n, dtype=np.float32)
        near = np.full(n, (r_min + r_max) * 0.5, dtype=np.float32)
        half = circ_phys * 0.5
        for item in items:
            strength = max(0.0, float(item.strength))
            if strength <= 0.0:
                continue
            sampled = sample_single_curve_canonical(item.points, item.target_radius)
            if sampled[0] is None:
                continue
            xs, ys, rs = sampled
            r_inf = max(float(item.influence_radius), 1e-6)
            cbest = np.full(n, np.inf, dtype=np.float32)
            cbest_r = np.zeros(n, dtype=np.float32)
            for idx in range(len(xs)):
                px = float(xs[idx])
                py = float(ys[idx])
                pr = float(rs[idx])
                j0 = int(math.floor((py - r_inf - z_min) / dz_tex)) if dz_tex > 0 else 0
                j1 = int(math.floor((py + r_inf - z_min) / dz_tex)) if dz_tex > 0 else height - 1
                if j1 < 0 or j0 > height - 1:
                    continue
                j0c = max(0, j0)
                j1c = min(height - 1, j1)
                i0 = int(math.floor((px - r_inf) / dx_tex))
                i1 = int(math.floor((px + r_inf) / dx_tex))
                if i1 - i0 + 1 >= width:
                    cols = np.arange(width)
                else:
                    cols = np.mod(np.arange(i0, i1 + 1), width)
                xcols = cols.astype(np.float64) * dx_tex
                dxs = np.mod(xcols - px + half, circ_phys) - half
                zrows = zs_centers[j0c:j1c + 1]
                dy = zrows - py
                d = np.sqrt(dxs[None, :] ** 2 + dy[:, None] ** 2).astype(np.float32)
                rows = np.arange(j0c, j1c + 1, dtype=np.int64)[:, None] * width + cols[None, :]
                flat = rows.ravel()
                dv = d.ravel()
                upd = dv < cbest[flat]
                if np.any(upd):
                    sel = flat[upd]
                    cbest[sel] = dv[upd]
                    cbest_r[sel] = pr
            t = np.clip(1.0 - cbest / r_inf, 0.0, 1.0)
            falloff = (t * t * (3.0 - 2.0 * t) * strength).astype(np.float32)
            mask = falloff > pot
            pot[mask] = falloff[mask]
            near[mask] = cbest_r[mask]
        return pot, near

    # 1. 谷（Root）カーブの距離変換およびマルチカーブ合成
    r_potential, near_r_v_all = splat_channel(norm_root)

    r_channel = np.clip(r_potential * 255.0, 0.0, 255.0).astype(np.uint8)
    norm_r_v = np.clip((near_r_v_all - r_min) / (r_max - r_min), 0.0, 1.0)
    b_channel = (norm_r_v * 255.0).astype(np.uint8)

    # 2. 山（Crest）カーブの距離変換およびマルチカーブ合成
    g_potential, near_r_c_all = splat_channel(norm_crest)

    g_channel = np.clip(g_potential * 255.0, 0.0, 255.0).astype(np.uint8)
    norm_r_c = np.clip((near_r_c_all - r_min) / (r_max - r_min), 0.0, 1.0)
    a_channel = (norm_r_c * 255.0).astype(np.uint8)

    # RGBA 配列の構築 (height, width, 4)
    rgba = np.stack([
        r_channel.reshape((height, width)),
        g_channel.reshape((height, width)),
        b_channel.reshape((height, width)),
        a_channel.reshape((height, width)),
    ], axis=-1)

    return WrinkleTexture2D(
        width=width,
        height=height,
        texture_bytes=rgba.tobytes(),
        z_min=float(z_min),
        z_max=float(z_max),
        r_min=float(r_min),
        r_max=float(r_max),
        rgba_array=rgba,
    )


def resolve_bone_radius(
    explicit: float = 0.0,
    head_r: float = 0.0,
    tail_r: float = 0.0,
    length: float = 0.0,
) -> float:
    """ボーン代理半径を解決する (Blender非依存の純粋関数)。

    優先順位: 明示値 > 先尾平均 > 骨長フォールバック。
    旧来の先端側 (`head_radius`) のみの参照はテーパーを無視するため、
    両端の平均を用いる。`Bone.head_radius` は本来Envelope変形用の関節半径の
    流用であり、実体表との一致保証はない (明示値での上書きを推奨)。
    """
    if explicit is not None and float(explicit) > 1e-6:
        return float(explicit)
    vals = [float(v) for v in (head_r, tail_r) if v is not None and float(v) > 1e-4]
    if vals:
        return float(sum(vals) / len(vals))
    return float(max(1e-4, float(length) * 0.15))


def derive_wrinkle_influence_spans(
    points: np.ndarray,
    origin: Sequence[float],
    axis: Sequence[float],
    normal: Optional[Sequence[float]] = None,
    explicit_target_radius: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """単一カーブ点群から影響範囲表示用の角度スパン・Z範囲・目標半径を導出する。

    重み付け自体は `bake_wrinkle_2d_sdf_texture` の角度・長さ距離に基づくため、
    本関数は表示用の外形 (theta_min/max, z_min/max, target_r) のみを返す。
    閉曲線 (始終点近接または角度全周) では全周 (2π) を返す。
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0 or len(pts) < 2:
        return None
    thetas, zs, rs = project_points_to_cylindrical(pts, origin, axis, normal)

    t_unwrapped = np.unwrap(thetas.astype(np.float64))
    span = float(np.max(t_unwrapped) - np.min(t_unwrapped))
    closed = bool(
        float(np.linalg.norm(pts[0] - pts[-1])) < 1e-3 or span >= 1.85 * math.pi
    )
    if closed:
        theta_min = 0.0
        theta_max = 2.0 * math.pi
    else:
        theta_min = float(np.min(t_unwrapped))
        theta_max = float(np.max(t_unwrapped))
        if theta_max - theta_min < 1e-6:
            theta_max = theta_min + 1e-3

    if explicit_target_radius is not None and float(explicit_target_radius) > 1e-6:
        target_r = float(explicit_target_radius)
    else:
        target_r = float(np.mean(rs)) if len(rs) else 0.0

    return {
        "theta_min": float(theta_min),
        "theta_max": float(theta_max),
        "z_min": float(np.min(zs)),
        "z_max": float(np.max(zs)) if float(np.max(zs)) > float(np.min(zs)) else float(np.min(zs)) + 1e-3,
        "target_r": float(target_r),
        "is_closed": closed,
    }
