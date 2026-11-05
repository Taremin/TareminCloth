# -*- coding: utf-8 -*-
"""2Dアフィン変換およびバウンディングボックス計算モジュール (Blender非依存)."""

import math
from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np

Vec2 = Tuple[float, float]


@dataclass(frozen=True)
class Affine2D:
    """2Dアフィン変換 (平行移動、回転、拡大縮小)."""

    tx: float = 0.0
    ty: float = 0.0
    rotation_rad: float = 0.0
    sx: float = 1.0
    sy: float = 1.0

    def apply(self, point: Vec2) -> Vec2:
        """単一の2D点にアフィン変換を適用する."""
        px, py = float(point[0]), float(point[1])
        # 拡大縮小
        sx_px = px * self.sx
        sy_py = py * self.sy
        # 回転
        cos_r = math.cos(self.rotation_rad)
        sin_r = math.sin(self.rotation_rad)
        rx = cos_r * sx_px - sin_r * sy_py
        ry = sin_r * sx_px + cos_r * sy_py
        # 平行移動
        return (rx + self.tx, ry + self.ty)

    def apply_array(self, pts: np.ndarray) -> np.ndarray:
        """NumPy配列 (N, 2) に対しアフィン変換を一括適用する."""
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
        arr = np.asarray(pts, dtype=np.float32)
        scaled = arr * np.array([self.sx, self.sy], dtype=np.float32)
        cos_r = math.cos(self.rotation_rad)
        sin_r = math.sin(self.rotation_rad)
        r_mat = np.array([[cos_r, sin_r], [-sin_r, cos_r]], dtype=np.float32)
        rotated = scaled @ r_mat
        transformed = rotated + np.array([self.tx, self.ty], dtype=np.float32)
        return transformed

    def inverse_apply(self, point: Vec2) -> Vec2:
        """逆アフィン変換を単一の点に適用する."""
        px = float(point[0]) - self.tx
        py = float(point[1]) - self.ty
        cos_r = math.cos(-self.rotation_rad)
        sin_r = math.sin(-self.rotation_rad)
        rx = cos_r * px - sin_r * py
        ry = sin_r * px + cos_r * py
        inv_sx = 1.0 / self.sx if abs(self.sx) > 1e-9 else 0.0
        inv_sy = 1.0 / self.sy if abs(self.sy) > 1e-9 else 0.0
        return (rx * inv_sx, ry * inv_sy)

    def transform_point(self, point: Vec2) -> Vec2:
        """単一の2D点にアフィン変換を適用する (apply の別名)."""
        return self.apply(point)

    def inverse_point(self, point: Vec2) -> Vec2:
        """逆アフィン変換を単一の点に適用する (inverse_apply の別名)."""
        return self.inverse_apply(point)


def compute_bounding_box(points: Sequence[Vec2]) -> tuple[Vec2, Vec2]:
    """点列から軸平行バウンディングボックス ((min_x, min_y), (max_x, max_y)) を返す."""
    if not points:
        return ((0.0, 0.0), (0.0, 0.0))

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for p in points:
        x, y = float(p[0]), float(p[1])
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y

    return ((min_x, min_y), (max_x, max_y))
