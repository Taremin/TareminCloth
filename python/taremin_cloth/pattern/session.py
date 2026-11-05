# -*- coding: utf-8 -*-
"""型紙エディタのセッション状態、ビューポート座標変換、スナップ計算 (Blender非依存)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

from .document import Document, EdgeRange, Piece
from .geometry import Vec2


@dataclass
class ViewTransform2D:
    """2Dビューポートの画面ピクセル座標とワールド座標（メートル）の相互変換.
    
    pan: ワールド原点 (0, 0) が画面上のどこに位置するか (ピクセル).
    zoom: 1メートルあたりのピクセル数 (pixels / meter).
          例: 1000.0 px/m の場合、1m = 1000px, 1cm = 10px.
    """
    pan_x: float = 400.0
    pan_y: float = 300.0
    zoom: float = 1000.0  # 1m = 1000px

    def world_to_screen(self, world_pos: tuple[float, float]) -> tuple[float, float]:
        """ワールド座標 (m) を画面ピクセル座標に変換する."""
        sx = self.pan_x + world_pos[0] * self.zoom
        sy = self.pan_y + world_pos[1] * self.zoom
        return (sx, sy)

    def screen_to_world(self, screen_pos: tuple[float, float]) -> tuple[float, float]:
        """画面ピクセル座標をワールド座標 (m) に変換する."""
        if abs(self.zoom) < 1e-6:
            return (0.0, 0.0)
        wx = (screen_pos[0] - self.pan_x) / self.zoom
        wy = (screen_pos[1] - self.pan_y) / self.zoom
        return (wx, wy)

    def pixel_to_world_dist(self, px: float) -> float:
        """ピクセル長さをワールド長さ (m) に換算する."""
        return float(px / max(self.zoom, 1e-6))

    def world_to_pixel_dist(self, world_dist: float) -> float:
        """ワールド長さ (m) をピクセル長さに換算する."""
        return float(world_dist * self.zoom)

    def zoom_at(self, screen_center: tuple[float, float], factor: float) -> None:
        """指定された画面座標を中心にズーム倍率を変更する."""
        factor = max(0.1, min(10.0, factor))
        cx, cy = screen_center
        # ズーム前のワールド座標
        wx, wy = self.screen_to_world(screen_center)
        self.zoom = max(10.0, min(100000.0, self.zoom * factor))
        # ズーム後も同一画面位置に留まるように pan を逆算
        self.pan_x = cx - wx * self.zoom
        self.pan_y = cy - wy * self.zoom


class SnapKind(str, Enum):
    """スナップ種別."""
    NONE = "NONE"
    GRID = "GRID"
    NODE = "NODE"


@dataclass
class SnapResult:
    """スナップ計算結果."""
    world_pos: tuple[float, float]
    kind: SnapKind = SnapKind.NONE
    snapped_node_id: Optional[str] = None


@dataclass
class SnapSettings:
    """スナップ動作設定."""
    enabled: bool = True
    snap_grid: bool = True
    snap_nodes: bool = True
    grid_size: float = 0.01  # 1cm (0.01m)
    snap_threshold_pixels: float = 10.0  # ピクセル半径以内なら吸着


@dataclass
class SelectionState:
    """現在の選択状態コンテナ."""
    piece_ids: set[str] = field(default_factory=set)
    node_ids: set[str] = field(default_factory=set)
    edge_ranges: list[EdgeRange] = field(default_factory=list)
    active_piece_id: Optional[str] = None

    def clear(self) -> None:
        """全選択をクリアする."""
        self.piece_ids.clear()
        self.node_ids.clear()
        self.edge_ranges.clear()
        self.active_piece_id = None

    def select_piece(self, piece_id: str, *, add: bool = False) -> None:
        """ピースを選択する."""
        if not add:
            self.clear()
        self.piece_ids.add(piece_id)
        self.active_piece_id = piece_id

    def select_node(self, piece_id: str, node_id: str, *, add: bool = False) -> None:
        """ノードを選択する."""
        if not add:
            self.clear()
        self.piece_ids.add(piece_id)
        self.node_ids.add(node_id)
        self.active_piece_id = piece_id

    def is_piece_selected(self, piece_id: str) -> bool:
        """ピースが選択されているかどうか."""
        return piece_id in self.piece_ids

    def is_node_selected(self, node_id: str) -> bool:
        """ノードが選択されているかどうか."""
        return node_id in self.node_ids


def compute_snap(
    world_pos: tuple[float, float],
    doc: Document,
    *,
    view: ViewTransform2D,
    settings: Optional[SnapSettings] = None,
    exclude_node_ids: Optional[set[str]] = None,
) -> SnapResult:
    """ワールド座標に対するスナップ（既存ノード吸着、グリッド吸着）を評価して返す.
    
    優先度: 既存ノード吸着 > グリッド吸着
    """
    opts = settings or SnapSettings()
    if not opts.enabled:
        return SnapResult(world_pos=world_pos, kind=SnapKind.NONE)

    threshold_dist = view.pixel_to_world_dist(opts.snap_threshold_pixels)
    best_dist = threshold_dist
    best_node_pos: Optional[tuple[float, float]] = None
    best_node_id: Optional[str] = None
    exclude_ids = exclude_node_ids or set()

    # 1. 既存ノード吸着
    if opts.snap_nodes:
        for piece in doc.pieces.values():
            for path in piece.paths.values():
                for node in path.nodes:
                    if node.id in exclude_ids:
                        continue
                    w_pos = piece.transform.transform_point(node.position)
                    dx = w_pos[0] - world_pos[0]
                    dy = w_pos[1] - world_pos[1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist <= best_dist:
                        best_dist = dist
                        best_node_pos = w_pos
                        best_node_id = node.id

        if best_node_pos is not None:
            return SnapResult(
                world_pos=best_node_pos,
                kind=SnapKind.NODE,
                snapped_node_id=best_node_id,
            )

    # 2. グリッド吸着
    if opts.snap_grid and opts.grid_size > 1e-6:
        gx = round(world_pos[0] / opts.grid_size) * opts.grid_size
        gy = round(world_pos[1] / opts.grid_size) * opts.grid_size
        dx = gx - world_pos[0]
        dy = gy - world_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= threshold_dist:
            return SnapResult(
                world_pos=(gx, gy),
                kind=SnapKind.GRID,
            )

    return SnapResult(world_pos=world_pos, kind=SnapKind.NONE)
