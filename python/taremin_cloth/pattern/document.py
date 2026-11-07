# -*- coding: utf-8 -*-
"""型紙エディタのコア文書データモデル (Single Source of Truth, Blender非依存)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional, Sequence

from .ids import new_id
from .geometry import (
    ArcLengthTable,
    Vec2,
    make_segment_controls,
    compute_auto_handles,
    build_arc_table,
    arc_length_at_t,
    point_at_arc_length,
    evaluate_cubic,
    split_cubic_bezier,
    merge_breakpoints_by_arc_length,
    signed_area,
    ensure_ccw,
    has_self_intersection,
    Affine2D,
)

DOCUMENT_VERSION: int = 1


class PathRole(str, Enum):
    """パスの役割定義."""
    OUTER = "OUTER"
    INTERNAL = "INTERNAL"
    HOLE = "HOLE"
    DART = "DART"


class NodeKind(str, Enum):
    """制御点ノードの幾何種別."""
    CORNER = "CORNER"
    SMOOTH = "SMOOTH"
    AUTO = "AUTO"


class SeamMode(str, Enum):
    """縫合モード."""
    SEGMENT = "SEGMENT"
    FREE = "FREE"
    CHAIN = "CHAIN"


class NotchKind(str, Enum):
    """合印（ノッチ）の形状種別."""
    V = "V"
    U = "U"
    SLIT = "SLIT"


class AnchorKind(str, Enum):
    """ピン・タック等のアンカー種別."""
    PIN = "PIN"
    TACK = "TACK"


@dataclass
class Node:
    """パス上の制御点."""
    id: str
    position: tuple[float, float]
    kind: NodeKind = NodeKind.CORNER
    in_handle: Optional[tuple[float, float]] = None   # position からの相対座標 (dx, dy)
    out_handle: Optional[tuple[float, float]] = None  # position からの相対座標 (dx, dy)


@dataclass
class Path:
    """複数のNodeを連結した輪郭または内部線."""
    id: str
    role: PathRole = PathRole.OUTER
    nodes: list[Node] = field(default_factory=list)
    closed: bool = True

    def get_segment_controls(self, seg_index: int) -> tuple[Vec2, Vec2, Vec2, Vec2]:
        """指定セグメント (nodes[seg_index] -> nodes[next]) の実効3次ベジェ制御点 (p0, c1, c2, p3) を返す (AUTOハンドル自動計算込み)."""
        n_nodes = len(self.nodes)
        num_segs = n_nodes if self.closed else n_nodes - 1
        if seg_index < 0 or seg_index >= num_segs:
            raise IndexError(f"Segment index {seg_index} out of range [0, {num_segs})")

        n0 = self.nodes[seg_index]
        n1 = self.nodes[(seg_index + 1) % n_nodes]

        out_h = n0.out_handle
        in_h = n1.in_handle

        if n0.kind == NodeKind.AUTO:
            p_prev = self.nodes[(seg_index - 1) % n_nodes].position if self.closed else self.nodes[max(0, seg_index - 1)].position
            _, out_h = compute_auto_handles(p_prev, n0.position, n1.position)

        if n1.kind == NodeKind.AUTO:
            p_next = self.nodes[(seg_index + 2) % n_nodes].position if self.closed else self.nodes[min(n_nodes - 1, seg_index + 2)].position
            in_h, _ = compute_auto_handles(n0.position, n1.position, p_next)

        return make_segment_controls(n0.position, out_h, in_h, n1.position)

    def resolve_edge_range(self, er: EdgeRange) -> Optional[tuple[int, float, float]]:
        """EdgeRange をこのパス上の順方向セグメント (seg_index, p_t0, p_t1) に解決する.

        逆方向ノード指定 (nodes[i+1] -> nodes[i]) の場合は (i, 1.0 - t0, 1.0 - t1) を返す。
        該当する隣接セグメントが存在しない場合は None を返す。
        """
        n_nodes = len(self.nodes)
        if n_nodes < 2:
            return None
        num_segs = n_nodes if self.closed else n_nodes - 1
        raw_t0 = max(0.0, min(1.0, float(er.t0)))
        raw_t1 = max(0.0, min(1.0, float(er.t1)))

        for i in range(num_segs):
            if self.nodes[i].id == er.start_node_id and self.nodes[(i + 1) % n_nodes].id == er.end_node_id:
                return (i, raw_t0, raw_t1)
        for i in range(num_segs):
            if self.nodes[i].id == er.end_node_id and self.nodes[(i + 1) % n_nodes].id == er.start_node_id:
                return (i, 1.0 - raw_t0, 1.0 - raw_t1)
        return None

    def sample_outline_points(self, *, steps_per_segment: int = 24) -> list[Vec2]:
        """パス全体をベジェ評価してサンプリングした2D頂点列 (ローカル座標) を返す.
        
        全ノードが CORNER かつハンドルなしの場合は各ノード座標のみを返し、曲線が含まれる場合は steps_per_segment で分割する.
        閉ループの場合は末尾が先頭と一致しない一巡の点列を返す.
        """
        n_nodes = len(self.nodes)
        if n_nodes == 0:
            return []
        if n_nodes == 1:
            return [self.nodes[0].position]

        # 曲線が含まれるか判定
        is_curved = any(
            n.kind != NodeKind.CORNER or n.in_handle is not None or n.out_handle is not None
            for n in self.nodes
        )

        if not is_curved:
            return [n.position for n in self.nodes]

        num_segs = n_nodes if self.closed else n_nodes - 1
        sampled_pts: list[Vec2] = []

        for i in range(num_segs):
            ctrls = self.get_segment_controls(i)
            table = build_arc_table(*ctrls, steps=steps_per_segment)
            pts = table.points
            # セグメントの最終点以外を追加（次のセグメントの始点と重複しないようにする）
            sampled_pts.extend(pts[:-1])

        if not self.closed and num_segs > 0:
            # 開いた折れ線の場合は最後のセグメントの最終点を追加
            last_ctrls = self.get_segment_controls(num_segs - 1)
            last_table = build_arc_table(*last_ctrls, steps=steps_per_segment)
            sampled_pts.append(last_table.points[-1])

        return sampled_pts


@dataclass
class Piece:
    """単一の型紙パーツ（外周パス、内部線、2Dアフィン配置、物性設定）."""
    id: str
    name: str
    paths: dict[str, Path] = field(default_factory=dict)
    outer_path_id: str = ""
    transform: Affine2D = field(default_factory=Affine2D)
    target_edge_length: float = 0.015
    layer_id: int = 0


@dataclass
class EdgeRange:
    """パス上の一区間または部分区間を指す参照."""
    piece_id: str
    path_id: str
    start_node_id: str
    end_node_id: str
    t0: float = 0.0
    t1: float = 1.0


@dataclass
class Waypoint:
    """3D空間内で縫合エッジをアバター周囲へ迂回させるための中継点."""
    id: str
    position_3d: tuple[float, float, float]


@dataclass
class Seam:
    """2つの辺範囲（および任意の中継点列）を結ぶ縫合定義."""
    id: str
    a: EdgeRange
    b: EdgeRange
    chain_a: list[EdgeRange] = field(default_factory=list)
    chain_b: list[EdgeRange] = field(default_factory=list)
    reverse: bool = False
    mode: SeamMode = SeamMode.SEGMENT
    shrink: float = 1.0
    waypoints: list[Waypoint] = field(default_factory=list)

    def effective_a(self) -> list[EdgeRange]:
        """A側の実効 EdgeRange リストを返す."""
        return self.chain_a if self.chain_a else [self.a]

    def effective_b(self) -> list[EdgeRange]:
        """B側の実効 EdgeRange リストを返す."""
        return self.chain_b if self.chain_b else [self.b]

    def iter_unique_edge_ranges(self) -> list[EdgeRange]:
        """a, b, chain_a, chain_b 内の全 EdgeRange をオブジェクト同一性で重複排除して返す."""
        seen: set[int] = set()
        result: list[EdgeRange] = []
        for er in [self.a, self.b] + self.chain_a + self.chain_b:
            oid = id(er)
            if oid not in seen:
                seen.add(oid)
                result.append(er)
        return result

    def sync_primary_from_chains(self) -> None:
        """chain_a / chain_b が非空のとき先頭要素を a / b に同期し、複数要素時は SeamMode.CHAIN に設定する."""
        if self.chain_a:
            self.a = self.chain_a[0]
        if self.chain_b:
            self.b = self.chain_b[0]
        if len(self.chain_a) > 1 or len(self.chain_b) > 1:
            self.mode = SeamMode.CHAIN


@dataclass
class Notch:
    """辺上に配置する合印切り込み."""
    id: str
    piece_id: str
    path_id: str
    segment_start_id: str
    t: float = 0.5
    kind: NotchKind = NotchKind.V
    depth: float = 0.005
    width: float = 0.004


@dataclass
class Anchor:
    """素体やボーンへの拘束アンカー（ピン留め / タック）."""
    id: str
    piece_id: str
    position: tuple[float, float]
    kind: AnchorKind = AnchorKind.PIN
    target_object: Optional[str] = None
    target_bone: Optional[str] = None
    weight: float = 1.0


@dataclass
class Placement:
    """型紙パーツの3D空間初期配置パラメータ."""
    piece_id: str
    offset_3d: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler_3d: tuple[float, float, float] = (0.0, 0.0, 0.0)
    preset: str = "CHEST_FRONT"
    target_bone: str = ""
    bone_t: float = 0.5
    bone_angle_deg: float = 0.0
    surface_offset: float = 0.08
    is_tubular: bool = False


@dataclass
class ReferenceImage:
    """キャンバス背景に表示する参考下絵画像."""
    image_path: str = ""
    offset: tuple[float, float] = (0.0, 0.0)
    width: float = 1.0  # ワールドメートル単位での表示幅
    opacity: float = 0.5
    visible: bool = True
    aspect_ratio: float = 1.0  # 幅に対する高さの比率 (height = width * aspect_ratio)


@dataclass
class Document:
    """型紙エディタの全データを保持するルートドキュメント."""
    version: int = DOCUMENT_VERSION
    name: str = "Untitled Pattern"
    pieces: dict[str, Piece] = field(default_factory=dict)
    seams: dict[str, Seam] = field(default_factory=dict)
    notches: dict[str, Notch] = field(default_factory=dict)
    anchors: dict[str, Anchor] = field(default_factory=dict)
    placements: dict[str, Placement] = field(default_factory=dict)
    reference_image: Optional[ReferenceImage] = None

    @classmethod
    def new(cls, *, name: str = "Untitled Pattern") -> Document:
        """新規の空ドキュメントを作成して返す."""
        return cls(name=name)

    def _unique_piece_name(self, base_name: str) -> str:
        """既存ピースと重複しない一意なピース名を生成する."""
        existing_names = {p.name for p in self.pieces.values()}
        if base_name not in existing_names:
            return base_name
        idx = 1
        while f"{base_name}_{idx}" in existing_names:
            idx += 1
        return f"{base_name}_{idx}"

    def add_rectangle_piece(
        self,
        p0: Vec2,
        p1: Vec2,
        *,
        name: Optional[str] = None,
    ) -> Piece:
        """対角2点から反時計回り (CCW) 4頂点の矩形ピースを作成し、文書に追加する."""
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        min_x, max_x = min(x0, x1), max(x0, x1)
        min_y, max_y = min(y0, y1), max(y0, y1)

        if (max_x - min_x < 0.001) or (max_y - min_y < 0.001):
            raise ValueError(f"Rectangle size too small: {max_x - min_x} x {max_y - min_y}")

        coords = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        ]

        p_name = self._unique_piece_name(name or "Piece_Rect")
        piece_id = new_id()
        path_id = new_id()

        nodes = [
            Node(id=new_id(), position=c, kind=NodeKind.CORNER)
            for c in coords
        ]

        path = Path(id=path_id, role=PathRole.OUTER, nodes=nodes, closed=True)
        piece = Piece(
            id=piece_id,
            name=p_name,
            paths={path_id: path},
            outer_path_id=path_id,
        )

        self.pieces[piece_id] = piece
        self.placements[piece_id] = Placement(piece_id=piece_id)
        return piece

    def add_ellipse_piece(
        self,
        center: Vec2,
        radius: Vec2,
        *,
        name: Optional[str] = None,
    ) -> Piece:
        """中心と半径 (rx, ry) から4頂点ベジェ曲線の楕円ピースを作成し、文書に追加する."""
        cx, cy = float(center[0]), float(center[1])
        rx, ry = abs(float(radius[0])), abs(float(radius[1]))

        if rx < 0.001 or ry < 0.001:
            raise ValueError(f"Ellipse radius too small: rx={rx}, ry={ry}")

        # 4頂点ベジェ近似定数 kappa = 4*(sqrt(2)-1)/3
        kappa = 0.5522847498307935
        kx = rx * kappa
        ky = ry * kappa

        # 反時計回り: 底 -> 右 -> 上 -> 左
        nodes = [
            Node(
                id=new_id(),
                position=(cx, cy - ry),
                kind=NodeKind.SMOOTH,
                in_handle=(-kx, 0.0),
                out_handle=(kx, 0.0),
            ),
            Node(
                id=new_id(),
                position=(cx + rx, cy),
                kind=NodeKind.SMOOTH,
                in_handle=(0.0, -ky),
                out_handle=(0.0, ky),
            ),
            Node(
                id=new_id(),
                position=(cx, cy + ry),
                kind=NodeKind.SMOOTH,
                in_handle=(kx, 0.0),
                out_handle=(-kx, 0.0),
            ),
            Node(
                id=new_id(),
                position=(cx - rx, cy),
                kind=NodeKind.SMOOTH,
                in_handle=(0.0, ky),
                out_handle=(0.0, -ky),
            ),
        ]

        p_name = self._unique_piece_name(name or "Piece_Ellipse")
        piece_id = new_id()
        path_id = new_id()

        path = Path(id=path_id, role=PathRole.OUTER, nodes=nodes, closed=True)
        piece = Piece(
            id=piece_id,
            name=p_name,
            paths={path_id: path},
            outer_path_id=path_id,
        )

        self.pieces[piece_id] = piece
        self.placements[piece_id] = Placement(piece_id=piece_id)
        return piece

    def add_polygon_piece(
        self,
        points: Sequence[Vec2],
        *,
        name: Optional[str] = None,
    ) -> Piece:
        """点列から反時計回り (CCW) 多角形ピースを作成し、文書に追加する.
        
        頂点数が3未満または自己交差が存在する場合は ValueError を送出する.
        """
        raw_pts = [(float(p[0]), float(p[1])) for p in points]
        pts: list[tuple[float, float]] = []
        for p in raw_pts:
            if not pts or math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > 1e-7:
                pts.append(p)
        if len(pts) >= 2 and math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) <= 1e-7:
            pts.pop()

        if len(pts) < 3:
            raise ValueError(f"Polygon must have at least 3 distinct vertices, got {len(pts)}")

        if abs(signed_area(pts)) < 1e-8:
            raise ValueError("Polygon is degenerate (zero area)")

        if has_self_intersection(pts, closed=True):
            raise ValueError("Polygon has self-intersection")

        ccw_pts = ensure_ccw(pts)
        p_name = self._unique_piece_name(name or "Piece_Poly")
        piece_id = new_id()
        path_id = new_id()

        nodes = [
            Node(id=new_id(), position=c, kind=NodeKind.CORNER)
            for c in ccw_pts
        ]

        path = Path(id=path_id, role=PathRole.OUTER, nodes=nodes, closed=True)
        piece = Piece(
            id=piece_id,
            name=p_name,
            paths={path_id: path},
            outer_path_id=path_id,
        )

        self.pieces[piece_id] = piece
        self.placements[piece_id] = Placement(piece_id=piece_id)
        return piece

    def evaluate_path_segments(
        self,
        piece_id: str,
        path_id: str,
    ) -> list[tuple[str, str, ArcLengthTable]]:
        """パスの各区間 (start_node_id, end_node_id, ArcLengthTable) を評価して返す."""
        piece = self.pieces.get(piece_id)
        if not piece:
            return []
        path = piece.paths.get(path_id)
        if not path or len(path.nodes) < 2:
            return []

        nodes = path.nodes
        n_nodes = len(nodes)
        num_segs = n_nodes if path.closed else n_nodes - 1
        segments: list[tuple[str, str, ArcLengthTable]] = []

        for i in range(num_segs):
            curr_node = nodes[i]
            next_node = nodes[(i + 1) % n_nodes]
            ctrls = path.get_segment_controls(i)
            table = build_arc_table(*ctrls)
            segments.append((curr_node.id, next_node.id, table))

        return segments

    def compute_edge_range_length(self, edge_range: EdgeRange) -> float:
        """指定された EdgeRange の実寸弧長 (メートル) を計算して返す."""
        piece = self.pieces.get(edge_range.piece_id)
        if not piece:
            return 0.0
        path = piece.paths.get(edge_range.path_id)
        if not path or len(path.nodes) < 2:
            return 0.0

        resolved = path.resolve_edge_range(edge_range)
        if resolved is None:
            return 0.0
        seg_idx, p_t0, p_t1 = resolved
        ctrls = path.get_segment_controls(seg_idx)
        target_seg = build_arc_table(*ctrls)
        s0 = arc_length_at_t(target_seg, p_t0)
        s1 = arc_length_at_t(target_seg, p_t1)
        return float(abs(s1 - s0))

    def get_segment_notch_breakpoints(
        self,
        piece_id: str,
        start_node_id: str,
        *,
        path_id: Optional[str] = None,
    ) -> list[float]:
        """指定セグメント上のノッチ分割比率を [0.0, t_1, ..., 1.0] の昇順リストで返す."""
        p = self.pieces.get(piece_id)
        if not p:
            return [0.0, 1.0]
        actual_path_id = path_id or p.outer_path_id
        path = p.paths.get(actual_path_id)
        if not path or len(path.nodes) < 2:
            return [0.0, 1.0]

        nodes = path.nodes
        n_nodes = len(nodes)
        num_segs = n_nodes if path.closed else n_nodes - 1
        seg_idx = next((i for i in range(num_segs) if nodes[i].id == start_node_id), None)
        if seg_idx is None:
            return [0.0, 1.0]

        ctrls = path.get_segment_controls(seg_idx)
        table = build_arc_table(*ctrls)

        notches_t = [
            n.t for n in self.notches.values()
            if n.piece_id == piece_id
            and n.path_id == actual_path_id
            and n.segment_start_id == start_node_id
            and 0.0 < n.t < 1.0
        ]
        return merge_breakpoints_by_arc_length(table, notches_t, min_arc_gap=0.001)

    def move_notch(
        self,
        notch_id: str,
        new_t: float,
    ) -> bool:
        """ノッチの位置 (t) を更新し、接続中の EdgeRange の境界も同期更新する."""
        notch = self.notches.get(notch_id)
        if not notch:
            return False

        piece = self.pieces.get(notch.piece_id)
        if not piece:
            return False

        path = piece.paths.get(notch.path_id)
        if not path or len(path.nodes) < 2:
            return False

        nodes = path.nodes
        n_nodes = len(nodes)
        num_segs = n_nodes if path.closed else n_nodes - 1
        seg_idx = next((i for i in range(num_segs) if nodes[i].id == notch.segment_start_id), None)
        if seg_idx is None:
            return False

        ctrls = path.get_segment_controls(seg_idx)
        target_seg = build_arc_table(*ctrls)
        tot_len = target_seg.total_length

        if tot_len <= 0.002:
            return False

        # セグメント全長から 1mm (0.001m) 相当の最小ギャップ
        min_gap_s = 0.001
        _, min_t = point_at_arc_length(target_seg, min_gap_s)
        _, max_t = point_at_arc_length(target_seg, tot_len - min_gap_s)

        # 同一セグメント上の他のノッチを取得
        other_ts = sorted([
            n.t for n in self.notches.values()
            if n.id != notch_id
            and n.piece_id == notch.piece_id
            and n.path_id == notch.path_id
            and n.segment_start_id == notch.segment_start_id
        ])

        lower_bound = min_t
        upper_bound = max_t
        for ot in other_ts:
            s_ot = arc_length_at_t(target_seg, ot)
            if ot < notch.t:
                _, cand_t = point_at_arc_length(target_seg, min(tot_len, s_ot + min_gap_s))
                lower_bound = max(lower_bound, cand_t)
            elif ot > notch.t:
                _, cand_t = point_at_arc_length(target_seg, max(0.0, s_ot - min_gap_s))
                upper_bound = min(upper_bound, cand_t)

        if lower_bound > upper_bound:
            clamped_t = 0.5 * (lower_bound + upper_bound)
        else:
            clamped_t = max(lower_bound, min(upper_bound, new_t))

        old_t = notch.t
        if abs(old_t - clamped_t) < 1e-6:
            return False

        notch.t = clamped_t

        # 関連する EdgeRange の端点 (t0, t1) を連動更新
        seg_start_id = notch.segment_start_id
        seg_end_id = nodes[(seg_idx + 1) % n_nodes].id

        for seam in self.seams.values():
            for er in seam.iter_unique_edge_ranges():
                if er.piece_id == notch.piece_id and er.path_id == notch.path_id:
                    # 順方向一致
                    if er.start_node_id == seg_start_id and er.end_node_id == seg_end_id:
                        if abs(er.t0 - old_t) < 1e-4:
                            er.t0 = clamped_t
                        if abs(er.t1 - old_t) < 1e-4:
                            er.t1 = clamped_t
                    # 逆方向一致
                    elif er.start_node_id == seg_end_id and er.end_node_id == seg_start_id:
                        rev_old_t = 1.0 - old_t
                        rev_clamped_t = 1.0 - clamped_t
                        if abs(er.t0 - rev_old_t) < 1e-4:
                            er.t0 = rev_clamped_t
                        if abs(er.t1 - rev_old_t) < 1e-4:
                            er.t1 = rev_clamped_t

        return True

    def snap_notch_t(
        self,
        piece_id: str,
        start_node_id: str,
        *,
        raw_t: float,
        path_id: Optional[str] = None,
        snap_enabled: bool = True,
    ) -> float:
        """ノッチパラメータ t を 1cm 刻みおよび中点 (50%) に吸着させて返す."""
        p = self.pieces.get(piece_id)
        if not p:
            return max(0.001, min(0.999, raw_t))
        actual_path_id = path_id or p.outer_path_id
        segs = self.evaluate_path_segments(piece_id, actual_path_id)
        target_seg = next((tbl for s_id, _, tbl in segs if s_id == start_node_id), None)
        if not target_seg or target_seg.total_length < 1e-6:
            return max(0.001, min(0.999, raw_t))

        tot_len = target_seg.total_length
        min_gap = max(0.001, 0.001 / tot_len)
        clamped_raw = max(min_gap, min(1.0 - min_gap, raw_t))
        if not snap_enabled:
            return clamped_raw

        cur_s = arc_length_at_t(target_seg, clamped_raw)
        # 1cm 刻みの弧長
        snapped_s = round(cur_s / 0.01) * 0.01
        snapped_s = max(0.001, min(tot_len - 0.001, snapped_s))
        _, s_cand_t = point_at_arc_length(target_seg, snapped_s)

        # 中点 (t=0.5) の弧長判定
        mid_s = arc_length_at_t(target_seg, 0.5)
        if abs(cur_s - mid_s) < 0.008:
            return 0.5

        return max(min_gap, min(1.0 - min_gap, s_cand_t))

    def evaluate_edge_range_geometry(
        self,
        edge_range: EdgeRange,
        *,
        steps: int = 16,
    ) -> tuple[Vec2, Vec2, Vec2, list[Vec2]]:
        """EdgeRange の (ワールド始点, ワールド中点, ワールド終点, ワールドサンプリング点列) を返す."""
        piece = self.pieces.get(edge_range.piece_id)
        if not piece:
            return ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), [])

        path = piece.paths.get(edge_range.path_id)
        if not path or len(path.nodes) < 2:
            return ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), [])

        resolved = path.resolve_edge_range(edge_range)
        if resolved is None:
            # セグメントとして隣接していない場合、直接2点ノードを結ぶフォールバック
            n0 = next((n for n in path.nodes if n.id == edge_range.start_node_id), None)
            n1 = next((n for n in path.nodes if n.id == edge_range.end_node_id), None)
            if not n0 or not n1:
                return ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), [])
            w0 = piece.transform.transform_point(n0.position)
            w1 = piece.transform.transform_point(n1.position)
            wm = ((w0[0] + w1[0]) * 0.5, (w0[1] + w1[1]) * 0.5)
            return (w0, wm, w1, [w0, w1])

        seg_idx, p_t0, p_t1 = resolved
        ctrls = path.get_segment_controls(seg_idx)

        t_mid = 0.5 * (p_t0 + p_t1)
        p0_loc = evaluate_cubic(ctrls[0], ctrls[1], ctrls[2], ctrls[3], p_t0)
        pm_loc = evaluate_cubic(ctrls[0], ctrls[1], ctrls[2], ctrls[3], t_mid)
        p1_loc = evaluate_cubic(ctrls[0], ctrls[1], ctrls[2], ctrls[3], p_t1)

        step_count = max(1, steps)
        pts_loc = [
            evaluate_cubic(ctrls[0], ctrls[1], ctrls[2], ctrls[3], p_t0 + (p_t1 - p_t0) * (i / step_count))
            for i in range(step_count + 1)
        ]

        w_p0 = piece.transform.transform_point(p0_loc)
        w_pm = piece.transform.transform_point(pm_loc)
        w_p1 = piece.transform.transform_point(p1_loc)
        w_pts = [piece.transform.transform_point(pt) for pt in pts_loc]

        return (w_p0, w_pm, w_p1, w_pts)

    def insert_node_on_segment(
        self,
        piece_id: str,
        path_id: str,
        *,
        seg_idx: int,
        t: float,
    ) -> Optional[Node]:
        """指定パスの seg_idx セグメント上にパラメータ t で新ノードを挿入し、ノッチ・縫合を自動分割再マッピングする."""
        piece = self.pieces.get(piece_id)
        if not piece:
            return None
        path = piece.paths.get(path_id)
        if not path or len(path.nodes) < 2:
            return None

        nodes = path.nodes
        n_nodes = len(nodes)
        num_segs = n_nodes if path.closed else n_nodes - 1
        if seg_idx < 0 or seg_idx >= num_segs:
            return None

        t = max(0.01, min(0.99, float(t)))
        orig_start_node = nodes[seg_idx]
        orig_end_node = nodes[(seg_idx + 1) % n_nodes]
        orig_start_id = orig_start_node.id
        orig_end_id = orig_end_node.id

        ctrls = path.get_segment_controls(seg_idx)
        is_curved = (
            orig_start_node.kind != NodeKind.CORNER
            or orig_end_node.kind != NodeKind.CORNER
            or orig_start_node.out_handle is not None
            or orig_end_node.in_handle is not None
        )

        left_c, right_c = split_cubic_bezier(ctrls[0], ctrls[1], ctrls[2], ctrls[3], t)
        mid_pt = left_c[3]

        if not is_curved:
            new_node = Node(
                id=new_id(),
                position=mid_pt,
                kind=NodeKind.CORNER,
                in_handle=None,
                out_handle=None,
            )
            orig_start_node.out_handle = None
            orig_end_node.in_handle = None
        else:
            in_h = (left_c[2][0] - mid_pt[0], left_c[2][1] - mid_pt[1])
            out_h = (right_c[1][0] - mid_pt[0], right_c[1][1] - mid_pt[1])
            new_node = Node(
                id=new_id(),
                position=mid_pt,
                kind=NodeKind.SMOOTH,
                in_handle=in_h,
                out_handle=out_h,
            )
            if orig_start_node.kind == NodeKind.AUTO:
                orig_start_node.kind = NodeKind.SMOOTH
            if orig_end_node.kind == NodeKind.AUTO:
                orig_end_node.kind = NodeKind.SMOOTH

            orig_start_node.out_handle = (left_c[1][0] - ctrls[0][0], left_c[1][1] - ctrls[0][1])
            orig_end_node.in_handle = (right_c[2][0] - ctrls[3][0], right_c[2][1] - ctrls[3][1])

        nodes.insert(seg_idx + 1, new_node)

        # ノッチの再マッピング
        for notch in list(self.notches.values()):
            if (
                notch.piece_id == piece_id
                and notch.path_id == path_id
                and notch.segment_start_id == orig_start_id
            ):
                if abs(notch.t - t) <= 1e-4:
                    del self.notches[notch.id]
                elif notch.t < t:
                    notch.t = max(0.001, min(0.999, notch.t / t))
                else:
                    notch.segment_start_id = new_node.id
                    notch.t = max(0.001, min(0.999, (notch.t - t) / max(1e-4, 1.0 - t)))

        # 縫合の再マッピング (外周パス時)
        if path.role == PathRole.OUTER:
            for seam in self.seams.values():
                for side in ("a", "b"):
                    chains = seam.effective_a() if side == "a" else seam.effective_b()
                    new_chain: list[EdgeRange] = []
                    modified = False
                    for er in chains:
                        if er.piece_id != piece_id or er.path_id != path_id:
                            new_chain.append(er)
                            continue

                        # 順方向一致
                        if er.start_node_id == orig_start_id and er.end_node_id == orig_end_id:
                            p_t0 = max(0.0, min(1.0, float(er.t0)))
                            p_t1 = max(0.0, min(1.0, float(er.t1)))
                            is_rev = False
                        # 逆方向一致
                        elif er.start_node_id == orig_end_id and er.end_node_id == orig_start_id:
                            p_t0 = 1.0 - max(0.0, min(1.0, float(er.t0)))
                            p_t1 = 1.0 - max(0.0, min(1.0, float(er.t1)))
                            is_rev = True
                        else:
                            new_chain.append(er)
                            continue

                        modified = True
                        min_p = min(p_t0, p_t1)
                        max_p = max(p_t0, p_t1)

                        if max_p <= t + 1e-5:
                            # 左区間内
                            sub_t0 = p_t0 / t
                            sub_t1 = p_t1 / t
                            if not is_rev:
                                new_chain.append(EdgeRange(
                                    piece_id=piece_id,
                                    path_id=path_id,
                                    start_node_id=orig_start_id,
                                    end_node_id=new_node.id,
                                    t0=max(0.0, min(1.0, sub_t0)),
                                    t1=max(0.0, min(1.0, sub_t1)),
                                ))
                            else:
                                new_chain.append(EdgeRange(
                                    piece_id=piece_id,
                                    path_id=path_id,
                                    start_node_id=new_node.id,
                                    end_node_id=orig_start_id,
                                    t0=max(0.0, min(1.0, 1.0 - sub_t0)),
                                    t1=max(0.0, min(1.0, 1.0 - sub_t1)),
                                ))
                        elif min_p >= t - 1e-5:
                            # 右区間内
                            sub_t0 = (p_t0 - t) / (1.0 - t)
                            sub_t1 = (p_t1 - t) / (1.0 - t)
                            if not is_rev:
                                new_chain.append(EdgeRange(
                                    piece_id=piece_id,
                                    path_id=path_id,
                                    start_node_id=new_node.id,
                                    end_node_id=orig_end_id,
                                    t0=max(0.0, min(1.0, sub_t0)),
                                    t1=max(0.0, min(1.0, sub_t1)),
                                ))
                            else:
                                new_chain.append(EdgeRange(
                                    piece_id=piece_id,
                                    path_id=path_id,
                                    start_node_id=orig_end_id,
                                    end_node_id=new_node.id,
                                    t0=max(0.0, min(1.0, 1.0 - sub_t0)),
                                    t1=max(0.0, min(1.0, 1.0 - sub_t1)),
                                ))
                        else:
                            # t をまたぐ場合: 2つの区間に分割して展開
                            if p_t0 < p_t1:
                                # 順走
                                er_left = EdgeRange(
                                    piece_id=piece_id,
                                    path_id=path_id,
                                    start_node_id=orig_start_id,
                                    end_node_id=new_node.id,
                                    t0=max(0.0, min(1.0, p_t0 / t)),
                                    t1=1.0,
                                )
                                er_right = EdgeRange(
                                    piece_id=piece_id,
                                    path_id=path_id,
                                    start_node_id=new_node.id,
                                    end_node_id=orig_end_id,
                                    t0=0.0,
                                    t1=max(0.0, min(1.0, (p_t1 - t) / (1.0 - t))),
                                )
                                new_chain.extend([er_left, er_right])
                            else:
                                # 逆走
                                er_right = EdgeRange(
                                    piece_id=piece_id,
                                    path_id=path_id,
                                    start_node_id=new_node.id,
                                    end_node_id=orig_end_id,
                                    t0=(p_t0 - t) / (1.0 - t),
                                    t1=0.0,
                                )
                                er_left = EdgeRange(
                                    piece_id=piece_id,
                                    path_id=path_id,
                                    start_node_id=orig_start_id,
                                    end_node_id=new_node.id,
                                    t0=1.0,
                                    t1=p_t1 / t,
                                )
                                new_chain.extend([er_right, er_left])

                    if modified:
                        if side == "a":
                            seam.chain_a = new_chain
                        else:
                            seam.chain_b = new_chain
                        seam.sync_primary_from_chains()

        return new_node

    def get_seam_geometry(
        self,
        seam: Seam,
    ) -> tuple[Vec2, Vec2, Vec2, Vec2, Vec2, Vec2]:
        """縫合の A側 (始点, 中点, 終点) および B側 (対応始点, 対応中点, 対応終点) のワールド座標を返す."""
        edges_a = seam.chain_a if seam.chain_a else [seam.a]
        edges_b = seam.chain_b if seam.chain_b else [seam.b]

        w_a_start, _, _, _ = self.evaluate_edge_range_geometry(edges_a[0])
        _, _, w_a_end, _ = self.evaluate_edge_range_geometry(edges_a[-1])
        mid_idx = len(edges_a) // 2
        _, w_a_mid, _, _ = self.evaluate_edge_range_geometry(edges_a[mid_idx])

        w_b_start, _, _, _ = self.evaluate_edge_range_geometry(edges_b[0])
        _, _, w_b_end, _ = self.evaluate_edge_range_geometry(edges_b[-1])
        mid_idx_b = len(edges_b) // 2
        _, w_b_mid, _, _ = self.evaluate_edge_range_geometry(edges_b[mid_idx_b])

        if seam.reverse:
            b0 = w_b_end
            b1 = w_b_start
        else:
            b0 = w_b_start
            b1 = w_b_end

        return (w_a_start, w_a_mid, w_a_end, b0, w_b_mid, b1)

    def compute_seam_lengths(
        self,
        seam_id: str,
    ) -> tuple[float, float, float]:
        """縫合の両側の弧長と不一致率 (len_a, len_b, mismatch_ratio) を返す."""
        seam = self.seams.get(seam_id)
        if not seam:
            return (0.0, 0.0, 0.0)

        # chain があればそちらの総和、なければ a / b 単体
        if seam.chain_a:
            len_a = sum(self.compute_edge_range_length(er) for er in seam.chain_a)
        else:
            len_a = self.compute_edge_range_length(seam.a)

        if seam.chain_b:
            len_b = sum(self.compute_edge_range_length(er) for er in seam.chain_b)
        else:
            len_b = self.compute_edge_range_length(seam.b)

        max_len = max(len_a, len_b, 1e-9)
        mismatch_ratio = abs(len_a - len_b) / max_len
        return (len_a, len_b, mismatch_ratio)

    def to_dict(self) -> dict[str, Any]:
        """ドキュメント全体をシリアライズ可能な辞書形式に変換する."""
        data = asdict(self)
        # transform などをプレーン辞書化
        return data

    def clone(self) -> Document:
        """現在のドキュメントの完全なディープコピー（スナップショット）を返す."""
        return self.from_dict(self.to_dict())

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """JSON文字列として直列化する."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """辞書データから Document オブジェクトを復元する.
        
        バージョン整合性、必須フィールド、ID形式を検証する.
        """
        if not isinstance(data, dict):
            raise ValueError(f"Document data must be a dict, got {type(data)}")

        version = data.get("version")
        if version != DOCUMENT_VERSION:
            raise ValueError(f"Unsupported document version: {version} (expected {DOCUMENT_VERSION})")

        doc = cls(
            version=version,
            name=data.get("name", "Untitled Pattern"),
        )

        # pieces
        raw_pieces = data.get("pieces", {})
        for pid, pdata in raw_pieces.items():
            paths = {}
            for path_id, path_data in pdata.get("paths", {}).items():
                nodes = []
                for nd in path_data.get("nodes", []):
                    pos = tuple(nd["position"])
                    in_h = tuple(nd["in_handle"]) if nd.get("in_handle") else None
                    out_h = tuple(nd["out_handle"]) if nd.get("out_handle") else None
                    nodes.append(Node(
                        id=nd["id"],
                        position=pos,
                        kind=NodeKind(nd.get("kind", "CORNER")),
                        in_handle=in_h,
                        out_handle=out_h,
                    ))
                paths[path_id] = Path(
                    id=path_id,
                    role=PathRole(path_data.get("role", "OUTER")),
                    nodes=nodes,
                    closed=path_data.get("closed", True),
                )

            t_data = pdata.get("transform", {})
            aff = Affine2D(
                tx=t_data.get("tx", 0.0),
                ty=t_data.get("ty", 0.0),
                rotation_rad=t_data.get("rotation_rad", 0.0),
                sx=t_data.get("sx", 1.0),
                sy=t_data.get("sy", 1.0),
            )

            doc.pieces[pid] = Piece(
                id=pid,
                name=pdata.get("name", "Piece"),
                paths=paths,
                outer_path_id=pdata.get("outer_path_id", ""),
                transform=aff,
                target_edge_length=pdata.get("target_edge_length", 0.015),
                layer_id=pdata.get("layer_id", 0),
            )

        # seams
        raw_seams = data.get("seams", {})
        for sid, sdata in raw_seams.items():
            def make_edge_range(ed: dict) -> EdgeRange:
                return EdgeRange(
                    piece_id=ed["piece_id"],
                    path_id=ed["path_id"],
                    start_node_id=ed["start_node_id"],
                    end_node_id=ed["end_node_id"],
                    t0=ed.get("t0", 0.0),
                    t1=ed.get("t1", 1.0),
                )

            er_a = make_edge_range(sdata["a"])
            er_b = make_edge_range(sdata["b"])
            chain_a = [make_edge_range(x) for x in sdata.get("chain_a", [])]
            chain_b = [make_edge_range(x) for x in sdata.get("chain_b", [])]

            wps = [
                Waypoint(id=w["id"], position_3d=tuple(w["position_3d"]))
                for w in sdata.get("waypoints", [])
            ]

            doc.seams[sid] = Seam(
                id=sid,
                a=er_a,
                b=er_b,
                chain_a=chain_a,
                chain_b=chain_b,
                reverse=sdata.get("reverse", False),
                mode=SeamMode(sdata.get("mode", "SEGMENT")),
                shrink=sdata.get("shrink", 1.0),
                waypoints=wps,
            )

        # notches
        raw_notches = data.get("notches", {})
        for nid, ndata in raw_notches.items():
            doc.notches[nid] = Notch(
                id=nid,
                piece_id=ndata["piece_id"],
                path_id=ndata["path_id"],
                segment_start_id=ndata["segment_start_id"],
                t=ndata.get("t", 0.5),
                kind=NotchKind(ndata.get("kind", "V")),
                depth=ndata.get("depth", 0.005),
                width=ndata.get("width", 0.004),
            )

        # anchors
        raw_anchors = data.get("anchors", {})
        for aid, adata in raw_anchors.items():
            doc.anchors[aid] = Anchor(
                id=aid,
                piece_id=adata["piece_id"],
                position=tuple(adata["position"]),
                kind=AnchorKind(adata.get("kind", "PIN")),
                target_object=adata.get("target_object"),
                target_bone=adata.get("target_bone"),
                weight=adata.get("weight", 1.0),
            )

        # placements
        raw_placements = data.get("placements", {})
        for pid, pldata in raw_placements.items():
            doc.placements[pid] = Placement(
                piece_id=pid,
                offset_3d=tuple(pldata.get("offset_3d", (0.0, 0.0, 0.0))),
                euler_3d=tuple(pldata.get("euler_3d", (0.0, 0.0, 0.0))),
                preset=pldata.get("preset", "CHEST_FRONT"),
                target_bone=pldata.get("target_bone", ""),
                bone_t=float(pldata.get("bone_t", 0.5)),
                bone_angle_deg=float(pldata.get("bone_angle_deg", 0.0)),
                surface_offset=float(pldata.get("surface_offset", 0.08)),
                is_tubular=bool(pldata.get("is_tubular", False)),
            )

        # reference_image
        raw_ref = data.get("reference_image")
        if raw_ref and isinstance(raw_ref, dict):
            doc.reference_image = ReferenceImage(
                image_path=raw_ref.get("image_path", ""),
                offset=tuple(raw_ref.get("offset", (0.0, 0.0))),
                width=raw_ref.get("width", 1.0),
                opacity=raw_ref.get("opacity", 0.5),
                visible=raw_ref.get("visible", True),
                aspect_ratio=raw_ref.get("aspect_ratio", 1.0),
            )

        return doc

    @classmethod
    def from_json(cls, raw: str) -> Document:
        """JSON文字列から Document オブジェクトを復元する.
        
        NaN / Infinity 等の非有限値は ValueError で拒絶する.
        """
        def _reject_constant(val: str):
            raise ValueError(f"Invalid JSON constant found: '{val}'")

        try:
            data = json.loads(raw, parse_constant=_reject_constant)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse document JSON: {e}") from e

        return cls.from_dict(data)

    def save_to_file(self, filepath: str, *, indent: int = 2) -> None:
        """外部ファイル (.json) に UTF-8 でドキュメントを保存する."""
        import os
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=indent))

    @classmethod
    def load_from_file(cls, filepath: str) -> Document:
        """外部ファイル (.json) から UTF-8 でドキュメントを読み込む."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.from_json(content)

