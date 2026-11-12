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
    extract_sub_cubic,
    match_segment_shape,
    adjust_segment_arc_length,
    signed_area,
    ensure_ccw,
    has_self_intersection,
    Affine2D,
    compute_bounding_box,
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


class SeamRouting(str, Enum):
    """縫合エッジの3D空間ルーティング方式."""
    DIRECT = "DIRECT"
    BONE_WRAP = "BONE_WRAP"


def normalize_angle_deg(angle: float) -> float:
    """角度（度）の有限値を検証し [-180.0, 180.0] 範囲へ正規化する."""
    if not math.isfinite(angle):
        return 0.0
    norm = ((float(angle) + 180.0) % 360.0) - 180.0
    if norm == -180.0 and angle > 0.0:
        return 180.0
    return 0.0 if norm == 0.0 else norm


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

    def get_outer_path(self) -> Optional[Path]:
        """外周パス（OUTER）を取得する."""
        if self.outer_path_id and self.outer_path_id in self.paths:
            return self.paths[self.outer_path_id]
        for p in self.paths.values():
            if p.role == PathRole.OUTER:
                return p
        return None

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換する."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Piece:
        """辞書データから Piece オブジェクトを復元する."""
        paths = {}
        for path_id, path_data in data.get("paths", {}).items():
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

        t_data = data.get("transform", {})
        aff = Affine2D(
            tx=t_data.get("tx", 0.0),
            ty=t_data.get("ty", 0.0),
            rotation_rad=t_data.get("rotation_rad", 0.0),
            sx=t_data.get("sx", 1.0),
            sy=t_data.get("sy", 1.0),
        )

        return cls(
            id=data.get("id", ""),
            name=data.get("name", "Piece"),
            paths=paths,
            outer_path_id=data.get("outer_path_id", ""),
            transform=aff,
            target_edge_length=data.get("target_edge_length", 0.015),
            layer_id=data.get("layer_id", 0),
        )


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
    routing: SeamRouting = SeamRouting.DIRECT
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
    pattern_up_angle_deg: float = 0.0

    def copy_for_piece(self, piece_id: str, *, mirror_horizontal: bool = False) -> Placement:
        """配置パラメータを指定ピース向けに複製する（必要に応じ左右反転も適用）."""
        if not mirror_horizontal:
            return Placement(
                piece_id=piece_id,
                offset_3d=self.offset_3d,
                euler_3d=self.euler_3d,
                preset=self.preset,
                target_bone=self.target_bone,
                bone_t=self.bone_t,
                bone_angle_deg=self.bone_angle_deg,
                surface_offset=self.surface_offset,
                pattern_up_angle_deg=self.pattern_up_angle_deg,
            )

        from .placement import flip_bone_name_left_right
        new_preset = self.preset
        if new_preset == "ARM_LEFT":
            new_preset = "ARM_RIGHT"
        elif new_preset == "ARM_RIGHT":
            new_preset = "ARM_LEFT"

        return Placement(
            piece_id=piece_id,
            offset_3d=(-self.offset_3d[0], self.offset_3d[1], self.offset_3d[2]),
            euler_3d=(self.euler_3d[0], -self.euler_3d[1], -self.euler_3d[2]),
            preset=new_preset,
            target_bone=flip_bone_name_left_right(self.target_bone),
            bone_t=self.bone_t,
            bone_angle_deg=normalize_angle_deg(-self.bone_angle_deg),
            surface_offset=self.surface_offset,
            pattern_up_angle_deg=normalize_angle_deg(-self.pattern_up_angle_deg),
        )


@dataclass
class ReferenceImage:
    """キャンバス背景に表示する参考下絵画像."""
    image_path: str = ""
    offset: tuple[float, float] = (0.0, 0.0)
    width: float = 1.0  # ワールドメートル単位での表示幅
    opacity: float = 0.5
    visible: bool = True
    aspect_ratio: float = 1.0  # 幅に対する高さの比率 (height = width * aspect_ratio)



def order_edge_chain(path: Path, edge_ranges: list[EdgeRange]) -> Optional[list[EdgeRange]]:
    """指定されたEdgeRange群が同一パス上で1本の連続した一本道チェーンを形成しているか判定し、
    始点から終点へ順序正しく並べたリストを返す.
    
    連続していない（飛び飛び）場合や枝分かれしている場合は None を返す。
    """
    if not edge_ranges:
        return None
    if len(edge_ranges) == 1:
        res = path.resolve_edge_range(edge_ranges[0])
        return list(edge_ranges) if res is not None else None

    n_nodes = len(path.nodes)
    num_segs = n_nodes if path.closed else n_nodes - 1

    seg_to_er: dict[int, EdgeRange] = {}
    for er in edge_ranges:
        res = path.resolve_edge_range(er)
        if not res:
            return None
        s_idx, _, _ = res
        if s_idx in seg_to_er:
            return None
        seg_to_er[s_idx] = er

    k = len(seg_to_er)
    if k != len(edge_ranges):
        return None

    sorted_indices = sorted(seg_to_er.keys())
    # 1. 通常の連続ケース (例: [2, 3, 4])
    if sorted_indices[-1] - sorted_indices[0] == k - 1:
        return [seg_to_er[i] for i in sorted_indices]

    # 2. 閉ループの境界跨ぎケース (例: [0, 1, n_nodes-1])
    if path.closed:
        gaps = [i for i in range(k) if (sorted_indices[(i + 1) % k] - sorted_indices[i]) % num_segs != 1]
        if len(gaps) == 1:
            cut = (gaps[0] + 1) % k
            ordered_indices = sorted_indices[cut:] + sorted_indices[:cut]
            return [seg_to_er[i] for i in ordered_indices]

    return None


def extract_edge_chain_data(
    piece: Piece,
    path: Path,
    *,
    chain_ranges: list[EdgeRange],
) -> Optional[dict]:
    """連続したEdgeRangeチェーンから、正規化局所基底における各セグメントの制御点・ノード情報を抽出する."""
    if not chain_ranges:
        return None

    all_seg_wctrls: list[tuple[Vec2, Vec2, Vec2, Vec2]] = []
    seg_lengths: list[float] = []
    seg_node_kinds: list[tuple[NodeKind, NodeKind]] = []

    for er in chain_ranges:
        res = path.resolve_edge_range(er)
        if not res:
            return None
        seg_idx, t0, t1 = res
        ctrls = path.get_segment_controls(seg_idx)
        sub_c = extract_sub_cubic(ctrls[0], ctrls[1], ctrls[2], ctrls[3], t0=t0, t1=t1)
        w_ctrls = tuple(piece.transform.apply(p) for p in sub_c)
        table = build_arc_table(*w_ctrls)
        all_seg_wctrls.append(w_ctrls)
        seg_lengths.append(table.total_length)

        n0 = path.nodes[seg_idx]
        n1 = path.nodes[(seg_idx + 1) % len(path.nodes)]
        seg_node_kinds.append((n0.kind, n1.kind))

    total_len = sum(seg_lengths)

    w_start = all_seg_wctrls[0][0]
    w_end = all_seg_wctrls[-1][3]

    dx = w_end[0] - w_start[0]
    dy = w_end[1] - w_start[1]
    chord_len = math.hypot(dx, dy)

    if chord_len < 1e-9:
        ux, uy = 1.0, 0.0
        vx, vy = 0.0, 1.0
    else:
        ux = dx / chord_len
        uy = dy / chord_len
        vx = -uy
        vy = ux

    def to_norm(pt: tuple[float, float]) -> tuple[float, float]:
        px = pt[0] - w_start[0]
        py = pt[1] - w_start[1]
        denom = max(1e-9, chord_len)
        alpha = (px * ux + py * uy) / denom
        beta = (px * vx + py * vy) / denom
        return (alpha, beta)

    segments_data = []
    for i, w_ctrls in enumerate(all_seg_wctrls):
        c_norm = tuple(to_norm(p) for p in w_ctrls)
        segments_data.append({
            "controls_norm": c_norm,
            "length": seg_lengths[i],
            "node_start_kind": seg_node_kinds[i][0],
            "node_end_kind": seg_node_kinds[i][1],
        })

    return {
        "num_segments": len(chain_ranges),
        "total_length": total_len,
        "chord_vector": (dx, dy),
        "chord_length": chord_len,
        "segments": segments_data,
        "w_start": w_start,
        "w_end": w_end,
    }


def replace_path_segment_chain(
    path: Path,
    start_node_id: str,
    end_node_id: str,
    *,
    new_intermediate_nodes: list[Node],
) -> bool:
    """path上の start_node_id から end_node_id までの間にある中間ノード群を new_intermediate_nodes で置き換える.
    
    audit: allow-positional
    """
    n_nodes = len(path.nodes)
    if n_nodes < 2:
        return False

    i_start = next((i for i, nd in enumerate(path.nodes) if nd.id == start_node_id), None)
    i_end = next((i for i, nd in enumerate(path.nodes) if nd.id == end_node_id), None)
    if i_start is None or i_end is None:
        return False

    if i_start == i_end:
        return False

    if i_start < i_end:
        path.nodes = path.nodes[: i_start + 1] + new_intermediate_nodes + path.nodes[i_end :]
    else:
        path.nodes = path.nodes[i_end : i_start + 1] + new_intermediate_nodes

    return True


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

    def _symmetric_name(self, name: str, *, fallback_to_sym: bool = True) -> str:
        """左右対称名（_L -> _R, .L -> .R 等）を導出する."""
        for suf_l, suf_r in [("_L", "_R"), (".L", ".R"), ("_Left", "_Right"), ("-L", "-R")]:
            if name.endswith(suf_l):
                return self._unique_piece_name(name[:-len(suf_l)] + suf_r)
            elif name.endswith(suf_r):
                return self._unique_piece_name(name[:-len(suf_r)] + suf_l)
        if fallback_to_sym:
            return self._unique_piece_name(f"{name}_Sym")
        return name

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

    @staticmethod
    def _materialize_auto_node(path: Path, node_idx: int) -> None:
        """AUTOノードのハンドルを実体化し、CORNERノードに変換する."""
        nodes = path.nodes
        n_nodes = len(nodes)
        if n_nodes < 2 or node_idx < 0 or node_idx >= n_nodes:
            return
        node = nodes[node_idx]
        if node.kind != NodeKind.AUTO:
            return
        prev_idx = (node_idx - 1 + n_nodes) % n_nodes if path.closed else max(0, node_idx - 1)
        next_idx = (node_idx + 1) % n_nodes if path.closed else min(n_nodes - 1, node_idx + 1)
        in_h, out_h = compute_auto_handles(
            nodes[prev_idx].position,
            node.position,
            nodes[next_idx].position,
        )
        node.in_handle = in_h
        node.out_handle = out_h
        node.kind = NodeKind.CORNER

    def subdivide_edge(self, edge_range: EdgeRange, *, ratio: float = 0.5) -> Optional[Node]:
        """指定された辺範囲を実寸弧長比率 ratio（既定0.5）で中点分割し、新ノードを返す."""
        piece = self.pieces.get(edge_range.piece_id)
        if not piece:
            return None
        path = piece.paths.get(edge_range.path_id)
        if not path:
            return None
        res = path.resolve_edge_range(edge_range)
        if not res:
            return None
        seg_idx, p_t0, p_t1 = res
        ctrls = path.get_segment_controls(seg_idx)
        table = build_arc_table(*ctrls)
        if table.total_length < 0.002:
            return None

        s0 = arc_length_at_t(table, p_t0)
        s1 = arc_length_at_t(table, p_t1)
        clamped_r = max(0.05, min(0.95, float(ratio)))
        s_mid = s0 + clamped_r * (s1 - s0)
        _, split_t = point_at_arc_length(table, s_mid)
        return self.insert_node_on_segment(piece.id, path.id, seg_idx=seg_idx, t=split_t)

    def duplicate_piece(
        self,
        piece_id: str,
        *,
        offset: Optional[tuple[float, float]] = None,
    ) -> Optional[Piece]:
        """ピース単体を複製し、オフセット位置に配置して返す（ノッチ・内部パス・自己完結Seam・Placementも複製）."""
        orig = self.pieces.get(piece_id)
        if not orig:
            return None
        outer = orig.get_outer_path()
        if not outer:
            return None

        if offset is None:
            sample_pts = outer.sample_outline_points(steps_per_segment=16)
            w_pts = [orig.transform.apply(p) for p in sample_pts]
            min_pt, max_pt = compute_bounding_box(w_pts)
            w = max_pt[0] - min_pt[0]
            dx = max(0.05, w + 0.05)
            dy = 0.0
        else:
            dx, dy = float(offset[0]), float(offset[1])

        new_pid = new_id()
        base_name = orig.name
        new_name = self._unique_piece_name(f"{base_name}_Copy")

        new_transform = Affine2D(
            tx=orig.transform.tx + dx,
            ty=orig.transform.ty + dy,
            rotation_rad=orig.transform.rotation_rad,
            sx=orig.transform.sx,
            sy=orig.transform.sy,
        )

        new_paths: dict[str, Path] = {}
        path_map: dict[str, str] = {}
        node_map: dict[str, str] = {}
        for p in orig.paths.values():
            new_path_id = new_id()
            path_map[p.id] = new_path_id
            new_nodes: list[Node] = []
            for nd in p.nodes:
                new_nid = new_id()
                node_map[nd.id] = new_nid
                new_nodes.append(Node(
                    id=new_nid,
                    position=nd.position,
                    kind=nd.kind,
                    in_handle=nd.in_handle,
                    out_handle=nd.out_handle,
                ))
            new_paths[new_path_id] = Path(
                id=new_path_id,
                role=p.role,
                nodes=new_nodes,
                closed=p.closed,
            )

        new_outer_path_id = path_map.get(orig.outer_path_id, "")
        new_piece = Piece(
            id=new_pid,
            name=new_name,
            paths=new_paths,
            outer_path_id=new_outer_path_id,
            transform=new_transform,
            target_edge_length=orig.target_edge_length,
            layer_id=orig.layer_id,
        )
        self.pieces[new_pid] = new_piece

        # ノッチ複製
        for notch in list(self.notches.values()):
            if notch.piece_id == piece_id and notch.path_id in path_map:
                new_nid = new_id()
                self.notches[new_nid] = Notch(
                    id=new_nid,
                    piece_id=new_pid,
                    path_id=path_map[notch.path_id],
                    segment_start_id=node_map.get(notch.segment_start_id, notch.segment_start_id),
                    t=notch.t,
                    kind=notch.kind,
                    depth=notch.depth,
                    width=notch.width,
                )

        # アンカー複製
        for anchor in list(self.anchors.values()):
            if anchor.piece_id == piece_id:
                new_aid = new_id()
                self.anchors[new_aid] = Anchor(
                    id=new_aid,
                    piece_id=new_pid,
                    position=anchor.position,
                    kind=anchor.kind,
                    target_object=anchor.target_object,
                    target_bone=anchor.target_bone,
                    weight=anchor.weight,
                )

        # ピース内自己完結 Seam 複製（ダーツ等）
        for seam in list(self.seams.values()):
            if all(er.piece_id == piece_id for er in seam.iter_unique_edge_ranges()):
                def map_er(er: EdgeRange) -> EdgeRange:
                    return EdgeRange(
                        piece_id=new_pid,
                        path_id=path_map.get(er.path_id, er.path_id),
                        start_node_id=node_map.get(er.start_node_id, er.start_node_id),
                        end_node_id=node_map.get(er.end_node_id, er.end_node_id),
                        t0=er.t0,
                        t1=er.t1,
                    )
                new_sid = new_id()
                new_a = [map_er(er) for er in seam.chain_a] if seam.chain_a else map_er(seam.a)
                new_b = [map_er(er) for er in seam.chain_b] if seam.chain_b else map_er(seam.b)
                self.seams[new_sid] = Seam(
                    id=new_sid,
                    a=new_a if isinstance(new_a, EdgeRange) else new_a[0],
                    b=new_b if isinstance(new_b, EdgeRange) else new_b[0],
                    chain_a=new_a if isinstance(new_a, list) else [],
                    chain_b=new_b if isinstance(new_b, list) else [],
                    reverse=seam.reverse,
                    mode=seam.mode,
                    shrink=seam.shrink,
                    routing=seam.routing,
                    waypoints=[Waypoint(id=new_id(), position_3d=wp.position_3d) for wp in seam.waypoints],
                )

        # Placement 複製
        if piece_id in self.placements:
            self.placements[new_pid] = self.placements[piece_id].copy_for_piece(new_pid)

        return new_piece

    def flip_piece(
        self,
        piece_id: str,
        *,
        direction: str = "HORIZONTAL",
        flip_name: bool = True,
    ) -> bool:
        """ピースをローカルバウンディングボックス中心基準で反転する（閉パスはCCW維持のためノード順逆転）."""
        if direction not in ("HORIZONTAL", "VERTICAL"):
            raise ValueError(f"Unsupported flip direction: '{direction}'")
        piece = self.pieces.get(piece_id)
        if not piece:
            return False
        outer = piece.get_outer_path()
        if not outer:
            return False

        outer_pts = outer.sample_outline_points(steps_per_segment=16)
        min_pt, max_pt = compute_bounding_box(outer_pts)
        cx = 0.5 * (min_pt[0] + max_pt[0])
        cy = 0.5 * (min_pt[1] + max_pt[1])

        if direction == "HORIZONTAL":
            def reflect_pos(p: tuple[float, float]) -> tuple[float, float]:
                return (2.0 * cx - p[0], p[1])

            def reflect_vec(v: Optional[tuple[float, float]]) -> Optional[tuple[float, float]]:
                return (-v[0], v[1]) if v is not None else None
        else:
            def reflect_pos(p: tuple[float, float]) -> tuple[float, float]:
                return (p[0], 2.0 * cy - p[1])

            def reflect_vec(v: Optional[tuple[float, float]]) -> Optional[tuple[float, float]]:
                return (v[0], -v[1]) if v is not None else None

        for path in piece.paths.values():
            if path.closed:
                n = len(path.nodes)
                next_id_map = {path.nodes[i].id: path.nodes[(i + 1) % n].id for i in range(n)}
                rev_nodes: list[Node] = []
                for orig_n in reversed(path.nodes):
                    rev_nodes.append(Node(
                        id=orig_n.id,
                        position=reflect_pos(orig_n.position),
                        kind=orig_n.kind,
                        in_handle=reflect_vec(orig_n.out_handle),
                        out_handle=reflect_vec(orig_n.in_handle),
                    ))
                path.nodes = rev_nodes
                # ノッチ更新
                for notch in self.notches.values():
                    if notch.piece_id == piece_id and notch.path_id == path.id:
                        if notch.segment_start_id in next_id_map:
                            notch.segment_start_id = next_id_map[notch.segment_start_id]
                            notch.t = max(0.001, min(0.999, 1.0 - float(notch.t)))
            else:
                for nd in path.nodes:
                    nd.position = reflect_pos(nd.position)
                    nd.in_handle = reflect_vec(nd.in_handle)
                    nd.out_handle = reflect_vec(nd.out_handle)

        for a in self.anchors.values():
            if a.piece_id == piece_id:
                a.position = reflect_pos(a.position)

        if direction == "HORIZONTAL":
            if piece_id in self.placements:
                self.placements[piece_id] = self.placements[piece_id].copy_for_piece(
                    piece_id,
                    mirror_horizontal=True,
                )

            if flip_name:
                piece.name = self._symmetric_name(piece.name, fallback_to_sym=False)

        return True

    def unfold_piece_along_edge(self, edge_range: EdgeRange) -> bool:
        """指定された外周辺を対称軸として単一ピースに対称展開（わ展開）する."""
        piece = self.pieces.get(edge_range.piece_id)
        if not piece:
            return False
        outer = piece.get_outer_path()
        if not outer or edge_range.path_id != outer.id:
            return False

        nodes = outer.nodes
        n_orig = len(nodes)
        if n_orig < 3:
            return False

        res = outer.resolve_edge_range(edge_range)
        if not res:
            return False
        seg_idx, _, _ = res
        i0 = seg_idx
        i1 = (seg_idx + 1) % n_orig

        p_a = nodes[i1].position
        p_b = nodes[i0].position
        vx = p_b[0] - p_a[0]
        vy = p_b[1] - p_a[1]
        v_len = math.hypot(vx, vy)
        if v_len < 0.001:
            return False

        nx = -vy / v_len
        ny = vx / v_len

        def reflect_point(p: tuple[float, float]) -> tuple[float, float]:
            wx = p[0] - p_a[0]
            wy = p[1] - p_a[1]
            d = wx * nx + wy * ny
            return (p[0] - 2.0 * d * nx, p[1] - 2.0 * d * ny)

        def reflect_vec(v: Optional[tuple[float, float]]) -> Optional[tuple[float, float]]:
            if v is None:
                return None
            d = v[0] * nx + v[1] * ny
            return (v[0] - 2.0 * d * nx, v[1] - 2.0 * d * ny)

        self._materialize_auto_node(outer, i0)
        self._materialize_auto_node(outer, i1)

        unfolded_nodes: list[Node] = []
        orig_idx_to_new_id: dict[int, str] = {}
        mirror_idx_to_new_id: dict[int, str] = {}

        curr = i1
        while True:
            orig_n = nodes[curr]
            nid = new_id()
            orig_idx_to_new_id[curr] = nid
            in_h = orig_n.in_handle
            out_h = orig_n.out_handle
            kind = orig_n.kind
            if curr == i1:
                in_h = reflect_vec(orig_n.out_handle)
                kind = NodeKind.CORNER
            elif curr == i0:
                out_h = reflect_vec(orig_n.in_handle)
                kind = NodeKind.CORNER
            unfolded_nodes.append(Node(
                id=nid,
                position=orig_n.position,
                kind=kind,
                in_handle=in_h,
                out_handle=out_h,
            ))
            if curr == i0:
                break
            curr = (curr + 1) % n_orig

        mirror_idx_to_new_id[i0] = orig_idx_to_new_id[i0]
        mirror_idx_to_new_id[i1] = orig_idx_to_new_id[i1]

        curr = (i0 - 1 + n_orig) % n_orig
        while curr != i1:
            orig_n = nodes[curr]
            nid = new_id()
            mirror_idx_to_new_id[curr] = nid
            unfolded_nodes.append(Node(
                id=nid,
                position=reflect_point(orig_n.position),
                kind=orig_n.kind,
                in_handle=reflect_vec(orig_n.out_handle),
                out_handle=reflect_vec(orig_n.in_handle),
            ))
            curr = (curr - 1 + n_orig) % n_orig

        outer.nodes = unfolded_nodes

        old_notches = [
            n for n in list(self.notches.values())
            if n.piece_id == piece.id and n.path_id == outer.id
        ]
        for n in old_notches:
            del self.notches[n.id]

        for n in old_notches:
            seg_i = next((i for i, nd in enumerate(nodes) if nd.id == n.segment_start_id), None)
            if seg_i is not None and seg_i != seg_idx:
                nid_fwd = new_id()
                self.notches[nid_fwd] = Notch(
                    id=nid_fwd,
                    piece_id=piece.id,
                    path_id=outer.id,
                    segment_start_id=orig_idx_to_new_id[seg_i],
                    t=n.t,
                    kind=n.kind,
                    depth=n.depth,
                    width=n.width,
                )
                nid_mir = new_id()
                seg_end_i = (seg_i + 1) % n_orig
                mir_start_id = mirror_idx_to_new_id[seg_end_i]
                self.notches[nid_mir] = Notch(
                    id=nid_mir,
                    piece_id=piece.id,
                    path_id=outer.id,
                    segment_start_id=mir_start_id,
                    t=1.0 - n.t,
                    kind=n.kind,
                    depth=n.depth,
                    width=n.width,
                )

        # 対称軸を参照する Seam の削除およびノードID付け替え
        for sid, s in list(self.seams.items()):
            involved = False
            for er in s.iter_unique_edge_ranges():
                if er.piece_id == piece.id and er.path_id == outer.id:
                    if (er.start_node_id == nodes[i0].id and er.end_node_id == nodes[i1].id) or (
                        er.start_node_id == nodes[i1].id and er.end_node_id == nodes[i0].id
                    ):
                        del self.seams[sid]
                        involved = True
                        break
            if not involved:
                def remap_er(er: EdgeRange) -> EdgeRange:
                    if er.piece_id == piece.id and er.path_id == outer.id:
                        s_idx = next((i for i, nd in enumerate(nodes) if nd.id == er.start_node_id), None)
                        e_idx = next((i for i, nd in enumerate(nodes) if nd.id == er.end_node_id), None)
                        if s_idx is not None and e_idx is not None and s_idx in orig_idx_to_new_id and e_idx in orig_idx_to_new_id:
                            return EdgeRange(
                                piece_id=er.piece_id,
                                path_id=er.path_id,
                                start_node_id=orig_idx_to_new_id[s_idx],
                                end_node_id=orig_idx_to_new_id[e_idx],
                                t0=er.t0,
                                t1=er.t1,
                            )
                    return er
                if s.chain_a:
                    s.chain_a = [remap_er(er) for er in s.chain_a]
                s.a = remap_er(s.a)
                if s.chain_b:
                    s.chain_b = [remap_er(er) for er in s.chain_b]
                s.b = remap_er(s.b)

        # 内部パス・穴・アンカーの鏡像追加
        def dist_to_axis(pt: tuple[float, float]) -> float:
            wx = pt[0] - p_a[0]
            wy = pt[1] - p_a[1]
            return abs(wx * nx + wy * ny)

        for p in list(piece.paths.values()):
            if p.id == outer.id:
                continue
            avg_d = sum(dist_to_axis(nd.position) for nd in p.nodes) / max(1, len(p.nodes))
            if avg_d > 0.001:
                mir_pid = new_id()
                mir_nodes: list[Node] = []
                if p.closed:
                    for nd in reversed(p.nodes):
                        mir_nodes.append(Node(
                            id=new_id(),
                            position=reflect_point(nd.position),
                            kind=nd.kind,
                            in_handle=reflect_vec(nd.out_handle),
                            out_handle=reflect_vec(nd.in_handle),
                        ))
                else:
                    for nd in p.nodes:
                        mir_nodes.append(Node(
                            id=new_id(),
                            position=reflect_point(nd.position),
                            kind=nd.kind,
                            in_handle=reflect_vec(nd.in_handle),
                            out_handle=reflect_vec(nd.out_handle),
                        ))
                piece.paths[mir_pid] = Path(id=mir_pid, role=p.role, nodes=mir_nodes, closed=p.closed)

        for a in list(self.anchors.values()):
            if a.piece_id == piece.id and dist_to_axis(a.position) > 0.001:
                mir_aid = new_id()
                self.anchors[mir_aid] = Anchor(
                    id=mir_aid,
                    piece_id=piece.id,
                    position=reflect_point(a.position),
                    kind=a.kind,
                    target_object=a.target_object,
                    target_bone=a.target_bone,
                    weight=a.weight,
                )

        return True

    def mirror_duplicate_across_edge(
        self,
        edge_range: EdgeRange,
        *,
        create_seam: bool = True,
    ) -> Optional[Piece]:
        """指定された外周辺を軸に対称な別ピースを生成し、外側へ5cm離して配置する（create_seam=Trueで自動縫合）."""
        piece = self.pieces.get(edge_range.piece_id)
        if not piece:
            return None
        outer = piece.get_outer_path()
        if not outer or edge_range.path_id != outer.id:
            return None

        nodes = outer.nodes
        n_orig = len(nodes)
        if n_orig < 2:
            return None

        res = outer.resolve_edge_range(edge_range)
        if not res:
            return None
        seg_idx, _, _ = res
        i0 = seg_idx
        i1 = (seg_idx + 1) % n_orig
        n0 = nodes[i0]
        n1 = nodes[i1]
        p0 = n0.position
        p1 = n1.position
        vx = p1[0] - p0[0]
        vy = p1[1] - p0[1]
        seg_len = math.hypot(vx, vy)
        if seg_len < 0.001:
            return None

        nx = vy / seg_len
        ny = -vx / seg_len

        def reflect_point(p: tuple[float, float]) -> tuple[float, float]:
            wx = p[0] - p0[0]
            wy = p[1] - p0[1]
            d = wx * nx + wy * ny
            return (p[0] - 2.0 * d * nx, p[1] - 2.0 * d * ny)

        def reflect_vec(v: Optional[tuple[float, float]]) -> Optional[tuple[float, float]]:
            if v is None:
                return None
            d = v[0] * nx + v[1] * ny
            return (v[0] - 2.0 * d * nx, v[1] - 2.0 * d * ny)

        new_pid = new_id()
        new_name = self._symmetric_name(piece.name)

        path_map: dict[str, str] = {}
        node_map: dict[str, str] = {}
        new_paths: dict[str, Path] = {}
        for p in piece.paths.values():
            new_path_id = new_id()
            path_map[p.id] = new_path_id
            new_nodes: list[Node] = []
            if p.closed:
                n = len(p.nodes)
                next_id_map = {p.nodes[i].id: p.nodes[(i + 1) % n].id for i in range(n)}
                for orig_n in reversed(p.nodes):
                    nid = new_id()
                    node_map[orig_n.id] = nid
                    new_nodes.append(Node(
                        id=nid,
                        position=reflect_point(orig_n.position),
                        kind=orig_n.kind,
                        in_handle=reflect_vec(orig_n.out_handle),
                        out_handle=reflect_vec(orig_n.in_handle),
                    ))
                new_paths[new_path_id] = Path(id=new_path_id, role=p.role, nodes=new_nodes, closed=True)
                for notch in list(self.notches.values()):
                    if notch.piece_id == piece.id and notch.path_id == p.id:
                        mapped_start = node_map.get(next_id_map.get(notch.segment_start_id, ""))
                        if mapped_start:
                            new_nid = new_id()
                            self.notches[new_nid] = Notch(
                                id=new_nid,
                                piece_id=new_pid,
                                path_id=new_path_id,
                                segment_start_id=mapped_start,
                                t=max(0.001, min(0.999, 1.0 - float(notch.t))),
                                kind=notch.kind,
                                depth=notch.depth,
                                width=notch.width,
                            )
            else:
                for orig_n in p.nodes:
                    nid = new_id()
                    node_map[orig_n.id] = nid
                    new_nodes.append(Node(
                        id=nid,
                        position=reflect_point(orig_n.position),
                        kind=orig_n.kind,
                        in_handle=reflect_vec(orig_n.in_handle),
                        out_handle=reflect_vec(orig_n.out_handle),
                    ))
                new_paths[new_path_id] = Path(id=new_path_id, role=p.role, nodes=new_nodes, closed=False)

        from .placement import flip_bone_name_left_right
        for a in list(self.anchors.values()):
            if a.piece_id == piece.id:
                new_aid = new_id()
                self.anchors[new_aid] = Anchor(
                    id=new_aid,
                    piece_id=new_pid,
                    position=reflect_point(a.position),
                    kind=a.kind,
                    target_object=a.target_object,
                    target_bone=flip_bone_name_left_right(a.target_bone) if a.target_bone else "",
                    weight=a.weight,
                )

        rot = piece.transform.rotation_rad
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        wnx = cos_r * nx - sin_r * ny
        wny = sin_r * nx + cos_r * ny
        new_transform = Affine2D(
            tx=piece.transform.tx + wnx * 0.05,
            ty=piece.transform.ty + wny * 0.05,
            rotation_rad=piece.transform.rotation_rad,
            sx=piece.transform.sx,
            sy=piece.transform.sy,
        )
        new_outer_path_id = path_map.get(piece.outer_path_id, "")
        new_piece = Piece(
            id=new_pid,
            name=new_name,
            paths=new_paths,
            outer_path_id=new_outer_path_id,
            transform=new_transform,
            target_edge_length=piece.target_edge_length,
            layer_id=piece.layer_id,
        )
        self.pieces[new_pid] = new_piece

        # ピース内自己完結 Seam（ダーツ等）の鏡像複製
        for seam in list(self.seams.values()):
            if all(er.piece_id == piece.id for er in seam.iter_unique_edge_ranges()):
                def map_mirror_er(er: EdgeRange) -> EdgeRange:
                    orig_p = piece.paths.get(er.path_id)
                    new_path_id = path_map.get(er.path_id, er.path_id)
                    if orig_p and orig_p.closed:
                        return EdgeRange(
                            piece_id=new_pid,
                            path_id=new_path_id,
                            start_node_id=node_map.get(er.end_node_id, er.end_node_id),
                            end_node_id=node_map.get(er.start_node_id, er.start_node_id),
                            t0=1.0 - er.t1,
                            t1=1.0 - er.t0,
                        )
                    return EdgeRange(
                        piece_id=new_pid,
                        path_id=new_path_id,
                        start_node_id=node_map.get(er.start_node_id, er.start_node_id),
                        end_node_id=node_map.get(er.end_node_id, er.end_node_id),
                        t0=er.t0,
                        t1=er.t1,
                    )

                new_sid = new_id()
                new_chain_a = [map_mirror_er(er) for er in seam.chain_a] if seam.chain_a else []
                new_chain_b = [map_mirror_er(er) for er in seam.chain_b] if seam.chain_b else []
                if len(new_chain_a) > 1:
                    orig_pa = piece.paths.get(seam.a.path_id)
                    if orig_pa and orig_pa.closed:
                        new_chain_a.reverse()
                if len(new_chain_b) > 1:
                    orig_pb = piece.paths.get(seam.b.path_id)
                    if orig_pb and orig_pb.closed:
                        new_chain_b.reverse()
                new_a = new_chain_a[0] if new_chain_a else map_mirror_er(seam.a)
                new_b = new_chain_b[0] if new_chain_b else map_mirror_er(seam.b)
                self.seams[new_sid] = Seam(
                    id=new_sid,
                    a=new_a,
                    b=new_b,
                    chain_a=new_chain_a,
                    chain_b=new_chain_b,
                    reverse=seam.reverse,
                    mode=seam.mode,
                    shrink=seam.shrink,
                    routing=seam.routing,
                    waypoints=[Waypoint(id=new_id(), position_3d=wp.position_3d) for wp in seam.waypoints],
                )

        if piece.id in self.placements:
            self.placements[new_pid] = self.placements[piece.id].copy_for_piece(
                new_pid,
                mirror_horizontal=True,
            )

        if create_seam:
            sym_outer_id = path_map[outer.id]
            er_a = EdgeRange(piece_id=piece.id, path_id=outer.id, start_node_id=n0.id, end_node_id=n1.id)
            er_b = EdgeRange(piece_id=new_pid, path_id=sym_outer_id, start_node_id=node_map[n1.id], end_node_id=node_map[n0.id])
            seam_id = new_id()
            self.seams[seam_id] = Seam(
                id=seam_id,
                a=er_a,
                b=er_b,
                reverse=True,
                mode=SeamMode.SEGMENT,
                shrink=1.0,
            )

        return new_piece

    def match_edge_shape(
        self,
        source_er: EdgeRange,
        target_er: EdgeRange,
        *,
        mode: str = "SIMILAR",
        reverse: bool = True,
        flip_curvature: bool = False,
        ease_offset: float = 0.0,
        target_length_override: Optional[float] = None,
    ) -> bool:
        """参照元辺 source_er のベジェ形状・弧長を対象辺 target_er へ転写する."""
        src_piece = self.pieces.get(source_er.piece_id)
        if not src_piece:
            return False
        src_path = src_piece.paths.get(source_er.path_id)
        if not src_path:
            return False
        chain_data = extract_edge_chain_data(src_piece, src_path, chain_ranges=[source_er])
        if not chain_data:
            return False
        eff_ease = ease_offset
        if target_length_override is not None:
            eff_ease = (target_length_override - chain_data["total_length"]) + ease_offset
        res = self.match_edge_chain_shape(
            [target_er],
            chain_data,
            mode=mode,
            reverse=reverse,
            flip_curvature=flip_curvature,
            apply_endpoint_offset=False,
            ease_offset=eff_ease,
        )
        return bool(res)

    def match_edge_chain_shape(
        self,
        dst_ranges: list[EdgeRange],
        src_chain_data: dict,
        *,
        mode: str = "SIMILAR",
        reverse: bool = False,
        flip_curvature: bool = False,
        apply_endpoint_offset: bool = False,
        ease_offset: float = 0.0,
    ) -> Optional[list[EdgeRange]]:
        """クリップボードに保存されたエッジチェーン形状を、対象のdst_rangesチェーンへ反映する."""
        if not dst_ranges or not src_chain_data:
            return None

        first_dst = dst_ranges[0]
        dst_piece = self.pieces.get(first_dst.piece_id)
        if not dst_piece:
            return None
        dst_path = dst_piece.paths.get(first_dst.path_id)
        if not dst_path:
            return None

        ordered_dst = order_edge_chain(dst_path, dst_ranges)
        if not ordered_dst:
            return None

        first_res = dst_path.resolve_edge_range(ordered_dst[0])
        last_res = dst_path.resolve_edge_range(ordered_dst[-1])
        if not first_res or not last_res:
            return None

        n_dst_nodes = len(dst_path.nodes)
        start_seg_idx = first_res[0]
        last_seg_idx = last_res[0]
        end_node_idx = (last_seg_idx + 1) % n_dst_nodes

        # 隣接する非対象セグメントの AUTO_SMOOTH ハンドルを事前に実体化して保護
        self._materialize_auto_node(dst_path, start_seg_idx)
        self._materialize_auto_node(dst_path, end_node_idx)

        start_node = dst_path.nodes[start_seg_idx]
        end_node = dst_path.nodes[end_node_idx]

        # 始点・終点のワールド座標
        w_start = dst_piece.transform.apply(start_node.position)

        if apply_endpoint_offset:
            src_vec = src_chain_data["chord_vector"]
            eff_vec = (-src_vec[0], -src_vec[1]) if reverse else src_vec
            w_end = (w_start[0] + eff_vec[0], w_start[1] + eff_vec[1])
            end_node.position = dst_piece.transform.inverse_apply(w_end)
        else:
            w_end = dst_piece.transform.apply(end_node.position)
            if mode == "LENGTH_MOVE_END":
                src_total_len = max(1e-6, float(src_chain_data["total_length"]))
                src_chord_len = float(src_chain_data["chord_length"])
                goal_len = max(0.001, src_total_len + ease_offset)
                cur_dx = w_end[0] - w_start[0]
                cur_dy = w_end[1] - w_start[1]
                cur_chord = math.hypot(cur_dx, cur_dy)
                if cur_chord > 1e-9 and src_chord_len > 1e-9:
                    target_chord = goal_len * (src_chord_len / src_total_len)
                    scale = target_chord / cur_chord
                    new_w_end = (w_start[0] + cur_dx * scale, w_start[1] + cur_dy * scale)
                    end_node.position = dst_piece.transform.inverse_apply(new_w_end)
                    w_end = new_w_end

        # ペースト先局所基底
        dx = w_end[0] - w_start[0]
        dy = w_end[1] - w_start[1]
        chord_len = math.hypot(dx, dy)
        if chord_len < 1e-9:
            ux, uy = 1.0, 0.0
            vx, vy = 0.0, 1.0
        else:
            ux = dx / chord_len
            uy = dy / chord_len
            vx = -uy
            vy = ux

        src_segments = list(src_chain_data["segments"])
        if reverse:
            src_segments.reverse()

        k = len(src_segments)
        restored_w_ctrls: list[tuple[Vec2, Vec2, Vec2, Vec2]] = []

        for seg in src_segments:
            c_norm = list(seg["controls_norm"])
            if reverse:
                c_norm = [c_norm[3], c_norm[2], c_norm[1], c_norm[0]]
                c_norm = [(1.0 - a, -b) for a, b in c_norm]
            if flip_curvature:
                c_norm = [(a, -b) for a, b in c_norm]

            w_pts = []
            for a, b in c_norm:
                wx = w_start[0] + a * dx + b * chord_len * vx
                wy = w_start[1] + a * dy + b * chord_len * vy
                w_pts.append((wx, wy))
            restored_w_ctrls.append((w_pts[0], w_pts[1], w_pts[2], w_pts[3]))

        # 中間ノード群の生成
        new_intermediate_nodes: list[Node] = []
        for i in range(k - 1):
            w_mid = restored_w_ctrls[i][3]
            l_mid = dst_piece.transform.inverse_apply(w_mid)
            orig_kind = src_segments[i]["node_end_kind"] if not reverse else src_segments[i]["node_start_kind"]
            new_node = Node(
                id=new_id(),
                position=l_mid,
                kind=orig_kind,
            )
            new_intermediate_nodes.append(new_node)

        before_node_ids = {nd.id for nd in dst_path.nodes}

        # パストポロジーを置換
        if not replace_path_segment_chain(
            dst_path,
            start_node.id,
            end_node.id,
            new_intermediate_nodes=new_intermediate_nodes,
        ):
            return None

        after_node_ids = {nd.id for nd in dst_path.nodes}
        removed_node_ids = before_node_ids - after_node_ids
        if removed_node_ids:
            for nid, notch in list(self.notches.items()):
                if notch.piece_id == dst_piece.id and notch.path_id == dst_path.id:
                    if notch.segment_start_id in removed_node_ids:
                        del self.notches[nid]

        # 全ノードのイン/アウトハンドルを設定
        chain_nodes = [start_node] + new_intermediate_nodes + [end_node]
        for i in range(k):
            wc0, wc1, wc2, wc3 = restored_w_ctrls[i]

            if mode == "LENGTH_ADJUST_BULGE" or (mode == "SIMILAR" and abs(ease_offset) > 1e-6):
                seg_orig_len = float(src_segments[i]["length"])
                weight = seg_orig_len / max(1e-6, float(src_chain_data["total_length"]))
                goal_seg_len = max(0.001, seg_orig_len + ease_offset * weight)
                seg_chord = math.hypot(wc3[0] - wc0[0], wc3[1] - wc0[1])
                # ターゲット弦長が目標弧長より短く膨らませられる場合に調整（目標長が弦長以下の場合は相似カーブを維持）
                if goal_seg_len > seg_chord + 1e-6:
                    wc0, wc1, wc2, wc3 = adjust_segment_arc_length(
                        wc0,
                        wc1,
                        wc2,
                        wc3,
                        target_length=goal_seg_len,
                        mode="LENGTH_ADJUST_BULGE",
                        flip_curvature=flip_curvature,
                    )

            lc0 = dst_piece.transform.inverse_apply(wc0)
            lc1 = dst_piece.transform.inverse_apply(wc1)
            lc2 = dst_piece.transform.inverse_apply(wc2)
            lc3 = dst_piece.transform.inverse_apply(wc3)

            nd_a = chain_nodes[i]
            nd_b = chain_nodes[i + 1]

            vx_s = lc3[0] - lc0[0]
            vy_s = lc3[1] - lc0[1]
            b1 = (lc0[0] + vx_s / 3.0, lc0[1] + vy_s / 3.0)
            b2 = (lc0[0] + 2.0 * vx_s / 3.0, lc0[1] + 2.0 * vy_s / 3.0)
            is_linear = (
                math.hypot(lc1[0] - b1[0], lc1[1] - b1[1]) < 1e-6
                and math.hypot(lc2[0] - b2[0], lc2[1] - b2[1]) < 1e-6
            )

            if is_linear:
                nd_a.out_handle = None
                nd_b.in_handle = None
            else:
                nd_a.out_handle = (lc1[0] - lc0[0], lc1[1] - lc0[1])
                nd_b.in_handle = (lc2[0] - lc3[0], lc2[1] - lc3[1])
                if i == 0:
                    nd_a.kind = NodeKind.CORNER
                if i == k - 1:
                    nd_b.kind = NodeKind.CORNER

        new_edge_ranges = []
        for i in range(k):
            er = EdgeRange(
                piece_id=dst_piece.id,
                path_id=dst_path.id,
                start_node_id=chain_nodes[i].id,
                end_node_id=chain_nodes[i + 1].id,
                t0=0.0,
                t1=1.0,
            )
            new_edge_ranges.append(er)

        return new_edge_ranges

    def match_seam_shape_and_length(
        self,
        seam_id: str,
        *,
        source_side: str = "A",
        mode: str = "LENGTH_ADJUST_BULGE",
        flip_curvature: bool = False,
        apply_endpoint_offset: bool = False,
        ease_value: float = 0.0,
        ease_is_ratio: bool = False,
    ) -> bool:
        """Seam の片側（AまたはB）の形状と弧長をもう片側へ反映する."""
        seam = self.seams.get(seam_id)
        if not seam:
            return False

        a_ranges = seam.effective_a()
        b_ranges = seam.effective_b()
        if not a_ranges or not b_ranges:
            return False

        if source_side == "A":
            eff_src = a_ranges
            eff_dst = b_ranges
        else:
            eff_src = b_ranges
            eff_dst = a_ranges

        src_er = eff_src[0]
        dst_er = eff_dst[0]
        src_piece = self.pieces.get(src_er.piece_id)
        dst_piece = self.pieces.get(dst_er.piece_id)
        if not src_piece or not dst_piece:
            return False
        src_path = src_piece.paths.get(src_er.path_id)
        dst_path = dst_piece.paths.get(dst_er.path_id)
        if not src_path or not dst_path:
            return False

        ordered_src = order_edge_chain(src_path, eff_src)
        ordered_dst = order_edge_chain(dst_path, eff_dst)
        if not ordered_src or not ordered_dst:
            return False

        src_chain_data = extract_edge_chain_data(src_piece, src_path, chain_ranges=ordered_src)
        if not src_chain_data:
            return False

        total_src_len = src_chain_data["total_length"]
        total_ease_m = (total_src_len * ease_value) if ease_is_ratio else ease_value

        new_dst_ranges = self.match_edge_chain_shape(
            ordered_dst,
            src_chain_data,
            mode=mode,
            reverse=seam.reverse,
            flip_curvature=flip_curvature,
            apply_endpoint_offset=apply_endpoint_offset,
            ease_offset=total_ease_m,
        )

        if new_dst_ranges:
            if source_side == "A":
                seam.chain_b = new_dst_ranges
                seam.b = new_dst_ranges[0]
            else:
                seam.chain_a = new_dst_ranges
                seam.a = new_dst_ranges[0]
            if len(new_dst_ranges) > 1 or (len(seam.chain_a) > 1 or len(seam.chain_b) > 1):
                seam.mode = SeamMode.CHAIN
            seam.sync_primary_from_chains()
            return True

        return False

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
            piece = Piece.from_dict(pdata)
            piece.id = pid
            doc.pieces[pid] = piece

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

            raw_routing = sdata.get("routing", SeamRouting.DIRECT.value)
            try:
                routing = SeamRouting(raw_routing)
            except ValueError:
                routing = SeamRouting.DIRECT

            doc.seams[sid] = Seam(
                id=sid,
                a=er_a,
                b=er_b,
                chain_a=chain_a,
                chain_b=chain_b,
                reverse=sdata.get("reverse", False),
                mode=SeamMode(sdata.get("mode", "SEGMENT")),
                shrink=sdata.get("shrink", 1.0),
                routing=routing,
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
                bone_angle_deg=normalize_angle_deg(float(pldata.get("bone_angle_deg", 0.0))),
                surface_offset=float(pldata.get("surface_offset", 0.08)),
                pattern_up_angle_deg=normalize_angle_deg(float(pldata.get("pattern_up_angle_deg", 0.0))),
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

