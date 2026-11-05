# -*- coding: utf-8 -*-
"""型紙エディタのコア文書データモデル (Single Source of Truth, Blender非依存)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional, Sequence, Tuple

from .ids import new_id, is_valid_id
from .geometry import (
    ArcLengthTable,
    Vec2,
    make_segment_controls,
    compute_auto_handles,
    build_arc_table,
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

    def add_polygon_piece(
        self,
        points: Sequence[Vec2],
        *,
        name: Optional[str] = None,
    ) -> Piece:
        """点列から反時計回り (CCW) 多角形ピースを作成し、文書に追加する.
        
        頂点数が3未満または自己交差が存在する場合は ValueError を送出する.
        """
        pts = [(float(p[0]), float(p[1])) for p in points]
        if len(pts) < 3:
            raise ValueError(f"Polygon must have at least 3 vertices, got {len(pts)}")

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
        """パスの各区間 (start_node_id, end_node_id, ArcLengthTable) を評価して返す.
        
        NodeKind.AUTO のノードは前後のノード位置からハンドルを動的に算出する.
        """
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

        # AUTOノード用ハンドルの事前計算
        effective_handles: list[tuple[Optional[Vec2], Optional[Vec2]]] = []
        for i, node in enumerate(nodes):
            if node.kind == NodeKind.AUTO:
                prev_node = nodes[(i - 1) % n_nodes] if path.closed else nodes[max(0, i - 1)]
                next_node = nodes[(i + 1) % n_nodes] if path.closed else nodes[min(n_nodes - 1, i + 1)]
                in_h, out_h = compute_auto_handles(prev_node.position, node.position, next_node.position)
                effective_handles.append((in_h, out_h))
            else:
                effective_handles.append((node.in_handle, node.out_handle))

        for i in range(num_segs):
            curr_node = nodes[i]
            next_node = nodes[(i + 1) % n_nodes]

            out_h = effective_handles[i][1]
            in_h = effective_handles[(i + 1) % n_nodes][0]

            ctrls = make_segment_controls(curr_node.position, out_h, in_h, next_node.position)
            table = build_arc_table(*ctrls)
            segments.append((curr_node.id, next_node.id, table))

        return segments

    def compute_edge_range_length(self, edge_range: EdgeRange) -> float:
        """指定された EdgeRange の実寸弧長 (メートル) を計算して返す."""
        segs = self.evaluate_path_segments(edge_range.piece_id, edge_range.path_id)
        target_seg = None
        for start_id, end_id, tbl in segs:
            if start_id == edge_range.start_node_id and end_id == edge_range.end_node_id:
                target_seg = tbl
                break

        if target_seg is None:
            return 0.0

        t0 = max(0.0, min(1.0, edge_range.t0))
        t1 = max(0.0, min(1.0, edge_range.t1))
        ratio_span = abs(t1 - t0)
        return float(target_seg.total_length * ratio_span)

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
