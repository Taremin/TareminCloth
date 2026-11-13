# -*- coding: utf-8 -*-
"""型紙パターンエディタの共通2D幾何変換・整列・スナップエンジン (Blender非依存)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .document import (
    Anchor,
    Document,
    EdgeRange,
    Node,
    NodeKind,
    Path,
    PathRole,
    Piece,
)
from .geometry import (
    Vec2,
    build_arc_table,
    compute_bounding_box,
    segment_intersects_aabb,
    signed_area,
)
from .session import SelectionState, SnapSettings, ViewTransform2D, compute_snap

_MIN_SCALE_ABS: float = 1e-4
_MAX_MITER_MULT: float = 3.0


@dataclass(frozen=True)
class NodeSnapshot:
    """変形開始時における単一ノードのローカル・ワールド座標および法線スナップショット."""

    piece_id: str
    path_id: str
    node_id: str
    local_pos: Vec2
    local_in_handle: Optional[Vec2]
    local_out_handle: Optional[Vec2]
    world_pos: Vec2
    world_in_handle_vec: Optional[Vec2]
    world_out_handle_vec: Optional[Vec2]
    miter_normal: Vec2 = (0.0, 0.0)


@dataclass(frozen=True)
class AnchorSnapshot:
    """変形開始時におけるアンカーのスナップショット."""

    anchor_id: str
    piece_id: str
    local_pos: Vec2
    world_pos: Vec2


@dataclass(frozen=True)
class PieceTransformSnapshot:
    """変形開始時におけるピース全体の2Dアフィン変換スナップショット."""

    piece_id: str
    tx: float
    ty: float
    rotation_rad: float
    sx: float
    sy: float


@dataclass
class TransformSnapshot:
    """ドラッグ移動・G/R/Sモーダル変形全体で共有する開始状態スナップショット."""

    piece_only_mode: bool
    piece_transforms: dict[str, PieceTransformSnapshot] = field(default_factory=dict)
    node_snapshots: dict[tuple[str, str, str], NodeSnapshot] = field(default_factory=dict)
    anchor_snapshots: dict[str, AnchorSnapshot] = field(default_factory=dict)
    snap_candidate_points: list[Vec2] = field(default_factory=list)
    pivot_world: Vec2 = (0.0, 0.0)
    active_edge_normal: Optional[Vec2] = None
    exclude_node_ids: set[str] = field(default_factory=set)
    full_path_keys: set[tuple[str, str]] = field(default_factory=set)


@dataclass
class TransformModalSession:
    """G (移動) / R (回転) / S (拡大縮小) モーダル変形の実行状態."""

    mode: str  # "MOVE" | "ROTATE" | "SCALE"
    snapshot: TransformSnapshot
    start_cursor_world: Vec2
    current_cursor_world: Vec2
    snap_reference_world: Optional[Vec2] = None
    axis_lock: Optional[str] = None  # None | "X" | "Y" | "NORMAL" | "NORMAL_ACTIVE"
    numeric_buffer: str = ""
    current_scale: tuple[float, float] = (1.0, 1.0)
    current_angle_rad: float = 0.0
    current_delta: Vec2 = (0.0, 0.0)


def _to_world_handle_vec(
    piece: Piece,
    node: Node,
    *,
    rel_handle: Optional[Vec2],
    world_node_pos: Vec2,
) -> Optional[Vec2]:
    """ノード相対のローカルベジェハンドルをワールド空間相対ベクトルへ変換する."""
    if rel_handle is None:
        return None
    w_tip = piece.transform.transform_point((node.position[0] + rel_handle[0], node.position[1] + rel_handle[1]))
    return (w_tip[0] - world_node_pos[0], w_tip[1] - world_node_pos[1])


def _apply_world_pos_and_handles_to_node(
    piece: Piece,
    node: Node,
    *,
    new_world_pos: Vec2,
    new_world_in_vec: Optional[Vec2] = None,
    new_world_out_vec: Optional[Vec2] = None,
    update_handles: bool = False,
) -> None:
    """ワールド座標（および必要時ベジェハンドル）をピースローカル座標へ書き戻す."""
    new_local_pos = piece.transform.inverse_apply(new_world_pos)
    node.position = new_local_pos
    if not update_handles:
        return

    if new_world_in_vec is not None:
        w_in_tip = (new_world_pos[0] + new_world_in_vec[0], new_world_pos[1] + new_world_in_vec[1])
        l_in_tip = piece.transform.inverse_apply(w_in_tip)
        node.in_handle = (l_in_tip[0] - new_local_pos[0], l_in_tip[1] - new_local_pos[1])

    if new_world_out_vec is not None:
        w_out_tip = (new_world_pos[0] + new_world_out_vec[0], new_world_pos[1] + new_world_out_vec[1])
        l_out_tip = piece.transform.inverse_apply(w_out_tip)
        node.out_handle = (l_out_tip[0] - new_local_pos[0], l_out_tip[1] - new_local_pos[1])


def _compute_segment_world_normal(piece: Piece, path: Path, *, seg_idx: int, is_ccw_outer: bool) -> Vec2:
    """指定パスセグメントのワールド空間における単位外向き/垂直法線を返す."""
    n_nodes = len(path.nodes)
    n0 = path.nodes[seg_idx]
    n1 = path.nodes[(seg_idx + 1) % n_nodes]
    w0 = piece.transform.transform_point(n0.position)
    w1 = piece.transform.transform_point(n1.position)
    dx = w1[0] - w0[0]
    dy = w1[1] - w0[1]
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-8:
        # 端点が一致している場合はベジェ中点付近の接線にフォールバック
        ctrls = path.get_segment_controls(seg_idx)
        w_c0 = piece.transform.transform_point(ctrls[0])
        w_c1 = piece.transform.transform_point(ctrls[1])
        dx = w_c1[0] - w_c0[0]
        dy = w_c1[1] - w_c0[1]
        seg_len = math.hypot(dx, dy)
    if seg_len < 1e-8:
        return (0.0, 1.0)

    ux = dx / seg_len
    uy = dy / seg_len
    # CCW 外周の場合は進行方向の右向き (uy, -ux) が外向き法線
    if is_ccw_outer or path.role != PathRole.OUTER or not path.closed:
        return (uy, -ux)
    return (-uy, ux)


def _combine_miter_normal(normals: list[Vec2]) -> Vec2:
    """1頂点に接続する選択辺群の単位法線リストから平行オフセット用マイターベクトルを算出する."""
    if not normals:
        return (0.0, 0.0)
    if len(normals) == 1:
        return normals[0]

    # 主な2法線の合算から二等分線方向を求める
    bx = sum(n[0] for n in normals)
    by = sum(n[1] for n in normals)
    b_len = math.hypot(bx, by)
    if b_len < 1e-5:
        return normals[0]

    bx /= b_len
    by /= b_len
    if len(normals) == 2:
        dot_val = bx * normals[0][0] + by * normals[0][1]
        if abs(dot_val) > 1.0 / _MAX_MITER_MULT:
            scale = 1.0 / dot_val
            return (bx * scale, by * scale)
        scale = _MAX_MITER_MULT if dot_val >= 0.0 else -_MAX_MITER_MULT
        return (bx * scale, by * scale)

    return (bx, by)


def _compute_path_is_ccw_world(piece: Piece, path: Path) -> bool:
    """パスがワールド空間で反時計回り (CCW) か判定する."""
    if not path.closed or len(path.nodes) < 3:
        return True
    pts = [piece.transform.transform_point(p) for p in path.sample_outline_points(steps_per_segment=8)]
    return signed_area(pts) >= 0.0


def can_start_modal_transform(doc: Document, selection: SelectionState, *, mode: str) -> tuple[bool, str]:
    """現在の選択状態で指定モーダル変形 (MOVE / ROTATE / SCALE) を開始できるか判定する."""
    has_targets = bool(selection.piece_ids or selection.node_ids or selection.edge_ranges)
    if not has_targets:
        return (False, "")

    snapshot = capture_transform_snapshot(doc, selection)
    if snapshot.piece_only_mode:
        if not snapshot.piece_transforms:
            return (False, "")
        return (True, "")

    if not snapshot.node_snapshots:
        return (False, "")

    if mode in ("ROTATE", "SCALE") and len(snapshot.node_snapshots) < 2:
        return (False, "Select 2+ nodes, an edge, or a piece to rotate/scale")

    return (True, "")


def capture_transform_snapshot(doc: Document, selection: SelectionState) -> TransformSnapshot:
    """現在の選択要素から変形計算用の開始時スナップショットを生成する."""
    has_nodes_or_edges = bool(selection.node_ids or selection.edge_ranges)
    piece_only_mode = bool(selection.piece_ids) and not has_nodes_or_edges

    piece_transforms: dict[str, PieceTransformSnapshot] = {}
    node_snapshots: dict[tuple[str, str, str], NodeSnapshot] = {}
    anchor_snapshots: dict[str, AnchorSnapshot] = {}
    snap_candidate_points: list[Vec2] = []
    exclude_node_ids: set[str] = set()
    full_path_keys: set[tuple[str, str]] = set()
    edge_normals_by_node: dict[tuple[str, str, str], list[Vec2]] = {}
    active_edge_normal: Optional[Vec2] = None

    # 1. 選択中の全ピース（piece_ids）
    for pid in selection.piece_ids:
        piece = doc.pieces.get(pid)
        if not piece:
            continue
        piece_transforms[pid] = PieceTransformSnapshot(
            piece_id=pid,
            tx=piece.transform.tx,
            ty=piece.transform.ty,
            rotation_rad=piece.transform.rotation_rad,
            sx=piece.transform.sx,
            sy=piece.transform.sy,
        )
        for path_id, path in piece.paths.items():
            if path.closed and len(path.nodes) >= 3:
                full_path_keys.add((pid, path_id))
            for node in path.nodes:
                w_pos = piece.transform.transform_point(node.position)
                snap_candidate_points.append(w_pos)
                exclude_node_ids.add(node.id)
                key = (pid, path_id, node.id)
                node_snapshots[key] = NodeSnapshot(
                    piece_id=pid,
                    path_id=path_id,
                    node_id=node.id,
                    local_pos=node.position,
                    local_in_handle=node.in_handle,
                    local_out_handle=node.out_handle,
                    world_pos=w_pos,
                    world_in_handle_vec=_to_world_handle_vec(
                        piece, node, rel_handle=node.in_handle, world_node_pos=w_pos
                    ),
                    world_out_handle_vec=_to_world_handle_vec(
                        piece, node, rel_handle=node.out_handle, world_node_pos=w_pos
                    ),
                )
        for aid, anchor in doc.anchors.items():
            if anchor.piece_id == pid:
                w_anc = piece.transform.transform_point(anchor.position)
                anchor_snapshots[aid] = AnchorSnapshot(
                    anchor_id=aid,
                    piece_id=pid,
                    local_pos=anchor.position,
                    world_pos=w_anc,
                )

    # 2. 選択中の全辺 (edge_ranges) を展開し法線計算および両端ノード収集
    active_er = selection.get_active_edge_range()
    path_ccw_cache: dict[tuple[str, str], bool] = {}

    for er in selection.edge_ranges:
        piece = doc.pieces.get(er.piece_id)
        if not piece:
            continue
        path = piece.paths.get(er.path_id)
        if not path or len(path.nodes) < 2:
            continue
        res = path.resolve_edge_range(er)
        if not res:
            continue
        seg_idx, _, _ = res
        n_nodes = len(path.nodes)
        n0 = path.nodes[seg_idx]
        n1 = path.nodes[(seg_idx + 1) % n_nodes]

        cache_key = (piece.id, path.id)
        if cache_key not in path_ccw_cache:
            path_ccw_cache[cache_key] = _compute_path_is_ccw_world(piece, path)
        seg_normal = _compute_segment_world_normal(
            piece, path, seg_idx=seg_idx, is_ccw_outer=path_ccw_cache[cache_key]
        )
        if active_er is er or active_edge_normal is None:
            active_edge_normal = seg_normal

        for nd in (n0, n1):
            nkey = (piece.id, path.id, nd.id)
            edge_normals_by_node.setdefault(nkey, []).append(seg_normal)
            if nkey not in node_snapshots:
                w_pos = piece.transform.transform_point(nd.position)
                snap_candidate_points.append(w_pos)
                exclude_node_ids.add(nd.id)
                node_snapshots[nkey] = NodeSnapshot(
                    piece_id=piece.id,
                    path_id=path.id,
                    node_id=nd.id,
                    local_pos=nd.position,
                    local_in_handle=nd.in_handle,
                    local_out_handle=nd.out_handle,
                    world_pos=w_pos,
                    world_in_handle_vec=_to_world_handle_vec(
                        piece, nd, rel_handle=nd.in_handle, world_node_pos=w_pos
                    ),
                    world_out_handle_vec=_to_world_handle_vec(
                        piece, nd, rel_handle=nd.out_handle, world_node_pos=w_pos
                    ),
                )

    # 3. 明示選択ノード (node_ids) の収集
    if selection.node_ids:
        for pid, piece in doc.pieces.items():
            for path_id, path in piece.paths.items():
                for nd in path.nodes:
                    if nd.id not in selection.node_ids:
                        continue
                    nkey = (pid, path_id, nd.id)
                    if nkey not in node_snapshots:
                        w_pos = piece.transform.transform_point(nd.position)
                        snap_candidate_points.append(w_pos)
                        exclude_node_ids.add(nd.id)
                        node_snapshots[nkey] = NodeSnapshot(
                            piece_id=pid,
                            path_id=path_id,
                            node_id=nd.id,
                            local_pos=nd.position,
                            local_in_handle=nd.in_handle,
                            local_out_handle=nd.out_handle,
                            world_pos=w_pos,
                            world_in_handle_vec=_to_world_handle_vec(
                                piece, nd, rel_handle=nd.in_handle, world_node_pos=w_pos
                            ),
                            world_out_handle_vec=_to_world_handle_vec(
                                piece, nd, rel_handle=nd.out_handle, world_node_pos=w_pos
                            ),
                        )

    # 全ノードが選択対象になっている閉パスをマーク（負スケール時のCCW補正判定用）
    if not piece_only_mode:
        for pid, piece in doc.pieces.items():
            for path_id, path in piece.paths.items():
                if path.closed and len(path.nodes) >= 3:
                    if all((pid, path_id, nd.id) in node_snapshots for nd in path.nodes):
                        full_path_keys.add((pid, path_id))

    # マイター法線ベクトル確定
    default_normal = active_edge_normal or (0.0, 0.0)
    for nkey, nsnap in list(node_snapshots.items()):
        normals = edge_normals_by_node.get(nkey)
        miter = _combine_miter_normal(normals) if normals else default_normal
        if miter != nsnap.miter_normal:
            node_snapshots[nkey] = NodeSnapshot(
                piece_id=nsnap.piece_id,
                path_id=nsnap.path_id,
                node_id=nsnap.node_id,
                local_pos=nsnap.local_pos,
                local_in_handle=nsnap.local_in_handle,
                local_out_handle=nsnap.local_out_handle,
                world_pos=nsnap.world_pos,
                world_in_handle_vec=nsnap.world_in_handle_vec,
                world_out_handle_vec=nsnap.world_out_handle_vec,
                miter_normal=miter,
            )

    # ピボット座標（全対象点群のバウンディングボックス中心）
    all_pts = [ns.world_pos for ns in node_snapshots.values()]
    if all_pts:
        b_min, b_max = compute_bounding_box(all_pts)
        pivot = ((b_min[0] + b_max[0]) * 0.5, (b_min[1] + b_max[1]) * 0.5)
    elif piece_transforms:
        pts_t = [(ts.tx, ts.ty) for ts in piece_transforms.values()]
        b_min, b_max = compute_bounding_box(pts_t)
        pivot = ((b_min[0] + b_max[0]) * 0.5, (b_min[1] + b_max[1]) * 0.5)
    else:
        pivot = (0.0, 0.0)

    return TransformSnapshot(
        piece_only_mode=piece_only_mode,
        piece_transforms=piece_transforms,
        node_snapshots=node_snapshots,
        anchor_snapshots=anchor_snapshots,
        snap_candidate_points=snap_candidate_points,
        pivot_world=pivot,
        active_edge_normal=active_edge_normal,
        exclude_node_ids=exclude_node_ids,
        full_path_keys=full_path_keys,
    )


def find_nearest_snap_reference(snapshot: TransformSnapshot, cursor_world: Vec2) -> Optional[Vec2]:
    """開始時スナップショットからカーソルに最も近い頂点のワールド座標を単一スナップ基準点として返す."""
    if snapshot.snap_candidate_points:
        return min(
            snapshot.snap_candidate_points,
            key=lambda p: (p[0] - cursor_world[0]) ** 2 + (p[1] - cursor_world[1]) ** 2,
        )
    if snapshot.piece_transforms:
        return snapshot.pivot_world
    return None


def restore_snapshot(doc: Document, snapshot: TransformSnapshot) -> None:
    """ドキュメントの対象要素をスナップショット開始時点の状態へ復元する."""
    for pid, ts in snapshot.piece_transforms.items():
        piece = doc.pieces.get(pid)
        if piece:
            piece.transform = piece.transform.__class__(
                tx=ts.tx,
                ty=ts.ty,
                rotation_rad=ts.rotation_rad,
                sx=ts.sx,
                sy=ts.sy,
            )

    # 高速引当て用のノード辞書
    node_map = _build_doc_node_lookup(doc)
    for nkey, nsnap in snapshot.node_snapshots.items():
        node = node_map.get(nkey)
        if node:
            node.position = nsnap.local_pos
            node.in_handle = nsnap.local_in_handle
            node.out_handle = nsnap.local_out_handle

    for aid, asnap in snapshot.anchor_snapshots.items():
        anchor = doc.anchors.get(aid)
        if anchor:
            anchor.position = asnap.local_pos


def _build_doc_node_lookup(doc: Document) -> dict[tuple[str, str, str], Node]:
    lookup: dict[tuple[str, str, str], Node] = {}
    for pid, piece in doc.pieces.items():
        for path_id, path in piece.paths.items():
            for node in path.nodes:
                lookup[(pid, path_id, node.id)] = node
    return lookup


def _constrain_delta_to_axis(
    delta: Vec2,
    *,
    axis_lock: Optional[str],
    active_edge_normal: Optional[Vec2],
    clamp_ortho: bool,
) -> Vec2:
    dx, dy = float(delta[0]), float(delta[1])
    if axis_lock == "X":
        return (dx, 0.0)
    if axis_lock == "Y":
        return (0.0, dy)
    if axis_lock in ("NORMAL", "NORMAL_ACTIVE") and active_edge_normal is not None:
        nx, ny = active_edge_normal
        proj = dx * nx + dy * ny
        return (proj * nx, proj * ny)
    if clamp_ortho and axis_lock is None:
        if abs(dx) >= abs(dy):
            return (dx, 0.0)
        return (0.0, dy)
    return (dx, dy)


def apply_translation(
    doc: Document,
    snapshot: TransformSnapshot,
    *,
    delta_world: Vec2,
    view: Optional[ViewTransform2D] = None,
    snap_settings: Optional[SnapSettings] = None,
    snap_reference_world: Optional[Vec2] = None,
    axis_lock: Optional[str] = None,
    clamp_ortho: bool = False,
) -> Vec2:
    """スナップショット基準で平行移動（単一基準点スナップ・軸拘束・法線オフセット対応）を適用する."""
    constrained = _constrain_delta_to_axis(
        delta_world,
        axis_lock=axis_lock,
        active_edge_normal=snapshot.active_edge_normal,
        clamp_ortho=clamp_ortho,
    )

    # 個別辺マイター法線オフセットモード
    if (
        axis_lock == "NORMAL"
        and not snapshot.piece_only_mode
        and snapshot.active_edge_normal is not None
    ):
        nx, ny = snapshot.active_edge_normal
        scalar_offset = constrained[0] * nx + constrained[1] * ny
        node_map = _build_doc_node_lookup(doc)
        for nkey, nsnap in snapshot.node_snapshots.items():
            node = node_map.get(nkey)
            piece = doc.pieces.get(nsnap.piece_id)
            if not node or not piece:
                continue
            mx, my = nsnap.miter_normal
            new_w = (nsnap.world_pos[0] + mx * scalar_offset, nsnap.world_pos[1] + my * scalar_offset)
            _apply_world_pos_and_handles_to_node(piece, node, new_world_pos=new_w, update_handles=False)
        return (nx * scalar_offset, ny * scalar_offset)

    final_dx, final_dy = constrained

    # 単一基準点1回スナップ（複数頂点の相対形状を維持）
    if (
        snap_settings is not None
        and snap_settings.enabled
        and snap_reference_world is not None
        and view is not None
    ):
        raw_target = (snap_reference_world[0] + final_dx, snap_reference_world[1] + final_dy)
        snap_res = compute_snap(
            raw_target,
            doc,
            view=view,
            settings=snap_settings,
            exclude_node_ids=snapshot.exclude_node_ids,
        )
        snapped_delta = (
            snap_res.world_pos[0] - snap_reference_world[0],
            snap_res.world_pos[1] - snap_reference_world[1],
        )
        final_dx, final_dy = _constrain_delta_to_axis(
            snapped_delta,
            axis_lock=axis_lock,
            active_edge_normal=snapshot.active_edge_normal,
            clamp_ortho=clamp_ortho,
        )

    if snapshot.piece_only_mode:
        for pid, ts in snapshot.piece_transforms.items():
            piece = doc.pieces.get(pid)
            if piece:
                piece.transform = piece.transform.__class__(
                    tx=ts.tx + final_dx,
                    ty=ts.ty + final_dy,
                    rotation_rad=ts.rotation_rad,
                    sx=ts.sx,
                    sy=ts.sy,
                )
        return (final_dx, final_dy)

    node_map = _build_doc_node_lookup(doc)
    for nkey, nsnap in snapshot.node_snapshots.items():
        node = node_map.get(nkey)
        piece = doc.pieces.get(nsnap.piece_id)
        if not node or not piece:
            continue
        new_w = (nsnap.world_pos[0] + final_dx, nsnap.world_pos[1] + final_dy)
        _apply_world_pos_and_handles_to_node(piece, node, new_world_pos=new_w, update_handles=False)

    return (final_dx, final_dy)


def apply_rotation(
    doc: Document,
    snapshot: TransformSnapshot,
    *,
    angle_rad: float,
) -> None:
    """ピボット中心で対象ノード・ベジェハンドル・アンカーを2D回転しローカル座標へ書き戻す."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    px, py = snapshot.pivot_world

    def rot_point(w: Vec2) -> Vec2:
        rx = w[0] - px
        ry = w[1] - py
        return (px + cos_a * rx - sin_a * ry, py + sin_a * rx + cos_a * ry)

    def rot_vec(v: Optional[Vec2]) -> Optional[Vec2]:
        if v is None:
            return None
        return (cos_a * v[0] - sin_a * v[1], sin_a * v[0] + cos_a * v[1])

    node_map = _build_doc_node_lookup(doc)
    for nkey, nsnap in snapshot.node_snapshots.items():
        node = node_map.get(nkey)
        piece = doc.pieces.get(nsnap.piece_id)
        if not node or not piece:
            continue
        new_w = rot_point(nsnap.world_pos)
        new_in = rot_vec(nsnap.world_in_handle_vec)
        new_out = rot_vec(nsnap.world_out_handle_vec)
        _apply_world_pos_and_handles_to_node(
            piece,
            node,
            new_world_pos=new_w,
            new_world_in_vec=new_in,
            new_world_out_vec=new_out,
            update_handles=True,
        )

    for aid, asnap in snapshot.anchor_snapshots.items():
        anchor = doc.anchors.get(aid)
        piece = doc.pieces.get(asnap.piece_id)
        if anchor and piece:
            anchor.position = piece.transform.inverse_apply(rot_point(asnap.world_pos))


def _guard_scale_factor(val: float) -> float:
    if not math.isfinite(val):
        return 1.0
    if abs(val) < _MIN_SCALE_ABS:
        return -_MIN_SCALE_ABS if val < 0.0 else _MIN_SCALE_ABS
    return float(val)


def apply_scale(
    doc: Document,
    snapshot: TransformSnapshot,
    *,
    sx: float,
    sy: float,
) -> tuple[float, float]:
    """ピボット中心で対象ノード・ベジェハンドル・アンカーを2D拡大縮小しローカル座標へ書き戻す."""
    eff_sx = _guard_scale_factor(sx)
    eff_sy = _guard_scale_factor(sy)
    px, py = snapshot.pivot_world

    def scale_point(w: Vec2) -> Vec2:
        return (px + (w[0] - px) * eff_sx, py + (w[1] - py) * eff_sy)

    def scale_vec(v: Optional[Vec2]) -> Optional[Vec2]:
        if v is None:
            return None
        return (v[0] * eff_sx, v[1] * eff_sy)

    node_map = _build_doc_node_lookup(doc)
    for nkey, nsnap in snapshot.node_snapshots.items():
        node = node_map.get(nkey)
        piece = doc.pieces.get(nsnap.piece_id)
        if not node or not piece:
            continue
        new_w = scale_point(nsnap.world_pos)
        new_in = scale_vec(nsnap.world_in_handle_vec)
        new_out = scale_vec(nsnap.world_out_handle_vec)
        _apply_world_pos_and_handles_to_node(
            piece,
            node,
            new_world_pos=new_w,
            new_world_in_vec=new_in,
            new_world_out_vec=new_out,
            update_handles=True,
        )

    for aid, asnap in snapshot.anchor_snapshots.items():
        anchor = doc.anchors.get(aid)
        piece = doc.pieces.get(asnap.piece_id)
        if anchor and piece:
            anchor.position = piece.transform.inverse_apply(scale_point(asnap.world_pos))

    return (eff_sx, eff_sy)


def finalize_scale_orientation(
    doc: Document,
    snapshot: TransformSnapshot,
    *,
    sx: float,
    sy: float,
) -> None:
    """負の拡大縮小確定時に反転して CW になった閉パスの巻き方向 (CCW) およびノッチを補正する."""
    if sx * sy >= 0.0:
        return

    for pid, path_id in snapshot.full_path_keys:
        piece = doc.pieces.get(pid)
        if not piece:
            continue
        path = piece.paths.get(path_id)
        if not path or not path.closed or len(path.nodes) < 3:
            continue
        pts_local = path.sample_outline_points(steps_per_segment=12)
        if signed_area(pts_local) >= 0.0:
            continue

        n = len(path.nodes)
        next_id_map = {path.nodes[i].id: path.nodes[(i + 1) % n].id for i in range(n)}
        rev_nodes: list[Node] = []
        for orig_n in reversed(path.nodes):
            rev_nodes.append(
                Node(
                    id=orig_n.id,
                    position=orig_n.position,
                    kind=orig_n.kind,
                    in_handle=orig_n.out_handle,
                    out_handle=orig_n.in_handle,
                )
            )
        path.nodes = rev_nodes

        for notch in doc.notches.values():
            if notch.piece_id == pid and notch.path_id == path_id:
                if notch.segment_start_id in next_id_map:
                    notch.segment_start_id = next_id_map[notch.segment_start_id]
                    notch.t = max(0.001, min(0.999, 1.0 - float(notch.t)))


def align_selected_nodes(
    doc: Document,
    selection: SelectionState,
    *,
    axis: str,
    mode: str,
) -> bool:
    """選択中の複数頂点（または選択辺の端点）をワールド座標 X / Y で整列させる."""
    axis_u = axis.upper()
    mode_u = mode.upper()
    if axis_u not in ("X", "Y"):
        raise ValueError(f"Unsupported align axis: '{axis}'")
    if mode_u not in ("MIN", "CENTER", "MAX", "AVG"):
        raise ValueError(f"Unsupported align mode: '{mode}'")

    # piece_ids が選ばれず node_ids / edge_ranges がある対象のみ収集
    node_keys: dict[tuple[str, str, str], tuple[Piece, Node, Vec2]] = {}
    node_map = _build_doc_node_lookup(doc)

    for er in selection.edge_ranges:
        piece = doc.pieces.get(er.piece_id)
        if not piece:
            continue
        path = piece.paths.get(er.path_id)
        if not path:
            continue
        res = path.resolve_edge_range(er)
        if not res:
            continue
        seg_idx, _, _ = res
        n_nodes = len(path.nodes)
        for nd in (path.nodes[seg_idx], path.nodes[(seg_idx + 1) % n_nodes]):
            key = (piece.id, path.id, nd.id)
            if key not in node_keys:
                node_keys[key] = (piece, nd, piece.transform.transform_point(nd.position))

    if selection.node_ids:
        for (pid, path_id, nid), nd in node_map.items():
            if nid in selection.node_ids:
                piece = doc.pieces[pid]
                key = (pid, path_id, nid)
                if key not in node_keys:
                    node_keys[key] = (piece, nd, piece.transform.transform_point(nd.position))

    if len(node_keys) < 2:
        return False

    idx = 0 if axis_u == "X" else 1
    coords = [w_pos[idx] for (_, _, w_pos) in node_keys.values()]
    if mode_u == "MIN":
        target = min(coords)
    elif mode_u == "MAX":
        target = max(coords)
    elif mode_u == "CENTER":
        target = 0.5 * (min(coords) + max(coords))
    else:
        target = sum(coords) / len(coords)

    for piece, node, w_pos in node_keys.values():
        new_w = (target, w_pos[1]) if idx == 0 else (w_pos[0], target)
        node.position = piece.transform.inverse_apply(new_w)

    return True


def find_edges_intersecting_box(
    doc: Document,
    *,
    box_min: Vec2,
    box_max: Vec2,
) -> list[EdgeRange]:
    """2Dワールド矩形と交差または内包される全ピースの辺・内部線セグメントを EdgeRange リストとして返す."""
    hits: list[EdgeRange] = []
    for piece_id, piece in doc.pieces.items():
        for path_id, path in piece.paths.items():
            n_nodes = len(path.nodes)
            if n_nodes < 2:
                continue
            num_segs = n_nodes if path.closed else n_nodes - 1
            for i in range(num_segs):
                n0 = path.nodes[i]
                n1 = path.nodes[(i + 1) % n_nodes]
                is_curved = (
                    n0.kind != NodeKind.CORNER
                    or n1.kind != NodeKind.CORNER
                    or n0.out_handle is not None
                    or n1.in_handle is not None
                )
                if not is_curved:
                    w0 = piece.transform.transform_point(n0.position)
                    w1 = piece.transform.transform_point(n1.position)
                    if segment_intersects_aabb(w0, w1, box_min=box_min, box_max=box_max):
                        hits.append(
                            EdgeRange(
                                piece_id=piece_id,
                                path_id=path_id,
                                start_node_id=n0.id,
                                end_node_id=n1.id,
                            )
                        )
                else:
                    ctrls = path.get_segment_controls(i)
                    tbl = build_arc_table(*ctrls, steps=12)
                    w_pts = [piece.transform.transform_point(pt) for pt in tbl.points]
                    if any(
                        segment_intersects_aabb(w_pts[k], w_pts[k + 1], box_min=box_min, box_max=box_max)
                        for k in range(len(w_pts) - 1)
                    ):
                        hits.append(
                            EdgeRange(
                                piece_id=piece_id,
                                path_id=path_id,
                                start_node_id=n0.id,
                                end_node_id=n1.id,
                            )
                        )
    return hits


def summarize_multi_node_info(doc: Document, selection: SelectionState) -> tuple[list[tuple[Piece, Node, Vec2]], Vec2]:
    """選択中ノード一覧（または選択ノード群）とワールドBBoxサイズ (w, h) を返す."""
    nodes_info: list[tuple[Piece, Node, Vec2]] = []
    for pid, piece in doc.pieces.items():
        for path in piece.paths.values():
            for nd in path.nodes:
                if nd.id in selection.node_ids:
                    nodes_info.append((piece, nd, piece.transform.transform_point(nd.position)))
    if not nodes_info:
        return ([], (0.0, 0.0))
    b_min, b_max = compute_bounding_box([w for _, _, w in nodes_info])
    return (nodes_info, (b_max[0] - b_min[0], b_max[1] - b_min[1]))
