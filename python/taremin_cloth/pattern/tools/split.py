# -*- coding: utf-8 -*-
"""辺分割ツール (SplitEdgeTool) および辺上分割スナップ・実行共通ヘルパー."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..document import _NOTCH_MIN_GAP_M, Document, NotchSnapResult
from ..selection import HitResult, HitTargetKind, hit_test
from ..session import EdgePointPreviewState
from ..units import format_cm
from ...i18n import trans
from .base import BaseTool

if TYPE_CHECKING:
    from ..editor import EditorController, EditorEvent


def resolve_split_candidate_from_hit(
    controller: EditorController,
    hit: Optional[HitResult],
    *,
    snap_enabled: bool,
) -> Optional[tuple[int, NotchSnapResult]]:
    """辺ヒット結果から分割対象セグメント添字とスナップ結果を解決する."""
    if not hit or hit.kind not in (HitTargetKind.EDGE, HitTargetKind.INTERNAL_PATH):
        return None
    if not hit.segment_start_node_id or not hit.path_id:
        return None

    piece = controller.doc.pieces.get(hit.piece_id)
    if not piece:
        return None
    path = piece.paths.get(hit.path_id)
    if not path:
        return None

    n_nodes = len(path.nodes)
    num_segs = n_nodes if path.closed else n_nodes - 1
    seg_idx = next(
        (i for i in range(num_segs) if path.nodes[i].id == hit.segment_start_node_id),
        None,
    )
    if seg_idx is None:
        return None

    snap_res = controller.doc.resolve_edge_point_snap(
        hit.piece_id,
        hit.segment_start_node_id,
        raw_t=hit.edge_t,
        mode="SPLIT",
        path_id=hit.path_id,
        snap_enabled=snap_enabled,
        clipboard_length=controller.copied_edge_length,
    )
    if snap_res.total_length <= 2.0 * _NOTCH_MIN_GAP_M:
        return None

    return (seg_idx, snap_res)


def resolve_split_candidate(
    controller: EditorController,
    *,
    world_pos: tuple[float, float],
    edge_threshold: float,
    snap_enabled: bool,
) -> Optional[tuple[HitResult, int, NotchSnapResult]]:
    """指定ワールド座標から分割対象の辺ヒット、セグメント添字、スナップ結果を解決する."""
    hit = hit_test(
        controller.doc,
        world_pos,
        edge_threshold=edge_threshold,
        allowed_kinds={HitTargetKind.EDGE, HitTargetKind.INTERNAL_PATH},
    )
    resolved = resolve_split_candidate_from_hit(controller, hit, snap_enabled=snap_enabled)
    if resolved is None or hit is None:
        return None
    seg_idx, snap_res = resolved
    return (hit, seg_idx, snap_res)


def build_split_preview_state(
    hit: HitResult,
    snap_res: NotchSnapResult,
    *,
    snap_enabled: bool,
) -> EdgePointPreviewState:
    """スナップ結果から辺分割プレビュー用の EdgePointPreviewState を構築する."""
    return EdgePointPreviewState.from_snap_result(
        snap_res,
        mode="CREATE",
        piece_id=hit.piece_id,
        path_id=hit.path_id or "",
        segment_start_id=hit.segment_start_node_id or "",
        snap_enabled=snap_enabled,
        target_kind="SPLIT",
    )


def execute_split_edge(
    controller: EditorController,
    *,
    hit: HitResult,
    seg_idx: int,
    snap_res: NotchSnapResult,
    before_doc: Optional[Document] = None,
    clear_selection: bool = False,
    description: str = "Split Edge",
) -> bool:
    """解決済みスナップ位置で辺を分割し、新ノード選択・ステータス更新・履歴記録を行う."""
    doc_snapshot = before_doc if before_doc is not None else controller.doc.clone()
    new_node = controller.doc.insert_node_on_segment(
        hit.piece_id,
        hit.path_id or "",
        seg_idx=seg_idx,
        t=snap_res.t,
    )
    if new_node is None:
        return False

    if clear_selection:
        controller.selection.clear()
    controller.selection.select_node(hit.piece_id, new_node.id)
    controller.status_message = (
        f"{trans('Split Edge')}: {format_cm(snap_res.split_left_len)} / {format_cm(snap_res.split_right_len)}"
    )
    controller._record_doc_change(description=description, before_doc=doc_snapshot)
    controller.is_dirty = True
    return True


class SplitEdgeTool(BaseTool):
    """外周・内部線の辺上を任意位置で分割して頂点（Node）を挿入するツール."""

    def on_mouse_down(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.button != "LEFT":
            return False

        cand = resolve_split_candidate(
            controller,
            world_pos=event.world_pos,
            edge_threshold=controller.view.pixel_to_world_dist(12.0),
            snap_enabled=not event.alt,
        )
        if cand is None:
            return False

        hit, seg_idx, snap_res = cand
        return execute_split_edge(
            controller,
            hit=hit,
            seg_idx=seg_idx,
            snap_res=snap_res,
            description="Split Edge",
        )

    def on_mouse_move(self, controller: EditorController, event: EditorEvent) -> bool:
        prev = controller.edge_preview
        cand = resolve_split_candidate(
            controller,
            world_pos=event.world_pos,
            edge_threshold=controller.view.pixel_to_world_dist(12.0),
            snap_enabled=not event.alt,
        )
        if cand is None:
            controller.edge_preview = None
        else:
            hit, _, snap_res = cand
            controller.edge_preview = build_split_preview_state(
                hit,
                snap_res,
                snap_enabled=not event.alt,
            )

        if controller.edge_preview != prev:
            controller.is_dirty = True
            return True
        return False

    def on_mouse_up(self, controller: EditorController, event: EditorEvent) -> bool:
        return False

    def on_key_press(self, controller: EditorController, event: EditorEvent) -> bool:
        return False

    def cancel(self, controller: EditorController) -> None:
        if controller.edge_preview is not None:
            controller.edge_preview = None
            controller.is_dirty = True
