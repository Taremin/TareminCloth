# -*- coding: utf-8 -*-
"""合印（ノッチ）作成ツール (NotchTool).

外周の辺上をクリックして指定位置（パラメータ t）に合印（Notch）を配置する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..document import Notch, NotchKind
from ..geometry import build_arc_table, arc_length_at_t
from ..ids import new_id
from ..selection import HitTargetKind, hit_test
from .base import BaseTool

if TYPE_CHECKING:
    from ..editor import EditorController, EditorEvent


class NotchTool(BaseTool):
    """外周の辺に合印ノッチを追加するツール."""

    def __init__(self, *, default_kind: NotchKind = NotchKind.V) -> None:
        self.default_kind = default_kind

    def on_mouse_down(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.button != "LEFT":
            return False

        # 辺のヒットテスト（許容距離 12px 相当のワールド距離）
        threshold = controller.view.pixel_to_world_dist(12.0)
        hit = hit_test(
            controller.doc,
            event.world_pos,
            edge_threshold=threshold,
            allowed_kinds={HitTargetKind.EDGE},
        )

        if not hit or hit.kind != HitTargetKind.EDGE:
            return False

        if not hit.segment_start_node_id or not hit.path_id:
            return False

        piece = controller.doc.pieces.get(hit.piece_id)
        if not piece:
            return False

        path = piece.paths.get(hit.path_id)
        if not path:
            return False

        snapped_t = controller.doc.snap_notch_t(
            hit.piece_id,
            hit.segment_start_node_id,
            raw_t=hit.edge_t,
            path_id=hit.path_id,
            snap_enabled=not event.alt,
        )

        # 同一セグメント上の既存ノッチとの実寸弧長差を検査 (1mm未満なら既存ノッチを選択して重複抑止)
        nodes = path.nodes
        n_nodes = len(nodes)
        num_segs = n_nodes if path.closed else n_nodes - 1
        seg_idx = next((i for i in range(num_segs) if nodes[i].id == hit.segment_start_node_id), None)
        if seg_idx is not None:
            ctrls = path.get_segment_controls(seg_idx)
            table = build_arc_table(*ctrls)
            s_snapped = arc_length_at_t(table, snapped_t)

            for existing in controller.doc.notches.values():
                if (
                    existing.piece_id == hit.piece_id
                    and existing.path_id == hit.path_id
                    and existing.segment_start_id == hit.segment_start_node_id
                ):
                    s_ex = arc_length_at_t(table, existing.t)
                    if abs(s_ex - s_snapped) < 0.001:
                        controller.selection.select_notch(hit.piece_id, existing.id)
                        controller.is_dirty = True
                        return True

        before_doc = controller.doc.clone()
        notch_id = new_id()
        notch = Notch(
            id=notch_id,
            piece_id=hit.piece_id,
            path_id=hit.path_id,
            segment_start_id=hit.segment_start_node_id,
            t=snapped_t,
            kind=self.default_kind,
            depth=0.005,  # 5mm
            width=0.004,  # 4mm
        )
        controller.doc.notches[notch_id] = notch

        # 作成したノッチを選択状態にする
        controller.selection.select_notch(hit.piece_id, notch_id)

        controller.history.record_change(
            description="Add Notch",
            before_doc=before_doc,
            after_doc=controller.doc,
        )
        controller.is_dirty = True
        return True

    def on_mouse_move(self, controller: EditorController, event: EditorEvent) -> bool:
        return False

    def on_mouse_up(self, controller: EditorController, event: EditorEvent) -> bool:
        return False

    def on_key_press(self, controller: EditorController, event: EditorEvent) -> bool:
        return False

    def cancel(self, controller: EditorController) -> None:
        pass
