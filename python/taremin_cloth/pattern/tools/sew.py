# -*- coding: utf-8 -*-
"""型紙エディタの縫合定義ツール (Blender非依存)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..document import Document, EdgeRange, Seam, SeamMode
from ..ids import new_id
from ..selection import HitResult, HitTargetKind, hit_test
from .base import BaseTool

if TYPE_CHECKING:
    from ..editor import EditorController, EditorEvent


class SewTool(BaseTool):
    """2つの辺範囲をクリック選択して縫合 (Seam) を定義するツール.
    
    通常クリック: 1辺目 -> 2辺目で即座に縫合
    Shift+クリック: 複数辺を順次追加して鎖縫合 (Chain Seam) を構築 (EnterでA確定/完了)
    """

    name: str = "SEW"

    def __init__(self) -> None:
        self.step: int = 0  # 0: A側選択中, 1: B側選択中
        self.chain_a: list[EdgeRange] = []
        self.chain_b: list[EdgeRange] = []

    def on_mouse_down(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.button != "LEFT":
            return False

        # 辺のヒットテスト
        threshold = controller.view.pixel_to_world_dist(12.0)
        hit = hit_test(
            controller.doc,
            event.world_pos,
            edge_threshold=threshold,
            allowed_kinds={HitTargetKind.EDGE},
        )
        if hit is None or hit.kind != HitTargetKind.EDGE or not hit.path_id:
            return False

        edge_range = self._resolve_hit_edge_range(controller.doc, hit, bypass_notches=event.alt)
        if edge_range is None:
            return False

        if event.shift:
            # Shift+クリック: 鎖縫合の辺追加
            if self.step == 0:
                self.chain_a.append(edge_range)
                controller.preview_seam_candidates = list(self.chain_a)
                controller.is_dirty = True
                return True
            else:
                self.chain_b.append(edge_range)
                controller.preview_seam_candidates = list(self.chain_a) + list(self.chain_b)
                controller.is_dirty = True
                return True
        else:
            # 通常クリック
            if self.step == 0:
                self.chain_a = [edge_range]
                self.step = 1
                controller.preview_seam_candidates = [edge_range]
                controller.is_dirty = True
                return True
            else:
                self.chain_b = [edge_range]
                return self._finalize_seam(controller)

    def on_mouse_move(self, controller: EditorController, event: EditorEvent) -> bool:
        threshold = controller.view.pixel_to_world_dist(12.0)
        hit = hit_test(
            controller.doc,
            event.world_pos,
            edge_threshold=threshold,
            allowed_kinds={HitTargetKind.EDGE},
        )
        if hit and hit.kind == HitTargetKind.EDGE and hit.path_id:
            hover_edge = self._resolve_hit_edge_range(controller.doc, hit, bypass_notches=event.alt)
            if hover_edge is not None:
                base_cands = list(self.chain_a) if self.step == 0 else list(self.chain_a) + list(self.chain_b)
                controller.preview_seam_candidates = base_cands + [hover_edge]
                controller.is_dirty = True
                return True

        base_cands = list(self.chain_a) if self.step == 0 else list(self.chain_a) + list(self.chain_b)
        if len(controller.preview_seam_candidates) != len(base_cands):
            controller.preview_seam_candidates = base_cands
            controller.is_dirty = True
            return True
        return False

    def _resolve_hit_edge_range(
        self,
        doc: Document,
        hit: HitResult,
        *,
        bypass_notches: bool,
    ) -> Optional[EdgeRange]:
        """ヒット結果から部分または全体の EdgeRange を構築して返す."""
        if not hit.path_id or not hit.segment_start_node_id or not hit.segment_end_node_id:
            return None

        t0 = 0.0
        t1 = 1.0
        if not bypass_notches:
            bps = doc.get_segment_notch_breakpoints(
                hit.piece_id,
                hit.segment_start_node_id,
                path_id=hit.path_id,
            )
            cur_t = max(0.0, min(1.0, hit.edge_t))
            for i in range(len(bps) - 1):
                if cur_t <= bps[i + 1] + 1e-6 or i == len(bps) - 2:
                    t0 = bps[i]
                    t1 = bps[i + 1]
                    break

        return EdgeRange(
            piece_id=hit.piece_id,
            path_id=hit.path_id,
            start_node_id=hit.segment_start_node_id,
            end_node_id=hit.segment_end_node_id,
            t0=t0,
            t1=t1,
        )

    def on_mouse_up(self, controller: EditorController, event: EditorEvent) -> bool:
        return False

    def on_key_press(self, controller: EditorController, event: EditorEvent) -> bool:
        key_upper = event.key.upper()
        if key_upper == "ESCAPE":
            self.cancel(controller)
            return True

        if key_upper in ("RET", "RETURN", "ENTER", "NUMPAD_ENTER", "SPACE"):
            if self.step == 0 and self.chain_a:
                self.step = 1
                controller.is_dirty = True
                return True
            elif self.step == 1 and self.chain_b:
                return self._finalize_seam(controller)

        if key_upper in ("R", "F"):
            if controller.flip_selected_seams():
                return True

        return False

    def _finalize_seam(self, controller: EditorController) -> bool:
        """集めた辺から縫合を確定する."""
        if not self.chain_a or not self.chain_b:
            self.cancel(controller)
            return False

        first_a = self.chain_a[0]
        first_b = self.chain_b[0]

        # 同一辺の自己縫合（区間重複）は禁止
        for er_a in self.chain_a:
            piece_a = controller.doc.pieces.get(er_a.piece_id)
            if not piece_a:
                continue
            path_a = piece_a.paths.get(er_a.path_id)
            if not path_a:
                continue
            res_a = path_a.resolve_edge_range(er_a)
            if res_a is None:
                continue
            seg_a, a_t0, a_t1 = res_a
            min_a = min(a_t0, a_t1)
            max_a = max(a_t0, a_t1)

            for er_b in self.chain_b:
                if er_a.piece_id != er_b.piece_id or er_a.path_id != er_b.path_id:
                    continue
                res_b = path_a.resolve_edge_range(er_b)
                if res_b is None:
                    continue
                seg_b, b_t0, b_t1 = res_b
                if seg_a != seg_b:
                    continue
                min_b = min(b_t0, b_t1)
                max_b = max(b_t0, b_t1)
                # 区間が重なっている場合は禁止
                if max(min_a, min_b) < min(max_a, max_b) - 1e-4:
                    self.cancel(controller)
                    return True

        before_doc = Document.from_json(controller.doc.to_json())
        seam_id = new_id()
        is_chain = len(self.chain_a) > 1 or len(self.chain_b) > 1
        seam = Seam(
            id=seam_id,
            a=first_a,
            b=first_b,
            chain_a=list(self.chain_a),
            chain_b=list(self.chain_b),
            mode=SeamMode.CHAIN if is_chain else SeamMode.SEGMENT,
        )
        controller.doc.seams[seam_id] = seam

        controller.history.record_change(
            description="Add Seam",
            before_doc=before_doc,
            after_doc=controller.doc,
        )

        self.cancel(controller)
        controller.selection.select_seam(seam_id)
        controller.is_dirty = True
        return True

    def cancel(self, controller: EditorController) -> None:
        self.step = 0
        self.chain_a.clear()
        self.chain_b.clear()
        controller.preview_seam_candidates.clear()
        controller.is_dirty = True
