# -*- coding: utf-8 -*-
"""型紙エディタの選択およびドラッグ移動ツール (Blender非依存)."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Optional

from ..document import (
    Document,
    EdgeRange,
    NodeKind,
)
from ..geometry import (
    build_arc_table,
    closest_point_on_table,
)
from ..selection import HitResult, HitTargetKind, hit_test
from ..transform_ops import (
    TransformSnapshot,
    apply_translation,
    capture_transform_snapshot,
    find_edges_intersecting_box,
    find_nearest_snap_reference,
)
from ...utils.logger import logger
from .base import BaseTool

if TYPE_CHECKING:
    from ..editor import EditorController, EditorEvent

_DRAG_THRESHOLD_PX: float = 3.0


class SelectTool(BaseTool):
    """要素選択およびドラッグ移動ツール."""

    name: str = "SELECT"

    def __init__(self) -> None:
        self.is_dragging: bool = False
        self.is_box_selecting: bool = False
        self.drag_start_world: tuple[float, float] = (0.0, 0.0)
        self.drag_start_screen: tuple[float, float] = (0.0, 0.0)
        self.drag_target: Optional[HitResult] = None
        self.before_drag_doc: Optional[Document] = None
        self._has_drag_moved: bool = False
        self._pending_click_hit: Optional[HitResult] = None
        self._pending_shift_toggle: bool = False
        self._drag_snapshot: Optional[TransformSnapshot] = None
        self._snap_ref_world: Optional[tuple[float, float]] = None
        self._last_edge_click_time: float = 0.0
        self._last_edge_click_screen: tuple[float, float] = (0.0, 0.0)
        self._last_edge_click_seg: Optional[tuple[str, str, str]] = None

    @staticmethod
    def _is_hit_selected(controller: EditorController, hit: HitResult) -> bool:
        sel = controller.selection
        if hit.kind == HitTargetKind.NODE and hit.node_id:
            return sel.is_node_selected(hit.node_id)
        if hit.kind == HitTargetKind.NOTCH and hit.notch_id:
            return sel.is_notch_selected(hit.notch_id)
        if hit.kind == HitTargetKind.SEAM and hit.seam_id:
            return hit.seam_id in sel.seam_ids
        if (
            hit.kind in (HitTargetKind.EDGE, HitTargetKind.INTERNAL_PATH)
            and hit.path_id
            and hit.segment_start_node_id
            and hit.segment_end_node_id
        ):
            return sel.is_edge_selected(
                hit.piece_id,
                hit.path_id,
                start_node_id=hit.segment_start_node_id,
                end_node_id=hit.segment_end_node_id,
            )
        if hit.kind == HitTargetKind.PIECE_BODY:
            return sel.is_piece_selected(hit.piece_id)
        return False

    @staticmethod
    def _apply_hit_selection(controller: EditorController, hit: HitResult, *, add: bool, toggle: bool) -> None:
        sel = controller.selection
        if hit.kind == HitTargetKind.NODE and hit.node_id:
            if toggle:
                sel.toggle_node(hit.piece_id, hit.node_id)
            else:
                sel.select_node(hit.piece_id, hit.node_id, add=add)
        elif hit.kind == HitTargetKind.NOTCH and hit.notch_id:
            if toggle:
                sel.toggle_notch(hit.piece_id, hit.notch_id)
            else:
                sel.select_notch(hit.piece_id, hit.notch_id, add=add)
        elif hit.kind == HitTargetKind.SEAM and hit.seam_id:
            if toggle:
                sel.toggle_seam(hit.seam_id)
            else:
                sel.select_seam(hit.seam_id, add=add)
        elif (
            hit.kind in (HitTargetKind.EDGE, HitTargetKind.INTERNAL_PATH)
            and hit.path_id
            and hit.segment_start_node_id
            and hit.segment_end_node_id
        ):
            er = EdgeRange(
                piece_id=hit.piece_id,
                path_id=hit.path_id,
                start_node_id=hit.segment_start_node_id,
                end_node_id=hit.segment_end_node_id,
            )
            if toggle:
                sel.toggle_edge(er)
            else:
                sel.select_edge(er, add=add)
        elif hit.kind == HitTargetKind.PIECE_BODY:
            if toggle:
                sel.toggle_piece(hit.piece_id)
            else:
                sel.select_piece(hit.piece_id, add=add)

    def _reset_drag_state(self) -> None:
        self.is_dragging = False
        self.drag_target = None
        self.before_drag_doc = None
        self._has_drag_moved = False
        self._pending_click_hit = None
        self._pending_shift_toggle = False
        self._drag_snapshot = None
        self._snap_ref_world = None

    def on_mouse_down(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.button != "LEFT":
            return False

        # 画面ピクセル基準の許容距離を動的換算（ズーム倍率に左右されないクリック精度）
        r_node = controller.view.pixel_to_world_dist(12.0)
        r_handle = controller.view.pixel_to_world_dist(10.0)
        r_edge = controller.view.pixel_to_world_dist(8.0)

        # ヒットテスト実行
        allowed = {HitTargetKind.NODE, HitTargetKind.EDGE, HitTargetKind.INTERNAL_PATH} if event.ctrl else None
        hit = hit_test(
            controller.doc,
            event.world_pos,
            node_radius=r_node,
            handle_radius=r_handle,
            edge_threshold=r_edge,
            selected_node_ids=controller.selection.node_ids,
            allowed_kinds=allowed,
        )

        if hit:
            logger.debug(
                f"[Pattern] MouseDown Hit: kind={hit.kind}, piece={hit.piece_id}, "
                f"start_node={hit.segment_start_node_id}, end_node={hit.segment_end_node_id}, "
                f"node={hit.node_id}, t={hit.edge_t:.3f}"
            )
        else:
            logger.debug(f"[Pattern] MouseDown Hit: NONE (pos={event.world_pos})")

        self._reset_drag_state()
        self.before_drag_doc = controller.doc.clone()
        self.drag_start_world = event.world_pos
        self.drag_start_screen = event.screen_pos
        self.drag_target = hit

        # Ctrl+クリックによる辺上への新規ノード挿入
        if (
            event.ctrl
            and hit
            and hit.kind in (HitTargetKind.EDGE, HitTargetKind.INTERNAL_PATH)
            and hit.segment_start_node_id
            and hit.path_id
        ):
            piece = controller.doc.pieces.get(hit.piece_id)
            if piece:
                path = piece.paths.get(hit.path_id)
                if path:
                    seg_idx = next(
                        (i for i, n in enumerate(path.nodes) if n.id == hit.segment_start_node_id),
                        None,
                    )
                    if seg_idx is not None:
                        new_node = controller.doc.insert_node_on_segment(
                            hit.piece_id,
                            hit.path_id,
                            seg_idx=seg_idx,
                            t=max(0.01, min(0.99, hit.edge_t)),
                        )
                        if new_node is not None:
                            controller.selection.select_node(piece.id, new_node.id)
                            controller.history.record_change(
                                description="Insert Node on Edge",
                                before_doc=self.before_drag_doc,
                                after_doc=controller.doc,
                            )
                            self.before_drag_doc = None
                            self.drag_target = None
                            controller.is_dirty = True
                            return True

        if hit is None:
            if not event.shift:
                controller.selection.clear()
            self.before_drag_doc = None
            self.is_box_selecting = True
            return True

        self.is_dragging = True

        # 辺・内部線上でのダブルクリック（ノード挿入）
        if hit.kind in (HitTargetKind.EDGE, HitTargetKind.INTERNAL_PATH) and not event.ctrl and not event.shift:
            now = time.monotonic()
            dt = now - self._last_edge_click_time
            sx, sy = event.screen_pos
            lx, ly = self._last_edge_click_screen
            screen_dist = math.hypot(sx - lx, sy - ly)
            same_seg = self._last_edge_click_seg == (hit.piece_id, hit.path_id, hit.segment_start_node_id)
            is_dbl = getattr(event, "is_double_click", False) or (
                same_seg and dt <= 0.35 and screen_dist <= 6.0
            )
            if is_dbl:
                before_doc = controller.doc.clone()
                t_val = max(0.01, min(0.99, float(hit.edge_t)))
                p = controller.doc.pieces.get(hit.piece_id)
                pth = p.paths.get(hit.path_id) if (p and hit.path_id) else None
                seg_i = (
                    next((i for i, nd in enumerate(pth.nodes) if nd.id == hit.segment_start_node_id), None)
                    if pth
                    else None
                )
                new_node = (
                    controller.doc.insert_node_on_segment(
                        hit.piece_id,
                        hit.path_id or "",
                        seg_idx=seg_i,
                        t=t_val,
                    )
                    if (seg_i is not None and hit.path_id)
                    else None
                )
                if new_node:
                    controller.selection.clear()
                    controller.selection.select_node(hit.piece_id, new_node.id)
                    controller.history.record_change(
                        description="Insert Node on Edge",
                        before_doc=before_doc,
                        after_doc=controller.doc,
                    )
                    self._last_edge_click_time = 0.0
                    self._last_edge_click_seg = None
                    self._reset_drag_state()
                    controller.is_dirty = True
                    return True

        if hit.kind in (HitTargetKind.EDGE, HitTargetKind.INTERNAL_PATH) and hit.segment_start_node_id:
            self._last_edge_click_time = time.monotonic()
            self._last_edge_click_screen = event.screen_pos
            self._last_edge_click_seg = (hit.piece_id, hit.path_id or "", hit.segment_start_node_id)

        # 既に選択中の要素をクリックした場合は複数選択維持のため即解除せず、離した時点まで確定を遅延する
        if hit.kind != HitTargetKind.HANDLE and self._is_hit_selected(controller, hit):
            self._pending_click_hit = hit
            self._pending_shift_toggle = bool(event.shift)
        else:
            self._apply_hit_selection(controller, hit, add=event.shift, toggle=False)

        logger.debug(
            f"[Pattern] SelectionState after mousedown: pieces={controller.selection.piece_ids}, "
            f"edges={len(controller.selection.edge_ranges)}, nodes={len(controller.selection.node_ids)}"
        )
        return True

    def on_mouse_move(self, controller: EditorController, event: EditorEvent) -> bool:
        if self.is_box_selecting:
            p0 = self.drag_start_world
            p1 = event.world_pos
            controller.preview_points = [
                p0,
                (p1[0], p0[1]),
                p1,
                (p0[0], p1[1]),
                p0,
            ]
            controller.is_dirty = True
            return True

        if not self.is_dragging or self.drag_target is None:
            return False

        dx = event.world_pos[0] - self.drag_start_world[0]
        dy = event.world_pos[1] - self.drag_start_world[1]
        screen_dist = math.hypot(
            event.screen_pos[0] - self.drag_start_screen[0],
            event.screen_pos[1] - self.drag_start_screen[1],
        )
        world_dist = math.hypot(dx, dy)
        threshold_world = controller.view.pixel_to_world_dist(_DRAG_THRESHOLD_PX)

        if not self._has_drag_moved:
            if screen_dist <= _DRAG_THRESHOLD_PX and world_dist <= threshold_world:
                return True
            self._has_drag_moved = True
            self._pending_click_hit = None
            self._pending_shift_toggle = False

        target = self.drag_target

        if target.kind == HitTargetKind.HANDLE and target.node_id:
            piece = controller.doc.pieces.get(target.piece_id)
            if piece and target.path_id:
                path = piece.paths.get(target.path_id)
                if path:
                    node = next((n for n in path.nodes if n.id == target.node_id), None)
                    if node:
                        local_pos = piece.transform.inverse_apply(event.world_pos)
                        h_vec = (local_pos[0] - node.position[0], local_pos[1] - node.position[1])
                        if target.handle_type == "in":
                            node.in_handle = h_vec
                            if node.kind == NodeKind.SMOOTH:
                                node.out_handle = (-h_vec[0], -h_vec[1])
                        elif target.handle_type == "out":
                            node.out_handle = h_vec
                            if node.kind == NodeKind.SMOOTH:
                                node.in_handle = (-h_vec[0], -h_vec[1])
            return True

        if target.kind == HitTargetKind.NOTCH and target.notch_id:
            notch = controller.doc.notches.get(target.notch_id)
            if notch:
                piece = controller.doc.pieces.get(notch.piece_id)
                if piece:
                    path = piece.paths.get(notch.path_id)
                    if path:
                        seg_idx = next(
                            (i for i, n in enumerate(path.nodes) if n.id == notch.segment_start_id),
                            None,
                        )
                        if seg_idx is not None:
                            ctrls = path.get_segment_controls(seg_idx)
                            tbl = build_arc_table(*ctrls)
                            local_pos = piece.transform.inverse_apply(event.world_pos)
                            _, raw_t, _ = closest_point_on_table(tbl, local_pos)
                            snapped_t = controller.doc.snap_notch_t(
                                notch.piece_id,
                                notch.segment_start_id,
                                raw_t=raw_t,
                                path_id=notch.path_id,
                                snap_enabled=not event.alt,
                            )
                            controller.doc.move_notch(target.notch_id, snapped_t)
            return True

        if target.kind in (
            HitTargetKind.PIECE_BODY,
            HitTargetKind.EDGE,
            HitTargetKind.INTERNAL_PATH,
            HitTargetKind.NODE,
        ):
            if self._drag_snapshot is None:
                snap_doc = self.before_drag_doc or controller.doc
                self._drag_snapshot = capture_transform_snapshot(snap_doc, controller.selection)
                self._snap_ref_world = find_nearest_snap_reference(self._drag_snapshot, self.drag_start_world)
                if target.kind == HitTargetKind.NODE and target.node_id:
                    for ns in self._drag_snapshot.node_snapshots.values():
                        if ns.node_id == target.node_id:
                            self._snap_ref_world = ns.world_pos
                            break

            apply_translation(
                controller.doc,
                self._drag_snapshot,
                delta_world=(dx, dy),
                view=controller.view,
                snap_settings=controller.snap_settings,
                snap_reference_world=self._snap_ref_world,
                clamp_ortho=bool(event.shift),
            )
            return True

        return False

    def on_mouse_up(self, controller: EditorController, event: EditorEvent) -> bool:
        if self.is_box_selecting:
            self.is_box_selecting = False
            controller.preview_points.clear()
            p0 = self.drag_start_world
            p1 = event.world_pos
            min_x, max_x = min(p0[0], p1[0]), max(p0[0], p1[0])
            min_y, max_y = min(p0[1], p1[1]), max(p0[1], p1[1])

            # 微小ドラッグでない場合、矩形範囲内のノード -> 交差辺 -> 重心内包ピース の順で選択
            if max_x - min_x > 0.005 or max_y - min_y > 0.005:
                selected_nodes = 0
                for pid, piece in controller.doc.pieces.items():
                    for path in piece.paths.values():
                        for node in path.nodes:
                            w_pos = piece.transform.transform_point(node.position)
                            if min_x <= w_pos[0] <= max_x and min_y <= w_pos[1] <= max_y:
                                controller.selection.select_node(pid, node.id, add=True)
                                selected_nodes += 1
                if selected_nodes == 0:
                    hit_edges = find_edges_intersecting_box(
                        controller.doc,
                        box_min=(min_x, min_y),
                        box_max=(max_x, max_y),
                    )
                    if hit_edges:
                        for er in hit_edges:
                            controller.selection.select_edge(er, add=True)
                    else:
                        for pid, piece in controller.doc.pieces.items():
                            outer = piece.paths.get(piece.outer_path_id)
                            if outer and outer.nodes:
                                cx = sum(n.position[0] for n in outer.nodes) / len(outer.nodes)
                                cy = sum(n.position[1] for n in outer.nodes) / len(outer.nodes)
                                c_w = piece.transform.transform_point((cx, cy))
                            else:
                                c_w = piece.transform.transform_point((0.0, 0.0))
                            if min_x <= c_w[0] <= max_x and min_y <= c_w[1] <= max_y:
                                controller.selection.select_piece(pid, add=True)

            controller.is_dirty = True
            return True

        if not self.is_dragging:
            return False

        # ドラッグ閾値を超えずにボタンを離した場合のクリック確定（トグル解除 / 単一絞り込み）
        if not self._has_drag_moved and self._pending_click_hit is not None:
            self._apply_hit_selection(
                controller,
                self._pending_click_hit,
                add=False,
                toggle=self._pending_shift_toggle,
            )
            self._reset_drag_state()
            controller.is_dirty = True
            return True

        before_doc = self.before_drag_doc
        drag_target = self.drag_target
        moved = self._has_drag_moved
        self._reset_drag_state()

        if moved and before_doc is not None:
            if drag_target and drag_target.kind == HitTargetKind.HANDLE:
                desc = "Edit Handle"
            elif drag_target and drag_target.kind == HitTargetKind.NOTCH:
                desc = "Move Notch"
            else:
                desc = "Move Element"
            controller.history.record_change(
                description=desc,
                before_doc=before_doc,
                after_doc=controller.doc,
            )
            controller.is_dirty = True

        return True

    def on_key_press(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.key == "ESCAPE":
            self.cancel(controller)
            return True

        if event.key.upper() == "F":
            if controller.flip_selected_seams():
                return True

        if event.key.upper() == "C":
            # 1. ノードが明示的に選択されている場合 -> 選択ノードのみを対象にする
            target_node_ids = set(controller.selection.node_ids)

            # 2. ピースが選択されている場合 -> そのピースの外周輪郭ノードのみを一括対象にする
            if not target_node_ids and controller.selection.piece_ids:
                for pid in controller.selection.piece_ids:
                    p = controller.doc.pieces.get(pid)
                    if p:
                        outer = p.paths.get(p.outer_path_id)
                        if outer:
                            for n in outer.nodes:
                                target_node_ids.add(n.id)

            if not target_node_ids:
                return True

            before_doc = controller.doc.clone()
            modified = False

            for piece in controller.doc.pieces.values():
                for path in piece.paths.values():
                    nodes = path.nodes
                    n_nodes = len(nodes)
                    if n_nodes < 2:
                        continue
                    for i, node in enumerate(nodes):
                        if node.id in target_node_ids:
                            if node.kind == NodeKind.CORNER:
                                node.kind = NodeKind.SMOOTH
                                prev_pos = nodes[(i - 1) % n_nodes].position
                                next_pos = nodes[(i + 1) % n_nodes].position
                                dx = (next_pos[0] - prev_pos[0]) * 0.166
                                dy = (next_pos[1] - prev_pos[1]) * 0.166
                                node.in_handle = (-dx, -dy)
                                node.out_handle = (dx, dy)
                                modified = True
                            elif node.kind == NodeKind.SMOOTH:
                                node.kind = NodeKind.AUTO
                                node.in_handle = None
                                node.out_handle = None
                                modified = True
                            elif node.kind == NodeKind.AUTO:
                                node.kind = NodeKind.CORNER
                                node.in_handle = None
                                node.out_handle = None
                                modified = True

            if modified:
                controller.history.record_change(
                    description="Cycle Node Kind",
                    before_doc=before_doc,
                    after_doc=controller.doc,
                )
                controller.is_dirty = True
            return True

        if event.key.upper() == "W":
            return controller.subdivide_selected_edge()

        if event.key.upper() == "D" and event.shift:
            return controller.duplicate_selected_pieces()

        if event.key.upper() == "M":
            return controller.flip_selected_pieces(direction="VERTICAL" if event.shift else "HORIZONTAL")

        if event.key.upper() == "U":
            return controller.mirror_duplicate_selected() if event.shift else controller.unfold_selected()

        return False

    def cancel(self, controller: EditorController) -> None:
        if self.is_dragging and self.before_drag_doc is not None:
            controller.doc = self.before_drag_doc
        self._reset_drag_state()
        if self.is_box_selecting:
            self.is_box_selecting = False
            controller.preview_points.clear()
