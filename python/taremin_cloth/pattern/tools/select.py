# -*- coding: utf-8 -*-
"""型紙エディタの選択およびドラッグ移動ツール (Blender非依存)."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Optional

from ..document import (
    Document,
    EdgeRange,
    Node,
    NodeKind,
    Notch,
    Path,
    PathRole,
    Piece,
    Placement,
    Seam,
)
from ..geometry import (
    build_arc_table,
    closest_point_on_table,
)
from ..ids import new_id
from ..selection import HitResult, HitTargetKind, hit_test
from ..session import compute_snap
from ...utils.logger import logger
from .base import BaseTool

if TYPE_CHECKING:
    from ..editor import EditorController, EditorEvent


class SelectTool(BaseTool):
    """要素選択およびドラッグ移動ツール."""

    name: str = "SELECT"

    def __init__(self) -> None:
        self.is_dragging: bool = False
        self.is_box_selecting: bool = False
        self.drag_start_world: tuple[float, float] = (0.0, 0.0)
        self.drag_target: Optional[HitResult] = None
        self.before_drag_doc: Optional[Document] = None
        self._last_edge_click_time: float = 0.0
        self._last_edge_click_screen: tuple[float, float] = (0.0, 0.0)
        self._last_edge_click_seg: Optional[tuple[str, str, str]] = None

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
            logger.debug(f"[Pattern] MouseDown Hit: kind={hit.kind}, piece={hit.piece_id}, start_node={hit.segment_start_node_id}, end_node={hit.segment_end_node_id}, node={hit.node_id}, t={hit.edge_t:.3f}")
        else:
            logger.debug(f"[Pattern] MouseDown Hit: NONE (pos={event.world_pos})")

        self.before_drag_doc = controller.doc.clone()
        self.drag_start_world = event.world_pos
        self.drag_target = hit

        # Ctrl+クリックによる辺上への新規ノード挿入
        if event.ctrl and hit and hit.kind in (HitTargetKind.EDGE, HitTargetKind.INTERNAL_PATH) and hit.segment_start_node_id and hit.path_id:
            piece = controller.doc.pieces.get(hit.piece_id)
            if piece:
                path = piece.paths.get(hit.path_id)
                if path:
                    nodes = path.nodes
                    seg_idx = next((i for i, n in enumerate(nodes) if n.id == hit.segment_start_node_id), None)
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
                            controller.is_dirty = True
                            return True

        if hit is None:
            if not event.shift:
                controller.selection.clear()
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
            same_seg = (self._last_edge_click_seg == (hit.piece_id, hit.path_id, hit.segment_start_node_id))
            is_dbl = getattr(event, "is_double_click", False) or (same_seg and dt <= 0.35 and screen_dist <= 6.0)
            if is_dbl:
                before_doc = controller.doc.clone()
                t_val = max(0.01, min(0.99, float(hit.edge_t)))
                p = controller.doc.pieces.get(hit.piece_id)
                pth = p.paths.get(hit.path_id) if p else None
                seg_i = next((i for i, nd in enumerate(pth.nodes) if nd.id == hit.segment_start_node_id), None) if pth else None
                new_node = controller.doc.insert_node_on_segment(
                    hit.piece_id,
                    hit.path_id,
                    seg_idx=seg_i if seg_i is not None else 0,
                    t=t_val,
                ) if seg_i is not None else None
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
                    controller.is_dirty = True
                    return True

        if hit.kind == HitTargetKind.NODE and hit.node_id:
            controller.selection.select_node(hit.piece_id, hit.node_id, add=event.shift)
        elif hit.kind == HitTargetKind.NOTCH and hit.notch_id:
            controller.selection.select_notch(hit.piece_id, hit.notch_id, add=event.shift)
        elif hit.kind == HitTargetKind.SEAM and hit.seam_id:
            controller.selection.select_seam(hit.seam_id, add=event.shift)
        elif hit.kind == HitTargetKind.HANDLE and hit.node_id:
            # ハンドル選択（ノード選択状態は維持）
            pass
        elif hit.kind == HitTargetKind.EDGE and hit.segment_start_node_id and hit.segment_end_node_id:
            er = EdgeRange(
                piece_id=hit.piece_id,
                path_id=hit.path_id,
                start_node_id=hit.segment_start_node_id,
                end_node_id=hit.segment_end_node_id,
            )
            controller.selection.select_edge(er, add=event.shift)
            self._last_edge_click_time = time.monotonic()
            self._last_edge_click_screen = event.screen_pos
            self._last_edge_click_seg = (hit.piece_id, hit.path_id, hit.segment_start_node_id)
            logger.debug(f"[Pattern] Select Edge: {er}")
        elif hit.kind in (HitTargetKind.PIECE_BODY, HitTargetKind.INTERNAL_PATH):
            controller.selection.select_piece(hit.piece_id, add=event.shift)
            logger.debug(f"[Pattern] Select Piece Body/Internal: {hit.piece_id}")

        logger.debug(f"[Pattern] SelectionState after click: pieces={controller.selection.piece_ids}, edges={len(controller.selection.edge_ranges)}, nodes={len(controller.selection.node_ids)}")

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

        target = self.drag_target
        if target.kind in (HitTargetKind.PIECE_BODY, HitTargetKind.EDGE, HitTargetKind.INTERNAL_PATH):
            # 選択中の全ピースを一括移動 (基準ノードをワールド共通グリッド/他ノードへ吸着)
            if self.before_drag_doc:
                target_pids = set(controller.selection.piece_ids)
                if not target_pids and target.piece_id:
                    target_pids.add(target.piece_id)

                # ドラッグ対象ピースのうち掴んだ位置に最も近い外周ノードを基準点とする
                ref_piece = self.before_drag_doc.pieces.get(target.piece_id)
                if ref_piece is not None:
                    ref_outer = ref_piece.paths.get(ref_piece.outer_path_id)
                    if ref_outer and ref_outer.nodes:
                        ref_w = min(
                            (ref_piece.transform.transform_point(n.position) for n in ref_outer.nodes),
                            key=lambda p: (p[0] - self.drag_start_world[0]) ** 2 + (p[1] - self.drag_start_world[1]) ** 2,
                        )
                        moving_node_ids = {
                            n.id
                            for pid in target_pids
                            if pid in controller.doc.pieces
                            for pth in controller.doc.pieces[pid].paths.values()
                            for n in pth.nodes
                        }
                        snap = compute_snap(
                            (ref_w[0] + dx, ref_w[1] + dy),
                            controller.doc,
                            view=controller.view,
                            settings=controller.snap_settings,
                            exclude_node_ids=moving_node_ids,
                        )
                        dx = snap.world_pos[0] - ref_w[0]
                        dy = snap.world_pos[1] - ref_w[1]

                for pid in target_pids:
                    piece = controller.doc.pieces.get(pid)
                    orig_piece = self.before_drag_doc.pieces.get(pid)
                    if piece and orig_piece:
                        piece.transform = piece.transform.__class__(
                            tx=orig_piece.transform.tx + dx,
                            ty=orig_piece.transform.ty + dy,
                            rotation_rad=orig_piece.transform.rotation_rad,
                            sx=orig_piece.transform.sx,
                            sy=orig_piece.transform.sy,
                        )
            return True

        elif target.kind == HitTargetKind.NODE and target.node_id:
            # 選択中の全ノードを一括移動 (ワールド座標変換 & スナップ)
            if self.before_drag_doc:
                target_nids = set(controller.selection.node_ids)
                if not target_nids and target.node_id:
                    target_nids.add(target.node_id)

                for pid, piece in controller.doc.pieces.items():
                    orig_piece = self.before_drag_doc.pieces.get(pid)
                    if not orig_piece:
                        continue
                    for path_id, path in piece.paths.items():
                        orig_path = orig_piece.paths.get(path_id)
                        if not orig_path:
                            continue
                        for i, node in enumerate(path.nodes):
                            if node.id in target_nids:
                                orig_node = orig_path.nodes[i]
                                # 開始時ノードのワールド座標
                                orig_w = orig_piece.transform.transform_point(orig_node.position)
                                target_w = (orig_w[0] + dx, orig_w[1] + dy)
                                snap = compute_snap(
                                    target_w,
                                    controller.doc,
                                    view=controller.view,
                                    settings=controller.snap_settings,
                                    exclude_node_ids=target_nids,
                                )
                                # 新ワールド座標をローカル座標へ逆変換して代入
                                node.position = piece.transform.inverse_apply(snap.world_pos)
            return True

        elif target.kind == HitTargetKind.HANDLE and target.node_id:
            # ハンドル先端のドラッグ操作
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

        elif target.kind == HitTargetKind.NOTCH and target.notch_id:
            # ノッチの辺上ドラッグ移動
            notch = controller.doc.notches.get(target.notch_id)
            if notch:
                piece = controller.doc.pieces.get(notch.piece_id)
                if piece:
                    path = piece.paths.get(notch.path_id)
                    if path:
                        nodes = path.nodes
                        seg_idx = next((i for i, n in enumerate(nodes) if n.id == notch.segment_start_id), None)
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

        return False

    def on_mouse_up(self, controller: EditorController, event: EditorEvent) -> bool:
        if self.is_box_selecting:
            self.is_box_selecting = False
            controller.preview_points.clear()
            p0 = self.drag_start_world
            p1 = event.world_pos
            min_x, max_x = min(p0[0], p1[0]), max(p0[0], p1[0])
            min_y, max_y = min(p0[1], p1[1]), max(p0[1], p1[1])

            # 微小ドラッグでない場合、矩形範囲内のノードを選択
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
                    # ノードがなければ外周重心が入るピースを選択
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

        self.is_dragging = False
        if self.before_drag_doc is not None:
            if self.drag_target and self.drag_target.kind == HitTargetKind.HANDLE:
                desc = "Edit Handle"
            elif self.drag_target and self.drag_target.kind == HitTargetKind.NOTCH:
                desc = "Move Notch"
            else:
                desc = "Move Element"
            controller.history.record_change(
                description=desc,
                before_doc=self.before_drag_doc,
                after_doc=controller.doc,
            )
            self.before_drag_doc = None

        self.drag_target = None
        return True

    def on_key_press(self, controller: EditorController, event: EditorEvent) -> bool:
        if event.key == "ESCAPE":
            self.cancel(controller)
            return True

        if event.key.upper() in ("R", "F"):
            if controller.flip_selected_seams():
                return True

        if event.key.upper() == "C":
            # 1. ノードが明示的に選択されている場合 -> 選択ノードのみを対象にする
            target_node_ids = set(controller.selection.node_ids)

            # 2. ピースが選択されている場合 -> そのピースの「外周輪郭（outer_path）」ノードのみを一括対象にする（内部線は巻き込まない）
            if not target_node_ids and controller.selection.piece_ids:
                for pid in controller.selection.piece_ids:
                    p = controller.doc.pieces.get(pid)
                    if p:
                        outer = p.paths.get(p.outer_path_id)
                        if outer:
                            for n in outer.nodes:
                                target_node_ids.add(n.id)

            if not target_node_ids:
                # 対象がなくてもイベントを消費してBlenderのサークル選択を防止
                return True

            # 幾何種別を巡回 (CORNER -> SMOOTH -> AUTO -> CORNER)
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
                                # CORNER -> SMOOTH
                                node.kind = NodeKind.SMOOTH
                                prev_pos = nodes[(i - 1) % n_nodes].position
                                next_pos = nodes[(i + 1) % n_nodes].position
                                dx = (next_pos[0] - prev_pos[0]) * 0.166
                                dy = (next_pos[1] - prev_pos[1]) * 0.166
                                node.in_handle = (-dx, -dy)
                                node.out_handle = (dx, dy)
                                modified = True
                            elif node.kind == NodeKind.SMOOTH:
                                # SMOOTH -> AUTO
                                node.kind = NodeKind.AUTO
                                node.in_handle = None
                                node.out_handle = None
                                modified = True
                            elif node.kind == NodeKind.AUTO:
                                # AUTO -> CORNER
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
            self.before_drag_doc = None
        self.is_dragging = False
        self.drag_target = None

