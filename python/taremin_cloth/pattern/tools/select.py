# -*- coding: utf-8 -*-
"""型紙エディタの選択およびドラッグ移動ツール (Blender非依存)."""

from __future__ import annotations

import math
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

        if hit.kind == HitTargetKind.NODE and hit.node_id:
            controller.selection.select_node(hit.piece_id, hit.node_id, add=event.shift)
        elif hit.kind == HitTargetKind.NOTCH and hit.notch_id:
            controller.selection.select_notch(hit.piece_id, hit.notch_id, add=event.shift)
        elif hit.kind == HitTargetKind.SEAM and hit.seam_id:
            controller.selection.select_seam(hit.seam_id, add=event.shift)
        elif hit.kind == HitTargetKind.HANDLE and hit.node_id:
            # ハンドル選択（ノード選択状態は維持）
            pass
        elif hit.kind in (HitTargetKind.PIECE_BODY, HitTargetKind.EDGE, HitTargetKind.INTERNAL_PATH):
            controller.selection.select_piece(hit.piece_id, add=event.shift)

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

        if event.key.upper() == "U":
            # 選択中のピースを取得
            target_piece_id = controller.selection.active_piece_id
            if not target_piece_id and controller.selection.piece_ids:
                target_piece_id = next(iter(controller.selection.piece_ids))
            if not target_piece_id and len(controller.doc.pieces) == 1:
                target_piece_id = next(iter(controller.doc.pieces.keys()))

            if not target_piece_id or target_piece_id not in controller.doc.pieces:
                return True

            piece = controller.doc.pieces[target_piece_id]
            outer = piece.paths.get(piece.outer_path_id)
            if not outer or len(outer.nodes) < 3:
                return True

            before_doc = controller.doc.clone()

            if event.shift:
                # 【Shift+U】: 縫合付き対称複製 (Symmetric Duplicate with Seam)
                nodes = outer.nodes
                xs = [n.position[0] for n in nodes]
                min_x, max_x = min(xs), max(xs)
                w = max_x - min_x

                sym_nodes: list[Node] = []
                for n in reversed(nodes):
                    in_h = (-n.out_handle[0], n.out_handle[1]) if n.out_handle else None
                    out_h = (-n.in_handle[0], n.in_handle[1]) if n.in_handle else None
                    sym_nodes.append(Node(
                        id=new_id(),
                        position=(-n.position[0], n.position[1]),
                        kind=n.kind,
                        in_handle=in_h,
                        out_handle=out_h,
                    ))

                sym_piece_id = new_id()
                sym_path_id = new_id()
                new_transform = piece.transform.__class__(
                    tx=piece.transform.tx + w + 0.05,
                    ty=piece.transform.ty,
                    rotation_rad=piece.transform.rotation_rad,
                    sx=piece.transform.sx,
                    sy=piece.transform.sy,
                )
                if piece.name.endswith("_L"):
                    sym_name = f"{piece.name[:-2]}_R"
                elif piece.name.endswith(".L"):
                    sym_name = f"{piece.name[:-2]}.R"
                elif piece.name.endswith("_Left"):
                    sym_name = f"{piece.name[:-5]}_Right"
                else:
                    sym_name = f"{piece.name}_Sym"

                sym_piece = Piece(
                    id=sym_piece_id,
                    name=sym_name,
                    paths={sym_path_id: Path(id=sym_path_id, role=PathRole.OUTER, nodes=sym_nodes, closed=True)},
                    outer_path_id=sym_path_id,
                    transform=new_transform,
                    target_edge_length=piece.target_edge_length,
                    layer_id=piece.layer_id,
                )
                controller.doc.pieces[sym_piece_id] = sym_piece

                orig_pl = controller.doc.placements.get(piece.id)
                if orig_pl:
                    from ..placement import flip_bone_name_left_right
                    sym_bone = flip_bone_name_left_right(orig_pl.target_bone)
                    sym_angle = -orig_pl.bone_angle_deg
                    controller.doc.placements[sym_piece_id] = Placement(
                        piece_id=sym_piece_id,
                        offset_3d=(-orig_pl.offset_3d[0], orig_pl.offset_3d[1], orig_pl.offset_3d[2]),
                        euler_3d=(orig_pl.euler_3d[0], -orig_pl.euler_3d[1], -orig_pl.euler_3d[2]),
                        target_bone=sym_bone,
                        bone_t=orig_pl.bone_t,
                        bone_angle_deg=sym_angle,
                        surface_offset=orig_pl.surface_offset,
                        is_tubular=orig_pl.is_tubular,
                    )

                best_i = 0
                best_sum_x = float("inf")
                for i in range(len(nodes)):
                    sx = nodes[i].position[0] + nodes[(i + 1) % len(nodes)].position[0]
                    if sx < best_sum_x:
                        best_sum_x = sx
                        best_i = i

                n0 = nodes[best_i]
                n1 = nodes[(best_i + 1) % len(nodes)]
                sym_n1 = sym_nodes[len(nodes) - 1 - ((best_i + 1) % len(nodes))]
                sym_n0 = sym_nodes[len(nodes) - 1 - best_i]

                seam_id = new_id()
                controller.doc.seams[seam_id] = Seam(
                    id=seam_id,
                    a=EdgeRange(piece_id=piece.id, path_id=piece.outer_path_id, start_node_id=n0.id, end_node_id=n1.id),
                    b=EdgeRange(piece_id=sym_piece_id, path_id=sym_path_id, start_node_id=sym_n1.id, end_node_id=sym_n0.id),
                )

                # ノッチの対称複製
                for notch in list(controller.doc.notches.values()):
                    if notch.piece_id == piece.id and notch.path_id == piece.outer_path_id:
                        seg_i = next((i for i, n in enumerate(nodes) if n.id == notch.segment_start_id), None)
                        if seg_i is not None:
                            sym_start_id = sym_nodes[len(nodes) - 1 - ((seg_i + 1) % len(nodes))].id
                            sym_notch_id = new_id()
                            controller.doc.notches[sym_notch_id] = Notch(
                                id=sym_notch_id,
                                piece_id=sym_piece_id,
                                path_id=sym_path_id,
                                segment_start_id=sym_start_id,
                                t=max(0.001, min(0.999, 1.0 - notch.t)),
                                kind=notch.kind,
                                depth=notch.depth,
                                width=notch.width,
                            )

                controller.selection.clear()
                controller.selection.select_piece(sym_piece_id)
                controller.history.record_change(
                    description="Symmetric Duplicate Piece",
                    before_doc=before_doc,
                    after_doc=controller.doc,
                )
                controller.is_dirty = True
                return True

            else:
                # 【U】: 単一ピースの対称展開 (Unfold Piece - 自己交差のない2n-2頂点閉ループ)
                nodes = outer.nodes
                n_orig = len(nodes)
                if n_orig < 3:
                    return True

                # 中点のX座標が最小となる辺を対称軸として特定
                best_i = 0
                best_mid_x = float("inf")
                for i in range(n_orig):
                    p0 = nodes[i].position
                    p1 = nodes[(i + 1) % n_orig].position
                    mx = 0.5 * (p0[0] + p1[0])
                    if mx < best_mid_x:
                        best_mid_x = mx
                        best_i = i

                i0 = best_i
                i1 = (best_i + 1) % n_orig
                p_a = nodes[i1].position
                p_b = nodes[i0].position

                # 直線 A -> B に対する鏡像反射ヘルパー
                vx = p_b[0] - p_a[0]
                vy = p_b[1] - p_a[1]
                v_len = math.hypot(vx, vy)
                if v_len > 1e-9:
                    nx = -vy / v_len
                    ny = vx / v_len
                else:
                    nx, ny = 1.0, 0.0

                def reflect_point(p: tuple[float, float]) -> tuple[float, float]:
                    wx = p[0] - p_a[0]
                    wy = p[1] - p_a[1]
                    d = wx * nx + wy * ny
                    return (p[0] - 2.0 * d * nx, p[1] - 2.0 * d * ny)

                def reflect_vec(v: tuple[float, float]) -> tuple[float, float]:
                    d = v[0] * nx + v[1] * ny
                    return (v[0] - 2.0 * d * nx, v[1] - 2.0 * d * ny)

                # ループ構築: i1 から時計回りに i0 まで進み、折り返しとして (i0-1 down to i1+1) の鏡像を追加
                unfolded_nodes: list[Node] = []
                orig_idx_to_new_id: dict[int, str] = {}
                mirror_idx_to_new_id: dict[int, str] = {}

                # 前半: 元のノード i1 -> i0
                curr = i1
                while True:
                    orig_n = nodes[curr]
                    nid = new_id()
                    orig_idx_to_new_id[curr] = nid
                    unfolded_nodes.append(Node(
                        id=nid,
                        position=orig_n.position,
                        kind=orig_n.kind,
                        in_handle=orig_n.in_handle,
                        out_handle=orig_n.out_handle,
                    ))
                    if curr == i0:
                        break
                    curr = (curr + 1) % n_orig

                # 対称軸両端は不動点
                mirror_idx_to_new_id[i0] = orig_idx_to_new_id[i0]
                mirror_idx_to_new_id[i1] = orig_idx_to_new_id[i1]

                # 後半: 内側ノードの鏡像 (i0 - 1 down to i1 + 1)
                curr = (i0 - 1 + n_orig) % n_orig
                while curr != i1:
                    orig_n = nodes[curr]
                    mir_pos = reflect_point(orig_n.position)
                    mir_in = reflect_vec(orig_n.out_handle) if orig_n.out_handle else None
                    mir_out = reflect_vec(orig_n.in_handle) if orig_n.in_handle else None
                    nid = new_id()
                    mirror_idx_to_new_id[curr] = nid
                    unfolded_nodes.append(Node(
                        id=nid,
                        position=mir_pos,
                        kind=orig_n.kind,
                        in_handle=mir_in,
                        out_handle=mir_out,
                    ))
                    curr = (curr - 1 + n_orig) % n_orig

                outer.nodes = unfolded_nodes

                # ノッチの再配置および整理
                old_notches = [
                    n for n in list(controller.doc.notches.values())
                    if n.piece_id == piece.id and n.path_id == outer.id
                ]
                for n in old_notches:
                    del controller.doc.notches[n.id]

                for n in old_notches:
                    seg_i = next((i for i, nd in enumerate(nodes) if nd.id == n.segment_start_id), None)
                    if seg_i is not None and seg_i != best_i:
                        # 前半辺用ノッチ
                        nid_fwd = new_id()
                        controller.doc.notches[nid_fwd] = Notch(
                            id=nid_fwd,
                            piece_id=piece.id,
                            path_id=outer.id,
                            segment_start_id=orig_idx_to_new_id[seg_i],
                            t=n.t,
                            kind=n.kind,
                            depth=n.depth,
                            width=n.width,
                        )
                        # 後半鏡像辺用ノッチ
                        next_i = (seg_i + 1) % n_orig
                        mir_start_id = mirror_idx_to_new_id[next_i]
                        nid_rev = new_id()
                        controller.doc.notches[nid_rev] = Notch(
                            id=nid_rev,
                            piece_id=piece.id,
                            path_id=outer.id,
                            segment_start_id=mir_start_id,
                            t=max(0.001, min(0.999, 1.0 - n.t)),
                            kind=n.kind,
                            depth=n.depth,
                            width=n.width,
                        )

                # Seam の再マッピングと対称軸上の Seam 除去
                axis_node_ids = {nodes[i0].id, nodes[i1].id}
                dead_seams: list[str] = []
                for sid, seam in controller.doc.seams.items():
                    seam_ers = seam.iter_unique_edge_ranges()
                    affects_this_piece = any(er.piece_id == piece.id and er.path_id == outer.id for er in seam_ers)
                    if not affects_this_piece:
                        continue

                    # 対称軸辺を参照していた場合は削除
                    has_axis = False
                    for er in seam_ers:
                        if er.piece_id == piece.id and er.path_id == outer.id:
                            if er.start_node_id in axis_node_ids and er.end_node_id in axis_node_ids:
                                has_axis = True
                                break
                    if has_axis:
                        dead_seams.append(sid)
                        continue

                    # 前半ノードIDへの付け替え
                    for er in seam_ers:
                        if er.piece_id == piece.id and er.path_id == outer.id:
                            orig_start_idx = next((i for i, nd in enumerate(nodes) if nd.id == er.start_node_id), None)
                            orig_end_idx = next((i for i, nd in enumerate(nodes) if nd.id == er.end_node_id), None)
                            if orig_start_idx is not None and orig_start_idx in orig_idx_to_new_id:
                                er.start_node_id = orig_idx_to_new_id[orig_start_idx]
                            if orig_end_idx is not None and orig_end_idx in orig_idx_to_new_id:
                                er.end_node_id = orig_idx_to_new_id[orig_end_idx]

                for sid in dead_seams:
                    if sid in controller.doc.seams:
                        del controller.doc.seams[sid]
                controller.history.record_change(
                    description="Unfold Piece",
                    before_doc=before_doc,
                    after_doc=controller.doc,
                )
                controller.is_dirty = True
                return True

        return False

    def cancel(self, controller: EditorController) -> None:
        if self.is_dragging and self.before_drag_doc is not None:
            controller.doc = self.before_drag_doc
            self.before_drag_doc = None
        self.is_dragging = False
        self.drag_target = None

