# -*- coding: utf-8 -*-
"""Blender 既存 2D Curve オブジェクトから Document へのマイグレーションモジュール."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .document import Document, Node, NodeKind, Path, PathRole, Piece, Placement
from .geometry import Affine2D
from .ids import new_id

if TYPE_CHECKING:
    import bpy


def migrate_curve_objects_to_document(
    curve_objects: list[bpy.types.Object],
    *,
    target_doc: Optional[Document] = None,
) -> Document:
    """既存の 2D Curve オブジェクト群を走査し、Document に変換・統合して返す."""
    doc = target_doc or Document.new()

    for obj in curve_objects:
        if obj.type != "CURVE" or not obj.data:
            continue

        curve_data = obj.data
        if not curve_data.splines:
            continue

        piece_id = new_id()
        paths: dict[str, Path] = {}
        outer_path_id = ""

        for spline_idx, spline in enumerate(curve_data.splines):
            path_id = new_id()
            nodes: list[Node] = []

            if spline.type == "BEZIER":
                for pt in spline.bezier_points:
                    px = float(pt.co.x)
                    py = float(pt.co.y)
                    # ハンドル相対ベクトル
                    in_h = (float(pt.handle_left.x - pt.co.x), float(pt.handle_left.y - pt.co.y))
                    out_h = (float(pt.handle_right.x - pt.co.x), float(pt.handle_right.y - pt.co.y))

                    kind = NodeKind.CORNER
                    if pt.handle_left_type in ("ALIGNED", "AUTO") or pt.handle_right_type in ("ALIGNED", "AUTO"):
                        kind = NodeKind.SMOOTH

                    nodes.append(Node(
                        id=new_id(),
                        position=(px, py),
                        kind=kind,
                        in_handle=in_h if kind != NodeKind.CORNER else None,
                        out_handle=out_h if kind != NodeKind.CORNER else None,
                    ))
            elif spline.type == "POLY":
                for pt in spline.points:
                    nodes.append(Node(
                        id=new_id(),
                        position=(float(pt.co.x), float(pt.co.y)),
                        kind=NodeKind.CORNER,
                    ))

            if not nodes:
                continue

            role = PathRole.OUTER if spline_idx == 0 else PathRole.INTERNAL
            path = Path(
                id=path_id,
                role=role,
                nodes=nodes,
                closed=bool(spline.use_cyclic_u),
            )
            paths[path_id] = path
            if spline_idx == 0:
                outer_path_id = path_id

        if not outer_path_id:
            continue

        # オブジェクトの 2D 位置
        tx = float(obj.location.x)
        ty = float(obj.location.y)
        rot_z = float(obj.rotation_euler.z)

        piece = Piece(
            id=piece_id,
            name=obj.name,
            paths=paths,
            outer_path_id=outer_path_id,
            transform=Affine2D(tx=tx, ty=ty, rotation_rad=rot_z),
        )
        doc.pieces[piece_id] = piece

        # 3D配置プロキシパラメータ
        if hasattr(obj, "taremin_pattern") and obj.taremin_pattern.is_pattern_piece:
            pat = obj.taremin_pattern
            doc.placements[piece_id] = Placement(
                piece_id=piece_id,
                offset_3d=(float(pat.arrangement_offset[0]), float(pat.arrangement_offset[1]), float(pat.arrangement_offset[2])),
                euler_3d=(float(pat.arrangement_euler[0]), float(pat.arrangement_euler[1]), float(pat.arrangement_euler[2])),
            )

    return doc

