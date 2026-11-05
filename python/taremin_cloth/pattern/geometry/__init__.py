# -*- coding: utf-8 -*-
"""型紙用2D幾何計算・ベジェ評価・サンプリング・アフィン変換パッケージ."""

from .bezier import (
    ArcLengthTable,
    Vec2,
    make_segment_controls,
    compute_auto_handles,
    evaluate_cubic,
    subdivide_cubic,
    build_arc_table,
    point_at_arc_length,
    closest_point_on_table,
)
from .sampling import (
    compute_seam_subdivision_count,
    sample_edge_range_by_count,
)
from .polygon import (
    signed_area,
    ensure_ccw,
    point_in_polygon,
    has_self_intersection,
)
from .transform import (
    Affine2D,
    compute_bounding_box,
)

__all__ = [
    "ArcLengthTable",
    "Vec2",
    "make_segment_controls",
    "compute_auto_handles",
    "evaluate_cubic",
    "subdivide_cubic",
    "build_arc_table",
    "point_at_arc_length",
    "closest_point_on_table",
    "compute_seam_subdivision_count",
    "sample_edge_range_by_count",
    "signed_area",
    "ensure_ccw",
    "point_in_polygon",
    "has_self_intersection",
    "Affine2D",
    "compute_bounding_box",
]
