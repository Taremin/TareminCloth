# -*- coding: utf-8 -*-
"""型紙用2D幾何計算・ベジェ評価・サンプリング・アフィン変換パッケージ."""

from .bezier import (
    ArcLengthTable,
    Vec2,
    make_segment_controls,
    compute_auto_handles,
    evaluate_cubic,
    subdivide_cubic,
    split_cubic_bezier,
    build_arc_table,
    point_at_arc_length,
    arc_length_at_t,
    sample_sub_segment_points,
    closest_point_on_table,
    merge_breakpoints_by_arc_length,
    extract_sub_cubic,
    match_segment_shape,
    adjust_segment_arc_length,
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
    segment_intersection,
    segment_intersects_aabb,
    triangulate_polygon_2d,
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
    "split_cubic_bezier",
    "build_arc_table",
    "point_at_arc_length",
    "arc_length_at_t",
    "sample_sub_segment_points",
    "closest_point_on_table",
    "merge_breakpoints_by_arc_length",
    "extract_sub_cubic",
    "match_segment_shape",
    "adjust_segment_arc_length",
    "compute_seam_subdivision_count",
    "sample_edge_range_by_count",
    "signed_area",
    "ensure_ccw",
    "point_in_polygon",
    "has_self_intersection",
    "segment_intersection",
    "segment_intersects_aabb",
    "triangulate_polygon_2d",
    "Affine2D",
    "compute_bounding_box",
]
