# -*- coding: utf-8 -*-
"""2D型紙（パターン）モデリング・製図および衣装ビルドシステム."""

from .ids import new_id, is_valid_id
from .units import m_to_cm, cm_to_m, format_cm, parse_length_input
from .document import (
    DOCUMENT_VERSION,
    Anchor,
    AnchorKind,
    Document,
    EdgeRange,
    Node,
    NodeKind,
    Notch,
    NotchKind,
    Path,
    PathRole,
    Piece,
    Placement,
    ReferenceImage,
    Seam,
    SeamMode,
    SeamRouting,
    Waypoint,
    normalize_angle_deg,
)
from .history import HistoryManager, HistoryEntry
from .placement import (
    QUICK_PLACEMENT_PRESETS,
    estimate_body_reference_pure,
    get_quick_placement_transform,
)
from .migration import migrate_curve_objects_to_document

__all__ = [
    "new_id",
    "is_valid_id",
    "m_to_cm",
    "cm_to_m",
    "format_cm",
    "parse_length_input",
    "DOCUMENT_VERSION",
    "Anchor",
    "AnchorKind",
    "Document",
    "EdgeRange",
    "Node",
    "NodeKind",
    "Notch",
    "NotchKind",
    "Path",
    "PathRole",
    "Piece",
    "Placement",
    "ReferenceImage",
    "Seam",
    "SeamMode",
    "SeamRouting",
    "Waypoint",
    "normalize_angle_deg",
    "HistoryManager",
    "HistoryEntry",
    "QUICK_PLACEMENT_PRESETS",
    "estimate_body_reference_pure",
    "get_quick_placement_transform",
    "migrate_curve_objects_to_document",
]
