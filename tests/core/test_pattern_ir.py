# -*- coding: utf-8 -*-
"""Milestone 2: PatternContour2D 中間表現・ノッチ同期分割・中継ガイド縫合生成の単体テスト (Blender非依存)."""
import os
import sys
import unittest
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from taremin_cloth.utils.mesh_extract import extract_sewing_topology
from taremin_cloth.engine.waypoint import WaypointManager

try:
    from taremin_cloth.utils.pattern_ir import (
        BoundarySegmentIR,
        PatternContour2D,
        SeamPairIR,
        WaypointGuideIR,
        synchronize_seam_subdivisions,
        sample_pattern_boundary_loop,
        build_sewing_loose_edges,
    )

    _HAS_PATTERN_IR = True
except ImportError:
    _HAS_PATTERN_IR = False


def _make_rect_contour(name: str, width: float, height: float, target_len: float = 0.02) -> "PatternContour2D":
    """4辺 (bottom, right, top, left) を持つ矩形 PatternContour2D を作成する."""
    corners = [
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([width, 0.0], dtype=np.float32),
        np.array([width, height], dtype=np.float32),
        np.array([0.0, height], dtype=np.float32),
    ]
    seg_names = ["bottom", "right", "top", "left"]
    segments = []
    for i in range(4):
        p0 = corners[i]
        p1 = corners[(i + 1) % 4]
        length = float(np.linalg.norm(p1 - p0))
        segments.append(
            BoundarySegmentIR(
                segment_id=f"{name}_{seg_names[i]}",
                start_uid=f"{name}_c{i}",
                end_uid=f"{name}_c{(i + 1) % 4}",
                polyline_2d=np.stack([p0, p1], axis=0),
                arc_length=length,
            )
        )
    return PatternContour2D(
        piece_name=name,
        outer_segments=segments,
        target_edge_length=target_len,
    )


@unittest.skipUnless(_HAS_PATTERN_IR, "taremin_cloth.utils.pattern_ir 未実装のためスキップ")
class TestPatternIRAndWaypointSync(unittest.TestCase):
    def test_unequal_length_seam_synchronizes_vertex_counts(self):
        """長さが異なる2辺（ギャザー・いせ込み）を縫合ペアにした際、両辺の分割数 N が完全一致すること."""
        front = _make_rect_contour("Front", width=0.20, height=0.30, target_len=0.020)
        back = _make_rect_contour("Back", width=0.20, height=0.36, target_len=0.020)

        # Front右辺 (0.30m) と Back左辺 (0.36m) を縫合ペアに指定
        seam = SeamPairIR(
            pair_id="side_seam_r",
            piece_a="Front",
            segment_a_id="Front_right",
            piece_b="Back",
            segment_b_id="Back_left",
            reverse_b=True,
        )

        synchronize_seam_subdivisions([front, back], [seam])

        seg_a = next(s for s in front.outer_segments if s.segment_id == "Front_right")
        seg_b = next(s for s in back.outer_segments if s.segment_id == "Back_left")

        # 平均長 (0.30 + 0.36) / 2 = 0.33m -> 0.33 / 0.02 = 16.5 -> 16 または 17
        self.assertIsNotNone(seg_a.target_subdivisions)
        self.assertEqual(seg_a.target_subdivisions, seg_b.target_subdivisions)

        loop_a, seg_map_a = sample_pattern_boundary_loop(front)
        loop_b, seg_map_b = sample_pattern_boundary_loop(back)

        idx_a = seg_map_a["Front_right"]
        idx_b = seg_map_b["Back_left"]
        self.assertEqual(len(idx_a), len(idx_b))
        self.assertEqual(len(idx_a), seg_a.target_subdivisions + 1)

    def test_notch_sub_segments_align_precisely(self):
        """曲線辺の途中にノッチ（合印）で分割された2区間がある場合、区間ごとに分割数が同期すること."""
        front = _make_rect_contour("Front", width=0.20, height=0.30, target_len=0.020)
        # Front_right (0.30m) を ノッチ (y=0.10m, y=0.30m) の2区間に分割
        front.outer_segments[1] = BoundarySegmentIR(
            segment_id="Front_right_lower",
            start_uid="Front_c1",
            end_uid="Front_notch_1",
            polyline_2d=np.array([[0.20, 0.0], [0.20, 0.10]], dtype=np.float32),
            arc_length=0.10,
        )
        front.outer_segments.insert(
            2,
            BoundarySegmentIR(
                segment_id="Front_right_upper",
                start_uid="Front_notch_1",
                end_uid="Front_c2",
                polyline_2d=np.array([[0.20, 0.10], [0.20, 0.30]], dtype=np.float32),
                arc_length=0.20,
            ),
        )

        back = _make_rect_contour("Back", width=0.20, height=0.30, target_len=0.020)
        # Back側ではノッチ位置が y=0.14m (下側0.14m, 上側0.16m) と非対称な場合でもノッチ同士が一致すること
        back.outer_segments[3] = BoundarySegmentIR(
            segment_id="Back_left_upper",
            start_uid="Back_c3",
            end_uid="Back_notch_1",
            polyline_2d=np.array([[0.0, 0.30], [0.0, 0.14]], dtype=np.float32),
            arc_length=0.16,
        )
        back.outer_segments.insert(
            4,
            BoundarySegmentIR(
                segment_id="Back_left_lower",
                start_uid="Back_notch_1",
                end_uid="Back_c0",
                polyline_2d=np.array([[0.0, 0.14], [0.0, 0.0]], dtype=np.float32),
                arc_length=0.14,
            ),
        )

        seams = [
            SeamPairIR(
                pair_id="seam_lower",
                piece_a="Front",
                segment_a_id="Front_right_lower",
                piece_b="Back",
                segment_b_id="Back_left_lower",
                reverse_b=True,
            ),
            SeamPairIR(
                pair_id="seam_upper",
                piece_a="Front",
                segment_a_id="Front_right_upper",
                piece_b="Back",
                segment_b_id="Back_left_upper",
                reverse_b=True,
            ),
        ]

        synchronize_seam_subdivisions([front, back], seams)
        _, map_f = sample_pattern_boundary_loop(front)
        _, map_b = sample_pattern_boundary_loop(back)

        self.assertEqual(len(map_f["Front_right_lower"]), len(map_b["Back_left_lower"]))
        self.assertEqual(len(map_f["Front_right_upper"]), len(map_b["Back_left_upper"]))
        # ノッチ頂点（lowerの終点 == upperの始点）が両パーツで正確に共有・対応していること
        self.assertEqual(map_f["Front_right_lower"][-1], map_f["Front_right_upper"][0])
        self.assertEqual(map_b["Back_left_upper"][-1], map_b["Back_left_lower"][0])

    def test_waypoint_guides_generate_valid_single_path_seams_for_solver(self):
        """中継ガイド付きの縫合ペアが A - W1 - W2 - B のLoose Edgeを生成し、既存 extract_sewing_topology と完全互換であること."""
        front = _make_rect_contour("SleeveF", width=0.20, height=0.10, target_len=0.050)
        back = _make_rect_contour("SleeveB", width=0.20, height=0.10, target_len=0.050)

        # 腕の下を回り込む2本の中継ガイドライン G1, G2
        guide1 = WaypointGuideIR(
            guide_id="underarm_g1",
            polyline_3d=np.array([[0.20, -0.04, 0.42], [0.20, -0.04, 0.52]], dtype=np.float32),
        )
        guide2 = WaypointGuideIR(
            guide_id="underarm_g2",
            polyline_3d=np.array([[0.20, 0.04, 0.42], [0.20, 0.04, 0.52]], dtype=np.float32),
        )
        seam = SeamPairIR(
            pair_id="sleeve_underarm",
            piece_a="SleeveF",
            segment_a_id="SleeveF_right",
            piece_b="SleeveB",
            segment_b_id="SleeveB_left",
            reverse_b=True,
            waypoint_guides=[guide1, guide2],
        )

        synchronize_seam_subdivisions([front, back], [seam])
        loop_f, map_f = sample_pattern_boundary_loop(front)
        loop_b, map_b = sample_pattern_boundary_loop(back)

        # 2枚のパーツの3D頂点と仮の三角形面を構築
        n_f = len(loop_f)
        n_b = len(loop_b)
        pos_f = np.column_stack([loop_f[:, 0], np.full(n_f, -0.08, dtype=np.float32), loop_f[:, 1] + 0.42])
        pos_b = np.column_stack([loop_b[:, 0], np.full(n_b, 0.08, dtype=np.float32), loop_b[:, 1] + 0.42])
        base_positions = np.vstack([pos_f, pos_b]).astype(np.float32)

        # 境界頂点をすべて面頂点として登録するためのダミー三角形群
        tris = []
        for i in range(1, n_f - 1):
            tris.append([0, i, i + 1])
        for i in range(1, n_b - 1):
            tris.append([n_f, n_f + i, n_f + i + 1])
        faces = np.array(tris, dtype=np.uint32)

        seg_global_map = {
            "SleeveF": {k: v for k, v in map_f.items()},
            "SleeveB": {k: [ idx + n_f for idx in v ] for k, v in map_b.items()},
        }

        all_positions, loose_edges = build_sewing_loose_edges(
            base_positions=base_positions,
            segment_vertex_map=seg_global_map,
            seams=[seam],
        )

        # 既存の物理コア前処理 extract_sewing_topology にそのまま投入して検証
        simple_edges, waypoint_seams = extract_sewing_topology(loose_edges, faces, all_positions)
        expected_chains = len(map_f["SleeveF_right"])

        self.assertEqual(len(simple_edges), 0, "すべて中継点付きパスとして分類されること")
        self.assertEqual(len(waypoint_seams), expected_chains, "分割された各頂点ペアごとに独立したWaypointSeamが生成されること")
        for ws in waypoint_seams:
            self.assertEqual(len(ws.waypoint_indices), 2, "G1, G2 の2つの中継点を持つこと")

        # 既存の WaypointManager で初期化・更新がエラーなく動作すること
        inv_masses = np.ones(len(all_positions), dtype=np.float32)
        for ws in waypoint_seams:
            for wp_idx in ws.waypoint_indices:
                inv_masses[wp_idx] = 0.0
        wp_mgr = WaypointManager(waypoint_seams, inv_masses)
        self.assertTrue(wp_mgr.has_active_seams)


if __name__ == "__main__":
    unittest.main()
