# -*- coding: utf-8 -*-
"""Phase 4 Step 4.1: 3D配置プリセットおよび計算の単体テスト."""

import math
import unittest

from taremin_cloth.pattern.document import Document, Node, Path, PathRole, Piece, Placement
from taremin_cloth.pattern.editor import EditorController, EditorEvent
from taremin_cloth.pattern.placement import (
    compute_bone_aligned_frame,
    compute_tubular_waypoints_for_pair,
    estimate_body_reference_pure,
    flip_bone_name_left_right,
    get_quick_placement_transform,
    project_2d_points_to_bone_plane,
)


class TestPatternPlacement(unittest.TestCase):
    """3D配置プリセットと素体参照推定のテストケース."""

    def test_estimate_body_reference_pure(self) -> None:
        """素体寸法の計算が標準身長に対して妥当な値を返す."""
        ref = estimate_body_reference_pure(body_height=1.7)
        self.assertAlmostEqual(ref["chest_z"], 1.7 * 0.70)
        self.assertAlmostEqual(ref["arm_z"], 1.7 * 0.72)
        self.assertAlmostEqual(ref["waist_z"], 1.7 * 0.50)
        self.assertGreater(ref["thickness_y"], 0.10)

    def test_quick_placement_presets(self) -> None:
        """各プリセットが幾何学的に正しい位置と回転角を返す."""
        ref = estimate_body_reference_pure(body_height=1.7)

        # 前身頃: Yがマイナス (手前), X=0, Z=胸の高さ, X軸90度回転
        off_cf, rot_cf = get_quick_placement_transform("CHEST_FRONT", ref=ref)
        self.assertEqual(off_cf[0], 0.0)
        self.assertLess(off_cf[1], 0.0)
        self.assertAlmostEqual(off_cf[2], ref["chest_z"])
        self.assertAlmostEqual(rot_cf[0], math.pi * 0.5)

        # 後身頃: Yがプラス (奥), X=0, Z=胸の高さ, Z軸180度反転
        off_bk, rot_bk = get_quick_placement_transform("BACK", ref=ref)
        self.assertEqual(off_bk[0], 0.0)
        self.assertGreater(off_bk[1], 0.0)
        self.assertAlmostEqual(off_bk[2], ref["chest_z"])
        self.assertAlmostEqual(rot_bk[2], math.pi)

        # 左袖: Xがマイナス, Z軸-90度
        off_la, rot_la = get_quick_placement_transform("ARM_LEFT", ref=ref)
        self.assertLess(off_la[0], 0.0)
        self.assertAlmostEqual(rot_la[2], -math.pi * 0.5)

        # 右袖: Xがプラス, Z軸+90度
        off_ra, rot_ra = get_quick_placement_transform("ARM_RIGHT", ref=ref)
        self.assertGreater(off_ra[0], 0.0)
        self.assertAlmostEqual(rot_ra[2], math.pi * 0.5)

    def test_flip_bone_name_left_right(self) -> None:
        """左右ボーン名の各種表記パターン反転テスト."""
        self.assertEqual(flip_bone_name_left_right("UpperArm.L"), "UpperArm.R")
        self.assertEqual(flip_bone_name_left_right("UpperArm.R"), "UpperArm.L")
        self.assertEqual(flip_bone_name_left_right("ForeArm_L"), "ForeArm_R")
        self.assertEqual(flip_bone_name_left_right("ForeArm_R"), "ForeArm_L")
        self.assertEqual(flip_bone_name_left_right("HandLeft"), "HandRight")
        self.assertEqual(flip_bone_name_left_right("HandRight"), "HandLeft")
        self.assertEqual(flip_bone_name_left_right("leg_left"), "leg_right")
        self.assertEqual(flip_bone_name_left_right("leg_right"), "leg_left")
        self.assertEqual(flip_bone_name_left_right("Spine"), "Spine")
        self.assertEqual(flip_bone_name_left_right(""), "")

    def test_compute_bone_aligned_frame_vertical(self) -> None:
        """垂直ボーン（胴体・背骨など）に対する正面配置フレーム計算テスト."""
        head = (0.0, 0.0, 1.0)
        tail = (0.0, 0.0, 1.6)

        plane_center, ex, normal, ez, axis_origin = compute_bone_aligned_frame(
            head,
            tail,
            bone_t=0.5,
            bone_angle_deg=0.0,
            surface_offset=0.10,
        )

        # 軸中心点: Z = 1.3
        self.assertAlmostEqual(axis_origin[2], 1.3, places=4)
        self.assertAlmostEqual(axis_origin[0], 0.0, places=4)
        self.assertAlmostEqual(axis_origin[1], 0.0, places=4)

        # 正面角度 (0度) の法線は Blender正面 (-Y方向)
        self.assertAlmostEqual(normal[0], 0.0, places=3)
        self.assertAlmostEqual(normal[1], -1.0, places=3)
        self.assertAlmostEqual(normal[2], 0.0, places=3)

        # 平面中心は Y = -0.10, Z = 1.3
        self.assertAlmostEqual(plane_center[1], -0.10, places=3)
        self.assertAlmostEqual(plane_center[2], 1.3, places=3)

        # ez はワールド +Z
        self.assertGreater(ez[2], 0.99)

        # ex はワールド +X (右方向)
        self.assertAlmostEqual(ex[0], 1.0, places=3)

    def test_compute_bone_aligned_frame_inverted_bone(self) -> None:
        """下向きボーン（腕・脚など、head Z > tail Z）でも縦軸 ez が上向きに補正されることの検証."""
        head = (0.3, 0.0, 1.4)
        tail = (0.3, 0.0, 1.0)  # 下向き

        plane_center, ex, normal, ez, axis_origin = compute_bone_aligned_frame(
            head,
            tail,
            bone_t=0.5,
            bone_angle_deg=0.0,
            surface_offset=0.08,
        )

        # ez は必ずワールド +Z 側を向く（型紙の上下逆転を防止）
        self.assertGreater(ez[2], 0.99)

    def test_compute_bone_aligned_frame_rotation(self) -> None:
        """ロール角度 (bone_angle_deg) による法線ベクトルの回転検証."""
        head = (0.0, 0.0, 0.0)
        tail = (0.0, 0.0, 1.0)

        # 90度回転 (右側面向き)
        _, _, normal_90, _, _ = compute_bone_aligned_frame(
            head,
            tail,
            bone_angle_deg=90.0,
            surface_offset=0.10,
        )
        self.assertAlmostEqual(normal_90[0], 1.0, places=3)
        self.assertAlmostEqual(normal_90[1], 0.0, places=3)

        # 180度回転 (背面向き: +Y方向)
        _, _, normal_180, _, _ = compute_bone_aligned_frame(
            head,
            tail,
            bone_angle_deg=180.0,
            surface_offset=0.10,
        )
        self.assertAlmostEqual(normal_180[0], 0.0, places=3)
        self.assertAlmostEqual(normal_180[1], 1.0, places=3)

    def test_symmetric_duplicate_bone_placement(self) -> None:
        """Shift+U（対称複製）実行時にボーン名が反転し、角度が符号反転されることの検証."""
        doc = Document.new()
        p_id = "piece_left"
        path_id = "path_outer"

        nodes = [
            Node(id="n0", position=(0.0, 0.0)),
            Node(id="n1", position=(0.2, 0.0)),
            Node(id="n2", position=(0.2, 0.3)),
            Node(id="n3", position=(0.0, 0.3)),
        ]
        piece = Piece(
            id=p_id,
            name="Sleeve_L",
            paths={path_id: Path(id=path_id, role=PathRole.OUTER, nodes=nodes, closed=True)},
            outer_path_id=path_id,
        )
        doc.pieces[p_id] = piece
        doc.placements[p_id] = Placement(
            piece_id=p_id,
            target_bone="UpperArm.L",
            bone_t=0.4,
            bone_angle_deg=45.0,
            surface_offset=0.06,
            is_tubular=True,
        )

        ctrl = EditorController(doc=doc)
        ctrl.selection.select_piece(p_id)

        # Shift+U のシミュレーション
        u_event = EditorEvent(
            screen_pos=(0, 0),
            is_press=True,
            is_release=False,
            is_move=False,
            button="",
            shift=True,
            ctrl=False,
            alt=False,
            key="U",
        )
        handled = ctrl.handle_event(u_event)
        self.assertTrue(handled)

        # 複製ピースの存在確認
        self.assertEqual(len(ctrl.doc.pieces), 2)
        sym_p = next((p for pid, p in ctrl.doc.pieces.items() if pid != p_id), None)
        self.assertIsNotNone(sym_p)
        self.assertEqual(sym_p.name, "Sleeve_R")

        # 複製ピースの Placement 検証
        sym_pl = ctrl.doc.placements.get(sym_p.id)
        self.assertIsNotNone(sym_pl)
        self.assertEqual(sym_pl.target_bone, "UpperArm.R")
        self.assertAlmostEqual(sym_pl.bone_t, 0.4)
        self.assertAlmostEqual(sym_pl.bone_angle_deg, -45.0)
        self.assertAlmostEqual(sym_pl.surface_offset, 0.06)
        self.assertTrue(sym_pl.is_tubular)

    def test_project_2d_points_and_tubular_waypoints(self) -> None:
        """2D頂点列のボーン追従3D投影および筒モード裏側中継点算出の単体テスト."""
        head = (0.0, 0.0, 1.0)
        tail = (0.0, 0.0, 1.6)
        pts_2d = [(0.0, 0.0), (0.2, 0.0), (0.2, 0.4), (0.0, 0.4)]

        world_3d = project_2d_points_to_bone_plane(
            pts_2d,
            head_3d=head,
            tail_3d=tail,
            bone_t=0.5,
            bone_angle_deg=0.0,
            surface_offset=0.10,
        )
        self.assertEqual(world_3d.shape, (4, 3))
        # 全頂点が Y = -0.10 平面上に配置され、重心が Z = 1.3 になること
        for p in world_3d:
            self.assertAlmostEqual(float(p[1]), -0.10, places=4)
        self.assertAlmostEqual(float(world_3d[:, 2].mean()), 1.3, places=4)

        # 左右両端の頂点から裏側 (+Y側) の迂回中継点 (wp1, wp2) を算出
        wp1, wp2 = compute_tubular_waypoints_for_pair(
            world_3d[0],
            world_3d[1],
            head_3d=head,
            tail_3d=tail,
            bone_t=0.5,
            bone_angle_deg=0.0,
            surface_offset=0.10,
        )
        # ボーン裏側 (+Y > 0) に配置され、X符号が va -> wp1 -> wp2 -> vb で交差しないこと
        self.assertGreater(float(wp1[1]), 0.0)
        self.assertGreater(float(wp2[1]), 0.0)
        self.assertLess(float(wp1[0]), 0.0)
        self.assertGreater(float(wp2[0]), 0.0)


if __name__ == "__main__":
    unittest.main()

