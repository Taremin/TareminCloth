# -*- coding: utf-8 -*-
"""Phase 4 Step 4.2 & 4.3: パターン外部JSON保存・読込、下絵画像、Curveマイグレーションの単体テスト."""

import os
import unittest
from unittest.mock import MagicMock

from taremin_cloth.pattern.document import (
    Document,
    NodeKind,
    Notch,
    NotchKind,
    Placement,
    ReferenceImage,
)
from taremin_cloth.pattern.editor import EditorController
from taremin_cloth.pattern.migration import migrate_curve_objects_to_document
from taremin_cloth.pattern.renderer import DrawCommandKind, build_render_commands


class TestPatternIOMigration(unittest.TestCase):
    """JSON入出力・下絵画像・Curveマイグレーションのテストケース."""

    def test_json_file_export_import_roundtrip(self) -> None:
        """複合Documentを外部JSONファイルに保存し、同一内容で読み込めること."""
        doc = Document.new(name="TestGarment")
        piece = doc.add_rectangle_piece((-0.1, -0.1), (0.1, 0.1), name="Front")
        p_id = piece.id

        # ノッチ、配置、下絵画像を追加
        doc.notches["notch_1"] = Notch(
            id="notch_1",
            piece_id=p_id,
            path_id=doc.pieces[p_id].outer_path_id,
            segment_start_id="node_0",
            t=0.5,
            kind=NotchKind.V,
        )
        doc.placements[p_id] = Placement(
            piece_id=p_id,
            offset_3d=(0.0, -0.15, 1.15),
            euler_3d=(1.57, 0.0, 0.0),
        )
        doc.reference_image = ReferenceImage(
            image_path="test_pattern.png",
            offset=(-0.2, -0.2),
            width=0.4,
            opacity=0.6,
            visible=True,
        )

        test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scratch", "test_roundtrip.json"))
        try:
            doc.save_to_file(test_path)
            self.assertTrue(os.path.exists(test_path))

            loaded = Document.load_from_file(test_path)
            self.assertEqual(loaded.name, "TestGarment")
            self.assertIn(p_id, loaded.pieces)
            self.assertEqual(loaded.pieces[p_id].name, "Front")
            self.assertIn("notch_1", loaded.notches)
            self.assertIn(p_id, loaded.placements)
            self.assertAlmostEqual(loaded.placements[p_id].offset_3d[1], -0.15)

            # 下絵画像の復元確認
            self.assertIsNotNone(loaded.reference_image)
            self.assertEqual(loaded.reference_image.image_path, "test_pattern.png")
            self.assertAlmostEqual(loaded.reference_image.width, 0.4)
            self.assertAlmostEqual(loaded.reference_image.opacity, 0.6)
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

    def test_reference_image_render_command(self) -> None:
        """下絵画像が設定されている場合、IMAGE描画コマンドが発行されること."""
        doc = Document.new()
        doc.reference_image = ReferenceImage(
            image_path="sketch.png",
            offset=(0.0, 0.0),
            width=0.5,
            opacity=0.7,
            visible=True,
        )
        controller = EditorController(doc=doc)
        commands = build_render_commands(doc, controller, screen_size=(800, 600))

        image_cmds = [c for c in commands if c.kind == DrawCommandKind.IMAGE]
        self.assertEqual(len(image_cmds), 1)
        self.assertEqual(image_cmds[0].image_path, "sketch.png")
        self.assertAlmostEqual(image_cmds[0].color[3], 0.7)
        self.assertEqual(len(image_cmds[0].points), 4)

    def test_migrate_curve_objects_to_document(self) -> None:
        """Blender CurveモックオブジェクトからDocumentへ正常に変換されること."""
        # モックBezierスプライン
        pt0 = MagicMock()
        pt0.co.x, pt0.co.y = 0.0, 0.0
        pt0.handle_left.x, pt0.handle_left.y = -0.02, 0.0
        pt0.handle_right.x, pt0.handle_right.y = 0.02, 0.0
        pt0.handle_left_type = "ALIGNED"
        pt0.handle_right_type = "ALIGNED"

        pt1 = MagicMock()
        pt1.co.x, pt1.co.y = 0.2, 0.0
        pt1.handle_left.x, pt1.handle_left.y = 0.18, 0.0
        pt1.handle_right.x, pt1.handle_right.y = 0.22, 0.0
        pt1.handle_left_type = "VECTOR"
        pt1.handle_right_type = "VECTOR"

        pt2 = MagicMock()
        pt2.co.x, pt2.co.y = 0.2, 0.3
        pt2.handle_left.x, pt2.handle_left.y = 0.2, 0.28
        pt2.handle_right.x, pt2.handle_right.y = 0.2, 0.32
        pt2.handle_left_type = "VECTOR"
        pt2.handle_right_type = "VECTOR"

        spline = MagicMock()
        spline.type = "BEZIER"
        spline.bezier_points = [pt0, pt1, pt2]
        spline.use_cyclic_u = True

        mock_obj = MagicMock()
        mock_obj.type = "CURVE"
        mock_obj.name = "Bodice_Front"
        mock_obj.location.x, mock_obj.location.y = 0.5, 0.5
        mock_obj.rotation_euler.z = 0.0
        mock_obj.data.splines = [spline]
        mock_obj.taremin_pattern.is_pattern_piece = True
        mock_obj.taremin_pattern.arrangement_offset = (0.0, -0.2, 1.2)
        mock_obj.taremin_pattern.arrangement_euler = (1.57, 0.0, 0.0)

        doc = migrate_curve_objects_to_document([mock_obj])
        self.assertEqual(len(doc.pieces), 1)

        piece = next(iter(doc.pieces.values()))
        self.assertEqual(piece.name, "Bodice_Front")
        self.assertEqual(piece.transform.tx, 0.5)
        self.assertEqual(piece.transform.ty, 0.5)

        outer = piece.paths[piece.outer_path_id]
        self.assertEqual(len(outer.nodes), 3)
        self.assertEqual(outer.nodes[0].kind, NodeKind.SMOOTH)
        self.assertEqual(outer.nodes[1].kind, NodeKind.CORNER)

        # 3D配置パラメータの移行確認
        self.assertIn(piece.id, doc.placements)
        self.assertAlmostEqual(doc.placements[piece.id].offset_3d[1], -0.2)


class TestReferenceImageSerialization(unittest.TestCase):
    """下絵参照画像 (ReferenceImage) のシリアライズ・複製・履歴テスト。"""

    def test_reference_image_roundtrip_and_clone(self):
        from taremin_cloth.pattern.document import Document, ReferenceImage

        doc = Document()
        doc.reference_image = ReferenceImage(
            image_path="//textures/pattern_ref.png",
            offset=(0.15, -0.25),
            width=1.4,
            aspect_ratio=1.25,
            opacity=0.65,
            visible=False,
        )

        payload = doc.to_dict()
        restored = Document.from_dict(payload)
        self.assertIsNotNone(restored.reference_image)
        self.assertEqual(restored.reference_image.image_path, "//textures/pattern_ref.png")
        self.assertEqual(restored.reference_image.offset, (0.15, -0.25))
        self.assertAlmostEqual(restored.reference_image.width, 1.4)
        self.assertAlmostEqual(restored.reference_image.aspect_ratio, 1.25)
        self.assertAlmostEqual(restored.reference_image.opacity, 0.65)
        self.assertFalse(restored.reference_image.visible)

        cloned = restored.clone()
        cloned.reference_image.opacity = 0.2
        self.assertAlmostEqual(restored.reference_image.opacity, 0.65)


if __name__ == "__main__":
    unittest.main()
