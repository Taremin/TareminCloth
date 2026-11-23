# -*- coding: utf-8 -*-
"""パターン外部JSON保存・読込および下絵画像の単体テスト."""

import os
import tempfile
import unittest

from taremin_cloth.pattern.document import (
    Document,
    Notch,
    NotchKind,
    Placement,
    ReferenceImage,
)
from taremin_cloth.pattern.editor import EditorController
from taremin_cloth.pattern.renderer import DrawCommandKind, build_render_commands


class TestPatternIO(unittest.TestCase):
    """JSON入出力・下絵画像のテストケース."""

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

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = os.path.join(tmp_dir, "test_roundtrip.json")
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
