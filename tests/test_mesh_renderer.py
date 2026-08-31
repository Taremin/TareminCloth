"""
taremin_cloth.mesh_renderer モジュールのユニットテスト
表面（白）、裏面（赤）のUnlitラスタライズおよび赤色ピクセルカウントの検証
"""

import os
import sys
import tempfile
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)

from taremin_cloth.mesh_renderer import (
    render_mesh_to_image,
    render_mesh_to_file,
    count_red_pixels,
)


class TestMeshRenderer(unittest.TestCase):
    def test_render_frontface_plane(self):
        """カメラに向かって正面を向いている面（白）の描画検証"""
        # カメラは [0, 0, 3] から原点 [0, 0, 0] を見下ろす
        # 反時計回り(CCW)の三角形 -> 表面（白）
        positions = np.array([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)

        img = render_mesh_to_image(
            positions, faces,
            width=200, height=200,
            camera_pos=(0.0, 0.0, 3.0),
            camera_target=(0.0, 0.0, 0.0)
        )

        red_count = count_red_pixels(img)
        self.assertEqual(red_count, 0, "表面のみ描画されたメッシュで赤色ピクセルが検出されました")

        # 白色ピクセルが存在することを確認
        arr = np.array(img)
        white_count = np.sum((arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 200))
        self.assertTrue(white_count > 100, "表面の白ピクセルが十分に描画されていません")

    def test_render_backface_plane(self):
        """カメラに対して裏面を向いている面（赤）の描画検証"""
        # 時計回り(CW)の三角形 -> 裏面（赤）
        positions = np.array([
            [-1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)

        img = render_mesh_to_image(
            positions, faces,
            width=200, height=200,
            camera_pos=(0.0, 0.0, 3.0),
            camera_target=(0.0, 0.0, 0.0)
        )

        red_count = count_red_pixels(img)
        self.assertTrue(red_count > 100, "裏面の赤ピクセルが検出されませんでした")

    def test_render_mesh_to_file(self):
        """画像ファイル保存の検証"""
        positions = np.array([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)

        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = os.path.join(tmpdir, "render_test.png")
            out_file, red_count = render_mesh_to_file(png_path, positions, faces, width=100, height=100)
            self.assertTrue(os.path.exists(out_file))
            self.assertTrue(os.path.getsize(out_file) > 0)
            self.assertEqual(red_count, 0)


if __name__ == "__main__":
    unittest.main()
