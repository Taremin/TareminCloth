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

    def test_render_scene_with_sewing_and_colliders(self):
        """縫合線（水色）およびコライダー（スレートグレー）の同時描画検証"""
        from taremin_cloth.mesh_renderer import render_scene_to_file
        from PIL import Image

        positions = np.array([
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 1.5, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        sewing_springs = np.array([[2, 3]], dtype=np.uint32)
        colliders = np.array([
            [-1.5, -1.5, -0.5, -0.5, -1.5, -0.5, -1.0, -0.5, -0.5]
        ], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = os.path.join(tmpdir, "scene_test.png")
            render_scene_to_file(
                png_path, positions, faces,
                sewing_springs=sewing_springs,
                mesh_colliders=colliders,
                width=100, height=100,
                camera_pos=(0.0, 0.0, 4.0),
                camera_target=(0.0, 0.0, 0.0),
            )
            self.assertTrue(os.path.exists(png_path))
            img = Image.open(png_path)
            arr = np.array(img)

            # 縫合エッジ (水色 [0, 204, 255])
            sew_mask = (arr[:, :, 0] == 0) & (arr[:, :, 1] == 204) & (arr[:, :, 2] == 255)
            self.assertTrue(np.sum(sew_mask) > 0, "縫合エッジ（水色）が検出されませんでした")

            # コライダー (スレートグレー [140, 160, 180] 前後)
            col_mask = (arr[:, :, 2] > arr[:, :, 0]) & (arr[:, :, 2] > 70) & (arr[:, :, 0] > 40)
            self.assertTrue(np.sum(col_mask) > 10, "コライダーが検出されませんでした")

    def test_create_animation_file(self):
        """連番フレーム画像からの APNG / GIF 生成検証"""
        from taremin_cloth.mesh_renderer import create_animation_file
        from PIL import Image

        # 差分のある2フレーム (表面と裏面)
        positions1 = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        positions2 = np.array([[-1.0, -1.0, 0.0], [0.0, 1.0, 0.0], [1.0, -1.0, 0.0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)

        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "f1.png")
            f2 = os.path.join(tmpdir, "f2.png")
            render_mesh_to_file(f1, positions1, faces, width=80, height=80)
            render_mesh_to_file(f2, positions2, faces, width=80, height=80)

            # 1. APNG
            apng_path = os.path.join(tmpdir, "anim.png")
            create_animation_file([f1, f2], apng_path, fps=10)
            self.assertTrue(os.path.exists(apng_path))
            self.assertTrue(os.path.getsize(apng_path) > 0)

            with Image.open(apng_path) as anim_img:
                self.assertTrue(getattr(anim_img, "is_animated", False))
                self.assertEqual(getattr(anim_img, "n_frames", 1), 2)

            # 2. GIF
            gif_path = os.path.join(tmpdir, "anim.gif")
            create_animation_file([f1, f2], gif_path, fps=10)
            self.assertTrue(os.path.exists(gif_path))
            with Image.open(gif_path) as gif_img:
                self.assertTrue(getattr(gif_img, "is_animated", False))
                self.assertEqual(getattr(gif_img, "n_frames", 1), 2)

    def test_primitive_collider_mesh_generation(self):
        """球・カプセル・平面のプリミティブコライダーメッシュ自動生成の検証"""
        from taremin_cloth.mesh_renderer import (
            generate_sphere_mesh,
            generate_capsule_mesh,
            generate_plane_mesh,
            convert_colliders_to_mesh,
        )

        # 1. 球
        s_mesh = generate_sphere_mesh([0, 0, 0], radius=1.0)
        self.assertGreater(len(s_mesh), 0)
        self.assertEqual(s_mesh.shape[1], 9)

        # 2. カプセル
        c_mesh = generate_capsule_mesh([0, 0, 0], [0, 0, 1.0], radius=0.2)
        self.assertGreater(len(c_mesh), 0)
        self.assertEqual(c_mesh.shape[1], 9)

        # 3. 平面
        p_mesh = generate_plane_mesh([0, 0, 0], [0, 0, 1.0], size=2.0)
        self.assertEqual(len(p_mesh), 2) # 2枚の三角形

        # 4. 一括変換
        cols_record = [
            {"type": "sphere", "center": [0, 1, 0], "radius": 0.5},
            {"type": "capsule", "point_a": [1, 0, 0], "point_b": [1, 1, 0], "radius": 0.2},
            {"type": "plane", "point": [0, 0, -1], "normal": [0, 0, 1]},
        ]
        all_tris = convert_colliders_to_mesh(cols_record)
        self.assertIsNotNone(all_tris)
        self.assertGreater(len(all_tris), 10)

    def test_heatmap_computations(self):
        """速度・歪み・法線ヒートマップ計算の検証"""
        from taremin_cloth.mesh_renderer import (
            colormap_jet,
            compute_velocity_colors,
            compute_strain_colors,
            compute_normal_colors,
        )

        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.uint32)
        rest_lens = np.array([0.8, 1.4, 0.8], dtype=np.float32) # 引き伸ばされている

        # 1. 速度ヒートマップ
        vels = np.array([[0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]], dtype=np.float32)
        v_colors = compute_velocity_colors(vels)
        self.assertEqual(v_colors.shape, (3, 3))
        self.assertEqual(v_colors.dtype, np.uint8)

        # 2. 歪みヒートマップ
        s_colors = compute_strain_colors(positions, edges, rest_lens)
        self.assertEqual(s_colors.shape, (3, 3))
        self.assertEqual(s_colors.dtype, np.uint8)

        # 3. 法線カラーマップ
        n_colors = compute_normal_colors(positions, faces)
        self.assertEqual(n_colors.shape, (3, 3))
        self.assertEqual(n_colors.dtype, np.uint8)

    def test_render_with_vertex_colors_and_extra_lines(self):
        """頂点カラー（ヒートマップ）および追加3Dライン（SDF BBOX）描画の検証"""
        from taremin_cloth.mesh_renderer import render_scene_to_file
        from PIL import Image

        positions = np.array([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        v_colors = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ], dtype=np.uint8)
        extra_lines = np.array([
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 255.0, 170.0, 0.0], # ゴールドの線
        ], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_p = os.path.join(tmpdir, "test_vc.png")
            render_scene_to_file(
                out_p, positions, faces,
                vertex_colors=v_colors,
                extra_lines=extra_lines,
                width=100, height=100,
                camera_pos=(0, 0, 3), camera_target=(0, 0, 0)
            )
            self.assertTrue(os.path.exists(out_p))
            img = Image.open(out_p)
            arr = np.array(img)
            # ゴールド (255, 170, 0) の線が存在すること
            gold_mask = (arr[:, :, 0] == 255) & (arr[:, :, 1] == 170) & (arr[:, :, 2] == 0)
            self.assertTrue(np.any(gold_mask), "追加3Dライン（ゴールド）が描画されていません")

    def test_wireframe_only_and_voxels_render(self):
        """wireframe_only モードとボクセル描画、および render_scene_to_image の動作検証"""
        from taremin_cloth.mesh_renderer import render_scene_to_image
        from PIL import Image

        positions = np.array([
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)

        # 鮮やかなシアンのボクセル [0, 240, 255]
        voxels = np.array([
            [0.0, 0.0, 0.0, 0.4, 0.0, 240.0, 255.0],
        ], dtype=np.float32)

        # 1. render_scene_to_image で直接 PIL.Image を生成
        img = render_scene_to_image(
            positions, faces,
            voxels=voxels,
            wireframe_only=True,
            width=100, height=100,
            camera_pos=(0, 0, 3), camera_target=(0, 0, 0)
        )
        self.assertIsInstance(img, Image.Image)
        arr = np.array(img)

        # ワイヤーフレームのオレンジ [255, 140, 20] が検出されること
        orange_mask = (arr[:, :, 0] == 255) & (arr[:, :, 1] == 140) & (arr[:, :, 2] == 20)
        self.assertTrue(np.any(orange_mask), "布のオレンジワイヤーフレームが描画されていません")

        # ボクセルのシアン [0, >100, >100] が検出されること
        cyan_mask = (arr[:, :, 0] == 0) & (arr[:, :, 1] > 100) & (arr[:, :, 2] > 100)
        self.assertTrue(np.any(cyan_mask), "ボクセルのシアンが描画されていません")

        # 2. voxel_screen_size=2.0 でスクリーンスペース固定サイズ描画
        img_ss = render_scene_to_image(
            positions, faces,
            voxels=voxels,
            voxel_screen_size=2.0,
            wireframe_only=True,
            width=100, height=100,
            camera_pos=(0, 0, 3), camera_target=(0, 0, 0)
        )
        arr_ss = np.array(img_ss)
        cyan_mask_ss = (arr_ss[:, :, 0] == 0) & (arr_ss[:, :, 1] == 240) & (arr_ss[:, :, 2] == 255)
        # 2x2 正方形なのでちょうど 4 ピクセル描画されること
        self.assertEqual(np.count_nonzero(cyan_mask_ss), 4, "スクリーンスペース 2x2 ボクセルのピクセル数が4ではありません")

    def test_extract_sdf_surface_voxels_adaptive(self):
        """非等方AABBに対する適応的サンプリング（adaptive stride）の検証"""
        from taremin_cloth.mesh_renderer import extract_sdf_surface_voxels
        from types import SimpleNamespace

        # モックの BakeResult: すねを模した極端な縦長直方体 (10cm x 60cm x 15cm, res=64)
        # base_d = [1.56mm, 9.38mm, 2.34mm]
        res = 64
        tex = np.zeros((res, res, res, 2), dtype=np.float16)
        # 表面付近のボクセルを設定 (dist=0.001, alpha=0.5)
        tex[:, :, :, 0] = 0.001
        tex[:, :, :, 1] = 0.5

        info_row = [0.0] * 20
        info_row[0:3] = [-0.05, 0.0, -0.075] # min
        info_row[4:7] = [0.05, 0.6, 0.075]   # max (box: 0.1 x 0.6 x 0.15)
        info_row[7] = float(res)
        info_row[19] = 0.01 # weight_threshold

        mock_bake = SimpleNamespace(
            texture_bytes=tex.tobytes(),
            bone_names=["lower_leg"],
            bone_infos=[info_row],
            width=res,
            height=res,
            depth=res,
        )

        # 1. adaptive=False の場合: stride は全軸一律
        res_non_adaptive = extract_sdf_surface_voxels(mock_bake, stride=2, adaptive=False)
        self.assertEqual(len(res_non_adaptive), 1)
        _, pts_non_ad, vox_sz_non_ad, _ = res_non_adaptive[0]
        self.assertGreater(len(pts_non_ad), 0)
        # vox_sz のアスペクト比は元のAABBのまま (約 1:6:1.5)
        self.assertAlmostEqual(vox_sz_non_ad[1] / vox_sz_non_ad[0], 6.0, delta=0.1)

        # 2. adaptive=True の場合: 粗い長軸(Y)に合わせ、他軸が間引かれて等方化される
        res_adaptive = extract_sdf_surface_voxels(mock_bake, stride=1, adaptive=True)
        _, pts_ad, vox_sz_ad, _ = res_adaptive[0]
        self.assertGreater(len(pts_ad), 0)
        # vox_sz は等方化されているため、アスペクト比がほぼ 1:1 に近い
        ratio_yx = vox_sz_ad[1] / vox_sz_ad[0]
        self.assertTrue(0.8 <= ratio_yx <= 1.25, f"等方化されたアスペクト比が期待外: {ratio_yx}")

        # 3. target_spacing を指定した場合
        target_sp = 0.02 # 20mm
        res_target = extract_sdf_surface_voxels(mock_bake, target_spacing=target_sp, adaptive=True)
        _, pts_tgt, vox_sz_tgt, _ = res_target[0]
        self.assertGreater(len(pts_tgt), 0)
        for d in vox_sz_tgt:
            self.assertTrue(0.015 <= d <= 0.025, f"target_spacing に一致していません: {d}")


if __name__ == "__main__":
    unittest.main()
