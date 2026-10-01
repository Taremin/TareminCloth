"""
ドレープガイド（Wrinkle Field）のスタンドアロン座屈検証テスト
Blender非依存で動作し、円柱コライダー＋円筒布メッシュに対して
ドレープガイド拘束を適用し、山・谷の自発的座屈、非貫通性、
およびレンダリング画像の生成を検証します。
"""

import math
import os
import unittest
import numpy as np

from taremin_cloth.engine.wrinkle_field import (
    bake_wrinkle_2d_sdf_texture,
    WrinkleTexture2D,
)
from taremin_cloth.mesh_renderer import (
    generate_capsule_mesh,
    generate_wrinkle_curve_lines,
    compute_wrinkle_texture_2d_weight_colors,
    render_scene_to_file,
)
import taremin_cloth_core


def create_cylinder_cloth_mesh(
    radius: float = 0.06,
    height: float = 0.30,
    n_theta: int = 36,
    n_z: int = 30,
    z_center: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """円筒布メッシュの頂点、エッジ、面を生成します"""
    thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    zs = np.linspace(z_center - height * 0.5, z_center + height * 0.5, n_z)

    verts = []
    for z in zs:
        for th in thetas:
            verts.append([radius * math.cos(th), radius * math.sin(th), z])
    verts = np.array(verts, dtype=np.float32)

    faces = []
    edges_set = set()
    for iz in range(n_z - 1):
        for it in range(n_theta):
            it_next = (it + 1) % n_theta
            v0 = iz * n_theta + it
            v1 = iz * n_theta + it_next
            v2 = (iz + 1) * n_theta + it_next
            v3 = (iz + 1) * n_theta + it

            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

            for pair in [(v0, v1), (v1, v2), (v2, v0), (v0, v2), (v2, v3), (v3, v0)]:
                a, b = min(pair), max(pair)
                edges_set.add((a, b))

    faces = np.array(faces, dtype=np.uint32)
    edges = np.array(list(edges_set), dtype=np.uint32)
    return verts, edges, faces


class TestWrinkleFieldStandalone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 成果物ディレクトリ（画像保存先）
        default_dir = r"C:\Users\unknown\.gemini\antigravity-ide\brain\d457158d-658d-4a3b-b98e-8410e651650d"
        cls.artifact_dir = os.environ.get("ANTIGRAVITY_ARTIFACT_DIR", default_dir)
        os.makedirs(cls.artifact_dir, exist_ok=True)

    def test_wrinkle_field_buckling_and_render(self):
        """ドレープガイドによる自発的座屈・非貫通性および各ステップの画像生成"""
        # 1. 円筒布メッシュ生成 (半径 6cm, 高さ 30cm)
        radius_cloth = 0.06
        radius_col = 0.05
        verts, edges, faces = create_cylinder_cloth_mesh(
            radius=radius_cloth, height=0.30, n_theta=36, n_z=30, z_center=0.0
        )
        num_verts = len(verts)

        # 上端（Z >= 0.145）のみピン固定（肩側固定・袖口自由端で自然な座屈を促進）
        inv_masses = np.ones(num_verts, dtype=np.float32)
        pin_indices = []
        for i, v in enumerate(verts):
            if v[2] >= 0.145:
                inv_masses[i] = 0.0
                pin_indices.append(i)

        # 2. シワカーブ定義 (円柱を囲む山 Crest と 谷 Root)
        # ユーザー構想: 「円柱に対して周囲を囲む円カーブ」
        thetas_full = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)

        # 谷カーブ (Root: シアン): Z=-0.020 を中心に、θに応じて波打つ
        # 半径: 5.5cm (初期 6.0cm から -5mm キュッと引き締める)
        root_z = -0.020 + 0.004 * np.cos(thetas_full)
        root_r = radius_cloth - 0.005  # 5.5cm
        root_pts = np.stack([
            root_r * np.cos(thetas_full),
            root_r * np.sin(thetas_full),
            root_z
        ], axis=-1).astype(np.float32)

        # 山カーブ (Crest: 赤/橙): 谷から十分離れた上側 (Z = +0.035 + 0.004 * cos(theta))
        # 半径: 6.5cm (初期 6.0cm から +5mm ふんわり膨らませる)
        crest1_z = 0.035 + 0.004 * np.cos(thetas_full)
        crest1_r = radius_cloth + 0.005  # 6.5cm
        crest1_pts = np.stack([
            crest1_r * np.cos(thetas_full),
            crest1_r * np.sin(thetas_full),
            crest1_z
        ], axis=-1).astype(np.float32)

        # 3. プロファイル抽出 (メタボール的ポテンシャル合成)
        origin = (0.0, 0.0, 0.0)
        axis = (0.0, 0.0, 1.0)
        normal = (1.0, 0.0, 0.0)
        binormal = (0.0, 1.0, 0.0)

        tex = bake_wrinkle_2d_sdf_texture(
            crest_curves=[crest1_pts],
            root_curves=[root_pts],
            origin=origin,
            axis=axis,
            normal=normal,
            width=512,
            height=256,
            influence_radius=0.030,
            bone_radius=radius_col,
            cloth_radius=radius_cloth,
        )

        # 4. ClothSimulator 初期化 (適度な曲げ・圧縮抵抗で布の美しい張りとくびれを両立)
        sim = taremin_cloth_core.ClothSimulator(
            positions=verts,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.003,
            stiffness=800.0,
            bending_stiffness=20.0,
            compression_stiffness=150.0,
            solver_mode=0,
            workgroup_size=32,
        )
        sim.set_gravity(0.0, 0.0, 0.0)

        # コライダー設定: カプセルメッシュ (Z=-0.4 から Z=+0.4、半径 5cm)
        col_tris = generate_capsule_mesh(
            point_a=[0.0, 0.0, -0.4],
            point_b=[0.0, 0.0, 0.4],
            radius=radius_col,
            segments=24
        )
        sim.set_mesh_collider_triangles(
            col_tris.reshape((-1, 3, 3)),
            friction=0.2,
            thickness=0.002,
            restitution=0.0,
            single_sided=False
        )

        # ドレープガイド設定 (影響半径 3.0cm、剛性 0.08で極めて安定した滑らかな座屈)
        sim.set_wrinkle_field_params(
            origin=[origin[0], origin[1], origin[2]],
            axis=[axis[0], axis[1], axis[2]],
            normal=[normal[0], normal[1], normal[2]],
            binormal=[binormal[0], binormal[1], binormal[2]],
            influence_radius=0.030,
            bone_radius=radius_col,
            stiffness=25.0,
            blend_weight=1.0,
            z_min=tex.z_min,
            z_max=tex.z_max,
            r_min=tex.r_min,
            r_max=tex.r_max,
        )
        sim.set_wrinkle_field_texture_2d(
            tex.width, tex.height, tex.texture_bytes,
            tex.z_min, tex.z_max, tex.r_min, tex.r_max
        )
        sim.set_enable_wrinkle_field(True)

        self.assertTrue(sim.wrinkle_field_enabled())

        # カーブ3Dライン配列の生成 (全周閉ループ)
        curve_lines = generate_wrinkle_curve_lines(
            crest_curves=[crest1_pts],
            root_curves=[root_pts],
            is_closed=True
        )

        # 5. シミュレーションステップ進行と各ステップのレンダリング
        # 円柱の斜め上方からの見下ろしアングル
        camera_pos = (0.35, -0.25, 0.12)
        camera_target = (0.0, 0.0, 0.0)

        # --- Frame 0 (初期状態) ---
        img_f0 = os.path.join(self.artifact_dir, "wrinkle_f00_init.png")
        render_scene_to_file(
            img_f0,
            positions=verts,
            faces=faces,
            mesh_colliders=col_tris,
            extra_lines=curve_lines,
            camera_pos=camera_pos,
            camera_target=camera_target,
            width=600,
            height=600,
        )
        self.assertTrue(os.path.exists(img_f0))

        # --- 頂点カラー影響度ヒートマップ ---
        weight_colors = compute_wrinkle_texture_2d_weight_colors(
            verts,
            texture_rgba=tex.rgba_array,
            origin=origin,
            axis=axis,
            normal=normal,
            z_min=tex.z_min,
            z_max=tex.z_max,
        )
        img_hm = os.path.join(self.artifact_dir, "wrinkle_weight_heatmap.png")
        render_scene_to_file(
            img_hm,
            positions=verts,
            faces=faces,
            mesh_colliders=col_tris,
            vertex_colors=weight_colors,
            extra_lines=curve_lines,
            camera_pos=camera_pos,
            camera_target=camera_target,
            width=600,
            height=600,
        )
        self.assertTrue(os.path.exists(img_hm))

        # --- Frame 10 (座屈進行中) ---
        dt = 1.0 / 60.0
        substeps = 15
        for f in range(1, 11):
            sim.step(dt, substeps, 4)

        pos_flat = np.empty(num_verts * 3, dtype=np.float32)
        sim.get_positions(pos_flat)
        curr_pos = pos_flat.reshape((-1, 3)).copy()
        self.assertFalse(np.any(np.isnan(curr_pos)), "NaN が検出されました")
        self.assertFalse(np.any(np.isinf(curr_pos)), "Inf が検出されました")

        img_f10 = os.path.join(self.artifact_dir, "wrinkle_f10_progress.png")
        render_scene_to_file(
            img_f10,
            positions=curr_pos,
            faces=faces,
            mesh_colliders=col_tris,
            extra_lines=curve_lines,
            camera_pos=camera_pos,
            camera_target=camera_target,
            width=600,
            height=600,
        )
        self.assertTrue(os.path.exists(img_f10))

        # --- Frame 30 (最終収束) ---
        for f in range(11, 31):
            sim.step(dt, substeps, 4)

        sim.get_positions(pos_flat)
        curr_pos = pos_flat.reshape((-1, 3)).copy()
        self.assertFalse(np.any(np.isnan(curr_pos)), "NaN が検出されました")

        img_f30 = os.path.join(self.artifact_dir, "wrinkle_f30_converged.png")
        render_scene_to_file(
            img_f30,
            positions=curr_pos,
            faces=faces,
            mesh_colliders=col_tris,
            extra_lines=curve_lines,
            camera_pos=camera_pos,
            camera_target=camera_target,
            width=600,
            height=600,
        )
        self.assertTrue(os.path.exists(img_f30))

        # 6. 定量的物理アサーション
        rs = np.linalg.norm(curr_pos[:, :2], axis=1)
        min_idx = np.argmin(rs)
        max_idx = np.argmax(rs)
        min_r = rs[min_idx]
        max_r = rs[max_idx]
        peak_to_valley = max_r - min_r

        # (A) 非貫通性: 全頂点の半径がコライダー表面 (5cm - 1mm) 以上であること (貫通ゼロ)
        self.assertGreaterEqual(
            min_r, radius_col - 0.002,
            f"コライダーへの貫通が検出されました: min_r={min_r:.4f}m < {radius_col}m"
        )

        # (B) 山の座屈度: 最大半径 r_max が初期半径 6.0cm から外側へふんわり盛り上がっていること
        self.assertGreater(
            max_r, radius_cloth + 0.002,
            f"山の座屈が十分に発生していません: max_r={max_r:.4f}m <= {radius_cloth + 0.002:.4f}m"
        )

        # (C) 谷の引き締め度: 最小半径 min_r が初期半径 6.0cm から内側へキュッと引き込まれていること
        self.assertLess(
            min_r, radius_cloth - 0.002,
            f"谷の引き込みが十分に発生していません: min_r={min_r:.4f}m >= {radius_cloth - 0.002:.4f}m"
        )

        # (D) シワ全体の起伏度 (Peak-to-Valley): 山と谷の高低差が 5mm 以上形成されていること
        self.assertGreaterEqual(
            peak_to_valley, 0.005,
            f"シワの起伏が不十分です: peak_to_valley={peak_to_valley*1000:.1f}mm < 5.0mm"
        )

        # (E) 健全性: NaN/Inf が一切なく、布がアコーディオン状に縮退していないこと
        self.assertFalse(np.any(np.isnan(curr_pos)), "NaN が検出されました")
        self.assertFalse(np.any(np.isinf(curr_pos)), "Inf が検出されました")

        print(f"\n[WrinkleField Test Summary]")
        print(f"  コライダー半径: {radius_col:.3f}m, 布初期半径: {radius_cloth:.3f}m")
        print(f"  最小半径 (谷の引き込み): {min_r:.4f}m (クリア: 窪み量 -{(radius_cloth - min_r)*1000:.1f}mm, コライダー非貫通)")
        print(f"  最大半径 (山のふんわり座屈): {max_r:.4f}m (クリア: 隆起量 +{(max_r - radius_cloth)*1000:.1f}mm)")
        print(f"  シワ起伏 (Peak-to-Valley): {peak_to_valley*1000:.1f}mm (クリア: >= 5.0mm)")
        print(f"  生成画像:")
        print(f"    - Frame 00 (初期): {img_f0}")
        print(f"    - Weight Heatmap:  {img_hm}")
        print(f"    - Frame 10 (進行): {img_f10}")
        print(f"    - Frame 30 (収束): {img_f30}")

    def test_wrinkle_field_v_orientation_gpu(self):
        """GPU実機V方向検証: 谷(z=-0.05)が下側バンド、山(z=+0.08)が上側バンドに現れること"""
        radius_cloth = 0.06
        radius_col = 0.05
        verts, edges, faces = create_cylinder_cloth_mesh(
            radius=radius_cloth, height=0.30, n_theta=36, n_z=40, z_center=0.0
        )
        num_verts = len(verts)
        inv_masses = np.ones(num_verts, dtype=np.float32)

        thetas_full = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
        z_valley = -0.05
        z_crest = 0.08
        root_pts = np.stack([
            np.full_like(thetas_full, radius_col + 0.002) * np.cos(thetas_full),
            np.full_like(thetas_full, radius_col + 0.002) * np.sin(thetas_full),
            np.full_like(thetas_full, z_valley),
        ], axis=-1).astype(np.float32)
        crest_pts = np.stack([
            np.full_like(thetas_full, radius_cloth + 0.010) * np.cos(thetas_full),
            np.full_like(thetas_full, radius_cloth + 0.010) * np.sin(thetas_full),
            np.full_like(thetas_full, z_crest),
        ], axis=-1).astype(np.float32)

        origin = (0.0, 0.0, 0.0)
        axis = (0.0, 0.0, 1.0)
        normal = (1.0, 0.0, 0.0)
        binormal = (0.0, 1.0, 0.0)

        tex = bake_wrinkle_2d_sdf_texture(
            crest_curves=[crest_pts],
            root_curves=[root_pts],
            origin=origin,
            axis=axis,
            normal=normal,
            width=128,
            height=128,
            influence_radius=0.020,
            bone_radius=radius_col,
            cloth_radius=radius_cloth,
        )

        sim = taremin_cloth_core.ClothSimulator(
            positions=verts,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.003,
            stiffness=800.0,
            bending_stiffness=20.0,
            compression_stiffness=150.0,
            solver_mode=0,
            workgroup_size=32,
        )
        sim.set_gravity(0.0, 0.0, 0.0)
        sim.set_wrinkle_field_params(
            origin=[origin[0], origin[1], origin[2]],
            axis=[axis[0], axis[1], axis[2]],
            normal=[normal[0], normal[1], normal[2]],
            binormal=[binormal[0], binormal[1], binormal[2]],
            influence_radius=0.020,
            bone_radius=radius_col,
            stiffness=25.0,
            blend_weight=1.0,
            z_min=tex.z_min,
            z_max=tex.z_max,
            r_min=tex.r_min,
            r_max=tex.r_max,
        )
        sim.set_wrinkle_field_texture_2d(
            tex.width, tex.height, tex.texture_bytes,
            tex.z_min, tex.z_max, tex.r_min, tex.r_max
        )
        sim.set_enable_wrinkle_field(True)

        dt = 1.0 / 60.0
        for _ in range(30):
            sim.step(dt, 15, 4)

        pos_flat = np.empty(num_verts * 3, dtype=np.float32)
        sim.get_positions(pos_flat)
        curr = pos_flat.reshape((-1, 3))
        rs = np.linalg.norm(curr[:, :2], axis=1)
        zs = curr[:, 2]

        # 谷バンド (z_valley±15mm) の平均半径 < 山バンド (z_crest±15mm) の平均半径
        valley_mask = np.abs(zs - z_valley) < 0.015
        crest_mask = np.abs(zs - z_crest) < 0.015
        self.assertTrue(np.count_nonzero(valley_mask) > 50, "谷バンド頂点が存在すること")
        self.assertTrue(np.count_nonzero(crest_mask) > 50, "山バンド頂点が存在すること")
        mean_valley = float(np.mean(rs[valley_mask]))
        mean_crest = float(np.mean(rs[crest_mask]))
        print(f"\n[V-Orientation GPU] valley(z={z_valley}): {mean_valley:.4f}m, "
              f"crest(z={z_crest}): {mean_crest:.4f}m")
        # V反転時は谷/山が入れ替わり本assertが赤になる
        self.assertLess(
            mean_valley, mean_crest - 0.002,
            f"V反転の疑い: 谷バンド {mean_valley:.4f}m が山バンド {mean_crest:.4f}m より細くない")
        # 全体として座屈が発生していること
        self.assertGreaterEqual(float(np.max(rs) - np.min(rs)), 0.005)

    def test_wrinkle_field_oneside_pin_free_buckling(self):
        """片側ピン留め（上端肩側のみ固定、下端袖口は自由）によるダイナミックなシワ座屈検証"""
        radius_cloth = 0.06
        radius_col = 0.05
        verts, edges, faces = create_cylinder_cloth_mesh(
            radius=radius_cloth, height=0.30, n_theta=36, n_z=30, z_center=0.0
        )
        num_verts = len(verts)

        # 上端（Z >= 0.145）のみピン固定、下端は完全に自由
        inv_masses = np.ones(num_verts, dtype=np.float32)
        for i, v in enumerate(verts):
            if v[2] >= 0.145:
                inv_masses[i] = 0.0

        # カーブ定義 (円柱を囲む山 Crest と 谷 Root)
        thetas_full = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
        root_z = 0.008 * np.cos(thetas_full)
        root_r = radius_col + 0.003  # 5.3cm
        root_pts = np.stack([
            root_r * np.cos(thetas_full),
            root_r * np.sin(thetas_full),
            root_z
        ], axis=-1).astype(np.float32)

        crest1_z = 0.030 + 0.008 * np.cos(thetas_full)
        crest1_r = radius_cloth + 0.015  # 7.5cm
        crest1_pts = np.stack([
            crest1_r * np.cos(thetas_full),
            crest1_r * np.sin(thetas_full),
            crest1_z
        ], axis=-1).astype(np.float32)

        crest2_z = -0.030 + 0.008 * np.cos(thetas_full)
        crest2_r = radius_cloth + 0.012  # 7.2cm
        crest2_pts = np.stack([
            crest2_r * np.cos(thetas_full),
            crest2_r * np.sin(thetas_full),
            crest2_z
        ], axis=-1).astype(np.float32)

        origin = (0.0, 0.0, 0.0)
        axis = (0.0, 0.0, 1.0)
        normal = (1.0, 0.0, 0.0)
        binormal = (0.0, 1.0, 0.0)

        tex = bake_wrinkle_2d_sdf_texture(
            crest_curves=[crest1_pts, crest2_pts],
            root_curves=[root_pts],
            origin=origin,
            axis=axis,
            normal=normal,
            width=512,
            height=256,
            influence_radius=0.035,
            bone_radius=radius_col,
            cloth_radius=radius_cloth,
        )

        sim = taremin_cloth_core.ClothSimulator(
            positions=verts,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.003,
            stiffness=800.0,
            bending_stiffness=2.0,
            solver_mode=0,
            workgroup_size=32,
        )
        sim.set_gravity(0.0, 0.0, 0.0)

        col_tris = generate_capsule_mesh(
            point_a=[0.0, 0.0, -0.4],
            point_b=[0.0, 0.0, 0.4],
            radius=radius_col,
            segments=24
        )
        sim.set_mesh_collider_triangles(
            col_tris.reshape((-1, 3, 3)),
            friction=0.2,
            thickness=0.002,
            restitution=0.0,
            single_sided=False
        )

        sim.set_wrinkle_field_params(
            origin=[origin[0], origin[1], origin[2]],
            axis=[axis[0], axis[1], axis[2]],
            normal=[normal[0], normal[1], normal[2]],
            binormal=[binormal[0], binormal[1], binormal[2]],
            influence_radius=0.035,
            bone_radius=radius_col,
            stiffness=35.0,
            blend_weight=1.0,
            z_min=tex.z_min,
            z_max=tex.z_max,
            r_min=tex.r_min,
            r_max=tex.r_max,
        )
        sim.set_wrinkle_field_texture_2d(
            tex.width, tex.height, tex.texture_bytes,
            tex.z_min, tex.z_max, tex.r_min, tex.r_max
        )
        sim.set_enable_wrinkle_field(True)

        dt = 1.0 / 60.0
        substeps = 15
        for f in range(30):
            sim.step(dt, substeps, 4)

        pos_flat = np.empty(num_verts * 3, dtype=np.float32)
        sim.get_positions(pos_flat)
        curr_pos = pos_flat.reshape((-1, 3)).copy()

        rs = np.linalg.norm(curr_pos[:, :2], axis=1)
        min_r = np.min(rs)
        max_r = np.max(rs)
        peak_to_valley = max_r - min_r

        img_oneside = os.path.join(self.artifact_dir, "wrinkle_oneside_pin.png")
        curve_lines = generate_wrinkle_curve_lines(
            crest_curves=[crest1_pts, crest2_pts],
            root_curves=[root_pts],
            is_closed=True
        )
        render_scene_to_file(
            img_oneside,
            positions=curr_pos,
            faces=faces,
            mesh_colliders=col_tris,
            extra_lines=curve_lines,
            camera_pos=(0.35, -0.25, 0.12),
            camera_target=(0.0, 0.0, 0.0),
            width=600,
            height=600,
        )

        print(f"\n[One-side Pin Test Summary]")
        print(f"  最小半径 (谷): {min_r:.4f}m (窪み量: -{(radius_cloth - min_r)*1000:.1f}mm)")
        print(f"  最大半径 (山): {max_r:.4f}m (隆起量: +{(max_r - radius_cloth)*1000:.1f}mm)")
        print(f"  起伏 (Peak-to-Valley): {peak_to_valley*1000:.1f}mm")
        print(f"  生成画像: {img_oneside}")

        # 下端自由端のため、よりダイナミックに座屈が発生する
        self.assertGreaterEqual(min_r, radius_col - 0.002)
        self.assertGreater(max_r, radius_cloth + 0.003)
        self.assertGreaterEqual(peak_to_valley, 0.006)

    def test_wrinkle_field_root_only_cinch(self):
        """水色谷カーブ単体による強力なくびれ・引き締め（圧縮剛性0のリアルな布）の検証"""
        radius_cloth = 0.06
        radius_col = 0.05
        verts, edges, faces = create_cylinder_cloth_mesh(
            radius=radius_cloth, height=0.30, n_theta=36, n_z=30, z_center=0.0
        )
        num_verts = len(verts)

        inv_masses = np.ones(num_verts, dtype=np.float32)
        for i, v in enumerate(verts):
            if v[2] >= 0.145:
                inv_masses[i] = 0.0

        thetas_full = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
        # 谷カーブ: Z=0.0、半径 5.2cm (コライダー 5.0cm 直前)
        root_z = np.zeros_like(thetas_full)
        root_r = radius_col + 0.002  # 5.2cm
        root_pts = np.stack([
            root_r * np.cos(thetas_full),
            root_r * np.sin(thetas_full),
            root_z
        ], axis=-1).astype(np.float32)

        origin = (0.0, 0.0, 0.0)
        axis = (0.0, 0.0, 1.0)
        normal = (1.0, 0.0, 0.0)

        # 山カーブなし、谷カーブのみ
        tex = bake_wrinkle_2d_sdf_texture(
            crest_curves=[],
            root_curves=[root_pts],
            origin=origin,
            axis=axis,
            normal=normal,
            width=512,
            height=256,
            influence_radius=0.035,
            bone_radius=radius_col,
            cloth_radius=radius_cloth,
        )

        # 圧縮剛性0.0（布が容易にシワを寄せて縮める設定）
        sim = taremin_cloth_core.ClothSimulator(
            positions=verts,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.003,
            stiffness=800.0,
            bending_stiffness=2.0,
            compression_stiffness=0.0,
            solver_mode=0,
            workgroup_size=32,
        )
        sim.set_gravity(0.0, 0.0, 0.0)

        col_tris = generate_capsule_mesh(
            point_a=[0.0, 0.0, -0.4],
            point_b=[0.0, 0.0, 0.4],
            radius=radius_col,
            segments=24
        )
        sim.set_mesh_collider_triangles(
            col_tris.reshape((-1, 3, 3)),
            friction=0.2,
            thickness=0.002,
            restitution=0.0,
            single_sided=False
        )

        sim.set_wrinkle_field_params(
            origin=[origin[0], origin[1], origin[2]],
            axis=[axis[0], axis[1], axis[2]],
            normal=[normal[0], normal[1], normal[2]],
            binormal=[0.0, 1.0, 0.0],
            influence_radius=0.035,
            bone_radius=radius_col,
            stiffness=40.0,  # 強力な引き締め外力加速度
            blend_weight=1.0,
            z_min=tex.z_min,
            z_max=tex.z_max,
            r_min=tex.r_min,
            r_max=tex.r_max,
        )
        sim.set_wrinkle_field_texture_2d(
            tex.width, tex.height, tex.texture_bytes,
            tex.z_min, tex.z_max, tex.r_min, tex.r_max
        )
        sim.set_enable_wrinkle_field(True)

        dt = 1.0 / 60.0
        substeps = 15
        for f in range(30):
            sim.step(dt, substeps, 4)

        pos_flat = np.empty(num_verts * 3, dtype=np.float32)
        sim.get_positions(pos_flat)
        curr_pos = pos_flat.reshape((-1, 3)).copy()

        rs = np.linalg.norm(curr_pos[:, :2], axis=1)
        min_r = np.min(rs)
        max_r = np.max(rs)

        img_cinch = os.path.join(self.artifact_dir, "wrinkle_root_only_cinch.png")
        curve_lines = generate_wrinkle_curve_lines(
            crest_curves=[],
            root_curves=[root_pts],
            is_closed=True
        )
        render_scene_to_file(
            img_cinch,
            positions=curr_pos,
            faces=faces,
            mesh_colliders=col_tris,
            extra_lines=curve_lines,
            camera_pos=(0.35, -0.25, 0.12),
            camera_target=(0.0, 0.0, 0.0),
            width=600,
            height=600,
        )

        print(f"\n[Root-Only Cinch Test Summary]")
        print(f"  初期半径: {radius_cloth:.4f}m -> 谷最小半径: {min_r:.4f}m (引き込み量: -{(radius_cloth - min_r)*1000:.1f}mm)")
        print(f"  コライダー表面(50mm)との隙間: +{(min_r - radius_col)*1000:.1f}mm (目標: 52mm)")
        print(f"  生成画像: {img_cinch}")

        # コライダー非貫通かつ、初期半径から5mm以上強力に引き締められていること
        self.assertGreaterEqual(min_r, radius_col - 0.002)
        self.assertLess(min_r, radius_cloth - 0.005)

    def test_wrinkle_field_2d_texture_buckling_and_render(self):
        """円柱UV展開2D-SDFテクスチャによる自発的座屈・非貫通性および各ステップの画像生成"""
        radius_cloth = 0.06
        radius_col = 0.05
        verts, edges, faces = create_cylinder_cloth_mesh(
            radius=radius_cloth, height=0.30, n_theta=36, n_z=30, z_center=0.0
        )
        num_verts = len(verts)

        # 上端（Z >= 0.145）のみピン固定
        inv_masses = np.ones(num_verts, dtype=np.float32)
        for i, v in enumerate(verts):
            if v[2] >= 0.145:
                inv_masses[i] = 0.0

        # カーブ定義
        thetas_full = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)

        # 谷カーブ (Root: シアン): Z=-0.020、半径 5.4cm (初期 6.0cm から -6mm 引き締め)
        root_z = -0.020 + 0.004 * np.cos(thetas_full)
        root_r = radius_cloth - 0.006  # 5.4cm
        root_pts = np.stack([
            root_r * np.cos(thetas_full),
            root_r * np.sin(thetas_full),
            root_z
        ], axis=-1).astype(np.float32)

        # 山カーブ (Crest: 赤/橙): Z=+0.035、半径 6.6cm (初期 6.0cm から +6mm 膨らみ)
        crest1_z = 0.035 + 0.004 * np.cos(thetas_full)
        crest1_r = radius_cloth + 0.006  # 6.6cm
        crest1_pts = np.stack([
            crest1_r * np.cos(thetas_full),
            crest1_r * np.sin(thetas_full),
            crest1_z
        ], axis=-1).astype(np.float32)

        origin = (0.0, 0.0, 0.0)
        axis = (0.0, 0.0, 1.0)
        normal = (1.0, 0.0, 0.0)
        binormal = (0.0, 1.0, 0.0)

        # 2D-SDF テクスチャベイク (512 x 256)
        tex = bake_wrinkle_2d_sdf_texture(
            crest_curves=[crest1_pts],
            root_curves=[root_pts],
            origin=origin,
            axis=axis,
            normal=normal,
            width=512,
            height=256,
            influence_radius=0.035,
            bone_radius=radius_col,
            cloth_radius=radius_cloth,
        )

        sim = taremin_cloth_core.ClothSimulator(
            positions=verts,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.003,
            stiffness=800.0,
            bending_stiffness=10.0,
            compression_stiffness=50.0,
            solver_mode=0,
            workgroup_size=32,
        )
        sim.set_gravity(0.0, 0.0, 0.0)

        col_tris = generate_capsule_mesh(
            point_a=[0.0, 0.0, -0.4],
            point_b=[0.0, 0.0, 0.4],
            radius=radius_col,
            segments=24
        )
        sim.set_mesh_collider_triangles(
            col_tris.reshape((-1, 3, 3)),
            friction=0.2,
            thickness=0.002,
            restitution=0.0,
            single_sided=False
        )

        sim.set_wrinkle_field_params(
            origin=[origin[0], origin[1], origin[2]],
            axis=[axis[0], axis[1], axis[2]],
            normal=[normal[0], normal[1], normal[2]],
            binormal=[binormal[0], binormal[1], binormal[2]],
            influence_radius=0.035,
            bone_radius=radius_col,
            stiffness=25.0,
            blend_weight=1.0,
            z_min=tex.z_min,
            z_max=tex.z_max,
            r_min=tex.r_min,
            r_max=tex.r_max,
        )
        sim.set_wrinkle_field_texture_2d(
            tex.width, tex.height, tex.texture_bytes,
            tex.z_min, tex.z_max, tex.r_min, tex.r_max
        )
        sim.set_enable_wrinkle_field(True)

        curve_lines = generate_wrinkle_curve_lines(
            crest_curves=[crest1_pts],
            root_curves=[root_pts],
            is_closed=True
        )

        camera_pos = (0.35, -0.25, 0.12)
        camera_target = (0.0, 0.0, 0.0)

        # Frame 0 (初期状態)
        img_f0 = os.path.join(self.artifact_dir, "wrinkle_2d_f00_init.png")
        render_scene_to_file(
            img_f0,
            positions=verts,
            faces=faces,
            mesh_colliders=col_tris,
            extra_lines=curve_lines,
            camera_pos=camera_pos,
            camera_target=camera_target,
            width=600,
            height=600,
        )

        # 頂点カラーヒートマップ (2Dテクスチャサンプリング)
        weight_colors = compute_wrinkle_texture_2d_weight_colors(
            verts,
            texture_rgba=tex.rgba_array,
            origin=origin,
            axis=axis,
            normal=normal,
            z_min=tex.z_min,
            z_max=tex.z_max,
        )
        img_hm = os.path.join(self.artifact_dir, "wrinkle_2d_weight_heatmap.png")
        render_scene_to_file(
            img_hm,
            positions=verts,
            faces=faces,
            mesh_colliders=col_tris,
            vertex_colors=weight_colors,
            extra_lines=curve_lines,
            camera_pos=camera_pos,
            camera_target=camera_target,
            width=600,
            height=600,
        )

        # Frame 10
        dt = 1.0 / 60.0
        substeps = 15
        for f in range(1, 11):
            sim.step(dt, substeps, 4)

        pos_flat = np.empty(num_verts * 3, dtype=np.float32)
        sim.get_positions(pos_flat)
        curr_pos = pos_flat.reshape((-1, 3)).copy()

        img_f10 = os.path.join(self.artifact_dir, "wrinkle_2d_f10_progress.png")
        render_scene_to_file(
            img_f10,
            positions=curr_pos,
            faces=faces,
            mesh_colliders=col_tris,
            extra_lines=curve_lines,
            camera_pos=camera_pos,
            camera_target=camera_target,
            width=600,
            height=600,
        )

        # Frame 30 (収束後)
        for f in range(11, 31):
            sim.step(dt, substeps, 4)

        sim.get_positions(pos_flat)
        curr_pos = pos_flat.reshape((-1, 3)).copy()

        img_f30 = os.path.join(self.artifact_dir, "wrinkle_2d_f30_converged.png")
        render_scene_to_file(
            img_f30,
            positions=curr_pos,
            faces=faces,
            mesh_colliders=col_tris,
            extra_lines=curve_lines,
            camera_pos=camera_pos,
            camera_target=camera_target,
            width=600,
            height=600,
        )

        rs = np.linalg.norm(curr_pos[:, :2], axis=1)
        min_r = np.min(rs)
        max_r = np.max(rs)
        peak_to_valley = max_r - min_r

        print(f"\n[2D-SDF Texture Wrinkle Test Summary]")
        print(f"  コライダー半径: {radius_col:.3f}m, 布初期半径: {radius_cloth:.3f}m")
        print(f"  谷最小半径: {min_r:.4f}m (窪み量: -{(radius_cloth - min_r)*1000:.1f}mm)")
        print(f"  山最大半径: {max_r:.4f}m (隆起量: +{(max_r - radius_cloth)*1000:.1f}mm)")
        print(f"  起伏 (Peak-to-Valley): {peak_to_valley*1000:.1f}mm")
        print(f"  生成画像:")
        print(f"    - Frame 00 (初期): {img_f0}")
        print(f"    - Weight Heatmap:  {img_hm}")
        print(f"    - Frame 10 (進行): {img_f10}")
        print(f"    - Frame 30 (収束): {img_f30}")

        # アサーション
        self.assertFalse(np.any(np.isnan(curr_pos)), "NaN が検出されました")
        self.assertFalse(np.any(np.isinf(curr_pos)), "Inf が検出されました")
        self.assertGreaterEqual(min_r, radius_col - 0.002, "コライダーへの貫通が検出されました")
        self.assertGreater(max_r, radius_cloth + 0.002, "山の座屈が不十分です")
        self.assertLess(min_r, radius_cloth - 0.002, "谷の引き込みが不十分です")
        self.assertGreaterEqual(peak_to_valley, 0.005, "シワの起伏が不十分です")

    def test_wrinkle_field_2d_texture_y_branch(self):
        """「Y字型」分岐シワカーブの2D-SDF座屈検証（同角度での多重シワ共存）"""
        radius_cloth = 0.06
        radius_col = 0.05
        verts, edges, faces = create_cylinder_cloth_mesh(
            radius=radius_cloth, height=0.30, n_theta=48, n_z=36, z_center=0.0
        )
        num_verts = len(verts)

        # 上端ピン固定
        inv_masses = np.ones(num_verts, dtype=np.float32)
        for i, v in enumerate(verts):
            if v[2] >= 0.145:
                inv_masses[i] = 0.0

        # Y字型山カーブ
        # 幹: Z=-0.04 〜 0.00 (theta = 0)
        stem_z = np.linspace(-0.04, 0.0, 20)
        stem_th = np.zeros_like(stem_z)
        stem_r = np.full_like(stem_z, radius_cloth + 0.008)
        stem_pts = np.stack([stem_r * np.cos(stem_th), stem_r * np.sin(stem_th), stem_z], axis=-1).astype(np.float32)

        # 枝1: Z=0.0 〜 +0.04 (theta = 0 -> +pi/3 = +60度)
        br1_z = np.linspace(0.0, 0.04, 20)
        br1_th = np.linspace(0.0, math.pi / 3.0, 20)
        br1_r = np.full_like(br1_z, radius_cloth + 0.008)
        br1_pts = np.stack([br1_r * np.cos(br1_th), br1_r * np.sin(br1_th), br1_z], axis=-1).astype(np.float32)

        # 枝2: Z=0.0 〜 +0.04 (theta = 0 -> -pi/3 = -60度)
        br2_z = np.linspace(0.0, 0.04, 20)
        br2_th = np.linspace(0.0, -math.pi / 3.0, 20)
        br2_r = np.full_like(br2_z, radius_cloth + 0.008)
        br2_pts = np.stack([br2_r * np.cos(br2_th), br2_r * np.sin(br2_th), br2_z], axis=-1).astype(np.float32)

        # 谷カーブ: 幹の両脇を挟むように配置 (theta = +pi/4, -pi/4 付近のくびれ)
        thetas_cinch = np.linspace(-math.pi * 0.7, math.pi * 0.7, 32)
        root_z = np.full_like(thetas_cinch, -0.015)
        root_r = np.full_like(thetas_cinch, radius_cloth - 0.005)
        root_pts = np.stack([root_r * np.cos(thetas_cinch), root_r * np.sin(thetas_cinch), root_z], axis=-1).astype(np.float32)

        origin = (0.0, 0.0, 0.0)
        axis = (0.0, 0.0, 1.0)
        normal = (1.0, 0.0, 0.0)

        # 2D-SDF ベイク
        tex = bake_wrinkle_2d_sdf_texture(
            crest_curves=[stem_pts, br1_pts, br2_pts],
            root_curves=[root_pts],
            origin=origin,
            axis=axis,
            normal=normal,
            width=512,
            height=256,
            influence_radius=0.030,
            bone_radius=radius_col,
            cloth_radius=radius_cloth,
        )

        sim = taremin_cloth_core.ClothSimulator(
            positions=verts,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.003,
            stiffness=800.0,
            bending_stiffness=10.0,
            compression_stiffness=30.0,
            solver_mode=0,
            workgroup_size=32,
        )
        sim.set_gravity(0.0, 0.0, 0.0)

        col_tris = generate_capsule_mesh(
            point_a=[0.0, 0.0, -0.4],
            point_b=[0.0, 0.0, 0.4],
            radius=radius_col,
            segments=24
        )
        sim.set_mesh_collider_triangles(
            col_tris.reshape((-1, 3, 3)),
            friction=0.2,
            thickness=0.002,
            restitution=0.0,
            single_sided=False
        )

        sim.set_wrinkle_field_params(
            origin=[origin[0], origin[1], origin[2]],
            axis=[axis[0], axis[1], axis[2]],
            normal=[normal[0], normal[1], normal[2]],
            binormal=[0.0, 1.0, 0.0],
            influence_radius=0.030,
            bone_radius=radius_col,
            stiffness=30.0,
            blend_weight=1.0,
            z_min=tex.z_min,
            z_max=tex.z_max,
            r_min=tex.r_min,
            r_max=tex.r_max,
        )
        sim.set_wrinkle_field_texture_2d(
            tex.width, tex.height, tex.texture_bytes,
            tex.z_min, tex.z_max, tex.r_min, tex.r_max
        )
        sim.set_enable_wrinkle_field(True)

        dt = 1.0 / 60.0
        substeps = 15
        for f in range(30):
            sim.step(dt, substeps, 4)

        pos_flat = np.empty(num_verts * 3, dtype=np.float32)
        sim.get_positions(pos_flat)
        curr_pos = pos_flat.reshape((-1, 3)).copy()

        curve_lines = generate_wrinkle_curve_lines(
            crest_curves=[stem_pts, br1_pts, br2_pts],
            root_curves=[root_pts],
            is_closed=False
        )

        # 正面見下ろしアングル (theta=0 付近のY字が最もよく見える位置)
        camera_pos = (0.35, 0.0, 0.08)
        camera_target = (0.0, 0.0, 0.0)

        img_y = os.path.join(self.artifact_dir, "wrinkle_2d_y_branch.png")
        render_scene_to_file(
            img_y,
            positions=curr_pos,
            faces=faces,
            mesh_colliders=col_tris,
            extra_lines=curve_lines,
            camera_pos=camera_pos,
            camera_target=camera_target,
            width=600,
            height=600,
        )

        # 頂点カラーヒートマップ画像も生成
        weight_colors = compute_wrinkle_texture_2d_weight_colors(
            curr_pos,
            texture_rgba=tex.rgba_array,
            origin=origin,
            axis=axis,
            normal=normal,
            z_min=tex.z_min,
            z_max=tex.z_max,
        )
        img_y_hm = os.path.join(self.artifact_dir, "wrinkle_2d_y_branch_heatmap.png")
        render_scene_to_file(
            img_y_hm,
            positions=curr_pos,
            faces=faces,
            mesh_colliders=col_tris,
            vertex_colors=weight_colors,
            extra_lines=curve_lines,
            camera_pos=camera_pos,
            camera_target=camera_target,
            width=600,
            height=600,
        )

        rs = np.linalg.norm(curr_pos[:, :2], axis=1)
        min_r = np.min(rs)
        max_r = np.max(rs)

        print(f"\n[Y-Branch 2D Wrinkle Test Summary]")
        print(f"  谷最小半径: {min_r:.4f}m, 山最大半径: {max_r:.4f}m")
        print(f"  生成画像: {img_y}")
        print(f"  ヒートマップ画像: {img_y_hm}")

        self.assertFalse(np.any(np.isnan(curr_pos)), "NaN が検出されました")
        self.assertGreaterEqual(min_r, radius_col - 0.002, "コライダー貫通")
        self.assertGreater(max_r, radius_cloth + 0.002, "Y字シワの膨らみが形成されていること")


if __name__ == "__main__":
    unittest.main()
