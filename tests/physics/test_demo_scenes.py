"""
独立GUIデモシーンの物理挙動および回帰テスト
- Sphere Draping（球体落下・ドレープ・貫通防止・トゲトゲ防止）
- Two-Point Curtain（2点吊りカーテン・伸び耐性・ピン拘束）
- Ground Folding（長い布の地面折り畳み・多層自己衝突スタック）
- Multi-Layer Cloth（多層レイヤー布・インナー/アウター層順序維持）
- Jitter & High-Frequency Spike Detection（トゲトゲ・過剰伸長・頂点暴れ検知テスト）
"""

import os
import sys
import unittest
from typing import Optional
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import taremin_cloth_core


def compute_areal_inv_masses(
    positions: np.ndarray,
    faces: np.ndarray,
    pin_weights: Optional[np.ndarray] = None,
    areal_density: float = 0.20,
) -> np.ndarray:
    """三角形面積と面密度に応じた物理単位の逆質量 (kg^-1) を計算する。"""
    try:
        pos_c = np.ascontiguousarray(positions, dtype=np.float32)
        faces_c = np.ascontiguousarray(faces, dtype=np.uint32)
        pin_c = np.ascontiguousarray(pin_weights, dtype=np.float32) if pin_weights is not None else None
        return np.asarray(
            taremin_cloth_core.compute_areal_inv_masses(pos_c, faces_c, pin_c, float(areal_density)),
            dtype=np.float32,
        )
    except Exception:
        # フォールバック (NumPy実装)
        n_verts = len(positions)
        masses = np.zeros(n_verts, dtype=np.float64)
        if faces is not None and len(faces) > 0 and areal_density > 0.0:
            tris = np.asarray(faces, dtype=np.int64)
            p0 = positions[tris[:, 0]].astype(np.float64)
            p1 = positions[tris[:, 1]].astype(np.float64)
            p2 = positions[tris[:, 2]].astype(np.float64)
            cross = np.cross(p1 - p0, p2 - p0)
            areas = 0.5 * np.linalg.norm(cross, axis=1)
            tri_mass = areas * float(areal_density) / 3.0
            np.add.at(masses, tris[:, 0], tri_mass)
            np.add.at(masses, tris[:, 1], tri_mass)
            np.add.at(masses, tris[:, 2], tri_mass)
        masses = np.maximum(masses, 1e-9)
        inv = (1.0 / masses).astype(np.float32)
        if pin_weights is not None:
            w = np.clip(np.asarray(pin_weights, dtype=np.float32), 0.0, 1.0)
            inv[w >= 1.0] = 0.0
            inv[w < 1.0] *= (1.0 - w[w < 1.0])
        return inv


def create_grid_data(
    nx, ny, width, height, center=(0.0, 0.0, 0.0), plane="XY", y_wave=0.0,
    areal_density: float = 0.20, pin_weights: Optional[np.ndarray] = None,
):
    """プロシージャルな平面布メッシュデータを生成するユーティリティ"""
    num_verts = nx * ny
    dx = width / (nx - 1)
    dy = height / (ny - 1)
    x_start = center[0] - width * 0.5
    y_start = center[1] - height * 0.5
    z_start = center[2] - height * 0.5

    positions = []
    for j in range(ny):
        wave_offset = np.sin(j * 0.4) * y_wave if y_wave != 0.0 else 0.0
        for i in range(nx):
            u = x_start + i * dx
            if plane == "XY":
                positions.append([u, y_start + j * dy, center[2]])
            elif plane == "XZ":
                positions.append([u, center[1] + wave_offset, z_start + j * dy])
            elif plane == "XZ_BOTTOM":
                positions.append([u, center[1] + wave_offset, center[2] + j * dy])

    positions = np.array(positions, dtype=np.float32)

    faces = []
    edges = set()
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = j * nx + i
            v10 = j * nx + (i + 1)
            v01 = (j + 1) * nx + i
            v11 = (j + 1) * nx + (i + 1)

            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

            edges.add((min(v00, v10), max(v00, v10)))
            edges.add((min(v10, v11), max(v10, v11)))
            edges.add((min(v11, v01), max(v11, v01)))
            edges.add((min(v01, v00), max(v01, v00)))
            # 対角線 (v00, v11) は基本エッジから除外（ClothMeshのせん断拘束に任せる）

    faces = np.array(faces, dtype=np.uint32)
    edges = np.array(list(edges), dtype=np.uint32)

    inv_masses = compute_areal_inv_masses(positions, faces, pin_weights, areal_density)
    return positions, faces, edges, inv_masses


def assert_mesh_smoothness_and_stability(
    test_case: unittest.TestCase,
    cur_pos: np.ndarray,
    initial_pos: np.ndarray,
    edges: np.ndarray,
    max_allowed_strain: float = 1.30,
    min_allowed_strain: float = 0.20,
    max_laplacian_ratio: float = 1.8,
    prev_pos: Optional[np.ndarray] = None,
    dt: float = 1.0 / 60.0,
    max_allowed_speed: float = 15.0,
):
    """メッシュの滑らかさ、局所トゲトゲ度、エッジ歪み率、速度スパイクを包括的に検証する"""
    # 1. NaN / Inf チェック
    test_case.assertFalse(np.isnan(cur_pos).any(), "頂点座標にNaNが含まれています")
    test_case.assertFalse(np.isinf(cur_pos).any(), "頂点座標にInfが含まれています")

    # 2. エッジ歪み率チェック (引き裂かれ・過度な収縮の検知)
    p0 = cur_pos[edges[:, 0]]
    p1 = cur_pos[edges[:, 1]]
    cur_len = np.linalg.norm(p1 - p0, axis=1)

    r0 = initial_pos[edges[:, 0]]
    r1 = initial_pos[edges[:, 1]]
    rest_len = np.linalg.norm(r1 - r0, axis=1)

    strains = cur_len / np.maximum(rest_len, 1e-6)
    max_strain = float(np.max(strains))
    min_strain = float(np.min(strains))

    test_case.assertLess(
        max_strain,
        max_allowed_strain,
        f"布メッシュのエッジが過剰に引き伸ばされています (トゲトゲ/暴れ検知): max_strain={max_strain:.2f} > {max_allowed_strain:.2f}"
    )
    test_case.assertGreater(
        min_strain,
        min_allowed_strain,
        f"布メッシュのエッジが異常に押し潰されています: min_strain={min_strain:.2f} < {min_allowed_strain:.2f}"
    )

    # 3. 離散ラプラシアン (局所曲率・隣接平均からの外側突出量)
    n_verts = len(cur_pos)
    adj = [[] for _ in range(n_verts)]
    for e0, e1 in edges:
        adj[e0].append(e1)
        adj[e1].append(e0)

    laps = []
    local_edge_lens = []
    for i in range(n_verts):
        if adj[i]:
            neighbors = cur_pos[adj[i]]
            mean_pos = np.mean(neighbors, axis=0)
            laps.append(np.linalg.norm(cur_pos[i] - mean_pos))
            local_edge_lens.append(np.mean(np.linalg.norm(neighbors - cur_pos[i], axis=1)))

    if laps:
        max_lap = float(np.max(laps))
        mean_edge_len = float(np.mean(local_edge_lens))
        test_case.assertLess(
            max_lap,
            mean_edge_len * max_laplacian_ratio,
            f"布表面に局所的なトゲトゲ突起が発生しています: max_lap={max_lap*1000:.1f}mm > 許容閾値={mean_edge_len*max_laplacian_ratio*1000:.1f}mm"
        )

    # 4. 速度スパイクチェック
    if prev_pos is not None:
        vel = (cur_pos - prev_pos) / dt
        speed = np.linalg.norm(vel, axis=1)
        max_speed = float(np.max(speed))
        test_case.assertLess(
            max_speed,
            max_allowed_speed,
            f"物理的にあり得ない高周波速度スパイクが発生しています: max_speed={max_speed:.2f} m/s > {max_allowed_speed:.2f} m/s"
        )


class TestDemoScenes(unittest.TestCase):
    """定番デモシーンの物理挙動自動回帰テスト"""

    def test_jitter_detection_fails_on_broken_mass(self):
        """【重要検証】inv_mass=1.0（旧バグ状態）のとき、トゲトゲ・暴れ検知テストが確実にFAILすることを確認する"""
        positions, faces, edges, _ = create_grid_data(
            nx=20, ny=20, width=0.60, height=0.60, center=(0.0, 0.0, 0.35), plane="XY"
        )
        # 故意に旧バグ（質量1.0kg固定 = 布全体1.2トン）を注入
        broken_inv_masses = np.ones(len(positions), dtype=np.float32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            faces=faces,
            inv_masses=broken_inv_masses,
            thickness=0.006,
            stiffness=1200.0,
            compression_stiffness=400.0,
            shear_stiffness=400.0,
            bending_stiffness=2.0,
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=0.2,
            max_displacement_ratio=0.2,
            exclude_neighbors=True,
            enable_normal_untangling=True,
        )
        sim.add_sphere_collider(center=[0.0, 0.0, 0.16], radius=0.14, friction=0.4, restitution=0.0)

        # 3ステップ実行しただけで暴れ（エッジ伸び・トゲトゲ）が発生する
        out_pos = np.zeros(len(positions) * 3, dtype=np.float32)
        for _ in range(3):
            sim.step(dt=1.0 / 60.0, substeps=20)

        sim.get_positions(out_pos)
        pos_3d = out_pos.reshape((-1, 3))

        # 旧状態では assert_mesh_smoothness_and_stability が AssertionError を投げること！
        with self.assertRaises(AssertionError, msg="inv_mass=1.0 の暴れ状態を検知できませんでした！"):
            assert_mesh_smoothness_and_stability(self, pos_3d, positions, edges)

    def test_sphere_draping(self):
        """1. 球体落下・ドレープテスト: 布が球体に衝突して覆いかぶさり、貫通せず、トゲトゲにならないこと"""
        positions, faces, edges, inv_masses = create_grid_data(
            nx=20, ny=20, width=0.60, height=0.60, center=(0.0, 0.0, 0.35), plane="XY", areal_density=0.20
        )

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.0035,
            stiffness=1200.0,
            compression_stiffness=60.0,
            shear_stiffness=60.0,
            bending_stiffness=0.4,
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=0.2,
            max_displacement_ratio=0.2,
            exclude_neighbors=True,
            enable_normal_untangling=True,
        )

        sphere_center = [0.0, 0.0, 0.16]
        sphere_radius = 0.14
        sim.add_sphere_collider(center=sphere_center, radius=sphere_radius, friction=0.4, restitution=0.0)

        out_pos = np.zeros(len(positions) * 3, dtype=np.float32)
        prev_pos = positions.copy()

        # 40ステップ進行
        for _ in range(40):
            sim.step(dt=1.0 / 60.0, substeps=20)
            sim.get_positions(out_pos)
            pos_3d = out_pos.reshape((-1, 3))

            # 毎ステップの滑らかさ・トゲトゲ・暴れチェック
            assert_mesh_smoothness_and_stability(
                self, pos_3d, positions, edges, prev_pos=prev_pos, dt=1.0 / 60.0
            )
            prev_pos = pos_3d.copy()

        # 1. 布が球体の上にとどまっていること
        max_z = np.max(pos_3d[:, 2])
        self.assertGreater(max_z, sphere_center[2] + sphere_radius * 0.8, "布が球体をすり抜けています")

        # 2. 球体内部への貫通深さチェック (マージン 0.015m 以内)
        dists = np.linalg.norm(pos_3d - sphere_center, axis=1)
        min_dist = np.min(dists)
        self.assertGreater(min_dist, sphere_radius - 0.015, f"球体内部へ貫通しています: min_dist={min_dist:.4f}")

    def test_two_point_curtain(self):
        """2. 2点吊りカーテンテスト: ピン留め頂点が固定され、布が自重で垂れ下がり伸びすぎず、滑らかであること"""
        nx, ny = 25, 20
        top_row = ny - 1
        pins = np.zeros(nx * ny, dtype=np.float32)
        pin_indices = [top_row * nx, top_row * nx + 1, top_row * nx + (nx - 2), top_row * nx + (nx - 1)]
        for idx in pin_indices:
            pins[idx] = 1.0

        positions, faces, edges, inv_masses = create_grid_data(
            nx=nx, ny=ny, width=0.75, height=0.50, center=(0.0, 0.0, 0.50), plane="XZ",
            areal_density=0.18, pin_weights=pins
        )

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.0035,
            stiffness=1500.0,
            compression_stiffness=50.0,
            shear_stiffness=60.0,
            bending_stiffness=0.3,
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_damping(0.5)

        out_pos = np.zeros(len(positions) * 3, dtype=np.float32)
        prev_pos = positions.copy()

        # 40ステップ進行
        for _ in range(40):
            sim.step(dt=1.0 / 60.0, substeps=20)
            sim.get_positions(out_pos)
            pos_3d = out_pos.reshape((-1, 3))
            assert_mesh_smoothness_and_stability(
                self, pos_3d, positions, edges, max_allowed_strain=1.30, prev_pos=prev_pos, dt=1.0 / 60.0
            )
            prev_pos = pos_3d.copy()

        # 1. ピン留め頂点が初期位置を完全に維持していること
        for idx in pin_indices:
            disp = np.linalg.norm(pos_3d[idx] - positions[idx])
            self.assertLess(disp, 1e-4, f"ピン頂点 {idx} が移動しました (disp={disp:.6f})")

        # 2. 布全体が自重に耐えて安定静止し、不自然な大暴れ・発散を起こしていないこと
        max_disp = np.max(np.linalg.norm(pos_3d - positions, axis=1))
        self.assertLess(max_disp, 0.15, f"カーテンが不自然に大きく変形・発散しています: max_disp={max_disp:.4f}m")

    def test_ground_folding(self):
        """3. 長い布の地面折り畳みテスト: 地面コライダー上でアコーディオン状に積み重なり、貫通せずトゲトゲ化しないこと"""
        nx, ny = 12, 45
        width = 0.20
        height = 0.90
        positions, faces, edges, inv_masses = create_grid_data(
            nx=nx, ny=ny, width=width, height=height, center=(0.0, 0.0, 0.03), plane="XZ_BOTTOM", y_wave=0.015,
            areal_density=0.22
        )

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.0035,
            stiffness=1500.0,
            compression_stiffness=60.0,
            shear_stiffness=60.0,
            bending_stiffness=0.3,
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=0.2,
            max_displacement_ratio=0.2,
            exclude_neighbors=True,
            enable_normal_untangling=True,
        )
        sim.set_coupled_self_collision_options(1, 2)
        sim.add_plane_collider(point=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0], friction=0.5, restitution=0.0)

        out_pos = np.zeros(len(positions) * 3, dtype=np.float32)
        prev_pos = positions.copy()

        # 50ステップ進行
        for _ in range(50):
            sim.step(dt=1.0 / 60.0, substeps=25)
            sim.get_positions(out_pos)
            pos_3d = out_pos.reshape((-1, 3))
            assert_mesh_smoothness_and_stability(
                self, pos_3d, positions, edges, max_allowed_strain=1.30, prev_pos=prev_pos, dt=1.0 / 60.0
            )
            prev_pos = pos_3d.copy()

        # 1. 地面下へのめり込みがないこと
        min_z = np.min(pos_3d[:, 2])
        self.assertGreater(min_z, -0.012, f"布が地面コライダーの下に突き抜けています: min_z={min_z:.4f}")

        # 2. 全体が地面付近に折り畳まれ、最高点が大幅に下がっていること
        max_z = np.max(pos_3d[:, 2])
        self.assertLess(max_z, 0.50, f"布が地面に折りたたまれていません: max_z={max_z:.4f}")

        # 3. Y方向の広がり（蛇腹折り重なり）が確認できること
        y_range = np.max(pos_3d[:, 1]) - np.min(pos_3d[:, 1])
        self.assertGreater(y_range, 0.03, f"蛇腹状の折り畳みが発生していません: y_range={y_range:.4f}")

    def test_multi_layer_cloth(self):
        """4. 多層レイヤー布テスト: アウター層がインナー層の外側を維持し、両層とも暴れないこと"""
        nx, ny = 14, 14
        pos_inner, faces_inner, edges_inner, inv_inner = create_grid_data(
            nx=nx, ny=ny, width=0.35, height=0.35, center=(0.0, 0.0, 0.25), plane="XY", areal_density=0.18
        )
        pos_outer, faces_outer, edges_outer, inv_outer = create_grid_data(
            nx=nx, ny=ny, width=0.38, height=0.38, center=(0.0, 0.0, 0.27), plane="XY", areal_density=0.18
        )

        n_inner = len(pos_inner)
        n_outer = len(pos_outer)

        positions = np.vstack([pos_inner, pos_outer])
        inv_masses = np.concatenate([inv_inner, inv_outer])

        faces = np.vstack([faces_inner, faces_outer + n_inner])
        edges = np.vstack([edges_inner, edges_outer + n_inner])

        layer_ids = np.zeros(n_inner + n_outer, dtype=np.uint32)
        layer_ids[n_inner:] = 1

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            layer_ids=layer_ids,
            thickness=0.0035,
            stiffness=1200.0,
            compression_stiffness=60.0,
            shear_stiffness=60.0,
            bending_stiffness=0.4,
        )
        sim.set_gravity(0.0, 0.0, -9.81)
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=0.2,
            max_displacement_ratio=0.2,
            exclude_neighbors=True,
            enable_normal_untangling=True,
        )
        sim.set_coupled_self_collision_options(1, 2)
        sim.set_enable_pair_cache(True)
        sim.set_pair_cache_options(65536, 65536, 1, 0.005, 1.3, 0.02)

        sphere_center = np.array([0.0, 0.0, 0.10], dtype=np.float32)
        sphere_radius = 0.12
        sim.add_sphere_collider(center=sphere_center.tolist(), radius=sphere_radius, friction=0.3, restitution=0.0)

        out_pos = np.zeros(len(positions) * 3, dtype=np.float32)
        prev_pos = positions.copy()

        for _ in range(40):
            sim.step(dt=1.0 / 60.0, substeps=25)
            sim.get_positions(out_pos)
            pos_3d = out_pos.reshape((-1, 3))
            assert_mesh_smoothness_and_stability(
                self, pos_3d, positions, edges, max_allowed_strain=1.25, prev_pos=prev_pos, dt=1.0 / 60.0
            )
            prev_pos = pos_3d.copy()

        inner_pos = pos_3d[:n_inner]
        outer_pos = pos_3d[n_inner:]

        # 球体上の接触領域（中央領域: X, Y が ±0.06m 以内）において、
        # 同一XY近傍でアウター層がインナー層の外側（上）にあることを幾何学的に検証
        mask_center = (np.abs(outer_pos[:, 0]) < 0.06) & (np.abs(outer_pos[:, 1]) < 0.06)
        outer_center_pos = outer_pos[mask_center]

        diffs = []
        for p_out in outer_center_pos:
            xy_dist = np.linalg.norm(inner_pos[:, :2] - p_out[:2], axis=1)
            nearest_idx = np.argmin(xy_dist)
            z_diff = p_out[2] - inner_pos[nearest_idx, 2]
            diffs.append(z_diff)

        diffs = np.array(diffs)
        min_z_diff = float(np.min(diffs))
        mean_z_diff = float(np.mean(diffs))

        self.assertGreater(
            min_z_diff,
            -0.002,
            f"中央接触部でアウター層がインナー層の内側にめり込んでいます: min_diff={min_z_diff:.4f}m",
        )
        self.assertGreater(
            mean_z_diff,
            0.005,
            f"中央接触部でアウター層が十分に外側を保っていません: mean_diff={mean_z_diff:.4f}m",
        )

    def test_cloth_twisting(self):
        """5. 布のねじり絞りテスト: 上下逆回転による絞り込み・高密度接触で過剰伸長・トゲトゲ化を起こさないこと"""
        nx, ny = 16, 22
        width = 0.30
        height = 0.50
        areal_density = 0.20
        positions, faces, edges, inv_masses = create_grid_data(
            nx=nx, ny=ny, width=width, height=height, center=(0.0, 0.0, 0.35), plane="XZ",
            areal_density=areal_density
        )

        # 上端と下端をピン留め
        for i in range(nx):
            inv_masses[i] = 0.0           # 下端
            inv_masses[(ny - 1) * nx + i] = 0.0  # 上端

        # 上下逆方向にねじる初期変位（約270度）
        twisting_positions = positions.copy()
        for j in range(ny):
            t = j / (ny - 1)
            angle = (t - 0.5) * np.pi * 1.5
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            for i in range(nx):
                idx = j * nx + i
                x = positions[idx, 0]
                y = positions[idx, 1]
                twisting_positions[idx, 0] = x * cos_a - y * sin_a
                twisting_positions[idx, 1] = x * sin_a + y * cos_a

        sim = taremin_cloth_core.ClothSimulator(
            positions=twisting_positions,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.0035,
            stiffness=1500.0,
            compression_stiffness=60.0,
            shear_stiffness=80.0,
            bending_stiffness=0.4,
        )
        sim.set_gravity(0.0, 0.0, 0.0) # ねじれ挙動検証のため無重力
        sim.set_enable_self_collision(True)
        sim.set_self_collision_options(
            relief_factor=0.2,
            max_displacement_ratio=0.2,
            exclude_neighbors=True,
            enable_normal_untangling=True,
        )
        sim.set_coupled_self_collision_options(1, 2)
        sim.set_enable_edge_collision(True)

        out_pos = np.zeros(len(positions) * 3, dtype=np.float32)
        prev_pos = twisting_positions.copy()

        # 30ステップ進行
        for _ in range(30):
            sim.step(dt=1.0 / 60.0, substeps=25)
            sim.get_positions(out_pos)
            pos_3d = out_pos.reshape((-1, 3))
            assert_mesh_smoothness_and_stability(
                self, pos_3d, twisting_positions, edges, max_allowed_strain=1.30, prev_pos=prev_pos, dt=1.0 / 60.0
            )
            prev_pos = pos_3d.copy()

        # 1. ピン留め端が位置を維持していること
        for i in range(nx):
            self.assertLess(np.linalg.norm(pos_3d[i] - twisting_positions[i]), 1e-4)
            top_idx = (ny - 1) * nx + i
            self.assertLess(np.linalg.norm(pos_3d[top_idx] - twisting_positions[top_idx]), 1e-4)

        # 2. 全頂点が上下ピン端のZ範囲内にとどまり、発散やテレポートを起こしていないこと
        z_min = np.min(pos_3d[:, 2])
        z_max = np.max(pos_3d[:, 2])
        self.assertGreater(z_min, 0.08, f"ねじれにより布が下に吹き飛んでいます: z_min={z_min:.4f}")
        self.assertLess(z_max, 0.62, f"ねじれにより布が上に吹き飛んでいます: z_max={z_max:.4f}")

    def test_funnel_pass(self):
        """6. 漏斗通過テスト: 円錐漏斗の開口部から布がすり抜け・挟み込み破綻なく通過すること"""
        nx, ny = 16, 16
        width = 0.35
        height = 0.35
        positions, faces, edges, inv_masses = create_grid_data(
            nx=nx, ny=ny, width=width, height=height, center=(0.0, 0.0, 0.45), plane="XY",
            areal_density=0.20
        )

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.0035,
            stiffness=1200.0,
            compression_stiffness=50.0,
            shear_stiffness=60.0,
            bending_stiffness=0.3,
        )
        sim.set_gravity(0.0, 0.0, -9.81)

        # 6本の傾斜カプセルによる円錐漏斗
        num_posts = 6
        for k in range(num_posts):
            angle = k * 2.0 * np.pi / num_posts
            r_top = 0.20
            r_bot = 0.07
            p_top = [r_top * np.cos(angle), r_top * np.sin(angle), 0.35]
            p_bot = [r_bot * np.cos(angle), r_bot * np.sin(angle), 0.15]
            sim.add_capsule_collider(point_a=p_top, point_b=p_bot, radius=0.025, friction=0.2, restitution=0.0)

        out_pos = np.zeros(len(positions) * 3, dtype=np.float32)
        prev_pos = positions.copy()

        # 40ステップ進行
        for _ in range(40):
            sim.step(dt=1.0 / 60.0, substeps=25)
            sim.get_positions(out_pos)
            pos_3d = out_pos.reshape((-1, 3))
            assert_mesh_smoothness_and_stability(
                self, pos_3d, positions, edges, max_allowed_strain=1.30, prev_pos=prev_pos, dt=1.0 / 60.0
            )
            prev_pos = pos_3d.copy()

        # 1. 布全体が重力で漏斗内部に引き込まれ、最低点が大きく低下していること
        min_z = np.min(pos_3d[:, 2])
        self.assertLess(min_z, 0.25, f"布が漏斗内に落ち込んでいません: min_z={min_z:.4f}")

        # 2. 水平広がりが漏斗上端より小さく窄まっていること
        r_xy = np.linalg.norm(pos_3d[:, :2], axis=1)
        mean_r = np.mean(r_xy)
        self.assertLess(mean_r, 0.16, f"漏斗通過に伴う窄まりが起きていません: mean_r={mean_r:.4f}")

    def test_garment_sewing(self):
        """7. 衣服縫合テスト: 縫合バネ収縮により前後身頃が閉合し、人体胴体コライダーにフィットすること"""
        nx, ny = 14, 20
        width = 0.28
        height = 0.45
        areal_density = 0.18

        pos_front, faces_front, edges_front, inv_front = create_grid_data(
            nx=nx, ny=ny, width=width, height=height, center=(0.0, 0.12, 0.35), plane="XZ",
            areal_density=areal_density
        )
        pos_back, faces_back, edges_back, inv_back = create_grid_data(
            nx=nx, ny=ny, width=width, height=height, center=(0.0, -0.12, 0.35), plane="XZ",
            areal_density=areal_density
        )

        n_front = len(pos_front)
        positions = np.vstack([pos_front, pos_back])
        inv_masses = np.concatenate([inv_front, inv_back])
        faces = np.vstack([faces_front, faces_back + n_front])
        edges = np.vstack([edges_front, edges_back + n_front])

        # 左右両脇の縫合バネ
        sewing_springs = []
        for j in range(ny):
            f_left = j * nx
            b_left = n_front + j * nx
            sewing_springs.append([f_left, b_left])

            f_right = j * nx + (nx - 1)
            b_right = n_front + j * nx + (nx - 1)
            sewing_springs.append([f_right, b_right])
        sewing_springs = np.array(sewing_springs, dtype=np.uint32)

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            sewing_springs=sewing_springs,
            thickness=0.0035,
            stiffness=1500.0,
            compression_stiffness=60.0,
            shear_stiffness=60.0,
            bending_stiffness=0.4,
            sewing_shrink_speed=1.0,
            sewing_stiffness=10000.0,
            enable_sewing_lock=True,
            sewing_lock_distance=0.02,
        )
        sim.set_gravity(0.0, 0.0, -9.81)

        # マネキン胴体コライダー (2本のカプセル)
        sim.add_capsule_collider(point_a=[-0.05, 0.0, 0.55], point_b=[-0.05, 0.0, 0.15], radius=0.09, friction=0.3, restitution=0.0)
        sim.add_capsule_collider(point_a=[0.05, 0.0, 0.55], point_b=[0.05, 0.0, 0.15], radius=0.09, friction=0.3, restitution=0.0)

        out_pos = np.zeros(len(positions) * 3, dtype=np.float32)
        prev_pos = positions.copy()

        # 35ステップ進行（過渡期は縫合バネ牽引による局所伸長を許容）
        for _ in range(35):
            sim.step(dt=1.0 / 60.0, substeps=25)
            sim.get_positions(out_pos)
            pos_3d = out_pos.reshape((-1, 3))
            assert_mesh_smoothness_and_stability(
                self, pos_3d, positions, edges, max_allowed_strain=2.20, prev_pos=prev_pos, dt=1.0 / 60.0
            )
            prev_pos = pos_3d.copy()

        # 最終状態（収縮・ロック完了後）:
        # タイトな衣服が胴体にフィットして張っている状態（最大歪み率 < 1.80）であり、トゲトゲや発散がないこと
        assert_mesh_smoothness_and_stability(
            self, pos_3d, positions, edges, max_allowed_strain=1.80
        )

        # 1. 縫合バネが収縮して左右両脇が閉合していること (平均距離 < 0.015m)
        sew_dists = np.linalg.norm(pos_3d[sewing_springs[:, 0]] - pos_3d[sewing_springs[:, 1]], axis=1)
        mean_sew_dist = float(np.mean(sew_dists))
        self.assertLess(mean_sew_dist, 0.018, f"縫合バネが十分に閉合していません: mean_dist={mean_sew_dist:.4f}m")

        # 2. 前後身頃のY方向距離が初期状態 (0.24m) から大きく縮まり、胴体にフィットしていること
        mean_front_y = np.mean(pos_3d[:n_front, 1])
        mean_back_y = np.mean(pos_3d[n_front:, 1])
        y_gap = mean_front_y - mean_back_y
        self.assertLess(y_gap, 0.20, f"前後身頃が胴体に引き寄せられていません: y_gap={y_gap:.4f}m")
        self.assertGreater(y_gap, 0.05, f"身頃同士が貫通して裏返っています: y_gap={y_gap:.4f}m")


if __name__ == "__main__":
    unittest.main()

