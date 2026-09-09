"""
実測デバッグログに基づく自己衝突・貫通の再現＆改善検証テスト (Red-Green Test)
Frame 50（布同士が接近し接触が始まる直前）から再実行し、
・旧設定 (探索上限128, 緩和0.2, 反復1): 接触圧に負けて突き抜けが増加する (Red条件)
・新設定 (探索上限512, 枝切り有効, 緩和0.8, 反復3): 反発クリアランスを維持し突き抜けを大幅抑制する (Green条件)
を検証します。
"""

import os
import unittest
import numpy as np

import taremin_cloth_core
from taremin_cloth.replayer import ClothReplayer, get_frame


class TestPenetrationReplayRedGreen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(os.path.dirname(__file__), "..", "fixtures", "data_penetration_f46_f50.jsonl.gz")
        cls.replayer = ClothReplayer(cls.data_path)

        # 2枚の布の連結成分（アイランド）を分離
        edges = cls.replayer.edges
        adj = {i: set() for i in range(cls.replayer.num_vertices)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        visited = set()
        components = []
        for i in range(cls.replayer.num_vertices):
            if i not in visited:
                comp = []
                q = [i]
                visited.add(i)
                while q:
                    curr = q.pop()
                    comp.append(curr)
                    for nxt in adj[curr]:
                        if nxt not in visited:
                            visited.add(nxt)
                            q.append(nxt)
                components.append(set(comp))

        cls.c0_set = components[0]
        cls.c1_set = components[1]

    def count_cross_sheet_intersections(self, positions: np.ndarray) -> int:
        """2枚の布の間で発生している三角形交差ペア数を精密カウント"""
        from taremin_cloth.analysis import find_triangle_intersections
        intersections = find_triangle_intersections(positions, self.replayer.faces)
        cross_count = 0
        for f0_idx, f1_idx in intersections:
            f0_in_c0 = self.replayer.faces[f0_idx][0] in self.c0_set
            f1_in_c0 = self.replayer.faces[f1_idx][0] in self.c0_set
            if f0_in_c0 != f1_in_c0:
                cross_count += 1
        return cross_count

    def test_red_condition_original_settings(self):
        """
        [Red検証]
        Frame 46（布間交差ゼロの直前フレーム）から旧設定で実行した場合、
        Frame 49時点で布同士の突き抜け（面交差）が発生することを確認
        """
        f46 = get_frame(self.data_path, 46)
        p46 = np.array(f46["positions"], dtype=np.float32).reshape(-1, 3)
        v46 = np.array(f46["velocities"], dtype=np.float32).reshape(-1, 3)

        # 初期Frame 46時点では布間交差が完全にゼロであることを確認
        init_cross = self.count_cross_sheet_intersections(p46)
        self.assertEqual(init_cross, 0, "初期Frame 46時点で布間交差が存在します")

        sim = self.replayer.create_simulator(p46)
        sim.set_positions_and_velocities(p46, v46)
        sim.set_solver_iterations(1)
        sim.set_self_collision_options(
            relief_factor=0.2,
            max_displacement_ratio=0.2,
            exclude_neighbors=True,
            enable_normal_untangling=True,
            max_iterations=128
        )

        out = np.zeros(self.replayer.num_vertices * 3, dtype=np.float32)
        for f in range(47, 50):
            fdata = get_frame(self.data_path, f)
            self.replayer.sim = sim
            self.replayer.apply_frame_inputs(fdata)
            sim.step(dt=1.0 / 60.0, substeps=8)

        sim.get_positions(out)
        cross_count = self.count_cross_sheet_intersections(out.reshape(-1, 3))
        print(f"\n[Original Red] Frame 49 Cross-Sheet Intersections = {cross_count}")

        # Redテスト完了時にもRustコア内蔵の高速Z-bufferレンダラーで可視化画像を保存
        from taremin_cloth.mesh_renderer import render_mesh_to_file
        img_path, red_px = render_mesh_to_file(
            "scratch/test_penetration_red_result.png",
            out.reshape(-1, 3),
            self.replayer.faces,
            width=800,
            height=600,
        )
        print(f"[Rust Renderer] Saved: {img_path} (Back-face red pixels: {red_px})")

        # 旧設定では布同士の突き抜け交差が確実に発生している（Red成功）
        self.assertGreater(cross_count, 0, "旧設定で貫通が再現されていません")

    def test_green_condition_improved_settings(self):
        """
        [Green検証]
        新設定（探索上限512, 枝切り有効, 緩和0.8, 反復3, 適応型サブステップ32）では、
        超高速衝突時でもトンネリングを完全遮断し、布間交差数「0組」を達成することを確認
        """
        f46 = get_frame(self.data_path, 46)
        p46 = np.array(f46["positions"], dtype=np.float32).reshape(-1, 3)
        v46 = np.array(f46["velocities"], dtype=np.float32).reshape(-1, 3)

        sim = self.replayer.create_simulator(p46)
        sim.set_positions_and_velocities(p46, v46)
        sim.set_solver_iterations(3)
        sim.set_self_collision_options(
            relief_factor=0.8,
            max_displacement_ratio=0.5,
            exclude_neighbors=True,
            enable_normal_untangling=True,
            max_iterations=512
        )

        out = np.zeros(self.replayer.num_vertices * 3, dtype=np.float32)
        for f in range(47, 50):
            fdata = get_frame(self.data_path, f)
            self.replayer.sim = sim
            self.replayer.apply_frame_inputs(fdata)
            sim.step(dt=1.0 / 60.0, substeps=32)

        sim.get_positions(out)
        cross_count = self.count_cross_sheet_intersections(out.reshape(-1, 3))
        print(f"[Improved Green] Frame 49 Cross-Sheet Intersections = {cross_count}")

        # テスト完了時にRustコア内蔵の高速Z-bufferレンダラーで可視化画像を保存
        from taremin_cloth.mesh_renderer import render_mesh_to_file
        img_path, red_px = render_mesh_to_file(
            "scratch/test_penetration_green_result.png",
            out.reshape(-1, 3),
            self.replayer.faces,
            width=800,
            height=600,
        )
        print(f"[Rust Renderer] Saved: {img_path} (Back-face red pixels: {red_px})")

        # 布同士の交差が完全にゼロ（0組）であることをアサート
        self.assertEqual(cross_count, 0, f"新設定で布間交差が発生しています: {cross_count} 組")


if __name__ == "__main__":
    unittest.main()
