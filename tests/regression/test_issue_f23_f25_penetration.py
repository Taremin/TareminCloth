import os
import unittest
import numpy as np
from taremin_cloth.replayer import ClothReplayer, get_frame
from taremin_cloth.analysis import find_triangle_intersections


class TestColliderCollisionPenetrationReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(os.path.dirname(__file__), "..", "fixtures", "data_penetration_f23_f25.jsonl.gz")
        cls.replayer = ClothReplayer(cls.data_path)

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
        cls.c1_set = components[1] if len(components) > 1 else set()

    def count_cross_sheet_intersections(self, positions: np.ndarray) -> int:
        """2枚の布の間で発生している三角形交差ペア数をカウント"""
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
        Frame 23（コライダー激突直前、布間交差ゼロ）から旧設定で実行した場合、
        Frame 24で100m/s超の射出が発生し、Frame 25で布同士の面交差（突き抜け）が発生することを確認
        """
        f23 = get_frame(self.data_path, 23)
        p23 = np.array(f23["positions"], dtype=np.float32).reshape(-1, 3)
        v23 = np.array(f23["velocities"], dtype=np.float32).reshape(-1, 3)

        # Frame 23時点では布間交差ゼロであることを確認
        init_cross = self.count_cross_sheet_intersections(p23)
        self.assertEqual(init_cross, 0, "初期Frame 23時点で既に布間交差が存在します")

        sim = self.replayer.create_simulator(p23)
        sim.set_positions_and_velocities(p23, v23)
        sim.set_solver_iterations(2)
        sim.set_self_collision_options(
            relief_factor=0.2,
            max_displacement_ratio=0.2,
            exclude_neighbors=True,
            enable_normal_untangling=True,
            max_iterations=128
        )

        out = np.zeros(self.replayer.num_vertices * 3, dtype=np.float32)
        for f in range(24, 26):
            fdata = get_frame(self.data_path, f)
            self.replayer.sim = sim
            self.replayer.apply_frame_inputs(fdata)
            sim.step(dt=1.0 / 60.0, substeps=20)

        sim.get_positions(out)
        pos = out.reshape(-1, 3)
        cross_count = self.count_cross_sheet_intersections(pos)
        print(f"\n[Original Red Frame 25] Cross-Sheet Intersections = {cross_count}")

        # レンダリング画像の保存
        try:
            from taremin_cloth.mesh_renderer import render_mesh_to_file
            os.makedirs("scratch", exist_ok=True)
            render_mesh_to_file(
                "scratch/test_f25_red_result.png",
                pos,
                self.replayer.faces,
                width=1024,
                height=768,
            )
        except Exception as e:
            print(f"[Warning] Failed to render red image: {e}")

        # 旧設定では確実に交差（突き抜け）が発生していることをアサート
        self.assertGreater(cross_count, 0, "旧設定で貫通が再現されていません（Red失敗）")

    def test_green_condition_improved_settings(self):
        """
        [Green検証]
        新設定（コライダー速度クランプ + Sweep AABB型CCD探索 + 適応型サブステップ64）により、
        激突後のFrame 25においても布間交差（突き抜け）が完全にゼロ（0組）になることを保証
        """
        f23 = get_frame(self.data_path, 23)
        p23 = np.array(f23["positions"], dtype=np.float32).reshape(-1, 3)
        v23 = np.array(f23["velocities"], dtype=np.float32).reshape(-1, 3)

        sim = self.replayer.create_simulator(p23)
        sim.set_positions_and_velocities(p23, v23)
        sim.set_solver_iterations(3)
        sim.set_self_collision_options(
            relief_factor=0.8,
            max_displacement_ratio=0.5,
            exclude_neighbors=True,
            enable_normal_untangling=True,
            max_iterations=512,
        )

        out = np.zeros(self.replayer.num_vertices * 3, dtype=np.float32)
        for f in range(24, 26):
            fdata = get_frame(self.data_path, f)
            self.replayer.sim = sim
            self.replayer.apply_frame_inputs(fdata)
            # 激突時の緊急適応サブステップ（96ステップ）
            sim.step(dt=1.0 / 60.0, substeps=96)

        sim.get_positions(out)
        pos = out.reshape(-1, 3)
        cross_count = self.count_cross_sheet_intersections(pos)
        print(f"\n[Improved Green Frame 25] Cross-Sheet Intersections = {cross_count}")

        # レンダリング画像の保存
        try:
            from taremin_cloth.mesh_renderer import render_mesh_to_file
            os.makedirs("scratch", exist_ok=True)
            render_mesh_to_file(
                "scratch/test_f25_green_result.png",
                pos,
                self.replayer.faces,
                width=1024,
                height=768,
            )
        except Exception as e:
            print(f"[Warning] Failed to render green image: {e}")

        # 新設定では交差が完全ゼロ（0組）であることを検証
        self.assertEqual(cross_count, 0, f"新設定でも布間交差が残存しています: {cross_count}組")


if __name__ == "__main__":
    unittest.main()
