import unittest
import numpy as np
import taremin_cloth_core

class TestGrabCollisionPipeline(unittest.TestCase):
    def test_grab_does_not_tunnel_through_cloth(self):
        """Grab (ピン留め強制移動) を行っても、自己衝突が拘束後に適用されて相手の布を貫通しないことを検証"""
        # 2枚の平行な 5x5 平面メッシュ (上布 Z=0.05, 下布 Z=0.0)
        grid_size = 5
        x = np.linspace(-0.2, 0.2, grid_size, dtype=np.float32)
        y = np.linspace(-0.2, 0.2, grid_size, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        # 下布 (Z=0.0)
        bottom_pos = np.stack([xx.ravel(), yy.ravel(), np.zeros(grid_size*grid_size, dtype=np.float32)], axis=1)
        # 上布 (Z=0.05)
        top_pos = np.stack([xx.ravel(), yy.ravel(), np.full(grid_size*grid_size, 0.05, dtype=np.float32)], axis=1)

        positions = np.vstack([bottom_pos, top_pos])
        n_verts = len(positions)

        # エッジと面の生成
        def create_grid_topology(offset):
            edges = []
            faces = []
            for r in range(grid_size):
                for c in range(grid_size):
                    idx = offset + r * grid_size + c
                    if c + 1 < grid_size:
                        edges.append([idx, idx + 1])
                    if r + 1 < grid_size:
                        edges.append([idx, idx + grid_size])
                    if r + 1 < grid_size and c + 1 < grid_size:
                        v0 = idx
                        v1 = idx + 1
                        v2 = idx + grid_size + 1
                        v3 = idx + grid_size
                        faces.append([v0, v1, v2])
                        faces.append([v0, v2, v3])
                        edges.append([v0, v2])
            return edges, faces

        b_edges, b_faces = create_grid_topology(0)
        t_edges, t_faces = create_grid_topology(grid_size*grid_size)

        edges = np.array(b_edges + t_edges, dtype=np.uint32)
        faces = np.array(b_faces + t_faces, dtype=np.uint32)
        inv_masses = np.ones(n_verts, dtype=np.float32)

        # 下布の四隅を固定 (ピン留め)
        for pin_idx in [0, grid_size-1, grid_size*(grid_size-1), grid_size*grid_size-1]:
            inv_masses[pin_idx] = 0.0

        sim = taremin_cloth_core.ClothSimulator(
            positions=positions,
            edges=edges,
            faces=faces,
            inv_masses=inv_masses,
            thickness=0.005,
            stiffness=10000.0,
            compression_stiffness=10000.0,
            shear_stiffness=1000.0,
            bending_stiffness=10.0,
        )
        sim.set_enable_self_collision(True)
        sim.set_gravity(0.0, 0.0, 0.0) # 無重力

        # 上布の中央頂点を Grab して、下布の反対側 (Z = -0.05) へ向かって押し込む！
        grab_vert = grid_size * grid_size + (grid_size * grid_size) // 2 # 上布の中央頂点
        initial_grab_pos = positions[grab_vert].copy()

        # 10 ステップかけて、下布の反対側 (Z = -0.05) へ向かって無理やり Grab 移動させる
        target_z = -0.05
        for step in range(1, 20):
            current_target = initial_grab_pos.copy()
            alpha = min(1.0, step / 10.0)
            current_target[2] = initial_grab_pos[2] * (1.0 - alpha) + target_z * alpha
            sim.set_pin(grab_vert, [float(current_target[0]), float(current_target[1]), float(current_target[2])], 1.0)
            sim.step(dt=1.0 / 60.0, substeps=20)

        # シミュレーション結果を取得
        out_coords = np.zeros(n_verts * 3, dtype=np.float32)
        sim.get_positions(out_coords)
        result_pos = out_coords.reshape((-1, 3))


        
        # 下布の中央頂点の Z 座標
        bottom_center_vert = (grid_size * grid_size) // 2
        grabbed_final_z = result_pos[grab_vert, 2]
        bottom_final_z = result_pos[bottom_center_vert, 2]

        print(f"\n[Grab貫通テスト結果]")
        print(f"  Grab目標位置 Z = {target_z:.4f}")
        print(f"  Grab頂点最終位置 Z = {grabbed_final_z:.4f}")
        print(f"  下布中央頂点最終位置 Z = {bottom_final_z:.4f}")
        print(f"  上布と下布の相対間隔 (上 - 下) = {(grabbed_final_z - bottom_final_z)*1000:.2f} mm")

        # 判定: 上布が下布を突き抜けていないこと (grabbed_final_z >= bottom_final_z - 1e-4)
        self.assertGreaterEqual(
            grabbed_final_z, bottom_final_z - 1e-3,
            f"上布のGrab頂点が下布を突き抜けています！ (上: {grabbed_final_z:.4f}, 下: {bottom_final_z:.4f})"
        )

if __name__ == '__main__':
    unittest.main()
