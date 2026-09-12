"""
Taremin Cloth GUI IPC通信プロトコル自動テスト
ヘッドレスサーバーを起動し、Python クライアントから接続して
初期化、ステップ進行、ノンブロッキングフレーム取得、ポーズ適用の通信シーケンスを自動検証する。
"""

import os
import subprocess
import time
import unittest
import numpy as np

# プロジェクトルートをインポートパスに追加
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
addon_root = os.path.dirname(os.path.dirname(current_dir))
python_pkg_dir = os.path.join(addon_root, "python")
if python_pkg_dir not in sys.path:
    sys.path.insert(0, python_pkg_dir)

from taremin_cloth.engine.gui_client import ClothGuiClient


class TestGuiIpc(unittest.TestCase):
    SERVER_PORT = 9059
    server_process = None

    @classmethod
    def setUpClass(cls):
        # taremin_cloth_gui.exe のパスを解決
        exe_path = os.path.join(addon_root, "target", "release", "taremin_cloth_gui.exe")
        if not os.path.isfile(exe_path):
            exe_path = os.path.join(addon_root, "target", "debug", "taremin_cloth_gui.exe")

        if not os.path.isfile(exe_path):
            raise unittest.SkipTest(f"taremin_cloth_gui.exe not found at {exe_path}")

        # ヘッドレスモードでサーバーを起動
        cls.server_process = subprocess.Popen(
            [exe_path, "--port", str(cls.SERVER_PORT), "--headless"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # 起動待機
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        if cls.server_process and cls.server_process.poll() is None:
            cls.server_process.terminate()
            try:
                cls.server_process.wait(timeout=2.0)
            except Exception:
                cls.server_process.kill()

    def test_01_connect_and_disconnect(self):
        """接続と切断の基本テスト"""
        client = ClothGuiClient("127.0.0.1", self.SERVER_PORT)
        self.assertTrue(client.connect(timeout=2.0))
        self.assertTrue(client.is_connected)
        client.disconnect()
        self.assertFalse(client.is_connected)

    def test_02_init_scene_and_step(self):
        """メッシュ送信、ステップ実行、座標取得の統合テスト"""
        client = ClothGuiClient("127.0.0.1", self.SERVER_PORT)
        self.assertTrue(client.connect(timeout=2.0))

        # 4頂点・2三角形の正方形メッシュ
        positions = [
            [-0.5, -0.5, 1.0],
            [ 0.5, -0.5, 1.0],
            [ 0.5,  0.5, 1.0],
            [-0.5,  0.5, 1.0],
        ]
        faces = [
            [0, 1, 2],
            [0, 2, 3],
        ]
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0], [0, 2],
        ]
        inv_masses = [0.0, 1.0, 1.0, 1.0] # 頂点0を固定ピン

        # Mock オブジェクト
        class MockMesh:
            pass

        data = {
            "type": "InitScene",
            "data": {
                "object_name": "TestCloth",
                "positions": positions,
                "faces": faces,
                "edges": edges,
                "sewing_springs": None,
                "inv_masses": inv_masses,
                "layer_id": 0,
                "thickness": 0.01,
                "stiffness": 100.0,
                "compression_stiffness": 100.0,
                "shear_stiffness": 50.0,
                "bending_stiffness": 1.0,
                "sewing_shrink_speed": 0.5,
                "workgroup_size": 32,
                "solver_mode": 0,
            }
        }

        self.assertTrue(client._send_packet(data))
        resp = client._recv_packet(timeout=10.0)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.get("type"), "Ack")

        # 1ステップ実行要求
        ok = client.step(dt=1.0 / 60.0, substeps=10, solver_iterations=2)
        print("step ok:", ok)
        time.sleep(0.05)

        # 最新座標の取得
        client._send_packet({"type": "GetLatestCoords"})
        raw_resp = client._recv_packet(timeout=2.0)
        print("RAW RESP:", raw_resp)
        if raw_resp and raw_resp.get("type") == "Coords":
            data = raw_resp.get("data", {})
            print("COORDS DATA positions len:", len(data.get("positions", [])))

        res = client.get_latest_coords(timeout=2.0)
        print("get_latest_coords res:", res is not None)
        if res is None:
            time.sleep(0.1)
            import select
            # Windowsではパイプにselectが使えないため、非ブロッキングで読むかterminateしてcommunicate
            self.server_process.terminate()
            out, err = self.server_process.communicate(timeout=2.0)
            print("SERVER STDOUT:\n", out.decode('utf-8', errors='ignore'))
            print("SERVER STDERR:\n", err.decode('utf-8', errors='ignore'))
        self.assertIsNotNone(res)
        seq, coords, fps, ms = res
        self.assertGreater(seq, 0)
        self.assertEqual(len(coords), 12) # 4頂点 * 3

        # 頂点0は固定ピンなので初期位置 (-0.5, -0.5, 1.0) のままであること
        np.testing.assert_allclose(coords[0:3], [-0.5, -0.5, 1.0], atol=1e-3)

        # 頂点1〜3は重力(-Z)で落下していること (z < 1.0)
        self.assertLess(coords[5], 1.0) # 頂点1のZ
        self.assertLess(coords[8], 1.0) # 頂点2のZ

        # ポーズ適用要求
        applied = client.apply_pose()
        self.assertIsNotNone(applied)
        self.assertEqual(len(applied), 12)

        client.disconnect()

    def test_03_collider_transfer_and_collision(self):
        """コライダー（球コライダー）を転送し、布が球の上で衝突停止することの検証"""
        client = ClothGuiClient("127.0.0.1", self.SERVER_PORT)
        self.assertTrue(client.connect(timeout=2.0))

        # (0, 0, 0.8) に自由落下頂点を配置
        positions = [[0.0, 0.0, 0.8]]
        faces = []
        edges = []
        inv_masses = [1.0]

        # (0, 0, 0) に半径 0.5 の球コライダーを配置
        colliders = [
            {
                "collider_type": 0,
                "friction": 0.5,
                "restitution": 0.0,
                "point_a": [0.0, 0.0, 0.0],
                "radius": 0.5,
                "point_b": [0.0, 0.0, 0.0],
            }
        ]

        data = {
            "type": "InitScene",
            "data": {
                "object_name": "ColliderTestCloth",
                "positions": positions,
                "faces": faces,
                "edges": edges,
                "sewing_springs": None,
                "inv_masses": inv_masses,
                "layer_id": 0,
                "thickness": 0.01,
                "stiffness": 100.0,
                "compression_stiffness": 100.0,
                "shear_stiffness": 50.0,
                "bending_stiffness": 1.0,
                "sewing_shrink_speed": 0.5,
                "workgroup_size": 32,
                "solver_mode": 0,
                "colliders": colliders,
                "mesh_triangles": [],
            }
        }

        self.assertTrue(client._send_packet(data))
        resp = client._recv_packet(timeout=5.0)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.get("type"), "Ack")

        # 30ステップ（約0.5秒分）シミュレーションを進める
        for _ in range(30):
            client.step(dt=1.0 / 60.0, substeps=10, solver_iterations=2)

        res = client.get_latest_coords(timeout=2.0)
        self.assertIsNotNone(res)
        seq, coords, fps, ms = res
        self.assertEqual(len(coords), 3)

        # 球の半径は 0.5、布の厚みは 0.01 なので、衝突によって z >= 0.49 で留まっていること！
        # もしコライダーが無視されていれば、0.5秒の自由落下で z < 0.0 またはマイナスになっている。
        z_pos = coords[2]
        print(f"Collider collision final z position: {z_pos:.4f} (expected >= 0.49)")
        self.assertGreaterEqual(z_pos, 0.49)
        self.assertLessEqual(z_pos, 0.60)

        client.disconnect()

    def test_04_full_parameters_and_self_collision(self):
        """全物理パラメータ（剛性4種・減衰4種・重力）およびCoupled XPBD自己衝突・伸縮グループの初期化テスト"""
        client = ClothGuiClient("127.0.0.1", self.SERVER_PORT)
        self.assertTrue(client.connect(timeout=2.0))

        positions = [
            [-0.2, -0.2, 1.0],
            [ 0.2, -0.2, 1.0],
            [ 0.2,  0.2, 1.0],
            [-0.2,  0.2, 1.0],
        ]
        faces = [[0, 1, 2], [0, 2, 3]]
        edges = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]]
        inv_masses = [0.0, 1.0, 1.0, 1.0]

        data = {
            "type": "InitScene",
            "data": {
                "object_name": "FullParamsCloth",
                "positions": positions,
                "faces": faces,
                "edges": edges,
                "sewing_springs": None,
                "inv_masses": inv_masses,
                "layer_id": 1,
                "thickness": 0.02,
                "stiffness": 150.0,
                "compression_stiffness": 80.0,
                "shear_stiffness": 40.0,
                "bending_stiffness": 5.0,
                "air_damping": 0.05,
                "tension_damping": 0.1,
                "compression_damping": 0.1,
                "shear_damping": 0.05,
                "bending_damping": 0.02,
                "gravity": [0.0, 0.0, -9.81],
                "sewing_shrink_speed": 0.5,
                "workgroup_size": 32,
                "solver_mode": 0,
                "solver_iterations": 15,
                "substeps": 20,
                "self_collision": {
                    "enabled": True,
                    "coupled_mode": 2, # FULL_COUPLED
                    "post_relaxation_iters": 3,
                    "relief_factor": 0.3,
                    "max_displacement_ratio": 0.2,
                    "exclude_neighbors": True,
                    "enable_normal_untangling": True,
                    "enable_edge_collision": True,
                    "edge_margin_scale": 1.2,
                    "edge_margin_offset": 0.001,
                },
                "elastic_bands": {
                    "edge_indices": [0, 1],
                    "scales": [0.8, 0.8],
                },
                "colliders": [],
                "mesh_triangles": [],
                "voxels": [[0.0, 0.0, 0.5, 0.05, 1.0, 0.0, 0.0]],
                "extra_lines": [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]],
            }
        }

        self.assertTrue(client._send_packet(data))
        resp = client._recv_packet(timeout=5.0)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.get("type"), "Ack")

        # ステップ進行テスト
        for _ in range(10):
            self.assertTrue(client.step(dt=1.0 / 60.0, substeps=10, solver_iterations=10))

        res = client.get_latest_coords(timeout=2.0)
        self.assertIsNotNone(res)
        seq, coords, fps, ms = res
        self.assertEqual(len(coords), 12)
        # 固定ピン（頂点0）が固定されていること
        np.testing.assert_allclose(coords[0:3], [-0.2, -0.2, 1.0], atol=1e-3)
        client.disconnect()

    def test_05_bone_sdf_transfer_and_collision(self):
        """BONE_SDF（3Dテクスチャ・Base64転送）と布の衝突判定テスト"""
        import base64
        client = ClothGuiClient("127.0.0.1", self.SERVER_PORT)
        self.assertTrue(client.connect(timeout=2.0))

        # 16x16x16 の球状SDFボリューム（半径0.4、中心[0, 0, 0]、Rg16Float形式）
        res = 16
        xs = np.linspace(-0.5, 0.5, res, dtype=np.float32)
        ys = np.linspace(-0.5, 0.5, res, dtype=np.float32)
        zs = np.linspace(-0.5, 0.5, res, dtype=np.float32)
        grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='ij')
        dist = np.sqrt(grid_x**2 + grid_y**2 + grid_z**2) - 0.4

        rg_f16 = np.zeros((res, res, res, 2), dtype=np.float16)
        rg_f16[..., 0] = dist.astype(np.float16)
        rg_f16[..., 1] = np.float16(1.0) # alpha
        sdf_b64 = base64.b64encode(rg_f16.tobytes()).decode('ascii')

        # 頂点 (0, 0, 0.7) から自由落下
        positions = [[0.0, 0.0, 0.7]]
        inv_masses = [1.0]

        # GpuBoneInfo: 20要素
        # aabb_min(4), aabb_max(4), uvw_scale(4), uvw_offset(4), params(4)
        bone_info = [
            -0.5, -0.5, -0.5, 0.0,  # aabb_min
             0.5,  0.5,  0.5, 0.0,  # aabb_max
             1.0,  1.0,  1.0, 0.05, # uvw_scale, blend_k
             0.5,  0.5,  0.5, 0.0,  # uvw_offset
             0.3,  0.02, 0.0, 0.0,  # friction=0.3, thickness=0.02, restitution=0.0, threshold=0.0
        ]

        data = {
            "type": "InitScene",
            "data": {
                "object_name": "BoneSdfCloth",
                "positions": positions,
                "faces": [],
                "edges": [],
                "sewing_springs": None,
                "inv_masses": inv_masses,
                "layer_id": 0,
                "thickness": 0.01,
                "stiffness": 100.0,
                "compression_stiffness": 100.0,
                "shear_stiffness": 50.0,
                "bending_stiffness": 1.0,
                "gravity": [0.0, 0.0, -9.81],
                "sewing_shrink_speed": 0.5,
                "workgroup_size": 32,
                "solver_mode": 0,
                "solver_iterations": 5,
                "substeps": 10,
                "bone_sdf": {
                    "width": res,
                    "height": res,
                    "depth": res,
                    "texture_base64": sdf_b64,
                    "bone_infos": [bone_info],
                },
                "colliders": [],
                "mesh_triangles": [],
            }
        }

        self.assertTrue(client._send_packet(data))
        resp = client._recv_packet(timeout=5.0)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.get("type"), "Ack")

        # 30ステップ（約0.5秒）実行
        for _ in range(30):
            client.step(dt=1.0 / 60.0, substeps=10, solver_iterations=5)

        res = client.get_latest_coords(timeout=2.0)
        self.assertIsNotNone(res)
        seq, coords, fps, ms = res
        self.assertEqual(len(coords), 3)

        # SDFの球半径は 0.40、布厚みは 0.01 なので、衝突により z >= 0.38 付近で停止すること！
        z_pos = coords[2]
        print(f"Bone SDF collision final z position: {z_pos:.4f} (expected >= 0.38)")
        self.assertGreaterEqual(z_pos, 0.38)
        self.assertLessEqual(z_pos, 0.55)

        client.disconnect()

    def test_06_dynamic_updates(self):
        """ボーン変換行列・伸縮グループ・動的ピン・パラメータ更新のリアルタイム送信テスト"""
        client = ClothGuiClient("127.0.0.1", self.SERVER_PORT)
        self.assertTrue(client.connect(timeout=2.0))

        positions = [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]
        edges = [[0, 1]]
        inv_masses = [1.0, 1.0]

        data = {
            "type": "InitScene",
            "data": {
                "object_name": "DynamicUpdateCloth",
                "positions": positions,
                "faces": [],
                "edges": edges,
                "sewing_springs": None,
                "inv_masses": inv_masses,
                "layer_id": 0,
                "thickness": 0.01,
                "stiffness": 100.0,
                "compression_stiffness": 100.0,
                "shear_stiffness": 50.0,
                "bending_stiffness": 1.0,
                "gravity": [0.0, 0.0, -9.81],
                "sewing_shrink_speed": 0.5,
                "workgroup_size": 32,
                "solver_mode": 0,
                "colliders": [],
                "mesh_triangles": [],
            }
        }
        self.assertTrue(client._send_packet(data))
        self.assertIsNotNone(client._recv_packet(timeout=5.0))

        # 1. ボーン姿勢行列の動的更新
        bone_mats = [np.eye(4, dtype=np.float32).tolist()]
        self.assertTrue(client.send_update_bone_transforms(bone_mats))

        # 2. 伸縮グループスケールの動的更新
        self.assertTrue(client.send_update_elastic_scales([0], [0.75]))

        # 3. 動的ピンの更新
        self.assertTrue(client.send_update_pins([0], [[0.0, 0.0, 1.2]], [1.0]))

        # 4. パラメータのリアルタイム更新
        self.assertTrue(client.send_params(
            gravity=[0.0, 0.0, -5.0],
            air_damping=0.1,
            stiffness=200.0,
            solver_iterations=8,
        ))

        # ステップ進行が正常に動作することを確認
        self.assertTrue(client.step(dt=1.0 / 60.0, substeps=5, solver_iterations=5))
        res = client.get_latest_coords(timeout=2.0)
        self.assertIsNotNone(res)

        client.disconnect()

    def test_07_numpy_types_serialization(self):
        """NumPyスカラー型（np.float32, np.int64等）やndarrayがJSONシリアライズ可能であることの検証"""
        client = ClothGuiClient("127.0.0.1", self.SERVER_PORT)
        self.assertTrue(client.connect(timeout=2.0))

        # numpy.float32 や numpy.int64、numpy.ndarray をそのままパケットデータに含める
        data = {
            "type": "InitScene",
            "data": {
                "object_name": "NumpyTypeCloth",
                "positions": np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                "faces": [],
                "edges": [],
                "sewing_springs": None,
                "inv_masses": [np.float32(1.0)],
                "layer_id": np.int32(0),
                "thickness": np.float32(0.01),
                "stiffness": np.float64(100.0),
                "compression_stiffness": np.float32(100.0),
                "shear_stiffness": np.float32(50.0),
                "bending_stiffness": np.float32(1.0),
                "gravity": [np.float32(0.0), np.float32(0.0), np.float32(-9.81)],
                "sewing_shrink_speed": np.float32(0.5),
                "workgroup_size": np.int32(32),
                "solver_mode": np.int32(0),
                "colliders": [],
                "mesh_triangles": [],
            }
        }
        # エラーなく送信され、Ackが返ってくること
        self.assertTrue(client._send_packet(data))
        resp = client._recv_packet(timeout=5.0)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.get("type"), "Ack")

        client.disconnect()


if __name__ == "__main__":
    unittest.main()


