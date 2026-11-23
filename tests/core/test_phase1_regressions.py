"""
Phase 1 回帰テスト群（Blender非依存・スタンドアロン実行可能）。

検証対象:
1. collider._extract_primitive_collider_data: PLANE法線 / CAPSULE端点 / SPHEREスケール変換
2. collider.sync_colliders: BONE_SDF 関節メッシュ接続、複数SDF抑止、SDF 0件時クリア
3. topology.apply_dynamic_diagonal_triangulation: bytes 形式レスト座標復元
4. runner.bake_step_frame / cloth_frame_handler: 紐パーツ付き布のタイムライン保存・復元
5. brush.range_grab.RangeGrabTool: ユーザーピンの保護（除外・非解放）
6. brush.smooth.SmoothTool: ActiveRegionManager の初期化・凍結・解放連動
"""

import math
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
for p in (project_root, python_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import tests  # noqa: F401,E402  -- Blender モックおよび sys.path を安全に初期化

from taremin_cloth.engine import cache, collider, runner
from taremin_cloth.engine.sdf_baker import BoneSdfBakeResult
from taremin_cloth.utils.curve_extract import CurvePartRuntimeData
from taremin_cloth.brush.range_grab import RangeGrabTool
from taremin_cloth.brush.smooth import SmoothTool
from taremin_cloth.brush.base import BrushContext


class _FakeVec:
    """テスト用の最小 3D ベクトル実装。"""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __getitem__(self, idx):
        return (self.x, self.y, self.z)[idx]

    @property
    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def normalized(self):
        ln = self.length
        if ln > 1e-8:
            return _FakeVec(self.x / ln, self.y / ln, self.z / ln)
        return _FakeVec(0.0, 0.0, 1.0)

    def copy(self):
        return _FakeVec(self.x, self.y, self.z)

    def __neg__(self):
        return _FakeVec(-self.x, -self.y, -self.z)

    def __sub__(self, other):
        return _FakeVec(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other):
        return _FakeVec(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, s):
        return _FakeVec(self.x * s, self.y * s, self.z * s)

    def __rmul__(self, s):
        return _FakeVec(self.x * s, self.y * s, self.z * s)


def _rot_x_matrix(angle_rad: float, *, scale=(1.0, 1.0, 1.0), translation=(0.0, 0.0, 0.0)):
    """X軸回転・スケール・平行移動を持つ 4x4 numpy 変換行列を作成する。"""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )
    scl = np.diag([float(scale[0]), float(scale[1]), float(scale[2])])
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot @ scl
    mat[:3, 3] = np.array(translation, dtype=np.float64)
    return mat


class TestPhase1ColliderRegressions(unittest.TestCase):
    """解析コライダー変換および sync_colliders の回帰テスト。"""

    def setUp(self):
        collider.clear_collider_cache()
        collider.clear_collider_motion_cache()

    def tearDown(self):
        collider.clear_collider_cache()
        collider.clear_collider_motion_cache()

    def test_primitive_collider_transforms_plane_capsule_sphere(self):
        """PLANE法線・CAPSULE端点回転/水平スケール・SPHEREワールドスケールが正しく計算されること。"""
        # 1. X軸+90度回転 + 非一様スケール (2, 1, 0.5) の PLANE
        #    ローカル法線 +Z (0,0,1) -> ワールド法線は (0, -1, 0)
        plane_obj = SimpleNamespace(
            matrix_world=_rot_x_matrix(math.pi / 2.0, scale=(2.0, 1.0, 0.5), translation=(1.0, 2.0, 3.0)),
            scale=(1.0, 1.0, 1.0),
        )
        plane_set = SimpleNamespace(collider_type='PLANE', friction=0.4, restitution=0.1, radius=1.0)
        pdata = collider._extract_primitive_collider_data(plane_obj, plane_set)
        self.assertIsNotNone(pdata)
        self.assertEqual(pdata["collider_type"], 2)
        self.assertAlmostEqual(pdata["point_a"][0], 1.0, places=5)
        self.assertAlmostEqual(pdata["point_a"][1], 2.0, places=5)
        self.assertAlmostEqual(pdata["point_a"][2], 3.0, places=5)
        self.assertAlmostEqual(pdata["point_b"][0], 0.0, places=5)
        self.assertAlmostEqual(pdata["point_b"][1], -1.0, places=5)
        self.assertAlmostEqual(pdata["point_b"][2], 0.0, places=5)

        # 2. X軸+90度回転 + 水平スケール X=2.0 の CAPSULE
        #    ローカル (0,0,-r) -> (0, +r, 0), (0,0,+r) -> (0, -r, 0), radius = r * 2.0
        cap_obj = SimpleNamespace(
            matrix_world=_rot_x_matrix(math.pi / 2.0, scale=(2.0, 1.0, 1.0), translation=(0.0, 0.0, 1.0)),
            scale=(1.0, 1.0, 1.0),
        )
        cap_set = SimpleNamespace(collider_type='CAPSULE', friction=0.5, restitution=0.0, radius=0.25)
        cdata = collider._extract_primitive_collider_data(cap_obj, cap_set)
        self.assertIsNotNone(cdata)
        self.assertEqual(cdata["collider_type"], 1)
        self.assertAlmostEqual(cdata["radius"], 0.50, places=5)
        self.assertAlmostEqual(cdata["point_a"][1], 0.25, places=5)
        self.assertAlmostEqual(cdata["point_a"][2], 1.0, places=5)
        self.assertAlmostEqual(cdata["point_b"][1], -0.25, places=5)
        self.assertAlmostEqual(cdata["point_b"][2], 1.0, places=5)

        # 3. matrix_world に親スケール 3.0 を持つ SPHERE (obj.scale は (1,1,1))
        sph_obj = SimpleNamespace(
            matrix_world=_rot_x_matrix(0.0, scale=(3.0, 1.0, 1.0), translation=(0.0, 0.0, 0.0)),
            scale=(1.0, 1.0, 1.0),
        )
        sph_set = SimpleNamespace(collider_type='SPHERE', friction=0.3, restitution=0.0, radius=0.5)
        sdata = collider._extract_primitive_collider_data(sph_obj, sph_set)
        self.assertIsNotNone(sdata)
        self.assertAlmostEqual(sdata["radius"], 1.5, places=5)

    def test_sync_colliders_joint_mesh_multi_sdf_guard_and_clear(self):
        """sync_colliders で BONE_SDF 関節三角形送信・複数SDF上書き抑止・0件時クリアが動作すること。"""
        class FakeSim:
            def __init__(self):
                self.bone_sdf_calls = []
                self.clear_bone_calls = 0
                self.mesh_tris_calls = []
                self.bone_enabled = False
                self.bone_transform_calls = 0

            def clear_colliders(self):
                pass

            def set_bone_sdf_colliders(self, w, h, d, tex_bytes, bone_infos):
                self.bone_sdf_calls.append((w, h, d))
                self.bone_enabled = True

            def clear_bone_sdf_colliders(self):
                self.clear_bone_calls += 1
                self.bone_enabled = False

            def bone_sdf_enabled(self):
                return self.bone_enabled

            def set_mesh_collider_triangles(self, tris, *, attributes=None):
                self.mesh_tris_calls.append(np.asarray(tris))

            def update_bone_transforms(self, w_arr, inv_arr, dirty=None):
                self.bone_transform_calls += 1

            def set_collider_options(self, **kwargs):
                pass

        class _Mat4:
            def __init__(self):
                self.translation = _FakeVec(0.0, 0.0, 0.0)
                self._m = np.eye(4, dtype=np.float32)

            def __array__(self, dtype=None, copy=None):
                return self._m.astype(dtype) if dtype is not None else self._m

            def __getitem__(self, idx):
                return self._m[idx]

        arm_a = SimpleNamespace(
            name="ArmA",
            pose=SimpleNamespace(bones={}),
            matrix_world=_Mat4(),
            animation_data=None,
            data=None,
            evaluated_get=lambda dg: SimpleNamespace(matrix_world=_Mat4(), pose=SimpleNamespace(bones={})),
        )
        arm_b = SimpleNamespace(
            name="ArmB",
            pose=SimpleNamespace(bones={}),
            matrix_world=_Mat4(),
            animation_data=None,
            data=None,
            evaluated_get=lambda dg: SimpleNamespace(matrix_world=_Mat4(), pose=SimpleNamespace(bones={})),
        )
        arm_mod_a = SimpleNamespace(type='ARMATURE', object=arm_a)
        arm_mod_b = SimpleNamespace(type='ARMATURE', object=arm_b)

        def _make_body(name, arm_mod):
            col_set = SimpleNamespace(
                is_collider=True,
                enabled=True,
                collider_type='BONE_SDF',
                sdf_update_mode='STATIC',
                sdf_resolution='64',
                sdf_margin=0.1,
                thickness=0.005,
                friction=0.5,
                restitution=0.0,
                radius=0.1,
                weight_threshold=0.02,
                blend_k=0.05,
                enable_joint_mesh=True,
                joint_weight_threshold=0.85,
                joint_rotation_threshold=0.0,
                mesh_sdf_voxel_size=0.004,
                mesh_sdf_margin=0.02,
                mesh_sdf_max_vram_mb=128,
                mesh_sdf_auto_scale=True,
                close_open_boundaries=True,
                single_sided=True,
                enable_single_sided_recovery=True,
                enable_cluster_culling=False,
                sweep_margin_offset=0.05,
                anim=None,
            )
            return SimpleNamespace(
                name=name,
                type='MESH',
                taremin_cloth_collider=col_set,
                matrix_world=_Mat4(),
                modifiers=[arm_mod],
                animation_data=None,
                data=SimpleNamespace(vertices=[0, 1, 2], shape_keys=None),
                evaluated_get=lambda dg: SimpleNamespace(matrix_world=_Mat4()),
            )

        body_a = _make_body("BodyA", arm_mod_a)
        body_b = _make_body("BodyB", arm_mod_b)

        bone_infos = np.zeros((1, 20), dtype=np.float32)
        bone_infos[0, 7] = 4.0
        fake_bake = BoneSdfBakeResult(
            texture_bytes=b"\x00" * (4 * 4 * 4 * 4),
            width=4,
            height=4,
            depth=4,
            bone_infos=bone_infos,
            bone_names=["Bone0"],
            bind_matrices=np.eye(4, dtype=np.float32).reshape((1, 4, 4)),
            joint_face_indices=np.array([0, 1], dtype=np.int32),
            joint_faces_by_pair={},
        )
        joint_tris = np.zeros((2, 3, 3), dtype=np.float32)
        joint_attrs = np.zeros((2, 4), dtype=np.float32)

        sim = FakeSim()
        scene_two = SimpleNamespace(name="Scene", frame_current=1, objects=[body_a, body_b])
        scene_empty = SimpleNamespace(name="Scene", frame_current=2, objects=[])

        with patch("taremin_cloth.engine.collider.get_or_bake_bone_sdf_for_object", return_value=fake_bake) as mock_bake, \
             patch("taremin_cloth.engine.collider.get_collider_eval_mesh", return_value=(object(), None, [])), \
             patch("taremin_cloth.engine.collider._extract_mesh_collider_triangles", return_value=(joint_tris, joint_attrs, [])), \
             patch.object(collider.bpy.data, "objects", {"ArmA": arm_a, "ArmB": arm_b, "BodyA": body_a, "BodyB": body_b}):
            # 2体の BONE_SDF が存在する場合: 1体目のみ登録され2体目はスキップ、関節三角形は同期される
            collider.sync_colliders(sim, scene_two)
            self.assertEqual(mock_bake.call_count, 1, "2体目の SDF はスキップされベイクされないこと")
            self.assertEqual(len(sim.bone_sdf_calls), 1)
            self.assertEqual(len(sim.mesh_tris_calls), 1)
            self.assertEqual(len(sim.mesh_tris_calls[0]), 2, "STATIC BONE_SDF の関節三角形が同期されること")
            self.assertIsInstance(collider._bone_sdf_cache[id(sim)][0], str, "キャッシュはオブジェクト名文字列を保持すること")

            # 次フレーム（同一状態、force=True で再進入しても再ベイクされないこと）
            scene_two.frame_current = 2
            collider.sync_colliders(sim, scene_two, force=True)
            self.assertEqual(mock_bake.call_count, 1, "キャッシュ済みの SDF は次フレームでリベイクされないこと")

            # SDF コライダー除去後は clear_bone_sdf_colliders() が呼ばれること
            collider.sync_colliders(sim, scene_empty)
            self.assertEqual(sim.clear_bone_calls, 1)
            self.assertNotIn(id(sim), collider._bone_sdf_cache)


class TestPhase1TopologyBytesRest(unittest.TestCase):
    """apply_dynamic_diagonal_triangulation の bytes 型レスト座標復元テスト。"""

    def test_bytes_rest_positions_decoded_correctly(self):
        """_taremin_cloth_rest_positions が bytes のとき正しく復元して歪みベース分割が行われること。"""
        from taremin_cloth.utils import topology

        # レスト形状: 単位正方形 (0,0)-(1,0)-(1,1)-(0,1)
        rest_pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        # 現在形状: 対角線 (0,2) が強く引き伸ばされるよう頂点 2 を移動
        curr_pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.5, 2.5, 0.5],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        class _FakeBMSeq(list):
            def ensure_lookup_table(self):
                pass

        bm_verts = _FakeBMSeq(SimpleNamespace(index=i, co=curr_pts[i]) for i in range(4))
        bm_face = SimpleNamespace(verts=bm_verts)
        split_calls = []

        fake_bm = MagicMock()
        fake_bm.verts = bm_verts
        fake_bm.faces = _FakeBMSeq([bm_face])

        class _MockMeshData:
            def __init__(self):
                self.vertices = MagicMock()
                self.vertices.__len__ = lambda _s: 4
                self.vertices.foreach_get = lambda attr, arr: arr.__setitem__(slice(None), curr_pts.flatten())
                self.polygons = [0]

            def update(self):
                pass

            def copy(self):
                return self

        class _MockObj:
            def __init__(self):
                self.name = "QuadCloth"
                self.type = 'MESH'
                self.data = _MockMeshData()
                self._props = {
                    "_taremin_cloth_rest_positions": rest_pts.tobytes(),
                }

            def __contains__(self, key):
                return key in self._props

            def __getitem__(self, key):
                return self._props[key]

            def __setitem__(self, key, val):
                self._props[key] = val

        obj = _MockObj()
        with patch.object(topology.bmesh, "new", return_value=fake_bm), \
             patch.object(topology.bmesh.utils, "face_split", side_effect=lambda f, va, vb: split_calls.append((va.index, vb.index))), \
             patch.object(topology, "backup_pre_subdivision_mesh"):
            ok = topology.apply_dynamic_diagonal_triangulation(obj, preserve_flat=False)

        self.assertTrue(ok)
        self.assertEqual(len(split_calls), 1)
        # bytes 復元されずに curr_pts フォールバックされた場合は (1, 3) が選ばれるケースになるよう、
        # (0, 2) と (1, 3) の歪み差を期待する: rest vs curr で evaluate_quad_diagonal_by_strain と一致すること
        expected_02 = topology.evaluate_quad_diagonal_by_strain(
            rest_pts[0], rest_pts[1], rest_pts[2], rest_pts[3],
            curr_pts[0], curr_pts[1], curr_pts[2], curr_pts[3],
        )
        expected_pair = (0, 2) if expected_02 else (1, 3)
        self.assertEqual(split_calls[0], expected_pair)


class TestPhase1RunnerCurveBake(unittest.TestCase):
    """紐パーツ付き布オブジェクトのベイクキャッシュ保存・フレーム再生テスト。"""

    def setUp(self):
        cache.clear_simulators()

    def tearDown(self):
        cache.clear_simulators()

    def test_bake_step_and_frame_handler_restore_mesh_and_curve(self):
        """紐付き布の bake_step_frame と cloth_frame_handler でキャッシュヒットし両方復元されること。"""
        applied_curve_coords = []

        class _MockVertices:
            def __init__(self, arr):
                self.arr = arr.copy()

            def __len__(self):
                return len(self.arr) // 3

            def foreach_set(self, attr, values):
                if attr == "co":
                    self.arr = np.asarray(values, dtype=np.float32).copy()

            def foreach_get(self, attr, out):
                if attr == "co":
                    out[:] = self.arr

        mesh_rest = np.zeros(4 * 3, dtype=np.float32)
        mesh_data = MagicMock()
        mesh_data.vertices = _MockVertices(mesh_rest)
        mesh_data.edges = []
        mesh_data.polygons = []

        cloth_settings = SimpleNamespace(
            is_cloth=True,
            enabled=True,
            solver_iterations=2,
            string_parts=[],
        )
        cloth_obj = SimpleNamespace(
            name="HoodieCloth",
            type='MESH',
            data=mesh_data,
            taremin_cloth=cloth_settings,
            matrix_world=np.eye(4, dtype=np.float32),
            _props={},
        )
        # dict 風アクセスの代わりに属性辞書ラップを使う
        class _DictObj:
            def __init__(self):
                self.name = "HoodieCloth"
                self.type = 'MESH'
                self.data = mesh_data
                self.taremin_cloth = cloth_settings
                self.matrix_world = np.eye(4, dtype=np.float32)
                self._p = {"_taremin_cloth_is_deformed": False}

            def get(self, k, d=None):
                return self._p.get(k, d)

            def __contains__(self, k):
                return k in self._p

            def __getitem__(self, k):
                return self._p[k]

            def __setitem__(self, k, v):
                self._p[k] = v

            def update_tag(self):
                pass

        cloth_obj = _DictObj()
        curve_obj = SimpleNamespace(
            name="DrawstringCurve",
            type='CURVE',
            matrix_world=np.eye(4, dtype=np.float32),
            data=SimpleNamespace(splines=[]),
        )

        # 4 mesh verts + 3 curve verts = 7 verts
        sim_coords = np.arange(7 * 3, dtype=np.float32).reshape((7, 3)) * 0.1
        flat_sim_coords = sim_coords.flatten()
        fake_sim = MagicMock()
        cache._simulators[cloth_obj.name] = (fake_sim, flat_sim_coords)
        cache._timeline_frame_cache[cloth_obj.name] = {}
        cache.set_curve_parts_runtime(
            cloth_obj.name,
            [
                CurvePartRuntimeData(
                    curve_obj_name="DrawstringCurve",
                    vertex_offset=4,
                    vertex_count=3,
                    edge_offset=0,
                    left_tail_edge_indices=[],
                    channel_edge_indices=[],
                    right_tail_edge_indices=[],
                    seam_bridge_edge_indices=[],
                    initial_left_len=0.1,
                    initial_channel_len=0.1,
                    initial_right_len=0.1,
                    guide_cloth_verts=[],
                    eyelet_pairs=[],
                )
            ],
        )

        scene = SimpleNamespace(
            frame_start=1,
            frame_current=2,
            objects=[cloth_obj],
            frame_set=lambda f: None,
        )
        bake_ctx = {
            "scene": scene,
            "cloth_objs": [cloth_obj],
            "start_frame": 1,
            "end_frame": 2,
            "total_steps": 1,
            "fps": 60,
            "dt": 1.0 / 60.0,
            "baked_count": 1,
        }

        with patch.object(runner, "step_cloth_object", return_value=(fake_sim, flat_sim_coords)), \
             patch.object(runner.bpy.data, "objects", {"HoodieCloth": cloth_obj, "DrawstringCurve": curve_obj}):
            runner.bake_step_frame(bake_ctx, 2)

        # メッシュキャッシュ (4*3 = 12要素) と紐キャッシュ (3*3) が別々に記録されること
        self.assertIn(2, cache._timeline_frame_cache[cloth_obj.name])
        self.assertEqual(len(cache._timeline_frame_cache[cloth_obj.name][2]), 12)
        self.assertIn(2, cache._timeline_curve_cache[cloth_obj.name])

        # メッシュ頂点をゼロへ戻した上で cloth_frame_handler(frame=2) を呼んでもキャッシュヒットで復元されること
        cloth_obj.data.vertices.arr[:] = 0.0
        with patch.object(runner, "check_and_update_params_signature", return_value=False), \
             patch.object(runner, "restore_rest_positions") as mock_restore, \
             patch.object(runner.bpy.data, "objects", {"HoodieCloth": cloth_obj, "DrawstringCurve": curve_obj}), \
             patch("taremin_cloth.utils.curve_extract.apply_coords_to_curve", side_effect=lambda o, c, **kw: applied_curve_coords.append((o.name, c.copy(), kw))):
            runner.cloth_frame_handler(scene)

        mock_restore.assert_not_called()
        np.testing.assert_allclose(cloth_obj.data.vertices.arr, flat_sim_coords[:12], atol=1e-5)
        self.assertEqual(len(applied_curve_coords), 1)
        self.assertEqual(applied_curve_coords[0][0], "DrawstringCurve")
        np.testing.assert_allclose(applied_curve_coords[0][1], sim_coords[4:7], atol=1e-5)


class TestPhase1BrushRegressions(unittest.TestCase):
    """RangeGrabTool のユーザーピン保護と SmoothTool の ActiveRegion 連動テスト。"""

    def test_range_grab_protects_pinned_verts(self):
        """RangeGrabTool で _pinned_verts 内の頂点が掴まれず、解放対象にもならないこと。"""
        class FakeSim:
            def __init__(self):
                self.set_calls = []
                self.release_calls = []

            def set_pin(self, idx, pos, w):
                self.set_calls.append(int(idx))

            def set_pins_batch(self, indices, targets, weights):
                for idx in indices:
                    self.set_calls.append(int(idx))

            def release_pin(self, idx):
                self.release_calls.append(int(idx))

            def release_pins_batch(self, indices):
                for idx in indices:
                    self.release_calls.append(int(idx))

        coords = np.array(
            [
                [0.0, 0.0, 0.0],   # 0: 範囲内・非ピン
                [0.01, 0.0, 0.0],  # 1: 範囲内・ユーザーピン済み
                [0.02, 0.0, 0.0],  # 2: 範囲内・非ピン
                [5.0, 0.0, 0.0],   # 3: ActiveRegion 外・非ピン
            ],
            dtype=np.float32,
        )
        fake_res = {
            "center": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "origin": _FakeVec(0.0, 0.0, 2.0),
            "direction": _FakeVec(0.0, 0.0, -1.0),
            "hit_t": 2.0,
            "pos_2d": coords,
        }
        sim = FakeSim()
        brush_set = SimpleNamespace(radius=0.1, strength=0.8, falloff_shape='SMOOTH', use_active_region=True, active_region_scale=2.0)
        obj = SimpleNamespace(name="BrushCloth", taremin_cloth=SimpleNamespace(brush=brush_set))
        op = SimpleNamespace(_pinned_verts={1})
        region_data = MagicMock()
        fake_col = [_FakeVec(1, 0, 0), _FakeVec(0, 1, 0), _FakeVec(0, 0, 1)]
        region_data.view_matrix.inverted.return_value.to_3x3.return_value.col = fake_col
        blender_ctx = SimpleNamespace(region=MagicMock(), region_data=region_data)
        ctx = BrushContext(op, blender_ctx, obj, sim, coords.flatten(), SimpleNamespace(), (100, 100))

        tool = RangeGrabTool()
        with patch("taremin_cloth.brush.range_grab.resolve_brush_center", return_value=fake_res), \
             patch("taremin_cloth.brush.range_grab.update_brush_circle"), \
             patch("taremin_cloth.brush.range_grab.mathutils.Vector", side_effect=lambda v: _FakeVec(*v)):
            ok = tool.on_press(ctx)

        self.assertTrue(ok)
        self.assertNotIn(1, tool.grab)
        self.assertNotIn(1, sim.set_calls, "ユーザーピン済み頂点 1 に対して set_pin が呼ばれないこと")
        self.assertIn(0, tool.grab)
        self.assertIn(3, tool.active_region._temp_frozen_indices)

        tool.on_release(ctx, op._pinned_verts)
        self.assertNotIn(1, sim.release_calls, "解放時にユーザーピン済み頂点 1 が解除されないこと")
        self.assertIn(0, sim.release_calls)
        self.assertIn(3, sim.release_calls)
        self.assertFalse(tool.active_region.is_active)

    def test_smooth_tool_active_region_lifecycle(self):
        """SmoothTool が active_region を持ち on_press で凍結、on_release/abort で解放すること。"""
        class FakeSim:
            def __init__(self):
                self.set_calls = []
                self.release_calls = []

            def set_pin(self, idx, pos, w):
                self.set_calls.append(int(idx))

            def set_pins_batch(self, indices, targets, weights):
                for idx in indices:
                    self.set_calls.append(int(idx))

            def release_pin(self, idx):
                self.release_calls.append(int(idx))

            def release_pins_batch(self, indices):
                for idx in indices:
                    self.release_calls.append(int(idx))

        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [0.0, 0.02, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        fake_res = {
            "center": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "origin": _FakeVec(0.0, 0.0, 2.0),
            "direction": _FakeVec(0.0, 0.0, -1.0),
            "hit_t": 2.0,
            "pos_2d": coords,
        }

        tool = SmoothTool()
        self.assertTrue(hasattr(tool, "active_region"))
        tool.adj_offsets = np.array([0, 2, 3, 4, 4], dtype=np.uint32)
        tool.adj_indices = np.array([1, 2, 0, 0], dtype=np.uint32)
        tool.adj_num_verts = 4

        sim = FakeSim()
        brush_set = SimpleNamespace(radius=0.1, strength=0.5, falloff_shape='SMOOTH', use_active_region=True, active_region_scale=2.0)
        obj = SimpleNamespace(name="SmoothCloth", taremin_cloth=SimpleNamespace(brush=brush_set))
        op = SimpleNamespace(_pinned_verts={3})
        ctx = BrushContext(op, SimpleNamespace(region=MagicMock(), region_data=MagicMock()), obj, sim, coords.flatten(), SimpleNamespace(), (100, 100))

        # 頂点3を遠方かつピン除外対象から外すため、頂点4 (far_unpinned) も確認用に追加する場合は coords 5点にする方が厳密
        coords5 = np.vstack([coords, np.array([[6.0, 0.0, 0.0]], dtype=np.float32)])
        fake_res["pos_2d"] = coords5
        tool.adj_offsets = np.array([0, 2, 3, 4, 4, 4], dtype=np.uint32)
        tool.adj_num_verts = 5
        ctx.coords = coords5.flatten()

        with patch("taremin_cloth.brush.smooth.resolve_brush_center", return_value=fake_res), \
             patch("taremin_cloth.brush.smooth.update_brush_circle"):
            ok = tool.on_press(ctx)

        self.assertTrue(ok)
        self.assertTrue(tool.active_region.is_active)
        self.assertEqual(tool.active_region._temp_frozen_indices, [4])
        self.assertNotIn(3, sim.set_calls)

        tool.on_release(ctx, op._pinned_verts)
        self.assertFalse(tool.active_region.is_active)
        self.assertIn(4, sim.release_calls)
        self.assertNotIn(3, sim.release_calls)


if __name__ == "__main__":
    unittest.main()
