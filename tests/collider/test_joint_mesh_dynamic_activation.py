import math
from pathlib import Path
import sys
import unittest
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tests  # noqa: F401
from taremin_cloth.engine.collider import _extract_active_joint_mesh_triangles
from taremin_cloth.engine.sdf_baker import BoneSdfBakeResult


class _MockPoseBone:
    def __init__(
        self,
        rotation_quaternion=(1.0, 0.0, 0.0, 0.0),
        rotation_mode='QUATERNION',
        rotation_axis_angle=(0.0, 1.0, 0.0, 0.0),
        rotation_euler=(0.0, 0.0, 0.0),
    ):
        self.rotation_quaternion = rotation_quaternion
        self.rotation_mode = rotation_mode
        self.rotation_axis_angle = rotation_axis_angle
        self.rotation_euler = type("_Eul", (), {
            "x": rotation_euler[0], "y": rotation_euler[1], "z": rotation_euler[2]
        })()


class _MockPose:
    def __init__(self, bones_dict):
        self.bones = bones_dict


class _MockArmatureObject:
    def __init__(self, bones_dict):
        self.pose = _MockPose(bones_dict)


class _MockArmMod:
    def __init__(self, arm_obj):
        self.object = arm_obj


class _MockColSettings:
    def __init__(self, rot_threshold_deg: float):
        self.enable_joint_mesh = True
        self.joint_rotation_threshold = rot_threshold_deg
        self.friction = 0.4
        self.thickness = 0.02
        self.restitution = 0.0
        self.single_sided = True
        self.enable_single_sided_recovery = True


class _FakeVerts:
    def __init__(self, coords_flat: np.ndarray):
        self._co = coords_flat
        self._n = len(coords_flat) // 3

    def __len__(self):
        return self._n

    def foreach_get(self, attr, out):
        assert attr == "co"
        out[:] = self._co


class _FakeLoopTris:
    def __init__(self, indices_flat: np.ndarray):
        self._idx = indices_flat
        self._n = len(indices_flat) // 3

    def __len__(self):
        return self._n

    def foreach_get(self, attr, out):
        assert attr == "vertices"
        out[:] = self._idx


class _FakeMesh:
    """三角形 f の第0頂点 x 座標に float(f) を格納した識別用テストメッシュ。"""

    def __init__(self, n_tris: int = 40):
        verts = []
        tris = []
        for f in range(n_tris):
            base = f * 3
            verts.extend([
                [float(f), 0.0, 0.0],
                [float(f), 0.1, 0.0],
                [float(f), 0.0, 0.1],
            ])
            tris.append([base, base + 1, base + 2])
        self.vertices = _FakeVerts(np.asarray(verts, dtype=np.float32).ravel())
        self.loop_triangles = _FakeLoopTris(np.asarray(tris, dtype=np.int32).ravel())

    def calc_loop_triangles(self):
        pass


class _FakeTargetObj:
    def __init__(self, n_tris: int = 40):
        self.data = _FakeMesh(n_tris)
        self.matrix_world = np.eye(4, dtype=np.float32)
        self.modifiers = []


class TestJointMeshDynamicActivation(unittest.TestCase):
    def setUp(self):
        self.faces_spine = np.array([10, 11, 12, 13], dtype=np.int32)
        self.faces_knee_l = np.array([20, 21, 22], dtype=np.int32)
        self.faces_knee_r = np.array([30, 31, 32], dtype=np.int32)
        all_faces = np.unique(np.concatenate([self.faces_spine, self.faces_knee_l, self.faces_knee_r]))

        joint_faces_by_pair = {
            ("hips", "spine"): self.faces_spine,
            ("thigh.L", "shin.L"): self.faces_knee_l,
            ("thigh.R", "shin.R"): self.faces_knee_r,
        }

        self.bake_res = BoneSdfBakeResult(
            texture_bytes=b"",
            width=16, height=16, depth=16,
            bone_infos=np.zeros((3, 20), dtype=np.float32),
            bone_names=["hips", "spine", "thigh.L", "shin.L", "thigh.R", "shin.R"],
            bind_matrices=np.zeros((3, 4, 4), dtype=np.float32),
            joint_face_indices=all_faces,
            joint_faces_by_pair=joint_faces_by_pair,
        )
        self.obj = _FakeTargetObj(n_tris=40)

    @staticmethod
    def _extracted_face_ids(tri_coords):
        if tri_coords is None:
            return []
        return sorted(int(round(float(tri[0, 0]))) for tri in tri_coords)

    def test_static_pose_returns_zero_faces(self):
        """直立静止ポーズ（回転角0度）では本番関数が None (SDF単体モード) を返すこと"""
        bones = {
            "spine": _MockPoseBone((1.0, 0.0, 0.0, 0.0)),
            "shin.L": _MockPoseBone((1.0, 0.0, 0.0, 0.0)),
            "shin.R": _MockPoseBone((1.0, 0.0, 0.0, 0.0)),
        }
        arm_mod = _MockArmMod(_MockArmatureObject(bones))

        tri_coords, attr_block, dict_list = _extract_active_joint_mesh_triangles(
            self.obj,
            _MockColSettings(rot_threshold_deg=2.0),
            arm_mod=arm_mod,
            bake_res=self.bake_res,
            depsgraph=None,
        )
        self.assertIsNone(tri_coords)
        self.assertIsNone(attr_block)
        self.assertIsNone(dict_list)

    def test_below_threshold_returns_zero_faces(self):
        """閾値（2.0度）未満の微小な回転（1.0度）では本番関数がアクティブ面を返さないこと"""
        ang_rad = math.radians(1.0)
        q = (math.cos(ang_rad * 0.5), math.sin(ang_rad * 0.5), 0.0, 0.0)
        bones = {
            "spine": _MockPoseBone(q),
            "shin.L": _MockPoseBone((1.0, 0.0, 0.0, 0.0)),
            "shin.R": _MockPoseBone((1.0, 0.0, 0.0, 0.0)),
        }
        arm_mod = _MockArmMod(_MockArmatureObject(bones))

        tri_coords, attr_block, dict_list = _extract_active_joint_mesh_triangles(
            self.obj,
            _MockColSettings(rot_threshold_deg=2.0),
            arm_mod=arm_mod,
            bake_res=self.bake_res,
            depsgraph=None,
        )
        self.assertIsNone(tri_coords)
        self.assertIsNone(attr_block)
        self.assertIsNone(dict_list)

    def test_bent_joint_selectively_activated(self):
        """背骨のみ前屈（30度）した際、本番関数が腰・背骨関節面のみを正確に抽出すること"""
        ang_rad = math.radians(30.0)
        q = (math.cos(ang_rad * 0.5), math.sin(ang_rad * 0.5), 0.0, 0.0)
        bones = {
            "spine": _MockPoseBone(q),
            "shin.L": _MockPoseBone((1.0, 0.0, 0.0, 0.0)),
            "shin.R": _MockPoseBone((1.0, 0.0, 0.0, 0.0)),
        }
        arm_mod = _MockArmMod(_MockArmatureObject(bones))

        tri_coords, attr_block, dict_list = _extract_active_joint_mesh_triangles(
            self.obj,
            _MockColSettings(rot_threshold_deg=2.0),
            arm_mod=arm_mod,
            bake_res=self.bake_res,
            depsgraph=None,
        )
        self.assertIsNotNone(tri_coords)
        self.assertEqual(tri_coords.shape, (len(self.faces_spine), 3, 3))
        self.assertEqual(attr_block.shape, (len(self.faces_spine), 4))
        self.assertEqual(len(dict_list), len(self.faces_spine))
        self.assertEqual(self._extracted_face_ids(tri_coords), self.faces_spine.tolist())

    def test_zero_threshold_activates_all(self):
        """閾値0度では本番関数が全関節面を抽出すること"""
        bones = {
            "spine": _MockPoseBone((1.0, 0.0, 0.0, 0.0)),
            "shin.L": _MockPoseBone((1.0, 0.0, 0.0, 0.0)),
            "shin.R": _MockPoseBone((1.0, 0.0, 0.0, 0.0)),
        }
        arm_mod = _MockArmMod(_MockArmatureObject(bones))

        tri_coords, attr_block, dict_list = _extract_active_joint_mesh_triangles(
            self.obj,
            _MockColSettings(rot_threshold_deg=0.0),
            arm_mod=arm_mod,
            bake_res=self.bake_res,
            depsgraph=None,
        )
        self.assertIsNotNone(tri_coords)
        expected = self.bake_res.joint_face_indices.tolist()
        self.assertEqual(self._extracted_face_ids(tri_coords), expected)
        self.assertEqual(attr_block.shape, (len(expected), 4))
        self.assertEqual(len(dict_list), len(expected))


if __name__ == "__main__":
    unittest.main()
