import os
import sys
import math
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)

from taremin_cloth.engine.sdf_baker import BoneSdfBakeResult


class MockPoseBone:
    def __init__(self, rotation_quaternion=(1.0, 0.0, 0.0, 0.0), rotation_mode='QUATERNION'):
        self.rotation_quaternion = rotation_quaternion
        self.rotation_mode = rotation_mode


class MockPose:
    def __init__(self, bones_dict):
        self.bones = bones_dict


class MockArmatureObject:
    def __init__(self, bones_dict):
        self.pose = MockPose(bones_dict)


def compute_active_joint_faces(bake_res, arm_obj, rot_threshold_deg=2.0):
    """collider.py の動的アクティブ化判定ロジックの参照実装"""
    rot_threshold_rad = math.radians(rot_threshold_deg)

    if rot_threshold_deg <= 0.0 or not bake_res.joint_faces_by_pair:
        return bake_res.joint_face_indices

    active_pair_faces = []
    if arm_obj and arm_obj.pose:
        for (p_name, c_name), p_faces in bake_res.joint_faces_by_pair.items():
            pb_c = arm_obj.pose.bones.get(c_name)
            if not pb_c:
                continue
            delta_angle = 0.0
            rot_mode = pb_c.rotation_mode
            if rot_mode == 'QUATERNION':
                q = pb_c.rotation_quaternion
                delta_angle = 2.0 * math.acos(min(abs(float(q[0])), 1.0))
            elif rot_mode == 'AXIS_ANGLE':
                delta_angle = abs(float(pb_c.rotation_axis_angle[0]))
            else:
                e = pb_c.rotation_euler
                delta_angle = math.sqrt(float(e.x)**2 + float(e.y)**2 + float(e.z)**2)

            if delta_angle >= rot_threshold_rad:
                active_pair_faces.append(p_faces)

    if active_pair_faces:
        return np.unique(np.concatenate(active_pair_faces))
    return np.empty(0, dtype=np.int32)


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

    def test_static_pose_returns_zero_faces(self):
        """直立静止ポーズ（回転角0度）ではアクティブ面数が0面（SDF単体モード）になること"""
        bones = {
            "spine": MockPoseBone((1.0, 0.0, 0.0, 0.0)),
            "shin.L": MockPoseBone((1.0, 0.0, 0.0, 0.0)),
            "shin.R": MockPoseBone((1.0, 0.0, 0.0, 0.0)),
        }
        arm = MockArmatureObject(bones)

        active = compute_active_joint_faces(self.bake_res, arm, rot_threshold_deg=2.0)
        self.assertEqual(len(active), 0)

    def test_below_threshold_returns_zero_faces(self):
        """閾値（2.0度）未満の微小な回転（1.0度）ではアクティブ面数が0面になること"""
        # 1度回転: q = cos(0.5 deg), sin(0.5 deg)
        ang_rad = math.radians(1.0)
        q = (math.cos(ang_rad * 0.5), math.sin(ang_rad * 0.5), 0.0, 0.0)
        bones = {
            "spine": MockPoseBone(q),
            "shin.L": MockPoseBone((1.0, 0.0, 0.0, 0.0)),
            "shin.R": MockPoseBone((1.0, 0.0, 0.0, 0.0)),
        }
        arm = MockArmatureObject(bones)

        active = compute_active_joint_faces(self.bake_res, arm, rot_threshold_deg=2.0)
        self.assertEqual(len(active), 0)

    def test_bent_joint_selectively_activated(self):
        """背骨のみ前屈（30度）した際、腰・背骨関節面のみが正確にアクティブ化されること"""
        ang_rad = math.radians(30.0)
        q = (math.cos(ang_rad * 0.5), math.sin(ang_rad * 0.5), 0.0, 0.0)
        bones = {
            "spine": MockPoseBone(q),
            "shin.L": MockPoseBone((1.0, 0.0, 0.0, 0.0)),
            "shin.R": MockPoseBone((1.0, 0.0, 0.0, 0.0)),
        }
        arm = MockArmatureObject(bones)

        active = compute_active_joint_faces(self.bake_res, arm, rot_threshold_deg=2.0)
        # 背骨関節面のみアクティブ
        self.assertEqual(len(active), len(self.faces_spine))
        np.testing.assert_array_equal(active, self.faces_spine)

    def test_zero_threshold_activates_all(self):
        """閾値0度では常時全関節面がアクティブになること"""
        bones = {
            "spine": MockPoseBone((1.0, 0.0, 0.0, 0.0)),
            "shin.L": MockPoseBone((1.0, 0.0, 0.0, 0.0)),
            "shin.R": MockPoseBone((1.0, 0.0, 0.0, 0.0)),
        }
        arm = MockArmatureObject(bones)

        active = compute_active_joint_faces(self.bake_res, arm, rot_threshold_deg=0.0)
        self.assertEqual(len(active), len(self.bake_res.joint_face_indices))
        np.testing.assert_array_equal(active, self.bake_res.joint_face_indices)


if __name__ == "__main__":
    unittest.main()
