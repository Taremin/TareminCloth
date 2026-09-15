"""
コライダー自動判別モジュール (collider_detect.py) の単体テスト
Blender非依存のモックオブジェクトを用いて、形状や構造に基づく自動推定ロジックを検証する。
"""

import os
import sys
import unittest
from types import SimpleNamespace

# プロジェクトルートパスの設定
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)

from taremin_cloth.utils.collider_detect import (
    detect_collider_type,
    PURPOSE_TO_COLLIDER_TYPE,
)


class MockModifier:
    def __init__(self, mod_type: str, armature_obj=None):
        self.type = mod_type
        self.object = armature_obj


class MockMesh:
    def __init__(self, num_polygons: int):
        self.polygons = [SimpleNamespace() for _ in range(num_polygons)]


class MockObject:
    def __init__(
        self,
        name: str = "TestObj",
        obj_type: str = 'MESH',
        num_polygons: int = 100,
        modifiers=None,
        vertex_groups=None,
        bound_box=None,
    ):
        self.name = name
        self.type = obj_type
        self.data = MockMesh(num_polygons) if obj_type == 'MESH' else None
        self.modifiers = modifiers or []
        self.vertex_groups = vertex_groups or []
        # デフォルトは 1m x 1m x 1m の立方体BBox
        self.bound_box = bound_box or [
            (-0.5, -0.5, -0.5),
            (-0.5, -0.5, 0.5),
            (-0.5, 0.5, 0.5),
            (-0.5, 0.5, -0.5),
            (0.5, -0.5, -0.5),
            (0.5, -0.5, 0.5),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, -0.5),
        ]


class TestColliderDetect(unittest.TestCase):
    """コライダー自動判別機能の検証"""

    def test_purpose_mapping(self):
        """目的別マッピング辞書が正しく定義されているかを検証"""
        self.assertIsNone(PURPOSE_TO_COLLIDER_TYPE['AUTO'])
        self.assertEqual(PURPOSE_TO_COLLIDER_TYPE['CHARACTER'], 'BONE_SDF')
        self.assertEqual(PURPOSE_TO_COLLIDER_TYPE['MANNEQUIN'], 'MESH_SDF')
        self.assertEqual(PURPOSE_TO_COLLIDER_TYPE['FLOOR'], 'PLANE')
        self.assertEqual(PURPOSE_TO_COLLIDER_TYPE['SPHERE'], 'SPHERE')
        self.assertIsNone(PURPOSE_TO_COLLIDER_TYPE['CUSTOM'])

    def test_detect_bone_sdf_for_rigged_character(self):
        """Armatureモディファイアと頂点グループを持つ素体キャラクタは BONE_SDF と判定される"""
        armature_mock = SimpleNamespace(type='ARMATURE')
        obj = MockObject(
            name="CharacterBody",
            num_polygons=5000,
            modifiers=[MockModifier('ARMATURE', armature_obj=armature_mock)],
            vertex_groups=[SimpleNamespace(name="DEF-Spine"), SimpleNamespace(name="DEF-Arm")],
        )
        detected = detect_collider_type(obj)
        self.assertEqual(detected, 'BONE_SDF')

    def test_detect_plane_for_floor_mesh(self):
        """極めて薄い（Z方向の厚みが極小）メッシュは PLANE と判定される"""
        # 10m x 10m x 0.001m の床メッシュ
        floor_bbox = [
            (-5.0, -5.0, 0.0),
            (-5.0, -5.0, 0.001),
            (-5.0, 5.0, 0.001),
            (-5.0, 5.0, 0.0),
            (5.0, -5.0, 0.0),
            (5.0, -5.0, 0.001),
            (5.0, 5.0, 0.001),
            (5.0, 5.0, 0.0),
        ]
        obj = MockObject(name="Floor", num_polygons=10, bound_box=floor_bbox)
        detected = detect_collider_type(obj)
        self.assertEqual(detected, 'PLANE')

    def test_detect_sphere(self):
        """3軸サイズがほぼ等しく、名前または形状から球体とみなせるメッシュは SPHERE と判定される"""
        sphere_bbox = [
            (-1.0, -1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (1.0, 1.0, -1.0),
        ]
        # 名前に sphere を含む場合
        obj1 = MockObject(name="Sphere_Head", num_polygons=200, bound_box=sphere_bbox)
        self.assertEqual(detect_collider_type(obj1), 'SPHERE')

        # 名前に ball を含む場合
        obj2 = MockObject(name="BallCollision", num_polygons=200, bound_box=sphere_bbox)
        self.assertEqual(detect_collider_type(obj2), 'SPHERE')

    def test_detect_mesh_sdf_for_dense_mannequin(self):
        """リグなしで面数が300以上の高密度剛体メッシュは MESH_SDF と判定される"""
        # 直方体マネキン (0.4m x 0.2m x 1.7m)
        mannequin_bbox = [
            (-0.2, -0.1, 0.0),
            (-0.2, -0.1, 1.7),
            (-0.2, 0.1, 1.7),
            (-0.2, 0.1, 0.0),
            (0.2, -0.1, 0.0),
            (0.2, -0.1, 1.7),
            (0.2, 0.1, 1.7),
            (0.2, 0.1, 0.0),
        ]
        obj = MockObject(name="Mannequin_Torso", num_polygons=1200, bound_box=mannequin_bbox)
        detected = detect_collider_type(obj)
        self.assertEqual(detected, 'MESH_SDF')

    def test_detect_mesh_for_low_poly_obstacle(self):
        """リグなしで面数が300未満の低ポリゴン障害物は MESH と判定される"""
        low_poly_bbox = [
            (-0.5, -0.2, -0.5),
            (-0.5, -0.2, 0.5),
            (-0.5, 0.2, 0.5),
            (-0.5, 0.2, -0.5),
            (0.5, -0.2, -0.5),
            (0.5, -0.2, 0.5),
            (0.5, 0.2, 0.5),
            (0.5, 0.2, -0.5),
        ]
        obj = MockObject(name="CubeObstacle", num_polygons=12, bound_box=low_poly_bbox)
        detected = detect_collider_type(obj)
        self.assertEqual(detected, 'MESH')

    def test_detect_none_for_invalid_object(self):
        """None や非メッシュオブジェクトの場合はデフォルトの MESH を返す"""
        self.assertEqual(detect_collider_type(None), 'MESH')
        non_mesh = SimpleNamespace(type='EMPTY', bound_box=None)
        self.assertEqual(detect_collider_type(non_mesh), 'MESH')


if __name__ == '__main__':
    unittest.main()
