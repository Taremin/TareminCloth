"""
ボーンチェーン補間およびシワカーブスライド幾何モジュールの単体テスト (test_wrinkle_slide.py)
"""

import unittest
import numpy as np

from taremin_cloth.engine.wrinkle_slide import (
    BoneSegment,
    ChainEvalResult,
    BoneChain,
    slide_curves_along_chain,
)


class TestWrinkleSlide(unittest.TestCase):
    """BoneChain および slide_curves_along_chain の幾何テスト"""

    def test_continuous_bone_chain_evaluation(self):
        """2本の連続ボーンが正しくチェーン化され、位置・軸・半径が連続評価されること"""
        bones = [
            ("UpperArm", [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.05),
            ("ForeArm", [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], 0.04),
        ]
        chain = BoneChain.from_bones(bones)
        self.assertEqual(len(chain.segments), 2)
        self.assertAlmostEqual(chain.total_length, 2.0, places=5)
        self.assertFalse(chain.segments[0].is_virtual)
        self.assertFalse(chain.segments[1].is_virtual)

        # t = 0.0 (始点: UpperArm Head)
        ev0 = chain.evaluate(0.0)
        np.testing.assert_allclose(ev0.position, [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(ev0.axis, [0.0, 0.0, 1.0], atol=1e-5)
        self.assertAlmostEqual(ev0.radius, 0.05, places=5)

        # t = 0.5 (関節位置: UpperArm Tail)
        ev_joint = chain.evaluate(0.5)
        np.testing.assert_allclose(ev_joint.position, [0.0, 0.0, 1.0], atol=1e-5)
        self.assertAlmostEqual(ev_joint.radius, 0.05, places=5)

        # t = 0.75 (ForeArm 中央)
        ev_forearm = chain.evaluate(0.75)
        np.testing.assert_allclose(ev_forearm.position, [0.0, 0.0, 1.5], atol=1e-5)
        self.assertAlmostEqual(ev_forearm.radius, 0.04, places=5)

        # t = 1.0 (終点: ForeArm Tail)
        ev1 = chain.evaluate(1.0)
        np.testing.assert_allclose(ev1.position, [0.0, 0.0, 2.0], atol=1e-5)
        self.assertAlmostEqual(ev1.radius, 0.04, places=5)

    def test_tapered_bone_chain_evaluation(self):
        """テーパーボーン (head_radius, tail_radius) のチェーンで半径が滑らかに線形補間されること"""
        bones = [
            ("UpperArm", [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], (0.06, 0.04)),
            ("ForeArm", [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], (0.04, 0.02)),
        ]
        chain = BoneChain.from_bones(bones)
        # t=0.0 -> 0.06
        self.assertAlmostEqual(chain.evaluate(0.0).radius, 0.06, places=5)
        # t=0.25 (UpperArm 中央) -> 0.05
        self.assertAlmostEqual(chain.evaluate(0.25).radius, 0.05, places=5)
        # t=0.5 (関節部) -> 0.04
        self.assertAlmostEqual(chain.evaluate(0.5).radius, 0.04, places=5)
        # t=0.75 (ForeArm 中央) -> 0.03
        self.assertAlmostEqual(chain.evaluate(0.75).radius, 0.03, places=5)
        # t=1.0 (終点) -> 0.02
        self.assertAlmostEqual(chain.evaluate(1.0).radius, 0.02, places=5)

    def test_discontinuous_bone_chain_virtual_link(self):
        """非連続ボーン（親Tail〜子Headのギャップ）に仮想リンクが自動補間されること"""
        bones = [
            ("BoneA", [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.06),
            # ギャップ: (0, 0, 1.0) -> (0, 0, 1.5) の 0.5m
            ("BoneB", [0.0, 0.0, 1.5], [0.0, 0.0, 2.5], 0.04),
        ]
        chain = BoneChain.from_bones(bones, gap_threshold=0.001)

        # セグメントは 3つ（BoneA, 仮想リンク, BoneB）になるはず
        self.assertEqual(len(chain.segments), 3)
        self.assertAlmostEqual(chain.total_length, 2.5, places=5)

        v_link = chain.segments[1]
        self.assertTrue(v_link.is_virtual)
        self.assertEqual(v_link.name, "Link_BoneA_to_BoneB")
        np.testing.assert_allclose(v_link.head, [0.0, 0.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(v_link.tail, [0.0, 0.0, 1.5], atol=1e-5)
        self.assertAlmostEqual(v_link.length, 0.5, places=5)

        # 仮想リンクの真ん中 (累積距離 1.25m -> t = 1.25 / 2.5 = 0.5) を評価
        ev_mid = chain.evaluate(0.5)
        self.assertTrue(ev_mid.is_virtual)
        np.testing.assert_allclose(ev_mid.position, [0.0, 0.0, 1.25], atol=1e-5)
        # 半径は 0.06 から 0.04 の線形中間 -> 0.05
        self.assertAlmostEqual(ev_mid.radius, 0.05, places=5)

    def test_bent_joint_basis_orthonormality_and_continuity(self):
        """90度曲がった関節で正規直交基底が維持され、接線がブレンドされること"""
        bones = [
            ("Root", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.05),  # +X方向 (長1m)
            ("Limb", [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 0.05),  # +Y方向 (長1m)
        ]
        chain = BoneChain.from_bones(bones)
        self.assertAlmostEqual(chain.total_length, 2.0, places=5)

        # チェーン全体のサンプリング点ですべて正規直交であることを確認
        for t in np.linspace(0.0, 1.0, 21):
            ev = chain.evaluate(t)
            # 各ベクトルのノルムが 1.0
            self.assertAlmostEqual(float(np.linalg.norm(ev.axis)), 1.0, places=5)
            self.assertAlmostEqual(float(np.linalg.norm(ev.normal)), 1.0, places=5)
            self.assertAlmostEqual(float(np.linalg.norm(ev.binormal)), 1.0, places=5)
            # ベクトル同士が直交 (内積が 0)
            self.assertAlmostEqual(float(np.dot(ev.axis, ev.normal)), 0.0, places=5)
            self.assertAlmostEqual(float(np.dot(ev.axis, ev.binormal)), 0.0, places=5)
            self.assertAlmostEqual(float(np.dot(ev.normal, ev.binormal)), 0.0, places=5)

    def test_slide_curves_along_chain_straight(self):
        """直進ボーンに沿って円カーブをスライドさせたときの幾何位置と拡縮"""
        bones = [
            ("Spine", [0.0, 0.0, 0.0], [0.0, 0.0, 2.0], 0.05),
        ]
        chain = BoneChain.from_bones(bones)

        # t=0.2 (z = 0.4) の位置に半径 0.05 の円カーブを作成
        num_pts = 16
        angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        src_circle = np.zeros((num_pts, 3), dtype=np.float32)
        src_circle[:, 0] = 0.05 * np.cos(angles)
        src_circle[:, 1] = 0.05 * np.sin(angles)
        src_circle[:, 2] = 0.4

        # t=0.2 から t=0.7 (z = 1.4) へスライド (scale_radius = 1.5)
        slid_curves = slide_curves_along_chain(
            [src_circle],
            chain,
            source_t=0.2,
            target_t=0.7,
            scale_radius=1.5,
        )

        self.assertEqual(len(slid_curves), 1)
        res = slid_curves[0]
        # Z座標は 1.4 に移動しているはず
        np.testing.assert_allclose(res[:, 2], 1.4, atol=1e-5)
        # 半径は 0.05 * 1.5 = 0.075 にスケールされているはず
        r_dist = np.linalg.norm(res[:, :2], axis=1)
        np.testing.assert_allclose(r_dist, 0.075, atol=1e-5)

    def test_slide_curves_across_bent_joint(self):
        """L字に曲がったボーンをまたいでスライドしたとき、円の向き（法線）がボーン軸に追従すること"""
        bones = [
            ("BoneX", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.05),  # 軸は +X
            ("BoneY", [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 0.05),  # 軸は +Y
        ]
        chain = BoneChain.from_bones(bones)

        # BoneX 上 (t=0.25, x=0.5) に、YZ平面上の円カーブを配置
        num_pts = 8
        angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        ev_src = chain.evaluate(0.25)
        src_circle = np.zeros((num_pts, 3), dtype=np.float32)
        for i, a in enumerate(angles):
            src_circle[i] = ev_src.position + ev_src.normal * (0.05 * np.cos(a)) + ev_src.binormal * (0.05 * np.sin(a))

        # BoneY 上 (t=0.75, y=0.5, x=1.0) へスライド
        slid_curves = slide_curves_along_chain(
            [src_circle],
            chain,
            source_t=0.25,
            target_t=0.75,
            scale_radius=1.0,
        )

        res = slid_curves[0]
        ev_tgt = chain.evaluate(0.75)
        # スライド後の中心点
        center = np.mean(res, axis=0)
        np.testing.assert_allclose(center, ev_tgt.position, atol=1e-5)

        # すべての頂点が目標位置 ev_tgt.position から半径 0.05 の距離にあること
        diffs = res - ev_tgt.position
        radii = np.linalg.norm(diffs, axis=1)
        np.testing.assert_allclose(radii, 0.05, atol=1e-5)

        # 頂点群が target_axis (+Y方向) と直交していること (内積がほぼ0)
        axis_dots = np.dot(diffs, ev_tgt.axis)
        np.testing.assert_allclose(axis_dots, 0.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
