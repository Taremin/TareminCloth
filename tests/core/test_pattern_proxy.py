# -*- coding: utf-8 -*-
"""3D配置プロキシ & クイック配置プリセットの単体テスト."""

import math
import os
import sys
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from taremin_cloth.pattern.placement import (
    QUICK_PLACEMENT_PRESETS,
    estimate_body_reference_pure,
    get_quick_placement_transform,
)


class TestPatternProxy(unittest.TestCase):
    def test_quick_placement_transforms(self):
        """主要なクイック配置プリセットのオフセットと回転角が正しいこと."""
        # 1. 前身頃 (Front / Chest): build_garment_mesh_geometry は XZ 平面直立 [du, 0, dv] を基準とするため euler_x = 0.0
        offset_f, euler_f = get_quick_placement_transform('CHEST_FRONT')
        self.assertAlmostEqual(offset_f[0], 0.0)
        self.assertLess(offset_f[1], 0.0)  # Yマイナス（体の前）
        self.assertGreater(offset_f[2], 0.9)  # 胸の高さ
        self.assertAlmostEqual(euler_f[0], 0.0, places=4)

        # 2. 後身頃 (Back): Yプラス側・Z軸180度反転で直立
        offset_b, euler_b = get_quick_placement_transform('BACK')
        self.assertAlmostEqual(offset_b[0], 0.0)
        self.assertGreater(offset_b[1], 0.0)  # Yプラス（体の後ろ）
        self.assertAlmostEqual(euler_b[0], 0.0, places=4)
        self.assertAlmostEqual(abs(euler_b[2]), math.pi, places=4)  # 180度反転

        # 3. 左袖 (Left Arm)
        offset_l, euler_l = get_quick_placement_transform('ARM_LEFT')
        self.assertLess(offset_l[0], 0.0)  # Xマイナス（左腕側）
        self.assertAlmostEqual(euler_l[0], 0.0, places=4)
        self.assertAlmostEqual(euler_l[2], -math.pi / 2, places=4)

        # 4. 右袖 (Right Arm)
        offset_r, euler_r = get_quick_placement_transform('ARM_RIGHT')
        self.assertGreater(offset_r[0], 0.0)  # Xプラス（右腕側）
        self.assertAlmostEqual(euler_r[0], 0.0, places=4)
        self.assertAlmostEqual(euler_r[2], math.pi / 2, places=4)

        # 5. 身長スケール参照の純粋計算が反映されること
        ref_short = estimate_body_reference_pure(body_height=1.2)
        offset_short, _ = get_quick_placement_transform('CHEST_FRONT', ref=ref_short)
        self.assertLess(offset_short[2], offset_f[2])

    def test_preset_items_defined(self):
        """プリセット定義リストが空でなく主要パーツを網羅していること."""
        keys = [item[0] for item in QUICK_PLACEMENT_PRESETS]
        self.assertIn('CHEST_FRONT', keys)
        self.assertIn('BACK', keys)
        self.assertIn('ARM_LEFT', keys)
        self.assertIn('ARM_RIGHT', keys)
        self.assertIn('COLLAR', keys)
        self.assertIn('WAIST_SKIRT', keys)

    def test_quick_placement_builds_upright_in_3d(self):
        """CHEST_FRONT / BACK / ARM_* を 3D ビルドした際、2D 高さ方向がワールド Z 軸に直立展開されること."""
        from taremin_cloth.pattern.build_types import build_garment_mesh_geometry
        from taremin_cloth.pattern.document import Document, Placement

        for preset_id in ('CHEST_FRONT', 'BACK', 'ARM_LEFT', 'ARM_RIGHT'):
            doc = Document()
            # 横 0.2m x 縦 0.4m の縦長矩形パネル
            piece = doc.add_rectangle_piece((0.0, 0.0), (0.2, 0.4), name=preset_id)
            piece.target_edge_length = 0.05
            offset_p, euler_p = get_quick_placement_transform(preset_id)
            doc.placements[piece.id] = Placement(
                piece_id=piece.id,
                offset_3d=offset_p,
                euler_3d=euler_p,
                preset=preset_id,
            )
            geom = build_garment_mesh_geometry(doc)
            self.assertGreater(len(geom.vertices_3d), 0)
            xs = [p[0] for p in geom.vertices_3d]
            ys = [p[1] for p in geom.vertices_3d]
            zs = [p[2] for p in geom.vertices_3d]
            span_z = max(zs) - min(zs)
            span_h = max(max(xs) - min(xs), max(ys) - min(ys))
            # 直立しているため Z 方向の広がりが水平幅より大きく約 0.4m ある
            self.assertGreater(span_z, 0.35, f"{preset_id} が水平に倒れています: span_z={span_z}, span_h={span_h}")
            self.assertLess(span_h, 0.25, f"{preset_id} の水平幅が想定外です: span_h={span_h}")


if __name__ == '__main__':
    unittest.main()
