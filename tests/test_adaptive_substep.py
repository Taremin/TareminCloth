import unittest
import bpy
import numpy as np
import sys
from pathlib import Path

# python ディレクトリを sys.path に追加
addon_dir = Path(__file__).parent.parent / "python"
if str(addon_dir) not in sys.path:
    sys.path.insert(0, str(addon_dir))

import taremin_cloth
from taremin_cloth.operators import (
    get_or_create_simulator,
    get_effective_substeps,
    clear_simulators,
    _mesh_char_len_cache,
    _effective_substeps_cache,
    _prev_coords_cache,
    _collider_prev_locs_cache,
)


class TestAdaptiveSubstep(unittest.TestCase):
    def setUp(self):
        taremin_cloth.register()
        clear_simulators()

        # テスト用メッシュプレーンを作成
        bpy.ops.mesh.primitive_plane_add(size=2.0)
        self.cloth_obj = bpy.context.active_object
        self.cloth_obj.taremin_cloth.is_cloth = True
        self.cloth_obj.taremin_cloth.enable_adaptive_substep = True
        self.cloth_obj.taremin_cloth.substeps = 20
        self.cloth_obj.taremin_cloth.min_substeps = 4

    def tearDown(self):
        clear_simulators()
        # シーン内のオブジェクトを全削除
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        taremin_cloth.unregister()

    def test_mesh_characteristic_length_cache(self):
        """メッシュの最小エッジ長と厚みから特性長が正しく計算・キャッシュされることを検証"""
        sim, coords = get_or_create_simulator(self.cloth_obj)
        self.assertIn(self.cloth_obj.name, _mesh_char_len_cache)
        char_len = _mesh_char_len_cache[self.cloth_obj.name]
        # 2mの平面 (頂点間隔は 2.0m, 厚みは 0.005m * 2 = 0.01m)
        self.assertGreater(char_len, 0.001)
        self.assertLessEqual(char_len, 2.0)

    def test_smooth_step_reduction(self):
        """静止時にサブステップ数が急落せず、最大2ステップずつ滑らかに減少してmin_substepsに収束することを検証"""
        sim, coords = get_or_create_simulator(self.cloth_obj)
        _mesh_char_len_cache[self.cloth_obj.name] = 0.05

        # 1フレーム目: 初期状態（変位計算なし）は base_steps (20)
        step1 = get_effective_substeps(self.cloth_obj, coords, 1.0 / 60.0)
        self.assertEqual(step1, 20)

        # 2フレーム目: 変位ゼロ → 目標は min_steps(4) だが、スムージングにより 20 - 2 = 18
        step2 = get_effective_substeps(self.cloth_obj, coords, 1.0 / 60.0)
        self.assertEqual(step2, 18)

        # 3フレーム目: 18 - 2 = 16
        step3 = get_effective_substeps(self.cloth_obj, coords, 1.0 / 60.0)
        self.assertEqual(step3, 16)

        # 複数フレーム進めて min_substeps (4) に収束することを確認
        for _ in range(10):
            last_step = get_effective_substeps(self.cloth_obj, coords, 1.0 / 60.0)
        self.assertEqual(last_step, 4)

    def test_immediate_step_boost_on_rapid_motion(self):
        """急激な頂点変位が発生した際、即座にサブステップが引き上げられることを検証 (CFL条件)"""
        sim, coords = get_or_create_simulator(self.cloth_obj)
        _mesh_char_len_cache[self.cloth_obj.name] = 0.02  # cfl_margin = 0.01m

        # 静止状態にして min_substeps (4) まで落とす
        for _ in range(15):
            get_effective_substeps(self.cloth_obj, coords, 1.0 / 60.0)
        self.assertEqual(_effective_substeps_cache[self.cloth_obj.name], 4)

        # 急激に頂点座標を 0.15m 移動 (15 * 0.01m = 15ステップ以上)
        moved_coords = coords.copy()
        moved_coords[0] += 0.15

        step_boost = get_effective_substeps(self.cloth_obj, moved_coords, 1.0 / 60.0)
        # 即座に引き上げられる（ヒステリシスによる制限を受けない）
        self.assertGreaterEqual(step_boost, 15)
        self.assertLessEqual(step_boost, 20)

    def test_collider_motion_boosts_substeps(self):
        """布が静止していても、コライダーが高速移動した際にサブステップ数が引き上げられることを検証"""
        sim, coords = get_or_create_simulator(self.cloth_obj)
        _mesh_char_len_cache[self.cloth_obj.name] = 0.02

        # 静止状態にして 4 まで落とす
        for _ in range(15):
            get_effective_substeps(self.cloth_obj, coords, 1.0 / 60.0)
        self.assertEqual(_effective_substeps_cache[self.cloth_obj.name], 4)

        # 球コライダーを作成
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0, 0, 0))
        col_obj = bpy.context.active_object
        col_obj.taremin_collider.is_collider = True
        col_obj.taremin_collider.collider_type = 'SPHERE'

        # 1回目の呼び出しでコライダー位置キャッシュを登録
        get_effective_substeps(self.cloth_obj, coords, 1.0 / 60.0, scene=bpy.context.scene)

        # コライダーを高速移動 (0.2m移動)
        col_obj.location.x += 0.2
        bpy.context.view_layer.update()

        # 布座標は不変だがコライダー移動によりサブステップが即時ブースト
        step_after_col_move = get_effective_substeps(
            self.cloth_obj, coords, 1.0 / 60.0, scene=bpy.context.scene
        )
        self.assertEqual(step_after_col_move, 20)


if __name__ == "__main__":
    unittest.main()
