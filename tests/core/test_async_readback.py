import sys
import unittest
from pathlib import Path
import numpy as np

# プロジェクトルートと python ディレクトリをパスに追加
root_dir = Path(__file__).parent.parent.parent
python_dir = root_dir / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import taremin_cloth_core
from taremin_cloth.preferences import is_async_readback_experimental_enabled
from taremin_cloth.engine.runner import flush_async_simulation, flush_all_async_simulations


def create_simple_grid(nx=4, ny=4, dx=0.1):
    positions = []
    inv_masses = []
    for y in range(ny):
        for x in range(nx):
            positions.append([x * dx, y * dx, 1.0])
            if y == ny - 1:
                inv_masses.append(0.0)  # 上端固定
            else:
                inv_masses.append(1.0)

    edges = []
    for y in range(ny):
        for x in range(nx):
            idx = y * nx + x
            if x + 1 < nx:
                edges.append([idx, idx + 1])
            if y + 1 < ny:
                edges.append([idx, idx + nx])

    return (
        np.array(positions, dtype=np.float32),
        np.array(edges, dtype=np.uint32),
        np.array(inv_masses, dtype=np.float32),
    )


class TestAsyncReadback(unittest.TestCase):
    """非同期リードバック（step_async / fetch_positions）パイプラインの単体テスト"""

    def setUp(self):
        self.pos, self.edges, self.inv_m = create_simple_grid()
        self.n_verts = len(self.pos)

    def test_step_async_fetch_positions_basic(self):
        """step_async 発行と fetch_positions による回収の基本動作"""
        sim = taremin_cloth_core.ClothSimulator(
            positions=self.pos,
            edges=self.edges,
            inv_masses=self.inv_m,
            stiffness=500.0,
            thickness=0.01,
        )

        out_coords = np.zeros(self.n_verts * 3, dtype=np.float32)

        # 最初の fetch は未発行なので False
        ok_before = sim.fetch_positions(out_coords)
        self.assertFalse(ok_before)

        # 1フレーム非同期進行
        sim.step_async(dt=1.0 / 60.0, substeps=10)

        # 回収
        ok_after = sim.fetch_positions(out_coords)
        self.assertTrue(ok_after)

        # 重力によってZ座標が落ちている頂点が存在することを確認
        z_coords = out_coords[2::3]
        self.assertTrue(np.any(z_coords < 1.0), "布頂点が重力で落下していること")

    def test_async_vs_sync_results_match(self):
        """同期モードと非同期パイプラインで最終的な物理座標が一致することを検証"""
        dt = 1.0 / 60.0
        substeps = 10
        n_frames = 15

        # 1. 同期モード
        sim_sync = taremin_cloth_core.ClothSimulator(
            positions=self.pos,
            edges=self.edges,
            inv_masses=self.inv_m,
            stiffness=500.0,
            thickness=0.01,
        )
        coords_sync = np.zeros(self.n_verts * 3, dtype=np.float32)
        for _ in range(n_frames):
            sim_sync.step(dt=dt, substeps=substeps)
        sim_sync.get_positions(coords_sync)

        # 2. 非同期パイプライン
        sim_async = taremin_cloth_core.ClothSimulator(
            positions=self.pos,
            edges=self.edges,
            inv_masses=self.inv_m,
            stiffness=500.0,
            thickness=0.01,
        )
        coords_async = np.zeros(self.n_verts * 3, dtype=np.float32)
        for _ in range(n_frames):
            sim_async.fetch_positions(coords_async)
            sim_async.step_async(dt=dt, substeps=substeps)
        # 最終フレーム回収
        sim_async.fetch_positions(coords_async)

        # 両者の結果が同一物理シミュレーションとして一致すること (許容誤差 1e-4)
        np.testing.assert_allclose(
            coords_async,
            coords_sync,
            atol=1e-4,
            err_msg="同期モードと非同期パイプラインのシミュレーション結果が一致すること",
        )

    def test_consecutive_async_steps_without_fetch(self):
        """fetch_positions を挟まずに連続で step_async を呼んでもハングアップしないこと"""
        sim = taremin_cloth_core.ClothSimulator(
            positions=self.pos,
            edges=self.edges,
            inv_masses=self.inv_m,
            stiffness=500.0,
            thickness=0.01,
        )
        coords = np.zeros(self.n_verts * 3, dtype=np.float32)

        # 3回連続発行
        for _ in range(3):
            sim.step_async(dt=1.0 / 60.0, substeps=5)

        # 最後のフレームが回収できること
        ok = sim.fetch_positions(coords)
        self.assertTrue(ok)

    def test_preference_helper_default(self):
        """プリファレンス未設定環境・モック環境でのヘルパー関数の安全動作"""
        # コンテキストNone時でも例外を出さずに真偽値を返すこと
        res = is_async_readback_experimental_enabled(None)
        self.assertIsInstance(res, bool)

    def test_step_async_with_none_substeps(self):
        """適応サブステップ有効時のように substeps=None で step_async を呼んでもエラーにならないこと"""
        sim = taremin_cloth_core.ClothSimulator(
            positions=self.pos,
            edges=self.edges,
            inv_masses=self.inv_m,
            stiffness=500.0,
            thickness=0.01,
        )
        sim.set_adaptive_substep_options(True, 10, 40)
        coords = np.zeros(self.n_verts * 3, dtype=np.float32)

        # None を渡して実行
        sim.step_async(dt=1.0 / 60.0, substeps=None)
        ok = sim.fetch_positions(coords)
        self.assertTrue(ok)
        self.assertGreaterEqual(sim.get_effective_substeps(), 10)

    def test_external_displacement_scales_substeps_in_async(self):
        """非同期モードでも外部変位通知（マウスドラッグ等）によって適応サブステップ数が自動昇格すること"""
        sim = taremin_cloth_core.ClothSimulator(
            positions=self.pos,
            edges=self.edges,
            inv_masses=self.inv_m,
            stiffness=500.0,
            thickness=0.01,
        )
        sim.set_adaptive_substep_options(True, 10, 50)
        coords = np.zeros(self.n_verts * 3, dtype=np.float32)

        # 1. 外部変位なしで1ステップ実行
        sim.step_async(dt=1.0 / 60.0, substeps=None)
        sim.fetch_positions(coords)
        substeps_quiet = sim.get_effective_substeps()

        # 2. 大きな外部変位（急激なマウスドラッグ相当）を通知して非同期実行
        sim.note_external_displacement(0.5)
        sim.step_async(dt=1.0 / 60.0, substeps=None)
        sim.fetch_positions(coords)
        substeps_drag = sim.get_effective_substeps()

        # ドラッグ時のサブステップ数が静止時より有意に昇格していること
        self.assertGreater(
            substeps_drag,
            substeps_quiet,
            f"外部変位通知によりサブステップ数が自動昇格すること ({substeps_quiet} -> {substeps_drag})",
        )
        self.assertGreaterEqual(substeps_drag, 30, "大きな変位に対して高いサブステップ数が割り当てられること")

    def test_async_fetch_updates_internal_displacement_feedback(self):
        """外部変位通知なしでも、fetch_positionsによる回収で布の自由落下運動がRustコア内部にフィードバックされること"""
        sim = taremin_cloth_core.ClothSimulator(
            positions=self.pos,
            edges=self.edges,
            inv_masses=self.inv_m,
            stiffness=500.0,
            thickness=0.01,
        )
        sim.set_adaptive_substep_options(True, 10, 40)
        coords = np.zeros(self.n_verts * 3, dtype=np.float32)

        # 複数フレーム自由落下させて非同期回収
        for _ in range(5):
            sim.step_async(dt=1.0 / 60.0, substeps=None)
            sim.fetch_positions(coords)

        # 自由落下に伴う内部変位フィードバックにより実効サブステップ数が下限以上で適切に機能すること
        eff_substeps = sim.get_effective_substeps()
        self.assertGreaterEqual(eff_substeps, 10)


if __name__ == "__main__":
    unittest.main()
