"""コードレビュー バッチ 1/2 の回帰テスト集 (Blender 不要スタンドアロン CI 対応)."""

import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np

_PROJECT_ROOT = os.path.abspath(str(Path(__file__).resolve().parents[2]))
_PYTHON_DIR = os.path.join(_PROJECT_ROOT, "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import tests  # noqa: F401
import taremin_cloth_core
from taremin_cloth.engine import cache, sdf_baker
from taremin_cloth.utils import drawing
from tools import blender_manager, release


class _Vec3:
    """Blender Vector 互換の最小 3D ベクトルモック."""

    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __sub__(self, other):
        return _Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    @property
    def length(self):
        return float((self.x * self.x + self.y * self.y + self.z * self.z) ** 0.5)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z


class _IdentityMatrix:
    """`matrix @ vec` と `np.asarray(matrix)` の両方をサポートする単位行列モック."""

    def __matmul__(self, other):
        return other

    def __array__(self, dtype=None, copy=None):
        arr = np.eye(4, dtype=np.float32)
        return arr.astype(dtype) if dtype is not None else arr


class _VertexCollection(list):
    """`len()` / 添字参照に加えて `foreach_get('co', flat_out)` を提供する頂点リストモック."""

    def foreach_get(self, attr, flat_out):
        if attr != "co":
            raise KeyError(attr)
        coords = np.array([[v.co.x, v.co.y, v.co.z] for v in self], dtype=np.float32).ravel()
        flat_out[:] = coords


def _make_mock_cloth_object(*, num_verts: int, pin_count: int, pin_interactive_only: bool, elastic_interactive_only: bool, has_elastic_group: bool = False):
    """3D オーバーレイ描画テスト用の布オブジェクトモックを作成する."""
    vg_pin = MagicMock()
    vg_pin.index = 0

    verts = []
    for i in range(num_verts):
        v = MagicMock()
        v.index = i
        v.co = _Vec3(float(i) * 0.01, 0.0, 0.0)
        if i < pin_count:
            g = MagicMock()
            g.group = 0
            g.weight = 1.0
            v.groups = [g]
        else:
            v.groups = []
        verts.append(v)

    mesh = MagicMock()
    mesh.vertices = _VertexCollection(verts)
    mesh.edges = []

    settings = MagicMock()
    settings.is_cloth = True
    settings.pin_vertex_group = "Pin"
    settings.pin_color = (1.0, 0.45, 0.0, 0.9)
    settings.overlay_depth_test = False
    settings.show_pin_overlay = True
    settings.pin_overlay_interactive_only = pin_interactive_only
    settings.show_elastic_overlay = True
    settings.elastic_overlay_interactive_only = elastic_interactive_only
    settings.use_wrinkle_field = False
    settings.enable_sewing = False
    if has_elastic_group:
        grp = MagicMock()
        grp.enabled = True
        grp.get_edge_indices = MagicMock(return_value=[0])
        grp.color = (1.0, 0.2, 0.2, 0.8)
        grp.scale = 1.0
        settings.elastic_groups = [grp]
    else:
        settings.elastic_groups = []

    obj = MagicMock()
    obj.name = "MockCloth"
    obj.type = "MESH"
    obj.data = mesh
    obj.taremin_cloth = settings
    obj.visible_get = MagicMock(return_value=True)
    obj.hide_viewport = False
    obj.matrix_world = _IdentityMatrix()
    obj.vertex_groups = {"Pin": vg_pin}
    return obj


class TestDrawingOverlayRegressions(unittest.TestCase):
    """drawing.py の 33+ ピン UnboundLocalError および interactive_only 分岐テスト."""

    def setUp(self):
        self._orig_interactive = drawing._interactive_active
        drawing._pin_indices_cache.clear()
        drawing._sewing_edge_indices_cache.clear()

    def tearDown(self):
        drawing._interactive_active = self._orig_interactive
        drawing._pin_indices_cache.clear()
        drawing._sewing_edge_indices_cache.clear()

    def test_should_draw_pin_and_elastic_overlay_predicates(self):
        """should_draw_pin_overlay / should_draw_elastic_overlay が interactive_only を正しく判定すること."""
        settings = MagicMock()
        settings.show_pin_overlay = True
        settings.pin_overlay_interactive_only = False
        settings.show_elastic_overlay = True
        settings.elastic_overlay_interactive_only = False

        drawing._interactive_active = False
        self.assertTrue(drawing.should_draw_pin_overlay(settings))
        self.assertTrue(drawing.should_draw_elastic_overlay(settings))

        settings.pin_overlay_interactive_only = True
        settings.elastic_overlay_interactive_only = True
        self.assertFalse(drawing.should_draw_pin_overlay(settings))
        self.assertFalse(drawing.should_draw_elastic_overlay(settings))

        drawing._interactive_active = True
        self.assertTrue(drawing.should_draw_pin_overlay(settings))
        self.assertTrue(drawing.should_draw_elastic_overlay(settings))

        settings.show_elastic_overlay = False
        self.assertTrue(drawing.should_draw_pin_overlay(settings))
        self.assertFalse(drawing.should_draw_elastic_overlay(settings))

    def test_draw_callback_3d_over_32_pins_and_non_interactive_visibility(self):
        """40 頂点ピン描画時に UnboundLocalError が起きず、非インタラクティブ常時表示分岐も動作すること."""
        obj = _make_mock_cloth_object(
            num_verts=48,
            pin_count=40,
            pin_interactive_only=False,
            elastic_interactive_only=False,
            has_elastic_group=False,
        )
        point_shader = MagicMock()
        batch_calls = []

        def fake_batch(shader, mode, data):
            batch_calls.append((shader, mode, data))
            return MagicMock()

        scene = MagicMock()
        scene.objects = [obj]

        with (
            patch.object(drawing, "bpy") as mock_bpy,
            patch.object(drawing, "gpu"),
            patch.object(drawing, "get_point_shader", return_value=point_shader),
            patch.object(drawing, "get_line_shader", return_value=None),
            patch.object(drawing, "get_smooth_color_line_shader", return_value=None),
            patch.object(drawing, "batch_for_shader", side_effect=fake_batch),
            patch.dict(cache._simulators, {}, clear=True),
        ):
            mock_bpy.context.scene = scene
            mock_bpy.context.region_data = None
            # 非インタラクティブ + interactive_only=False: 40 ピンが描画され UnboundLocalError が出ない
            drawing._interactive_active = False
            drawing.draw_callback_3d()
            self.assertEqual(len(batch_calls), 1)
            self.assertEqual(batch_calls[0][1], "POINTS")
            self.assertEqual(len(batch_calls[0][2]["pos"]), 40)

            # 非インタラクティブ + elastic_overlay_interactive_only=False（ピンは interactive_only=True）: 対象抽出を通り描画が試行されること
            batch_calls.clear()
            obj.taremin_cloth.pin_overlay_interactive_only = True
            e0 = MagicMock()
            e0.vertices = (0, 1)
            obj.data.edges = [e0]
            vg_el = MagicMock()
            vg_el.index = 1
            obj.vertex_groups["Elastic"] = vg_el
            for v in obj.data.vertices[:2]:
                ge = MagicMock()
                ge.group = 1
                ge.weight = 1.0
                v.groups.append(ge)
            grp = MagicMock()
            grp.enabled = True
            grp.get_edge_indices = MagicMock(return_value=[0])
            grp.color = (1.0, 0.2, 0.2, 0.8)
            grp.scale = 1.0
            obj.taremin_cloth.elastic_groups = [grp]
            obj.taremin_cloth.elastic_overlay_interactive_only = False
            with patch.object(drawing, "get_smooth_color_line_shader", return_value=MagicMock()):
                drawing.draw_callback_3d()
            self.assertGreaterEqual(len(batch_calls), 1)
            self.assertEqual(batch_calls[0][1], "LINES")

            # 非インタラクティブ + 両方 interactive_only=True: 早期スキップされ描画されない
            batch_calls.clear()
            obj.taremin_cloth.elastic_overlay_interactive_only = True
            drawing.draw_callback_3d()
            self.assertEqual(len(batch_calls), 0)

            # インタラクティブ実行中は 40 ピンが再び描画される（このケースも UnboundLocalError なし）
            batch_calls.clear()
            obj.taremin_cloth.pin_overlay_interactive_only = False
            drawing._interactive_active = True
            with patch.dict(cache._simulators, {obj.name: MagicMock()}, clear=True):
                drawing.draw_callback_3d()
                self.assertEqual(len(batch_calls), 1)
                self.assertEqual(batch_calls[0][1], "POINTS")
                self.assertEqual(len(batch_calls[0][2]["pos"]), 40)


class TestReleaseVersionConsistencyRegressions(unittest.TestCase):
    """tools/release.py のプレリリースバージョン整合判定回帰テスト."""

    def _create_mock_repo(self, root: Path, *, init_ver: tuple[int, int, int], text_ver: str, mismatched_cargo: str | None = None) -> None:
        (root / "__init__.py").write_text(
            f'bl_info = {{\n    "name": "Taremin Cloth",\n    "version": {init_ver},\n}}\n',
            encoding="utf-8",
        )
        pkg_init = root / "python" / "taremin_cloth" / "__init__.py"
        pkg_init.parent.mkdir(parents=True, exist_ok=True)
        pkg_init.write_text(
            f'bl_info = {{\n    "name": "Taremin Cloth",\n    "version": {init_ver},\n}}\n',
            encoding="utf-8",
        )
        (root / "pyproject.toml").write_text(
            f'[project]\nname = "taremin_cloth_core"\nversion = "{text_ver}"\n',
            encoding="utf-8",
        )
        for idx, crate in enumerate(("cloth_py", "cloth_core", "cloth_gui")):
            cargo = root / "crates" / crate / "Cargo.toml"
            cargo.parent.mkdir(parents=True, exist_ok=True)
            v_str = mismatched_cargo if (mismatched_cargo is not None and idx == 1) else text_ver
            cargo.write_text(
                f'[package]\nname = "{crate}"\nversion = "{v_str}"\n',
                encoding="utf-8",
            )

    def test_prerelease_version_consistency_match_and_mismatch(self):
        """0.2.0-rc.1 と __init__.py (0, 2, 0) で一致し、サフィックス不一致を検知すること."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_mock_repo(root, init_ver=(0, 2, 0), text_ver="0.2.0-rc.1")
            ok, primary, versions = release.check_version_consistency(root)
            self.assertTrue(ok, f"Versions should be consistent: {versions}")
            self.assertEqual(primary, release.SemVer(0, 2, 0, "rc.1"))

            self._create_mock_repo(
                root,
                init_ver=(0, 2, 0),
                text_ver="0.2.0-rc.1",
                mismatched_cargo="0.2.0-rc.2",
            )
            ok_bad, _primary_bad, _versions_bad = release.check_version_consistency(root)
            self.assertFalse(ok_bad)

            self._create_mock_repo(root, init_ver=(0, 1, 0), text_ver="0.2.0-rc.1")
            ok_bad_init, _p2, _v2 = release.check_version_consistency(root)
            self.assertFalse(ok_bad_init)


class TestSdfBakerCacheSafetyRegressions(unittest.TestCase):
    """sdf_baker.py の allow_pickle=False・アトミック保存・破損自動削除テスト."""

    def _make_valid_result(self, *, bone_names: list[str]) -> sdf_baker.BoneSdfBakeResult:
        # 2x2x2=8 ボクセルの非均一・非 NaN な float16 2チャンネルデータ (32 bytes)
        vals = np.arange(1, 17, dtype=np.float16)
        tex_bytes = vals.tobytes()
        n_bones = max(1, len(bone_names))
        return sdf_baker.BoneSdfBakeResult(
            texture_bytes=tex_bytes,
            width=2,
            height=2,
            depth=2,
            bone_infos=np.zeros((n_bones, 20), dtype=np.float32),
            bone_names=bone_names,
            bind_matrices=np.repeat(np.eye(4, dtype=np.float32)[None, :, :], n_bones, axis=0),
        )

    def test_roundtrip_empty_bones_and_corrupted_cleanup(self):
        """allow_pickle=False での保存/読込・空 bone_names 対応・破損/pickle 汚染ファイル自動削除を検証する."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(sdf_baker, "get_cache_dir", return_value=tmp):
                # 1. 通常ラウンドトリップ
                res = self._make_valid_result(bone_names=["Spine", "Chest"])
                sdf_baker.save_cached_sdf("valid_key", res)
                self.assertEqual([p.name for p in Path(tmp).glob("*.tmp*")], [])
                loaded = sdf_baker.load_cached_sdf("valid_key")
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded.bone_names, ["Spine", "Chest"])
                self.assertEqual(loaded.texture_bytes, res.texture_bytes)

                # 2. 空 bone_names=[] でも dtype=object にならず allow_pickle=False で読込可能
                res_empty = self._make_valid_result(bone_names=[])
                sdf_baker.save_cached_sdf("empty_bones", res_empty)
                loaded_empty = sdf_baker.load_cached_sdf("empty_bones")
                self.assertIsNotNone(loaded_empty)
                self.assertEqual(loaded_empty.bone_names, [])

                # 3. 破損テキスト/切り詰め npz は自動削除して None を返す
                bad_zip_path = Path(tmp) / "corrupt_key.npz"
                bad_zip_path.write_bytes(b"corrupted non-zip payload")
                self.assertIsNone(sdf_baker.load_cached_sdf("corrupt_key"))
                self.assertFalse(bad_zip_path.exists())

                # 4. pickle オブジェクト入り npz は allow_pickle=False により拒否＆自動削除
                pickle_path = Path(tmp) / "pickle_key.npz"
                np.savez(pickle_path, bone_names=np.array([{"evil": 1}], dtype=object))
                self.assertIsNone(sdf_baker.load_cached_sdf("pickle_key"))
                self.assertFalse(pickle_path.exists())


class TestBlenderManagerSecurityRegressions(unittest.TestCase):
    """blender_manager.py の SHA-256 検証および Zip Slip 防御テスト."""

    def test_verify_zip_sha256_match_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "sample.zip"
            payload = b"blender-archive-payload"
            zpath.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()

            blender_manager._verify_zip_sha256(zpath, expected_sha256=digest)
            bad_digest = ("0" if digest[0] != "0" else "1") + digest[1:]
            with self.assertRaises(RuntimeError):
                blender_manager._verify_zip_sha256(zpath, expected_sha256=bad_digest)

    def test_safe_extract_zip_blocks_zip_slip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ok_zip = root / "ok.zip"
            with zipfile.ZipFile(ok_zip, "w") as zf:
                zf.writestr("blender/bin.txt", "ok")
            dest_ok = root / "dest_ok"
            dest_ok.mkdir()
            blender_manager._safe_extract_zip(ok_zip, dest_dir=dest_ok)
            self.assertEqual((dest_ok / "blender" / "bin.txt").read_text(encoding="utf-8"), "ok")

            evil_zip = root / "evil.zip"
            with zipfile.ZipFile(evil_zip, "w") as zf:
                zf.writestr("../escaped.txt", "pwned")
            dest_evil = root / "nested" / "dest_evil"
            dest_evil.mkdir(parents=True)
            with self.assertRaises(RuntimeError):
                blender_manager._safe_extract_zip(evil_zip, dest_dir=dest_evil)
            self.assertFalse((root / "nested" / "escaped.txt").exists())


class TestPyO3SafetyAndWrinkleRegressions(unittest.TestCase):
    """taremin_cloth_core (PyO3) の Read-Only 拒否・境界検証・wrinkle_field 部分更新テスト."""

    def _make_minimal_sim(self) -> taremin_cloth_core.ClothSimulator:
        pos = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
        edges = np.array([[0, 1]], dtype=np.uint32)
        return taremin_cloth_core.ClothSimulator(positions=pos, edges=edges)

    def test_readonly_numpy_array_rejected(self):
        """Read-Only NumPy 配列への get_positions / fetch_positions / fetch_buffered_positions を拒否すること."""
        sim = self._make_minimal_sim()
        sim.step(1.0 / 60.0, 1, 1)

        ro_arr = np.zeros(6, dtype=np.float32)
        ro_arr.setflags(write=False)

        with self.assertRaises(ValueError):
            sim.get_positions(ro_arr)
        with self.assertRaises(ValueError):
            sim.fetch_positions(ro_arr)
        with self.assertRaises(ValueError):
            sim.fetch_buffered_positions(ro_arr)

    def test_invalid_inputs_raise_value_error(self):
        """範囲外インデックス・長さ不一致・テクスチャ乗算オーバーフローで ValueError が送出されること."""
        pos = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32)
        valid_edges = np.array([[0, 1]], dtype=np.uint32)

        with self.assertRaises(ValueError):
            taremin_cloth_core.ClothSimulator(
                positions=pos,
                edges=np.array([[0, 99]], dtype=np.uint32),
            )
        with self.assertRaises(ValueError):
            taremin_cloth_core.ClothSimulator(
                positions=pos,
                edges=valid_edges,
                faces=np.array([[0, 1, 99]], dtype=np.uint32),
            )
        with self.assertRaises(ValueError):
            taremin_cloth_core.ClothSimulator(
                positions=pos,
                edges=valid_edges,
                inv_masses=np.array([1.0], dtype=np.float32),
            )

        sim = taremin_cloth_core.ClothSimulator(positions=pos, edges=valid_edges)
        with self.assertRaises(ValueError):
            sim.set_wrinkle_field_texture_2d(0x80000000, 4, b"", 0.0, 1.0, 0.0, 1.0)

    def test_set_wrinkle_field_params_partial_range_update(self):
        """z_min/z_max と r_min/r_max の片側部分更新が未指定側レンジを破壊せず維持すること."""
        pos_init = np.array([[0.15, 0.0, 0.0], [0.15, 0.1, 0.0]], dtype=np.float32)
        edges = np.array([[0, 1]], dtype=np.uint32)
        inv_masses = np.array([1.0, 1.0], dtype=np.float32)
        sim = taremin_cloth_core.ClothSimulator(
            positions=pos_init,
            edges=edges,
            inv_masses=inv_masses,
            stiffness=0.0,
            bending_stiffness=0.0,
        )
        sim.set_gravity(0.0, 0.0, 0.0)
        sim.set_damping(1.0)

        # G=255 (crest_w=1.0), A=255 (target_crest_r = r_max) -> +r (+x) 方向に押すテクスチャ
        tex_bytes = bytes([0, 255, 0, 255] * 4)
        sim.set_wrinkle_field_texture_2d(2, 2, tex_bytes, -1.0, 1.0, 0.05, 0.30)
        sim.set_enable_wrinkle_field(True)
        sim.set_wrinkle_field_params(
            origin=[0.0, 0.0, 0.0],
            axis=[0.0, 0.0, 1.0],
            normal=[1.0, 0.0, 0.0],
            binormal=[0.0, 1.0, 0.0],
            influence_radius=0.05,
            bone_radius=0.01,
            stiffness=50.0,
            blend_weight=1.0,
            z_min=-1.0,
            z_max=1.0,
            r_min=0.05,
            r_max=0.30,
        )

        # 1. z_min のみ更新 (r_min/r_max=None): 既存 r_range=[0.05, 0.30] が維持され +x 方向へ変位する
        sim.set_wrinkle_field_params(
            origin=[0.0, 0.0, 0.0],
            axis=[0.0, 0.0, 1.0],
            normal=[1.0, 0.0, 0.0],
            binormal=[0.0, 1.0, 0.0],
            influence_radius=0.05,
            bone_radius=0.01,
            stiffness=50.0,
            blend_weight=1.0,
            z_min=-0.5,
        )
        sim.step(1.0 / 60.0, 1, 1)
        out1 = np.empty(6, dtype=np.float32)
        sim.get_positions(out1)
        dx1 = float(out1[0] - 0.15)
        self.assertGreater(dx1, 1e-4, f"z_min 部分更新後に r_range が失われました: dx={dx1}")

        # 2. リセットして r_min/r_max のみ更新 (z_min/z_max=None): r_cutoff = 0.08 + 0.05 = 0.13 < 0.15 となりほぼ無変位
        sim.set_positions_and_velocities(pos_init, np.zeros_like(pos_init))
        sim.set_wrinkle_field_params(
            origin=[0.0, 0.0, 0.0],
            axis=[0.0, 0.0, 1.0],
            normal=[1.0, 0.0, 0.0],
            binormal=[0.0, 1.0, 0.0],
            influence_radius=0.05,
            bone_radius=0.01,
            stiffness=50.0,
            blend_weight=1.0,
            r_min=0.02,
            r_max=0.08,
        )
        sim.step(1.0 / 60.0, 1, 1)
        out2 = np.empty(6, dtype=np.float32)
        sim.get_positions(out2)
        dx2 = float(abs(out2[0] - 0.15))
        self.assertLess(dx2, 1e-6, f"r_min/r_max 部分更新が反映されませんでした: dx={dx2}")


if __name__ == "__main__":
    unittest.main()
