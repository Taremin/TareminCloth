"""SDFキャッシュ健全性ゲートの単体テスト（Blender・GPU不要）.

TDR等でゼロ埋めテクスチャが正常として保存・再読込される毒キャッシュ事故
（全零SDF→接触無反応→「コライダーを無視」）の再発防止を検証する。
"""

import os
import sys
import unittest

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_pkg = os.path.join(project_root, "python")
if python_pkg not in sys.path:
    sys.path.insert(0, python_pkg)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from taremin_cloth.engine.sdf_baker import (
    BoneSdfBakeResult,
    check_sdf_texture_integrity,
    get_cache_dir,
    load_cached_sdf,
    save_cached_sdf,
)


def _make_valid_texture(w, h, d):
    n = w * h * d
    vals = (np.arange(n, dtype=np.uint32) * 2654435761) & 0xFFFF
    # NaN/Infパターンを除外（上位5bitが129以上にならないよう下位に抑える）
    vals = vals & 0x3BFF
    packed = (vals | (np.full(n, 0x3C00, dtype=np.uint32) << 16)).astype(np.uint32)
    return packed.tobytes()


def _make_result(tex, w=8, h=8, d=8):
    return BoneSdfBakeResult(
        texture_bytes=tex,
        width=w,
        height=h,
        depth=d,
        bone_infos=np.zeros((1, 20), dtype=np.float32),
        bone_names=["TestBone"],
        bind_matrices=np.eye(4, dtype=np.float32).reshape(1, 4, 4),
    )


class TestSdfCacheIntegrity(unittest.TestCase):
    def test_zero_texture_rejected(self):
        ok, reason = check_sdf_texture_integrity(bytes(8 * 8 * 8 * 4), 8, 8, 8)
        self.assertFalse(ok)
        self.assertIn("uniform", reason)

    def test_uniform_nonzero_rejected(self):
        tex = (np.full(8 * 8 * 8, 0x3C003C00, dtype=np.uint32)).tobytes()
        ok, reason = check_sdf_texture_integrity(tex, 8, 8, 8)
        self.assertFalse(ok)

    def test_nan_texture_rejected(self):
        tex = bytearray(_make_valid_texture(8, 8, 8))
        # 先頭ボクセルのdist halfをNaN (0x7E00) に改変
        tex[0] = 0x00
        tex[1] = 0x7E
        ok, reason = check_sdf_texture_integrity(bytes(tex), 8, 8, 8)
        self.assertFalse(ok)
        self.assertIn("NaN", reason)

    def test_size_mismatch_rejected(self):
        ok, _ = check_sdf_texture_integrity(bytes(100), 8, 8, 8)
        self.assertFalse(ok)

    def test_valid_texture_accepted(self):
        ok, reason = check_sdf_texture_integrity(_make_valid_texture(8, 8, 8), 8, 8, 8)
        self.assertTrue(ok, reason)

    def test_poison_not_saved(self):
        key = "test_integrity_poison_unit"
        path = os.path.join(get_cache_dir(), f"{key}.npz")
        if os.path.exists(path):
            os.remove(path)
        save_cached_sdf(key, _make_result(bytes(8 * 8 * 8 * 4)))
        self.assertFalse(os.path.exists(path), "不正結果は保存されないこと")

    def test_valid_roundtrip(self):
        key = "test_integrity_valid_unit"
        path = os.path.join(get_cache_dir(), f"{key}.npz")
        try:
            save_cached_sdf(key, _make_result(_make_valid_texture(8, 8, 8)))
            self.assertTrue(os.path.exists(path))
            loaded = load_cached_sdf(key)
            self.assertIsNotNone(loaded)
            self.assertEqual((loaded.width, loaded.height, loaded.depth), (8, 8, 8))
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_poison_file_self_heals_on_load(self):
        """既存の毒ファイルは読込時に削除されNoneが返る（再ベイクが走る）."""
        import hashlib

        key = "test_integrity_heal_unit"
        path = os.path.join(get_cache_dir(), f"{key}.npz")
        res = _make_result(bytes(8 * 8 * 8 * 4))
        np.savez_compressed(
            path,
            texture_bytes=np.frombuffer(res.texture_bytes, dtype=np.uint8),
            width=res.width,
            height=res.height,
            depth=res.depth,
            bone_infos=res.bone_infos,
            bone_names=np.array(res.bone_names),
            bind_matrices=res.bind_matrices,
            joint_face_indices=np.empty(0, dtype=np.int32),
            pair_keys=np.array([]),
            pair_lens=np.array([], dtype=np.int32),
            pair_data=np.empty(0, dtype=np.int32),
        )
        self.assertTrue(os.path.exists(path))
        self.assertIsNone(load_cached_sdf(key))
        self.assertFalse(os.path.exists(path), "毒ファイルは削除されること")


if __name__ == "__main__":
    unittest.main()
