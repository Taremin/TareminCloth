"""Mesh SDF の VRAM 予算解決ロジックの単体テスト（Blender不要）.

背景: 旧ドキュメントの「wgpuのバッファサイズ上限 256MB」はデフォルト下限の
記述であり、実効上限はデバイス依存（dGPUで1〜2GB等）。予算解決がデバイス
実効上限と連動していることを検証する。
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

import taremin_cloth_core
from taremin_cloth.engine.sdf_baker import (
    DEFAULT_MESH_SDF_MAX_VRAM_MB,
    MAX_MESH_SDF_VRAM_MB,
    MIN_MESH_SDF_VRAM_MB,
    compute_effective_voxel_size,
    get_device_vram_limit_mb,
    resolve_effective_vram_budget,
)


class TestVramBudget(unittest.TestCase):
    def test_device_buffer_limits_exposed(self):
        """Rustコアがデバイス実効上限を公開し、少なくとも既定下限以上であること"""
        if not taremin_cloth_core.is_gpu_available():
            self.skipTest("GPUが利用できない環境のためスキップします")
        self.assertTrue(
            hasattr(taremin_cloth_core, "get_device_buffer_limits"),
            "get_device_buffer_limits が公開されていること",
        )
        limits = taremin_cloth_core.get_device_buffer_limits()
        self.assertGreaterEqual(
            int(limits["max_sdf_mb"]), 128, "実効上限は少なくとも既定下限オーダーであること"
        )
        self.assertEqual(
            int(limits["max_sdf_mb"]),
            get_device_vram_limit_mb(),
            "PythonヘルパーとRustコアの値が一致すること",
        )

    def test_resolve_budget_normalization(self):
        """要求予算の正規化（範囲クランプ）がGPU有無によらず成立すること"""
        self.assertEqual(
            resolve_effective_vram_budget(None), DEFAULT_MESH_SDF_MAX_VRAM_MB
        )
        self.assertEqual(resolve_effective_vram_budget(1), MIN_MESH_SDF_VRAM_MB)
        device_mb = get_device_vram_limit_mb()
        if device_mb is None:
            self.assertEqual(resolve_effective_vram_budget(4096), MAX_MESH_SDF_VRAM_MB)
        else:
            self.assertEqual(
                resolve_effective_vram_budget(10**9), min(MAX_MESH_SDF_VRAM_MB, device_mb)
            )
            self.assertLessEqual(resolve_effective_vram_budget(4096), device_mb)

    def test_explicit_256_budget_still_clamps(self):
        """明示256MB指定時は後方互換として従来通りクランプされること"""
        size = np.array([0.86, 0.86, 0.86], dtype=np.float32)
        eff_v, is_clamped, raw_vram = compute_effective_voxel_size(
            size, requested_voxel_size=0.002, max_vram_mb=256, auto_scale=True
        )
        self.assertGreater(raw_vram, 256.0)
        # デバイス上限が256MB以下の極端な環境でなければ従来通りクランプ
        device_mb = get_device_vram_limit_mb()
        if device_mb is None or device_mb >= 256:
            self.assertTrue(is_clamped)
            self.assertGreater(eff_v, 0.002)

    def test_device_aware_budget_avoids_needless_degrade(self):
        """デバイス上限が大きい環境では303MB要求が既定予算で劣化しないこと"""
        if not taremin_cloth_core.is_gpu_available():
            self.skipTest("GPUが利用できない環境のためスキップします")
        device_mb = get_device_vram_limit_mb()
        if device_mb is None or device_mb < 512:
            self.skipTest(f"デバイス上限が小さすぎるためスキップします ({device_mb}MB)")
        size = np.array([0.86, 0.86, 0.86], dtype=np.float32)
        eff_v, is_clamped, raw_vram = compute_effective_voxel_size(
            size,
            requested_voxel_size=0.002,
            max_vram_mb=DEFAULT_MESH_SDF_MAX_VRAM_MB,
            auto_scale=True,
        )
        self.assertGreater(raw_vram, 256.0, "旧既定256MBではクランプされていた要求サイズであること")
        self.assertFalse(is_clamped, "既定1024MB予算では劣化せず通ること")
        self.assertAlmostEqual(eff_v, 0.002)


if __name__ == "__main__":
    unittest.main()
