# -*- coding: utf-8 -*-
"""
エンジンコア機能・クライアント統合カバレッジの自動検証テスト

tools/audit_feature_coverage.py の解析ロジックを用いて、
Blenderアドオンおよび独立GUIの実装状況・パリティ退行をCIで自動検出する。
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.audit_feature_coverage import run_full_audit, METADATA_FIELDS


class TestFeatureCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_full_audit(PROJECT_ROOT)

    def test_blender_engine_full_parity(self):
        """SimConfigの物理パラメータがBlender Engine (simconfig.py) で100%網羅されていること"""
        phys_params = [p for p in self.report.params if p["name"] not in METADATA_FIELDS]
        missing = [p["name"] for p in phys_params if not p["blender_engine"]]
        self.assertEqual(
            missing,
            [],
            f"Blender Engine (simconfig.py) に未同期のSimConfigパラメータがあります: {missing}",
        )

    def test_gui_init_full_parity(self):
        """GUI IPC初期化 (SceneInitData) で全物理パラメータが100%同期されていること"""
        phys_params = [p for p in self.report.params if p["name"] not in METADATA_FIELDS]
        missing = [p["name"] for p in phys_params if not p["gui_init"]]
        self.assertEqual(
            missing,
            [],
            f"GUI IPC Init (SceneInitData) に未同期のパラメータがあります: {missing}",
        )

    def test_gui_update_full_parity(self):
        """GUI IPC動的更新 (GuiParamsUpdate) で全物理パラメータが100%同期されていること"""
        phys_params = [p for p in self.report.params if p["name"] not in METADATA_FIELDS]
        missing = [p["name"] for p in phys_params if not p["gui_update"]]
        self.assertEqual(
            missing,
            [],
            f"GUI IPC Update (GuiParamsUpdate) に未同期のパラメータがあります: {missing}",
        )

    def test_gui_standalone_ui_full_parity(self):
        """独立GUI 単体UI (app.rs) で全物理パラメータが100%網羅・操作可能であること"""
        phys_params = [p for p in self.report.params if p["name"] not in METADATA_FIELDS]
        missing = [p["name"] for p in phys_params if not p["gui_ui"]]
        self.assertEqual(
            missing,
            [],
            f"独立GUI 単体UI (app.rs) に未実装のパラメータがあります: {missing}",
        )

    def test_feature_matrix_integrity(self):
        """全3軸の解析が正常に完了し、各レイヤーの測定値が有効であること"""
        self.assertGreater(len(self.report.params), 50)
        self.assertGreater(len(self.report.apis), 50)
        self.assertGreater(len(self.report.features), 15)

        # 軸 B: APIメソッドが両クライアントで一定以上使われていること
        a_b, t_a = self.report.api_counts["blender_api"]
        self.assertGreaterEqual(a_b / t_a, 0.5, "BlenderでのAPI利用率が50%を下回っています")

        # 軸 C: Blender機能網羅率が85%以上であること
        f_b, t_f = self.report.feature_counts["blender_feature"]
        self.assertGreaterEqual(f_b / t_f, 0.85, "Blenderでの機能網羅率が85%を下回っています")


if __name__ == "__main__":
    unittest.main()
