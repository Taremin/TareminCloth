"""Taremin Cloth テストスイートパッケージ"""
import sys
from pathlib import Path

# python ディレクトリおよびリポジトリルートを sys.path に自動追加
_repo_root = Path(__file__).resolve().parent.parent
_python_dir = _repo_root / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
# Blender 固有モジュールがインストールされていない環境（CIやスタンドアロン環境）の場合、
# taremin_cloth の各モジュールインポートが失敗しないよう一括モック登録
_BLENDER_MODULES = [
    "bpy", "bpy.types", "bpy.props", "bpy.ops", "bpy.utils", "bpy.context", "bpy.path", "bpy.app",
    "bpy_extras", "bpy_extras.view3d_utils", "bpy_extras.batch",
    "gpu", "gpu.types", "gpu.shader", "gpu.matrix",
    "gpu_extras", "gpu_extras.batch",
    "blf",
    "bmesh", "bmesh.types", "bmesh.ops",
    "mathutils", "mathutils.kdtree", "mathutils.bvhtree", "mathutils.geometry",
]
for mod_name in _BLENDER_MODULES:
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except ImportError:
            from unittest.mock import MagicMock
            m = MagicMock()
            m.__name__ = mod_name
            sys.modules[mod_name] = m

try:
    from .fixtures import panel_test_utils
    # 後方互換用モジュールエイリアス
    sys.modules['tests.panel_test_utils'] = panel_test_utils
except ImportError:
    # bpy が利用できない環境（スタンドアロンPythonテスト実行時）
    panel_test_utils = None
