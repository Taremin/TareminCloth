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
# bpy がインストールされていない環境（CIやスタンドアロン環境）の場合、
# taremin_cloth のモジュールインポートが失敗しないようにモック bpy を登録
if "bpy" not in sys.modules:
    try:
        import bpy
    except ImportError:
        from unittest.mock import MagicMock
        bpy = MagicMock()
        bpy.__name__ = "bpy"
        sys.modules["bpy"] = bpy
        sys.modules["bpy.types"] = bpy.types
        sys.modules["bpy.props"] = bpy.props
        sys.modules["bpy.ops"] = bpy.ops
        sys.modules["bpy.utils"] = bpy.utils
        sys.modules["bpy.context"] = bpy.context

try:
    from .fixtures import panel_test_utils
    # 後方互換用モジュールエイリアス
    sys.modules['tests.panel_test_utils'] = panel_test_utils
except ImportError:
    # bpy が利用できない環境（スタンドアロンPythonテスト実行時）
    panel_test_utils = None
