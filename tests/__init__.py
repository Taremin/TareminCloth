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

try:
    from .fixtures import panel_test_utils
    # 後方互換用モジュールエイリアス
    sys.modules['tests.panel_test_utils'] = panel_test_utils
except ImportError:
    # bpy が利用できない環境（スタンドアロンPythonテスト実行時）
    panel_test_utils = None
