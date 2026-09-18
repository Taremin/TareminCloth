"""Taremin Cloth テストスイートパッケージ"""
import sys
from pathlib import Path

# Windows 環境下での UnicodeEncodeError (cp1252 charmap 等) を防止
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# python ディレクトリおよびリポジトリルートを sys.path に自動追加
_repo_root = Path(__file__).resolve().parent.parent
_python_dir = _repo_root / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


class _DummyTypesModule:
    """bpy.types 用の動的クラスプロバイダ。
    Operator や PropertyGroup 等が MagicMock になると継承先クラスの
    メソッド・属性・アノテーションが消失するため、通常のPythonクラスを提供する。
    """
    def __init__(self):
        self._types = {
            "Operator": type("Operator", (object,), {}),
            "Panel": type("Panel", (object,), {}),
            "PropertyGroup": type("PropertyGroup", (object,), {}),
            "UIList": type("UIList", (object,), {}),
            "AddonPreferences": type("AddonPreferences", (object,), {}),
            "Menu": type("Menu", (object,), {}),
            "Header": type("Header", (object,), {}),
            "Gizmo": type("Gizmo", (object,), {}),
            "GizmoGroup": type("GizmoGroup", (object,), {}),
            "Node": type("Node", (object,), {}),
            "NodeTree": type("NodeTree", (object,), {}),
            "NodeSocket": type("NodeSocket", (object,), {}),
            "UILayout": type("UILayout", (object,), {}),
        }

    def __getattr__(self, name):
        if name not in self._types:
            self._types[name] = type(name, (object,), {})
        return self._types[name]


# Blender 固有モジュールがインストールされていない環境（CIやスタンドアロン環境）の場合、
# taremin_cloth の各モジュールインポートが失敗しないよう一括モック登録
_BLENDER_MODULES = [
    "bpy", "bpy.props", "bpy.ops", "bpy.utils", "bpy.context", "bpy.path", "bpy.app",
    "bpy_extras", "bpy_extras.view3d_utils", "bpy_extras.batch",
    "gpu", "gpu.types", "gpu.shader", "gpu.matrix",
    "gpu_extras", "gpu_extras.batch",
    "blf",
    "bmesh", "bmesh.types", "bmesh.ops",
    "mathutils", "mathutils.kdtree", "mathutils.bvhtree", "mathutils.geometry",
]
from unittest.mock import MagicMock

if "bpy.types" not in sys.modules:
    types_mod = _DummyTypesModule()
    sys.modules["bpy.types"] = types_mod

for mod_name in _BLENDER_MODULES:
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except ImportError:
            m = MagicMock()
            m.__name__ = mod_name
            if mod_name == "bpy":
                m.types = sys.modules.get("bpy.types", _DummyTypesModule())
            sys.modules[mod_name] = m

try:
    from .fixtures import panel_test_utils
    # 後方互換用モジュールエイリアス
    sys.modules['tests.panel_test_utils'] = panel_test_utils
except ImportError:
    # bpy が利用できない環境（スタンドアロンPythonテスト実行時）
    panel_test_utils = None

