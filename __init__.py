bl_info = {
    "name": "Taremin Cloth",
    "author": "Taremin",
    "version": (0, 1, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > Taremin Cloth",
    "description": "GPU-accelerated XPBD Cloth Simulator for Blender",
    "category": "Physics",
    "license": "GPL-3.0-or-later",
}

import importlib
import sys
from pathlib import Path

import bpy

# pythonディレクトリおよびアドオンルートをsys.pathに追加
_addon_dir = Path(__file__).parent
_python_dir = _addon_dir / "python"
_inner_pkg_dir = _python_dir / "taremin_cloth"

if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))
if str(_addon_dir) not in sys.path:
    sys.path.insert(0, str(_addon_dir))

# サブモジュール（properties, operators, panels等）の探索先として内部パッケージディレクトリを追加
if hasattr(sys.modules[__name__], "__path__"):
    pkg_paths = getattr(sys.modules[__name__], "__path__")
    if str(_inner_pkg_dir) not in pkg_paths:
        pkg_paths.append(str(_inner_pkg_dir))

# サブパッケージおよびモジュール定義
subpackages = [
    (
        "utils",
        [
            "logger",
            "mesh_extract",
            "view3d",
            "drawing",
            "topology",
        ],
    ),
    (
        "",
        [
            "i18n",
            "preferences",
            "properties",
            "presets",
            "operators",
            "panels",
        ],
    ),
]

# モジュール読み込み & リロード
modules = []
pkg_prefix = f"{__package__}." if __package__ else ""
for subpkg, mod_names in subpackages:
    for name in mod_names:
        fullname = f"{pkg_prefix}{subpkg}.{name}" if subpkg else f"{pkg_prefix}{name}"
        if fullname in sys.modules:
            try:
                mod = importlib.reload(sys.modules[fullname])
            except Exception:
                mod = sys.modules[fullname]
        else:
            mod = importlib.import_module(fullname)
        modules.append(mod)

# drawing モジュールの取得
drawing = None
for mod in modules:
    if mod.__name__.endswith(".utils.drawing") or mod.__name__ == "utils.drawing":
        drawing = mod
        break


_is_registered = False


def register():
    global _is_registered
    if _is_registered:
        return
    for mod in modules:
        if hasattr(mod, "register"):
            mod.register()
    if drawing and hasattr(drawing, "register_draw_handler"):
        drawing.register_draw_handler()
    _is_registered = True


def unregister():
    global _is_registered
    if not _is_registered:
        return
    if drawing and hasattr(drawing, "unregister_draw_handler"):
        drawing.unregister_draw_handler()
    for mod in reversed(modules):
        if hasattr(mod, "unregister"):
            mod.unregister()
    _is_registered = False

    Path(__file__).touch()


if __name__ == "__main__":
    register()


