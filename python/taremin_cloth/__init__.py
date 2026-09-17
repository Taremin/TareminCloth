bl_info = {
    "name": "Taremin Cloth",
    "author": "Taremin",
    "version": (0, 1, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > Taremin Cloth",
    "description": "GPU-accelerated XPBD Cloth Simulation for Blender",
    "category": "Physics",
    "license": "GPL-3.0-or-later",
}

import importlib
import sys
from pathlib import Path

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
            "collider_detect",
            "self_collision_fit",
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

from .engine.runner import step_cloth_object, step_cloth_scene


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
