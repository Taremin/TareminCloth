"""
Blender非依存の高速ソフトウェアZ-bufferレンダラー
表面（表）を白、背面（裏・裏返り面）を赤、ライティングなし(Unlit)で高速レンダリングします。
Rustコア (taremin_cloth_core) に内蔵された高速C拡張ラスタライザを使用し、数ミリ秒で画像保存と赤色ピクセル判定を行います。
"""

import os
from typing import Optional, Tuple
import numpy as np

try:
    import taremin_cloth_core
    _HAS_RUST_CORE = True
except ImportError:
    _HAS_RUST_CORE = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def render_mesh_to_file(
    filepath: str,
    positions: np.ndarray,
    faces: np.ndarray,
    width: int = 800,
    height: int = 600,
    camera_pos: Optional[Tuple[float, float, float]] = None,
    camera_target: Optional[Tuple[float, float, float]] = None,
    fov: float = 45.0,
    **kwargs
) -> Tuple[str, int]:
    """
    メッシュをRust内蔵のソフトウェアZ-bufferラスタライザで描画し、PNG画像として保存します。

    Args:
        filepath: 保存先PNGファイルパス
        positions: [N, 3] 頂点座標配列
        faces: [M, 3] 三角形インデックス配列
        width: 画像幅
        height: 画像高さ
        camera_pos: カメラ位置 (未指定時はAABBから自動算出)
        camera_target: 注視点位置 (未指定時は重心)
        fov: 視野角(度)

    Returns:
        (filepath, red_pixels): 保存先パスと裏面露出（赤）ピクセル数のタプル
    """
    pos = np.ascontiguousarray(positions, dtype=np.float32)
    fcs = np.ascontiguousarray(faces, dtype=np.uint32)

    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)

    if _HAS_RUST_CORE and hasattr(taremin_cloth_core, "render_mesh_to_png"):
        red_pixels = taremin_cloth_core.render_mesh_to_png(
            filepath,
            pos,
            fcs,
            width=width,
            height=height,
            camera_pos=list(camera_pos) if camera_pos is not None else None,
            camera_target=list(camera_target) if camera_target is not None else None,
            fov=float(fov),
        )
        return filepath, red_pixels
    else:
        raise RuntimeError("taremin_cloth_core.render_mesh_to_png が利用できません。")


def render_mesh_to_image(
    positions: np.ndarray,
    faces: np.ndarray,
    width: int = 800,
    height: int = 600,
    camera_pos: Optional[Tuple[float, float, float]] = None,
    camera_target: Optional[Tuple[float, float, float]] = None,
    fov: float = 45.0,
    **kwargs
):
    """
    PIL.Imageオブジェクトを返却する互換用ヘルパー。
    内部で一時PNGを生成してロードします。
    """
    if not _HAS_PIL:
        raise RuntimeError("PIL (Pillow) がインストールされていません")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        render_mesh_to_file(
            tmp_path, positions, faces,
            width=width, height=height,
            camera_pos=camera_pos, camera_target=camera_target, fov=fov,
            **kwargs
        )
        img = Image.open(tmp_path)
        img.load()
        return img
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def count_red_pixels(
    image,
    red_threshold: int = 200,
    green_blue_max: int = 50
) -> int:
    """
    画像内の赤色ピクセル（裏返り・背面露出面）の総数をカウントします。
    """
    arr = np.array(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return 0

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    red_mask = (r >= red_threshold) & (g <= green_blue_max) & (b <= green_blue_max)
    return int(np.sum(red_mask))
