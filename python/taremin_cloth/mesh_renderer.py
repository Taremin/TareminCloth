"""
Blender非依存の高速ソフトウェアZ-bufferレンダラー
表面（表）を白、背面（裏・裏返り面）を赤、ライティング付きで高速レンダリングします。
縫合エッジ（水色）やコライダー（スレートグレー）の同時描画、
およびアニメーションPNG (APNG) / GIF の生成をサポートします。
"""

import os
from typing import List, Optional, Sequence, Tuple, Union
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


def render_scene_to_file(
    filepath: str,
    positions: np.ndarray,
    faces: np.ndarray,
    sewing_springs: Optional[np.ndarray] = None,
    mesh_colliders: Optional[np.ndarray] = None,
    vertex_colors: Optional[np.ndarray] = None,
    extra_lines: Optional[np.ndarray] = None,
    width: int = 800,
    height: int = 600,
    camera_pos: Optional[Tuple[float, float, float]] = None,
    camera_target: Optional[Tuple[float, float, float]] = None,
    fov: float = 45.0,
    draw_wireframe: bool = True,
    wire_width: float = 1.0,
    collider_color: Optional[Tuple[int, int, int]] = None,
    sewing_color: Optional[Tuple[int, int, int]] = None,
    **kwargs
) -> Tuple[str, int]:
    """
    シーン（布メッシュ、縫合エッジ、コライダー、頂点カラー、追加3D線分）を
    Rust内蔵のソフトウェアラスタライザで描画し、PNG画像として保存します。

    Args:
        filepath: 保存先PNGファイルパス
        positions: [N, 3] 布頂点座標配列
        faces: [M, 3] 布三角形インデックス配列
        sewing_springs: [K, 2] 縫合エッジ頂点インデックスペア
        mesh_colliders: [L, 9] コライダー三角形配列 (p0x..z, p1x..z, p2x..z)
        vertex_colors: [N, 3] 頂点ごとのRGBカラー (0-255 uint8, ヒートマップ等)
        extra_lines: [E, 9] 追加3Dライン配列 (p0x..z, p1x..z, r, g, b)
        width: 画像幅
        height: 画像高さ
        camera_pos: カメラ位置 (未指定時はAABBから自動算出)
        camera_target: 注視点位置 (未指定時はAABB中心)
        fov: 視野角(度)
        draw_wireframe: ワイヤーフレームを描画するか
        wire_width: ワイヤーフレーム太さ(px)
        collider_color: コライダー色 (RGB 0-255)
        sewing_color: 縫合エッジ色 (RGB 0-255)

    Returns:
        (filepath, red_pixels): 保存先パスと裏面露出（赤）ピクセル数のタプル
    """
    pos = np.ascontiguousarray(positions, dtype=np.float32)
    fcs = np.ascontiguousarray(faces, dtype=np.uint32)

    sew = None
    if sewing_springs is not None and len(sewing_springs) > 0:
        sew = np.ascontiguousarray(sewing_springs, dtype=np.uint32)

    cols = None
    if mesh_colliders is not None and len(mesh_colliders) > 0:
        cols_arr = np.ascontiguousarray(mesh_colliders, dtype=np.float32)
        if cols_arr.ndim == 3 and cols_arr.shape[1] == 3 and cols_arr.shape[2] == 3:
            cols_arr = cols_arr.reshape((-1, 9))
        cols = cols_arr

    vc = None
    if vertex_colors is not None and len(vertex_colors) > 0:
        vc = np.ascontiguousarray(vertex_colors, dtype=np.uint8)

    lines = None
    if extra_lines is not None and len(extra_lines) > 0:
        lines = np.ascontiguousarray(extra_lines, dtype=np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)

    if _HAS_RUST_CORE and hasattr(taremin_cloth_core, "render_scene_to_png"):
        red_pixels = taremin_cloth_core.render_scene_to_png(
            filepath,
            pos,
            fcs,
            sewing_springs=sew,
            mesh_colliders=cols,
            width=width,
            height=height,
            camera_pos=list(camera_pos) if camera_pos is not None else None,
            camera_target=list(camera_target) if camera_target is not None else None,
            fov=float(fov),
            draw_wireframe=bool(draw_wireframe),
            wire_width=float(wire_width),
            collider_color=list(collider_color) if collider_color is not None else None,
            sewing_color=list(sewing_color) if sewing_color is not None else None,
            vertex_colors=vc,
            extra_lines=lines,
        )
        return filepath, red_pixels
    elif _HAS_RUST_CORE and hasattr(taremin_cloth_core, "render_mesh_to_png"):
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
        raise RuntimeError("taremin_cloth_core のレンダリング機能が利用できません。")


def render_mesh_to_file(
    filepath: str,
    positions: np.ndarray,
    faces: np.ndarray,
    **kwargs
) -> Tuple[str, int]:
    """互換用ヘルパー関数: render_scene_to_file を呼び出します"""
    return render_scene_to_file(filepath, positions, faces, **kwargs)


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
        render_scene_to_file(
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


def create_animation_file(
    frames: Sequence[Union[str, "Image.Image"]],
    output_path: str,
    fps: int = 30,
    loop: int = 0
) -> str:
    """
    画像ファイルパス一覧（または PIL.Image 一覧）から、アニメーションPNG (APNG) または GIF を生成して保存します。

    Args:
        frames: 画像ファイルパス（または PIL.Image）のシーケンス
        output_path: 保存先パス (.png なら APNG、.gif なら GIF)
        fps: 再生フレームレート
        loop: ループ回数 (0: 無限ループ)

    Returns:
        output_path: 保存先ファイルパス
    """
    if not _HAS_PIL:
        raise RuntimeError("アニメーション生成には PIL (Pillow) が必要です。")

    if not frames:
        raise ValueError("アニメーション生成用のフレームが空です。")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    loaded_images = []
    opened_files = []

    try:
        for f in frames:
            if isinstance(f, str):
                img = Image.open(f)
                img.load()
                opened_files.append(img)
                loaded_images.append(img)
            else:
                loaded_images.append(f)

        duration_ms = max(1, int(1000.0 / max(1, fps)))
        ext = os.path.splitext(output_path)[1].lower()

        save_kwargs = {
            "save_all": True,
            "append_images": loaded_images[1:],
            "duration": duration_ms,
            "loop": loop,
        }

        if ext == ".png":
            # APNG 保存
            loaded_images[0].save(output_path, format="PNG", **save_kwargs)
        elif ext == ".gif":
            # GIF 保存 (RGB -> P 変換)
            adaptive = Image.Palette.ADAPTIVE if hasattr(Image, "Palette") and hasattr(Image.Palette, "ADAPTIVE") else 1
            gif_frames = [im.convert("P", palette=adaptive) for im in loaded_images]
            gif_frames[0].save(output_path, format="GIF", save_all=True, append_images=gif_frames[1:], duration=duration_ms, loop=loop)
        else:
            # デフォルトは APNG
            loaded_images[0].save(output_path, **save_kwargs)

        return output_path
    finally:
        for img in opened_files:
            try:
                img.close()
            except Exception:
                pass


# ==============================================================================
# コライダープリミティブの三角形メッシュ自動生成 (Step 3)
# ==============================================================================

def generate_sphere_mesh(center: Sequence[float], radius: float, u_segments: int = 16, v_segments: int = 8) -> np.ndarray:
    """球コライダーを三角形メッシュ配列 [T, 9] に変換します"""
    c = np.asarray(center, dtype=np.float32)
    u = np.linspace(0, 2 * np.pi, u_segments, endpoint=False)
    v = np.linspace(0, np.pi, v_segments + 1)

    x = radius * np.outer(np.cos(u), np.sin(v)).T
    y = radius * np.outer(np.sin(u), np.sin(v)).T
    z = radius * np.outer(np.ones_like(u), np.cos(v)).T

    verts = np.stack([x, y, z], axis=-1) + c

    tris = []
    for vi in range(v_segments):
        for ui in range(u_segments):
            next_ui = (ui + 1) % u_segments
            p00 = verts[vi, ui]
            p01 = verts[vi, next_ui]
            p10 = verts[vi + 1, ui]
            p11 = verts[vi + 1, next_ui]
            if vi > 0:
                tris.append(np.concatenate([p00, p01, p11]))
            if vi < v_segments - 1:
                tris.append(np.concatenate([p00, p11, p10]))
    return np.array(tris, dtype=np.float32) if tris else np.empty((0, 9), dtype=np.float32)


def generate_capsule_mesh(point_a: Sequence[float], point_b: Sequence[float], radius: float, segments: int = 12) -> np.ndarray:
    """カプセルコライダーを三角形メッシュ配列 [T, 9] に変換します"""
    pa = np.asarray(point_a, dtype=np.float32)
    pb = np.asarray(point_b, dtype=np.float32)
    axis = pb - pa
    h = float(np.linalg.norm(axis))
    if h < 1e-6:
        return generate_sphere_mesh(pa, radius, u_segments=segments, v_segments=max(4, segments // 2))

    dir_z = axis / h
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32) if abs(dir_z[1]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    dir_x = np.cross(dir_z, up)
    dir_x /= np.linalg.norm(dir_x)
    dir_y = np.cross(dir_z, dir_x)

    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    cos_t = np.cos(theta)[:, None]
    sin_t = np.sin(theta)[:, None]
    circle = cos_t * dir_x + sin_t * dir_y

    ring_a = pa + radius * circle
    ring_b = pb + radius * circle

    tris = []
    for i in range(segments):
        ni = (i + 1) % segments
        p_a0 = ring_a[i]
        p_a1 = ring_a[ni]
        p_b0 = ring_b[i]
        p_b1 = ring_b[ni]
        tris.append(np.concatenate([p_a0, p_b0, p_b1]))
        tris.append(np.concatenate([p_a0, p_b1, p_a1]))

    s_a = generate_sphere_mesh(pa, radius, u_segments=segments, v_segments=max(4, segments // 2))
    s_b = generate_sphere_mesh(pb, radius, u_segments=segments, v_segments=max(4, segments // 2))

    parts = [np.array(tris, dtype=np.float32)]
    if len(s_a) > 0:
        parts.append(s_a)
    if len(s_b) > 0:
        parts.append(s_b)
    return np.vstack(parts)


def generate_plane_mesh(point: Sequence[float], normal: Sequence[float], size: float = 2.0) -> np.ndarray:
    """平面コライダーを有限四角形メッシュ配列 [2, 9] に変換します"""
    pt = np.asarray(point, dtype=np.float32)
    n = np.asarray(normal, dtype=np.float32)
    norm_len = np.linalg.norm(n)
    if norm_len > 1e-6:
        n /= norm_len
    else:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u = np.cross(n, up)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    hs = size * 0.5
    p0 = pt - hs * u - hs * v
    p1 = pt + hs * u - hs * v
    p2 = pt + hs * u + hs * v
    p3 = pt - hs * u + hs * v

    t0 = np.concatenate([p0, p1, p2])
    t1 = np.concatenate([p0, p2, p3])
    return np.array([t0, t1], dtype=np.float32)


def convert_colliders_to_mesh(colliders: Sequence[dict], scene_bbox_size: float = 2.0) -> Optional[np.ndarray]:
    """
    ログやフレームデータ内の全コライダー（Mesh, Sphere, Capsule, Plane）を
    単一の三角形メッシュ配列 [T, 9] に統合変換します。
    """
    all_tris = []
    for c in colliders:
        c_type = c.get("type")
        if c_type == "mesh" and "triangles" in c:
            raw = c["triangles"]
            if raw:
                t = np.asarray(raw, dtype=np.float32).reshape((-1, 9))
                all_tris.append(t)
        elif c_type == "sphere" and "center" in c and "radius" in c:
            t = generate_sphere_mesh(c["center"], float(c["radius"]))
            if len(t) > 0:
                all_tris.append(t)
        elif c_type == "capsule" and "point_a" in c and "point_b" in c and "radius" in c:
            t = generate_capsule_mesh(c["point_a"], c["point_b"], float(c["radius"]))
            if len(t) > 0:
                all_tris.append(t)
        elif c_type == "plane" and "point" in c and "normal" in c:
            t = generate_plane_mesh(c["point"], c["normal"], size=scene_bbox_size * 2.0)
            if len(t) > 0:
                all_tris.append(t)

    if all_tris:
        return np.vstack(all_tris)
    return None


# ==============================================================================
# ヒートマップ・各種解析カラーマップ計算 (Step 4)
# ==============================================================================

def colormap_jet(values: np.ndarray, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """
    1D 配列を Jet カラーマップ (青 -> シアン -> 緑 -> 黄 -> 赤) の RGB [N, 3] uint8 配列に変換します。
    """
    val = np.asarray(values, dtype=np.float32)
    if vmin is None:
        vmin = float(np.nanmin(val)) if len(val) > 0 else 0.0
    if vmax is None:
        vmax = float(np.nanmax(val)) if len(val) > 0 else 1.0

    if abs(vmax - vmin) < 1e-7:
        norm = np.zeros_like(val)
    else:
        norm = np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0)

    # 4分割 Jet カラーランプ
    r = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=-1) * 255.0
    return rgb.astype(np.uint8)


def compute_velocity_colors(velocities: np.ndarray, vmin: float = 0.0, vmax: Optional[float] = None) -> np.ndarray:
    """頂点移動速度ノルム ||v|| の Jet ヒートマップ頂点カラー [N, 3] を計算します"""
    vel = np.asarray(velocities, dtype=np.float32)
    if vel.ndim == 2 and vel.shape[1] == 3:
        speeds = np.linalg.norm(vel, axis=1)
    else:
        speeds = vel
    return colormap_jet(speeds, vmin=vmin, vmax=vmax)


def compute_strain_colors(positions: np.ndarray, edges: np.ndarray, rest_lengths: np.ndarray, max_strain: float = 0.2) -> np.ndarray:
    """
    拘束エッジ伸長率 (L - L0) / L0 のテンションカラーマップ [N, 3] を計算します。
    圧縮（縮み）: 青 -> 自然長 (0): 緑/白 -> 引張（伸び）: 赤
    """
    pos = np.asarray(positions, dtype=np.float32)
    edg = np.asarray(edges, dtype=np.uint32)
    rl = np.asarray(rest_lengths, dtype=np.float32)
    n_verts = len(pos)
    if len(edg) == 0 or len(rl) == 0:
        return np.full((n_verts, 3), 255, dtype=np.uint8)

    p0 = pos[edg[:, 0]]
    p1 = pos[edg[:, 1]]
    curr_lens = np.linalg.norm(p0 - p1, axis=1)
    strain = np.where(rl > 1e-6, (curr_lens - rl) / rl, 0.0)

    vert_strain = np.zeros(n_verts, dtype=np.float32)
    vert_counts = np.zeros(n_verts, dtype=np.float32)
    np.add.at(vert_strain, edg[:, 0], strain)
    np.add.at(vert_strain, edg[:, 1], strain)
    np.add.at(vert_counts, edg[:, 0], 1.0)
    np.add.at(vert_counts, edg[:, 1], 1.0)
    avg_strain = np.where(vert_counts > 0, vert_strain / vert_counts, 0.0)

    norm = np.clip((avg_strain + max_strain) / (2.0 * max_strain), 0.0, 1.0)
    r = np.where(norm < 0.5, norm * 2.0, 1.0)
    g = np.where(norm < 0.5, norm * 2.0, (1.0 - norm) * 2.0)
    b = np.where(norm < 0.5, 1.0, (1.0 - norm) * 2.0)
    rgb = np.stack([r, g, b], axis=-1) * 255.0
    return rgb.astype(np.uint8)


def compute_normal_colors(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """面法線ベクトル (Nx, Ny, Nz) を直接 RGB に写像したカラーマップ [N, 3] を計算します"""
    pos = np.asarray(positions, dtype=np.float32)
    fcs = np.asarray(faces, dtype=np.uint32)
    n_verts = len(pos)
    if len(fcs) == 0:
        return np.full((n_verts, 3), 255, dtype=np.uint8)

    p0 = pos[fcs[:, 0]]
    p1 = pos[fcs[:, 1]]
    p2 = pos[fcs[:, 2]]
    face_norms = np.cross(p1 - p0, p2 - p0)

    vert_norms = np.zeros((n_verts, 3), dtype=np.float32)
    np.add.at(vert_norms, fcs[:, 0], face_norms)
    np.add.at(vert_norms, fcs[:, 1], face_norms)
    np.add.at(vert_norms, fcs[:, 2], face_norms)

    lens = np.linalg.norm(vert_norms, axis=1, keepdims=True)
    unit_norms = np.where(lens > 1e-6, vert_norms / lens, np.array([0.0, 0.0, 1.0], dtype=np.float32))

    rgb = (unit_norms + 1.0) * 0.5 * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def generate_sdf_bbox_lines(
    bone_infos: Sequence[dict],
    bone_transforms: Sequence[dict],
    color: Tuple[int, int, int] = (255, 170, 0)
) -> Optional[np.ndarray]:
    """各ボーンSDFの有効範囲AABB（直方体12辺）を3Dライン配列 [E, 9] に変換します"""
    lines = []
    c_r, c_g, c_b = float(color[0]), float(color[1]), float(color[2])

    for info, trans in zip(bone_infos, bone_transforms):
        aabb_min = np.array(info.get("aabb_min", [0, 0, 0])[:3], dtype=np.float32)
        aabb_max = np.array(info.get("aabb_max", [0, 0, 0])[:3], dtype=np.float32)
        world_m = np.array(trans.get("world_matrix", np.eye(4)), dtype=np.float32).reshape((4, 4))

        corners_local = np.array([
            [aabb_min[0], aabb_min[1], aabb_min[2], 1.0],
            [aabb_max[0], aabb_min[1], aabb_min[2], 1.0],
            [aabb_max[0], aabb_max[1], aabb_min[2], 1.0],
            [aabb_min[0], aabb_max[1], aabb_min[2], 1.0],
            [aabb_min[0], aabb_min[1], aabb_max[2], 1.0],
            [aabb_max[0], aabb_min[1], aabb_max[2], 1.0],
            [aabb_max[0], aabb_max[1], aabb_max[2], 1.0],
            [aabb_min[0], aabb_max[1], aabb_max[2], 1.0],
        ], dtype=np.float32)

        corners_world = (corners_local @ world_m.T)[:, :3]

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        for i0, i1 in edges:
            p0 = corners_world[i0]
            p1 = corners_world[i1]
            lines.append([p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], c_r, c_g, c_b])

    if lines:
        return np.array(lines, dtype=np.float32)
    return None
