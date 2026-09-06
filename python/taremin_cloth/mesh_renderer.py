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


def _prepare_scene_arrays(
    positions: np.ndarray,
    faces: np.ndarray,
    sewing_springs: Optional[np.ndarray] = None,
    mesh_colliders: Optional[np.ndarray] = None,
    vertex_colors: Optional[np.ndarray] = None,
    extra_lines: Optional[np.ndarray] = None,
    voxels: Optional[np.ndarray] = None,
):
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

    vox = None
    if voxels is not None and len(voxels) > 0:
        vox = np.ascontiguousarray(voxels, dtype=np.float32)

    return pos, fcs, sew, cols, vc, lines, vox


def render_scene_to_file(
    filepath: str,
    positions: np.ndarray,
    faces: np.ndarray,
    sewing_springs: Optional[np.ndarray] = None,
    mesh_colliders: Optional[np.ndarray] = None,
    vertex_colors: Optional[np.ndarray] = None,
    extra_lines: Optional[np.ndarray] = None,
    voxels: Optional[np.ndarray] = None,
    width: int = 800,
    height: int = 600,
    camera_pos: Optional[Tuple[float, float, float]] = None,
    camera_target: Optional[Tuple[float, float, float]] = None,
    fov: float = 45.0,
    draw_wireframe: bool = True,
    wire_width: float = 1.0,
    wireframe_only: bool = False,
    wire_color: Optional[Tuple[int, int, int]] = None,
    collider_color: Optional[Tuple[int, int, int]] = None,
    sewing_color: Optional[Tuple[int, int, int]] = None,
    voxel_screen_size: Optional[float] = None,
    **kwargs
) -> Tuple[str, int]:
    """
    シーン（布メッシュ、縫合エッジ、コライダー、頂点カラー、追加3D線分、ボクセル）を
    Rust内蔵のソフトウェアラスタライザで描画し、PNG画像として保存します。
    """
    pos, fcs, sew, cols, vc, lines, vox = _prepare_scene_arrays(
        positions, faces, sewing_springs, mesh_colliders, vertex_colors, extra_lines, voxels
    )

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
            wireframe_only=bool(wireframe_only),
            wire_color=list(wire_color) if wire_color is not None else None,
            collider_color=list(collider_color) if collider_color is not None else None,
            sewing_color=list(sewing_color) if sewing_color is not None else None,
            vertex_colors=vc,
            extra_lines=lines,
            voxels=vox,
            voxel_screen_size=float(voxel_screen_size) if voxel_screen_size is not None else None,
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


def render_scene_to_image(
    positions: np.ndarray,
    faces: np.ndarray,
    sewing_springs: Optional[np.ndarray] = None,
    mesh_colliders: Optional[np.ndarray] = None,
    vertex_colors: Optional[np.ndarray] = None,
    extra_lines: Optional[np.ndarray] = None,
    voxels: Optional[np.ndarray] = None,
    width: int = 800,
    height: int = 600,
    camera_pos: Optional[Tuple[float, float, float]] = None,
    camera_target: Optional[Tuple[float, float, float]] = None,
    fov: float = 45.0,
    draw_wireframe: bool = True,
    wire_width: float = 1.0,
    wireframe_only: bool = False,
    wire_color: Optional[Tuple[int, int, int]] = None,
    collider_color: Optional[Tuple[int, int, int]] = None,
    sewing_color: Optional[Tuple[int, int, int]] = None,
    voxel_screen_size: Optional[float] = None,
    **kwargs
):
    """
    シーンをメモリ上でラスタライズし、ディスクI/Oなしで直接 PIL.Image を生成して返却します。
    """
    if not _HAS_PIL:
        raise RuntimeError("アニメーション生成や画像直接取得には PIL (Pillow) が必要です。")

    pos, fcs, sew, cols, vc, lines, vox = _prepare_scene_arrays(
        positions, faces, sewing_springs, mesh_colliders, vertex_colors, extra_lines, voxels
    )

    if _HAS_RUST_CORE and hasattr(taremin_cloth_core, "render_scene_to_rgb"):
        raw_bytes, _red_pixels = taremin_cloth_core.render_scene_to_rgb(
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
            wireframe_only=bool(wireframe_only),
            wire_color=list(wire_color) if wire_color is not None else None,
            collider_color=list(collider_color) if collider_color is not None else None,
            sewing_color=list(sewing_color) if sewing_color is not None else None,
            vertex_colors=vc,
            extra_lines=lines,
            voxels=vox,
            voxel_screen_size=float(voxel_screen_size) if voxel_screen_size is not None else None,
        )
        # raw RGB バッファから PIL.Image をコピー生成
        img = Image.frombytes("RGB", (width, height), bytes(raw_bytes))
        return img
    else:
        # フォールバック: 一時ファイル経由
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            render_scene_to_file(
                tmp_path, positions, faces,
                sewing_springs=sewing_springs, mesh_colliders=mesh_colliders,
                vertex_colors=vertex_colors, extra_lines=extra_lines, voxels=voxels,
                width=width, height=height,
                camera_pos=camera_pos, camera_target=camera_target, fov=fov,
                draw_wireframe=draw_wireframe, wire_width=wire_width,
                wireframe_only=wireframe_only, wire_color=wire_color,
                collider_color=collider_color, sewing_color=sewing_color,
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
    """PIL.Imageオブジェクトを返却する互換用ヘルパー。render_scene_to_image を直接呼び出します"""
    return render_scene_to_image(
        positions, faces,
        width=width, height=height,
        camera_pos=camera_pos, camera_target=camera_target, fov=fov,
        **kwargs
    )


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


def extract_sdf_surface_voxels(
    bake_res,
    alpha_threshold: float = 0.03,
    dist_factor: float = 0.75,
    stride: int = 1,
    adaptive: bool = True,
    target_spacing: Optional[float] = None,
) -> List[Tuple[str, np.ndarray, np.ndarray, Tuple[int, int, int]]]:
    """
    SDFベイク結果（BakeResult）の3Dテクスチャから、各ボーンの表面付近ボクセルを抽出します。

    Args:
        bake_res: BakeResult オブジェクト
        alpha_threshold: ボーン影響度ウェイト閾値
        dist_factor: 表面近傍判定ファクター (|eff_d| <= dist_factor * avg_spacing)
        stride: ボクセルサンプリング間隔 (adaptive=False 時の一律間隔、または adaptive=True 時の基準倍率)
        adaptive: True の場合、各軸のアスペクト比に応じて stride を個別に適応化（等方サンプリング）
        target_spacing: ワールド/ローカル空間での目標間隔 (m単位)。指定時は全軸・全ボーンでこの間隔を基準に適応サンプリング。

    Returns:
        List of (bone_name, local_pts [V, 3], voxel_size [3], rgb_color [3])
    """
    raw_tex = np.frombuffer(bake_res.texture_bytes, dtype=np.float16).reshape(
        (bake_res.depth, bake_res.height, bake_res.width, 2)
    )
    bone_names = bake_res.bone_names
    bone_infos = bake_res.bone_infos

    bone_voxels = []
    b_per_layer = 16

    for b_idx, bname in enumerate(bone_names):
        info = bone_infos[b_idx]
        local_min = np.array(info[0:3], dtype=np.float32)
        local_max = np.array(info[4:7], dtype=np.float32)
        res = int(info[7])
        if res <= 0:
            bone_voxels.append((bname, np.empty((0, 3), dtype=np.float32), np.zeros(3, dtype=np.float32), (100, 140, 180)))
            continue

        weight_th = max(float(info[19]), 0.01)

        box_size = local_max - local_min
        base_d = box_size / float(res)

        if adaptive:
            if target_spacing is not None and target_spacing > 0.0:
                st_x = max(1, int(round(target_spacing / base_d[0])))
                st_y = max(1, int(round(target_spacing / base_d[1])))
                st_z = max(1, int(round(target_spacing / base_d[2])))
            else:
                target_pitch = float(np.max(base_d)) * float(max(1, int(stride)))
                st_x = max(1, int(round(target_pitch / base_d[0])))
                st_y = max(1, int(round(target_pitch / base_d[1])))
                st_z = max(1, int(round(target_pitch / base_d[2])))
        else:
            st = max(1, int(stride))
            st_x = st_y = st_z = st

        l_idx = b_idx // b_per_layer
        rem = b_idx % b_per_layer
        r_idx = rem // 4
        c_idx = rem % 4

        z0, z1 = l_idx * res, (l_idx + 1) * res
        y0, y1 = r_idx * res, (r_idx + 1) * res
        x0, x1 = c_idx * res, (c_idx + 1) * res

        slot_tex = raw_tex[z0:z1:st_z, y0:y1:st_y, x0:x1:st_x, :]
        raw_d = slot_tex[:, :, :, 0].astype(np.float32)
        alpha = slot_tex[:, :, :, 1].astype(np.float32)

        t = np.clip(alpha / weight_th, 0.0, 1.0)
        mask = t * t * (3.0 - 2.0 * t)
        eff_d = (1.0 - mask) * 10.0 + mask * raw_d

        eff_vox_size = base_d * np.array([st_x, st_y, st_z], dtype=np.float32)
        avg_spacing = float(np.mean(eff_vox_size))

        surf_mask = (np.abs(eff_d) <= avg_spacing * dist_factor) & (alpha > alpha_threshold)

        bname_l = bname.lower()
        if "hip" in bname_l:
            b_color = (255, 210, 40) # イエローゴールド
        elif "leg" in bname_l or "thigh" in bname_l or "foot" in bname_l:
            b_color = (60, 220, 120) # エメラルドグリーン
        elif "spine" in bname_l or "chest" in bname_l or "neck" in bname_l or "head" in bname_l:
            b_color = (0, 200, 255) # シアン
        elif "arm" in bname_l or "hand" in bname_l or "shoulder" in bname_l:
            b_color = (255, 100, 200) # マゼンタピンク
        else:
            b_color = (100, 140, 180) # スレートブルー

        if np.any(surf_mask):
            iz, iy, ix = np.where(surf_mask)
            fx = (ix.astype(np.float32) * st_x + 0.5) / float(res)
            fy = (iy.astype(np.float32) * st_y + 0.5) / float(res)
            fz = (iz.astype(np.float32) * st_z + 0.5) / float(res)

            px = local_min[0] + fx * box_size[0]
            py = local_min[1] + fy * box_size[1]
            pz = local_min[2] + fz * box_size[2]
            pts_loc = np.stack([px, py, pz], axis=-1)
            bone_voxels.append((bname, pts_loc, eff_vox_size, b_color))
        else:
            bone_voxels.append((bname, np.empty((0, 3), dtype=np.float32), eff_vox_size, b_color))

    return bone_voxels


def transform_sdf_voxels_to_world(
    bone_surface_voxels: List[Tuple[str, np.ndarray, np.ndarray, Tuple[int, int, int]]],
    bone_world_matrices: dict,
    scale_factor: float = 0.5
) -> Optional[np.ndarray]:
    """
    各ボーンのローカル表面ボクセルをボーンワールド行列で変換し、
    [N, 7] (x, y, z, size, r, g, b) のボクセル配列を生成します。
    """
    all_voxels = []
    for bname, pts_loc, vox_sz, b_color in bone_surface_voxels:
        if len(pts_loc) == 0:
            continue
        mat = bone_world_matrices.get(bname)
        if mat is None:
            continue

        mat4 = np.asarray(mat, dtype=np.float32).reshape((4, 4))
        pts_homo = np.hstack([pts_loc, np.ones((len(pts_loc), 1), dtype=np.float32)])
        pts_w = (pts_homo @ mat4.T)[:, :3]

        # ボクセルの実効サイズ（立方体の一辺）
        v_size = float(np.mean(vox_sz)) * scale_factor
        n = len(pts_w)

        vox_arr = np.empty((n, 7), dtype=np.float32)
        vox_arr[:, 0:3] = pts_w
        vox_arr[:, 3] = v_size
        vox_arr[:, 4] = b_color[0]
        vox_arr[:, 5] = b_color[1]
        vox_arr[:, 6] = b_color[2]
        all_voxels.append(vox_arr)

    if all_voxels:
        return np.vstack(all_voxels)
    return None


def extract_joint_mesh_wireframe_lines(
    mesh_verts: np.ndarray,
    mesh_tris: np.ndarray,
    face_indices: np.ndarray,
    color: Tuple[float, float, float] = (255.0, 255.0, 255.0),
) -> Optional[np.ndarray]:
    """
    指定された三角形面群から重複エッジを排除し、
    3Dワイヤーフレーム線分配列 [E, 9] (x0, y0, z0, x1, y1, z1, r, g, b) を生成します。

    Args:
        mesh_verts: [V, 3] 頂点座標配列
        mesh_tris: [F, 3] 三角形インデックス配列
        face_indices: [M] ワイヤーフレーム化する面の行インデックス配列
        color: 線のRGB色 (デフォルト: 白色 [255, 255, 255])

    Returns:
        [E, 9] の 3D 線分配列 (線分が存在しない場合は None)
    """
    if len(mesh_verts) == 0 or len(mesh_tris) == 0 or len(face_indices) == 0:
        return None

    n_tris = len(mesh_tris)
    valid_idx = face_indices[face_indices < n_tris]
    if len(valid_idx) == 0:
        return None

    sub_tris = mesh_tris[valid_idx]

    # 重複エッジの排除
    edges = set()
    for t in sub_tris:
        edges.add((min(t[0], t[1]), max(t[0], t[1])))
        edges.add((min(t[1], t[2]), max(t[1], t[2])))
        edges.add((min(t[2], t[0]), max(t[2], t[0])))

    if not edges:
        return None

    lines = []
    c0, c1, c2 = float(color[0]), float(color[1]), float(color[2])
    for e0, e1 in edges:
        p0 = mesh_verts[e0]
        p1 = mesh_verts[e1]
        lines.append([p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], c0, c1, c2])

    return np.array(lines, dtype=np.float32)


def filter_active_joint_faces(
    joint_faces_by_pair: dict,
    bone_rotations: dict,
    rotation_threshold_deg: float = 2.0,
) -> np.ndarray:
    """
    各ボーンのローカル回転角（クォータニオン [w, x, y, z]）から屈曲角を算出し、
    閾値(rotation_threshold_deg)以上屈曲しているボーンペアの関節面インデックスを動的に抽出します。

    Args:
        joint_faces_by_pair: {(parent, child): face_indices} の辞書
        bone_rotations: {bone_name: [w, x, y, z]} のクォータニオン回転辞書
        rotation_threshold_deg: アクティブ化する最小屈曲角（度）。0.0 以下の場合は全関節面を返却。

    Returns:
        アクティブな面インデックスのユニオン配列 (shape [M], dtype int32)
    """
    if not joint_faces_by_pair:
        return np.empty(0, dtype=np.int32)

    if rotation_threshold_deg <= 0.0:
        return np.unique(np.concatenate(list(joint_faces_by_pair.values()))).astype(np.int32)

    import math
    rot_th_rad = math.radians(rotation_threshold_deg)
    active_faces = []

    for (p_name, c_name), faces in joint_faces_by_pair.items():
        q = bone_rotations.get(c_name)
        if q is None or len(q) < 4:
            continue

        w = float(q[0])
        delta_angle = 2.0 * math.acos(min(abs(w), 1.0))
        if delta_angle >= rot_th_rad:
            active_faces.append(faces)

    if active_faces:
        return np.unique(np.concatenate(active_faces)).astype(np.int32)
    return np.empty(0, dtype=np.int32)

