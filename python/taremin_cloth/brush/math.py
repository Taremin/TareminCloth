"""
ブラシ純粋ロジック (taremin_cloth.brush.math)
Blender (bpy) 非依存。NumPyのみで動作する。
中心強度1・周縁へ減衰する重み付けはBlenderスカルプトのFalloff presetと同名・同系統。
"""

import numpy as np

BRUSH_RADIUS_MIN = 0.005
BRUSH_RADIUS_MAX = 0.5
BRUSH_STRENGTH_MIN = 0.01
BRUSH_STRENGTH_MAX = 1.0

FALLOFF_SHAPES = ('SMOOTH', 'SPHERE', 'SHARP', 'LINEAR', 'CONSTANT')


def falloff_weights(distances, radius, shape='SMOOTH'):
    """距離から減衰重み (0..1) を求める。中心=1、半径外=0"""
    d = np.asarray(distances, dtype=np.float32)
    r = max(float(radius), 1e-9)
    t = np.clip(1.0 - d / r, 0.0, 1.0).astype(np.float32)
    if shape == 'LINEAR':
        return t
    if shape == 'SPHERE':
        return np.sqrt(np.maximum(0.0, 1.0 - (1.0 - t) ** 2)).astype(np.float32)
    if shape == 'SHARP':
        return (t * t).astype(np.float32)
    if shape == 'CONSTANT':
        return np.where(d <= r, 1.0, 0.0).astype(np.float32)
    # SMOOTH (既定): smoothstep
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def verts_in_brush(vert_positions, brush_center, radius, shape='SMOOTH'):
    """ブラシ球内の頂点を列挙する。戻り値は (indices, weights)"""
    pos = np.asarray(vert_positions, dtype=np.float32)
    c = np.asarray(brush_center, dtype=np.float32).reshape(3)
    if len(pos) == 0:
        return np.empty((0,), dtype=np.uint32), np.empty((0,), dtype=np.float32)
    dists = np.linalg.norm(pos - c, axis=1)
    mask = dists <= float(radius)
    found = np.where(mask)[0].astype(np.uint32)
    if len(found) == 0:
        return found, np.empty((0,), dtype=np.float32)
    return found, falloff_weights(dists[mask], radius, shape)


def depth_keep_mask(world_points, origin, direction, hit_t, eps):
    """レイ命中面より奥の点を除外するマスク。レイ外れ (hit_t=None) 時は全て残す"""
    if hit_t is None:
        return np.ones(len(np.asarray(world_points)), dtype=bool)
    p = np.asarray(world_points, dtype=np.float32).reshape(-1, 3)
    o = np.asarray(origin, dtype=np.float32).reshape(3)
    d = np.asarray(direction, dtype=np.float32).reshape(3)
    t = (p - o) @ d
    return t <= float(hit_t) + float(eps)


def select_grab_pins(found_indices, falloff_weights_, threshold=0.01):
    """グラブ対象の選別: 閾値未満の裾野はピン留めせず物理に任せる。

    全頂点を weight=1.0 で固定すると裾野リングが運動学的壁となり、
    ブラシ外の布への伝播を遮断してしまう。減衰値をそのまま
    ピンウェイト (連続質量ブレンド + コンプライアント拘束で力比べ) とする。
    """
    idx = np.asarray(found_indices).tolist()
    w = np.asarray(falloff_weights_, dtype=np.float32).tolist()
    out_idx, out_w = [], []
    for i, weight in zip(idx, w):
        weight = max(0.0, min(1.0, float(weight)))
        if weight >= float(threshold):
            out_idx.append(int(i))
            out_w.append(weight)
    return out_idx, out_w


def radial_adjust(start_value, dx_px, min_value, max_value, px_scale=200.0):
    """ラジアル操作の相対式: 開始値×(1+dx/px_scale)を範囲へクリップする"""
    try:
        v = float(start_value) * (1.0 + float(dx_px) / max(float(px_scale), 1.0))
    except (TypeError, ValueError):
        v = float(min_value)
    return max(float(min_value), min(float(max_value), v))
