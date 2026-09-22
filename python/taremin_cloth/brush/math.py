"""
ブラシ純粋ロジック (taremin_cloth.brush.math)
Blender (bpy) 非依存。NumPyのみで動作する。
中心強度1・周縁へ減衰する重み付けはBlenderスカルプトのFalloff presetと同名・同系統。
"""

import math

import numpy as np

BRUSH_RADIUS_MIN = 0.005
BRUSH_RADIUS_MAX = 0.5
BRUSH_STRENGTH_MIN = 0.01
BRUSH_STRENGTH_MAX = 1.0

FALLOFF_SHAPES = ('SMOOTH', 'SPHERE', 'SHARP', 'LINEAR', 'CONSTANT')


def _clamp01_finite(v):
    """0..1クランプ。非数値・非有限値は 0.0 (Rust核と同一仕様)"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return max(0.0, min(1.0, f))


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


def build_adjacency_csr(num_vertices, edges):
    """無向辺配列から1-ring隣接CSRを構築する。戻り値は (offsets, indices)。

    offsets: shape [N+1] uint32、各頂点の隣接開始位置。
    indices: shape [2E] uint32、頂点毎に昇順ソート済み (決定性・将来のRust parity用)。
    範囲外の辺は防御的に除外する。
    """
    n = int(num_vertices)
    if n <= 0:
        return (np.zeros((1,), dtype=np.uint32), np.empty((0,), dtype=np.uint32))
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2) if edges is not None else np.empty((0, 2), dtype=np.int64)
    if len(e) > 0:
        valid = (e[:, 0] >= 0) & (e[:, 0] < n) & (e[:, 1] >= 0) & (e[:, 1] < n) & (e[:, 0] != e[:, 1])
        e = e[valid]
    if len(e) == 0:
        return (np.zeros((n + 1,), dtype=np.uint32), np.empty((0,), dtype=np.uint32))
    flat = np.concatenate([e[:, 0], e[:, 1]])
    degrees = np.bincount(flat, minlength=n).astype(np.int64)
    offsets = np.zeros((n + 1,), dtype=np.int64)
    np.cumsum(degrees, out=offsets[1:])
    indices = np.empty((int(offsets[n]),), dtype=np.int64)
    cursor = offsets[:-1].copy()
    for a, b in e.tolist():
        indices[cursor[a]] = b
        cursor[a] += 1
        indices[cursor[b]] = a
        cursor[b] += 1
    # 頂点毎に昇順ソート (決定性確保)
    for v in range(n):
        s, t = int(offsets[v]), int(offsets[v + 1])
        if t - s > 1:
            indices[s:t] = np.sort(indices[s:t])
    return (offsets.astype(np.uint32), indices.astype(np.uint32))


def laplacian_smooth_targets(positions, adj_offsets, adj_indices,
                             brush_indices, brush_weights, strength, exclude=None):
    """1-ring平均への減衰ブレンド目標を計算する (Blender非依存)。

    target[v] = pos[v] + (mean(pos[nbrs(v)]) - pos[v]) * clamp(w * strength).
    孤立頂点・除外頂点は現位置をそのまま返す。除外集合はターゲット側のみに
    適用し、平均化の参照からは除かない (固定境界への馴染みを残す)。
    """
    pos = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    n = len(pos)
    idx = np.asarray(brush_indices).ravel()
    w = np.asarray(brush_weights, dtype=np.float32).ravel()
    if len(idx) == 0:
        return np.empty((0, 3), dtype=np.float32)
    if len(idx) != len(w):
        raise ValueError("brush_indices and brush_weights must have the same length")
    try:
        k_global = float(strength)
    except (TypeError, ValueError):
        k_global = 0.0
    if not np.isfinite(k_global):
        k_global = 0.0
    k_global = max(0.0, min(1.0, k_global))
    offsets = np.asarray(adj_offsets).ravel()
    nbrs_all = np.asarray(adj_indices).ravel()
    excluded = set()
    if exclude is not None:
        try:
            excluded = {int(v) for v in np.asarray(list(exclude)).ravel().tolist()}
        except TypeError:
            excluded = set()
    out = np.empty((len(idx), 3), dtype=np.float32)
    for n_i, (v_raw, w_raw) in enumerate(zip(idx.tolist(), w.tolist())):
        v = int(v_raw)
        if v < 0 or v >= n:
            raise ValueError(f"brush vertex index out of range: {v}")
        pv = pos[v]
        if v in excluded:
            out[n_i] = pv
            continue
        wv = _clamp01_finite(w_raw)
        k = wv * k_global
        if k <= 0.0 or len(offsets) != n + 1:
            out[n_i] = pv
            continue
        s, t = int(offsets[v]), int(offsets[v + 1])
        nbrs = nbrs_all[s:t] if 0 <= s <= t <= len(nbrs_all) else np.empty((0,), dtype=np.int64)
        nbrs = [int(u) for u in np.asarray(nbrs).tolist() if 0 <= int(u) < n]
        if not nbrs:
            out[n_i] = pv
            continue
        avg = np.mean(pos[np.asarray(nbrs, dtype=np.int64)], axis=0).astype(np.float32)
        out[n_i] = (pv + (avg - pv) * np.float32(k)).astype(np.float32)
    return out


# 平滑化1回分の外向き拡張ゲイン (弛み逃がし用)。strength・falloffでスケール
# されるため dwell しても弾性抵抗と釣り合って発散しない水準に抑える。
TENSION_EXPAND = 0.05


def radial_expand_targets(positions, center, brush_indices, brush_weights, strength,
                          expand=TENSION_EXPAND):
    """ブラシ中心からの放射方向へ目標をわずかに拡張する (張力アシスト)。

    t' = c + (t - c) * (1 + expand * w * strength)。中心一致・強度0は不変。
    シワの余剰長をパッチ外へ逃がし、リリース後の再座屈を抑える。
    """
    pos = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    n = len(pos)
    c = np.asarray(center, dtype=np.float32).reshape(3)
    idx = np.asarray(brush_indices).ravel()
    w = np.asarray(brush_weights, dtype=np.float32).ravel()
    if len(idx) == 0:
        return np.empty((0, 3), dtype=np.float32)
    if len(idx) != len(w):
        raise ValueError("brush_indices and brush_weights must have the same length")
    try:
        k_global = max(0.0, min(1.0, float(strength)))
    except (TypeError, ValueError):
        k_global = 0.0
    if not np.isfinite(k_global):
        k_global = 0.0
    try:
        gain = max(0.0, float(expand))
    except (TypeError, ValueError):
        gain = 0.0
    out = np.empty((len(idx), 3), dtype=np.float32)
    for n_i, (v_raw, w_raw) in enumerate(zip(idx.tolist(), w.tolist())):
        v = int(v_raw)
        if v < 0 or v >= n:
            raise ValueError(f"brush vertex index out of range: {v}")
        pv = pos[v]
        wv = _clamp01_finite(w_raw)
        k = gain * wv * k_global
        if k <= 0.0:
            out[n_i] = pv
            continue
        d = pv - c
        dist = float(np.linalg.norm(d))
        if dist <= 1e-9:
            out[n_i] = pv
            continue
        out[n_i] = (c + d * np.float32(1.0 + k)).astype(np.float32)
    return out
