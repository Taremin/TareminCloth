"""
taremin_cloth 自己衝突パラメータ自動フィッティングユーティリティ
メッシュのエッジ長スケール・頂点密度、および布の用途（一般服、スカート、薄手など）に基づき、
破綻や自己交差、過剰なエッジ伸びを防ぐ最適な自己衝突パラメータを算出・適用する。
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from .args_guard import allow_positional


SELF_COLLISION_PURPOSE_ITEMS = [
    ('STANDARD', "Standard (一般衣服)", "シャツ、ズボン、ワンピース等の標準的な布地向け（負荷と安定性のバランス）"),
    ('SKIRT', "Skirt / Folds (プリーツ・多重折り)", "スカート、フリル、リボン等、布が密集して重なり合う形状向け（高精度・伸び抑制）"),
    ('THIN', "Thin / Delicate (薄手・シルク)", "スカーフ、シルク、極薄の布地向け（マイルドな反発で破裂防止）"),
    ('CUSTOM', "Custom (手動設定)", "詳細モードで自由にパラメータを微調整するモード"),
]


def calculate_mesh_edge_stats(obj) -> Optional[Tuple[float, float, int]]:
    """
    オブジェクトのワールド変換を考慮した平均エッジ長、最小エッジ長、頂点数を計算する。
    返り値: (avg_edge_length, min_edge_length, num_vertices) または None
    """
    if not obj or getattr(obj, "type", None) != 'MESH':
        return None

    mesh = getattr(obj, "data", None)
    if not mesh:
        return None

    edges = getattr(mesh, "edges", [])
    vertices = getattr(mesh, "vertices", [])
    if len(edges) == 0 or len(vertices) == 0:
        return None

    # ワールドスケールの考慮
    mat = getattr(obj, "matrix_world", None)

    # 面を構成する頂点集合が取得できる場合は、縫合用ルーズエッジ（長距離の接続線）を除外する
    polygons = getattr(mesh, "polygons", None)
    face_vert_set = None
    if polygons is not None and len(polygons) > 0:
        face_vert_set = set()
        for poly in polygons:
            for vid in getattr(poly, "vertices", ()):
                face_vert_set.add(int(vid))

    edge_lengths = []
    valid_edges = []
    n_verts = len(vertices)
    for e in edges:
        if getattr(e, "is_loose", False):
            continue
        v0_idx = int(e.vertices[0])
        v1_idx = int(e.vertices[1])
        if face_vert_set is not None and (v0_idx not in face_vert_set or v1_idx not in face_vert_set):
            continue
        if v0_idx < n_verts and v1_idx < n_verts:
            co0 = vertices[v0_idx].co
            co1 = vertices[v1_idx].co
            if mat is not None:
                # mathutils.Matrix または numpy 配列の乗算をサポート
                try:
                    p0 = mat @ co0
                    p1 = mat @ co1
                    length = (p1 - p0).length
                except (TypeError, AttributeError):
                    # モック等で @ 演算子が使えない場合のフォールバック
                    p0 = np.array(co0)
                    p1 = np.array(co1)
                    length = float(np.linalg.norm(p1 - p0))
            else:
                p0 = np.array(co0)
                p1 = np.array(co1)
                length = float(np.linalg.norm(p1 - p0))

            edge_lengths.append(length)
            valid_edges.append((v0_idx, v1_idx, length))

    if not edge_lengths:
        return None

    avg_len = float(np.mean(edge_lengths))
    # 外れ値（襟元やボタン穴等の極小エッジ）に引きずられないよう、
    # エッジ数が十分（20本以上）ある場合は下位5パーセンタイル値を実効最小エッジ長として採用する。
    # さらに複数アイランド（身頃＋高密度なフード等）で構成される場合は、20本以上のエッジを持つ
    # 各アイランドの下位5パーセンタイル値の最小値を採用し、局所的な高密度パーツの3ホップ干渉を防ぐ。
    if len(edge_lengths) >= 20:
        min_len = float(np.percentile(edge_lengths, 5))
        parent = list(range(n_verts))

        def _find(x: int) -> int:
            r = x
            while parent[r] != r:
                r = parent[r]
            curr = x
            while curr != r:
                nxt = parent[curr]
                parent[curr] = r
                curr = nxt
            return r

        for u, v, _ in valid_edges:
            ru, rv = _find(u), _find(v)
            if ru != rv:
                parent[rv] = ru

        island_edges: Dict[int, list] = {}
        for u, _, length in valid_edges:
            root = _find(u)
            island_edges.setdefault(root, []).append(length)

        island_p5 = [
            float(np.percentile(lengths, 5))
            for lengths in island_edges.values()
            if len(lengths) >= 20
        ]
        if island_p5:
            min_len = min(min_len, min(island_p5))
    else:
        min_len = float(np.min(edge_lengths))
    return avg_len, min_len, n_verts


# エッジ統計量キャッシュ辞書: key=(id(obj), id(mesh), n_verts, n_edges, scale_tuple)
_EDGE_STATS_CACHE: Dict[Tuple, Tuple[float, float, int]] = {}


def clear_edge_stats_cache() -> None:
    """エッジ統計量のメモリキャッシュを明示的にクリアする"""
    _EDGE_STATS_CACHE.clear()


def get_cached_mesh_edge_stats(obj) -> Optional[Tuple[float, float, int]]:
    """
    オブジェクトのメッシュエッジ統計量（平均エッジ長、最小エッジ長、頂点数）をキャッシュ付きで取得する。
    UIの描画ループ（draw）で毎フレーム数千〜数万エッジの再計算が発生するのを防ぐ。
    """
    if not obj or getattr(obj, "type", None) != 'MESH':
        return None

    mesh = getattr(obj, "data", None)
    if not mesh:
        return None

    edges = getattr(mesh, "edges", None)
    vertices = getattr(mesh, "vertices", None)
    if edges is None or vertices is None:
        return None

    n_edges = len(edges)
    n_verts = len(vertices)
    if n_edges == 0 or n_verts == 0:
        return None

    # スケール情報をキャッシュキーに含める
    scale = getattr(obj, "scale", None)
    if scale is not None:
        try:
            scale_tuple = (round(float(scale[0]), 5), round(float(scale[1]), 5), round(float(scale[2]), 5))
        except (IndexError, TypeError, ValueError):
            scale_tuple = (1.0, 1.0, 1.0)
    else:
        scale_tuple = (1.0, 1.0, 1.0)

    cache_key = (id(obj), id(mesh), n_verts, n_edges, scale_tuple)
    if cache_key in _EDGE_STATS_CACHE:
        return _EDGE_STATS_CACHE[cache_key]

    stats = calculate_mesh_edge_stats(obj)
    if stats is not None:
        # キャッシュサイズ上限（肥大化防止）
        if len(_EDGE_STATS_CACHE) > 100:
            _EDGE_STATS_CACHE.clear()
        _EDGE_STATS_CACHE[cache_key] = stats

    return stats



@allow_positional
def compute_self_collision_params(
    avg_edge_len: float,
    min_edge_len: float,
    num_vertices: int,
    *,
    purpose: str = 'STANDARD',
) -> Dict[str, Any]:
    """
    エッジ長と用途から、推奨される自己衝突パラメータ辞書を算出する。
    """
    if purpose == 'SKIRT':
        # プリーツスカート・多重折り:
        # 密集したヒダ同士が挟まっても過度な膨らみを起こさないよう厚みはやや薄め（15%）
        # トンネリングを防ぎつつ弾けを抑えるため relief は 0.8、FULL_COUPLED で伸びを7割抑制
        min_floor = 0.0010  # 1.0mm
        max_cap = 0.0080    # 8.0mm
        thickness = max(min_floor, min(max_cap, avg_edge_len * 0.15))
        relief_factor = 0.80
        max_displacement_ratio = 0.30
        max_iterations = '512'
        coupled_mode = 'FULL_COUPLED'
        relaxation_iters = 2
    elif purpose == 'THIN':
        # 薄手・シルク・フリル:
        # 極薄布向けに厚み 10%（最小 0.8mm）、トンネリング防止と滑らかな反発を両立
        min_floor = 0.0008  # 0.8mm
        max_cap = 0.0050    # 5.0mm
        thickness = max(min_floor, min(max_cap, avg_edge_len * 0.10))
        relief_factor = 0.80
        max_displacement_ratio = 0.30
        max_iterations = '256'
        coupled_mode = 'OFF'
        relaxation_iters = 0
    else:
        # STANDARD (一般の衣服) およびデフォルト:
        # 平均エッジ長の 20%（最小 1.5mm、最大 10mm）
        # 物理質量化に伴いシワ自縛ロック防止のため Coupled は OFF、
        # トンネリングトラップ（貫通脱出不能）防止のため relief=1.0, max_ratio=0.5 が標準
        min_floor = 0.0015  # 1.5mm
        max_cap = 0.0100    # 10.0mm
        thickness = max(min_floor, min(max_cap, avg_edge_len * 0.20))
        relief_factor = 1.00
        max_displacement_ratio = 0.50
        max_iterations = '512' if num_vertices >= 2000 else '256'
        coupled_mode = 'OFF'
        relaxation_iters = 0

    # 安全策: 最小エッジ長の40%を超えないようにクランプ（自縄自縛・初期埋没防止）
    # ただし、襟元等の極小外れ値（平均エッジ長の20%未満）によって用途ごとの下限フロア (min_floor) を
    # 破壊しないよう保護しつつ、フード等の正当な高密度領域（min_edge_len >= avg_edge_len * 0.20）では
    # 3ホップ初期干渉防止の絶対下限 (0.8mm) まで min_edge_len * 0.4 クランプを優先する
    if min_edge_len > 0:
        if avg_edge_len > 0 and min_edge_len < avg_edge_len * 0.20:
            effective_floor = min_floor
        else:
            effective_floor = min(min_floor, 0.0008)
        safe_clamped = max(effective_floor, min_edge_len * 0.4)
        thickness = min(thickness, safe_clamped)

    return {
        'thickness': thickness,
        'self_collision_relief_factor': relief_factor,
        'self_collision_max_displacement_ratio': max_displacement_ratio,
        'self_collision_max_iterations': max_iterations,
        'coupled_self_collision_mode': coupled_mode,
        'post_collision_relaxation_iters': relaxation_iters,
        'enable_normal_untangling': True,
    }


def fit_self_collision_for_object(obj, *, settings=None, purpose: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    オブジェクトのメッシュを解析し、settings に最適な自己衝突パラメータを適用する。
    適用されたパラメータ辞書を返す（失敗時は None）。
    """
    if settings is None:
        settings = getattr(obj, "taremin_cloth", None)
    if not settings:
        return None

    target_purpose = purpose or getattr(settings, "self_collision_purpose", 'STANDARD')
    if target_purpose == 'CUSTOM':
        return None

    stats = calculate_mesh_edge_stats(obj)
    if not stats:
        # エッジ情報が取れない場合のデフォルト値フォールバック
        avg_len = 0.05
        min_len = 0.01
        n_verts = 100
    else:
        avg_len, min_len, n_verts = stats

    params = compute_self_collision_params(avg_len, min_len, n_verts, purpose=target_purpose)

    # settings に適用
    for k, v in params.items():
        if hasattr(settings, k):
            setattr(settings, k, v)

    return params


def get_thickness_warning_info(obj, *, settings=None) -> Optional[Dict[str, Any]]:
    """
    自己衝突が有効な布オブジェクトに対して、厚み（thickness）がメッシュのエッジ長に対して
    過大（接触判定の自爆反発や高周波ジッター・暴れの原因）になっていないかを判定する。

    警告がある場合は警告レベル ('DANGER' / 'WARNING') や推奨安全上限厚みなどの情報を辞書で返す。
    自己衝突が無効な場合やメッシュ情報が取得できない場合は None を返す。
    """
    if settings is None:
        settings = getattr(obj, "taremin_cloth", None)
    if not settings:
        return None

    # 布オブジェクトでない、または自己衝突が無効な場合は対象外
    is_cloth = getattr(settings, "is_cloth", True)
    enable_self_collision = getattr(settings, "enable_self_collision", False)
    if not is_cloth or not enable_self_collision:
        return None

    stats = get_cached_mesh_edge_stats(obj)
    if not stats:
        return None

    avg_edge_len, min_edge_len, num_vertices = stats
    thickness = float(getattr(settings, "thickness", 0.005) or 0.005)

    # 接触判定球の直径は 2 * thickness。
    # 3ホップ離れた頂点との自己衝突誤爆および距離拘束との競合による高周波発振を防ぐため、
    # 推奨安全上限は min(min_edge * 0.40, avg_edge * 0.25)。
    # 暴れ（激しいジッター・発散）の危険閾値は min(min_edge * 0.45, avg_edge * 0.30)。
    safe_max_thick = max(0.0008, min(min_edge_len * 0.40, avg_edge_len * 0.25))
    danger_thick = max(0.0010, min(min_edge_len * 0.45, avg_edge_len * 0.30))

    if thickness > danger_thick:
        level = 'DANGER'
        has_warning = True
    elif thickness > safe_max_thick:
        level = 'WARNING'
        has_warning = True
    else:
        level = 'OK'
        has_warning = False

    return {
        'has_warning': has_warning,
        'level': level,
        'thickness': thickness,
        'max_safe_thickness': safe_max_thick,
        'avg_edge_len': avg_edge_len,
        'min_edge_len': min_edge_len,
        'num_vertices': num_vertices,
    }


def draw_thickness_warning_ui(layout, obj, *, settings=None, compact: bool = False) -> bool:
    """
    自己衝突の厚み過大警告UIを描画する。
    警告ボックスを描画した場合は True、警告がない場合は False を返す。
    """
    from .. import i18n
    info = get_thickness_warning_info(obj, settings=settings)
    if not info or not info.get('has_warning'):
        return False

    is_danger = (info['level'] == 'DANGER')
    rec_mm = info['max_safe_thickness'] * 1000.0
    cur_mm = info['thickness'] * 1000.0
    avg_mm = info['avg_edge_len'] * 1000.0

    box = layout.box()
    box.alert = is_danger
    col = box.column(align=True)

    if compact:
        row = col.row(align=True)
        if is_danger:
            row.label(text=i18n.trans("Self-collision jitter risk!"), icon='ERROR')
        else:
            row.label(text=i18n.trans("Thickness may cause jitter."), icon='INFO')
        row.operator("taremin_cloth.auto_fit_self_collision", text=i18n.trans("Auto Fit"), icon='FIXED_SIZE')
        row_sub = col.row(align=True)
        row_sub.label(text=f"{i18n.trans('Current:')} {cur_mm:.1f}mm <= {rec_mm:.1f}mm")
    else:
        row_title = col.row(align=True)
        if is_danger:
            row_title.label(text=i18n.trans("Self-collision jitter risk!"), icon='ERROR')
        else:
            row_title.label(text=i18n.trans("Thickness may cause jitter."), icon='INFO')

        col.label(text=i18n.trans("Thickness is too large for mesh edge length."))

        row_stat = col.row(align=True)
        row_stat.label(text=f"{i18n.trans('Current:')} {cur_mm:.1f}mm ({i18n.trans('Avg Edge:')} {avg_mm:.1f}mm)")

        row_btn = col.row(align=True)
        row_btn.label(text=f"{i18n.trans('Recommended:')} <= {rec_mm:.1f} mm")
        row_btn.operator("taremin_cloth.auto_fit_self_collision", text=i18n.trans("Auto Fit"), icon='FIXED_SIZE')

    return True

