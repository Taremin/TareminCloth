"""
taremin_cloth 自己衝突パラメータ自動フィッティングユーティリティ
メッシュのエッジ長スケール・頂点密度、および布の用途（一般服、スカート、薄手など）に基づき、
破綻や自己交差、過剰なエッジ伸びを防ぐ最適な自己衝突パラメータを算出・適用する。
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


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

    edge_lengths = []
    for e in edges:
        v0_idx = e.vertices[0]
        v1_idx = e.vertices[1]
        if v0_idx < len(vertices) and v1_idx < len(vertices):
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

    if not edge_lengths:
        return None

    avg_len = float(np.mean(edge_lengths))
    min_len = float(np.min(edge_lengths))
    return avg_len, min_len, len(vertices)


def compute_self_collision_params(
    avg_edge_len: float,
    min_edge_len: float,
    num_vertices: int,
    purpose: str = 'STANDARD',
) -> Dict[str, Any]:
    """
    エッジ長と用途から、推奨される自己衝突パラメータ辞書を算出する。
    """
    if purpose == 'SKIRT':
        # プリーツスカート・多重折り:
        # 密集したヒダ同士が挟まっても過度な膨らみを起こさないよう厚みはやや薄め（3.5%）
        # 急激な弾きによる破裂を防ぐため relief は 0.15、FULL_COUPLED で伸びを7割抑制
        thickness = max(0.0008, min(0.015, avg_edge_len * 0.035))
        relief_factor = 0.15
        max_displacement_ratio = 0.15
        max_iterations = '512'
        coupled_mode = 'FULL_COUPLED'
        relaxation_iters = 2
    elif purpose == 'THIN':
        # 薄手・シルク・フリル:
        # 極薄布向けに厚み 2.5%（最小 0.5mm）、マイルドな反発
        thickness = max(0.0005, min(0.008, avg_edge_len * 0.025))
        relief_factor = 0.10
        max_displacement_ratio = 0.10
        max_iterations = '256'
        coupled_mode = 'RELAXATION'
        relaxation_iters = 2
    else:
        # STANDARD (一般の衣服) およびデフォルト:
        # 平均エッジ長の 5%（最小 1mm、最大 20mm）
        thickness = max(0.001, min(0.02, avg_edge_len * 0.05))
        relief_factor = 0.20
        max_displacement_ratio = 0.20
        max_iterations = '512' if num_vertices >= 2000 else '256'
        coupled_mode = 'RELAXATION'
        relaxation_iters = 2

    # 安全策: 厚みは最小エッジ長の40%を超えないようにクランプ（自縄自縛・爆発防止）
    if min_edge_len > 0:
        thickness = min(thickness, min_edge_len * 0.4)

    return {
        'thickness': thickness,
        'self_collision_relief_factor': relief_factor,
        'self_collision_max_displacement_ratio': max_displacement_ratio,
        'self_collision_max_iterations': max_iterations,
        'coupled_self_collision_mode': coupled_mode,
        'post_collision_relaxation_iters': relaxation_iters,
        'enable_normal_untangling': True,
    }


def fit_self_collision_for_object(obj, settings=None, purpose: Optional[str] = None) -> Optional[Dict[str, Any]]:
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

    params = compute_self_collision_params(avg_len, min_len, n_verts, target_purpose)

    # settings に適用
    for k, v in params.items():
        if hasattr(settings, k):
            setattr(settings, k, v)

    return params
