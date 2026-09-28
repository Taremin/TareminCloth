"""
イラスト風シワフィールド（Wrinkle Field）のプリセット管理・アセット化モジュール

ボーン円柱座標系（θ, z/L, r/R）に基づき、
山・谷カーブ群を正規化されたプリセットとして定義・保存・復元・管理します。
Blender非依存の純粋な幾何・JSON層として実装され、CLI単体テストが可能です。
"""

from dataclasses import asdict, dataclass, field
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .wrinkle_field import (
    WrinkleCurveItem,
    build_orthonormal_basis,
    project_points_to_cylindrical,
)


@dataclass
class WrinklePresetCurve:
    """プリセット内の単一カーブ定義（正規化円柱座標系）"""
    type: str                  # "crest" (山) または "root" (谷)
    thetas: List[float]        # 円周角 θ (0.0 〜 2π)
    rel_zs: List[float]        # ボーン長に対する相対位置 z / L (0.0=Head, 1.0=Tail)
    rel_rs: List[float]        # ボーン半径に対する比率 r / R_bone
    strength: float = 1.0      # 個別強度倍率
    influence_radius: float = 0.03  # 個別影響半径 (m)
    name: str = ""             # カーブ識別名 (例: "Valley_Cinch")


@dataclass
class WrinklePreset:
    """シワフィールドのプリセット定義データ"""
    name: str                                  # プリセット名 (例: "Pinch_and_Puff")
    category: str = "general"                  # カテゴリ ("elbow_knee", "wrist_ankle", "torso", "general")
    description: str = ""                      # 説明
    curves: List[WrinklePresetCurve] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WrinklePreset":
        curves_data = data.get("curves", [])
        curves = [WrinklePresetCurve(**c) for c in curves_data]
        return cls(
            name=data.get("name", "Unnamed"),
            category=data.get("category", "general"),
            description=data.get("description", ""),
            curves=curves,
        )


# =========================================================================
# 組み込みプリセット (Built-in Presets)
# =========================================================================

def _make_circle_thetas(num_pts: int = 48) -> List[float]:
    """完全周回する円周角サンプルの生成 (末尾で 2π に戻る閉曲線)"""
    return [float(th) for th in np.linspace(0.0, 2.0 * math.pi, num_pts, endpoint=True)]


def get_builtin_presets() -> Dict[str, WrinklePreset]:
    """アドオン組み込みの標準シワプリセット辞書を返す"""
    presets = {}

    # 1. 1本くびれ (Cinch Single): 谷1本
    th_circle = _make_circle_thetas(48)
    presets["cinch_single"] = WrinklePreset(
        name="1本くびれ (Cinch Single)",
        category="general",
        description="中央部をキュッと引き締める谷カーブ1本",
        curves=[
            WrinklePresetCurve(
                type="root",
                thetas=th_circle,
                rel_zs=[0.5] * len(th_circle),
                rel_rs=[1.05] * len(th_circle),
                strength=1.0,
                influence_radius=0.03,
                name="Root_Cinch",
            )
        ],
    )

    # 2. ふんわり山 (Puff Single): 山1本
    presets["puff_single"] = WrinklePreset(
        name="ふんわり山 (Puff Single)",
        category="general",
        description="中央部をふんわり外側へ膨らませる山カーブ1本",
        curves=[
            WrinklePresetCurve(
                type="crest",
                thetas=th_circle,
                rel_zs=[0.5] * len(th_circle),
                rel_rs=[1.35] * len(th_circle),
                strength=1.0,
                influence_radius=0.04,
                name="Crest_Puff",
            )
        ],
    )

    # 3. ピンチ＆パフ (Pinch & Puff): 谷1本＋山1本のセット
    presets["pinch_and_puff"] = WrinklePreset(
        name="ピンチ＆パフ (Pinch & Puff)",
        category="general",
        description="谷の引き締めと山の膨らみによる定番の立体シワペア",
        curves=[
            WrinklePresetCurve(
                type="root",
                thetas=th_circle,
                rel_zs=[0.42] * len(th_circle),
                rel_rs=[1.05] * len(th_circle),
                strength=1.2,
                influence_radius=0.025,
                name="Root_Pinch",
            ),
            WrinklePresetCurve(
                type="crest",
                thetas=th_circle,
                rel_zs=[0.58] * len(th_circle),
                rel_rs=[1.35] * len(th_circle),
                strength=1.0,
                influence_radius=0.035,
                name="Crest_Puff",
            ),
        ],
    )

    # 4. 2連アコーディオン (Double Accordion): 谷2本＋山2本の連続蛇腹
    presets["accordion_double"] = WrinklePreset(
        name="2連アコーディオン (Double Accordion)",
        category="wrist_ankle",
        description="袖口や足首の溜まりシワに適した連続蛇腹シワ",
        curves=[
            WrinklePresetCurve(
                type="root",
                thetas=th_circle,
                rel_zs=[0.30] * len(th_circle),
                rel_rs=[1.05] * len(th_circle),
                strength=1.0,
                influence_radius=0.02,
                name="Root_1",
            ),
            WrinklePresetCurve(
                type="crest",
                thetas=th_circle,
                rel_zs=[0.43] * len(th_circle),
                rel_rs=[1.30] * len(th_circle),
                strength=1.0,
                influence_radius=0.025,
                name="Crest_1",
            ),
            WrinklePresetCurve(
                type="root",
                thetas=th_circle,
                rel_zs=[0.57] * len(th_circle),
                rel_rs=[1.05] * len(th_circle),
                strength=1.0,
                influence_radius=0.02,
                name="Root_2",
            ),
            WrinklePresetCurve(
                type="crest",
                thetas=th_circle,
                rel_zs=[0.70] * len(th_circle),
                rel_rs=[1.30] * len(th_circle),
                strength=1.0,
                influence_radius=0.025,
                name="Crest_2",
            ),
        ],
    )

    # 5. Y字屈曲シワ (Y-Branch Joint): 肘・膝の内側屈曲用
    # θ=π (内側) を中心にY字に分岐する山と、根本を留める谷
    num_branch_pts = 20
    # 幹: z=0.35〜0.50, θ=π
    stem_z = list(np.linspace(0.35, 0.50, num_branch_pts))
    stem_th = [float(math.pi)] * num_branch_pts
    stem_r = [1.30] * num_branch_pts

    # 枝1: z=0.50〜0.68, θ=π〜0.6π
    b1_z = list(np.linspace(0.50, 0.68, num_branch_pts))
    b1_th = [float(th) for th in np.linspace(math.pi, 0.6 * math.pi, num_branch_pts)]
    b1_r = [1.30] * num_branch_pts

    # 枝2: z=0.50〜0.68, θ=π〜1.4π
    b2_z = list(np.linspace(0.50, 0.68, num_branch_pts))
    b2_th = [float(th) for th in np.linspace(math.pi, 1.4 * math.pi, num_branch_pts)]
    b2_r = [1.30] * num_branch_pts

    presets["y_branch_joint"] = WrinklePreset(
        name="Y字関節シワ (Y-Branch Joint)",
        category="elbow_knee",
        description="肘や膝の内側の屈曲座屈に最適なY字分岐シワ",
        curves=[
            WrinklePresetCurve(
                type="root",
                thetas=th_circle,
                rel_zs=[0.35] * len(th_circle),
                rel_rs=[1.05] * len(th_circle),
                strength=1.2,
                influence_radius=0.025,
                name="Root_Base",
            ),
            WrinklePresetCurve(
                type="crest",
                thetas=stem_th,
                rel_zs=stem_z,
                rel_rs=stem_r,
                strength=1.0,
                influence_radius=0.025,
                name="Crest_Stem",
            ),
            WrinklePresetCurve(
                type="crest",
                thetas=b1_th,
                rel_zs=b1_z,
                rel_rs=b1_r,
                strength=1.0,
                influence_radius=0.025,
                name="Crest_Branch1",
            ),
            WrinklePresetCurve(
                type="crest",
                thetas=b2_th,
                rel_zs=b2_z,
                rel_rs=b2_r,
                strength=1.0,
                influence_radius=0.025,
                name="Crest_Branch2",
            ),
        ],
    )

    return presets


# =========================================================================
# 幾何変換: ワールド座標 ⇔ プリセット正規化座標
# =========================================================================

def normalize_curves_to_preset(
    curves: Sequence[WrinkleCurveItem],
    curve_types: Sequence[str],
    bone_head: Sequence[float],
    bone_tail: Sequence[float],
    bone_radius: float,
    normal: Optional[Sequence[float]] = None,
    name: str = "Custom_Preset",
    category: str = "general",
    description: str = "",
) -> WrinklePreset:
    """
    Blenderワールド座標上のカーブ群を、指定ボーン基準の正規化円柱座標系へ変換して WrinklePreset を生成します。
    
    - bone_head: ボーン始点 (3D)
    - bone_tail: ボーン終点 (3D)
    - bone_radius: ボーン基準半径 (m)
    """
    b_head = np.array(bone_head, dtype=np.float32)
    b_tail = np.array(bone_tail, dtype=np.float32)
    axis = b_tail - b_head
    bone_len = float(np.linalg.norm(axis))
    if bone_len < 1e-6:
        raise ValueError("ボーンの長さが短すぎます (0除外)")

    axis_norm, norm, binorm = build_orthonormal_basis(axis)
    if normal is not None:
        norm = np.array(normal, dtype=np.float32)
        norm = norm - np.dot(norm, axis_norm) * axis_norm
        norm = norm / (np.linalg.norm(norm) + 1e-8)
        binorm = np.cross(axis_norm, norm)

    preset_curves = []
    for idx, (item, c_type) in enumerate(zip(curves, curve_types)):
        if len(item.points) < 2:
            continue
        thetas, zs, rs = project_points_to_cylindrical(item.points, b_head, axis_norm, norm)
        
        # ボーン長と半径に対する正規化比率の算出
        rel_zs = [float(z / bone_len) for z in zs]
        rel_rs = [float(r / max(bone_radius, 1e-4)) for r in rs]
        th_list = [float(th) for th in thetas]

        p_curve = WrinklePresetCurve(
            type="crest" if c_type.lower() in ("crest", "ridge") else "root",
            thetas=th_list,
            rel_zs=rel_zs,
            rel_rs=rel_rs,
            strength=float(item.strength),
            influence_radius=float(item.influence_radius),
            name=f"{c_type.capitalize()}_{idx + 1}",
        )
        preset_curves.append(p_curve)

    return WrinklePreset(
        name=name,
        category=category,
        description=description,
        curves=preset_curves,
    )


def instantiate_preset_on_bone(
    preset: WrinklePreset,
    bone_head: Sequence[float],
    bone_tail: Sequence[float],
    bone_radius: float,
    normal: Optional[Sequence[float]] = None,
    scale_radius: float = 1.0,
) -> Tuple[List[WrinkleCurveItem], List[WrinkleCurveItem]]:
    """
    WrinklePreset を指定ボーンの位置・向き・スケールに合わせて実体化し、
    山カーブ群 (crest_items) と谷カーブ群 (root_items) の WrinkleCurveItem を返します。
    """
    b_head = np.array(bone_head, dtype=np.float32)
    b_tail = np.array(bone_tail, dtype=np.float32)
    axis = b_tail - b_head
    bone_len = float(np.linalg.norm(axis))
    if bone_len < 1e-6:
        raise ValueError("ボーンの長さが短すぎます")

    axis_norm, norm, binorm = build_orthonormal_basis(axis)
    if normal is not None:
        norm = np.array(normal, dtype=np.float32)
        norm = norm - np.dot(norm, axis_norm) * axis_norm
        norm = norm / (np.linalg.norm(norm) + 1e-8)
        binorm = np.cross(axis_norm, norm)

    crest_items = []
    root_items = []

    for c in preset.curves:
        pts = []
        n_pts = len(c.thetas)
        for i in range(n_pts):
            th = c.thetas[i]
            z = c.rel_zs[i] * bone_len
            r = c.rel_rs[i] * bone_radius * scale_radius
            
            # 円柱ローカルからワールド座標へ展開
            # p_world = origin + axis * z + r * (cos(th)*norm + sin(th)*binorm)
            rad_dir = math.cos(th) * norm + math.sin(th) * binorm
            p_world = b_head + axis_norm * z + rad_dir * r
            pts.append(p_world)

        if len(pts) >= 2:
            pts_arr = np.array(pts, dtype=np.float32)
            item = WrinkleCurveItem(
                points=pts_arr,
                strength=c.strength,
                influence_radius=c.influence_radius,
                target_radius=None,
            )
            if c.type.lower() in ("crest", "ridge"):
                crest_items.append(item)
            else:
                root_items.append(item)

    return crest_items, root_items


def instantiate_preset_on_chain(
    preset: WrinklePreset,
    chain: Any,
    t_param: float = 0.5,
    scale_radius: float = 1.0,
    target_bone_len: Optional[float] = None,
) -> Tuple[List[WrinkleCurveItem], List[WrinkleCurveItem]]:
    """
    WrinklePreset を BoneChain 上の指定位置 t_param に合わせて実体化し、
    山カーブ群 (crest_items) と谷カーブ群 (root_items) の WrinkleCurveItem を返します。
    """
    ev = chain.evaluate(t_param)
    if target_bone_len is None:
        if hasattr(chain, "segments") and chain.segments:
            seg = chain.segments[min(ev.segment_idx, len(chain.segments) - 1)] if ev.segment_idx >= 0 else chain.segments[0]
            target_bone_len = float(getattr(seg, "length", 0.2))
        else:
            target_bone_len = 0.2
    target_bone_len = float(max(0.05, min(1.0, target_bone_len)))

    target_head = ev.position - ev.axis * (target_bone_len * 0.5)
    target_tail = ev.position + ev.axis * (target_bone_len * 0.5)
    target_radius = float(ev.radius * scale_radius)

    return instantiate_preset_on_bone(
        preset=preset,
        bone_head=target_head,
        bone_tail=target_tail,
        bone_radius=target_radius,
        normal=ev.normal,
        scale_radius=1.0,
    )


# =========================================================================
# プリセットファイル I/O (JSON)
# =========================================================================

def get_user_presets_directory() -> str:
    """ユーザー定義プリセットの保存先ディレクトリパスを返す"""
    try:
        import bpy
        user_dir = bpy.utils.user_resource('SCRIPTS', path="presets/taremin_cloth/wrinkles", create=True)
        return user_dir
    except Exception:
        home = os.path.expanduser("~")
        p = os.path.join(home, ".taremin_cloth", "wrinkle_presets")
        os.makedirs(p, exist_ok=True)
        return p


def get_user_preset_files() -> List[str]:
    """ユーザーディレクトリ内の全プリセットJSONファイルパスを返す"""
    try:
        user_dir = get_user_presets_directory()
        if not isinstance(user_dir, str) or not os.path.exists(user_dir) or not os.path.isdir(user_dir):
            return []
        files = []
        for fname in sorted(os.listdir(user_dir)):
            if fname.endswith(".json"):
                files.append(os.path.join(user_dir, fname))
        return files
    except Exception:
        return []


def save_preset_to_json(preset: WrinklePreset, filepath: Optional[str] = None) -> str:
    """プリセットをJSONファイルに保存し、保存先パスを返す"""
    if not filepath:
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in preset.name)
        filepath = os.path.join(get_user_presets_directory(), f"{safe_name}.json")
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(preset.to_dict(), f, indent=2, ensure_ascii=False)
    return filepath


def load_preset_from_json(filepath: str) -> Optional[WrinklePreset]:
    """JSONファイルからプリセットを読み込み"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return WrinklePreset.from_dict(data)
    except Exception:
        return None


def list_available_presets() -> Dict[str, WrinklePreset]:
    """組み込みプリセットおよびユーザー保存プリセットを統合して返す"""
    presets = get_builtin_presets()

    for fpath in get_user_preset_files():
        p = load_preset_from_json(fpath)
        if p:
            key = f"user_{p.name}"
            presets[key] = p

    return presets
