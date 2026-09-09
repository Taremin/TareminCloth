#!/usr/bin/env python3
"""
Taremin Cloth Blender ファイル インスペクションツール

指定された .blend ファイルをヘッドレスBlenderで開き、
シーン構成、布オブジェクト、コライダー、アーマチュア設定をCLI上にサマリー表示します。

使用例:
  python tools/inspect_blend.py path/to/model.blend
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# blender_managerのインポート
tools_dir = Path(__file__).resolve().parent
repo_root = tools_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.blender_manager import resolve_blender


BLENDER_INSPECT_SCRIPT = r"""
import sys
import json
import bpy

def inspect():
    scene = bpy.context.scene
    data = {
        "blender_version": bpy.app.version_string,
        "scene": {
            "name": scene.name,
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "fps": scene.render.fps,
        },
        "cloth_objects": [],
        "colliders": [],
        "armatures": [],
        "other_meshes": [],
    }

    for obj in bpy.data.objects:
        cloth = getattr(obj, "taremin_cloth", None)
        collider = getattr(obj, "taremin_cloth_collider", None)

        is_cloth = bool(cloth and getattr(cloth, "is_cloth", False))
        is_col = bool(collider and getattr(collider, "is_collider", False))

        if is_cloth:
            data["cloth_objects"].append({
                "name": obj.name,
                "vertices": len(obj.data.vertices),
                "faces": len(obj.data.polygons),
                "substeps": cloth.substeps,
                "solver_iterations": cloth.solver_iterations,
                "tension_stiffness": cloth.tension_stiffness,
                "bending_stiffness": cloth.bending_stiffness,
                "air_damping": cloth.air_damping,
                "thickness": cloth.thickness,
                "vertex_groups": [vg.name for vg in obj.vertex_groups],
            })
        elif is_col:
            data["colliders"].append({
                "name": obj.name,
                "type": obj.type,
                "collider_type": collider.collider_type,
                "vertices": len(obj.data.vertices) if obj.data and hasattr(obj.data, "vertices") else 0,
                "faces": len(obj.data.polygons) if obj.data and hasattr(obj.data, "polygons") else 0,
                "thickness": collider.thickness,
                "friction": collider.friction,
                "single_sided": collider.single_sided,
            })
        elif obj.type == "ARMATURE":
            data["armatures"].append({
                "name": obj.name,
                "bones_count": len(obj.data.bones),
                "bone_names": [b.name for b in obj.data.bones[:20]],
                "has_pose": bool(obj.pose),
            })
        elif obj.type == "MESH":
            data["other_meshes"].append({
                "name": obj.name,
                "vertices": len(obj.data.vertices),
                "faces": len(obj.data.polygons),
            })

    print("___TAREMIN_INSPECT_JSON_START___")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print("___TAREMIN_INSPECT_JSON_END___")

inspect()
"""


def inspect_blend_file(blend_path: Path, blender_version: str | None = None) -> dict:
    if not blend_path.exists():
        raise FileNotFoundError(f".blend ファイルが見つかりません: {blend_path}")

    blender_exe = resolve_blender(blender_version or "latest-lts")

    cmd = [
        str(blender_exe),
        "-b",
        "--factory-startup",
        str(blend_path),
        "--python-expr",
        BLENDER_INSPECT_SCRIPT,
    ]

    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    stdout = res.stdout
    start_tag = "___TAREMIN_INSPECT_JSON_START___"
    end_tag = "___TAREMIN_INSPECT_JSON_END___"

    if start_tag not in stdout or end_tag not in stdout:
        print("[!] インスペクション結果の取得に失敗しました。Blenderログ:", file=sys.stderr)
        print(res.stderr or stdout, file=sys.stderr)
        sys.exit(1)

    json_text = stdout.split(start_tag)[1].split(end_tag)[0].strip()
    return json.loads(json_text)


def print_summary(data: dict, blend_path: Path):
    print("=" * 70)
    print(f"【Blend ファイル インスペクション】: {blend_path.name}")
    print(f"ファイルパス: {blend_path}")
    print(f"Blender環境: {data['blender_version']}")
    scene = data["scene"]
    print(f"シーン: '{scene['name']}', フレーム範囲: {scene['frame_start']}〜{scene['frame_end']} ({scene['fps']} fps)")
    print("=" * 70)

    cloth_objs = data["cloth_objects"]
    print(f"\n■ 布オブジェクト (Taremin Cloth): {len(cloth_objs)} 個")
    if cloth_objs:
        for c in cloth_objs:
            print(f"  - '{c['name']}': {c['vertices']:,} 頂点, {c['faces']:,} 面")
            print(f"    substeps={c['substeps']}, iters={c['solver_iterations']}, tension={c['tension_stiffness']}, bend={c['bending_stiffness']}")
            if c["vertex_groups"]:
                print(f"    頂点グループ: {', '.join(c['vertex_groups'][:8])}{'...' if len(c['vertex_groups']) > 8 else ''}")
    else:
        print("    (布オブジェクトは登録されていません)")

    colliders = data["colliders"]
    print(f"\n■ コライダー (Taremin Collider): {len(colliders)} 個")
    if colliders:
        total_col_faces = sum(c["faces"] for c in colliders)
        print(f"  (合計ポリゴン数: {total_col_faces:,} 面)")
        for col in colliders:
            print(f"  - '{col['name']}' ({col['type']}): 形状={col['collider_type']}, {col['vertices']:,} 頂点, {col['faces']:,} 面, 厚み={col['thickness']}m")
    else:
        print("    (コライダーは登録されていません)")

    armatures = data["armatures"]
    if armatures:
        print(f"\n■ アーマチュア (ボーン階層): {len(armatures)} 個")
        for arm in armatures:
            print(f"  - '{arm['name']}': {arm['bones_count']} ボーン (主要ボーン: {', '.join(arm['bone_names'][:6])}...)")

    other_meshes = data["other_meshes"]
    if other_meshes:
        print(f"\n■ その他のメッシュオブジェクト: {len(other_meshes)} 個")
        for m in other_meshes[:5]:
            print(f"  - '{m['name']}': {m['vertices']:,} 頂点, {m['faces']:,} 面")
        if len(other_meshes) > 5:
            print(f"    ... 他 {len(other_meshes) - 5} 個")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Taremin Cloth .blend インスペクションツール")
    parser.add_argument("blend_file", type=Path, help="解析対象の .blend ファイル")
    parser.add_argument("--blender", type=str, default=None, help="使用するBlenderバージョン (例: 5.2, 4.2)")
    parser.add_argument("--json", action="store_true", help="JSON形式で出力")

    args = parser.parse_args()
    data = inspect_blend_file(args.blend_file, args.blender)

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print_summary(data, args.blend_file)


if __name__ == "__main__":
    main()
