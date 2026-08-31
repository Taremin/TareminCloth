"""
メッシュトポロジー操作ユーティリティ
四角面（Quad）メッシュの十字分割（Cross-Subdivision / Poke Faces）および、
シミュレーション後の後処理（最適対角線2三角分割、Quad復元、アダプティブ判定）を提供する。
"""

import math
import bpy
import bmesh
import numpy as np
from .logger import logger


def is_cross_subdivided(obj) -> bool:
    """オブジェクトが現在十字分割されているかを判定する"""
    if not obj or obj.type != 'MESH':
        return False
    return "_taremin_cross_subdiv_map" in obj and len(obj["_taremin_cross_subdiv_map"]) > 0


def backup_pre_subdivision_mesh(obj):
    """十字分割前のメッシュ（トポロジー・座標・属性）を完全バックアップする"""
    if not obj or obj.type != 'MESH':
        return
    if "_taremin_backup_mesh" in obj:
        mesh_name = obj["_taremin_backup_mesh"]
        old_backup = bpy.data.meshes.get(mesh_name)
        if old_backup:
            # 変形中（is_deformed=True）の場合は初期状態保護のため上書きしない
            if obj.get("_taremin_is_deformed", False):
                return
            if len(old_backup.vertices) == len(obj.data.vertices) and len(old_backup.polygons) == len(obj.data.polygons):
                return
            bpy.data.meshes.remove(old_backup, do_unlink=True)

    # 変形中に新規バックアップを作成することは避ける（すでに変形しているため）
    if obj.get("_taremin_is_deformed", False) and "_taremin_backup_mesh" in obj:
        return

    backup = obj.data.copy()
    backup.name = f".taremin_backup_{obj.name}"
    backup.use_fake_user = True
    obj["_taremin_backup_mesh"] = backup.name
    logger.info(f"[Topology] Created pre-subdivision backup mesh for '{obj.name}' -> '{backup.name}' ({len(backup.vertices)} verts)")


def restore_pre_subdivision_mesh(obj) -> bool:
    """十字分割前の初期メッシュ（トポロジー・初期座標・属性）に完全復元する"""
    if not obj or obj.type != 'MESH':
        return False
    if "_taremin_backup_mesh" not in obj:
        return False

    mesh_name = obj["_taremin_backup_mesh"]
    backup = bpy.data.meshes.get(mesh_name)
    if not backup:
        clear_pre_subdivision_backup(obj)
        return False

    curr_verts = len(obj.data.vertices)
    backup_verts = len(backup.vertices)
    max_added_verts = len(backup.polygons)
    subdiv_map = obj.get("_taremin_cross_subdiv_map")
    if subdiv_map:
        max_added_verts = len(subdiv_map)

    # 十字分割中または後処理（ADAPTIVE/OPTIMAL_TRI/QUAD）後であれば、
    # 頂点数は [backup_verts, backup_verts + max_added_verts] の範囲内に収まる
    if not (backup_verts <= curr_verts <= backup_verts + max_added_verts):
        logger.warning(
            f"[Topology] Mesh '{obj.name}' was modified after subdivision (curr_verts={curr_verts}, "
            f"backup_verts={backup_verts}). Discarding stale backup to protect user edits."
        )
        clear_pre_subdivision_backup(obj)
        return False

    bm = bmesh.new()
    bm.from_mesh(backup)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    if "_taremin_cross_subdiv_map" in obj:
        del obj["_taremin_cross_subdiv_map"]
    if "_taremin_is_cross_subdivided" in obj:
        del obj["_taremin_is_cross_subdivided"]

    logger.info(f"[Topology] Restored pre-subdivision mesh for '{obj.name}' from backup '{mesh_name}'")
    return True


def clear_pre_subdivision_backup(obj):
    """バックアップMeshデータブロックを完全に破棄する"""
    if not obj or obj.type != 'MESH':
        return
    if "_taremin_backup_mesh" in obj:
        mesh_name = obj["_taremin_backup_mesh"]
        backup = bpy.data.meshes.get(mesh_name)
        if backup:
            bpy.data.meshes.remove(backup, do_unlink=True)
            logger.debug(f"[Topology] Removed backup mesh block '{mesh_name}'")
        del obj["_taremin_backup_mesh"]
    if "_taremin_is_cross_subdivided" in obj:
        del obj["_taremin_is_cross_subdivided"]
    if "_taremin_cross_subdiv_map" in obj:
        del obj["_taremin_cross_subdiv_map"]


def evaluate_optimal_diagonal(p0, p1, p2, p3, pc):
    """
    4頂点と中心点の3D座標から、(v0, v2) と (v1, v3) のどちらの対角線が
    変形・シワの稜線によりフィットしているかを判定する。
    
    戻り値:
        bool: Trueなら (v0, v2)、Falseなら (v1, v3) を選択
    """
    m02 = (p0 + p2) * 0.5
    m13 = (p1 + p3) * 0.5
    d02 = float(np.linalg.norm(pc - m02))
    d13 = float(np.linalg.norm(pc - m13))
    return d02 <= d13


def evaluate_quad_diagonal_by_strain(p0_rest, p1_rest, p2_rest, p3_rest, p0_curr, p1_curr, p2_curr, p3_curr):
    """
    四角面の初期（レスト）座標と現在座標から、対角線の歪み（伸び縮み率: Strain）を計算し、
    シワの稜線に沿う最適な対角線を判定する。

    物理的背景:
    面内が圧縮されたとき、圧縮方向（歪み率 eps が小さく負になる方向）がシワの幅・谷となり、
    保たれている/伸びている方向（歪み率 eps が大きい方向）が折り目の稜線となる。
    したがって、歪み率が大きい方の対角線を選択する。

    引数:
        p0_rest, p1_rest, p2_rest, p3_rest: 時計回りまたは反時計回りの四角面4頂点のレスト座標
        p0_curr, p1_curr, p2_curr, p3_curr: 同4頂点の現在（変形後）座標
    戻り値:
        bool: Trueなら対角線 (v0, v2)、Falseなら対角線 (v1, v3) を選択
    """
    # レスト長
    L02 = float(np.linalg.norm(p2_rest - p0_rest))
    L13 = float(np.linalg.norm(p3_rest - p1_rest))

    # 現在長
    l02 = float(np.linalg.norm(p2_curr - p0_curr))
    l13 = float(np.linalg.norm(p3_curr - p1_curr))

    # 歪み率 (Engineering Strain): (l - L) / L
    eps02 = (l02 - L02) / max(L02, 1e-7)
    eps13 = (l13 - L13) / max(L13, 1e-7)

    # 歪み率が大きい（相対的に伸びている / 縮んでいない）方向が稜線
    return eps02 >= eps13


def evaluate_quad_flatness(face_normals, threshold_deg=5.0):
    """
    Quadを構成する小三角形の法線群から、平坦（シワがない）かどうかを判定する。
    
    引数:
        face_normals: 各小三角形の法線ベクトルのリスト（3D）
        threshold_deg: 平坦とみなす最大許容角度（度）
    戻り値:
        bool: 平坦なら True
    """
    if len(face_normals) < 2:
        return True
    
    avg_n = np.mean(face_normals, axis=0)
    norm = np.linalg.norm(avg_n)
    if norm < 1e-6:
        return False
    avg_n /= norm
    
    cos_threshold = math.cos(math.radians(threshold_deg))
    for n in face_normals:
        dot = float(np.dot(n, avg_n))
        if dot < cos_threshold:
            return False
    return True


def is_dome_deformation(p0, p1, p2, p3, pc, threshold_ratio=0.05):
    """
    中心点がどちらの対角線からも大きく浮き上がっている（ドーム状・テント状の変形）かを判定する。
    どちらの対角線でも形状を近似できない場合、中心頂点を残して十字分割（4個の三角形）を維持する。
    
    引数:
        p0, p1, p2, p3: 外周4頂点の3D座標
        pc: 中心頂点の3D座標
        threshold_ratio: 対角線長に対する中心点の最小浮き上がり許容比率 (デフォルト5%)
    戻り値:
        bool: ドーム状に盛り上がっている場合 True
    """
    diag_len = float((np.linalg.norm(p2 - p0) + np.linalg.norm(p3 - p1)) * 0.5)
    if diag_len < 1e-6:
        return False
    
    m02 = (p0 + p2) * 0.5
    m13 = (p1 + p3) * 0.5
    d02 = float(np.linalg.norm(pc - m02))
    d13 = float(np.linalg.norm(pc - m13))
    
    min_dist = min(d02, d13)
    return (min_dist / diag_len) > threshold_ratio


def apply_cross_subdivision(obj) -> bool:
    """
    オブジェクトの四角面メッシュを十字分割（Poke Faces）する。
    - 四角面（4頂点面）のみを対象に中心頂点を追加して4つの三角形に分割する。
    - 元のQuad頂点インデックスと中心頂点インデックスの対応マップを obj["_taremin_cross_subdiv_map"] に保存。
    - 成功した場合は True を返す。
    """
    if not obj or obj.type != 'MESH':
        return False

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # 四角面（4頂点面）のみを抽出
    quad_faces = [f for f in bm.faces if len(f.verts) == 4]
    if not quad_faces:
        bm.free()
        return False

    # 初回十字分割時に元のQuadメッシュ構造を完全バックアップ
    backup_pre_subdivision_mesh(obj)

    subdiv_map = []

    # 各四角面を個別に十字分割して対応関係を正確に追跡
    for f in quad_faces:
        orig_vert_indices = [v.index for v in f.verts]
        # poke 実行
        poke_res = bmesh.ops.poke(bm, faces=[f], center_mode='MEAN')
        new_verts = poke_res.get('verts', [])
        if new_verts:
            center_v = new_verts[0]
            subdiv_map.append({
                "quad_verts": orig_vert_indices,
                "center_vert_index": center_v.index,
            })

    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    final_subdiv_map = []
    for entry in subdiv_map:
        c_idx = entry["center_vert_index"]
        final_subdiv_map.append({
            "quad_verts": entry["quad_verts"],
            "center_vert_index": c_idx,
        })

    bm.to_mesh(mesh)
    mesh.update()
    bm.free()

    # メタデータを保存
    obj["_taremin_cross_subdiv_map"] = final_subdiv_map
    logger.info(
        f"[Topology] Applied cross-subdivision to '{obj.name}': poked {len(final_subdiv_map)} quads "
        f"(mesh now has {len(mesh.vertices)} verts, {len(mesh.polygons)} polygons)"
    )
    return True


def apply_post_process(obj, mode='OPTIMAL_TRI', flatness_threshold=5.0) -> bool:
    """
    シミュレーション後のメッシュに対してトポロジー後処理を実行する。
    
    引数:
        obj: 対象メッシュオブジェクト
        mode: 'KEEP' | 'QUAD' | 'OPTIMAL_TRI' | 'ADAPTIVE'
        flatness_threshold: アダプティブ判定の角度閾値（度）
    戻り値:
        bool: 処理が実行されたら True
    """
    if not obj or obj.type != 'MESH':
        return False

    if mode == 'KEEP':
        return True

    if not is_cross_subdivided(obj):
        return False

    subdiv_map = list(obj.get("_taremin_cross_subdiv_map", []))
    if not subdiv_map:
        return False

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # 各Quadごとの判定結果を記録
    # entry: (quad_verts, action: 'QUAD' | 'SPLIT_02' | 'SPLIT_13')
    quad_actions = []
    verts_to_dissolve = []

    for entry in subdiv_map:
        q_indices = entry["quad_verts"]
        c_idx = entry["center_vert_index"]

        if c_idx >= len(bm.verts) or any(i >= len(bm.verts) for i in q_indices):
            continue

        vc = bm.verts[c_idx]
        v0 = bm.verts[q_indices[0]]
        v1 = bm.verts[q_indices[1]]
        v2 = bm.verts[q_indices[2]]
        v3 = bm.verts[q_indices[3]]

        p0 = np.array(v0.co, dtype=np.float32)
        p1 = np.array(v1.co, dtype=np.float32)
        p2 = np.array(v2.co, dtype=np.float32)
        p3 = np.array(v3.co, dtype=np.float32)
        pc = np.array(vc.co, dtype=np.float32)

        # 最適対角線の判定
        use_02 = evaluate_optimal_diagonal(p0, p1, p2, p3, pc)

        # 平坦度および中心盛り上がり（ドーム状）の判定
        is_flat = False
        is_dome = False
        if mode == 'ADAPTIVE':
            face_normals = [np.array(f.normal, dtype=np.float32) for f in vc.link_faces]
            is_flat = evaluate_quad_flatness(face_normals, flatness_threshold)
            if not is_flat:
                is_dome = is_dome_deformation(p0, p1, p2, p3, pc, threshold_ratio=0.05)

        if mode == 'QUAD' or (mode == 'ADAPTIVE' and is_flat):
            # 1. 平坦: 中心頂点を削除して元のQuadに戻す
            verts_to_dissolve.append(vc)
            quad_actions.append((q_indices, 'QUAD'))
        elif mode == 'ADAPTIVE' and is_dome:
            # 2. 中心部が盛り上がっている（ドーム状・テント状）: 十字分割（4三角形）を維持
            pass
        else:
            # 3. 対角線上のシワ: 最適な対角線で2つの三角形に分割
            verts_to_dissolve.append(vc)
            action = 'SPLIT_02' if use_02 else 'SPLIT_13'
            quad_actions.append((q_indices, action))

    # 1. 中心頂点を一括 Dissolve（すべて元のQuadに戻る）
    valid_dissolve_verts = [v for v in verts_to_dissolve if v.is_valid]
    if valid_dissolve_verts:
        bmesh.ops.dissolve_verts(bm, verts=valid_dissolve_verts)

    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # 2. 最適対角線で Quad を 2 つの三角形に分割 (face_split)
    for q_indices, action in quad_actions:
        if action == 'QUAD':
            continue

        i0, i1, i2, i3 = q_indices
        if any(idx >= len(bm.verts) for idx in (i0, i1, i2, i3)):
            continue

        v0 = bm.verts[i0]
        v1 = bm.verts[i1]
        v2 = bm.verts[i2]
        v3 = bm.verts[i3]

        if not (v0.is_valid and v1.is_valid and v2.is_valid and v3.is_valid):
            continue

        # 4頂点をすべて含む四角面を探す
        quad_face = None
        for f in v0.link_faces:
            if len(f.verts) == 4 and v1 in f.verts and v2 in f.verts and v3 in f.verts:
                quad_face = f
                break

        if quad_face is not None:
            try:
                if action == 'SPLIT_02':
                    bmesh.utils.face_split(quad_face, v0, v2)
                elif action == 'SPLIT_13':
                    bmesh.utils.face_split(quad_face, v1, v3)
            except Exception:
                pass

    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.to_mesh(mesh)
    mesh.update()
    bm.free()

    # マップを削除（後処理完了）
    if "_taremin_cross_subdiv_map" in obj:
        del obj["_taremin_cross_subdiv_map"]
    if mode == 'QUAD':
        clear_pre_subdivision_backup(obj)

    logger.info(
        f"[Topology] Applied post-process (mode='{mode}') to '{obj.name}': "
        f"mesh now has {len(mesh.vertices)} verts, {len(mesh.polygons)} polygons"
    )
    return True


def restore_original_quads(obj) -> bool:
    """
    十字分割または動的対角線分割されたメッシュを元の四角面メッシュに完全復元する。
    """
    if is_cross_subdivided(obj):
        res = apply_post_process(obj, mode='QUAD')
        if res:
            clear_pre_subdivision_backup(obj)
        return res
    else:
        # 動的対角線分割等のバックアップからの復元
        return restore_pre_subdivision_mesh(obj)


def apply_dynamic_diagonal_triangulation(obj, rest_positions=None, preserve_flat=False, flatness_threshold=5.0) -> bool:
    """
    シミュレーション後の四角面メッシュに対して、歪み（Strain）に基づく動的対角線分割を実行する。
    
    特徴:
    - 頂点数は一切増加しない。
    - 四角面ごとに、対角線の伸び縮み率（歪み）を計算し、シワの稜線にフィットする方向で2分割（face_split）する。
    - 事前バックアップを取得するため、restore_pre_subdivision_mesh または restore_original_quads で復元可能。

    引数:
        obj: 対象メッシュオブジェクト
        rest_positions: レスト頂点座標の配列 (N, 3)。None の場合は "_taremin_rest_positions" または初期バックアップを参照
        preserve_flat: 平坦な四角面を分割せずそのまま保持するかどうか
        flatness_threshold: 平坦判定の角度閾値（度）
    戻り値:
        bool: 分割が実行されたら True
    """
    if not obj or obj.type != 'MESH':
        return False

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # 四角面（4頂点面）のみを抽出
    quad_faces = [f for f in bm.faces if len(f.verts) == 4]
    if not quad_faces:
        bm.free()
        return False

    # 復元用バックアップの作成
    backup_pre_subdivision_mesh(obj)

    # レスト座標の取得
    n_verts = len(bm.verts)
    rest_pos_arr = None
    if rest_positions is not None and len(rest_positions) == n_verts:
        rest_pos_arr = np.array(rest_positions, dtype=np.float32)
    elif "_taremin_rest_positions" in obj:
        raw_rest = obj["_taremin_rest_positions"]
        if len(raw_rest) == n_verts * 3:
            rest_pos_arr = np.array(raw_rest, dtype=np.float32).reshape((n_verts, 3))
    
    if rest_pos_arr is None:
        # バックアップメッシュの座標を参照
        if "_taremin_backup_mesh" in obj:
            b_mesh = bpy.data.meshes.get(obj["_taremin_backup_mesh"])
            if b_mesh and len(b_mesh.vertices) == n_verts:
                c = np.empty(n_verts * 3, dtype=np.float32)
                b_mesh.vertices.foreach_get("co", c)
                rest_pos_arr = c.reshape((n_verts, 3))

    if rest_pos_arr is None:
        # フォールバック: 現在座標をそのまま使用（この場合は幾何判定のみ）
        c = np.empty(n_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", c)
        rest_pos_arr = c.reshape((n_verts, 3))

    # 各Quadの現在頂点座標とレスト座標から分割方向を判定
    # entry: (quad_face, v_start, v_end)
    splits_to_perform = []

    for f in quad_faces:
        v0, v1, v2, v3 = f.verts[0], f.verts[1], f.verts[2], f.verts[3]
        i0, i1, i2, i3 = v0.index, v1.index, v2.index, v3.index

        p0_curr = np.array(v0.co, dtype=np.float32)
        p1_curr = np.array(v1.co, dtype=np.float32)
        p2_curr = np.array(v2.co, dtype=np.float32)
        p3_curr = np.array(v3.co, dtype=np.float32)

        p0_rest = rest_pos_arr[i0]
        p1_rest = rest_pos_arr[i1]
        p2_rest = rest_pos_arr[i2]
        p3_rest = rest_pos_arr[i3]

        if preserve_flat:
            # 2つの対角線分割それぞれの法線差を確認
            # (v0, v1, v2) と (v0, v2, v3)
            n1 = np.cross(p1_curr - p0_curr, p2_curr - p0_curr)
            n2 = np.cross(p2_curr - p0_curr, p3_curr - p0_curr)
            norm1 = np.linalg.norm(n1)
            norm2 = np.linalg.norm(n2)
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_ang = np.clip(np.dot(n1 / norm1, n2 / norm2), -1.0, 1.0)
                ang_deg = math.degrees(math.acos(cos_ang))
                if ang_deg < flatness_threshold:
                    # 平坦なので分割しない
                    continue

        use_02 = evaluate_quad_diagonal_by_strain(
            p0_rest, p1_rest, p2_rest, p3_rest,
            p0_curr, p1_curr, p2_curr, p3_curr
        )

        if use_02:
            splits_to_perform.append((f, v0, v2))
        else:
            splits_to_perform.append((f, v1, v3))

    split_count = 0
    for f, va, vb in splits_to_perform:
        try:
            bmesh.utils.face_split(f, va, vb)
            split_count += 1
        except Exception:
            pass

    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.to_mesh(mesh)
    mesh.update()
    bm.free()

    logger.info(
        f"[Topology] Applied dynamic diagonal triangulation to '{obj.name}': "
        f"split {split_count}/{len(quad_faces)} quads "
        f"(mesh now has {len(mesh.vertices)} verts, {len(mesh.polygons)} polygons)"
    )
    return split_count > 0
