"""
taremin_cloth GUI クライアントモジュール
Taremin Cloth GUI (taremin_cloth_gui.exe) とローカルTCPソケット経由で通信し、
シーン初期化、パラメータ同期、およびノンブロッキングな最新変形座標の取得を行う。
"""

import base64
import json
import socket
import struct
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


from ..utils.logger import logger
from ..utils.mesh_extract import extract_cloth_mesh_data

_global_client: Optional['ClothGuiClient'] = None


def get_gui_client() -> 'ClothGuiClient':
    """グローバルな ClothGuiClient インスタンスを取得する"""
    global _global_client
    if _global_client is None:
        _global_client = ClothGuiClient()
    return _global_client


def extract_scene_colliders(scene=None, depsgraph=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    シーン内の全コライダー（アナリティックコライダー、メッシュコライダー、および BONE_SDF の関節メッシュ）を抽出する。
    collider.extract_all_scene_colliders を直接利用して Blender In-process との 100% のパリティを保証する。
    """
    from .collider import extract_all_scene_colliders
    try:
        import bpy
        if depsgraph is None:
            try:
                depsgraph = bpy.context.evaluated_depsgraph_get()
            except Exception:
                depsgraph = None
    except Exception:
        pass

    col_data = extract_all_scene_colliders(scene, depsgraph)
    return col_data["analytic_colliders"], col_data["mesh_triangles_data"]


def extract_bone_sdf_data(scene=None) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], List[List[float]], List[List[float]]]:
    """
    シーン内のBONE_SDFまたはMESH_SDFコライダーを検出し、
    (bone_sdf_dict, joint_mesh_triangles, voxels_list, lines_list) を返す。
    """
    try:
        import bpy
    except ImportError:
        return None, [], [], []

    if scene is None:
        try:
            scene = bpy.context.scene
        except Exception:
            return None, [], [], []

    from .collider import (
        get_armature_modifier,
        get_or_bake_bone_sdf_for_object,
        get_or_bake_mesh_sdf_for_object,
    )
    from ..mesh_renderer import extract_sdf_surface_voxels, transform_sdf_voxels_to_world

    bone_sdf_dict = None
    joint_mesh_triangles = []
    voxels_list = []
    extra_lines = []

    for obj in getattr(scene, "objects", []):
        col_settings = getattr(obj, "taremin_cloth_collider", None)
        if not col_settings or not getattr(col_settings, "is_collider", False) or not getattr(col_settings, "enabled", True):
            continue

        if col_settings.collider_type == 'BONE_SDF':
            arm_mod = get_armature_modifier(obj)
            if not arm_mod or not arm_mod.object:
                continue
            arm_obj = arm_mod.object

            bake_res = get_or_bake_bone_sdf_for_object(obj, col_settings)
            if bake_res and bake_res.depth > 0:
                tex_b64 = base64.b64encode(bake_res.texture_bytes).decode('ascii')
                bone_infos_list = [[float(x) for x in info] for info in bake_res.bone_infos]
                bone_sdf_dict = {
                    "width": int(bake_res.width),
                    "height": int(bake_res.height),
                    "depth": int(bake_res.depth),
                    "texture_base64": tex_b64,
                    "bone_infos": bone_infos_list,
                }

                # ボーンワールド行列の収集 (collider.py と完全一致させる)
                bone_world_mats = {}
                bone_transforms = []
                mat_arm_world = np.array(arm_obj.matrix_world, dtype=np.float32)
                for bname in bake_res.bone_names:
                    pbone = arm_obj.pose.bones.get(bname) if arm_obj.pose else None
                    if pbone:
                        w_mat = mat_arm_world @ np.array(pbone.matrix, dtype=np.float32)
                    else:
                        w_mat = mat_arm_world
                    bone_world_mats[bname] = w_mat
                    bone_transforms.append(w_mat.T.tolist())

                bone_sdf_dict["bone_transforms"] = bone_transforms

                # 1. SDF表面ボクセルの抽出
                try:
                    bone_voxels = extract_sdf_surface_voxels(bake_res, stride=2, adaptive=True)
                    world_vox = transform_sdf_voxels_to_world(bone_voxels, bone_world_mats)
                    if world_vox is not None and len(world_vox) > 0:
                        w_copy = world_vox.copy()
                        if np.max(w_copy[:, 4:7]) > 1.0:
                            w_copy[:, 4:7] /= 255.0
                        voxels_list.extend(w_copy.tolist())
                except Exception as e:
                    logger.warning(f"[GuiClient] Failed to extract SDF voxels: {e}")

                # 2. SDF BBOX枠線（12エッジ）の抽出
                try:
                    for b_idx, bname in enumerate(bake_res.bone_names):
                        info = bake_res.bone_infos[b_idx]
                        min_pt = np.array(info[0:3], dtype=np.float32)
                        max_pt = np.array(info[4:7], dtype=np.float32)
                        b_mat = bone_world_mats.get(bname, np.eye(4, dtype=np.float32))

                        corners_local = np.array([
                            [min_pt[0], min_pt[1], min_pt[2]],
                            [max_pt[0], min_pt[1], min_pt[2]],
                            [max_pt[0], max_pt[1], min_pt[2]],
                            [min_pt[0], max_pt[1], min_pt[2]],
                            [min_pt[0], min_pt[1], max_pt[2]],
                            [max_pt[0], min_pt[1], max_pt[2]],
                            [max_pt[0], max_pt[1], max_pt[2]],
                            [min_pt[0], max_pt[1], max_pt[2]],
                        ], dtype=np.float32)
                        corners_homo = np.hstack([corners_local, np.ones((8, 1), dtype=np.float32)])
                        corners_world = (corners_homo @ b_mat.T)[:, :3]

                        edges = [
                            (0, 1), (1, 2), (2, 3), (3, 0),
                            (4, 5), (5, 6), (6, 7), (7, 4),
                            (0, 4), (1, 5), (2, 6), (3, 7)
                        ]
                        col = [0.2, 0.8, 1.0] # シアン
                        for e0, e1 in edges:
                            p0 = corners_world[e0].tolist()
                            p1 = corners_world[e1].tolist()
                            extra_lines.append(p0 + p1 + col)
                except Exception as e:
                    logger.warning(f"[GuiClient] Failed to generate BBOX lines: {e}")

                break

        elif col_settings.collider_type == 'MESH_SDF':
            bake_res = get_or_bake_mesh_sdf_for_object(obj, col_settings)
            if bake_res and bake_res.depth > 0:
                tex_b64 = base64.b64encode(bake_res.texture_bytes).decode('ascii')
                bone_infos_list = [[float(x) for x in info] for info in bake_res.bone_infos]
                w_mat = np.array(obj.matrix_world, dtype=np.float32)
                bone_sdf_dict = {
                    "width": int(bake_res.width),
                    "height": int(bake_res.height),
                    "depth": int(bake_res.depth),
                    "texture_base64": tex_b64,
                    "bone_infos": bone_infos_list,
                    "bone_transforms": [w_mat.T.tolist()],
                }
                break

    return bone_sdf_dict, joint_mesh_triangles, voxels_list, extra_lines


def extract_self_collision_data(settings) -> Optional[Dict[str, Any]]:
    """自己衝突オプションを抽出する"""
    if not settings:
        return None

    mode_str = getattr(settings, "coupled_self_collision_mode", "RELAXATION")
    if mode_str == "OFF":
        mode_int = 0
        relax_iters = 0
    elif mode_str == "FULL_COUPLED":
        mode_int = 3
        relax_iters = int(getattr(settings, "post_collision_relaxation_iters", 1))
        if relax_iters == 0:
            relax_iters = 1
    else:
        mode_int = 1
        relax_iters = int(getattr(settings, "post_collision_relaxation_iters", 2))
        if relax_iters == 0:
            relax_iters = 2

    return {
        "enabled": bool(getattr(settings, "enable_self_collision", False)),
        "coupled_mode": mode_int,
        "post_relaxation_iters": relax_iters,
        "relief_factor": float(getattr(settings, "self_collision_relief_factor", 0.2)),
        "max_displacement_ratio": float(getattr(settings, "self_collision_max_displacement_ratio", 0.2)),
        "exclude_neighbors": bool(getattr(settings, "self_collision_exclude_neighbors", True)),
        "enable_normal_untangling": bool(getattr(settings, "enable_normal_untangling", True)),
        "substep_interval": int(getattr(settings, "self_collision_substep_interval", 1)),
        "enable_edge_collision": bool(getattr(settings, "enable_edge_collision", False)),
        "edge_margin_scale": float(getattr(settings, "edge_margin_scale", 1.0)),
        "edge_margin_offset": float(getattr(settings, "edge_margin_offset", 0.0)),
    }


def extract_elastic_bands_data(obj) -> Optional[Dict[str, Any]]:
    """伸縮グループ設定を抽出する"""
    settings = getattr(obj, "taremin_cloth", None)
    if not settings or not getattr(settings, "elastic_groups", None):
        return None

    edge_scale_map = {}
    for g in settings.elastic_groups:
        if not getattr(g, "enabled", True):
            continue
        edges = g.get_edge_indices()
        scale = float(g.scale)
        for e_idx in edges:
            edge_scale_map[int(e_idx)] = scale

    if not edge_scale_map:
        return None

    indices = list(edge_scale_map.keys())
    scales = [edge_scale_map[idx] for idx in indices]
    return {
        "edge_indices": indices,
        "scales": scales,
    }



def _json_numpy_default(obj):
    """NumPyの型をPython標準型に変換してJSONシリアライズを可能にする"""
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class ClothGuiClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 9055):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self._is_connected = False
        self._last_frame_seq = 0
        self._last_physics_fps = 0.0
        self._last_render_fps = 0.0
        self._last_step_ms = 0.0
        self._last_sent_params = None

    @property
    def render_fps(self) -> float:
        return self._last_render_fps

    @property
    def physics_fps(self) -> float:
        return self._last_physics_fps

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self.sock is not None

    def connect(self, timeout: float = 3.0) -> bool:
        """GUIサーバーにTCPソケット接続する"""
        if self.is_connected:
            return True

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((self.host, self.port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setblocking(False)
            self.sock = sock
            self._is_connected = True
            logger.info(f"[GuiClient] Connected to Taremin Cloth GUI at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.warning(f"[GuiClient] Failed to connect to {self.host}:{self.port}: {e}")
            self.disconnect()
            return False

    def disconnect(self):
        """ソケットを切断する"""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None
        self._is_connected = False
        self._last_sent_params = None
        logger.info("[GuiClient] Disconnected")

    def _send_packet(self, data_dict: Dict[str, Any]) -> bool:
        """4バイト長さプレフィックス付きJSONパケットを送信する"""
        if not self.is_connected or not self.sock:
            return False

        try:
            payload = json.dumps(data_dict, default=_json_numpy_default).encode("utf-8")
            header = struct.pack(">I", len(payload))
            self.sock.sendall(header + payload)
            return True
        except Exception as e:
            logger.error(f"[GuiClient] Send error: {e}")
            self.disconnect()
            return False

    def _recv_packet(self, timeout: float = 0.5) -> Optional[Dict[str, Any]]:
        """パケットを1つ受信する（ノンブロッキング対応）"""
        if not self.is_connected or not self.sock:
            return None

        start_time = time.perf_counter()
        header_buf = b""

        while len(header_buf) < 4:
            try:
                chunk = self.sock.recv(4 - len(header_buf))
                if not chunk:
                    self.disconnect()
                    return None
                header_buf += chunk
            except BlockingIOError:
                if time.perf_counter() - start_time > timeout:
                    return None
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"[GuiClient] Recv header error: {e}")
                self.disconnect()
                return None

        length = struct.unpack(">I", header_buf)[0]
        payload_buf = b""

        while len(payload_buf) < length:
            try:
                chunk = self.sock.recv(min(length - len(payload_buf), 65536))
                if not chunk:
                    self.disconnect()
                    return None
                payload_buf += chunk
            except BlockingIOError:
                if time.perf_counter() - start_time > timeout:
                    return None
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"[GuiClient] Recv payload error: {e}")
                self.disconnect()
                return None

        try:
            return json.loads(payload_buf.decode("utf-8"))
        except Exception as e:
            logger.error(f"[GuiClient] JSON parse error: {e}")
            return None

    def send_init_scene(self, obj, scene=None) -> bool:
        """Blenderオブジェクトからメッシュデータ、全物理パラメータ、全コライダー、SDFを抽出してGUIサーバーに初期化送信する"""
        try:
            import bpy
            if scene is None:
                scene = bpy.context.scene
        except Exception:
            scene = None

        settings = getattr(obj, "taremin_cloth", None)
        cloth_data = extract_cloth_mesh_data(obj, settings)
        positions = cloth_data.positions.tolist()
        faces = cloth_data.faces.tolist() if cloth_data.faces is not None else []
        normal_edges = cloth_data.normal_edges.tolist()
        sewing_springs = cloth_data.sewing_edges.tolist() if cloth_data.sewing_edges is not None else None
        inv_masses = cloth_data.inv_masses.tolist()

        # 重力ベクトル & 重力倍率
        scale = float(getattr(settings, "gravity", 1.0)) if settings else 1.0
        gravity = [0.0, 0.0, -9.81 * scale]
        if scene and getattr(scene, "use_gravity", True):
            sg = scene.gravity
            gravity = [float(sg.x * scale), float(sg.y * scale), float(sg.z * scale)]
        elif scene:
            gravity = [0.0, 0.0, 0.0]
            scale = 0.0

        # コライダー抽出
        colliders, mesh_triangles = extract_scene_colliders(scene)

        # ボーンSDF & デバッグ可視化データ抽出
        bone_sdf_dict, joint_tris, voxels, extra_lines = extract_bone_sdf_data(scene)
        if joint_tris:
            mesh_triangles.extend(joint_tris)

        # 自己衝突オプション
        self_col = extract_self_collision_data(settings)

        # 伸縮グループ
        elastic_bands = extract_elastic_bands_data(obj)

        data = {
            "type": "InitScene",
            "data": {
                "object_name": obj.name,
                "positions": positions,
                "faces": faces,
                "edges": normal_edges,
                "sewing_springs": sewing_springs,
                "inv_masses": inv_masses,
                "layer_id": int(getattr(settings, "layer_id", 0)) if settings else 0,
                "thickness": float(getattr(settings, "thickness", 0.005)) if settings else 0.005,
                "stiffness": float(getattr(settings, "tension_stiffness", 100.0)) if settings else 100.0,
                "compression_stiffness": float(getattr(settings, "compression_stiffness", 100.0)) if settings else 100.0,
                "shear_stiffness": float(getattr(settings, "shear_stiffness", 50.0)) if settings else 50.0,
                "bending_stiffness": float(getattr(settings, "bending_stiffness", 1.0)) if settings else 1.0,
                "air_damping": float(getattr(settings, "air_damping", 0.01)) if settings else 0.01,
                "tension_damping": float(getattr(settings, "tension_damping", 0.0)) if settings else 0.0,
                "compression_damping": float(getattr(settings, "compression_damping", 0.0)) if settings else 0.0,
                "shear_damping": float(getattr(settings, "shear_damping", 0.0)) if settings else 0.0,
                "bending_damping": float(getattr(settings, "bending_damping", 0.0)) if settings else 0.0,
                "gravity": gravity,
                "gravity_scale": scale,
                "sewing_shrink_speed": float(getattr(settings, "sewing_shrink_speed", 0.5)) if settings else 0.5,
                "workgroup_size": int(getattr(settings, "workgroup_size", "32")) if settings else 32,
                "solver_mode": 1 if getattr(settings, "solver_mode", "COLORING") == 'ATOMIC' else 0,
                "solver_iterations": int(getattr(settings, "solver_iterations", 10)) if settings else 10,
                "substeps": int(getattr(settings, "substeps", 20)) if settings else 20,
                "fps": float(scene.render.fps) if scene and hasattr(scene, "render") else 60.0,
                "self_collision": self_col,
                "bone_sdf": bone_sdf_dict,
                "colliders": colliders,
                "mesh_triangles": mesh_triangles,
                "elastic_bands": elastic_bands,
                "voxels": voxels,
                "extra_lines": extra_lines,
            }
        }

        if not self._send_packet(data):
            return False

        resp = self._recv_packet(timeout=10.0)
        return resp is not None and resp.get("type") == "Ack"

    def send_update_colliders(self, scene=None) -> bool:
        """シーン内のコライダー位置・姿勢の最新状態をGUIサーバーに更新送信する"""
        colliders, mesh_triangles = extract_scene_colliders(scene)
        return self.send_command("UpdateColliders", {
            "colliders": colliders,
            "mesh_triangles": mesh_triangles,
        })

    def send_update_bone_transforms(self, transforms_or_scene=None) -> bool:
        """ボーン変換行列を送信する（直接リストまたはBlender scene）"""
        if transforms_or_scene is not None and isinstance(transforms_or_scene, (list, tuple)):
            return self.send_command("UpdateBoneTransforms", {"transforms": list(transforms_or_scene)})

        try:
            import bpy
            from .collider import get_armature_modifier, get_or_bake_bone_sdf_for_object
            scene = transforms_or_scene if transforms_or_scene is not None else bpy.context.scene
        except Exception:
            return False

        transforms = []
        for obj in getattr(scene, "objects", []):
            col_settings = getattr(obj, "taremin_cloth_collider", None)
            if not col_settings or not getattr(col_settings, "is_collider", False):
                continue

            if col_settings.collider_type == 'BONE_SDF':
                arm_mod = get_armature_modifier(obj)
                if not arm_mod or not arm_mod.object:
                    continue
                arm_obj = arm_mod.object
                bake_res = get_or_bake_bone_sdf_for_object(obj, col_settings)
                if bake_res and arm_obj.pose:
                    mat_arm_world = np.array(arm_obj.matrix_world, dtype=np.float32)
                    for bname in bake_res.bone_names:
                        pbone = arm_obj.pose.bones.get(bname)
                        if pbone is not None:
                            w_mat = mat_arm_world @ np.array(pbone.matrix, dtype=np.float32)
                        else:
                            w_mat = mat_arm_world
                        transforms.append(w_mat.T.tolist())
                    break

            elif col_settings.collider_type == 'MESH_SDF':
                w_mat = np.array(obj.matrix_world, dtype=np.float32)
                transforms.append(w_mat.T.tolist())
                break

        if not transforms:
            return True
        return self.send_command("UpdateBoneTransforms", {"transforms": transforms})

    def send_update_elastic_scales(self, edge_indices_or_obj, scales=None) -> bool:
        """伸縮グループの最新スケールを送信する（直接リストまたはBlender obj）"""
        if scales is not None:
            return self.send_command("UpdateElasticScales", {
                "edge_indices": list(edge_indices_or_obj),
                "scales": list(scales),
            })

        eb = extract_elastic_bands_data(edge_indices_or_obj)
        if not eb:
            return True
        return self.send_command("UpdateElasticScales", eb)

    def send_update_pins(self, indices_or_obj, positions=None, weights=None, scene=None) -> bool:
        """アタッチメントピンの最新位置を送信する（直接リストまたはBlender obj）"""
        if positions is not None and weights is not None:
            return self.send_command("UpdatePins", {
                "vertex_indices": list(indices_or_obj),
                "positions": list(positions),
                "weights": list(weights),
            })

        obj = indices_or_obj
        settings = getattr(obj, "taremin_cloth", None)
        if not settings or not getattr(settings, "pin_target_object", None):
            return True

        target_obj = settings.pin_target_object
        vg_name = settings.pin_vertex_group or "Pin"
        vg = obj.vertex_groups.get(vg_name)
        if not vg:
            return True

        target_mat = target_obj.matrix_world
        if getattr(settings, "pin_target_bone", None) and target_obj.type == 'ARMATURE' and target_obj.pose:
            bone = target_obj.pose.bones.get(settings.pin_target_bone)
            if bone:
                target_mat = target_mat @ bone.matrix

        inv_world = obj.matrix_world.inverted()
        local_target = inv_world @ target_mat.translation
        target_pt = [float(local_target.x), float(local_target.y), float(local_target.z)]

        v_indices = []
        pos_list = []
        weight_list = []

        mesh = obj.data
        for v in mesh.vertices:
            try:
                w = vg.weight(v.index)
            except RuntimeError:
                continue
            if w > 0.0:
                v_indices.append(v.index)
                pos_list.append(target_pt)
                weight_list.append(float(w))

        if not v_indices:
            return True

        return self.send_command("UpdatePins", {
            "vertex_indices": v_indices,
            "positions": pos_list,
            "weights": weight_list,
        })

    def send_params(
        self,
        gravity=None,
        gravity_scale=None,
        air_damping=None,
        tension_damping=None,
        compression_damping=None,
        shear_damping=None,
        bending_damping=None,
        stiffness=None,
        compression_stiffness=None,
        shear_stiffness=None,
        bending_stiffness=None,
        solver_iterations=None,
    ) -> bool:
        """物理パラメータの動的更新コマンドを送信する"""
        params = {}
        if gravity is not None:
            params["gravity"] = list(gravity)
        if gravity_scale is not None:
            params["gravity_scale"] = float(gravity_scale)
        if air_damping is not None:
            params["air_damping"] = float(air_damping)
        if tension_damping is not None:
            params["tension_damping"] = float(tension_damping)
        if compression_damping is not None:
            params["compression_damping"] = float(compression_damping)
        if shear_damping is not None:
            params["shear_damping"] = float(shear_damping)
        if bending_damping is not None:
            params["bending_damping"] = float(bending_damping)
        if stiffness is not None:
            params["stiffness"] = float(stiffness)
        if compression_stiffness is not None:
            params["compression_stiffness"] = float(compression_stiffness)
        if shear_stiffness is not None:
            params["shear_stiffness"] = float(shear_stiffness)
        if bending_stiffness is not None:
            params["bending_stiffness"] = float(bending_stiffness)
        if solver_iterations is not None:
            params["solver_iterations"] = int(solver_iterations)

        return self.send_command("SetParams", params)

    def send_update_params_from_settings(self, obj, scene=None) -> bool:
        """Blenderオブジェクトおよびシーンの現在の設定をGUIへ同期送信する (差分検知付き)"""
        settings = getattr(obj, "taremin_cloth", None)
        if not settings:
            return False

        # 重力ベクトル算出 (シーン重力 × オブジェクト重力倍率)
        if scene is not None:
            if getattr(scene, "use_gravity", True):
                sg = scene.gravity
                scale = float(getattr(settings, "gravity", 1.0))
                gravity_vec = [float(sg.x * scale), float(sg.y * scale), float(sg.z * scale)]
            else:
                scale = 0.0
                gravity_vec = [0.0, 0.0, 0.0]
        else:
            scale = float(getattr(settings, "gravity", 1.0))
            gravity_vec = [0.0, 0.0, -9.81 * scale]

        params = {
            "gravity": gravity_vec,
            "gravity_scale": scale,
            "air_damping": float(getattr(settings, "air_damping", 1.0)),
            "tension_damping": float(getattr(settings, "tension_damping", 5.0)),
            "compression_damping": float(getattr(settings, "compression_damping", 5.0)),
            "shear_damping": float(getattr(settings, "shear_damping", 5.0)),
            "bending_damping": float(getattr(settings, "bending_damping", 0.5)),
            "stiffness": float(getattr(settings, "tension_stiffness", 1000.0)),
            "compression_stiffness": float(getattr(settings, "compression_stiffness", 100.0)),
            "shear_stiffness": float(getattr(settings, "shear_stiffness", 100.0)),
            "bending_stiffness": float(getattr(settings, "bending_stiffness", 10.0)),
            "solver_iterations": int(getattr(settings, "solver_iterations", 2)),
        }

        # 差分検知: 前回送信したパラメータと完全一致している場合はスキップ
        if self._last_sent_params == params:
            return True

        if self.send_params(**params):
            self._last_sent_params = params
            return True
        return False



    def get_latest_coords(self, timeout: float = 0.5) -> Optional[Tuple[int, np.ndarray, float, float]]:
        """最新の変形座標を取得する (seq, flat_numpy_coords, fps, ms)"""
        if not self._send_packet({"type": "GetLatestCoords"}):
            return None

        start = time.perf_counter()
        resp = None
        while time.perf_counter() - start < timeout:
            resp = self._recv_packet(timeout=0.1)
            if resp and resp.get("type") == "Coords":
                break

        if not resp or resp.get("type") != "Coords":
            return None

        data = resp.get("data", {})
        seq = data.get("frame_seq", 0)
        fps = data.get("physics_fps", 0.0)
        ms = data.get("step_time_ms", 0.0)
        pos_list = data.get("positions", [])

        if not pos_list:
            return None

        self._last_frame_seq = seq
        self._last_physics_fps = fps
        self._last_render_fps = data.get("render_fps", 0.0)
        self._last_step_ms = ms

        arr = np.array(pos_list, dtype=np.float32).flatten()
        return seq, arr, fps, ms

    def get_status(self, timeout: float = 0.5) -> Optional[dict]:
        """軽量シミュレーションステータスを取得する (GPUリードバックなし)"""
        if not self._send_packet({"type": "GetStatus"}):
            return None

        start = time.perf_counter()
        resp = None
        while time.perf_counter() - start < timeout:
            resp = self._recv_packet(timeout=0.1)
            if resp and resp.get("type") == "Status":
                break

        if not resp or resp.get("type") != "Status":
            return None

        data = resp.get("data", {})
        self._last_frame_seq = data.get("current_frame", 0)
        self._last_physics_fps = data.get("physics_fps", 0.0)
        self._last_render_fps = data.get("render_fps", 0.0)
        self._last_step_ms = data.get("step_time_ms", 0.0)
        return data

    def apply_pose(self) -> Optional[np.ndarray]:
        """確定ポーズの座標を取得する"""
        if not self._send_packet({"type": "ApplyPose"}):
            return None

        start = time.perf_counter()
        resp = None
        while time.perf_counter() - start < 2.0:
            resp = self._recv_packet(timeout=0.2)
            if resp and resp.get("type") == "AppliedPose":
                break

        if not resp or resp.get("type") != "AppliedPose":
            return None

        data = resp.get("data", {})
        pos_list = data.get("positions", [])
        if not pos_list:
            return None

        return np.array(pos_list, dtype=np.float32).flatten()

    def send_command(self, cmd_type: str, data: Optional[Dict[str, Any]] = None, timeout: float = 1.0) -> bool:
        packet: Dict[str, Any] = {"type": cmd_type}
        if data is not None:
            packet["data"] = data
        if not self._send_packet(packet):
            return False
        resp = self._recv_packet(timeout=timeout)
        return resp is not None and resp.get("type") == "Ack"

    def play(self) -> bool:
        return self.send_command("Play")

    def pause(self) -> bool:
        return self.send_command("Pause")

    def reset(self) -> bool:
        return self.send_command("Reset")

    def step(self, dt: float = 1.0 / 60.0, substeps: int = 15, solver_iterations: int = 2) -> bool:
        return self.send_command("Step", {"dt": dt, "substeps": substeps, "solver_iterations": solver_iterations})
