"""heavy_test.blend から cloth設定値と体メッシュを抽出する。"""
import sys
import numpy as np

addon_path = sys.argv[sys.argv.index("--") + 1]
if addon_path not in sys.path:
    sys.path.insert(0, addon_path)

import bpy  # noqa: E402
import taremin_cloth  # noqa: E402

taremin_cloth.register()

out_path = sys.argv[sys.argv.index("--") + 2]

cloth_obj = bpy.data.objects.get("cloth_body")
s = cloth_obj.taremin_cloth
print("cloth_body settings:")
for k in ("enable_sewing", "sewing_shrink_speed", "sewing_stiffness", "enable_sewing_lock",
          "enable_sewing_priority", "sewing_priority_threshold", "sewing_priority_merge_dist",
          "sewing_priority_ramp_frames", "sewing_priority_max_frames",
          "enable_self_collision", "coupled_self_collision_mode", "post_collision_relaxation_iters",
          "substeps", "solver_iterations", "gravity", "tension_stiffness", "compression_stiffness",
          "shear_stiffness", "bending_stiffness", "thickness", "areal_density",
          "pin_vertex_group", "solver_mode", "enable_adaptive_substep"):
    try:
        print(f"  {k} = {getattr(s, k)}")
    except Exception as e:
        print(f"  {k} = <err {e}>")
print("  vertex_groups =", [vg.name for vg in cloth_obj.vertex_groups])
print("  use_gravity =", bpy.context.scene.use_gravity, "scene.gravity =", tuple(bpy.context.scene.gravity))
print("  fps =", bpy.context.scene.render.fps)

# 体メッシュ (評価済み・ワールド変換済み三角形)
body = bpy.data.objects.get("\u4f53")
depsgraph = bpy.context.evaluated_depsgraph_get()
eval_obj = body.evaluated_get(depsgraph)
mesh = eval_obj.to_mesh()
tris = []
verts = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
mesh.vertices.foreach_get("co", verts)
verts = verts.reshape((-1, 3))
mesh.calc_loop_triangles()
n = len(mesh.loop_triangles)
idx = np.empty(n * 3, dtype=np.int32)
mesh.loop_triangles.foreach_get("vertices", idx)
idx = idx.reshape((-1, 3))
M = np.array(body.matrix_world)
pts = np.concatenate([verts, np.ones((len(verts), 1))], axis=1) @ np.array(M).T
tris = pts[idx][:, :, :3].astype(np.float32)
eval_obj.to_mesh_clear()
np.savez(out_path, triangles=tris)
print(f"saved {out_path}: tris={len(tris)}")
