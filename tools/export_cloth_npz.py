"""heavy_test.blend から指定布オブジェクトのデータを抽出してnpz保存する。"""
import sys
import numpy as np

addon_path = sys.argv[sys.argv.index("--") + 1]
if addon_path not in sys.path:
    sys.path.insert(0, addon_path)

import bpy  # noqa: E402
import taremin_cloth  # noqa: E402

taremin_cloth.register()
from taremin_cloth.utils.mesh_extract import extract_cloth_mesh_data  # noqa: E402

out_path = sys.argv[sys.argv.index("--") + 2]
name = sys.argv[sys.argv.index("--") + 3]

cloth_obj = bpy.data.objects.get(name)
settings = cloth_obj.taremin_cloth
data = extract_cloth_mesh_data(cloth_obj, settings)
np.savez(out_path,
         positions=np.asarray(data.positions, dtype=np.float32),
         edges=np.asarray(data.normal_edges, dtype=np.uint32),
         faces=np.asarray(data.faces, dtype=np.uint32) if data.faces is not None else np.zeros((0, 3), dtype=np.uint32),
         sewing=(np.asarray(data.sewing_edges, dtype=np.uint32)
                 if data.sewing_edges is not None else np.zeros((0, 2), dtype=np.uint32)),
         inv_masses=np.asarray(data.inv_masses, dtype=np.float32),
         tension_stiffness=float(settings.tension_stiffness),
         compression_stiffness=float(settings.compression_stiffness),
         shear_stiffness=float(settings.shear_stiffness),
         bending_stiffness=float(settings.bending_stiffness),
         thickness=float(settings.thickness),
         areal_density=float(getattr(settings, "areal_density", 0.15)),
         sewing_shrink_speed=float(settings.sewing_shrink_speed),
         sewing_stiffness=float(getattr(settings, "sewing_stiffness", 10000.0)),
         substeps=int(settings.substeps),
         solver_iterations=int(settings.solver_iterations),
         gravity_scale=float(settings.gravity))
print(f"saved {out_path}: verts={len(data.positions)}, sews={len(data.sewing_edges) if data.sewing_edges is not None else 0}")
