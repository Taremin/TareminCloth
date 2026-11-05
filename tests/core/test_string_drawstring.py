"""
紐（Drawstring）パーツ統合および質量計算・パス探索の単体テスト (tests/core/test_string_drawstring.py)
Blender非依存で動作するスタンドアロンテスト。
"""

import unittest
import numpy as np
from taremin_cloth.utils.mesh_extract import (
    compute_areal_inv_masses,
    compute_combined_inv_masses,
)
from taremin_cloth.utils.curve_extract import (
    build_ordered_path_across_seams,
    extract_curve_polyline,
    apply_coords_to_curve,
    CurvePartRuntimeData,
)


class TestCombinedInvMasses(unittest.TestCase):
    """面密度と1D線密度の複合質量・逆質量計算テスト"""

    def test_pure_areal_matches_legacy(self):
        """面のみの場合、compute_combined_inv_masses と compute_areal_inv_masses が完全一致すること"""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)

        inv_legacy = compute_areal_inv_masses(positions, faces, areal_density=0.2)
        inv_combined = compute_combined_inv_masses(positions, faces, areal_density=0.2)

        np.testing.assert_allclose(inv_combined, inv_legacy, rtol=1e-5)

    def test_strand_linear_density(self):
        """1Dストランドエッジの線密度による質量分配が正しく機能すること"""
        # 長さ 2m の3頂点2エッジ (0)-(1)-(2)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ], dtype=np.float32)
        strand_edges = np.array([[0, 1], [1, 2]], dtype=np.uint32)
        linear_density = 0.1  # kg/m

        # 頂点0: エッジ(0,1)長 1m の半分 = 0.05kg -> inv_mass = 20.0
        # 頂点1: エッジ(0,1)長 1m の半分 + エッジ(1,2)長 1m の半分 = 0.10kg -> inv_mass = 10.0
        # 頂点2: エッジ(1,2)長 1m の半分 = 0.05kg -> inv_mass = 20.0
        inv_masses = compute_combined_inv_masses(
            positions,
            faces=None,
            strand_edges=strand_edges,
            linear_density=linear_density,
        )

        self.assertAlmostEqual(inv_masses[0], 20.0, places=4)
        self.assertAlmostEqual(inv_masses[1], 10.0, places=4)
        self.assertAlmostEqual(inv_masses[2], 20.0, places=4)


class TestCurvePathExtraction(unittest.TestCase):
    """選択エッジからの順序付きパス追跡および縫合跨ぎジャンプテスト"""

    def test_simple_linear_path(self):
        """単純な一続きのエッジ列から正しい順序の頂点列が得られること"""
        edges = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
        ], dtype=np.uint32)
        # 順不同で渡す
        selected_indices = [1, 0, 2]
        ordered_verts, seam_bridges = build_ordered_path_across_seams(edges, selected_indices)

        self.assertTrue(
            ordered_verts == [0, 1, 2, 3] or ordered_verts == [3, 2, 1, 0],
            f"Expected [0, 1, 2, 3] or reverse, got {ordered_verts}",
        )
        self.assertEqual(len(seam_bridges), 0)

    def test_path_across_seams(self):
        """左右に分かれた2枚の型紙境界エッジが縫合跨ぎで1本に繋がること"""
        # 左型紙: 0 - 1 - 2
        # 右型紙: 3 - 4 - 5
        # 頂点2 と 頂点3 が縫合線で接続されている
        edges = np.array([
            [0, 1],
            [1, 2],
            [3, 4],
            [4, 5],
        ], dtype=np.uint32)
        selected_indices = [0, 1, 2, 3]
        seam_cluster_map = {2: [3], 3: [2]}

        ordered_verts, seam_bridges = build_ordered_path_across_seams(
            edges,
            selected_indices,
            seam_cluster_map=seam_cluster_map,
        )

        expected_forward = [0, 1, 2, 3, 4, 5]
        expected_backward = [5, 4, 3, 2, 1, 0]
        self.assertTrue(
            ordered_verts == expected_forward or ordered_verts == expected_backward,
            f"Expected {expected_forward} or reverse, got {ordered_verts}",
        )
        self.assertIn((2, 3), seam_bridges)


class MockSplinePoint:
    def __init__(self, co, radius=1.0):
        self.co = co
        self.radius = radius


class MockCollection(list):
    def foreach_get(self, attr, out_arr):
        if attr == "co":
            for i, p in enumerate(self):
                if len(p.co) == 4:
                    out_arr[i * 4:(i + 1) * 4] = p.co
                else:
                    out_arr[i * 3:(i + 1) * 3] = p.co
        elif attr == "radius":
            for i, p in enumerate(self):
                out_arr[i] = p.radius

    def foreach_set(self, attr, in_arr):
        if attr == "co":
            for i, p in enumerate(self):
                if len(p.co) == 4:
                    p.co = list(in_arr[i * 4:(i + 1) * 4])
                else:
                    p.co = list(in_arr[i * 3:(i + 1) * 3])


class MockSpline:
    def __init__(self, points, spline_type='POLY'):
        self.type = spline_type
        self.points = MockCollection(points)
        self.bezier_points = MockCollection(points)


class MockCurveData:
    def __init__(self, splines, bevel_depth=0.005):
        self.splines = splines
        self.bevel_depth = bevel_depth


class MockCurveObj:
    def __init__(self, splines, bevel_depth=0.005):
        self.name = "MockCurve"
        self.data = MockCurveData(splines, bevel_depth=bevel_depth)
        self.matrix_world = np.eye(4, dtype=np.float32)


class TestCurveExtractMock(unittest.TestCase):
    """Curveオブジェクトのモック抽出および書き戻しテスト"""

    def test_extract_and_apply(self):
        pts = [
            MockSplinePoint([0.0, 0.0, 0.0, 1.0], radius=1.0),
            MockSplinePoint([1.0, 0.0, 0.0, 1.0], radius=1.0),
            MockSplinePoint([2.0, 0.0, 0.0, 1.0], radius=1.0),
        ]
        spline = MockSpline(pts, spline_type='POLY')
        obj = MockCurveObj([spline], bevel_depth=0.01)

        coords, edges, radii = extract_curve_polyline(obj, apply_world_matrix=False)

        self.assertEqual(len(coords), 3)
        self.assertEqual(len(edges), 2)
        np.testing.assert_allclose(radii, [0.01, 0.01, 0.01])

        # 新しい座標を書き戻す
        new_coords = coords + np.array([0.0, 1.0, 0.0], dtype=np.float32)
        apply_coords_to_curve(obj, new_coords)

        # 反映されていることを確認
        coords_after, _, _ = extract_curve_polyline(obj, apply_world_matrix=False)
        np.testing.assert_allclose(coords_after, new_coords)


class MockVert:
    def __init__(self, index, co):
        self.index = index
        self.co = co
        self.groups = []


class MockEdge:
    def __init__(self, v0, v1):
        self.vertices = (v0, v1)


class MockLoopTri:
    def __init__(self, v0, v1, v2, poly_idx=0):
        self.vertices = (v0, v1, v2)
        self.polygon_index = poly_idx


class MockVertCollection(list):
    def foreach_get(self, attr, out_arr):
        if attr == "co":
            for i, v in enumerate(self):
                out_arr[i * 3:(i + 1) * 3] = v.co


class MockEdgeCollection(list):
    def foreach_get(self, attr, out_arr):
        if attr == "vertices":
            for i, e in enumerate(self):
                out_arr[i * 2] = e.vertices[0]
                out_arr[i * 2 + 1] = e.vertices[1]


class MockLoopTriCollection(list):
    def foreach_get(self, attr, out_arr):
        if attr == "vertices":
            for i, t in enumerate(self):
                out_arr[i * 3:(i + 1) * 3] = t.vertices
        elif attr == "polygon_index":
            for i, t in enumerate(self):
                out_arr[i] = t.polygon_index


class MockMeshData:
    def __init__(self, verts, edges, tris):
        self.vertices = MockVertCollection([MockVert(i, c) for i, c in enumerate(verts)])
        self.edges = MockEdgeCollection([MockEdge(e[0], e[1]) for e in edges])
        self.loop_triangles = MockLoopTriCollection([MockLoopTri(t[0], t[1], t[2], i) for i, t in enumerate(tris)])
        self.polygons = tris
        self.attributes = {}

    def calc_loop_triangles(self):
        pass


class MockStringPartSetting:
    def __init__(self, curve_obj, guide_verts, seam_bridges, tail_left=2, tail_right=2):
        self.enabled = True
        self.curve_object = curve_obj
        self.guide_vert_indices_str = ",".join(str(v) for v in guide_verts)
        self.seam_bridge_pairs_str = ";".join(f"{min(a, b)},{max(a, b)}" for a, b in seam_bridges)
        self.tail_vert_count_left = tail_left
        self.tail_vert_count_right = tail_right
        self.eyelet_stride = 1
        self.linear_density = 0.02
        self.radius = 0.004
        self.pull_length = 0.0
        self.pull_offset_left = 0.0
        self.pull_offset_right = 0.0


class MockClothSettings:
    def __init__(self, string_parts):
        self.is_cloth = True
        self.enabled = True
        self.enable_sewing = True
        self.areal_density = 0.15
        self.thickness = 0.005
        self.layer_id = 0
        self.pin_vertex_group = ""
        self.string_parts = string_parts


class MockClothObj:
    def __init__(self, name, mesh_data, settings):
        self.name = name
        self.type = 'MESH'
        self.data = mesh_data
        self.taremin_cloth = settings
        self.vertex_groups = {}
        self.matrix_world = np.eye(4, dtype=np.float32)


class TestSplitHoodAndDrawstringIntegration(unittest.TestCase):
    """分割型紙の縫合ギャップ跨ぎ・弧長保存・物理引き絞りの結合テスト"""

    def test_split_hood_seam_bridge_excluded_from_channel_length(self):
        """左右離れた分割型紙で、型紙間の巨大ギャップ距離が紐の初期トンネル長や質量に混入しないこと"""
        from taremin_cloth.utils.mesh_extract import extract_cloth_mesh_data

        # 左パネル: 頂点 0,1,2 (x=0.0 ~ 0.1, 長さ0.1m)
        # 右パネル: 頂点 3,4,5 (x=1.0 ~ 1.1, 長さ0.1m, 左パネルから 0.9m 離れている)
        verts = [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [1.0, 0.1, 0.0],
        ]
        # 面を構成するエッジ + 頂点1と3を結ぶ縫合エッジ (ギャップ 0.9m)
        edges = [
            [0, 1], [1, 2], [2, 0],
            [3, 4], [4, 5], [5, 3],
            [1, 3],  # loose sewing edge
        ]
        tris = [[0, 1, 2], [3, 4, 5]]

        # 紐の制御点: 左テール2点 + ガイド4点 (v0, v1, v3, v4) + 右テール2点 = 計8点
        curve_pts = [
            MockSplinePoint([-0.2, 0.0, 0.0, 1.0]),
            MockSplinePoint([-0.1, 0.0, 0.0, 1.0]),
            MockSplinePoint([0.0, 0.0, 0.0, 1.0]),   # v0対応 (idx 2)
            MockSplinePoint([0.1, 0.0, 0.0, 1.0]),   # v1対応 (idx 3)
            MockSplinePoint([1.0, 0.0, 0.0, 1.0]),   # v3対応 (idx 4) -> (3,4)間は0.9mの縫合ギャップ!
            MockSplinePoint([1.1, 0.0, 0.0, 1.0]),   # v4対応 (idx 5)
            MockSplinePoint([1.2, 0.0, 0.0, 1.0]),
            MockSplinePoint([1.3, 0.0, 0.0, 1.0]),
        ]
        curve_obj = MockCurveObj([MockSpline(curve_pts, 'POLY')], bevel_depth=0.004)
        sp_setting = MockStringPartSetting(
            curve_obj=curve_obj,
            guide_verts=[0, 1, 3, 4],
            seam_bridges=[(1, 3)],
            tail_left=2,
            tail_right=2,
        )
        cloth_obj = MockClothObj(
            "SplitHood",
            MockMeshData(verts, edges, tris),
            MockClothSettings([sp_setting]),
        )

        data = extract_cloth_mesh_data(cloth_obj, settings=cloth_obj.taremin_cloth, apply_world_matrix=False)
        self.assertIsNotNone(data.curve_parts)
        self.assertEqual(len(data.curve_parts), 1)
        c_rt = data.curve_parts[0]

        # ギャップ 0.9m が除外され、トンネル初期長は左縁(0.1m) + 右縁(0.1m) = 0.2m であること
        self.assertAlmostEqual(c_rt.initial_channel_len, 0.2, places=4)
        self.assertAlmostEqual(c_rt.initial_left_len, 0.2, places=4)
        self.assertAlmostEqual(c_rt.initial_right_len, 0.2, places=4)

        # ギャップを跨ぐ紐頂点ペア (6+3=9, 6+4=10) が sewing_edges に登録されていること
        sew_set = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in data.sewing_edges}
        self.assertIn((9, 10), sew_set)

    def test_sync_string_parts_arc_length_conservation_and_sewing_priority(self):
        """sync_string_parts が左右オフセットを含めて総弧長を厳密に保存し、縫合優先中は引き絞りを待機すること"""
        from taremin_cloth.engine.cache import set_curve_parts_runtime, clear_simulator_for_object
        from taremin_cloth.engine.params import sync_string_parts

        class DummySim:
            def __init__(self):
                self.priority_active = True
                self.last_indices = None
                self.last_scales = None

            def is_sewing_priority_active(self):
                return self.priority_active

            def set_edge_rest_length_scales(self, indices, scales):
                self.last_indices = indices
                self.last_scales = scales

        c_rt = CurvePartRuntimeData(
            curve_obj_name="StrCurve",
            vertex_offset=10,
            vertex_count=7,
            edge_offset=20,
            left_tail_edge_indices=[20, 21],
            channel_edge_indices=[22, 23],
            right_tail_edge_indices=[24, 25],
            seam_bridge_edge_indices=[],
            initial_left_len=0.20,
            initial_channel_len=0.40,
            initial_right_len=0.20,
            guide_cloth_verts=[0, 1, 2],
            eyelet_pairs=[(0, 12), (2, 14)],
        )
        sp = MockStringPartSetting(MockCurveObj([]), [0, 1, 2], [])
        sp.pull_length = 0.16
        sp.pull_offset_left = 0.03
        sp.pull_offset_right = -0.01

        obj = MockClothObj("HoodTest", None, MockClothSettings([sp]))
        set_curve_parts_runtime(obj.name, [c_rt])
        sim = DummySim()

        try:
            # 1. 縫合優先モード中は引き絞りが抑止され、全スケールが 1.0 であること
            sync_string_parts(sim, obj)
            np.testing.assert_allclose(sim.last_scales, np.ones(6, dtype=np.float32), atol=1e-5)

            # 2. 縫合優先モード完了後、収束するまで同期を回すと総弧長が厳密に保存されること
            sim.priority_active = False
            for _ in range(40):
                sync_string_parts(sim, obj)

            scale_map = dict(zip(sim.last_indices.tolist(), sim.last_scales.tolist()))
            c_scale = scale_map[22]
            l_scale = scale_map[20]
            r_scale = scale_map[24]

            total_init = c_rt.initial_channel_len + c_rt.initial_left_len + c_rt.initial_right_len
            total_curr = (
                c_rt.initial_channel_len * c_scale
                + c_rt.initial_left_len * l_scale
                + c_rt.initial_right_len * r_scale
            )
            self.assertAlmostEqual(total_curr, total_init, places=5)
            # 左テールの方が右テールより長く引き出されていること
            self.assertGreater(c_rt.initial_left_len * l_scale, c_rt.initial_right_len * r_scale)
        finally:
            clear_simulator_for_object(obj.name)

    def test_cloth_simulator_drawstring_cinch_gathers_cloth(self):
        """実GPUシミュレータ上で紐を引き絞った際、アイレット間の布が座屈してギャザーとなり弧長を保ったまま幅が縮むこと"""
        import taremin_cloth_core

        # 布メッシュ: X方向に並ぶ 6セグメント (7列 x 2行 = 14頂点, 全長 0.6m, 幅 0.1m)
        # アイレット間引き (stride=2: v=0, 2, 4, 6 を紐に結合し、間の v=1, 3, 5 は自由頂点として座屈を許容)
        cloth_x = np.linspace(0.0, 0.6, 7, dtype=np.float32)
        cloth_verts = []
        for row_idx, y in enumerate([0.0, 0.1]):
            for col_idx, x in enumerate(cloth_x):
                # 対称破れ（オイラー座屈のトリガー）として奇数頂点に微小なZ変位(2mm)を付与
                z_pert = 0.002 if (col_idx % 2 == 1) else 0.0
                cloth_verts.append([x, y, z_pert])
        cloth_verts = np.array(cloth_verts, dtype=np.float32)

        faces = []
        edges_set = set()
        for i in range(6):
            v00, v10 = i, i + 1
            v01, v11 = i + 7, i + 8
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
            for a, b in [(v00, v10), (v10, v11), (v11, v00), (v00, v11), (v11, v01), (v01, v00)]:
                edges_set.add((min(a, b), max(a, b)))
        cloth_edges = np.array(sorted(list(edges_set)), dtype=np.uint32)
        faces_arr = np.array(faces, dtype=np.uint32)
        n_cloth_edges = len(cloth_edges)

        # 紐ストランド: 左テール2点 + トンネル4点 (x=0.0, 0.2, 0.4, 0.6) + 右テール2点 = 8頂点 (v=14..21)
        strand_x = np.array([-0.2, -0.1, 0.0, 0.2, 0.4, 0.6, 0.7, 0.8], dtype=np.float32)
        strand_verts = np.column_stack([strand_x, np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32)])
        strand_edges = np.column_stack([np.arange(14, 21, dtype=np.uint32), np.arange(15, 22, dtype=np.uint32)])

        all_verts = np.vstack([cloth_verts, strand_verts])
        all_edges = np.vstack([cloth_edges, strand_edges])

        # アイレット間引き結合 (stride=2): 布下辺 v=0, 2, 4, 6 と 紐トンネル v=16, 17, 18, 19 を連結
        sewing_springs = np.array([
            [0, 16],
            [2, 17],
            [4, 18],
            [6, 19],
        ], dtype=np.uint32)

        inv_masses = compute_combined_inv_masses(
            all_verts,
            faces=faces_arr,
            strand_edges=strand_edges,
            areal_density=0.15,
            linear_density=0.05,
        )
        sim = taremin_cloth_core.ClothSimulator(
            positions=all_verts,
            edges=all_edges,
            faces=faces_arr,
            inv_masses=inv_masses,
            sewing_springs=sewing_springs,
            stiffness=5000.0,
            compression_stiffness=5000.0,
            bending_stiffness=0.05,
            sewing_stiffness=10000.0,
            sewing_shrink_speed=5.0,
            enable_sewing_lock=True,
        )
        sim.set_gravity(0.0, 0.0, 0.0)

        # 紐のトンネルエッジ (n_cloth_edges + 2 .. n_cloth_edges + 4 の3本, 初期長0.6m) を 0.5倍 (0.3m) に引き絞り、
        # 縮んだ 0.3m 分を左右テール (各2本, 初期長0.2m -> 0.35m, 1.75倍) へ転送
        left_tail_e = [n_cloth_edges + 0, n_cloth_edges + 1]
        channel_e = [n_cloth_edges + 2, n_cloth_edges + 3, n_cloth_edges + 4]
        right_tail_e = [n_cloth_edges + 5, n_cloth_edges + 6]

        scale_indices = np.array(left_tail_e + channel_e + right_tail_e, dtype=np.uint32)
        scales = np.array([1.75, 1.75] + [0.5, 0.5, 0.5] + [1.75, 1.75], dtype=np.float32)
        sim.set_edge_rest_length_scales(scale_indices, scales)

        out_coords = np.empty(len(all_verts) * 3, dtype=np.float32)
        for _ in range(40):
            sim.step(dt=1.0 / 60.0, substeps=20, solver_iterations=2)
        sim.get_positions(out_coords)
        pos_after = out_coords.reshape((-1, 3))

        # 1. 布の両端 (v=0 と v=6) の直線距離が初期の 0.6m から引き絞られて大幅に縮んでいること
        cloth_span = float(np.linalg.norm(pos_after[6] - pos_after[0]))
        self.assertLess(cloth_span, 0.36, f"Expected cloth span < 0.36m after cinching, got {cloth_span:.4f}m")

        # 2. 布自体の下辺弧長（各エッジ長の総和）は自然長0.6mを維持していること（ギャザー座屈により布面積が保たれている証明）
        cloth_arc_len = sum(float(np.linalg.norm(pos_after[i + 1] - pos_after[i])) for i in range(6))
        self.assertGreater(cloth_arc_len, 0.55, f"Cloth edge arc length should remain ~0.6m, got {cloth_arc_len:.4f}m")

        # 3. アイレット間の自由頂点 (v=1, 3, 5) が面外（Z方向）へ大きく波打ってシワ（ギャザー）を形成していること
        z_arch = max(abs(float(pos_after[1, 2])), abs(float(pos_after[3, 2])), abs(float(pos_after[5, 2])))
        self.assertGreater(z_arch, 0.02, f"Expected out-of-plane gather buckling > 2cm, got {z_arch:.4f}m")


if __name__ == "__main__":
    unittest.main()
