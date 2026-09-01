#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use crate::context::GpuContext;
    use crate::mesh::{ClothMesh, GpuMeshTriangle};
    use crate::simulation::GpuClothSimulator;

    #[test]
    fn test_mesh_triangle_collision() {
        let ctx = Arc::new(GpuContext::new().expect("GPU Context creation"));

        // 頂点 (0, 0, 0.5) を自由落下させる
        let positions = vec![[0.0, 0.0, 0.5]];
        let edges = vec![];

        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            None,
            None,
            None,
            None,
            None,
            0,
            0.02,
            10000.0,
            10000.0,
            5000.0,
            0.0,
            1.0,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);

        // z=0 に水平な三角形メッシュコライダーを配置
        let tri = GpuMeshTriangle {
            p0: [-1.0, -1.0, 0.0],
            friction: 0.2,
            p1: [1.0, -1.0, 0.0],
            thickness: 0.02,
            p2: [0.0, 1.0, 0.0],
            restitution: 0.0,
            flags: 0,
            _pad: [0.0; 3],
        };
        sim.set_mesh_triangles(&[tri]);

        for _ in 0..60 {
            sim.step(1.0 / 60.0, 20);
        }

        let verts = sim.read_vertices();
        let z = verts[0].position[2];
        println!("[Test Mesh Triangle Collision] Final Vertex Z: {}", z);

        // 三角形の厚み (0.02) + 布の厚み (0.02) = 0.04 以上で止まること
        assert!(z >= 0.04 - 1e-3, "頂点がメッシュ三角形を貫通してはならない (z={})", z);
    }

    #[test]
    fn test_mesh_triangle_single_sided_recovery() {
        let ctx = Arc::new(GpuContext::new().expect("GPU Context creation"));

        // 頂点を意図的に三角形の裏側（内側 z = -0.1m）にめり込んだ状態で初期化
        let positions = vec![[0.0, 0.0, -0.1]];
        let edges = vec![];

        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            None,
            None,
            None,
            None,
            None,
            0,
            0.02,
            10000.0,
            10000.0,
            5000.0,
            0.0,
            1.0,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);

        // z=0 に法線上向きの片面三角形コライダーを配置 (flags=1: single_sided)
        let tri = GpuMeshTriangle {
            p0: [-1.0, -1.0, 0.0],
            friction: 0.2,
            p1: [1.0, -1.0, 0.0],
            thickness: 0.02,
            p2: [0.0, 1.0, 0.0],
            restitution: 0.0,
            flags: 1, // is_single_sided = true
            _pad: [0.0; 3],
        };
        sim.set_mesh_triangles(&[tri]);

        // シミュレーションを実行（裏側から表側へ押し出されるか検証）
        for _ in 0..10 {
            sim.step(1.0 / 60.0, 20);
        }

        let verts = sim.read_vertices();
        let z = verts[0].position[2];
        println!("[Test Single Sided Recovery] Recovered Vertex Z: {}", z);

        // 片面判定により、裏側から表側 (z >= 0.04 - 1e-3) へ押し戻されていること
        assert!(
            z >= 0.04 - 1e-3,
            "裏側に侵入した頂点が片面リカバリーによって表側へ押し戻されなければならない (z={})",
            z
        );
    }

    #[test]
    fn test_edge_rest_length_scaling() {
        let ctx = Arc::new(GpuContext::new().expect("GPU Context creation"));

        // 2頂点 (0, 0, 0) と (1.0, 0, 0) を結ぶエッジ（初期長 1.0）
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let edges = vec![[0, 1]];
        // 頂点0を固定ピン (inv_mass=0.0)、頂点1を自由頂点 (inv_mass=1.0)
        let inv_masses = vec![0.0, 1.0];

        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            None,
            Some(&inv_masses),
            None,
            None,
            None,
            0,
            0.005,
            10000.0,
            10000.0,
            5000.0,
            0.0,
            1.0,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.gravity = [0.0, 0.0, 0.0]; // 無重力

        // エッジ0のスケールを 0.5 (目標長 0.5) に縮小
        sim.set_edge_rest_length_scales(&[0], &[0.5]);

        // シミュレーションを進める
        for _ in 0..30 {
            sim.step(1.0 / 60.0, 20);
        }

        let verts = sim.read_vertices();
        let p0 = verts[0].position;
        let p1 = verts[1].position;
        let dist = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2) + (p1[2] - p0[2]).powi(2)).sqrt();
        println!("[Test Edge Scaling] Initial: 1.0, Target: 0.5, Result: {}", dist);

        // 許容誤差 5% 以内で 0.5 に収縮していること
        assert!((dist - 0.5).abs() < 0.03, "エッジが目標長 0.5 に収縮していない: dist={}", dist);

        // リセットすると元の長さに戻ること
        sim.reset_edge_rest_lengths();
        assert_eq!(sim.get_edge_rest_length(0), 1.0, "エッジの自然長が 1.0 にリセットされること");
    }

    #[test]
    fn test_self_collision_options_and_untangling() {
        let ctx = Arc::new(GpuContext::new().expect("GPU Context creation"));

        // 4頂点のクアッド
        let positions = vec![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0],
        ];
        let edges = vec![
            [0, 1],
            [1, 2],
            [2, 0],
            [1, 3],
            [3, 2],
        ];
        let faces = vec![
            [0, 1, 2],
            [2, 1, 3],
        ];

        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            Some(&faces),
            None,
            None,
            None,
            None,
            0,
            0.02, // 厚み2cm
            1000.0,
            1000.0,
            1000.0,
            10.0,
            1.0,
        );

        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.gravity = [0.0, 0.0, 0.0];
        sim.set_enable_self_collision(true);

        // 新オプションの設定
        sim.set_self_collision_options(0.1, 0.2, true, true, 256);
        assert_eq!(sim.self_collision_relief_factor, 0.1);
        assert_eq!(sim.self_collision_max_displacement_ratio, 0.2);
        assert!(sim.self_collision_exclude_neighbors);
        assert!(sim.enable_normal_untangling);
        assert_eq!(sim.self_collision_max_iterations, 256);

        // シミュレーションを実行してクラッシュやNaNが生じないこと
        for _ in 0..10 {
            sim.step(1.0 / 60.0, 10);
        }

        let verts = sim.read_vertices();
        for v in &verts {
            for c in v.position {
                assert!(!c.is_nan() && !c.is_infinite(), "頂点座標にNaNまたはInfが含まれてはいけない: {:?}", v.position);
            }
        }
    }
}
