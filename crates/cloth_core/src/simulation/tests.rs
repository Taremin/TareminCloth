#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use crate::context::GpuContext;
    use crate::mesh::{ClothMesh, GpuMeshTriangle};
    use crate::simulation::GpuClothSimulator;

    fn get_test_context() -> Option<Arc<GpuContext>> {
        match GpuContext::new() {
            Ok(c) => Some(Arc::new(c)),
            Err(crate::context::GpuContextError::AdapterNotFound) => {
                eprintln!("警告: GPU アダプタが検出されなかったため、テストをスキップします (CI環境の可能性があります)");
                None
            }
            Err(e) => panic!("GPU Context 作成エラー: {:?}", e),
        }
    }

    #[test]
    fn test_mesh_triangle_collision() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

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
            None,
            None,
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
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

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
            None,
            None,
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
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

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
            None,
            None,
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
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

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
            None,
            None,
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

    #[test]
    fn test_bone_sdf_collision() {
        use half::f16;
        use crate::simulation::types::{GpuBoneInfo, GpuBoneTransform};

        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // z=0.4 から自由落下する頂点
        let positions = vec![[0.0, 0.0, 0.4]];
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
            0.02, // 厚み 0.02
            10000.0,
            10000.0,
            5000.0,
            0.0,
            1.0,
            None,
            None,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);

        // 3D テクスチャ作成: 16x16x16
        // AABB: [-0.5, -0.5, -0.5] ~ [0.5, 0.5, 0.5]
        // z = 0 平面からの距離: d = z
        let size = 16usize;
        let mut tex_data = Vec::with_capacity(size * size * size * 4);
        for k in 0..size {
            let z = -0.5 + (k as f32 + 0.5) / size as f32 * 1.0;
            for _j in 0..size {
                for _i in 0..size {
                    let d = f16::from_f32(z);
                    let alpha = f16::from_f32(1.0);
                    tex_data.extend_from_slice(&d.to_le_bytes());
                    tex_data.extend_from_slice(&alpha.to_le_bytes());
                }
            }
        }

        let bone_info = GpuBoneInfo {
            aabb_min: [-0.5, -0.5, -0.5, 0.0],
            aabb_max: [0.5, 0.5, 0.5, 0.0],
            uvw_scale: [1.0, 1.0, 1.0, 0.05], // blend_k = 0.05
            uvw_offset: [0.5, 0.5, 0.5, 0.0],
            params: [0.3, 0.02, 0.0, 0.0], // friction: 0.3, thickness: 0.02, restitution: 0.0
        };

        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let bone_transform = GpuBoneTransform {
            world_matrix: identity,
            inv_world_matrix: identity,
        };

        sim.set_bone_sdf_colliders(size as u32, size as u32, size as u32, &tex_data, &[bone_info]);
        sim.update_bone_transforms(&[bone_transform]);

        // シミュレーション実行 (自由落下)
        for _ in 0..60 {
            sim.step(1.0 / 60.0, 20);
        }

        let verts = sim.read_vertices();
        let z = verts[0].position[2];
        println!("[Test Bone SDF Collision] Final Vertex Z: {}", z);

        // コライダー厚み (0.02) + 布厚み (0.02) = 0.04 で止まること
        assert!(z >= 0.04 - 2e-3, "頂点がボーンSDF表面 (z=0.04) で止まらなければならない (実測 z={})", z);
        assert!(z <= 0.06, "頂点が浮き上がりすぎてはならない (実測 z={})", z);

        // 動的テスト: ボーンを z=+0.1 移動させた場合、頂点も押し上げられること
        let moved_world = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.1, 1.0],
        ];
        let moved_inv = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -0.1, 1.0],
        ];
        sim.update_bone_transforms(&[GpuBoneTransform {
            world_matrix: moved_world,
            inv_world_matrix: moved_inv,
        }]);

        for _ in 0..30 {
            sim.step(1.0 / 60.0, 20);
        }

        let verts_moved = sim.read_vertices();
        let z_moved = verts_moved[0].position[2];
        println!("[Test Bone SDF Collision] Moved Vertex Z: {}", z_moved);
        assert!(z_moved >= 0.14 - 2e-3, "ボーン移動に伴い頂点が z=0.14 以上に押し上げられること (実測 z={})", z_moved);
    }

    fn make_pin_test_sim(
        ctx: &std::sync::Arc<crate::context::GpuContext>,
    ) -> GpuClothSimulator {
        // 3頂点の鎖 (0-1-2)、両端固定・中央自由。有限剛性で布側の抵抗を作る
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let edges = vec![[0, 1], [1, 2]];
        let inv_masses = vec![0.0, 1.0, 0.0];
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
            1000.0,
            1000.0,
            5000.0,
            0.0,
            1.0,
            None,
            None,
        );
        let mut sim = GpuClothSimulator::new(ctx.clone(), mesh);
        sim.gravity = [0.0, 0.0, 0.0]; // 無重力
        sim
    }

    #[test]
    fn test_compliant_pin_full_hold() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let mut sim = make_pin_test_sim(&ctx);
        // w=1.0: 完全固定で目標へ吸着し、隣接頂点を引き連れる
        sim.set_pin_target(1, [1.0, 0.5, 0.0], 1.0);
        for _ in 0..30 {
            sim.step(1.0 / 60.0, 20);
        }
        let verts = sim.read_vertices();
        let y1 = verts[1].position[1];
        println!("[Test Compliant Pin Full] y1={}", y1);
        assert!((y1 - 0.5).abs() < 1e-2, "w=1.0 hold failed (y1={})", y1);
    }

    #[test]
    fn test_compliant_pin_partial_blend() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let mut sim = make_pin_test_sim(&ctx);
        // w=0.5: 目標と布拘束のブレンド (0 < y < 0.5 の中間) になること
        sim.set_pin_target(1, [1.0, 0.5, 0.0], 0.5);
        for _ in 0..30 {
            sim.step(1.0 / 60.0, 20);
        }
        let verts = sim.read_vertices();
        let y1 = verts[1].position[1];
        println!("[Test Compliant Pin Partial] y1={}", y1);
        assert!(y1 > 1e-3, "w=0.5 must pull toward target (y1={})", y1);
        assert!(y1 < 0.5 - 1e-3, "w=0.5 must stay below target (y1={})", y1);
    }

}
