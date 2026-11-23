#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use crate::context::GpuContext;
    use crate::mesh::{ClothMesh, GpuMeshTriangle};
    use crate::simulation::GpuClothSimulator;

    /// テスト用 GPU コンテキストをプロセス内で共有する。
    /// 本番 (`GLOBAL_CONTEXT`) と同様に単一デバイスを使い回すことで、
    /// 共有パイプラインキャッシュのヒット経路も検証対象に含める。
    static TEST_CONTEXT: Mutex<Option<Arc<GpuContext>>> = Mutex::new(None);

    fn get_test_context() -> Option<Arc<GpuContext>> {
        let mut guard = TEST_CONTEXT.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(ctx) = guard.as_ref() {
            return Some(Arc::clone(ctx));
        }
        match GpuContext::new() {
            Ok(c) => {
                let ctx = Arc::new(c);
                *guard = Some(Arc::clone(&ctx));
                Some(ctx)
            }
            Err(crate::context::GpuContextError::AdapterNotFound) => {
                eprintln!("警告: GPU アダプタが検出されなかったため、テストをスキップします (CI環境の可能性があります)");
                None
            }
            Err(e) => panic!("GPU Context 作成エラー: {:?}", e),
        }
    }

    fn make_quad_mesh() -> ClothMesh {
        let positions = vec![
            [0.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 0.5],
        ];
        let edges = vec![[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]];
        let faces = vec![[0, 1, 2], [0, 2, 3]];
        ClothMesh::from_raw(
            &positions,
            &edges,
            Some(&faces),
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
        )
    }

    #[test]
    fn test_shared_pipeline_cache_hit_and_lazy() {
        use crate::simulation::pipeline_cache::{
            cache_info, get_or_create_shared, normalize_workgroup_size,
        };

        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // workgroup_size 正規化
        assert_eq!(normalize_workgroup_size(32), 32);
        assert_eq!(normalize_workgroup_size(64), 64);
        assert_eq!(normalize_workgroup_size(99), 32);

        // 同一キーでは同一 Arc が返る (共有ヒット)
        let (s1, _) = get_or_create_shared(&ctx, 32);
        let (s2, built2) = get_or_create_shared(&ctx, 32);
        assert!(Arc::ptr_eq(&s1, &s2));
        assert!(!built2, "2 回目はキャッシュヒットでなければならない");

        // workgroup_size 違いでは別エントリ
        let (s64, _) = get_or_create_shared(&ctx, 64);
        assert!(!Arc::ptr_eq(&s1, &s64));

        // 遅延 ensure は冪等 (同一オブジェクトを返す)
        let p1 = s1.ensure_pair() as *const _;
        let p2 = s1.ensure_pair() as *const _;
        assert_eq!(p1, p2);
        let a1 = s1.ensure_atomic() as *const _;
        assert_eq!(a1, s1.ensure_atomic() as *const _);
        let e1 = s1.ensure_edge() as *const _;
        assert_eq!(e1, s1.ensure_edge() as *const _);
        let sc = s1.ensure_self_collision();
        assert_eq!(sc as *const _, s1.ensure_self_collision() as *const _);
        let h = s1.ensure_hash();
        assert_eq!(h as *const _, s1.ensure_hash() as *const _);

        // 遅延ビルドが計測ログに記録される
        let names: Vec<String> = s1
            .timings_snapshot()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        for want in [
            "predict",
            "distance",
            "pair_collect",
            "pair_solve_vt",
            "pair_solve_ee",
            "hash_clear",
        ] {
            assert!(names.contains(&want.to_string()), "計測ログに {} が必要", want);
        }

        // 2 つのシミュレータが同一 Shared を参照する
        let sim1 = GpuClothSimulator::with_options(Arc::clone(&ctx), make_quad_mesh(), 32, 0);
        let sim2 = GpuClothSimulator::with_options(Arc::clone(&ctx), make_quad_mesh(), 32, 0);
        assert!(Arc::ptr_eq(&sim1.shared, &sim2.shared));
        assert!(!sim1.build_timings().is_empty());
        assert!(!sim2.build_timings().is_empty());

        // 診断 API がエントリを返す
        assert!(!cache_info().is_empty());
    }

    #[test]
    fn test_lazy_pipeline_mid_sim_enable() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let mut sim = GpuClothSimulator::with_options(Arc::clone(&ctx), make_quad_mesh(), 32, 0);

        // 途中有効化の各パスがコンパイル・実行できること
        sim.set_enable_self_collision(true);
        sim.set_enable_pair_cache(true);
        sim.set_enable_edge_collision(true);
        sim.set_solver_mode(1);
        sim.step(1.0 / 60.0, 2);
        sim.set_solver_mode(0);
        sim.set_enable_edge_collision(false);
        sim.set_enable_pair_cache(false);
        sim.set_enable_self_collision(false);
        sim.step(1.0 / 60.0, 2);
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
    fn test_single_layer_disables_effective_untangling() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // 単一レイヤー: ユーザー設定ONでも実効OFF (法線パス省略・結果は同一)
        let sim = GpuClothSimulator::with_options(Arc::clone(&ctx), make_quad_mesh(), 32, 0);
        assert!(!sim.has_multiple_layers, "単一quadは単一レイヤーでなければならない");
        assert!(sim.enable_normal_untangling, "ユーザー設定の既定はONのまま");
        assert!(!sim.effective_normal_untangling(), "単一レイヤーでは実効OFFでなければならない");

        // 複数レイヤー: 実効ONを維持
        let positions = vec![
            [0.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [1.0, 1.0, 0.5],
            [0.0, 1.0, 0.5],
        ];
        let edges = vec![[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]];
        let faces = vec![[0, 1, 2], [0, 2, 3]];
        let layers = vec![0u32, 0, 1, 1];
        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            Some(&faces),
            None,
            None,
            Some(&layers),
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
        let sim2 = GpuClothSimulator::with_options(Arc::clone(&ctx), mesh, 32, 0);
        assert!(sim2.has_multiple_layers, "レイヤー混在は複数レイヤーと判定されなければならない");
        assert!(sim2.effective_normal_untangling(), "複数レイヤーでは実効ONを維持しなければならない");

        // OFF設定時は複数レイヤーでも実効OFF
        let mut sim3 = sim2;
        sim3.set_self_collision_options(0.2, 0.2, true, false, 128);
        assert!(!sim3.effective_normal_untangling());
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
        for _ in 0..150 {
            sim.step(1.0 / 60.0, 20);
        }
        let verts = sim.read_vertices();
        let y1 = verts[1].position[1];
        println!("[Test Compliant Pin Partial] y1={}", y1);
        assert!(y1 > 1e-3, "w=0.5 must pull toward target (y1={})", y1);
        assert!(y1 < 0.5 - 1e-3, "w=0.5 must stay below target (y1={})", y1);
    }

    fn make_sewing_constraint(v0: u32, v1: u32) -> crate::mesh::GpuSewingConstraint {
        crate::mesh::GpuSewingConstraint {
            v0,
            v1,
            current_rest_len: 1.0,
            target_rest_len: 0.0,
            shrink_speed: 1.0,
            compliance: 0.0,
            lock_on_close: 1.0,
            _pad1: 0.0,
        }
    }

    #[test]
    fn test_sewing_closure_ratio_pure() {
        // GPU不要の純粋関数試験
        let sc = vec![make_sewing_constraint(0, 1), make_sewing_constraint(2, 3)];
        let pos = vec![
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0], // 結合 (1mm)
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], // 非結合
        ];
        let ratio = GpuClothSimulator::compute_sewing_closure_ratio(&pos, &sc, 0.005);
        assert!((ratio - 0.5).abs() < 1e-6, "ratio must be 0.5 (got {})", ratio);
        // 縫合なしは恒等 1.0
        let empty: Vec<crate::mesh::GpuSewingConstraint> = vec![];
        assert_eq!(GpuClothSimulator::compute_sewing_closure_ratio(&pos, &empty, 0.005), 1.0);
        // 範囲外インデックスはスキップされ、分母から除外される
        let bad = vec![make_sewing_constraint(0, 99)];
        assert_eq!(GpuClothSimulator::compute_sewing_closure_ratio(&pos, &bad, 0.005), 1.0);
    }

    #[test]
    fn test_sewing_priority_latch_and_ramp() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        // 2頂点・縫合1本の最小メッシュ
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let edges: Vec<[u32; 2]> = vec![];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, None, None, Some(&[[0u32, 1u32]]),
            None, None, 0, 0.005, 10000.0, 10.0, 5000.0, 0.0, 1.0, None, None,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        assert_eq!(sim.num_sewing_constraints, 1);
        sim.set_sewing_priority_options(true, 0.9, 0.005, 2, 0);

        // 未結合: スケール0
        let (r0, s0, lat0) = sim.update_sewing_priority_from_positions(&positions);
        assert_eq!(r0, 0.0);
        assert_eq!(s0, 0.0);
        assert!(!lat0);
        assert!(sim.is_sewing_priority_active());

        // 結合: ラッチし、ランプ1段目は smoothstep(0.5)=0.5
        let closed = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]];
        let (r1, s1, lat1) = sim.update_sewing_priority_from_positions(&closed);
        assert_eq!(r1, 1.0);
        assert!(lat1);
        assert!((s1 - 0.5).abs() < 1e-6, "ramp step1 must be 0.5 (got {})", s1);

        // 2段目で復帰完了
        let (_, s2, _) = sim.update_sewing_priority_from_positions(&closed);
        assert!((s2 - 1.0).abs() < 1e-6, "ramp step2 must be 1.0 (got {})", s2);

        // 片道ラッチ: 再び離れても戻らない
        let (_, s3, lat3) = sim.update_sewing_priority_from_positions(&positions);
        assert!(lat3);
        assert_eq!(s3, 1.0);
    }

    #[test]
    fn test_sewing_priority_max_frames_fallback() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let edges: Vec<[u32; 2]> = vec![];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, None, None, Some(&[[0u32, 1u32]]),
            None, None, 0, 0.005, 10000.0, 10.0, 5000.0, 0.0, 1.0, None, None,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        // 閾値1.0 (到達不能) + 上限2フレーム + 即時復帰
        sim.set_sewing_priority_options(true, 1.0, 0.005, 0, 2);
        let (_, s0, lat0) = sim.update_sewing_priority_from_positions(&positions);
        assert!(!lat0);
        assert_eq!(s0, 0.0);
        let (_, s1, lat1) = sim.update_sewing_priority_from_positions(&positions);
        assert!(lat1, "max_frames fallback must latch");
        assert_eq!(s1, 1.0);
    }

    #[test]
    fn test_sewing_priority_disabled_identity() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let edges: Vec<[u32; 2]> = vec![];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, None, None, Some(&[[0u32, 1u32]]),
            None, None, 0, 0.005, 10000.0, 10.0, 5000.0, 0.0, 1.0, None, None,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        // 既定OFF: 恒等動作
        assert_eq!(sim.sewing_priority_scale(), 1.0);
        assert!(!sim.is_sewing_priority_active());
        let (r, s, lat) = sim.update_sewing_priority_from_positions(&positions);
        assert_eq!((r, s, lat), (1.0, 1.0, true));
    }

    #[test]
    fn test_sewing_priority_wrinkle_field_suppression() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let edges: Vec<[u32; 2]> = vec![];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, None, None, Some(&[[0u32, 1u32]]),
            None, None, 0, 0.005, 10000.0, 10.0, 5000.0, 0.0, 1.0, None, None,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_enable_wrinkle_field(true);
        sim.set_sewing_priority_options(true, 0.9, 0.005, 2, 0);

        // 1. 未結合 (priority_scale == 0.0): 縫合フェーズ中はドレープガイド外力がスキップされる
        let (_, s0, lat0) = sim.update_sewing_priority_from_positions(&positions);
        assert_eq!(s0, 0.0);
        assert!(!lat0);
        sim.step(0.016, 2);

        // 2. 結合 (priority_scale -> 0.5 -> 1.0): ランプおよび完全復帰後も正常動作
        let closed = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]];
        let (_, s1, lat1) = sim.update_sewing_priority_from_positions(&closed);
        assert!(lat1);
        assert!((s1 - 0.5).abs() < 1e-6);
        sim.step(0.016, 2);

        let (_, s2, _) = sim.update_sewing_priority_from_positions(&closed);
        assert!((s2 - 1.0).abs() < 1e-6);
        sim.step(0.016, 2);
    }

    #[test]
    fn test_edge_collision_ee_pipeline() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // 水平な布エッジ (X軸方向: [-0.5, 0.0, 0.0] -> [0.5, 0.0, 0.0])
        let positions = vec![
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ];
        let edges = vec![[0, 1]];
        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            None,
            None,
            None,
            None,
            None,
            0,
            0.02, // 厚み 20mm
            10000.0,
            10000.0,
            5000.0,
            0.0,
            1.0,
            None,
            None,
        );

        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_enable_edge_collision(true);

        // Y軸方向のコライダーエッジを持つ三角形メッシュを追加
        // 直下 (Z = -0.01) に十字交差するよう配置
        let tri = GpuMeshTriangle {
            p0: [0.0, -0.5, -0.01],
            friction: 0.1,
            p1: [0.0, 0.5, -0.01],
            thickness: 0.02,
            p2: [0.5, 0.0, -0.01],
            restitution: 0.0,
            flags: 0,
            _pad: [0.0; 3],
        };
        sim.set_mesh_triangles(&[tri]);

        // コライダー稜線が抽出され、空間ハッシュに登録されていることを確認
        assert_eq!(sim.collider_edge_hash.num_edges, 3);

        // 重力を切って純粋なエッジ押し出しを検証
        sim.gravity = [0.0, 0.0, 0.0];

        // 1ステップ実行
        sim.step(0.016, 5);

        let verts = sim.read_vertices();
        // 布エッジの中点がコライダーエッジから離れる方向 (Z+) に押し出されていることを確認
        let mid_z = (verts[0].position[2] + verts[1].position[2]) * 0.5;
        assert!(mid_z > 0.0, "エッジ同士の接触により布がZ+方向に押し出されるべき (mid_z = {})", mid_z);
    }

    #[test]
    fn test_vertex_vertex_self_collision() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // 2つの対向する独立した三角形パッチ
        // パッチ1 (Z=0.0): (0,0,0), (0.1, 0, 0), (0, 0.1, 0)
        // パッチ2 (Z=0.015): (0,0,0.015), (0.1, 0, 0.015), (0, 0.1, 0.015)
        // 厚み thickness = 0.02 のため、初期距離 0.015 は衝突範囲内 (min_dist = 0.04)
        let positions = vec![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.015],
            [0.1, 0.0, 0.015],
            [0.0, 0.1, 0.015],
        ];
        let edges = vec![
            [0, 1], [1, 2], [2, 0],
            [3, 4], [4, 5], [5, 3],
        ];
        let faces = vec![[0, 1, 2], [3, 4, 5]];
        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            Some(&faces),
            None,
            None,
            None,
            None,
            0,
            0.02,
            1000.0,
            10.0,
            10.0,
            0.0,
            1.0,
            None,
            None,
        );

        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.gravity = [0.0, 0.0, 0.0];
        sim.set_enable_self_collision(true);
        // V-V モードに設定
        sim.set_self_collision_algorithm(2);
        assert_eq!(sim.get_self_collision_algorithm(), 2);

        // 仮想頂点が生成されていることを確認
        assert!(sim.num_virtual_vertices > 0, "仮想頂点が生成されているべき (num_virtual = {})", sim.num_virtual_vertices);

        // 1ステップ実行
        sim.step(0.016, 5);

        let verts = sim.read_vertices();
        assert_eq!(verts.len(), 6);

        // パッチ1のZ座標は下(Z-)へ、パッチ2のZ座標は上(Z+)へ押し離されていることを確認
        let z_lower = (verts[0].position[2] + verts[1].position[2] + verts[2].position[2]) / 3.0;
        let z_upper = (verts[3].position[2] + verts[4].position[2] + verts[5].position[2]) / 3.0;
        let final_sep = z_upper - z_lower;
        assert!(final_sep > 0.015, "V-V自己衝突により対向パッチが押し離されるべき (初期0.015 -> 実行後{})", final_sep);
    }

    #[test]
    fn test_adaptive_substeps_cfl_and_gravity_floor() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let positions = vec![
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
        ];
        let edges = vec![[0, 1], [1, 2], [2, 0]];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, None, None, None,
            None, None, 0, 0.005, 1000.0, 10.0, 1000.0, 0.0, 1.0, None, None,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_adaptive_substep_options(true, 4, 64);
        sim.base_substeps = 20;

        // 1. 静止時: 重力があっても変位がゼロなら min_substeps (4) に収束する（無駄なGPU負荷を排除）
        sim.gravity = [0.0, 0.0, -9.81];
        sim.last_step_max_displacement = 0.0;
        sim.effective_substeps = 4;
        let s = sim.compute_effective_substeps(0.016, 20);
        assert_eq!(s, 4, "静止時はmin_substepsになるべき (got {})", s);

        // 2. 高速移動時: 変位20cm (cfl_margin 5mm に対して40ステップ要求) で即座に40へ上昇
        sim.last_step_max_displacement = 0.20;
        let s_fast = sim.compute_effective_substeps(0.016, 20);
        assert_eq!(s_fast, 40, "危険変位時は安全重視で即座に40へ上昇すべき (got {})", s_fast);

        // 3. 上限クランプの厳守: max_substeps を 25 に設定した場合、40要求でも25で確実にキャップされること
        sim.set_adaptive_substep_options(true, 4, 25);
        sim.effective_substeps = 20;
        sim.last_step_max_displacement = 0.20;
        let s_clamped = sim.compute_effective_substeps(0.016, 20);
        assert_eq!(s_clamped, 25, "ユーザー指定のmax_substeps(25)を絶対に超えてはならない (got {})", s_clamped);

        // 4. 急停止時: ヒステリシス下降リミット(-2)により 25 -> 23 へ滑らかに減少
        sim.last_step_max_displacement = 0.0;
        let s_slow = sim.compute_effective_substeps(0.016, 20);
        assert_eq!(s_slow, 23, "ヒステリシス下降リミット(-2)により25->23になるべき (got {})", s_slow);
    }

    #[test]
    fn test_adaptive_substeps_preserve_history_on_config_update() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let positions = vec![
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
        ];
        let edges = vec![[0, 1], [1, 2], [2, 0]];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, None, None, None,
            None, None, 0, 0.005, 1000.0, 10.0, 1000.0, 0.0, 1.0, None, None,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_adaptive_substep_options(true, 4, 40);
        sim.effective_substeps = 20;

        // 適応有効のまま他設定（apply_config 相当）や min/max 更新が入っても履歴が保持・クランプされること
        let mut cfg = sim.export_config();
        cfg.gravity = [0.0, 0.0, 0.0];
        cfg.max_substeps = 18;
        sim.apply_config(&cfg);
        assert_eq!(sim.effective_substeps(), 18, "有効継続中の設定更新では履歴を保持し新範囲へクランプするべき");

        // 無効化時は履歴が 0 にリセットされ、無効期間中の外部変位も再有効化後に持ち越されないこと
        sim.set_adaptive_substep_options(false, 4, 40);
        assert_eq!(sim.effective_substeps(), 0);
        sim.note_external_displacement(0.50);
        sim.set_adaptive_substep_options(true, 4, 40);
        assert_eq!(sim.effective_substeps(), 0);
        sim.step(1.0 / 60.0, 0);
        assert_eq!(sim.effective_substeps(), 4, "無効期間中の外部変位は再有効化時にクリアされるべき");
    }

    #[test]
    fn test_adaptive_substeps_step_loop() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let positions = vec![
            [0.0, 0.0, 1.0],
            [0.05, 0.0, 1.0],
            [0.0, 0.05, 1.0],
        ];
        let edges = vec![[0, 1], [1, 2], [2, 0]];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, None, None, None,
            None, None, 0, 0.005, 1000.0, 10.0, 1000.0, 0.0, 1.0, None, None,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_adaptive_substep_options(true, 4, 40);
        sim.base_substeps = 10;
        sim.gravity = [0.0, 0.0, -9.81];

        // 1ステップ目 (落下開始前): 静止時は min_substeps (4)
        sim.step(0.016, 10);
        assert_eq!(sim.effective_substeps(), 4);

        // 自由落下により変位が増加し、ステップ数が上限方向に追従することを確認
        for _ in 0..10 {
            sim.step(0.016, 10);
        }
        assert!(sim.effective_substeps() >= 4, "自由落下中は min 以上を維持 (got {})", sim.effective_substeps());
    }

    #[test]
    fn test_strain_driven_adaptive_substep() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let positions = vec![
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
        ];
        let edges = vec![[0, 1], [1, 2], [2, 0]];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, None, None, None,
            None, None, 0, 0.005, 1000.0, 10.0, 1000.0, 0.0, 1.0, None, None,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_adaptive_substep_options(true, 4, 32);
        sim.base_substeps = 20;
        sim.solver_iterations = 2;
        sim.set_auto_compensate_iterations(true);
        sim.set_strain_adaptive_options(true, 0.008); // 0.8% 許容

        // 1. 歪みが正常範囲内 (0.2%): 通常の反復数補償 (target_solves 20 / 4 = 5)
        sim.last_step_max_strain = 0.002;
        let iters_normal = sim.compute_effective_iterations(4);
        assert_eq!(iters_normal, 5, "正常歪み時は通常の伝播補償値 (got {})", iters_normal);

        // 2. 歪みが超過 (1.6% = 許容値の2.0倍): 歪みブーストが発動 (target_solves 20*2 = 40 / 4 = 10 -> maxクランプ 8)
        sim.last_step_max_strain = 0.016;
        let iters_boosted = sim.compute_effective_iterations(4);
        assert_eq!(iters_boosted, 8, "歪み超過時は反復数が引き上げられ上限クランプ (8) に達するべき (got {})", iters_boosted);

        // solver_iterations = 4 の場合 (max_iters = 16)
        sim.solver_iterations = 4;
        let iters_strain_boost = sim.compute_effective_iterations(4);
        assert!(iters_strain_boost > 5, "歪み超過時はイテレーション数が引き上げられるべき (got {})", iters_strain_boost);

        // 3. サブステップ数の歪みフロア引き上げ: 深刻な歪み超過 (0.016 / 0.008 = 2.0倍)
        sim.last_step_max_displacement = 0.0;
        sim.effective_substeps = 4;
        let s_strain = sim.compute_effective_substeps(0.016, 20);
        assert!(s_strain >= 6, "歪み深刻超過時はサブステップ数も引き上げられるべき (got {})", s_strain);

        // 4. 無効時 (enable_strain_adaptive = false): 歪み超過があってもブーストされない
        sim.set_strain_adaptive_options(false, 0.008);
        sim.effective_substeps = 4;
        let s_disabled = sim.compute_effective_substeps(0.016, 20);
        assert_eq!(s_disabled, 4, "無効時は歪み超過による引き上げが発生しないべき (got {})", s_disabled);
    }

    #[test]
    fn test_large_pin_buffer_resize_and_batch_release() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // 10,000 頂点のメッシュを作成 (初期確保が 10,000 以上になる)
        let num_verts = 10000;
        let positions = vec![[0.0, 0.0, 0.0]; num_verts];
        let edges = vec![[0, 1]];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, None, None, None,
            None, None, 0, 0.01, 1000.0, 10.0, 1000.0, 0.0, 1.0, None, None,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);

        // 初期バッファ容量は 10,000 以上
        assert!(sim.pin_buffer_capacity >= 10000);

        // 9,000 個のピンを一括設定 (オーバーランせず正常終了)
        let pin_indices: Vec<u32> = (0..9000).collect();
        let pin_targets: Vec<[f32; 3]> = vec![[1.0, 2.0, 3.0]; 9000];
        let pin_weights: Vec<f32> = vec![1.0; 9000];
        sim.set_pin_targets_batch(&pin_indices, &pin_targets, &pin_weights);
        assert_eq!(sim.dynamic_pins.len(), 9000);

        // 一括解除 (release_pins_batch)
        sim.release_pins_batch(&pin_indices);
        assert_eq!(sim.dynamic_pins.len(), 0);

        // さらにバッファ容量を超えるピン設定による動的自動拡張を検証
        let huge_count = (sim.pin_buffer_capacity + 1000) as u32;
        for i in 0..huge_count {
            sim.dynamic_pins.insert(
                i,
                crate::mesh::GpuPinConstraint {
                    vertex_idx: 0,
                    weight: 1.0,
                    _pad: [0.0; 2],
                    target_pos: [0.0, 0.0, 0.0],
                    _pad2: 0.0,
                },
            );
        }
        let old_capacity = sim.pin_buffer_capacity;
        sim.upload_pins();
        // 容量が2のべき乗に拡張されていること
        assert!(sim.pin_buffer_capacity > old_capacity);
        assert!(sim.pin_buffer_capacity >= huge_count as usize);

        sim.clear_dynamic_pins();
        assert_eq!(sim.dynamic_pins.len(), 0);
    }

    #[test]
    fn test_boundary_ee_ccd_blocking() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // 正中面 X=0 を挟んで対向する2枚の独立三角形パネル
        // パネルA (X < 0): 辺 0-1 が X=-0.03 の開放境界エッジ
        // パネルB (X > 0): 辺 3-4 が X=+0.03 の開放境界エッジ
        let positions = vec![
            [-0.03, 0.0, 0.0],  // 0
            [-0.03, 0.1, 0.0],  // 1
            [-0.10, 0.05, 0.0], // 2
            [0.03, 0.0, 0.0],   // 3
            [0.03, 0.1, 0.0],   // 4
            [0.10, 0.05, 0.0],  // 5
        ];
        let edges = vec![
            [0, 1], [1, 2], [2, 0],
            [3, 4], [4, 5], [5, 3],
        ];
        let faces = vec![
            [0, 1, 2],
            [3, 4, 5],
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
            0.01,    // thickness = 1cm (半径 5mm)
            10000.0,
            10000.0,
            5000.0,
            0.0,
            1.0,
            None,
            None,
        );

        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_enable_self_collision(true);
        sim.set_gravity(0.0, 0.0, 0.0); // 無重力

        // 1 サブステップで相手を完全に突き抜ける大速度を設定 (+5.0 m/s と -5.0 m/s)
        let velocities = vec![
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [-5.0, 0.0, 0.0],
            [-5.0, 0.0, 0.0],
            [-5.0, 0.0, 0.0],
        ];
        sim.set_positions_and_velocities(&positions, Some(&velocities));

        // 1 ステップ実行 (substeps = 10)
        sim.step(1.0 / 60.0, 10);

        let verts = sim.read_vertices();
        let x_a0 = verts[0].position[0];
        let x_a1 = verts[1].position[0];
        let x_b0 = verts[3].position[0];
        let x_b1 = verts[4].position[0];

        println!(
            "[Test EE CCD] Final X - A0: {:.4}, A1: {:.4}, B0: {:.4}, B1: {:.4}",
            x_a0, x_a1, x_b0, x_b1
        );

        // CCD により開放境界エッジ同士が裏返らず、X_A <= X_B を維持して押し留められていること
        assert!(
            x_a0 <= x_b0 + 1e-4 && x_a1 <= x_b1 + 1e-4,
            "開放境界エッジが相手側へ貫通・裏返ってはならない (A0={:.4}, B0={:.4})",
            x_a0, x_b0
        );
    }

    #[test]
    fn test_one_sided_boundary_ee_ccd_blocking() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // 片側が開放境界エッジ（または1Dストランド）、もう片側が2面に挟まれた内部エッジの衝突
        // エッジA (0-1): 開放境界エッジ (X = -0.03)
        // エッジB (3-4): 三角形 2-3-4 と 3-4-5 に共有される内部エッジ (X = +0.03)
        let positions = vec![
            [-0.03, 0.0, 0.05],  // 0 (境界エッジ)
            [-0.03, 0.1, 0.05],  // 1 (境界エッジ)
            [-0.10, 0.05, 0.05], // 2 (パネルAの背面頂点)
            [0.03, 0.0, 0.05],   // 3 (内部エッジ始点)
            [0.03, 0.1, 0.05],   // 4 (内部エッジ終点)
            [0.08, 0.05, 0.00],  // 5 (パネルBの下三角形頂点)
            [0.08, 0.05, 0.10],  // 6 (パネルBの上三角形頂点)
        ];
        let edges = vec![
            [0, 1], [1, 2], [2, 0], // パネルA (0-1 は開放境界)
            [3, 4], [4, 5], [5, 3], // パネルB 三角形1
            [3, 6], [6, 4],         // パネルB 三角形2 (3-4 を共有)
        ];
        let faces = vec![
            [0, 1, 2],
            [3, 4, 5],
            [3, 6, 4], // 3-4 は2面で共有される内部エッジ
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
            0.01, // thickness = 1cm
            10000.0,
            10000.0,
            5000.0,
            0.0,
            1.0,
            None,
            None,
        );

        // 3-4 エッジ (gpu_edges 内) が開放境界ではないことを確認
        let edge_34_idx = mesh.edges.iter().position(|e| (e.v0 == 3 && e.v1 == 4) || (e.v0 == 4 && e.v1 == 3)).unwrap();
        let is_b_open = ((mesh.edge_boundary_bits[edge_34_idx >> 5] >> (edge_34_idx & 31)) & 1) != 0;
        assert!(!is_b_open, "エッジ 3-4 は2面で共有される内部エッジでなければならない");

        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_enable_self_collision(true);
        sim.set_gravity(0.0, 0.0, 0.0);

        // 1 サブステップで相手を完全に突き抜ける大速度を設定 (+5.0 m/s と -5.0 m/s)
        let velocities = vec![
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [-5.0, 0.0, 0.0],
            [-5.0, 0.0, 0.0],
            [-5.0, 0.0, 0.0],
            [-5.0, 0.0, 0.0],
        ];
        sim.set_positions_and_velocities(&positions, Some(&velocities));

        // 1 ステップ実行 (substeps = 10)
        sim.step(1.0 / 60.0, 10);

        let verts = sim.read_vertices();
        let x_a0 = verts[0].position[0];
        let x_a1 = verts[1].position[0];
        let x_b0 = verts[3].position[0];
        let x_b1 = verts[4].position[0];

        println!(
            "[Test One-Sided EE CCD] Final X - A0: {:.4}, A1: {:.4}, B0: {:.4}, B1: {:.4}",
            x_a0, x_a1, x_b0, x_b1
        );

        // 片側開放境界フラグにより E-E CCD が発動し、内部エッジを突き抜けて裏返らないこと
        assert!(
            x_a0 <= x_b0 + 1e-4 && x_a1 <= x_b1 + 1e-4,
            "開放境界エッジが内部エッジを突き抜けて裏返ってはならない (A0={:.4}, B0={:.4})",
            x_a0, x_b0
        );
    }

    #[test]
    fn test_vt_ccd_moving_triangle_penetration_blocking() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // 上から下（-Z方向）へ高速移動してくる三角形 (0, 1, 2) と、
        // 下で静止している（あるいは上へ向かう）頂点 3
        let positions = vec![
            [-0.05, -0.05, 0.04], // 0
            [0.05, -0.05, 0.04],  // 1
            [0.0, 0.05, 0.04],   // 2
            [0.0, 0.0, 0.00],    // 3 (ターゲット頂点)
            [0.02, 0.0, 0.00],   // 4 (頂点3のダミーエッジ用)
        ];
        let edges = vec![
            [0, 1], [1, 2], [2, 0], // 三角形
            [3, 4],                 // 孤立回避エッジ
        ];
        let faces = vec![
            [0, 1, 2],
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
            0.01, // thickness = 1cm
            10000.0,
            10000.0,
            5000.0,
            0.0,
            1.0,
            None,
            None,
        );

        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_enable_self_collision(true);
        sim.set_gravity(0.0, 0.0, 0.0);

        // 三角形に下向き大速度 -4.0 m/s、頂点3に上向き速度 +2.0 m/s を設定
        let velocities = vec![
            [0.0, 0.0, -4.0],
            [0.0, 0.0, -4.0],
            [0.0, 0.0, -4.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ];
        sim.set_positions_and_velocities(&positions, Some(&velocities));

        // 1 ステップ実行 (substeps = 10)
        sim.step(1.0 / 60.0, 10);

        let verts = sim.read_vertices();
        let z_tri = (verts[0].position[2] + verts[1].position[2] + verts[2].position[2]) / 3.0;
        let z_pt3 = verts[3].position[2];

        println!(
            "[Test VT CCD] Final Z - Tri: {:.4}, Pt3: {:.4}",
            z_tri, z_pt3
        );

        // V-T CCD により頂点3が三角形を突き抜けて上に突き抜けていないこと (Z_pt3 <= Z_tri)
        assert!(
            z_pt3 <= z_tri + 1e-4,
            "頂点3が移動三角形を上方に突き抜けてはならない (Pt3={:.4}, Tri={:.4})",
            z_pt3, z_tri
        );
    }

    #[test]
    fn test_seam_flap_closure_no_inversion() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // フード顔開口の最小モデル:
        // 下端頂点 0 (左) と 3 (右) をアンカー固定し、
        // 頂点 2 と 5 を上端縫合で引き寄せる2枚の対向三角形フラップ。
        // 辺 1-2 (左リム自由縁) と 辺 4-5 (右リム自由縁) が正面から合掌する。
        let positions = vec![
            [-0.05, 0.0, 0.0],  // 0 (固定アンカー)
            [-0.05, 0.1, 0.0],  // 1 (自由縁下)
            [-0.05, 0.2, 0.0],  // 2 (縫合端)
            [0.05, 0.0, 0.0],   // 3 (固定アンカー)
            [0.05, 0.1, 0.0],   // 4 (自由縁下)
            [0.05, 0.2, 0.0],   // 5 (縫合端)
        ];
        let edges = vec![
            [0, 1], [1, 2], [2, 0],
            [3, 4], [4, 5], [5, 3],
        ];
        let faces = vec![
            [0, 1, 2],
            [3, 4, 5],
        ];
        let sewing_pairs = vec![
            [2u32, 5u32], // 上端縫合
        ];
        let inv_masses = vec![
            0.0, 1.0, 1.0, // 0 固定
            0.0, 1.0, 1.0, // 3 固定
        ];

        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            Some(&faces),
            Some(&inv_masses),
            Some(&sewing_pairs),
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
        sim.set_enable_self_collision(true);
        sim.set_gravity(0.0, 0.0, 0.0);

        for _ in 0..30 {
            sim.step(1.0 / 60.0, 10);
        }

        let verts = sim.read_vertices();
        let x1 = verts[1].position[0];
        let x4 = verts[4].position[0];

        println!("[Test Flap Closure] Mid Flap X - L: {:.4}, R: {:.4}", x1, x4);

        // 自由縁リム中央が左右逆転（クロスオーバー）していないこと
        assert!(
            x1 <= x4 + 1e-4,
            "合掌したフラップの自由縁が左右逆転してはならない (L={:.4}, R={:.4})",
            x1, x4
        );
    }

    #[test]
    fn test_flat_seam_closure_no_stagnation() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // 2枚の平面クアッドを X 方向に 0.05m 離して配置し、シーム辺 1-2 と 4-5 を縫合
        let positions = vec![
            [-0.05, 0.0, 0.0], // 0
            [0.0, 0.0, 0.0],   // 1 (シーム)
            [0.0, 0.1, 0.0],   // 2 (シーム)
            [-0.05, 0.1, 0.0], // 3
            [0.05, 0.0, 0.0],  // 4 (シーム)
            [0.05, 0.1, 0.0],  // 5 (シーム)
            [0.10, 0.1, 0.0],  // 6
            [0.10, 0.0, 0.0],  // 7
        ];
        let edges = vec![
            [0, 1], [1, 2], [2, 3], [3, 0], [0, 2],
            [4, 5], [5, 6], [6, 7], [7, 4], [4, 6],
        ];
        let faces = vec![
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
        ];
        let sewing_pairs = vec![
            [1u32, 4u32],
            [2u32, 5u32],
        ];

        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            Some(&faces),
            None,
            Some(&sewing_pairs),
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
        sim.set_enable_self_collision(true);
        sim.set_gravity(0.0, 0.0, 0.0);

        // 縫合収縮を実行 (30フレーム)
        for _ in 0..30 {
            sim.step(1.0 / 60.0, 15);
        }

        let verts = sim.read_vertices();
        let p1 = verts[1].position;
        let p4 = verts[4].position;
        let p2 = verts[2].position;
        let p5 = verts[5].position;

        let d14 = ((p1[0] - p4[0]).powi(2) + (p1[1] - p4[1]).powi(2) + (p1[2] - p4[2]).powi(2)).sqrt();
        let d25 = ((p2[0] - p5[0]).powi(2) + (p2[1] - p5[1]).powi(2) + (p2[2] - p5[2]).powi(2)).sqrt();

        println!("[Test Flat Seam Closure] Final Seam Distances: d(1,4)={:.5}, d(2,5)={:.5}", d14, d25);

        // 展開レスト距離の誤設定による膠着がなく、1mm (0.001m) 未満まで完全に密着閉鎖すること
        assert!(
            d14 < 0.001,
            "縫合ペア 1-4 が完全密着閉鎖しなければならない (実測={:.5})",
            d14
        );
        assert!(
            d25 < 0.001,
            "縫合ペア 2-5 が完全密着閉鎖しなければならない (実測={:.5})",
            d25
        );
        assert_eq!(sim.dynamic_pins.len(), 0);
    }

    fn read_u32_buffer_sync(ctx: &GpuContext, buffer: &wgpu::Buffer, count: usize) -> Vec<u32> {
        let size = (count * std::mem::size_of::<u32>()) as u64;
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test u32 Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Copy u32 Buffer"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        ctx.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..size);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        ctx.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(receiver.receive()).unwrap().unwrap();

        let mapped = slice.get_mapped_range();
        let out: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        out
    }

    #[test]
    fn test_spatial_hash_virtual_particles_isolation_across_algos() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let mesh = make_quad_mesh();
        let real_count = mesh.vertices.len() as u32;
        let mut sim = GpuClothSimulator::new(Arc::clone(&ctx), mesh);
        assert!(
            sim.num_virtual_vertices > 0,
            "1m quad は仮想頂点を生成するべき (num_virtual_vertices={})",
            sim.num_virtual_vertices
        );
        let total_count = real_count + sim.num_virtual_vertices;
        let table_size = sim.spatial_hash.table_size as usize;

        sim.set_enable_self_collision(true);
        sim.gravity = [0.0, 0.0, 0.0];

        let check_hash_entries = |sim: &GpuClothSimulator, expected_count: u32, expect_only_real: bool, label: &str| {
            let cell_starts = read_u32_buffer_sync(&ctx, &sim.spatial_hash.cell_starts_buffer, table_size + 1);
            let actual_count = cell_starts[table_size];
            assert_eq!(
                actual_count, expected_count,
                "{}: 空間ハッシュ登録数が不正 (expected={}, actual={})",
                label, expected_count, actual_count
            );
            let sorted = read_u32_buffer_sync(&ctx, &sim.spatial_hash.sorted_indices_buffer, actual_count as usize);
            if expect_only_real {
                assert!(
                    sorted.iter().all(|&idx| idx < real_count),
                    "{}: sorted_indices に仮想頂点 (>= {}) が混入した: {:?}",
                    label,
                    real_count,
                    sorted
                );
            }
        };

        // 1. algo=1 (PairCache V-T/E-E): 実頂点のみ登録
        sim.set_self_collision_algorithm(1);
        sim.step(1.0 / 60.0, 2);
        check_hash_entries(&sim, real_count, true, "algo=1 (初期)");

        // 2. algo=2 (V-V 仮想頂点): 実頂点 + 仮想頂点の全粒子を登録
        sim.set_self_collision_algorithm(2);
        sim.step(1.0 / 60.0, 2);
        check_hash_entries(&sim, total_count, false, "algo=2");

        // 3. algo=0 (Direct V-T/E-E) へ戻した際も仮想頂点が完全に除外されること
        sim.set_self_collision_algorithm(0);
        sim.step(1.0 / 60.0, 2);
        check_hash_entries(&sim, real_count, true, "algo=0 (algo=2後)");

        // 4. 再び algo=1 へ戻した際も実頂点のみ登録されること
        sim.set_self_collision_algorithm(1);
        sim.step(1.0 / 60.0, 2);
        check_hash_entries(&sim, real_count, true, "algo=1 (algo=0後)");
    }

    /// 内部エッジのみが中央で直交通過し、V-T衝突・境界エッジ衝突が起きない2パッチ屋根型メッシュを生成する。
    /// - パッチ A (頂点 0..3): X軸沿いの尾根辺 [0, 1] が内部エッジ (z=0.0)、翼 [2, 3] は z=-0.5 へ退避
    /// - パッチ B (頂点 4..7): Y軸沿いの尾根辺 [4, 5] が内部エッジ (z=z_b)、翼 [6, 7] は z=z_b+0.5 へ退避
    fn make_orthogonal_internal_edge_roof_pair(z_b: f32, thickness: f32) -> ClothMesh {
        let positions = vec![
            // パッチ A (X軸沿い尾根)
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.3, -0.5],
            [0.0, -0.3, -0.5],
            // パッチ B (Y軸沿い尾根)
            [0.0, -1.0, z_b],
            [0.0, 1.0, z_b],
            [0.3, 0.0, z_b + 0.5],
            [-0.3, 0.0, z_b + 0.5],
        ];
        let edges = vec![
            [0, 1], // index 0: パッチ A の内部エッジ
            [1, 2],
            [2, 0],
            [0, 3],
            [3, 1],
            [4, 5], // index 5: パッチ B の内部エッジ
            [5, 6],
            [6, 4],
            [4, 7],
            [7, 5],
        ];
        let faces = vec![
            [0, 1, 2],
            [1, 0, 3],
            [4, 5, 6],
            [5, 4, 7],
        ];
        ClothMesh::from_raw(
            &positions,
            &edges,
            Some(&faces),
            None,
            None,
            None,
            None,
            0,
            thickness,
            10000.0,
            10000.0,
            5000.0,
            0.0,
            1.0,
            None,
            None,
        )
    }

    #[test]
    fn test_internal_ee_ccd_blocking() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let thickness = 0.005;
        let z_b = 0.008;
        let mesh = make_orthogonal_internal_edge_roof_pair(z_b, thickness);
        // 対象エッジ (0, 1) と (4, 5) がどちらも非境界（内部エッジ）であることを確認
        let idx_a = mesh
            .edges
            .iter()
            .position(|e| (e.v0.min(e.v1), e.v0.max(e.v1)) == (0, 1))
            .expect("edge (0, 1) が存在すること");
        let idx_b = mesh
            .edges
            .iter()
            .position(|e| (e.v0.min(e.v1), e.v0.max(e.v1)) == (4, 5))
            .expect("edge (4, 5) が存在すること");
        assert_eq!((mesh.edge_boundary_bits[idx_a >> 5] >> (idx_a & 31)) & 1, 0, "edge (0, 1) は内部エッジであるべき");
        assert_eq!((mesh.edge_boundary_bits[idx_b >> 5] >> (idx_b & 31)) & 1, 0, "edge (4, 5) は内部エッジであるべき");

        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_enable_self_collision(true);
        sim.set_enable_pair_cache(false);
        sim.set_gravity(0.0, 0.0, 0.0);

        let positions = vec![
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.3, -0.5],
            [0.0, -0.3, -0.5],
            [0.0, -1.0, z_b],
            [0.0, 1.0, z_b],
            [0.3, 0.0, z_b + 0.5],
            [-0.3, 0.0, z_b + 0.5],
        ];
        // パッチ B を 1 サブステップ (dt=1/60, substeps=1) で約 -0.020m 移動させ、
        // 有効厚み 0.010m を完全に飛び越えた先 (z≈-0.012m) へトンネルさせる (G1 ゲート対象、かつ空間セル 27近傍内)
        let velocities = vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.2],
            [0.0, 0.0, -1.2],
            [0.0, 0.0, -1.2],
            [0.0, 0.0, -1.2],
        ];
        sim.set_positions_and_velocities(&positions, Some(&velocities));
        sim.step(1.0 / 60.0, 1);

        let verts = sim.read_vertices();
        let z_mid_a = 0.5 * (verts[0].position[2] + verts[1].position[2]);
        let z_mid_b = 0.5 * (verts[4].position[2] + verts[5].position[2]);
        assert!(
            z_mid_b >= z_mid_a - 1e-4,
            "G1: 高速移動する内部エッジ B が A を貫通してはならない: z_mid_a={:.6}, z_mid_b={:.6}",
            z_mid_a,
            z_mid_b
        );
    }

    #[test]
    fn test_internal_ee_ccd_low_speed_close_crossover_blocking() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let thickness = 0.005; // eff_thick = 0.010, G1閾値 = 0.005
        let z_b = 0.002;
        let mesh = make_orthogonal_internal_edge_roof_pair(z_b, thickness);
        let idx_a = mesh
            .edges
            .iter()
            .position(|e| (e.v0.min(e.v1), e.v0.max(e.v1)) == (0, 1))
            .unwrap();
        let idx_b = mesh
            .edges
            .iter()
            .position(|e| (e.v0.min(e.v1), e.v0.max(e.v1)) == (4, 5))
            .unwrap();
        assert_eq!((mesh.edge_boundary_bits[idx_a >> 5] >> (idx_a & 31)) & 1, 0);
        assert_eq!((mesh.edge_boundary_bits[idx_b >> 5] >> (idx_b & 31)) & 1, 0);

        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_enable_self_collision(true);
        sim.set_enable_pair_cache(false);
        sim.set_gravity(0.0, 0.0, 0.0);

        let positions = vec![
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.3, -0.5],
            [0.0, -0.3, -0.5],
            [0.0, -1.0, z_b],
            [0.0, 1.0, z_b],
            [0.3, 0.0, z_b + 0.5],
            [-0.3, 0.0, z_b + 0.5],
        ];
        // 1 サブステップ移動量 -0.004m (< G1閾値 0.005m) だが z=+0.002 -> z=-0.002 へ符号反転交差する (G2 ゲート対象)
        let velocities = vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -0.24],
            [0.0, 0.0, -0.24],
            [0.0, 0.0, -0.24],
            [0.0, 0.0, -0.24],
        ];
        sim.set_positions_and_velocities(&positions, Some(&velocities));
        sim.step(1.0 / 60.0, 1);

        let verts = sim.read_vertices();
        let z_mid_a = 0.5 * (verts[0].position[2] + verts[1].position[2]);
        let z_mid_b = 0.5 * (verts[4].position[2] + verts[5].position[2]);
        assert!(
            z_mid_b > z_mid_a,
            "G2: 低速近接クロスオーバー時に内部エッジ B が A の裏側へ確定押し出しされてはならない: z_mid_a={:.6}, z_mid_b={:.6}",
            z_mid_a,
            z_mid_b
        );
    }

    #[test]
    fn test_internal_ee_ccd_folded_hinge_no_self_excitation() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        // 中央列 (col=1) で 2 つ折りした 3列x3行 (9頂点) メッシュ：ヒンジ周辺エッジ間の発振がないことを確認
        let mut positions = Vec::with_capacity(9);
        let init_pos: Vec<[f32; 3]>;
        for iy in 0..3 {
            let y = iy as f32 * 0.05;
            positions.push([0.0, y, 0.005]);  // col 0 (上折り返し側)
            positions.push([0.05, y, 0.0]);   // col 1 (ヒンジ)
            positions.push([0.0, y, -0.005]); // col 2 (下側)
        }
        init_pos = positions.clone();

        let mut edges = Vec::new();
        let mut faces = Vec::new();
        for iy in 0..3 {
            let base = iy * 3;
            edges.push([base, base + 1]);
            edges.push([base + 1, base + 2]);
            if iy + 1 < 3 {
                for ix in 0..3 {
                    edges.push([base + ix, base + 3 + ix]);
                }
                for ix in 0..2 {
                    let a = base + ix;
                    let b = base + ix + 1;
                    let c = base + 3 + ix + 1;
                    let d = base + 3 + ix;
                    faces.push([a, b, c]);
                    faces.push([a, c, d]);
                }
            }
        }
        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            Some(&faces),
            None,
            None,
            None,
            None,
            0,
            0.004, // eff_thick = 0.008 (ヒンジ側で接触あり・非貫通配置)
            5000.0,
            5000.0,
            2500.0,
            0.0,
            1.0,
            None,
            None,
        );

        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_enable_self_collision(true);
        sim.set_enable_pair_cache(false);
        sim.set_gravity(0.0, 0.0, 0.0);

        for _ in 0..20 {
            sim.step(1.0 / 60.0, 5);
        }

        let verts = sim.read_vertices();
        let mut max_disp = 0.0f32;
        for (i, v) in verts.iter().enumerate() {
            assert!(
                v.position.iter().all(|c| c.is_finite()),
                "折り畳みヒンジで NaN/Inf が発生: v[{}]={:?}",
                i,
                v.position
            );
            let dx = v.position[0] - init_pos[i][0];
            let dy = v.position[1] - init_pos[i][1];
            let dz = v.position[2] - init_pos[i][2];
            max_disp = max_disp.max((dx * dx + dy * dy + dz * dz).sqrt());
        }
        assert!(
            max_disp < 0.05,
            "折り畳みヒンジで過大な自己発振変位が発生: max_disp={:.6}",
            max_disp
        );
    }

    #[test]
    fn test_internal_ee_ccd_concurrent_double_hit_no_dilution() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let thickness = 0.005;
        let z_b = 0.008;
        let positions = vec![
            // パッチ A1 (y=-0.012 付近のX軸内部エッジ [0, 1])
            [-0.02, -0.012, 0.0],
            [0.02, -0.012, 0.0],
            [0.0, -0.006, -0.02],
            [0.0, -0.018, -0.02],
            // パッチ A2 (y=+0.012 付近のX軸内部エッジ [4, 5])
            [-0.02, 0.012, 0.0],
            [0.02, 0.012, 0.0],
            [0.0, 0.018, -0.02],
            [0.0, 0.006, -0.02],
            // パッチ B (Y軸内部エッジ [8, 9] が s=0.2 と s=0.8 で A1/A2 と同時交差)
            [0.0, -0.02, z_b],
            [0.0, 0.02, z_b],
            [0.006, 0.0, z_b + 0.02],
            [-0.006, 0.0, z_b + 0.02],
        ];
        let edges = vec![
            [0, 1], [1, 2], [2, 0], [0, 3], [3, 1],
            [4, 5], [5, 6], [6, 4], [4, 7], [7, 5],
            [8, 9], [9, 10], [10, 8], [8, 11], [11, 9],
        ];
        let faces = vec![
            [0, 1, 2], [1, 0, 3],
            [4, 5, 6], [5, 4, 7],
            [8, 9, 10], [9, 8, 11],
        ];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, Some(&faces), None, None, None, None,
            0, thickness, 10000.0, 10000.0, 5000.0, 0.0, 1.0, None, None,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_enable_self_collision(true);
        sim.set_enable_pair_cache(false);
        sim.set_gravity(0.0, 0.0, 0.0);

        let mut velocities = vec![[0.0, 0.0, 0.0]; 8];
        velocities.extend(vec![[0.0, 0.0, -1.5]; 4]);
        sim.set_positions_and_velocities(&positions, Some(&velocities));
        sim.step(1.0 / 60.0, 1);

        let verts = sim.read_vertices();
        let z_a1 = 0.5 * (verts[0].position[2] + verts[1].position[2]);
        let z_a2 = 0.5 * (verts[4].position[2] + verts[5].position[2]);
        let z_b_mid = 0.5 * (verts[8].position[2] + verts[9].position[2]);
        assert!(
            z_b_mid >= z_a1 - 1e-4 && z_b_mid >= z_a2 - 1e-4,
            "多重同時 E-E CCD で平均化希釈による貫通が起きてはならない: z_a1={:.6}, z_a2={:.6}, z_b={:.6}",
            z_a1, z_a2, z_b_mid
        );
    }

    #[test]
    fn test_internal_ee_discrete_untangling_symmetry_across_rotation() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let thickness = 0.005;
        let run_case = |rotate_x180: bool| -> f32 {
            let z_b = -0.003;
            let raw_positions = vec![
                // 内層 (layer_id=0): X軸尾根 [0, 1], 法線は +Z
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.3, -0.5],
                [0.0, -0.3, -0.5],
                // 外層 (layer_id=1): Y軸尾根 [4, 5], 内層の裏側 z=-0.003
                [0.0, -1.0, z_b],
                [0.0, 1.0, z_b],
                [0.3, 0.0, z_b + 0.5],
                [-0.3, 0.0, z_b + 0.5],
            ];
            let positions: Vec<[f32; 3]> = raw_positions
                .into_iter()
                .map(|p| if rotate_x180 { [p[0], -p[1], -p[2]] } else { p })
                .collect();
            let edges = vec![
                [0, 1], [1, 2], [2, 0], [0, 3], [3, 1],
                [4, 5], [5, 6], [6, 4], [4, 7], [7, 5],
            ];
            let faces = vec![
                [0, 1, 2], [1, 0, 3],
                [4, 5, 6], [5, 4, 7],
            ];
            let layer_ids = [0, 0, 0, 0, 1, 1, 1, 1];
            let mesh = ClothMesh::from_raw(
                &positions, &edges, Some(&faces), None, None, Some(&layer_ids), None,
                0, thickness, 10000.0, 10000.0, 5000.0, 0.0, 1.0, None, None,
            );
            let mut sim = GpuClothSimulator::new(ctx.clone(), mesh);
            sim.set_enable_self_collision(true);
            sim.set_enable_pair_cache(false);
            sim.set_gravity(0.0, 0.0, 0.0);
            sim.step(1.0 / 60.0, 1);

            let verts = sim.read_vertices();
            let z_outer = 0.5 * (verts[4].position[2] + verts[5].position[2]);
            if rotate_x180 { -z_outer } else { z_outer }
        };

        let z_upright = run_case(false);
        let z_rotated = run_case(true);
        assert!(
            (z_upright - z_rotated).abs() < 1e-5,
            "180度剛体回転で離散 E-E untangling が非対称: z_upright={:.6}, z_rotated={:.6}",
            z_upright, z_rotated
        );
    }

    #[test]
    fn test_set_positions_and_velocities_preserves_inv_masses_and_pins() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let mesh = make_quad_mesh();
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_vertex_inv_masses(&[0], &[0.5]);
        sim.set_pin_target(1, [1.0, 0.0, 0.5], 0.75);

        let positions = vec![
            [0.0, 0.0, 0.6],
            [1.0, 0.0, 0.6],
            [1.0, 1.0, 0.6],
            [0.0, 1.0, 0.6],
        ];
        let velocities = vec![[0.0; 3]; 4];
        sim.set_positions_and_velocities(&positions, Some(&velocities));

        let verts = sim.read_vertices();
        assert!((verts[0].inv_mass - 0.5).abs() < 1e-6);
        assert!((verts[1].inv_mass - 0.25).abs() < 1e-6);
        assert!((verts[2].inv_mass - 1.0).abs() < 1e-6);
        assert!((verts[0].position[2] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_record_current_frame_captures_inv_mass_deltas() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let mesh = make_quad_mesh();
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.start_debug_recording("InvMassDeltaTest", Some(16));

        sim.step(1.0 / 60.0, 2);
        sim.set_vertex_inv_masses(&[2], &[3.0]);
        sim.step(1.0 / 60.0, 2);
        sim.step(1.0 / 60.0, 2);

        let frames = &sim.debug_recorder.frames;
        assert_eq!(frames.len(), 3);
        assert!(frames[0].inv_mass_deltas.is_none());
        let d1 = frames[1].inv_mass_deltas.as_ref().expect("frame 1 must record inv_mass_deltas");
        assert_eq!(d1.len(), 1);
        assert_eq!(d1[0].vertex_idx, 2);
        assert!((d1[0].inv_mass - 3.0).abs() < 1e-6);
        assert!(frames[2].inv_mass_deltas.is_none());
    }

    #[test]
    fn test_pair_cache_workgroup_size_32_and_dynamic_switch() {
        // workgroup_size=32 の PairCache でスロット番号 >= 32 の後半頂点群まで確実に自己衝突解決され、
        // さらに set_workgroup_size(64) / (32) の動的切替後も全頂点が凍結せず動作することを検証
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        let nx = 8usize;
        let ny = 7usize;
        let spacing = 0.15f32;
        let thickness = 0.04f32;
        let per_layer = nx * ny; // 56頂点/層、2層合計112頂点

        let mut positions = Vec::with_capacity(per_layer * 2);
        let mut edges = Vec::new();
        let mut faces = Vec::new();
        let mut layer_ids = Vec::with_capacity(per_layer * 2);
        let mut inv_masses = Vec::with_capacity(per_layer * 2);

        for layer in 0..2u32 {
            // 上層 (layer=1) を下層 (layer=0, 表面法線 +Z) の表側に厚み (0.04*2=0.08) 未満の 0.005 で近接配置
            let z = if layer == 0 { 0.0 } else { 0.005 };
            let offset = (layer as usize) * per_layer;
            for iy in 0..ny {
                for ix in 0..nx {
                    positions.push([ix as f32 * spacing, iy as f32 * spacing, z]);
                    layer_ids.push(layer);
                    inv_masses.push(if layer == 0 { 0.0 } else { 1.0 });
                }
            }
            for iy in 0..ny {
                for ix in 0..nx {
                    let v0 = (offset + iy * nx + ix) as u32;
                    if ix + 1 < nx {
                        edges.push([v0, v0 + 1]);
                    }
                    if iy + 1 < ny {
                        edges.push([v0, v0 + nx as u32]);
                    }
                    if ix + 1 < nx && iy + 1 < ny {
                        let v1 = v0 + 1;
                        let v2 = v0 + nx as u32 + 1;
                        let v3 = v0 + nx as u32;
                        edges.push([v0, v2]);
                        faces.push([v0, v1, v2]);
                        faces.push([v0, v2, v3]);
                    }
                }
            }
        }

        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            Some(&faces),
            None,
            None,
            Some(&layer_ids),
            Some(&inv_masses),
            0,
            thickness,
            10000.0,
            10000.0,
            5000.0,
            0.0,
            1.0,
            None,
            None,
        );

        let mut sim = GpuClothSimulator::with_options(ctx, mesh, 32, 0);
        sim.set_enable_self_collision(true);
        sim.set_enable_pair_cache(true);
        sim.set_gravity(0.0, 0.0, 0.0);
        sim.step(1.0 / 60.0, 4);

        let verts = sim.read_vertices();
        let pushed_count = (per_layer..per_layer * 2)
            .filter(|&i| verts[i].position[2] > 0.02)
            .count();
        assert!(
            pushed_count > per_layer / 2,
            "wg=32 の PairCache で上層 (index >= 56) が半数以上押し戻されるべき: pushed={}/{}",
            pushed_count,
            per_layer
        );

        // 動的切替: workgroup_size=64 でも後半50% (index >= 56) が凍結せず重力落下する
        sim.set_enable_self_collision(false);
        sim.set_workgroup_size(64);
        assert_eq!(sim.workgroup_size, 64);
        assert_eq!(sim.shared.workgroup_size, 64);
        let reset_vel = vec![[0.0f32; 3]; positions.len()];
        sim.set_positions_and_velocities(&positions, Some(&reset_vel));
        let before_64 = sim.read_vertices();
        sim.set_gravity(0.0, 0.0, -9.81);
        sim.step(1.0 / 60.0, 2);
        let after_64 = sim.read_vertices();
        for i in per_layer..per_layer * 2 {
            assert!(
                after_64[i].position[2] < before_64[i].position[2] - 1e-4,
                "wg=64 切替後に頂点 {} が重力落下していない: before={}, after={}",
                i,
                before_64[i].position[2],
                after_64[i].position[2]
            );
        }

        // 動的切替: workgroup_size=32 へ戻しても同様に動作する
        sim.set_workgroup_size(32);
        assert_eq!(sim.workgroup_size, 32);
        assert_eq!(sim.shared.workgroup_size, 32);
        let before_32 = after_64;
        sim.step(1.0 / 60.0, 2);
        let after_32 = sim.read_vertices();
        for i in per_layer..per_layer * 2 {
            assert!(
                after_32[i].position[2] < before_32[i].position[2] - 1e-4,
                "wg=32 復帰後に頂点 {} が重力落下していない",
                i
            );
        }
    }

    #[test]
    fn test_extract_positions_non_multiple_of_32_with_virtual_vertices() {
        // 実頂点数が32の倍数ではない (35頂点) かつ仮想頂点が生成される状態で
        // extract_positions が compact_position_buffer に境界外書込しないことを検証
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        let nx = 5usize;
        let ny = 7usize; // 35 vertices (非32倍数)
        let spacing = 0.20f32;
        let thickness = 0.06f32; // target_spacing=0.09 < 0.20 -> 仮想頂点多数生成
        let mut positions = Vec::with_capacity(nx * ny);
        let mut edges = Vec::new();
        let mut faces = Vec::new();
        for iy in 0..ny {
            for ix in 0..nx {
                positions.push([ix as f32 * spacing, iy as f32 * spacing, 1.0]);
            }
        }
        for iy in 0..ny {
            for ix in 0..nx {
                let v0 = (iy * nx + ix) as u32;
                if ix + 1 < nx {
                    edges.push([v0, v0 + 1]);
                }
                if iy + 1 < ny {
                    edges.push([v0, v0 + nx as u32]);
                }
                if ix + 1 < nx && iy + 1 < ny {
                    let v1 = v0 + 1;
                    let v2 = v0 + nx as u32 + 1;
                    let v3 = v0 + nx as u32;
                    edges.push([v0, v2]);
                    faces.push([v0, v1, v2]);
                    faces.push([v0, v2, v3]);
                }
            }
        }

        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            Some(&faces),
            None,
            None,
            None,
            None,
            0,
            thickness,
            10000.0,
            10000.0,
            5000.0,
            0.0,
            1.0,
            None,
            None,
        );

        let mut sim = GpuClothSimulator::with_options(ctx, mesh, 32, 0);
        sim.set_enable_self_collision(true);
        sim.set_self_collision_algorithm(2);
        sim.step(1.0 / 60.0, 2);

        let mut pos = vec![0.0f32; nx * ny * 3];
        sim.get_positions_flat(&mut pos);
        assert_eq!(pos.len(), nx * ny * 3);
        assert!(pos.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_dynamic_resize_over_256_colliders_and_65536_mesh_tris() {
        // コライダー数 > 256 および メッシュ三角形 > 65536 でバッファが動的拡張されパニックしないことを検証
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        let mesh = make_quad_mesh();
        let mut sim = GpuClothSimulator::new(ctx, mesh);

        // 初期上限256を超える300個の球コライダーを設定
        for i in 0..300u32 {
            sim.add_sphere_collider([10.0 + i as f32 * 0.5, 0.0, 0.0], 0.1, 0.3, 0.0);
        }
        assert_eq!(sim.colliders.len(), 300);
        sim.step(1.0 / 60.0, 1);

        // 初期上限65536を超える65540個のメッシュ三角形を設定
        let tris = vec![
            GpuMeshTriangle {
                p0: [5.0, 0.0, 0.0],
                friction: 0.3,
                p1: [5.1, 0.0, 0.0],
                thickness: 0.01,
                p2: [5.0, 0.1, 0.0],
                restitution: 0.0,
                flags: 1,
                _pad: [0.0; 3],
            };
            65540
        ];
        sim.set_mesh_triangles(&tris);
        assert_eq!(sim.mesh_triangles.len(), 65540);
        sim.step(1.0 / 60.0, 1);

        let mut pos = vec![0.0f32; 12];
        sim.get_positions_flat(&mut pos);
        assert_eq!(pos.len(), 12);
        assert!(pos.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_empty_mesh_simulator_creation_safe() {
        // 0頂点メッシュでもバッファ生成・step・get_positions_flat がパニックしないことを検証
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        let mesh = ClothMesh::from_raw(
            &[],
            &[],
            None,
            None,
            None,
            None,
            None,
            0,
            0.01,
            1000.0,
            1000.0,
            500.0,
            0.0,
            1.0,
            None,
            None,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.step(1.0 / 60.0, 1);
        let mut pos = Vec::<f32>::new();
        sim.get_positions_flat(&mut pos);
        assert!(pos.is_empty());
    }

    #[test]
    fn test_quad_shear_and_bending_disjoint_and_symmetric() {
        // 2x1 Quad メッシュ:
        // 頂点: 0=(0,0,0), 1=(1,0,0), 2=(2,0,0), 3=(0,1,0), 4=(1,1,0), 5=(2,1,0)
        // オリジナル辺7本（内部対角線 [0,4] と [1,5] は含まない）
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ];
        let edges = [
            [0, 1], [1, 2], [3, 4], [4, 5],
            [0, 3], [1, 4], [2, 5],
        ];
        let faces = [
            [0, 1, 4], [0, 4, 3], // Quad 0 (分割対角線 0-4)
            [1, 2, 5], [1, 5, 4], // Quad 1 (分割対角線 1-5)
        ];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, Some(&faces),
            None, None, None, None,
            0, 0.005, 1000.0, 1000.0, 200.0, 10.0, 1.0,
            None, None,
        );

        let stretch_pairs: std::collections::HashSet<(u32, u32)> = mesh
            .distance_constraints
            .iter()
            .filter(|c| c.constraint_type == 0)
            .map(|c| (c.v0.min(c.v1), c.v0.max(c.v1)))
            .collect();
        let shear_pairs: std::collections::HashSet<(u32, u32)> = mesh
            .distance_constraints
            .iter()
            .filter(|c| c.constraint_type == 1)
            .map(|c| (c.v0.min(c.v1), c.v0.max(c.v1)))
            .collect();
        let bend_pairs: std::collections::HashSet<(u32, u32)> = mesh
            .bending_constraints
            .iter()
            .map(|c| (c.v2.min(c.v3), c.v2.max(c.v3)))
            .collect();

        // Stretch 7本、各QuadにX字対角線2本ずつでShear計4本、Quad間共有辺(1-4)の対向(0-5)にBending 1本
        assert_eq!(stretch_pairs.len(), 7);
        assert_eq!(
            shear_pairs,
            [(0, 4), (1, 3), (1, 5), (2, 4)].into_iter().collect()
        );
        assert_eq!(bend_pairs, [(0, 5)].into_iter().collect());
        // Shear と Bending の頂点ペアは完全に非重複であること
        assert!(shear_pairs.is_disjoint(&bend_pairs));

        // 純粋 Tri メッシュ（全三角形辺がオリジナル辺に含まれる2三角形）では Shear=0本、Bending=1本
        let tri_positions = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.5, 0.0],
            [1.0, 0.5, 0.0],
        ];
        let tri_edges = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3]];
        let tri_faces = [[0, 1, 2], [1, 0, 3]];
        let tri_mesh = ClothMesh::from_raw(
            &tri_positions, &tri_edges, Some(&tri_faces),
            None, None, None, None,
            0, 0.005, 1000.0, 1000.0, 200.0, 10.0, 1.0,
            None, None,
        );
        let tri_shear_count = tri_mesh
            .distance_constraints
            .iter()
            .filter(|c| c.constraint_type == 1)
            .count();
        assert_eq!(tri_shear_count, 0);
        assert_eq!(tri_mesh.bending_constraints.len(), 1);
    }

    #[test]
    fn test_strand_1d_bending_compliance_preserved_after_set_stiffness_all() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // 2D三角形 (0,1,2) + 共有辺で繋ぐ三角形 (1,0,3) + 1Dストランド (3-4-5)
        let positions = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            [2.0, 0.5, 0.0],
            [3.0, 0.5, 0.0],
        ];
        let edges = [
            [0, 1], [0, 2], [1, 2], [0, 3], [1, 3],
            [3, 4], [4, 5],
        ];
        let faces = [[0, 1, 2], [1, 0, 3]];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, Some(&faces),
            None, None, None, None,
            0, 0.005, 1000.0, 1000.0, 200.0, 10.0, 1.0,
            None, None,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);

        // bending_stiffness = 100.0 -> bend_comp = 0.01, strand = max(0.01 * 10, 0.5) = 0.5
        sim.set_stiffness_all(1000.0, 500.0, 200.0, 100.0);

        let face_bends: Vec<_> = sim
            .bending_constraints
            .iter()
            .filter(|c| c._pad[0] < 0.5)
            .collect();
        let strand_bends: Vec<_> = sim
            .bending_constraints
            .iter()
            .filter(|c| c._pad[0] > 1.5)
            .collect();

        assert_eq!(face_bends.len(), 1);
        assert_eq!(strand_bends.len(), 1);
        assert!((face_bends[0].compliance - 0.01).abs() < 1e-6);
        assert!((strand_bends[0].compliance - 0.5).abs() < 1e-6);

        // 非常に柔らかな曲げ (bending_stiffness = 2.0 -> bend_comp = 0.5) では strand = 5.0
        sim.set_stiffness_all(1000.0, 500.0, 200.0, 2.0);
        let strand_bends_soft: Vec<_> = sim
            .bending_constraints
            .iter()
            .filter(|c| c._pad[0] > 1.5)
            .collect();
        assert!((strand_bends_soft[0].compliance - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_air_damping_single_application_and_internal_damping_rigid_invariance() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // 1. 空気減衰の1回適用検証: g=-9.81, d=2.0, dt=1/60, substeps=10 を 60フレーム (計1.0秒)
        let make_edge_sim = |air_damping: f32, internal: [f32; 4]| {
            let pos = [[0.0, 0.0, 10.0], [1.0, 0.0, 10.0]];
            let edges = [[0, 1]];
            let mesh = ClothMesh::from_raw(
                &pos, &edges, None, Some(&[1.0, 1.0]),
                None, None, None,
                0, 0.005, 1000.0, 1000.0, 500.0, 10.0, 1.0,
                None, None,
            );
            let mut s = GpuClothSimulator::new(Arc::clone(&ctx), mesh);
            s.set_gravity(0.0, 0.0, -9.81);
            s.set_damping(air_damping);
            s.set_damping_all(internal[0], internal[1], internal[2], internal[3]);
            s
        };

        let dt_sub = (1.0f32 / 60.0) / 10.0;
        let mut expected_vz = 0.0f32;
        for _ in 0..600 {
            expected_vz = expected_vz * (1.0 - 2.0 * dt_sub).max(0.0) + (-9.81) * dt_sub;
        }

        let mut sim_air = make_edge_sim(2.0, [0.0; 4]);
        let mut sim_internal_only = make_edge_sim(0.0, [20.0, 20.0, 20.0, 20.0]);
        let mut sim_zero = make_edge_sim(0.0, [0.0; 4]);
        let mut sim_air_with_internal = make_edge_sim(2.0, [20.0, 20.0, 20.0, 20.0]);

        for _ in 0..60 {
            sim_air.step(1.0 / 60.0, 10);
            sim_internal_only.step(1.0 / 60.0, 10);
            sim_zero.step(1.0 / 60.0, 10);
            sim_air_with_internal.step(1.0 / 60.0, 10);
        }

        let v_air = sim_air.read_vertices()[0].velocity[2];
        let v_int = sim_internal_only.read_vertices()[0].velocity[2];
        let v_zero = sim_zero.read_vertices()[0].velocity[2];
        let v_both = sim_air_with_internal.read_vertices()[0].velocity[2];

        // 空気減衰1回適用の理論値 (~ -4.24 m/s) と一致 (旧2回適用は ~ -2.41 m/s)
        assert!(
            (v_air - expected_vz).abs() < 1e-3,
            "1回適用の理論値と不一致: actual={}, expected={}",
            v_air,
            expected_vz
        );
        // 内部減衰のみを大きく設定しても自由落下（剛体並進）には影響しないこと
        assert!(
            (v_int - v_zero).abs() < 1e-4,
            "内部減衰が剛体並進自由落下を減衰させている: v_int={}, v_zero={}",
            v_int,
            v_zero
        );
        assert!(
            (v_both - v_air).abs() < 1e-4,
            "内部減衰が空気減衰へ合算されている: v_both={}, v_air={}",
            v_both,
            v_air
        );
    }

    #[test]
    fn test_xpbd_rayleigh_internal_damping_and_atomic_clamp() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        let run_stretch_osc = |solver_mode: u32, substeps: u32, tension_damp: f32, compression_damp: f32| -> f32 {
            let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
            let edges = [[0, 1]];
            let mesh = ClothMesh::from_raw(
                &positions, &edges, None, Some(&[0.0, 50.0]),
                None, None, None,
                0, 0.005, 200.0, 200.0, 200.0, 10.0, 1.0,
                None, None,
            );
            let mut sim = GpuClothSimulator::with_options(Arc::clone(&ctx), mesh, 32, solver_mode);
            sim.set_gravity(0.0, 0.0, 0.0);
            sim.set_damping(0.0);
            sim.set_damping_all(tension_damp, compression_damp, 0.0, 0.0);
            sim.set_positions_and_velocities(&[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], None);

            let mut max_tail_speed = 0.0f32;
            for frame in 0..90 {
                sim.step(1.0 / 60.0, substeps);
                if frame >= 60 {
                    let v = sim.read_vertices()[1].velocity[0].abs();
                    max_tail_speed = max_tail_speed.max(v);
                }
            }
            max_tail_speed
        };

        // 1. 伸縮エッジの Rayleigh 引張・圧縮減衰 (Colored & Atomic 双方, m=0.02kg -> inv_mass=50, damp=25.0)
        for solver_mode in [0u32, 1u32] {
            let speed_undamped = run_stretch_osc(solver_mode, 20, 0.0, 0.0);
            let speed_damped = run_stretch_osc(solver_mode, 20, 25.0, 25.0);
            assert!(
                speed_damped < speed_undamped * 0.4,
                "solver_mode={} で伸縮Rayleigh減衰が効いていない: damped={}, undamped={}",
                solver_mode,
                speed_damped,
                speed_undamped
            );
        }

        // 2. 時間刻み幅 (substeps) 非依存性の確認: 同一 substeps 下の undamped 基準比で substeps=10 と 40 ともに明確減衰すること
        // (陰的オイラー法自体の数値減衰量が substeps に依存するため、各条件ごとの無減衰基準と比較する)
        let speed_sub10_undamped = run_stretch_osc(0, 10, 0.0, 0.0);
        let speed_sub10_damped = run_stretch_osc(0, 10, 25.0, 25.0);
        let speed_sub40_undamped = run_stretch_osc(0, 40, 0.0, 0.0);
        let speed_sub40_damped = run_stretch_osc(0, 40, 25.0, 25.0);
        assert!(
            speed_sub10_damped < speed_sub10_undamped * 0.4
                && speed_sub40_damped < speed_sub40_undamped * 0.4,
            "時間刻み幅依存でRayleigh減衰が消失: sub10=({}/{}) sub40=({}/{})",
            speed_sub10_damped,
            speed_sub10_undamped,
            speed_sub40_damped,
            speed_sub40_undamped
        );

        // 3. 曲げ振動の Rayleigh 曲げ減衰 (90度L字ヒンジでの一次開閉振動; 平坦位置では距離曲げ勾配が二次微小となるため)
        let run_bend_osc = |bending_damp: f32| -> f32 {
            let positions = [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.5, 0.0],
                [0.0, 0.5, 1.0],
            ];
            let edges = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3]];
            let faces = [[0, 1, 2], [1, 0, 3]];
            let mesh = ClothMesh::from_raw(
                &positions, &edges, Some(&faces), Some(&[0.0, 0.0, 0.0, 500.0]),
                None, None, None,
                0, 0.005, 5000.0, 5000.0, 500.0, 150.0, 1.0,
                None, None,
            );
            let mut sim = GpuClothSimulator::new(Arc::clone(&ctx), mesh);
            sim.set_gravity(0.0, 0.0, 0.0);
            sim.set_damping(0.0);
            sim.set_damping_all(0.0, 0.0, 0.0, bending_damp);
            let phi: f32 = 0.35;
            sim.set_positions_and_velocities(
                &[
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.5, 0.0],
                    [phi.sin(), 0.5, phi.cos()],
                ],
                None,
            );

            let mut max_tail_vx = 0.0f32;
            for frame in 0..90 {
                sim.step(1.0 / 60.0, 20);
                if frame >= 60 {
                    let vx = sim.read_vertices()[3].velocity[0].abs();
                    max_tail_vx = max_tail_vx.max(vx);
                }
            }
            max_tail_vx
        };

        let bend_undamped = run_bend_osc(0.0);
        let bend_damped = run_bend_osc(15.0);
        assert!(
            bend_damped < bend_undamped * 0.5,
            "曲げRayleigh減衰が効いていない: damped={}, undamped={}",
            bend_damped,
            bend_undamped
        );

        // 4. Atomic モードで極端な初期伸長 (10000m) を与えても固定小数点クランプにより NaN/Inf が発生しないこと
        let extreme_pos = [[0.0, 0.0, 0.0], [10000.0, 0.0, 0.0]];
        let edges = [[0, 1]];
        let extreme_mesh = ClothMesh::from_raw(
            &extreme_pos, &edges, None, Some(&[0.0, 1.0]),
            None, None, None,
            0, 0.005, 5000.0, 5000.0, 500.0, 10.0, 1.0,
            None, None,
        );
        let mut extreme_sim = GpuClothSimulator::with_options(ctx, extreme_mesh, 32, 1);
        extreme_sim.set_gravity(0.0, 0.0, 0.0);
        for _ in 0..5 {
            extreme_sim.step(1.0 / 60.0, 5);
        }
        for v in extreme_sim.read_vertices() {
            assert!(v.position.iter().all(|x| x.is_finite()));
            assert!(v.velocity.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn test_static_pin_edge_low_index_blocks_dynamic_edge() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let thickness = 0.005;
        let z_b = 0.008;
        let positions = vec![
            // パッチ A (edge 0 = [0, 1], 全頂点静止ピン留め)
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.3, -0.5],
            [0.0, -0.3, -0.5],
            // パッチ B (edge 5 = [4, 5], 動的)
            [0.0, -1.0, z_b],
            [0.0, 1.0, z_b],
            [0.3, 0.0, z_b + 0.5],
            [-0.3, 0.0, z_b + 0.5],
        ];
        let edges = vec![
            [0, 1], [1, 2], [2, 0], [0, 3], [3, 1],
            [4, 5], [5, 6], [6, 4], [4, 7], [7, 5],
        ];
        let faces = vec![
            [0, 1, 2], [1, 0, 3],
            [4, 5, 6], [5, 4, 7],
        ];
        let inv_masses = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, Some(&faces), Some(&inv_masses),
            None, None, None,
            0, thickness, 10000.0, 10000.0, 5000.0, 0.0, 1.0,
            None, None,
        );

        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_enable_self_collision(true);
        sim.set_self_collision_algorithm(0);
        sim.set_enable_pair_cache(false);
        sim.set_gravity(0.0, 0.0, 0.0);

        let velocities = vec![
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.2], [0.0, 0.0, -1.2], [0.0, 0.0, -1.2], [0.0, 0.0, -1.2],
        ];
        sim.set_positions_and_velocities(&positions, Some(&velocities));
        sim.step(1.0 / 60.0, 1);

        let verts = sim.read_vertices();
        let z_mid_a = 0.5 * (verts[0].position[2] + verts[1].position[2]);
        let z_mid_b = 0.5 * (verts[4].position[2] + verts[5].position[2]);
        assert!(z_mid_a.abs() < 1e-6, "静止ピンエッジ A は移動しないこと: {}", z_mid_a);
        assert!(
            z_mid_b >= z_mid_a - 1e-4,
            "低インデックス静止ピンエッジ A が動的エッジ B の通過をブロックするべき: z_mid_a={}, z_mid_b={}",
            z_mid_a,
            z_mid_b
        );
    }

    #[test]
    fn test_pair_cache_ee_ccd_blocks_high_speed_crossing() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let thickness = 0.005;
        let z_b = 0.006;
        // PairCache空間セル27近傍に確実に収まる 0.1m スケールの直交内部エッジペア
        let positions = vec![
            [-0.05, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.03, -0.05],
            [0.0, -0.03, -0.05],
            [0.0, -0.05, z_b],
            [0.0, 0.05, z_b],
            [0.03, 0.0, z_b + 0.05],
            [-0.03, 0.0, z_b + 0.05],
        ];
        let edges = vec![
            [0, 1], [1, 2], [2, 0], [0, 3], [3, 1],
            [4, 5], [5, 6], [6, 4], [4, 7], [7, 5],
        ];
        let faces = vec![
            [0, 1, 2], [1, 0, 3],
            [4, 5, 6], [5, 4, 7],
        ];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, Some(&faces), None,
            None, None, None,
            0, thickness, 10000.0, 10000.0, 5000.0, 0.0, 1.0,
            None, None,
        );

        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_enable_self_collision(true);
        sim.set_self_collision_algorithm(1);
        sim.set_enable_pair_cache(true);
        sim.set_gravity(0.0, 0.0, 0.0);

        // 1サブステップで -0.015m 移動 (z_pred = -0.009m): 距離は有効厚み 0.010m 内だが交差反転済み
        let velocities = vec![
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            [0.0, 0.0, -0.9], [0.0, 0.0, -0.9], [0.0, 0.0, -0.9], [0.0, 0.0, -0.9],
        ];
        sim.set_positions_and_velocities(&positions, Some(&velocities));
        sim.step(1.0 / 60.0, 1);

        let verts = sim.read_vertices();
        let z_mid_a = 0.5 * (verts[0].position[2] + verts[1].position[2]);
        let z_mid_b = 0.5 * (verts[4].position[2] + verts[5].position[2]);
        assert!(
            z_mid_b >= z_mid_a - 1e-4,
            "PairCache モード E-E CCD が交差貫通を阻止するべき: z_mid_a={:.6}, z_mid_b={:.6}",
            z_mid_a,
            z_mid_b
        );
    }

    #[test]
    fn test_collision_departure_velocity_and_segment_culling() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // 1. 接触押し出しにより厚み外へ退避し、非物理的な法線加速が起きないこと
        let mesh_dep = ClothMesh::from_raw(
            &[[0.0, 0.0, 0.005]], &[], None, None,
            None, None, None,
            0, 0.02, 10000.0, 10000.0, 5000.0, 0.0, 1.0,
            None, None,
        );
        let mut sim_dep = GpuClothSimulator::new(Arc::clone(&ctx), mesh_dep);
        sim_dep.set_gravity(0.0, 0.0, 0.0);
        sim_dep.set_damping(0.0);
        sim_dep.set_mesh_triangles(&[GpuMeshTriangle {
            p0: [-1.0, -1.0, 0.0],
            friction: 0.0,
            p1: [1.0, -1.0, 0.0],
            thickness: 0.02,
            p2: [0.0, 1.0, 0.0],
            restitution: 0.0,
            flags: 0,
            _pad: [0.0; 3],
        }]);
        sim_dep.set_positions_and_velocities(&[[0.0, 0.0, 0.005]], Some(&[[0.0, 0.0, 1.0]]));
        sim_dep.step(1.0 / 60.0, 1);

        let v_dep = sim_dep.read_vertices()[0];
        assert!(
            v_dep.position[2] >= 0.04 - 1e-3,
            "接触押し出しにより厚み外へ退避すること: z={}",
            v_dep.position[2]
        );
        assert!(
            v_dep.velocity[2].is_finite() && v_dep.velocity[2] >= -1e-4 && v_dep.velocity[2] <= 1.0 + 1e-4,
            "接触時の法線方向へ非物理的な加速が起きないこと: vz={}",
            v_dep.velocity[2]
        );

        // 2. 始点・終点ともに境界球外を高速横断する移動でも線分-境界球カリングで捕捉・阻止されること (直接パス & クラスタパス)
        let small_tri = GpuMeshTriangle {
            p0: [-0.1, -0.1, 0.0],
            friction: 0.0,
            p1: [0.1, -0.1, 0.0],
            thickness: 0.01,
            p2: [0.0, 0.1, 0.0],
            restitution: 0.0,
            flags: 0,
            _pad: [0.0; 3],
        };
        let mut cluster_tris = vec![small_tri];
        for _ in 0..32 {
            cluster_tris.push(GpuMeshTriangle {
                p0: [10.0, 10.0, 10.0],
                friction: 0.0,
                p1: [10.1, 10.0, 10.0],
                thickness: 0.01,
                p2: [10.0, 10.1, 10.0],
                restitution: 0.0,
                flags: 0,
                _pad: [0.0; 3],
            });
        }

        for enable_cluster in [false, true] {
            let mesh_seg = ClothMesh::from_raw(
                &[[0.0, 0.0, 0.50]], &[], None, None,
                None, None, None,
                0, 0.01, 10000.0, 10000.0, 5000.0, 0.0, 1.0,
                None, None,
            );
            let mut sim_seg = GpuClothSimulator::new(Arc::clone(&ctx), mesh_seg);
            sim_seg.set_gravity(0.0, 0.0, 0.0);
            sim_seg.set_damping(0.0);
            sim_seg.set_collider_options(enable_cluster, false, 0.0);
            if enable_cluster {
                sim_seg.set_mesh_triangles(&cluster_tris);
            } else {
                sim_seg.set_mesh_triangles(&[small_tri]);
            }
            // dt=1/60 で z=+0.50 -> z=-0.50 へ 1.0m ジャンプ（両端点が半径 r~0.25m の境界球外）
            sim_seg.set_positions_and_velocities(&[[0.0, 0.0, 0.50]], Some(&[[0.0, 0.0, -60.0]]));
            sim_seg.step(1.0 / 60.0, 1);

            let v_seg = sim_seg.read_vertices()[0];
            assert!(
                v_seg.position[2] >= 0.0,
                "enable_cluster={}: 始点・終点とも境界球外の線分貫通がカリングされずブロックされること: z={}",
                enable_cluster,
                v_seg.position[2]
            );
        }
    }

    #[test]
    fn test_edge_collision_single_sided_chord_pushes_outward() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // 布エッジの両端は屋根斜面の表側非接触位置 (signed_dist = 0.0389m > target_dist 0.030m) に置き、
        // 中点付近のみが片面凸稜線 (Z=0) の裏側へ弦状に潜り込んでいる状態 (z = -0.025m)
        let positions = vec![
            [0.0, -0.08, -0.025],
            [0.0, 0.08, -0.025],
        ];
        let edges = vec![[0, 1]];
        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            None,
            None,
            None,
            None,
            None,
            0,
            0.01,
            10000.0,
            10000.0,
            5000.0,
            0.0,
            1.0,
            None,
            None,
        );

        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_enable_edge_collision(true);
        sim.set_gravity(0.0, 0.0, 0.0);
        sim.set_damping(0.0);

        // X軸凸稜線 (-0.5,0,0)-(0.5,0,0) を共有する45度屋根形片面コライダー (flags=1, エッジ平均法線 = +Z)
        let roof_tris = [
            GpuMeshTriangle {
                p0: [-0.5, 0.0, 0.0],
                friction: 0.0,
                p1: [0.5, 0.0, 0.0],
                thickness: 0.02,
                p2: [0.0, 0.5, -0.5],
                restitution: 0.0,
                flags: 1,
                _pad: [0.0; 3],
            },
            GpuMeshTriangle {
                p0: [-0.5, 0.0, 0.0],
                friction: 0.0,
                p1: [0.0, -0.5, -0.5],
                thickness: 0.02,
                p2: [0.5, 0.0, 0.0],
                restitution: 0.0,
                flags: 1,
                _pad: [0.0; 3],
            },
        ];
        sim.set_mesh_triangles(&roof_tris);
        sim.step(0.016, 5);

        let verts = sim.read_vertices();
        let mid_z = 0.5 * (verts[0].position[2] + verts[1].position[2]);
        assert!(
            mid_z > 0.0,
            "片面凸稜線への弦状食い込みは内部(-Z)ではなく外向き法線(+Z)側へ押し戻されるべき (mid_z={})",
            mid_z
        );
    }

    #[test]
    fn test_edge_stiffness_multipliers_and_stiffness_all_composition() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let positions = vec![[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]];
        let edges = vec![[0, 1], [1, 2]];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, None, None, None, None, None,
            0, 0.005, 1000.0, 1000.0, 500.0, 10.0, 1.0, None, None,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);

        // 初期コンプライアンス = 1/1000 = 0.001
        assert!((sim.get_edge_tension_compliance(0) - 0.001).abs() < 1e-6);
        assert!((sim.get_edge_tension_compliance(1) - 0.001).abs() < 1e-6);

        // エッジ0のみ multiplier=4.0 を設定 -> コンプライアンス = 0.001 / 4 = 0.00025
        sim.set_edge_stiffness_multipliers(&[0], &[4.0]);
        assert!((sim.get_edge_tension_compliance(0) - 0.00025).abs() < 1e-6);
        assert!((sim.get_edge_tension_compliance(1) - 0.001).abs() < 1e-6);

        // set_stiffness_all でベース剛性を 2000.0 に変更しても multiplier=4.0 が維持され、
        // export_config の tension_stiffness はベース値 2000.0 を返すこと
        sim.set_stiffness_all(2000.0, 2000.0, 500.0, 10.0);
        assert!((sim.get_edge_tension_compliance(0) - 0.000125).abs() < 1e-6);
        assert!((sim.get_edge_tension_compliance(1) - 0.0005).abs() < 1e-6);
        assert!((sim.export_config().tension_stiffness - 2000.0).abs() < 1e-3);

        // リセットで 1.0 倍に戻ること
        sim.reset_edge_stiffness_multipliers();
        assert!((sim.get_edge_tension_compliance(0) - 0.0005).abs() < 1e-6);
    }

    #[test]
    fn test_dynamic_sewing_shrink_speed_preserves_rest_len() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let edges: Vec<[u32; 2]> = vec![];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, None, None, Some(&[[0u32, 1u32]]),
            None, None, 0, 0.005, 1000.0, 1000.0, 500.0, 0.0, 0.5, Some(1000.0), Some(false),
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_gravity(0.0, 0.0, 0.0);

        // dt=0.1s, shrink_speed=0.5 m/s -> 1ステップで自然長 1.0 -> 0.95m に収縮
        sim.step(0.1, 4);
        let r1 = sim.read_sewing_current_rest_lengths()[0];
        assert!((r1 - 0.95).abs() < 1e-3, "r1={}", r1);

        // 実行中に shrink_speed を 2.0 m/s に動的変更しても進行済み自然長 r1 が 1.0 に巻き戻らず、
        // 次ステップ dt=0.1s で 0.95 -> 0.75m へ加速収縮すること
        sim.set_sewing_shrink_speed(2.0);
        assert!((sim.export_config().sewing_shrink_speed - 2.0).abs() < 1e-6);
        let r_after_set = sim.read_sewing_current_rest_lengths()[0];
        assert!((r_after_set - 0.95).abs() < 1e-3, "r_after_set={}", r_after_set);

        sim.step(0.1, 4);
        let r2 = sim.read_sewing_current_rest_lengths()[0];
        assert!((r2 - 0.75).abs() < 1e-3, "r2={}", r2);
    }

    #[test]
    fn test_wrinkle_field_z_and_radial_range_guard() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };
        // 頂点0: 範囲内 (r=0.07, z=0.0)
        // 頂点1: Z範囲外 (r=0.07, z=0.50 > z_max=0.10)
        // 頂点2: 径方向範囲外 (r=0.30 > r_max(0.10) + influence_radius(0.03))
        // 頂点3: ボーン中心軸上特異点 (r=0.0, z=0.0)
        let positions = vec![
            [0.07, 0.0, 0.0],
            [0.07, 0.0, 0.50],
            [0.30, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let edges: Vec<[u32; 2]> = vec![];
        let mesh = ClothMesh::from_raw(
            &positions, &edges, None, None, None,
            None, None, 0, 0.005, 1000.0, 1000.0, 500.0, 0.0, 1.0, None, None,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_gravity(0.0, 0.0, 0.0);

        let wf_params = crate::simulation::WrinkleFieldParams {
            bone_origin: [0.0, 0.0, 0.0, 0.03],
            bone_axis: [0.0, 0.0, 1.0, 0.05],
            bone_normal: [1.0, 0.0, 0.0, 40.0],
            bone_binormal: [0.0, 1.0, 0.0, 1.0],
            z_range: [-0.10, 0.10],
            r_range: [0.04, 0.10],
            enabled: 1,
            use_texture: 1,
            valley_window: 0.015,
            crest_window: 0.020,
        };
        sim.set_wrinkle_field_params(&wf_params);
        // 全面で谷強度 R=255, G=0, B=0 (target_valley_r = 0.04), A=255
        let tex_bytes: Vec<u8> = [255u8, 0, 0, 255].repeat(16 * 16);
        sim.set_wrinkle_field_texture_2d(16, 16, &tex_bytes, -0.10, 0.10, 0.04, 0.10);
        sim.set_enable_wrinkle_field(true);

        sim.step(0.016, 2);
        let verts = sim.read_vertices();
        // 頂点0 (範囲内) は谷へ引き込まれて X < 0.07 になる
        assert!(verts[0].position[0] < 0.069, "v0 should be pulled in: {:?}", verts[0].position);
        // 頂点1 (Z範囲外)、頂点2 (径方向範囲外)、頂点3 (軸上特異点) は一切動かない
        assert!((verts[1].position[0] - 0.07).abs() < 1e-6, "v1 leaked: {:?}", verts[1].position);
        assert!((verts[2].position[0] - 0.30).abs() < 1e-6, "v2 leaked: {:?}", verts[2].position);
        assert!(verts[3].position[0].abs() < 1e-6 && verts[3].position[1].abs() < 1e-6, "v3 jumped: {:?}", verts[3].position);
    }

    #[test]
    fn test_untangling_no_false_positive_on_wrinkled_inner_and_face_to_face_flip() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // ケース1: 中央に凸シワが寄った内層(layer=0)の表側近傍(+3mm)に外層(layer=1)を配置
        // 旧V-V接平面Untanglingでは斜め頂点法線により外層が裏側と誤認され突き上げ/引き込みが発生していた
        let nx = 5usize;
        let ny = 5usize;
        let dx = 0.02f32;
        let mut positions = Vec::new();
        let mut edges = Vec::new();
        let mut faces = Vec::new();
        let mut layer_ids = Vec::new();
        let mut inv_masses = Vec::new();

        for layer in 0..2u32 {
            let base = positions.len() as u32;
            for iy in 0..ny {
                for ix in 0..nx {
                    let x = (ix as f32 - 2.0) * dx;
                    let y = (iy as f32 - 2.0) * dx;
                    // 中央列 ix=2 に高さ 12mm の凸シワを形成（法線が左右へ大きく傾く）
                    let wrinkle = if ix == 2 { 0.012 } else if ix == 1 || ix == 3 { 0.004 } else { 0.0 };
                    let z = wrinkle + if layer == 1 { 0.003 } else { 0.0 };
                    positions.push([x, y, z]);
                    layer_ids.push(layer);
                    // 両層の外周を固定、中央部は完全に動的として相互作用を確認
                    let is_border = ix == 0 || ix == nx - 1 || iy == 0 || iy == ny - 1;
                    inv_masses.push(if is_border { 0.0 } else { 1.0 });
                }
            }
            for iy in 0..ny {
                for ix in 0..nx {
                    let v0 = base + (iy * nx + ix) as u32;
                    if ix + 1 < nx { edges.push([v0, v0 + 1]); }
                    if iy + 1 < ny { edges.push([v0, v0 + nx as u32]); }
                    if ix + 1 < nx && iy + 1 < ny {
                        let v1 = v0 + 1;
                        let v2 = v0 + nx as u32 + 1;
                        let v3 = v0 + nx as u32;
                        edges.push([v0, v2]);
                        faces.push([v0, v1, v2]);
                        faces.push([v0, v2, v3]);
                    }
                }
            }
        }

        let mesh1 = ClothMesh::from_raw(
            &positions,
            &edges,
            Some(&faces),
            Some(&inv_masses),
            None,
            Some(&layer_ids),
            None,
            0,
            0.0014,
            1500.0,
            150.0,
            100.0,
            15.0,
            1.0,
            None,
            None,
        );
        let mut sim1 = GpuClothSimulator::new(Arc::clone(&ctx), mesh1);
        sim1.set_gravity(0.0, 0.0, 0.0);
        sim1.set_damping(2.0);
        sim1.set_enable_self_collision(true);
        sim1.set_self_collision_options(1.0, 0.5, true, true, 512);

        for _ in 0..10 {
            sim1.step(1.0 / 60.0, 10);
        }
        let verts1 = sim1.read_vertices();
        let per_layer = nx * ny;
        for i in 0..per_layer {
            let z_in = verts1[i].position[2];
            let z_out = verts1[per_layer + i].position[2];
            assert!(
                z_out > z_in - 1e-4,
                "凸シワ表側近傍で外層が内層裏側へ引き込まれた: v{} z_in={:.5}, z_out={:.5}",
                i, z_in, z_out
            );
            assert!(
                (z_out - positions[per_layer + i][2]).abs() < 0.008,
                "凸シワ誤爆により過剰変位が発生: v{} z_out={:.5} init={:.5}",
                i, z_out, positions[per_layer + i][2]
            );
        }

        // ケース2: 動的対向法線（表同士接触＝フード背中めくり状態）での衝突死角・誤発火防止
        // inner(layer=0): 7x7 (-0.06..+0.06), z=0.0, 法線 +Z（外周固定、内部動的）
        // outer(layer=1): 5x5 (-0.04..+0.04), z=0.006, 頂点順反転で法線 -Z（全面動的、内層内側領域へ落下）
        let nx0 = 7usize;
        let ny0 = 7usize;
        let nx1 = 5usize;
        let ny1 = 5usize;
        let mut pos2 = Vec::new();
        let mut edges2 = Vec::new();
        let mut faces2 = Vec::new();
        let mut layers2 = Vec::new();
        let mut inv2 = Vec::new();
        let mut vel2 = Vec::new();

        // layer 0 (0..49)
        for iy in 0..ny0 {
            for ix in 0..nx0 {
                let x = (ix as f32 - 3.0) * dx;
                let y = (iy as f32 - 3.0) * dx;
                pos2.push([x, y, 0.0]);
                layers2.push(0u32);
                let is_border = ix == 0 || ix == nx0 - 1 || iy == 0 || iy == ny0 - 1;
                inv2.push(if is_border { 0.0 } else { 1.0 });
                vel2.push([0.0, 0.0, 0.0]);
            }
        }
        for iy in 0..ny0 {
            for ix in 0..nx0 {
                let v0 = (iy * nx0 + ix) as u32;
                if ix + 1 < nx0 { edges2.push([v0, v0 + 1]); }
                if iy + 1 < ny0 { edges2.push([v0, v0 + nx0 as u32]); }
                if ix + 1 < nx0 && iy + 1 < ny0 {
                    let v1 = v0 + 1;
                    let v2 = v0 + nx0 as u32 + 1;
                    let v3 = v0 + nx0 as u32;
                    edges2.push([v0, v2]);
                    faces2.push([v0, v1, v2]);
                    faces2.push([v0, v2, v3]);
                }
            }
        }

        // layer 1 (49..74): 法線 -Z（内層の表面 +Z と表同士で向かい合う状態）
        let base1 = pos2.len() as u32;
        for iy in 0..ny1 {
            for ix in 0..nx1 {
                let x = (ix as f32 - 2.0) * dx;
                let y = (iy as f32 - 2.0) * dx;
                pos2.push([x, y, 0.006]);
                layers2.push(1u32);
                inv2.push(1.0);
                vel2.push([0.0, 0.0, -0.5]);
            }
        }
        for iy in 0..ny1 {
            for ix in 0..nx1 {
                let v0 = base1 + (iy * nx1 + ix) as u32;
                if ix + 1 < nx1 { edges2.push([v0, v0 + 1]); }
                if iy + 1 < ny1 { edges2.push([v0, v0 + nx1 as u32]); }
                if ix + 1 < nx1 && iy + 1 < ny1 {
                    let v1 = v0 + 1;
                    let v2 = v0 + nx1 as u32 + 1;
                    let v3 = v0 + nx1 as u32;
                    edges2.push([v0, v2]);
                    faces2.push([v0, v2, v1]);
                    faces2.push([v0, v3, v2]);
                }
            }
        }

        let mesh2 = ClothMesh::from_raw(
            &pos2,
            &edges2,
            Some(&faces2),
            Some(&inv2),
            None,
            Some(&layers2),
            None,
            0,
            0.002,
            1500.0,
            150.0,
            100.0,
            15.0,
            1.0,
            None,
            None,
        );
        let mut sim2 = GpuClothSimulator::new(ctx, mesh2);
        sim2.set_positions_and_velocities(&pos2, Some(&vel2));
        sim2.set_gravity(0.0, 0.0, -9.81);
        sim2.set_damping(1.0);
        sim2.set_enable_self_collision(true);
        sim2.set_self_collision_options(1.0, 0.5, true, true, 512);

        for _ in 0..15 {
            sim2.step(1.0 / 60.0, 15);
        }
        let verts2 = sim2.read_vertices();
        let inner_tris: Vec<[usize; 3]> = faces2
            .iter()
            .filter(|f| f[0] < base1 && f[1] < base1 && f[2] < base1)
            .map(|f| [f[0] as usize, f[1] as usize, f[2] as usize])
            .collect();

        let point_in_tri_xy = |px: f32, py: f32, a: [f32; 3], b: [f32; 3], c: [f32; 3]| -> Option<f32> {
            let v0x = b[0] - a[0];
            let v0y = b[1] - a[1];
            let v1x = c[0] - a[0];
            let v1y = c[1] - a[1];
            let det = v0x * v1y - v1x * v0y;
            if det.abs() < 1e-10 {
                return None;
            }
            let qx = px - a[0];
            let qy = py - a[1];
            let u = (qx * v1y - v1x * qy) / det;
            let v = (v0x * qy - qx * v0y) / det;
            let w = 1.0 - u - v;
            if u >= -1e-3 && v >= -1e-3 && w >= -1e-3 {
                Some(w * a[2] + u * b[2] + v * c[2])
            } else {
                None
            }
        };

        for out_idx in base1 as usize..verts2.len() {
            let po = verts2[out_idx].position;
            let mut matched_tri_z = None;
            for tri in &inner_tris {
                if let Some(tz) = point_in_tri_xy(
                    po[0],
                    po[1],
                    verts2[tri[0]].position,
                    verts2[tri[1]].position,
                    verts2[tri[2]].position,
                ) {
                    matched_tri_z = Some(tz);
                    break;
                }
            }
            let z_in = matched_tri_z.expect("外層頂点が内層XY領域の外へ外れた");
            assert!(
                po[2] > z_in + 0.0005,
                "表同士対向めくり接触で外層が内層を貫通した: out_idx={} pos={:?}, inner_z={:.5}",
                out_idx, po, z_in
            );
        }
    }

    #[test]
    fn test_untangling_1d_strand_behind_dynamic_inner_escapes() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // 内層 L0 (0..24): 5x5 平板 (dx=0.006m, z=0.0, 法線 +Z, 外周固定・内部 3x3 は動的)
        // 外層 1D ストランド L1 (25..29): L0 < L1 順序・faces なし・中央3頂点を裏側 z=-0.0018 に配置
        let nx = 5usize;
        let ny = 5usize;
        let dx = 0.006f32;
        let mut positions = Vec::new();
        let mut edges = Vec::new();
        let mut faces = Vec::new();
        let mut layer_ids = Vec::new();
        let mut inv_masses = Vec::new();
        let mut thicknesses = Vec::new();

        for iy in 0..ny {
            for ix in 0..nx {
                let x = (ix as f32 - 2.0) * dx;
                let y = (iy as f32 - 2.0) * dx;
                positions.push([x, y, 0.0]);
                layer_ids.push(0u32);
                let is_border = ix == 0 || ix == nx - 1 || iy == 0 || iy == ny - 1;
                inv_masses.push(if is_border { 0.0 } else { 200000.0 });
                thicknesses.push(0.0014);
            }
        }
        for iy in 0..ny {
            for ix in 0..nx {
                let v0 = (iy * nx + ix) as u32;
                if ix + 1 < nx { edges.push([v0, v0 + 1]); }
                if iy + 1 < ny { edges.push([v0, v0 + nx as u32]); }
                if ix + 1 < nx && iy + 1 < ny {
                    let v1 = v0 + 1;
                    let v2 = v0 + nx as u32 + 1;
                    let v3 = v0 + nx as u32;
                    edges.push([v0, v2]);
                    faces.push([v0, v1, v2]);
                    faces.push([v0, v2, v3]);
                }
            }
        }

        // L1 1D ストランド (25..29): x = -0.012 .. +0.012, y = 0.0
        let strand_base = positions.len() as u32;
        for i in 0..5usize {
            let x = (i as f32 - 2.0) * dx;
            let z = if i == 0 || i == 4 { 0.0025 } else { -0.0018 };
            positions.push([x, 0.0, z]);
            layer_ids.push(1u32);
            inv_masses.push(40000.0);
            thicknesses.push(0.0022);
        }
        for i in 0..4u32 {
            edges.push([strand_base + i, strand_base + i + 1]);
        }

        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            Some(&faces),
            Some(&inv_masses),
            None,
            Some(&layer_ids),
            Some(&thicknesses),
            0,
            0.0014,
            1500.0,
            150.0,
            100.0,
            15.0,
            1.0,
            None,
            None,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        sim.set_gravity(0.0, 0.0, 0.0);
        sim.set_damping(2.0);
        sim.set_enable_self_collision(true);
        sim.set_self_collision_options(1.0, 0.5, true, true, 512);

        for _ in 0..5 {
            sim.step(1.0 / 60.0, 15);
        }
        let verts = sim.read_vertices();
        for i in 0..5usize {
            let idx = strand_base as usize + i;
            let po = verts[idx].position;
            assert!(
                po[2] > 0.0015,
                "動的L0の裏側にある1Dストランドが表側へ脱出できていない: v{} pos={:?}",
                idx, po
            );
        }
    }

    #[test]
    fn test_apply_distance_compliances_preserves_disabled_sentinel_and_current_bending_quad() {
        let ctx = match get_test_context() {
            Some(c) => c,
            None => return,
        };

        // 1D ストランドのみ（_pad[0] == 2.0 が先頭になる構成）で曲げ剛性 200.0、伸縮剛性 0.0（無効 1e10）を設定
        let positions = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]];
        let edges = [[0, 1], [1, 2]];
        let mesh = ClothMesh::from_raw(
            &positions,
            &edges,
            None,
            Some(&[0.0, 1.0, 1.0]),
            None,
            None,
            None,
            0,
            0.005,
            0.0, // tension_stiffness = 0 -> base_tension_compliance = 1e10
            0.0,
            0.0,
            200.0,
            1.0,
            None,
            None,
        );
        let mut sim = GpuClothSimulator::new(ctx, mesh);
        assert!(sim.base_tension_compliance >= 1e9);

        // 個別剛性倍率 100.0 を与えても 1e10 が 1e8 へ復活しないこと
        sim.edge_stiffness_multipliers = vec![100.0; sim.distance_constraints.len()];
        sim.apply_distance_compliances();
        for c in &sim.distance_constraints {
            assert!(
                c.tension_compliance >= 1e9 && c.compression_compliance >= 1e9,
                "stiffness=0 の無効拘束が倍率適用で復活した: {:?}",
                c
            );
        }

        // 先頭が 1D ストランド (_pad[0] == 2.0, クランプ済み 0.5) でも export_config が 0.2 に誤逆算されないこと
        let cfg = sim.export_config();
        assert!(
            (cfg.bending_stiffness - 200.0).abs() < 1e-4,
            "1Dストランド先頭時の bending_stiffness が誤逆算された: {}",
            cfg.bending_stiffness
        );

        sim.set_stiffness_all(1000.0, 1000.0, 500.0, 12.5);
        let cfg2 = sim.export_config();
        assert!(
            (cfg2.bending_stiffness - 12.5).abs() < 1e-4,
            "set_stiffness_all 後の bending_stiffness が一致しない: {}",
            cfg2.bending_stiffness
        );
    }

    #[test]
    fn test_validate_dynamic_sdf_output_size_overflow_and_limits() {
        // aligned_bytes_per_row が u32 上限を超える幅でパニックせず Err を返す
        let err_row_u32 = GpuClothSimulator::validate_dynamic_sdf_output_size(
            1u32 << 30,
            1,
            1,
            64,
            u64::MAX,
            u64::MAX,
        );
        assert!(err_row_u32.is_err());

        // u64 総バイト数乗算オーバーフローを起こす巨大寸法でパニックせず Err を返す
        let err_total_u64 = GpuClothSimulator::validate_dynamic_sdf_output_size(
            1u32 << 28,
            1u32 << 20,
            1u32 << 20,
            64,
            u64::MAX,
            u64::MAX,
        );
        assert!(err_total_u64.is_err());

        // GPU バインディング上限超過で安全に拒否される
        let err_limit = GpuClothSimulator::validate_dynamic_sdf_output_size(
            512,
            512,
            512,
            64,
            512 * 1024 * 1024,
            128 * 1024 * 1024,
        );
        assert!(err_limit.is_err());

        // 解像度 0 拒否
        let err_zero = GpuClothSimulator::validate_dynamic_sdf_output_size(
            0,
            64,
            64,
            64,
            u64::MAX,
            u64::MAX,
        );
        assert!(err_zero.is_err());

        // 正常値受理
        let ok = GpuClothSimulator::validate_dynamic_sdf_output_size(
            64,
            64,
            64,
            64,
            256 * 1024 * 1024,
            128 * 1024 * 1024,
        );
        assert_eq!(ok.unwrap(), (256, 256 * 64 * 64));
    }
}

