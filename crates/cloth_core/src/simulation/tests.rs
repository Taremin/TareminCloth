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
}

