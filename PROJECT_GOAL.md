# 【GPU Cloth アドオン】自律実装用 ゴール仕様書 (Goal Specification)

## 1. プロジェクト概要 & 最終ゴール (Project Goal)

### 1.1 目的
Blender上で動作する、**GPUベンダー非依存（NVIDIA / AMD / Intel / Apple Silicon対応）** のリアルタイム・クロスシミュレーション・アドオンを開発する。
多層（マルチレイヤー）の布地シミュレーションと、3Dビューポート上でのインタラクティブな編集・ドラッグ操作を実現する。

### 1.2 達成条件 (Acceptance Criteria)
以下の条件をすべて満たした時点で本ゴールは達成（Goal Achieved）とする。

1. **GPUベンダー非依存コンピュートの動作**:
   - `wgpu` (WGSL Compute Shader) をバックエンドとし、CUDA等の特定ベンダー専用APIに依存しない。
   - Vulkan / DirectX 12 / Metal で動作可能であること。
2. **Blender統合とビルド自動化**:
   - Rustコアが `maturin` または `pyo3` 経由でPython拡張モジュール（`.pyd` / `.so`）としてビルドされ、Blenderから直接インポート可能であること。
   - Blenderのアドオンとして有効化でき、サイドバー（Nパネル）にUIが表示されること。
3. **リアルタイム XPBD シミュレーション**:
   - 距離拘束（Distance Constraints）、曲げ拘束（Bending Constraints）、ピン留め拘束（Pin/Attachment Constraints）がGPU上で計算されること。
   - 拘束の並列化（グラフ彩色 または 頂点中心Jacobi型PBD）により、データ競合なくGPU完全並列で実行されること。
4. **インタラクティブ操作（モーダルオペレーター）**:
   - 3Dビューポート上でシミュレーションを実行しながら、マウスドラッグ等で頂点を掴んで動かせること。
   - 剛性、重力、減衰などのパラメータ変更がシミュレーション中に即座に反映されること。
5. **ピン留め・アタッチメント（Pinning & Attachment）**:
   - 頂点グループ（ウェイト値）による固定ピン / 部分固定。
   - 他オブジェクト（ボーンやアバター）に追従するアタッチメント拘束。
   - 3Dビューポート上でインタラクティブにピンを追加・解除・移動できること。
6. **縫合機能（Sewing Constraints）**:
   - 2つの境界エッジ間（型紙の合わせ目）を縫い合わせるスプリング/距離拘束。
   - シミュレーション開始時に初期距離から目標長（通常ゼロ）へ徐々に収縮し、服を仕立てられること。
7. **マルチレイヤー（多層布）対応**:
   - 複数の布メッシュに異なるレイヤー番号（内側〜外側）および厚み（Thickness）を設定できること。
   - GPU空間ハッシュ（Spatial Hashing）等を用いた衝突判定により、レイヤー間の突き抜けが防止されること。
8. **テスト駆動開発 (TDD) による完全自動検証**:
   - すべての機能は「テスト先行」で実装され、`cargo test` および `python run_tests.py` で検証されること。

### 1.3 開発アプローチ：テスト駆動開発 (TDD) の徹底
自律エージェント開発における手戻り・退行（Regression）を防止するため、**厳格なTDDサイクル（Red-Green-Refactor）** を適用する。

```text
[ 1. テスト作成 (Red) ]
   ・実装前に期待される入力・出力をテストケースとして記述（テストが失敗することを確認）
      ↓
[ 2. 最小限の実装 (Green) ]
   ・テストを通過させるための最小限のコードを実装
      ↓
[ 3. 検証 & リファクタリング (Refactor) ]
   ・cargo test / python run_tests.py を実行してオールグリーンを確認し、コードを整理
```
* **Pure Rust コア**: `cloth_core` 内の拘束計算、彩色、トポロジ抽出はすべて Rust の単体テスト (`#[test]`) で駆動する。
* **Python / Blender 統合**: `tests/` 内の `unittest` を `python run_tests.py` でヘッドレス実行し、E2E で挙動を検証する。

#### ソフトウェアテスト工学に基づくテスト設計原則
形骸的な「正常系が1回通るだけ」の浅いテストは禁止し、以下の工学的観点に基づいた堅牢なテストケースを必ず設計すること：

1. **境界値分析 (Boundary Value Analysis) と同値分割**:
   - 剛性パラメータ（ゼロ剛性、標準剛性、極端に大きな剛性 $10^6$）での挙動。
   - 質量とピン留め（`inv_mass == 0.0` の固定ピン、極小質量、均一質量）。
   - サブステップ数（1ステップ、通常20ステップ、多サブステップ100）。
   - 頂点数・エッジ数の極値（1頂点、1三角形、1000頂点グリッド）。
2. **物理シミュレーション特有の不変量 (Invariants) & サニティチェック**:
   - **NaN / Inf の絶対検知**: シミュレーションの各ステップ後、全頂点の `position` および `velocity` が有限値（`is_finite()`）であることを保証。
   - **エネルギー保存と爆発（Blow-up）防止**: 減衰なし・外力なしの閉じた系で総エネルギーが急激に増大しないこと。
   - **幾何学的対称性テスト**: 左右対称なメッシュ・拘束・外力に対して、左右対称な変形結果が維持されること。
   - **拘束トレランス収束**: 距離拘束が目標長に対して許容誤差（例: $\pm 3\%$ 以内）に収束していること。
3. **エッジケースと異常系・頑健性テスト (Robustness & Edge Cases)**:
   - 縮退ジオメトリ（面積0の極小三角形、長さ0のエッジ、重複エッジ）。
   - 非多様体トポロジ（1本のエッジに3つ以上の面が集まる場合など）に対するエラー処理または安全なフォールバック。
   - ゼロ除算の回避（同一座標に配置された頂点ペアの距離計算・正規化）。
4. **状態遷移とライフサイクル検証**:
   - シミュレーションのリセット（Reset）時に、初期頂点座標が完全にビット単位で復元されること。
   - パラメータの動的変更（シミュレーション実行中に重力や剛性を書き換えた際）にクラッシュせず追従すること。

---

## 2. システムアーキテクチャ & 技術スタック

### 2.1 技術選定

| コンポーネント | 技術 | 役割・選定理由 |
| :--- | :--- | :--- |
| **GPU API** | **wgpu (v0.20+ / v22+)** | WebGPU仕様準拠のRust実装。DirectX 12, Vulkan, Metalへ自動抽象化。CUDA不要。 |
| **Shader言語** | **WGSL (WebGPU Shading Language)** | wgpu標準のコンピュートシェーダー言語。 |
| **ネイティブコア** | **Rust (2021 edition)** | 高速性、メモリ安全性、マルチプラットフォーム対応。 |
| **Python連携** | **PyO3 + rust-numpy** | NumPy配列のポインタをゼロコピーでRustに受け渡し。 |
| **ビルドツール** | **Maturin** | RustクレートをPythonパッケージ/拡張モジュールとしてワンコマンドでビルド。 |
| **アドオンUI/連携**| **Blender Python (`bpy`)** | Blender 3.6 LTS / 4.0+ 対応。モーダルオペレーター、パネルUI、メッシュ評価。 |

### 2.2 リポジトリ構造

```text
taremin_cloth/
├── Cargo.toml                  # Rust ワークスペース設定
├── pyproject.toml              # maturin ビルド設定
├── README.md
├── PROJECT_GOAL.md             # 本仕様書
├── crates/
│   ├── cloth_core/             # 物理・GPU計算コア（Pure Rust / wgpu）
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── context.rs      # wgpu Device / Queue 管理・GPUバックエンド/デバイス動的選択
│   │       ├── mesh.rs         # 頂点・トポロジ・拘束グラフデータ構造
│   │       ├── coloring.rs     # 拘束のグラフ彩色アルゴリズム
│   │       ├── spatial_hash.rs # GPU空間ハッシュグリッド構築
│   │       ├── simulation.rs   # XPBDシミュレーション実行・パイプライン管理
│   │       └── shaders/
│   │           ├── predict.wgsl      # 位置予測・外力適用
│   │           ├── distance.wgsl     # 距離拘束投影
│   │           ├── bending.wgsl      # 曲げ拘束投影
│   │           ├── pin.wgsl          # 固定ピン・追従ピン拘束
│   │           ├── sewing.wgsl       # 縫合拘束投影（収縮スプリング）
│   │           ├── collision.wgsl    # 衝突拘束・空間ハッシュ
│   │           └── update_vel.wgsl   # 速度更新・確定
│   └── cloth_py/               # PyO3 Python バインディング
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs          # Pythonモジュール定義 (taremin_cloth_core)
├── python/
│   └── taremin_cloth/          # Blenderアドオン本体
│       ├── __init__.py         # アドオン登録・メタデータ
│       ├── operators.py        # モーダルシミュレーション・インタラクション操作
│       ├── panels.py           # Nパネル UI定義
│       ├── properties.py       # プロパティグループ定義
│       └── utils/
│           ├── mesh_extract.py # bpy.types.Mesh から NumPy 配列への高速抽出
│           └── drawing.py      # ビューポートGPUプレビュー描画（bpy.gpu）
├── tools/
│   └── blender_manager.py      # Blender自動検出・DL・テスト実行マネージャー
├── run_tests.py                # Blenderヘッドレステスト実行ランナー
└── tests/
    ├── test_cloth_core.py      # Python単体でのコア機能テスト
    └── test_simulation_e2e.py  # Blender内ヘッドレス結合テスト
```

---

## 3. 詳細仕様

### 3.1 物理アルゴリズム (XPBD: Extended Position Based Dynamics)

各フレーム（$\Delta t \approx 1/60$ 秒）において、複数回のサブステップ（$N_{sub} = 10 \sim 30$）を実行する。
サブステップ時間刻み: $dt = \Delta t / N_{sub}$

#### 1サブステップ内の処理シーケンス:
1. **位置予測 (Predict Positions)**:
   $$v_i \leftarrow v_i + dt \cdot M^{-1} f_{ext}$$
   $$p_i \leftarrow x_i + dt \cdot v_i$$
   （$x_i$: 現在位置, $p_i$: 予測位置, $v_i$: 速度, $M$: 質量行列, $f_{ext}$: 重力・風など）
2. **空間ハッシュ構築 (Build Spatial Hash)**:
   - 各頂点のグリッドセルIDを算出し、GPUバッファにキー/値ペアとして格納。
   - ソートまたはセルポインタ配列を生成。
3. **拘束解消ループ (Constraint Projection Loop)**:
   - **距離拘束 (Distance Constraints)**:
     - エッジ長 $L_0$ に対して、伸縮剛性 $\alpha_{dist} = \frac{1}{k \cdot dt^2}$ を考慮したXPBD補正量を計算。
     - グラフ彩色された色グループごとに並列実行。
   - **曲げ拘束 (Bending Constraints)**:
     - 共有稜線を持つ隣接2三角形（4頂点）の2面角（Dihedral Angle）またはIsometric Bending拘束を投影。
     - 曲げ剛性 $\alpha_{bend}$ を適用。
   - **ピン留め・アタッチメント拘束 (Pin & Attachment Constraints)**:
     - 完全固定ピン（`inv_mass = 0.0`）、または目標座標・外部ボーン・アバターに追従するターゲット位置 $x_{target}$ へ補正（ウェイト値 $w_{pin} \in [0, 1]$ を適用）。
     - マウスドラッグ中の頂点をリアルタイムに追従させる。
   - **縫合拘束 (Sewing Constraints)**:
     - 衣服の型紙エッジ同士を縫い合わせる可変長スプリング拘束。
     - 目標自然長 $L(t) = \max(L_{target}, L_0 - v_{sew} \cdot t)$ により、シミュレーション開始とともに徐々に収縮して型紙を引き合わせる。
   - **衝突拘束 (Collision & Multi-layer Constraints)**:
     - 剛体コライダー（SDF球/カプセル/人体メッシュ）との侵入判定・位置押し戻し。
     - レイヤー間衝突: 空間ハッシュから近傍頂点を検索し、レイヤー番号が異なるメッシュ同士で距離が $Thickness_A + Thickness_B$ 未満の場合に反発補正を適用。
4. **速度更新と位置確定 (Velocity Update & Commit)**:
   $$v_i \leftarrow (p_i - x_i) / dt$$
   $$x_i \leftarrow p_i$$

### 3.2 データ構造 (Rust & GPU Layout)

GPUアライメント（16バイト境界）に注意したStruct定義を行う。

```rust
// 頂点データ (GPU Buffer: Read/Write)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 3],  // 現在位置 x_i
    pub inv_mass: f32,       // 逆質量 (0.0 = 固定ピン)
    pub prev_pos: [f32; 3],  // 直前位置 (または予測位置 p_i)
    pub layer_id: u32,       // レイヤー番号 (0, 1, 2...)
    pub velocity: [f32; 3],  // 速度 v_i
    pub thickness: f32,      // 布の厚み (m)
}

// 距離拘束 (GPU Buffer: Read-Only)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDistanceConstraint {
    pub v0: u32,             // 頂点インデックス 0
    pub v1: u32,             // 頂点インデックス 1
    pub rest_length: f32,    // 初期長
    pub compliance: f32,     // コンプライアンス (1 / 剛性)
}

// 曲げ拘束 (GPU Buffer: Read-Only)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBendingConstraint {
    pub v0: u32,             // 共有稜線 頂点0
    pub v1: u32,             // 共有稜線 頂点1
    pub v2: u32,             // 翼頂点 0
    pub v3: u32,             // 翼頂点 1
    pub rest_angle: f32,     // 初期2面角 (または初期曲率)
    pub compliance: f32,     // コンプライアンス
    pub _pad: [f32; 2],
}

// ピン・アタッチメント拘束 (GPU Buffer: Read/Write)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPinConstraint {
    pub vertex_idx: u32,     // 対象頂点インデックス
    pub weight: f32,         // ピン留め強度 (0.0〜1.0)
    pub _pad: [f32; 2],
    pub target_pos: [f32; 3],// 目標ワールド座標
    pub _pad2: f32,
}

// 縫合拘束 (GPU Buffer: Read-Only)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSewingConstraint {
    pub v0: u32,             // 縫合元頂点
    pub v1: u32,             // 縫合先頂点
    pub current_rest_len: f32,// 現在の自然長（時間経過で縮小）
    pub target_rest_len: f32, // 最終目標長 (通常 0.0)
    pub shrink_speed: f32,   // 収縮速度 (m/s)
    pub compliance: f32,     // コンプライアンス
    pub _pad: [f32; 2],
}
```

### 3.3 拘束並列化 (グラフ彩色仕様)

1. **彩色アルゴリズム**:
   - メッシュ初期化時（CPU・Rust側）にWelsh-Powell法（次数降順貪欲法）を実行。
   - 距離拘束リストを「色グループ配列」に再編成:
     - `color_offsets: Vec<u32>` (例: `[0, 15000, 30000, ...]`)
     - `color_counts: Vec<u32>` (各色の拘束数)
2. **GPUディスパッチ**:
   - ComputePass 内で、色グループごとに `set_pipeline` と `dispatch_workgroups((count + 255) / 256)` を発行。
   - ※初期プロトタイプ段階では、開発スピードを優先して「頂点中心Jacobi法（彩色不要）」を第1弾とし、パフォーマンス最適化フェーズでグラフ彩色に移行する選択肢も許容する。

### 3.4 Blender Python インターフェース

#### ゼロコピーデータ受け渡し
Blender側でNumPy配列の頂点バッファを用意し、Rustバインディングに渡す。

```python
# Blender Python 側の呼び出しイメージ
import numpy as np
import taremin_cloth_core as core

# 頂点座標の取得 (NumPy配列への直接コピー)
n_verts = len(mesh.vertices)
coords = np.empty(n_verts * 3, dtype=np.float32)
mesh.vertices.foreach_get("co", coords)

# シミュレータの初期化 (Rust)
sim = core.ClothSimulator(
    positions=coords,
    edges=edge_indices_np,
    faces=face_indices_np,
    layer_id=0,
    thickness=0.005,
    stiffness=1000.0,
)

# シミュレーション1ステップ進行 (GPU上で計算)
sim.step(dt=0.0166, substeps=20)

# 結果の頂点座標をBlenderメッシュに書き戻し
sim.get_positions(coords)
mesh.vertices.foreach_set("co", coords)
mesh.update()
```

#### インタラクティブ・モーダルオペレーター (`taremin.cloth_interactive`)
- ユーザーが「Interactive Mode」を開始すると、`modal()` が稼働。
- **イベントハンドリング**:
  - `TIMER`: 毎フレーム `sim.step()` を呼び出してメッシュを更新。
  - `LEFTMOUSE (Click & Drag)`: 3Dビューポートのレイキャスト（`bpy_extras.view3d_utils.region_2d_to_origin_3d`）で最も近い頂点を検出し、マウス移動差分に応じてピン位置をRust側に更新（`sim.set_pin_position(v_idx, new_pos)`）。
  - `ESC` / `RIGHTMOUSE`: インタラクティブモード終了（確定またはキャンセル）。

---

## 4. 段階的実装ロードマップ (TDD Milestones)

自律エージェントは各Phaseにおいて、**「1. テスト作成(Red) → 2. 実装(Green) → 3. 検証(Refactor/Test)」** の順序を厳格に守って進めること。

### Phase 1: 開発基盤 & プロジェクトスケルトン (TDD環境確立)
- [x] 【Red】`crates/cloth_core` の基本テスト（GPU Context初期化・Device/Queue取得）を作成。
- [x] 【Green】`Cargo.toml` (Workspace), `cloth_core`, `cloth_py`, `pyproject.toml` をセットアップしテストをパスさせる。
- [x] 【Refactor】`maturin develop` を実行し、Pythonから `taremin_cloth_core` がインポート可能であることを `tests/test_cloth_core.py` で確認。
- [x] 【Blender】アドオンの基本構成（`__init__.py`, NパネルUI）を配置し、`python run_tests.py` でBlender内アドオン登録テストをパスさせる。

### Phase 2: 単一布メッシュのGPU XPBD最小実装 (距離拘束・自由落下)
- [x] 【Red】グリッドメッシュの頂点落下およびエッジ距離拘束の誤差率を検証するテストを作成（初期状態では失敗）。
- [x] 【Green】WGSL シェーダー（`predict.wgsl`, `distance.wgsl`, `update_vel.wgsl`）とRustバッファ管理を実装。
- [x] 【Refactor】固定ピン（端点頂点の `inv_mass = 0.0`）を指定し、重力で垂れ下がる布の挙動を実装。
- [x] 【Verify】`cargo test` および `python run_tests.py` でエッジ長誤差が許容値（5%以内）に収まることを検証。

### Phase 3: 曲げ拘束 (Bending) & Blenderビューポート連携
- [x] 【Red】隣接三角形ペアの曲げ角度と曲げ剛性を評価するユニットテストを作成。
- [x] 【Green】CPU側の共有稜線探索（Winged-edge構造）と `bending.wgsl` を実装。
- [x] 【Blender】タイムライン再生ハンドラーを実装し、Blenderの3Dビューポート上で布がシワを保ちながら揺れる描画を確認。
- [x] 【Verify】曲げ剛性パラメータの変化により布の硬さが変わることを自動テストで検証。

### Phase 4: ピン留め（Pinning） & インタラクティブ操作 (Grab / Pin)
- [x] 【Red】動的ピン追加・移動・解放のAPIテストを作成（指定頂点が目標座標に追従することを検証）。
- [x] 【Green】`pin.wgsl` および Rust側の `set_pin_target` / `release_pin` APIを実装。
- [x] 【Blender】頂点グループからのピン留めウェイト読み込み、および `taremin.cloth_interactive` モーダルオペレーターを実装。
- [x] 【Verify】3Dビューポート上でマウスクリック＆ドラッグによる頂点追従インタラクションが動作することを確認。

### Phase 5: 縫合機能 (Sewing Constraints / Pattern Seaming)
- [x] 【Red】離れた2メッシュの境界エッジペアが時間経過とともに引き合わされ、目標長（0.0）に収縮するテストを作成。
- [x] 【Green】`sewing.wgsl` および可変自然長スプリング拘束を実装。収縮速度 $v_{sew}$ に応じた自然長の動的短縮を処理。
- [x] 【Blender】Blenderメッシュの境界エッジペアを「縫合線」として指定するUI/プロパティを実装。
- [x] 【Verify】型紙（前身頃・後身頃）が自動で引き寄せられ、縫い合わされて服の形状になるシミュレーションテストをパスさせる。

### Phase 6: 剛体コライダーとの衝突 (SDF Primitive & Body Mesh)
- [x] 【Red】球・カプセル・平面コライダーに対して頂点が貫通せず表面に押し戻されることを検証するテストを作成。
- [x] 【Green】SDF衝突計算シェーダー（`collision.wgsl`）を実装し、摩擦力および反発補正を適用。
- [x] 【Blender】シーン内のオブジェクト（人体モデルやSuzanne）をコライダーとして登録するUIを実装。
- [x] 【Verify】コライダーに布が覆い被さり、滑り落ちる挙動の自動テスト。

### Phase 7: マルチレイヤー & 空間ハッシュ (自己衝突 / レイヤー衝突)
- [x] 【Red】2枚の布（インナーとアウター）が互いに突き抜けず、外側レイヤーが内側レイヤーを押し込まないことを検証するテストを作成。
- [x] 【Green】GPU空間ハッシュグリッド（Uniform Grid）およびレイヤー間衝突・自己衝突シェーダーを実装。
- [x] 【Refactor】厚み（Thickness）とレイヤー優先度によるオフセット反発拘束を適用。
- [x] 【Verify】重ね着した2枚の布が破綻せず安定シミュレーションできることをE2Eテストで検証。

### Phase 8: 伸縮グループ（Elastic Bands / 辺の自然長スケーリング機能）
- [x] 【Red】指定エッジの自然長を動的にスケーリングし、XPBDで目標長に収縮・伸長する単体テストを作成。
- [x] 【Green】`ClothMesh` におけるグラフ彩色後のエッジインデックス対応表保持、GPUバッファ動的更新API、Pythonバインディングを実装。
- [x] 【Blender】Editモードでの事前選択（Alt+クリックでの辺ループ等）によるグループ登録UI（`TareminClothElasticGroup`）とリアルタイム平滑化追従を実装。
- [x] 【Drawing】3Dビューポートでのリアルタイムカラーオーバーレイ（青:収縮 / 赤:伸長 / 目標到達時のフェード）を実装。
- [x] 【Verify】単体テスト (`test_cloth_core.py`) および Blender E2E テスト (`test_elastic_groups.py`) でオールグリーンを確認。

---

## 5. 検証手順 (Verification Plan)

### 5.1 自動ユニットテスト (Rust)
```bash
cargo test --all
```
- メッシュ構築、グラフ彩色、トポロジ抽出が正しく動作するか検証。
- 境界値（0長エッジ、孤立頂点、極小メッシュ）でのパニック不在テスト。
- 各拘束プロジェクション関数における浮動小数点不変量（NaN/Inf検知）テスト。

### 5.2 Pythonコア統合テスト
```bash
maturin develop
python tests/test_cloth_core.py
```
- NumPy配列経由で頂点データを渡し、100ステップ回した後の健全性を厳格に自動アサート：
  1. `assert np.all(np.isfinite(positions))`（NaN/Infの完全不在）
  2. `assert max_edge_error < 0.05`（エッジ長誤差が5%以内）
  3. `assert not np.isnan(velocities).any()`（速度バッファの健全性）
  4. リセット操作後の座標一致（ビット完全性）検証。

### 5.3 Blenderヘッドレステスト (`run_tests.py`)
`TareminTextureAtlasGenarator` と共通のテスト基盤（`tools/blender_manager.py`）を利用。
ローカルのBlender（キャッシュ済みLTS版または指定版）を自動解決し、バックグラウンドで `tests/` 配下のunittestを実行する。

```bash
# キャッシュ済み/最新LTS Blenderで全テストを実行
python run_tests.py

# 特定のテストファイルのみ実行
python run_tests.py -t test_simulation_e2e.py

# 特定のBlenderバージョン（例: 4.2, 3.6）を指定して実行（未DL時は公式から自動取得）
python run_tests.py -b 4.2
```
- Blender内でアドオンを動的ロードし、グリッドメッシュに対してClothシミュレーションを実行。
- モーダルオペレーター、頂点グループピン留め、縫合機能、Depsgraph更新がヘッドレス環境で正常に完走することをE2Eで検証。

---

## 6. エラーハンドリング & 注意事項

1. **GPUバックエンドおよびデバイスの選択とフォールバック**:
   - ユーザーはアドオンパネルおよび設定（Preferences）から、計算バックエンド（Auto / DirectX 12 / Vulkan）とGPUデバイス（Discrete GPU / Integrated GPU / マルチGPU等）を選択・動的再初期化可能。
   - デフォルトは `Auto`（WindowsではDirectX 12のdGPU優先、Linux/macOSではVulkan/Metal優先）。
   - 指定されたバックエンド/デバイスの初期化に失敗した場合は、安全なデフォルト（Auto / PRIMARY）へ自動フォールバックし、Python側・UIへ通知。
2. **クラッシュ防止（Out-of-Bounds）**:
   - 頂点バッファ・拘束バッファのインデックスが範囲外にならないよう、メッシュ初期化時に厳密にバリデーション。
3. **BlenderのUndo/Redo**:
   - シミュレーション開始前の頂点位置をキャッシュし、リセット時やキャンセル時に確実に復元できるようにする。
