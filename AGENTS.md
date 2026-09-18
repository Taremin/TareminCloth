# AGENTS.md: AIアシスタント開発・デバッグ運用ガイドライン

このドキュメントは、taremin_clothプロジェクトにおいてAIエージェントおよび開発者が従うべき、テスト・デバッグ・再現・可視化の運用ルールと推奨ワークフローを定めたものです。

---

## 1. プロジェクト基本構成
- **Rust Core (`crates/cloth_core`)**:
  - wgpu (WGSL) ベースのXPBD物理シミュレーションコアおよび `.jsonl.gz` デバッグレコーダー。
- **PyO3 バインディング (`crates/cloth_py` -> `taremin_cloth_core`)**:
  - Python拡張モジュール（C-extension）。
- **Python アドオン & ツール層 (`python/taremin_cloth`)**:
  - BlenderアドオンUI/オペレーター、およびBlender非依存の解析・レンダラー・CLIツール群。

---

## 2. デバッグ・再現の鉄則（重要）

> [!IMPORTANT]
> **「ログの確認・再現・可視化にBlender（`bpy`）を起動しない」**
> - Blenderの起動は数秒〜十数秒の遅延とプロセス管理のオーバーヘッドを生みます。
> - 問題の再現、サブステップ解析、貫通確認レンダリング、テストケース作成はすべて **CLIツール (`python -m taremin_cloth.log_tools`)** および **Pythonテストコード** 上で完結させてください。
>
> **「Blenderシミュレーション検証時は `sim.step()` を直接呼ばず、共通関数 `step_cloth_scene` を使用する」**
> - `sim.step()` のみを単独で呼ぶと、UI実行時に毎フレーム実行されるパラメータ同期（`sync_cloth_parameters`）やコライダー同期（`sync_colliders`）がバイパスされ、アドオンUI実行時の挙動（パラメータ再設定の副作用など）を検知できません。
> - テストや再現スクリプトを作成する際は、必ず `from taremin_cloth import step_cloth_scene` を呼び出し、本番と完全に同一のライフサイクルで検証してください。

---

## 3. 基本ビルド＆テストコマンド

```powershell
# 1. Rustコアの単体テスト & WGSL構造体アライメント自動検証テスト
cargo test

# 2. PyO3 モジュールのビルドと配置
cargo build --release -p taremin_cloth_core
Copy-Item "target/release/taremin_cloth_core.dll" "taremin_cloth_core.pyd" -Force
Copy-Item "target/release/taremin_cloth_core.dll" "python/taremin_cloth/taremin_cloth_core.pyd" -Force

# 3. Python側の主要単体・統合テストの実行（高速スタンドアロン・Blender不要）
# 【推奨】GitHub Actions CIと同一の最小隔離環境（.venv_ci）で自動実行
python run_tests.py --ci
# 特定のテストモジュールだけをピンポイントで試行錯誤
python run_tests.py --ci -t test_quick_pinning.py

# または現在の環境上で直接実行
python -m unittest tests/core/test_mesh_analysis.py
python -m unittest tests/core/test_mesh_renderer.py
python -m unittest tests/core/test_replayer_standalone.py
python -m unittest tests/core/test_debug_recorder.py
python -m unittest tests/core/test_sdf_baker_gpu.py
# または core ディレクトリ配下を一括実行
python -m unittest discover -s tests/core -t .

# 4. Blenderアドオン結合・E2Eテストの実行（tools/blender_manager による自動解決）
python run_tests.py --test test_simulation_e2e.py
```


---

## 4. 推奨デバッグワークフロー（問題発生時）

シミュレーション中の破綻（貫通、裏返り、速度発散など）が発生した場合、以下の手順で最小テストケースを即座に構築してアルゴリズム修正に臨んでください。

### Step 1: ログの異常スキャン
```bash
python -m taremin_cloth.log_tools inspect path/to/cloth_debug.jsonl.gz
```
- 変位スパイクや急加速、NaNが起きたフレーム番号（例: Frame 71）が特定されます。

### Step 2: 貫通可視化レンダリング（0.1秒）
```bash
python -m taremin_cloth.log_tools render path/to/cloth_debug.jsonl.gz --frame 71 --output scratch/f71.png
```
- 表面（白）と裏面（赤）が Unlit で描画され、赤ピクセル（裏返り・貫通箇所）の個数が出力されます。

### Step 3: 最小限の再現テストスクリプトを自動生成
```bash
python -m taremin_cloth.log_tools make-test path/to/cloth_debug.jsonl.gz --frame 71 --output tests/test_issue_f71.py
```
- 直前数フレーム（Frame 68〜71）だけを切り出した極小データと、自己完結型の `unittest.TestCase` が自動作成されます。

### Step 4: 高速TDDループ（Blender不要）
```bash
# 生成されたテストを実行（ミリ秒〜数秒で完了）
python -m unittest tests/test_issue_f71.py
```
- Rust側のWGSLシェーダーや拘束解決コードを修正し、テストが PASS するまで素早くイテレーションを回します。

### Step 5: サブステップ顕微鏡解析（必要に応じて）
- `ClothReplayer.trace_substeps(frame_idx)` を呼び出し、該当フレーム内のサブステップ 1〜10 の最大変位・最大速度推移を1ステップ刻みで確認します。

---

## 5. CLIツール (`log_tools`) コマンドリファレンス

| コマンド | 説明 | 例 |
|---|---|---|
| `inspect` | ログ全体の異常値を走査してサマリー表示 | `python -m taremin_cloth.log_tools inspect input.jsonl.gz` |
| `slice` | 指定フレーム区間を切り出して新しい極小ログを作成 | `python -m taremin_cloth.log_tools slice input.jsonl.gz --start 68 --end 74 -o sliced.jsonl.gz` |
| `make-test` | 自己完結型 unittest テストスクリプトを自動生成 | `python -m taremin_cloth.log_tools make-test input.jsonl.gz -f 71 -o tests/test_f71.py` |
| `render` | 指定フレームまたは連番・APNG動画をレンダリング（縫合線・コライダー自動表示、速度・歪み・法線ヒートマップ対応、Blender不要） | `python -m taremin_cloth.log_tools render input.jsonl.gz -f 71 -o f71.png`<br>`python -m taremin_cloth.log_tools render input.jsonl.gz --frames all --format apng -o anim.png`<br>`python -m taremin_cloth.log_tools render input.jsonl.gz -f 20 --color-by velocity -o vel.png`<br>`python -m taremin_cloth.log_tools render input.jsonl.gz -f 40 --color-by strain -o strain.png` |
| `render-coloring` | 制約グラフ彩色（Welsh-Powell法）を可視化レンダリング | `python -m taremin_cloth.log_tools render-coloring input.jsonl.gz --type distance -o coloring.png` |
| `check-intersections` | 指定フレームの自己交差三角形ペアを検出 | `python -m taremin_cloth.log_tools check-intersections input.jsonl.gz -f 71` |
| `diff` | 2つのシミュレーションログ間で頂点位置の差分・誤差を比較 | `python -m taremin_cloth.log_tools diff log1.jsonl.gz log2.jsonl.gz --tolerance 1.0` |
| `export-obj` | 指定フレームをOBJ形式でエクスポート | `python -m taremin_cloth.log_tools export-obj input.jsonl.gz -f 71 -o f71.obj` |

---

## 6. Blender結合・E2Eテスト運用ガイド (`tools/blender_manager.py`)

Blender API (`bpy`) に依存する統合テスト（オペレーター登録、頂点グループピン設定、縫合E2E、タイムライン再生など）の実行や、特定のBlenderバージョンでの検証には、`run_tests.py` および `tools/blender_manager.py` を使用します。

### .blend ファイルのヘッドレス構造解析 (`tools/inspect_blend.py`)
ユーザーから提供された `.blend` ファイルを開かずに、布・コライダー・ボーン構成・頂点数・面数をCLIで瞬時に把握できます：
```bash
python tools/inspect_blend.py path/to/model.blend
```

### Blenderバージョンの自動解決とテスト実行


```bash
# 利用可能なBlender一覧とキャッシュ容量を確認
python run_tests.py --list-blenders

# Blender依存のE2Eテストを実行（未インストールの場合は公式最新LTSを自動DL）
python run_tests.py --test test_simulation_e2e.py

# 特定のBlenderバージョン（例: 5.2 / 4.2 / 3.6）を指定して実行
python run_tests.py --blender 5.2 --test test_simulation_e2e.py
```

### Pythonスクリプトから Blender パスを動的解決する場合

```python
from tools.blender_manager import resolve_blender

# 最新LTS（または指定バージョン）の blender.exe パスを取得（必要に応じて自動DL）
blender_exe = resolve_blender("5.2")  # または resolve_blender("latest-lts")
```
> [!TIP]
> テストスクリプトや検証ツール内で `C:\Blender\...` などのパスをハードコードせず、必ず `tools.blender_manager.resolve_blender()` を利用して実行環境に依存しないパス解決を行ってください。

---

## 7. アーキテクチャ仕様書の同期運用 (`docs/architecture.md`)

プロジェクトの物理アルゴリズム、GPUデータ構造、パイプライン設計は [docs/architecture.md](docs/architecture.md) に体系化されています。

> [!IMPORTANT]
> **「実装変更とアーキテクチャ仕様書の同期」**
> - 新たな拘束（Constraint）の追加、GPUバッファレイアウト（`SimParams`, `GpuVertex` 等）の変更、またはパイプライン設計の改修を行う際は、必ず [docs/architecture.md](docs/architecture.md) の記述を最新の実装に合わせて更新・同期してください。
> - 仕様とコードが乖離することを防ぎ、将来のAIエージェントおよび開発者が常に最新の設計意図を正確に把握できるように保守します。

---

## 8. アルゴリズム設計仕様書の参照と却下済み判断の再提案防止 (`docs/algorithms.md`)

Taremin Cloth の詳細な物理計算理論、接触・衝突判定（V-T, E-E, CCD）、Coupled XPBDの協調収束設計、および学術参考文献（一次資料リンク付き）は [docs/algorithms.md](docs/algorithms.md) に集約されています。

すべてのAIエージェントおよび開発者は、アルゴリズムの提案や不具合修正に臨む前に必ず [docs/algorithms.md](docs/algorithms.md) を確認し、以下の2原則を厳守してください：

### 原則 1: 過去に却下・破綻が確認されたアプローチの再提案禁止 (Anti-Patterns)
過去の議論および実機検証において不具合や破綻を招くことが判明した以下のアプローチは、**いかなる場合も繰り返し提案・再実装してはなりません**（詳細は [docs/algorithms.md 第4章](docs/algorithms.md#4-却下された判断アンチパターン一覧-rejected-approaches--anti-patterns) を参照）：
1. ❌ **面法線（表側）方向への一律押し戻し**: 自己衝突での裏返り・多重折り畳み時に食い込みを加速させるため禁止（幾何学最短距離ベクトルおよびトポロジーCCDを使用すること）。
2. ❌ **V-T / E-E 作用・反作用分配における相手頂点への強制無同期書き込み**: GPU並列競合によりメッシュ崩壊（くしゃくしゃ化）を引き起こすため禁止（頂点スレッド駆動の片側ペナルティとピン相対速度考慮に留めること）。
3. ❌ **コライダー貫通時の法線方向直立リカバリー**: 球体突き上げ時に布が風船のように球状に膨らむエンバグを招いたため原則禁止。
4. ❌ **CCD（連続衝突判定）変位の安易な間引き (`relief_factor` 等)**: 貫通解消が中途半端になり脱出不能になるため禁止（CCD変位は100%適用すること）。
5. ❌ **形状変形モディファイアの無差別バイパス**: ラティス等の形状変更を無視するとコリジョン形状と不整合が生じるため禁止。

### 原則 2: 不具合修正時の「本来の理想とするアルゴリズムからの乖離」チェック
眼前のバグや貫通を応急処置（ホットフィックス）するあまり、アドホックな値調整や打ち消し処理によって**本来の理想とする物理アルゴリズムから実装が離れていないか**を常にチェックしてください（詳細は [docs/algorithms.md 第5章](docs/algorithms.md#5-本来の理想とする物理アルゴリズムと現在実装との乖離チェック-gap-analysis--compromises) を参照）：
- [ ] **物理保存則**: エネルギー保存や運動量保存、対称性が崩れていないか？
- [ ] **幾何学的整合性**: 物理的な力や最短幾何ベクトルではなく、恣意的なマジックナンバーや非物理的な位置テレポートで誤魔化していないか？
- [ ] **Coupled収束ループの調和**: 拘束解決ループの内外の配置（Coupled XPBD）において、エッジ過剰伸長や不自然な伸びを招く構造になっていないか？
- [ ] **妥協点の明文化**: パフォーマンス制約等でやむを得ず理想と乖離した妥協実装を行う場合は、必ず [docs/algorithms.md](docs/algorithms.md) の「乖離分析」セクションに理由・トレードオフ・将来の改善策を記録すること。

---

## 9. 共通ユーティリティの利用義務と車輪の再発明禁止 (Code Reuse & Anti-Duplication)

> [!IMPORTANT]
> **「メッシュ抽出・評価メッシュ取得・ビュー制御をインラインで手動直書きしない」**
> - Blender API (`foreach_get` や `calc_loop_triangles`、画面再描画ループ等) を使った定型処理を各オペレーターやエンジン内にインラインでコピペ実装することを禁止します。
> - 必ず `taremin_cloth.utils` または共通モジュールに定義された関数を利用し、ロジックの一元性を保ってください。

### 共通ユーティリティ利用基準

| 用途 | ❌ 禁止アプローチ (直書き・重複) | ⭕ 必須アプローチ (共通ユーティリティ) |
|---|---|---|
| **布メッシュ全データ抽出** | 各所で `foreach_get("co")`、`calc_loop_triangles()`、エッジ分類、ピンウェイトを手動ループ | `from taremin_cloth.utils.mesh_extract import extract_cloth_mesh_data` |
| **汎用メッシュ頂点・面抽出** | 各所で `foreach_get("co")` や `loop_triangles.foreach_get` を直書き | `from taremin_cloth.utils.mesh_extract import extract_mesh_vertices_and_triangles` |
| **ピンウェイト・質量計算** | `vertex_groups.get()` と頂点ループでインバースマスを手動計算 | `from taremin_cloth.utils.mesh_extract import get_pin_inv_masses` |
| **3Dビュー再描画** | `for area in context.screen.areas: area.tag_redraw()` を手動ループ | `from taremin_cloth.utils.view3d import tag_redraw_view3d` |
| **アニメーション停止** | `bpy.ops.screen.animation_cancel()` を try-except で手動呼び出し | `from taremin_cloth.utils.view3d import stop_animation` |

### 実装前チェックリスト (Pre-Implementation Check)
コードを追加・修正する前に、すべてのAIエージェントおよび開発者は必ず以下の点検を行ってください：
1. 「これから書こうとしている処理（メッシュデータ取得・幾何変換・Blender操作など）は、すでに `taremin_cloth.utils` や既存モジュールに存在しないか？」を `grep_search` 等で確認すること。
2. 複数箇所で同一または類似の処理が必要になった場合、インラインで重複実装せず、必ず `utils/` に共通関数として定義・分離すること。

---

## 10. リリース運用ガイドライン (`tools/release.py`)

本プロジェクトでは、バージョン更新・事前テスト・Gitコミット・タグ付け・GitHub Actions自動リリース連携をワンストップで行うCLIツール `tools/release.py` を配備しています。

### 基本コマンド例

```bash
# 対話型メニューで次のバージョンを選択して実行
python tools/release.py

# パッチリリース (例: 0.0.1 -> 0.0.2)
python tools/release.py patch

# マイナーリリース (例: 0.0.1 -> 0.1.0)
python tools/release.py minor

# メジャーリリース (例: 0.0.1 -> 1.0.0)
python tools/release.py major

# メジャーのベータ / RC (例: 0.0.1 -> 1.0.0-beta.1 / 1.0.0-rc.1)
python tools/release.py major-beta
python tools/release.py major-rc

# マイナーのベータ / RC (例: 0.0.1 -> 0.1.0-beta.1 / 0.1.0-rc.1)
python tools/release.py minor-beta
python tools/release.py minor-rc

# プレリリース番号のインクリメント (例: 1.0.0-beta.1 -> 1.0.0-beta.2)
python tools/release.py next

# プレリリースから本番版への昇格 (例: 1.0.0-rc.1 -> 1.0.0)
python tools/release.py release

# 任意バージョン直接指定
python tools/release.py 0.0.1
```

### 主要オプション

| オプション | 説明 |
|---|---|
| `--dry-run` | ファイル変更やGit操作を行わず、動作内容をシミュレーション表示 |
| `--check` | 各ファイル（`__init__.py`, `pyproject.toml`, 各 `Cargo.toml`）のバージョン整合性チェックのみ実行 |
| `--skip-tests` | 時間短縮のため品質検証テスト（Rust/Python/Links）をスキップ |
| `--no-push` | Gitリモートプッシュを行わず、ローカルのコミット・タグ作成のみで終了 |
| `-y`, `--yes` | 確認プロンプトをすべて自動承認（CI/自動化向け） |

### リリース自動化の仕組み
1. `tools/release.py` により `__init__.py`、`pyproject.toml`、`Cargo.toml`（3クレート）、`Cargo.lock` が同期更新され、テスト通過後に `chore(release): vX.Y.Z` コミットと `vX.Y.Z` アノテーションタグが作成されます。
2. リモートへタグがプッシュされると、GitHub Actions の `release.yml` が自動起動し、Windows (x64)、Linux (x64)、macOS (Universal) のバイナリが並列ビルドされ、GitHub Releases に各プラットフォーム向けアドオンzipが自動添付・公開されます（`beta` や `rc` が含まれるタグは自動的に Pre-release として公開されます）。

