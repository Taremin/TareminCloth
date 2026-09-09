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
python -m unittest tests/test_mesh_analysis.py
python -m unittest tests/test_mesh_renderer.py
python -m unittest tests/test_replayer_standalone.py
python -m unittest tests/test_debug_recorder.py
python -m unittest tests/test_sdf_baker_gpu.py

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
| `export-obj` | 指定フレームをOBJ形式でエクスポート | `python -m taremin_cloth.log_tools export-obj input.jsonl.gz -f 71 -o f71.obj` |

---

## 6. Blender結合・E2Eテスト運用ガイド (`tools/blender_manager.py`)

Blender API (`bpy`) に依存する統合テスト（オペレーター登録、頂点グループピン設定、縫合E2E、タイムライン再生など）の実行や、特定のBlenderバージョンでの検証には、`run_tests.py` および `tools/blender_manager.py` を使用します。

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
