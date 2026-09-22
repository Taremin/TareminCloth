# HUD表示指針

> `python/taremin_cloth/utils/drawing.py` を唯一の描画集約点とする。
> 実装変更時は本書と `docs/architecture.md:1.4` を同期すること。

## 1. 目的・適用範囲
- 3Dビューポートの `POST_PIXEL` HUD（2D）と `POST_VIEW` オーバーレイ（3D）の表示内容・描画方法を統一する。
- 対象コード: `python/taremin_cloth/utils/drawing.py`（唯一の描画集約点）。

## 2. 基本原則
1. HUDは一過性の実行状態のみ表示する。永続情報・設定値はNパネルに置く。
2. 描画は共通ヘルパー経由のみ。背景クアッド・`blf` 定型処理の直書き禁止。
3. 表示文字列は `i18n.trans()` 必須。
4. 見た目の既定値を変えずに共通化する（後方互換維持）。

## 3. HudTheme（予定値・現行踏襲）
| 項目 | 値 | 由来 |
|---|---|---|
| 背景（上部バッジ/ガイド） | `(0.12, 0.12, 0.14, 0.75)` | `drawing.py` `_draw_top_badge`/`_draw_bottom_badge_2row` |
| 背景（下部進捗） | `(0.10, 0.10, 0.13, 0.85)` | `drawing.py` `_draw_bottom_center_progress` |
| 進捗バー | `(0.2, 0.7, 1.0, 0.9)` 高さ3px | `drawing.py` `_draw_bottom_center_progress` |
| text通常 | `(0.92,0.92,0.95,0.95)` / 進捗 `(0.95,0.95,0.98,1.0)` | `drawing.py` `_draw_bottom_badge_2row`ほか |
| FPS色 | >=50緑`(0.3,0.95,0.4)`、>=30黄、>0赤、0灰 | `drawing.py` `draw_callback_2d` |
| 縫合中/通常 | シアン`(0.35,0.85,1.0)` / 緑 | `drawing.py` `draw_callback_2d` |
| font | 上部13 / 下部12、`box_h=28`（2行時50）、pad 12-18 | `drawing.py` 共通ヘルパー |

## 4. 描画基盤（API草案）
- `_draw_top_badge(region, shader, x, y_top, text, color)` — FPS/縫合フェーズ用。FPSの固定幅も `min_w` で吸収する。
- `_draw_bottom_badge_2row(region, shader, line1, line2)` — Guidance用。下部中央・最大2行。1行連結の肥大化を防ぐ。
- `_draw_bottom_center_progress(region, shader, text, pct)` — ベイク/起動準備用。下部中央配置に統一。
- `_setup_font(size)` / `_measure_text(text)` — `blf.size/dimensions` のバージョン差分岐を集約。
- 禁止: `draw_callback_2d` 内での背景頂点列・`blend_set` 直書きの新設。`ops/interactive.py` でのブラシ分岐直書きも禁止（5章の供給I/F経由のみ）。

## 5. 表示分類
| 分類 | 内容 | 供給元 | 表示条件 |
|---|---|---|---|
| Status | FPS、縫合フェーズ | `ops/interactive.py` + `engine/sewing_priority` | interactive中、`show_fps_overlay` |
| Guidance | 1段目=コンテキストキー、2段目=ツール状態 | 各ブラシの `hud_info()`（下記） | interactive中、`show_hud_help` |
| Progress | ベイク/起動準備+Esc | `ops/bake.py` / `ops/interactive._update_init_progress` | 対応処理中のみ |
| Spatial | ピン/グラブ/縫合辺/Elastic/ブラシ円 | `brush/*`、`drawing.draw_callback_3d` | interactive中、個別トグルに従う |

### Guidance ブラシ表示仕様（UX起点・現状追従しない）
- 背景: 現状の1行連結（固定キー + `tool_text`）はブラシ増加で破綻するため、2層表示とする。
  - line1（操作）: 例 `[LMB]drag [E]switch [F]radius [Shift+F]strength`
  - line2（状態）: 例 `Tool: Range Grab r=50mm s=0.80`
- 各ブラシが表示内容を提供する（`ops/interactive.py` での分岐直書き禁止）:
  - `BaseBrushTool.hud_info(brush_settings, radial) -> {"tool", "label_key", "radius", "strength", "falloff", "show_falloff", "adjusting"}`
  - 集約窓口: `brush.get_brush_hud_info(brush_settings, radial)`（`TOOLS` registry経由）
  - 新規ブラシはクラス実装 + `brush/__init__.py:TOOLS` 登録のみで表示拡張できること。
- パラメータ表示規則:
  - GRAB: 半径なし。`[E]` 切替ヒントのみ。
  - RANGE_GRAB/SMOOTH: `r=NNmm s=0.80` を表示。`falloff` は値が既定外のときのみ付記。
  - `RadialController.active` 時は調整中パラメータを強調（例 `r=50mm <調整中>`）。数値円（Spatial）が主、テキストは補助。
- 役割分担: Nパネルが編集、HUDはライブ値の鏡。HUDにスライダー的詳細は持たせない。

## 6. 配置
- 上部: FPSアンカー5択（既定TOP_CENTER）→縫合は直下スタック。
- 下部中央: GuidanceとProgressは排他（同時表示しない）。
- 3Dは `overlay_depth_test` 既定OFF、`apply_view_depth_bias` 併用は維持。
