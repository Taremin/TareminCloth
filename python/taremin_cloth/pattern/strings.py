# -*- coding: utf-8 -*-
"""型紙エディタのUI表示文字列定義 (Blender非依存)."""

from __future__ import annotations

# ツール表示名
TOOL_NAMES: dict[str, str] = {
    "SELECT": "選択・移動 (Select/Move)",
    "RECTANGLE": "矩形作成 (Rectangle)",
    "POLYGON": "多角形作成 (Polygon)",
    "SEW": "縫合定義 (Sew)",
}

# ツール別操作ガイド
TOOL_GUIDES: dict[str, str] = {
    "SELECT": "クリックで選択 / ドラッグで移動 / Shift+クリックで複数選択 / Xで削除",
    "RECTANGLE": "ドラッグして対角を指定 / Escでキャンセル",
    "POLYGON": "クリックで頂点追加 / 開始点クリックまたはEnterで閉じる / Escでキャンセル",
    "SEW": "1つ目の辺をクリックし、続けて2つ目の辺をクリックして縫合",
}
