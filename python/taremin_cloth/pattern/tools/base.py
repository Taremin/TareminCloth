# -*- coding: utf-8 -*-
"""型紙エディタのツール基底クラス (Blender非依存)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..editor import EditorController, EditorEvent


class BaseTool(ABC):
    """型紙エディタ上で対話操作を行うツールの抽象基底クラス."""

    name: str = "BaseTool"

    @abstractmethod
    def on_mouse_down(self, controller: EditorController, event: EditorEvent) -> bool:
        """マウスボタン押下時のイベントハンドラー. イベントを消費した場合は True を返す."""
        return False

    @abstractmethod
    def on_mouse_move(self, controller: EditorController, event: EditorEvent) -> bool:
        """マウス移動時のイベントハンドラー. イベントを消費した場合は True を返す."""
        return False

    @abstractmethod
    def on_mouse_up(self, controller: EditorController, event: EditorEvent) -> bool:
        """マウスボタン解放時のイベントハンドラー. イベントを消費した場合は True を返す."""
        return False

    @abstractmethod
    def on_key_press(self, controller: EditorController, event: EditorEvent) -> bool:
        """キーボード押下時のイベントハンドラー. イベントを消費した場合は True を返す."""
        return False

    def cancel(self, controller: EditorController) -> None:
        """現在の操作をキャンセルしてツールを初期状態に復帰させる."""
        pass
