# -*- coding: utf-8 -*-
"""型紙エディタのUndo/Redo履歴管理マネージャー (Blender非依存)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .document import Document

DEFAULT_MAX_HISTORY: int = 50


@dataclass
class HistoryEntry:
    """単一の履歴変更エントリ."""
    description: str
    before_json: str
    after_json: str


class HistoryManager:
    """型紙ドキュメントのUndo/Redoスタックを管理するクラス.
    
    各操作の前後のJSONスナップショットを記録し、最大件数（既定50件）を超えた
    最古の履歴を自動的に切り捨てる.
    """

    def __init__(self, *, max_entries: int = DEFAULT_MAX_HISTORY) -> None:
        self.max_entries: int = max(1, max_entries)
        self.undo_stack: list[HistoryEntry] = []
        self.redo_stack: list[HistoryEntry] = []

    @property
    def can_undo(self) -> bool:
        """Undo可能な履歴が存在するかどうか."""
        return len(self.undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        """Redo可能な履歴が存在するかどうか."""
        return len(self.redo_stack) > 0

    def record_change(
        self,
        *,
        description: str,
        before_doc: Document,
        after_doc: Document,
    ) -> bool:
        """ドキュメントの変更前後の状態を履歴に記録する.
        
        前後の内容が同一の場合は記録をスキップして False を返す.
        記録に成功した場合は True を返す.
        """
        before_json = before_doc.to_json()
        after_json = after_doc.to_json()

        if before_json == after_json:
            return False

        entry = HistoryEntry(
            description=description,
            before_json=before_json,
            after_json=after_json,
        )

        self.undo_stack.append(entry)
        self.redo_stack.clear()

        if len(self.undo_stack) > self.max_entries:
            self.undo_stack.pop(0)

        return True

    def record_raw_json_change(
        self,
        *,
        description: str,
        before_json: str,
        after_json: str,
    ) -> bool:
        """事前シリアライズされたJSON文字列から直接履歴を記録する."""
        if before_json == after_json:
            return False

        entry = HistoryEntry(
            description=description,
            before_json=before_json,
            after_json=after_json,
        )

        self.undo_stack.append(entry)
        self.redo_stack.clear()

        if len(self.undo_stack) > self.max_entries:
            self.undo_stack.pop(0)

        return True

    def undo(self) -> Optional[tuple[Document, str]]:
        """直前の操作を取り消し、復元された Document と操作説明のタプルを返す.
        
        Undoスタックが空の場合は None を返す.
        """
        if not self.can_undo:
            return None

        entry = self.undo_stack.pop()
        self.redo_stack.append(entry)

        doc = Document.from_json(entry.before_json)
        return (doc, entry.description)

    def redo(self) -> Optional[tuple[Document, str]]:
        """取り消した操作をやり直し、復元された Document と操作説明のタプルを返す.
        
        Redoスタックが空の場合は None を返す.
        """
        if not self.can_redo:
            return None

        entry = self.redo_stack.pop()
        self.undo_stack.append(entry)

        doc = Document.from_json(entry.after_json)
        return (doc, entry.description)

    def clear(self) -> None:
        """全履歴をクリアする."""
        self.undo_stack.clear()
        self.redo_stack.clear()
