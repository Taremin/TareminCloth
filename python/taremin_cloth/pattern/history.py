# -*- coding: utf-8 -*-
"""型紙エディタのUndo/Redo履歴管理マネージャー (Blender非依存)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Optional

from .document import Document

DEFAULT_MAX_HISTORY: int = 50


def _compute_fingerprint(json_str: str) -> str:
    """JSON文字列のSHA-256ダイジェストを算出する."""
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


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
        self._active_transaction_label: Optional[str] = None
        self._active_transaction_before_json: Optional[str] = None
        self._active_transaction_before_hash: Optional[str] = None

    def begin_transaction(self, doc: Document, *, label: str = "Edit") -> None:
        """インタラクティブ操作（ドラッグ移動等）の開始時にトランザクションを開く."""
        raw = doc.to_json()
        self._active_transaction_label = label
        self._active_transaction_before_json = raw
        self._active_transaction_before_hash = _compute_fingerprint(raw)

    def commit_transaction(self, doc: Document) -> bool:
        """トランザクションを確定し、変更があれば履歴スタックに記録する.
        
        変更がない場合はスタックに積まず False を返す.
        """
        if self._active_transaction_before_json is None:
            return False

        after_json = doc.to_json()
        after_hash = _compute_fingerprint(after_json)

        label = self._active_transaction_label or "Edit"
        before_json = self._active_transaction_before_json
        before_hash = self._active_transaction_before_hash

        self._active_transaction_label = None
        self._active_transaction_before_json = None
        self._active_transaction_before_hash = None

        if before_hash == after_hash:
            return False

        entry = HistoryEntry(
            description=label,
            before_json=before_json,
            after_json=after_json,
        )
        self.undo_stack.append(entry)
        self.redo_stack.clear()
        if len(self.undo_stack) > self.max_entries:
            self.undo_stack.pop(0)
        return True

    def cancel_transaction(self, doc: Document) -> None:
        """アクティブなトランザクションを破棄し、docを開始時の状態に巻き戻す."""
        if self._active_transaction_before_json is not None:
            restored = Document.from_json(self._active_transaction_before_json)
            self._copy_doc_in_place(restored, doc)
        self._active_transaction_label = None
        self._active_transaction_before_json = None
        self._active_transaction_before_hash = None

    def undo_in_place(self, doc: Document) -> bool:
        """直前の操作を取り消し、渡された doc の内容を復元状態へインプレース更新する."""
        res = self.undo()
        if res is None:
            return False
        restored, _ = res
        self._copy_doc_in_place(restored, doc)
        return True

    def redo_in_place(self, doc: Document) -> bool:
        """直前の操作をやり直し、渡された doc の内容を復元状態へインプレース更新する."""
        res = self.redo()
        if res is None:
            return False
        restored, _ = res
        self._copy_doc_in_place(restored, doc)
        return True

    @staticmethod
    def _copy_doc_in_place(src: Document, dst: Document) -> None:
        dst.version = src.version
        dst.name = src.name
        dst.pieces = src.pieces
        dst.seams = src.seams
        dst.notches = src.notches
        dst.anchors = src.anchors
        dst.placements = src.placements
        dst.reference_image = src.reference_image

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
