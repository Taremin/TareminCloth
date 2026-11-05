# -*- coding: utf-8 -*-
"""型紙エディタのコントローラーおよびイベントモデル (Blender非依存)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .document import Document, EdgeRange, Piece
from .history import HistoryManager
from .selection import HitResult, HitTargetKind, hit_test
from .session import SelectionState, SnapSettings, ViewTransform2D, compute_snap
from .tools.base import BaseTool


@dataclass
class EditorEvent:
    """型紙エディタが処理する入力イベント."""
    screen_pos: tuple[float, float] = (0.0, 0.0)
    world_pos: tuple[float, float] = (0.0, 0.0)
    is_press: bool = False
    is_release: bool = False
    is_move: bool = False
    button: str = "LEFT"  # "LEFT", "RIGHT", "MIDDLE"
    shift: bool = False
    ctrl: bool = False
    alt: bool = False
    key: str = ""  # "ESCAPE", "ENTER", "DELETE", "X", "G", "Z", etc.


class EditorController:
    """型紙エディタの全状態を調停するメインコントローラー."""

    def __init__(
        self,
        *,
        doc: Optional[Document] = None,
        view: Optional[ViewTransform2D] = None,
    ) -> None:
        self.doc: Document = doc or Document.new()
        self.history: HistoryManager = HistoryManager()
        self.selection: SelectionState = SelectionState()
        self.view: ViewTransform2D = view or ViewTransform2D()
        self.snap_settings: SnapSettings = SnapSettings()

        self.tools: dict[str, BaseTool] = {}
        self.active_tool_name: str = "SELECT"
        self.is_dirty: bool = False

        # プレビュー用一時幾何（描画系が参照）
        self.preview_points: list[tuple[float, float]] = []
        self.preview_seam_candidates: list[EdgeRange] = []

        # 既定ツールの初期登録
        from .tools import SelectTool, RectangleTool, PolygonTool, SewTool
        self.register_tool("SELECT", SelectTool())
        self.register_tool("RECTANGLE", RectangleTool())
        self.register_tool("POLYGON", PolygonTool())
        self.register_tool("SEW", SewTool())

    def register_tool(self, name: str, tool: BaseTool) -> None:
        """ツールを登録する."""
        self.tools[name] = tool

    def set_active_tool(self, tool_name: str) -> None:
        """アクティブなツールを切り替える."""
        if tool_name not in self.tools:
            return
        if self.active_tool_name != tool_name:
            curr_tool = self.tools.get(self.active_tool_name)
            if curr_tool:
                curr_tool.cancel(self)
            self.active_tool_name = tool_name
            self.preview_points.clear()
            self.preview_seam_candidates.clear()
            self.is_dirty = True

    @property
    def active_tool(self) -> Optional[BaseTool]:
        """現在アクティブなツールインスタンスを返す."""
        return self.tools.get(self.active_tool_name)

    def handle_event(self, event: EditorEvent) -> bool:
        """入力イベントをアクティブツールまたはグローバルハンドラーへ配送する."""
        # ワールド座標を再計算
        event.world_pos = self.view.screen_to_world(event.screen_pos)

        # グローバルショートカット: Undo (Ctrl+Z) / Redo (Ctrl+Shift+Z または Ctrl+Y)
        if event.ctrl and event.key.upper() == "Z":
            if event.shift:
                self.redo()
            else:
                self.undo()
            return True

        if event.ctrl and event.key.upper() == "Y":
            self.redo()
            return True

        # グローバルショートカット: 選択削除 (X または Delete)
        if event.key.upper() in ("X", "DELETE"):
            if self.delete_selected():
                return True

        tool = self.active_tool
        if not tool:
            return False

        consumed = False
        if event.is_press:
            consumed = tool.on_mouse_down(self, event)
        elif event.is_move:
            consumed = tool.on_mouse_move(self, event)
        elif event.is_release:
            consumed = tool.on_mouse_up(self, event)
        elif event.key:
            consumed = tool.on_key_press(self, event)

        if consumed:
            self.is_dirty = True
        return consumed

    def delete_selected(self) -> bool:
        """現在選択中の要素（ノードまたはピース）を削除し履歴に記録する."""
        if not self.selection.piece_ids and not self.selection.node_ids:
            return False

        before_doc = Document.from_json(self.doc.to_json())
        modified = False

        # ピース削除
        for pid in list(self.selection.piece_ids):
            if pid in self.doc.pieces:
                del self.doc.pieces[pid]
                if pid in self.doc.placements:
                    del self.doc.placements[pid]
                # 関連する縫合・ノッチ・アンカーも削除
                for sid in [sid for sid, s in self.doc.seams.items() if s.a.piece_id == pid or s.b.piece_id == pid]:
                    del self.doc.seams[sid]
                for nid in [nid for nid, n in self.doc.notches.items() if n.piece_id == pid]:
                    del self.doc.notches[nid]
                for aid in [aid for aid, a in self.doc.anchors.items() if a.piece_id == pid]:
                    del self.doc.anchors[aid]
                modified = True

        # ノード削除
        for nid in list(self.selection.node_ids):
            for piece in self.doc.pieces.values():
                for path in piece.paths.values():
                    orig_len = len(path.nodes)
                    path.nodes = [n for n in path.nodes if n.id != nid]
                    if len(path.nodes) < orig_len:
                        modified = True

        if modified:
            self.selection.clear()
            self.history.record_change(
                description="Delete Selection",
                before_doc=before_doc,
                after_doc=self.doc,
            )
            self.is_dirty = True
            return True

        return False

    def undo(self) -> bool:
        """直前の操作を取り消す."""
        res = self.history.undo()
        if res:
            self.doc, _ = res
            self.selection.clear()
            self.is_dirty = True
            return True
        return False

    def redo(self) -> bool:
        """取り消した操作をやり直す."""
        res = self.history.redo()
        if res:
            self.doc, _ = res
            self.selection.clear()
            self.is_dirty = True
            return True
        return False
