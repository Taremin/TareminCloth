# -*- coding: utf-8 -*-
"""型紙エディタのコントローラーおよびイベントモデル (Blender非依存)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .document import Document, EdgeRange, PathRole
from .history import HistoryManager
from .session import SelectionState, SnapSettings, ViewTransform2D
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
    unicode: str = ""  # 印字可能文字 (例: "1", "2", ".", etc.)


KEY_CHAR_MAP: dict[str, str] = {
    "ZERO": "0",
    "ONE": "1",
    "TWO": "2",
    "THREE": "3",
    "FOUR": "4",
    "FIVE": "5",
    "SIX": "6",
    "SEVEN": "7",
    "EIGHT": "8",
    "NINE": "9",
    "NUMPAD_0": "0",
    "NUMPAD_1": "1",
    "NUMPAD_2": "2",
    "NUMPAD_3": "3",
    "NUMPAD_4": "4",
    "NUMPAD_5": "5",
    "NUMPAD_6": "6",
    "NUMPAD_7": "7",
    "NUMPAD_8": "8",
    "NUMPAD_9": "9",
    "PERIOD": ".",
    "NUMPAD_PERIOD": ".",
    "COMMA": ".",
    "MINUS": "-",
    "NUMPAD_MINUS": "-",
}

NOTCH_CARD_FIELDS: tuple[str, ...] = ("notch_from_start", "notch_ratio")
PIECE_CARD_FIELDS: tuple[str, ...] = (
    "target_edge_length",
    "layer_id",
    "target_bone",
    "bone_t",
    "bone_angle_deg",
    "surface_offset",
    "is_tubular",
)


class EditorController:
    """型紙エディタの全状態を調停するメインコントローラー."""

    @property
    def snap_settings(self) -> SnapSettings:
        """有効なスナップ設定を返す (Alt押下中は一時的に全スナップを無効化した設定を返す)."""
        if self._snap_bypass:
            return SnapSettings(enabled=False)
        return self._snap_settings

    @snap_settings.setter
    def snap_settings(self, value: SnapSettings) -> None:
        self._snap_settings = value

    def __init__(
        self,
        *,
        doc: Optional[Document] = None,
        view: Optional[ViewTransform2D] = None,
    ) -> None:
        self._snap_bypass: bool = False
        self.doc: Document = doc or Document.new()
        self.history: HistoryManager = HistoryManager()
        self.selection: SelectionState = SelectionState()
        self.view: ViewTransform2D = view or ViewTransform2D()
        self.snap_settings = SnapSettings()

        self.tools: dict[str, BaseTool] = {}
        self.active_tool_name: str = "SELECT"
        self.is_dirty: bool = False
        self.show_lengths: bool = True
        self.bones: dict[str, Any] = {}
        self.is_dragging_compass: bool = False
        self.drag_compass_before_doc: Optional[Document] = None

        # ボーン直接選択（Pick Bone）およびスライダー操作管理
        self.picking_bone: bool = False
        self.hovered_bone: str = ""
        self.sliding_field: Optional[str] = None
        self.slide_start_pos: tuple[float, float] = (0.0, 0.0)
        self.slide_start_val: float = 0.0
        self.is_slider_dragging: bool = False
        self.slide_before_doc: Optional[Document] = None

        # 画面サイズおよびHUDイベントフラグ
        self.screen_size: tuple[int, int] = (1280, 720)
        self.left_offset: float = 0.0
        self.right_offset: float = 0.0
        self.top_offset: float = 0.0
        self.bottom_offset: float = 0.0
        self.request_build: bool = False
        self.request_close: bool = False
        self.editing_field: Optional[str] = None
        self.editing_buffer: str = ""

        # プレビュー用一時幾何（描画系が参照）
        self.preview_points: list[tuple[float, float]] = []
        self.preview_seam_candidates: list[EdgeRange] = []

        # 既定ツールの初期登録
        from .tools import (
            SelectTool,
            RectangleTool,
            PolygonTool,
            EllipseTool,
            InternalLineTool,
            NotchTool,
            CutTool,
            DartTool,
            SewTool,
        )
        self.register_tool("SELECT", SelectTool())
        self.register_tool("RECTANGLE", RectangleTool())
        self.register_tool("POLYGON", PolygonTool())
        self.register_tool("ELLIPSE", EllipseTool())
        self.register_tool("INTERNAL_LINE", InternalLineTool())
        self.register_tool("NOTCH", NotchTool())
        self.register_tool("CUT", CutTool())
        self.register_tool("DART", DartTool())
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
        # Alt 押下中はグリッド/ノード吸着を一時無効化 (自由な微調整用)
        self._snap_bypass = bool(event.alt)

        # ワールド座標・スクリーン座標の同期
        if event.screen_pos == (0.0, 0.0) and event.world_pos != (0.0, 0.0):
            event.screen_pos = self.view.world_to_screen(event.world_pos)
        else:
            event.world_pos = self.view.screen_to_world(event.screen_pos)

        # インライン数値入力モード中のキー入力優先処理
        if self.editing_field is not None and (event.key or event.unicode):
            k = event.key.upper()
            if k == "ESCAPE":
                self.editing_field = None
                self.editing_buffer = ""
                self.is_dirty = True
                return True
            if k in ("ENTER", "NUMPAD_ENTER", "RET"):
                self._commit_inline_edit()
                self.editing_field = None
                self.editing_buffer = ""
                self.is_dirty = True
                return True
            if k in ("BACK_SPACE", "BACKSPACE"):
                self.editing_buffer = self.editing_buffer[:-1]
                self.is_dirty = True
                return True
            char = None
            if event.unicode and (event.unicode.isalnum() or event.unicode in ".-"):
                char = event.unicode
            elif k in KEY_CHAR_MAP:
                char = KEY_CHAR_MAP[k]
            elif len(event.key) == 1 and (event.key.isalnum() or event.key in ".-"):
                char = event.key
            elif k.startswith("NUMPAD_") and len(k) == 8 and k[7].isdigit():
                char = k[7]
            elif k == "NUMPAD_PERIOD":
                char = "."
            if char is not None:
                self.editing_buffer += char
                self.is_dirty = True
                return True
            return True

        # Pick Bone モード中の専用イベント処理（クリック確定・X/Delete解除・Esc/右クリック/Bキャンセル）
        if self.picking_bone:
            if event.is_move:
                return True
            if event.is_press:
                if event.button == "LEFT":
                    if self.hovered_bone and self.selection.piece_ids:
                        pid = next(iter(self.selection.piece_ids))
                        pl = self.doc.placements.get(pid)
                        if not pl:
                            from .document import Placement
                            pl = Placement(piece_id=pid)
                            self.doc.placements[pid] = pl
                        before_doc = self.doc.clone()
                        pl.target_bone = self.hovered_bone
                        self.history.record_change(
                            description=f"Pick Target Bone ({self.hovered_bone})",
                            before_doc=before_doc,
                            after_doc=self.doc,
                        )
                    self.picking_bone = False
                    self.hovered_bone = ""
                    self.is_dirty = True
                    return True
                elif event.button == "RIGHT":
                    self.picking_bone = False
                    self.hovered_bone = ""
                    self.is_dirty = True
                    return True
            if event.key:
                k = event.key.upper()
                if k in ("ESCAPE", "ESC", "B"):
                    self.picking_bone = False
                    self.hovered_bone = ""
                    self.is_dirty = True
                    return True
                elif k in ("X", "DELETE"):
                    if self.selection.piece_ids:
                        pid = next(iter(self.selection.piece_ids))
                        pl = self.doc.placements.get(pid)
                        if pl and pl.target_bone:
                            before_doc = self.doc.clone()
                            pl.target_bone = ""
                            self.history.record_change(
                                description="Clear Target Bone",
                                before_doc=before_doc,
                                after_doc=self.doc,
                            )
                    self.picking_bone = False
                    self.hovered_bone = ""
                    self.is_dirty = True
                    return True
            return True

        # スライダー操作中のマウス移動・マウス解放処理
        if self.sliding_field is not None:
            if event.is_move:
                px, _ = event.screen_pos
                dx = px - self.slide_start_pos[0]
                if abs(dx) > 3.0:
                    self.is_slider_dragging = True

                if self.is_slider_dragging and self.selection.piece_ids:
                    first_pid = next(iter(self.selection.piece_ids))
                    pl = self.doc.placements.get(first_pid)
                    if pl:
                        shift_mod = 0.2 if event.shift else 1.0
                        if self.sliding_field == "bone_t":
                            # 200px で 0.0〜1.0 (感度: 0.005)
                            new_val = self.slide_start_val + dx * 0.005 * shift_mod
                            pl.bone_t = max(0.0, min(1.0, new_val))
                        elif self.sliding_field == "bone_angle_deg":
                            # 1px で 1.0度 (感度: 1.0)
                            new_val = self.slide_start_val + dx * 1.0 * shift_mod
                            pl.bone_angle_deg = ((new_val + 180.0) % 360.0) - 180.0
                        elif self.sliding_field == "surface_offset":
                            # 100px で 0.1m = 10cm (感度: 0.001)
                            new_val = self.slide_start_val + dx * 0.001 * shift_mod
                            pl.surface_offset = max(0.001, min(1.0, new_val))
                        self.is_dirty = True
                return True
            elif event.is_release and event.button == "LEFT":
                field = self.sliding_field
                was_dragging = self.is_slider_dragging
                before_doc = self.slide_before_doc

                self.sliding_field = None
                self.is_slider_dragging = False
                self.slide_before_doc = None

                if was_dragging:
                    if before_doc is not None:
                        self.history.record_change(
                            description=f"Slide {field}",
                            before_doc=before_doc,
                            after_doc=self.doc,
                        )
                    self.is_dirty = True
                    return True
                else:
                    # 3px以内の単発クリック -> インライン数値直接入力を開始
                    if self.selection.piece_ids:
                        first_pid = next(iter(self.selection.piece_ids))
                        pl = self.doc.placements.get(first_pid)
                        if pl:
                            self.editing_field = field
                            if field == "bone_t":
                                self.editing_buffer = f"{int(pl.bone_t * 100)}"
                            elif field == "bone_angle_deg":
                                self.editing_buffer = f"{int(pl.bone_angle_deg)}"
                            elif field == "surface_offset":
                                from .units import m_to_cm
                                self.editing_buffer = f"{m_to_cm(pl.surface_offset):.1f}"
                            self.is_dirty = True
                    return True

        # コンパスドラッグ中のマウス移動・マウス解放処理
        if self.is_dragging_compass:
            if event.is_move:
                from .layout import compute_hud_layout
                layout = compute_hud_layout(
                    *self.screen_size,
                    left_offset=self.left_offset,
                    right_offset=self.right_offset,
                    top_offset=self.top_offset,
                    bottom_offset=self.bottom_offset,
                )
                px, py = event.screen_pos
                cx = layout.compass_rect.x + layout.compass_rect.w * 0.5
                cy = layout.compass_rect.y + layout.compass_rect.h * 0.5
                dx = px - cx
                dy = py - cy
                theta_deg = math.degrees(math.atan2(dx, dy))
                if event.shift:
                    theta_deg = round(theta_deg / 15.0) * 15.0
                if self.selection.piece_ids:
                    first_pid = next(iter(self.selection.piece_ids))
                    pl = self.doc.placements.get(first_pid)
                    if pl:
                        pl.bone_angle_deg = theta_deg
                        self.is_dirty = True
                return True
            elif event.is_release and event.button == "LEFT":
                self.is_dragging_compass = False
                if self.drag_compass_before_doc is not None:
                    self.history.record_change(
                        description="Rotate Placement Angle",
                        before_doc=self.drag_compass_before_doc,
                        after_doc=self.doc,
                    )
                    self.drag_compass_before_doc = None
                self.is_dirty = True
                return True

        # マウスクリック時のHUD 5領域（ボタン・レール・カード入力）ヒットテスト
        if event.is_press and event.button == "LEFT":
            from .layout import compute_hud_layout
            layout = compute_hud_layout(
                *self.screen_size,
                left_offset=self.left_offset,
                right_offset=self.right_offset,
                top_offset=self.top_offset,
                bottom_offset=self.bottom_offset,
            )
            px, py = event.screen_pos

            # ビルドボタン
            if layout.build_button.contains(px, py):
                self.request_build = True
                return True

            # 閉じるボタン
            if layout.close_button.contains(px, py):
                self.request_close = True
                return True

            # ツールレールボタン
            for tool_name, btn_rect in layout.tool_buttons.items():
                if btn_rect.contains(px, py):
                    self.set_active_tool(tool_name)
                    return True

            is_piece_card_mode = (
                not self.selection.notch_ids
                and not self.selection.seam_ids
                and len(self.selection.piece_ids) == 1
            )

            # コンパス領域クリック判定
            if is_piece_card_mode and layout.compass_rect.contains(px, py):
                self.is_dragging_compass = True
                self.drag_compass_before_doc = self.doc.clone()
                cx = layout.compass_rect.x + layout.compass_rect.w * 0.5
                cy = layout.compass_rect.y + layout.compass_rect.h * 0.5
                dx = px - cx
                dy = py - cy
                theta_deg = math.degrees(math.atan2(dx, dy))
                if event.shift:
                    theta_deg = round(theta_deg / 15.0) * 15.0
                first_pid = next(iter(self.selection.piece_ids))
                pl = self.doc.placements.get(first_pid)
                if not pl:
                    from .document import Placement
                    pl = Placement(piece_id=first_pid)
                    self.doc.placements[first_pid] = pl
                pl.bone_angle_deg = theta_deg
                self.is_dirty = True
                return True

            # 情報カード内のインライン編集可能フィールド
            if self.selection.notch_ids and layout.info_card.contains(px, py):
                nid = next(iter(self.selection.notch_ids))
                notch = self.doc.notches.get(nid)
                if notch:
                    for field_name in NOTCH_CARD_FIELDS:
                        f_rect = layout.card_fields.get(field_name)
                        if f_rect and f_rect.contains(px, py):
                            self.editing_field = field_name
                            segs = self.doc.evaluate_path_segments(notch.piece_id, notch.path_id)
                            target_seg = next((tbl for s_id, _, tbl in segs if s_id == notch.segment_start_id), None)
                            if field_name == "notch_from_start":
                                from .geometry import arc_length_at_t
                                from .units import m_to_cm
                                s_cur = arc_length_at_t(target_seg, notch.t) if target_seg else 0.0
                                self.editing_buffer = f"{m_to_cm(s_cur):.1f}"
                            elif field_name == "notch_ratio":
                                self.editing_buffer = f"{int(round(notch.t * 100))}"
                            self.is_dirty = True
                            return True

            elif is_piece_card_mode and layout.info_card.contains(px, py):
                first_pid = next(iter(self.selection.piece_ids))
                p = self.doc.pieces.get(first_pid)
                if p:
                    pl = self.doc.placements.get(first_pid)
                    if not pl:
                        from .document import Placement
                        pl = Placement(piece_id=first_pid)
                        self.doc.placements[first_pid] = pl

                    for field_name in PIECE_CARD_FIELDS:
                        f_rect = layout.card_fields.get(field_name)
                        if not f_rect or not f_rect.contains(px, py):
                            continue
                        if field_name == "target_bone":
                            # ボーン直接選択（Pick Bone）モードを開始
                            self.picking_bone = True
                            self.is_dirty = True
                            return True

                        elif field_name == "is_tubular":
                            before_doc = self.doc.clone()
                            pl.is_tubular = not pl.is_tubular
                            self.history.record_change(
                                description="Toggle Tubular Seam",
                                before_doc=before_doc,
                                after_doc=self.doc,
                            )
                            self.is_dirty = True
                            return True

                        elif field_name in ("bone_t", "bone_angle_deg", "surface_offset"):
                            # スライダー操作の準備（ドラッグでスライド調整、3px以内クリックでインライン直接入力）
                            self.sliding_field = field_name
                            self.slide_start_pos = event.screen_pos
                            self.is_slider_dragging = False
                            self.slide_before_doc = self.doc.clone()
                            if field_name == "bone_t":
                                self.slide_start_val = pl.bone_t
                            elif field_name == "bone_angle_deg":
                                self.slide_start_val = pl.bone_angle_deg
                            elif field_name == "surface_offset":
                                self.slide_start_val = pl.surface_offset
                            return True

                        self.editing_field = field_name
                        if field_name == "target_edge_length":
                            from .units import m_to_cm
                            self.editing_buffer = f"{m_to_cm(p.target_edge_length):.1f}"
                        elif field_name == "layer_id":
                            self.editing_buffer = str(p.layer_id)

                        self.is_dirty = True
                        return True

            # HUD矩形領域内の誤作図防止ガード
            if (
                layout.header_bar.contains(px, py)
                or layout.footer_bar.contains(px, py)
                or layout.left_rail.contains(px, py)
                or layout.info_card.contains(px, py)
            ):
                if self.editing_field is not None:
                    self.editing_field = None
                    self.editing_buffer = ""
                    self.is_dirty = True
                return True
            else:
                if self.editing_field is not None:
                    self.editing_field = None
                    self.editing_buffer = ""
                    self.is_dirty = True

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

        # グローバルショートカット: 縫合反転 (R または F)
        if not event.ctrl and not event.alt and event.key.upper() in ("R", "F"):
            if self.flip_selected_seams():
                return True

        # グローバルショートカット: 寸法線トグル (L)
        if not event.ctrl and not event.alt and event.key.upper() == "L":
            self.show_lengths = not self.show_lengths
            self.is_dirty = True
            return True

        # グローバルショートカット: ボーン選択モードトグル (B)
        if not event.ctrl and not event.alt and event.key.upper() == "B":
            if self.selection.piece_ids:
                self.picking_bone = not self.picking_bone
                self.is_dirty = True
                return True

        tool = self.active_tool
        if not tool:
            return False

        consumed = False
        if event.key:
            consumed = tool.on_key_press(self, event)
        elif event.is_press:
            consumed = tool.on_mouse_down(self, event)
        elif event.is_move:
            consumed = tool.on_mouse_move(self, event)
        elif event.is_release:
            consumed = tool.on_mouse_up(self, event)

        if consumed:
            self.is_dirty = True
        return consumed

    def _commit_inline_edit(self) -> None:
        """インライン入力バッファの内容を選択中ピースまたはノッチへ適用する."""
        if not self.editing_field or not self.editing_buffer:
            return

        if self.selection.notch_ids and self.editing_field in ("notch_from_start", "notch_ratio"):
            nid = next(iter(self.selection.notch_ids))
            notch = self.doc.notches.get(nid)
            if notch:
                segs = self.doc.evaluate_path_segments(notch.piece_id, notch.path_id)
                target_seg = next((tbl for s_id, _, tbl in segs if s_id == notch.segment_start_id), None)
                if target_seg:
                    before_doc = self.doc.clone()
                    if self.editing_field == "notch_from_start":
                        from .units import parse_length_input
                        try:
                            s_m = parse_length_input(self.editing_buffer, default_unit="cm")
                            from .geometry import point_at_arc_length
                            _, target_t = point_at_arc_length(target_seg, s_m)
                            if self.doc.move_notch(nid, target_t):
                                self.history.record_change(
                                    description="Edit Notch Distance",
                                    before_doc=before_doc,
                                    after_doc=self.doc,
                                )
                                self.is_dirty = True
                        except ValueError:
                            pass
                    elif self.editing_field == "notch_ratio":
                        try:
                            clean_buf = self.editing_buffer.replace("%", "").strip()
                            ratio_f = float(clean_buf)
                            if ratio_f > 1.0:
                                ratio_f /= 100.0
                            if self.doc.move_notch(nid, ratio_f):
                                self.history.record_change(
                                    description="Edit Notch Ratio",
                                    before_doc=before_doc,
                                    after_doc=self.doc,
                                )
                                self.is_dirty = True
                        except ValueError:
                            pass
            return

        if not self.selection.piece_ids:
            return
        from .units import parse_length_input
        before_doc = self.doc.clone()
        modified = False
        for pid in self.selection.piece_ids:
            piece = self.doc.pieces.get(pid)
            if not piece:
                continue
            pl = self.doc.placements.get(pid)
            if not pl:
                from .document import Placement
                pl = Placement(piece_id=pid)
                self.doc.placements[pid] = pl

            if self.editing_field == "target_edge_length":
                try:
                    val_m = parse_length_input(self.editing_buffer, default_unit="cm")
                    if 0.001 <= val_m <= 0.5:
                        piece.target_edge_length = val_m
                        modified = True
                except ValueError:
                    pass
            elif self.editing_field == "layer_id":
                try:
                    val_int = int(self.editing_buffer)
                    if -10 <= val_int <= 10:
                        piece.layer_id = val_int
                        modified = True
                except ValueError:
                    pass
            elif self.editing_field == "bone_t":
                try:
                    clean_buf = self.editing_buffer.replace("%", "").strip()
                    val_f = float(clean_buf)
                    if val_f > 1.0:
                        val_f /= 100.0
                    pl.bone_t = max(0.0, min(1.0, val_f))
                    modified = True
                except ValueError:
                    pass
            elif self.editing_field == "bone_angle_deg":
                try:
                    clean_buf = self.editing_buffer.replace("°", "").replace("deg", "").strip()
                    val_f = float(clean_buf)
                    pl.bone_angle_deg = ((val_f + 180.0) % 360.0) - 180.0
                    modified = True
                except ValueError:
                    pass
            elif self.editing_field == "surface_offset":
                try:
                    val_m = parse_length_input(self.editing_buffer, default_unit="cm")
                    if 0.001 <= val_m <= 1.0:
                        pl.surface_offset = val_m
                        modified = True
                except ValueError:
                    pass
        if modified:
            self.history.record_change(
                description=f"Edit {self.editing_field}",
                before_doc=before_doc,
                after_doc=self.doc,
            )
            self.is_dirty = True

    def flip_selected_seams(self) -> bool:
        """選択中の縫合（または直近/選択ピースに関連する縫合）の方向 (reverse) を反転する."""
        target_seam_ids = set(self.selection.seam_ids)
        if not target_seam_ids:
            if self.selection.piece_ids:
                for sid, seam in self.doc.seams.items():
                    if any(
                        er.piece_id in self.selection.piece_ids
                        for er in seam.iter_unique_edge_ranges()
                    ):
                        target_seam_ids.add(sid)
            elif len(self.doc.seams) == 1:
                target_seam_ids.add(next(iter(self.doc.seams.keys())))

        if not target_seam_ids:
            return False

        before_doc = self.doc.clone()
        modified = False
        for sid in target_seam_ids:
            seam = self.doc.seams.get(sid)
            if seam:
                seam.reverse = not seam.reverse
                modified = True

        if modified:
            self.history.record_change(
                description="Flip Seam Direction",
                before_doc=before_doc,
                after_doc=self.doc,
            )
            self.is_dirty = True
            return True
        return False

    def delete_selected(self) -> bool:
        """現在選択中の要素（ノッチ、縫合、ノード、またはピース）を優先順位に従って削除し履歴に記録する.
        
        優先順位:
        1. ノッチが選択されている場合 -> ノッチのみを削除
        2. 縫合が選択されている場合 -> 縫合のみを削除
        3. ノードが選択されている場合 -> ノードのみを削除
        4. 上記がなくピースが選択されている場合 -> ピース全体を削除
        """
        if (
            not self.selection.piece_ids
            and not self.selection.node_ids
            and not self.selection.notch_ids
            and not self.selection.seam_ids
        ):
            return False

        before_doc = self.doc.clone()
        modified = False

        # 1. ノッチ削除（ノッチ選択時はノッチのみを削除し、ピース本体は削除しない）
        if self.selection.notch_ids:
            for nid in list(self.selection.notch_ids):
                if nid in self.doc.notches:
                    del self.doc.notches[nid]
                    modified = True
            self.selection.notch_ids.clear()
            if modified:
                self.history.record_change(
                    description="Delete Notch",
                    before_doc=before_doc,
                    after_doc=self.doc,
                )
                self.is_dirty = True
                return True

        # 2. 縫合削除（縫合選択時は縫合のみを削除）
        if self.selection.seam_ids:
            for sid in list(self.selection.seam_ids):
                if sid in self.doc.seams:
                    del self.doc.seams[sid]
                    modified = True
            self.selection.seam_ids.clear()
            if modified:
                self.history.record_change(
                    description="Delete Seam",
                    before_doc=before_doc,
                    after_doc=self.doc,
                )
                self.is_dirty = True
                return True

        # 3. ノード削除（ノード選択時はノードのみを削除し、ピース本体は削除しない）
        if self.selection.node_ids:
            deleted_node_ids: set[str] = set()
            affected_prev_segments: set[tuple[str, str, str]] = set()
            for nid in list(self.selection.node_ids):
                for piece in self.doc.pieces.values():
                    for path in piece.paths.values():
                        if path.role == PathRole.OUTER and len(path.nodes) <= 3:
                            # 外周多角形が3頂点未満になる削除は拒絶
                            continue
                        n_len = len(path.nodes)
                        for i, n in enumerate(path.nodes):
                            if n.id == nid and (i > 0 or path.closed):
                                prev_n = path.nodes[(i - 1) % n_len]
                                affected_prev_segments.add((piece.id, path.id, prev_n.id))
                        orig_len = len(path.nodes)
                        path.nodes = [n for n in path.nodes if n.id != nid]
                        if len(path.nodes) < orig_len:
                            deleted_node_ids.add(nid)
                            modified = True

            # 削除されたノードおよび終点側だった直前セグメントを参照するノッチ、および縫合を除去
            if deleted_node_ids:
                dead_notches = [
                    nid for nid, n in self.doc.notches.items()
                    if n.segment_start_id in deleted_node_ids
                    or (n.piece_id, n.path_id, n.segment_start_id) in affected_prev_segments
                ]
                for nid in dead_notches:
                    del self.doc.notches[nid]

                dead_seams = [
                    sid for sid, s in self.doc.seams.items()
                    if any(
                        er.start_node_id in deleted_node_ids or er.end_node_id in deleted_node_ids
                        for er in s.iter_unique_edge_ranges()
                    )
                ]
                for sid in dead_seams:
                    del self.doc.seams[sid]

            self.selection.node_ids.clear()
            if modified:
                self.history.record_change(
                    description="Delete Node",
                    before_doc=before_doc,
                    after_doc=self.doc,
                )
                self.is_dirty = True
                return True

        # 4. ピース削除（明示的にピースが選択されている場合のみ）
        if self.selection.piece_ids:
            for pid in list(self.selection.piece_ids):
                if pid in self.doc.pieces:
                    del self.doc.pieces[pid]
                    if pid in self.doc.placements:
                        del self.doc.placements[pid]
                    dead_sids = [
                        sid for sid, s in self.doc.seams.items()
                        if any(er.piece_id == pid for er in s.iter_unique_edge_ranges())
                    ]
                    for sid in dead_sids:
                        del self.doc.seams[sid]
                    for nid in [nid for nid, n in self.doc.notches.items() if n.piece_id == pid]:
                        del self.doc.notches[nid]
                    for aid in [aid for aid, a in self.doc.anchors.items() if a.piece_id == pid]:
                        del self.doc.anchors[aid]
                    modified = True
            self.selection.piece_ids.clear()
            if modified:
                self.history.record_change(
                    description="Delete Piece",
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
