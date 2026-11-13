# -*- coding: utf-8 -*-
"""型紙エディタのコントローラーおよびイベントモデル (Blender非依存)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from .document import (
    Document,
    EdgeRange,
    Node,
    Path,
    PathRole,
    Placement,
    SeamRouting,
    normalize_angle_deg,
)
from .history import HistoryManager
from .ids import new_id
from .session import SelectionState, SnapSettings, ViewTransform2D
from .tools.base import BaseTool
from .transform_ops import (
    TransformModalSession,
    align_selected_nodes as apply_node_alignment,
    apply_rotation,
    apply_scale,
    apply_translation,
    can_start_modal_transform,
    capture_transform_snapshot,
    finalize_scale_orientation,
    find_nearest_snap_reference,
    restore_snapshot,
)
from .units import m_to_cm, parse_length_input
from ..i18n import trans
from ..utils.logger import logger


@dataclass
class EditorEvent:
    """型紙エディタが処理する入力イベント."""
    screen_pos: tuple[float, float] = (0.0, 0.0)
    world_pos: tuple[float, float] = (0.0, 0.0)
    is_press: bool = False
    is_release: bool = False
    is_move: bool = False
    is_double_click: bool = False
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
    "PLUS": "+",
    "NUMPAD_PLUS": "+",
}


def _extract_event_char(event: EditorEvent, *, allowed_chars: Optional[str] = None) -> Optional[str]:
    """キーボード入力イベントから数値・単位入力用の1文字を抽出する."""
    k = event.key.upper() if event.key else ""
    char: Optional[str] = None
    if event.unicode and (event.unicode.isalnum() or event.unicode in ".-+%"):
        char = event.unicode
    elif k in KEY_CHAR_MAP:
        char = KEY_CHAR_MAP[k]
    elif len(event.key) == 1 and (event.key.isalnum() or event.key in ".-+%"):
        char = event.key
    elif k.startswith("NUMPAD_") and len(k) == 8 and k[7].isdigit():
        char = k[7]
    elif k == "NUMPAD_PERIOD":
        char = "."

    if char is not None and allowed_chars is not None and char not in allowed_chars:
        return None
    return char


NOTCH_CARD_FIELDS: tuple[str, ...] = ("notch_from_start", "notch_ratio")
PIECE_CARD_FIELDS: tuple[str, ...] = (
    "target_edge_length",
    "layer_id",
    "target_bone",
    "bone_t",
    "bone_angle_deg",
    "surface_offset",
    "pattern_up_angle_deg",
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
        self.transform_session: Optional[TransformModalSession] = None
        self._last_world_pos: Optional[tuple[float, float]] = None

        # クリップボードおよび形状・弧長転写設定
        self.edge_clipboard: Optional[dict[str, Any]] = None
        self.piece_clipboard: list[dict[str, Any]] = []
        self.shape_match_mode: str = "LENGTH_ADJUST_BULGE"
        self.shape_match_reverse: bool = True
        self.shape_match_flip_curvature: bool = False
        self.shape_match_apply_endpoint_offset: bool = False
        self.shape_match_ease_value: float = 0.0
        self.shape_match_ease_is_ratio: bool = False
        self.status_message: str = ""

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
            if self.transform_session is not None:
                self._cancel_transform_session()
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

    def _is_active_tool_busy(self) -> bool:
        """現在アクティブなツールがドラッグ中または作図途中の状態か返す."""
        tool = self.active_tool
        if not tool:
            return False
        if (
            getattr(tool, "is_dragging", False)
            or getattr(tool, "is_box_selecting", False)
            or getattr(tool, "is_drawing", False)
            or getattr(tool, "is_cutting", False)
        ):
            return True
        if getattr(tool, "points", None) or getattr(tool, "local_points", None):
            return True
        if int(getattr(tool, "step", 0)) > 0:
            return True
        return False

    @property
    def is_modal_input_active(self) -> bool:
        """数値入力・モーダル変形・ドラッグ中など、ツール切替ホットキーやESC終了を抑止すべき状態か."""
        return bool(
            self.editing_field is not None
            or self.picking_bone
            or self.transform_session is not None
            or self.sliding_field is not None
            or self.is_dragging_compass
            or self._is_active_tool_busy()
        )

    def _get_or_create_placement(self, piece_id: str) -> Placement:
        """指定ピースの Placement を取得し、存在しない場合は生成して登録する."""
        pl = self.doc.placements.get(piece_id)
        if not pl:
            pl = Placement(piece_id=piece_id)
            self.doc.placements[piece_id] = pl
        return pl

    def _is_piece_card_mode(self) -> bool:
        """ピース情報カード（単一・複数一括共通）を表示・操作すべき選択状態か判定する."""
        return (
            not self.selection.notch_ids
            and not self.selection.seam_ids
            and not self.selection.edge_ranges
            and len(self.selection.piece_ids) >= 1
        )

    def _is_multi_node_align_mode(self) -> bool:
        """複数頂点整列カードを表示・操作すべき選択状態か判定する."""
        return (
            not self.selection.notch_ids
            and not self.selection.seam_ids
            and not self.selection.edge_ranges
            and not self.selection.piece_ids
            and len(self.selection.node_ids) >= 2
        )

    def handle_event(self, event: EditorEvent) -> bool:
        """入力イベントをアクティブツールまたはグローバルハンドラーへ配送する."""
        # Alt 押下中はグリッド/ノード吸着を一時無効化 (自由な微調整用)
        self._snap_bypass = bool(event.alt)

        # ワールド座標・スクリーン座標の同期（キー単体イベント時は直近カーソル位置を維持）
        is_key_only = (
            bool(event.key or event.unicode)
            and not event.is_press
            and not event.is_release
            and not event.is_move
        )
        if event.screen_pos == (0.0, 0.0) and event.world_pos != (0.0, 0.0):
            event.screen_pos = self.view.world_to_screen(event.world_pos)
            self._last_world_pos = event.world_pos
        elif is_key_only and event.screen_pos == (0.0, 0.0) and self._last_world_pos is not None:
            event.world_pos = self._last_world_pos
            event.screen_pos = self.view.world_to_screen(self._last_world_pos)
        else:
            event.world_pos = self.view.screen_to_world(event.screen_pos)
            self._last_world_pos = event.world_pos

        # インライン数値入力モード中のキー入力優先処理
        if self.editing_field is not None and (event.key or event.unicode) and not event.is_release:
            k = event.key.upper()
            if k in ("ESCAPE", "ESC"):
                self.editing_field = None
                self.editing_buffer = ""
                self.is_dirty = True
                return True
            if k in ("ENTER", "NUMPAD_ENTER", "RET", "RETURN"):
                self._commit_inline_edit()
                self.editing_field = None
                self.editing_buffer = ""
                self.is_dirty = True
                return True
            if k in ("BACK_SPACE", "BACKSPACE"):
                self.editing_buffer = self.editing_buffer[:-1]
                self.is_dirty = True
                return True
            char = _extract_event_char(event)
            if char is not None:
                self.editing_buffer += char
                self.is_dirty = True
                return True
            return True

        # モーダル変形（G / R / S）中の最優先イベント処理
        if self.transform_session is not None:
            return self._handle_transform_modal_event(event)

        # Pick Bone モード中の専用イベント処理（クリック確定・X/Delete解除・Esc/右クリック/Bキャンセル）
        if self.picking_bone:
            if event.is_move:
                return True
            if event.is_press:
                if event.button == "LEFT":
                    if self.hovered_bone and self.selection.piece_ids:
                        valid_pids = [pid for pid in self.selection.piece_ids if pid in self.doc.pieces]
                        if valid_pids:
                            before_doc = self.doc.clone()
                            for pid in valid_pids:
                                pl = self._get_or_create_placement(pid)
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
            if event.key and not event.is_release:
                k = event.key.upper()
                if k in ("ESCAPE", "ESC", "B"):
                    self.picking_bone = False
                    self.hovered_bone = ""
                    self.is_dirty = True
                    return True
                elif k in ("X", "DELETE"):
                    if self.selection.piece_ids:
                        has_bone = any(
                            (pl := self.doc.placements.get(pid)) is not None and bool(pl.target_bone)
                            for pid in self.selection.piece_ids
                        )
                        if has_bone:
                            before_doc = self.doc.clone()
                            for pid in self.selection.piece_ids:
                                pl = self.doc.placements.get(pid)
                                if pl and pl.target_bone:
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

        # スライダー操作中のマウス移動・マウス解放・キャンセル処理
        if self.sliding_field is not None:
            if (event.is_press and event.button == "RIGHT") or (
                event.key and not event.is_release and event.key.upper() in ("ESCAPE", "ESC")
            ):
                if self.slide_before_doc is not None:
                    self.doc = self.slide_before_doc
                self.sliding_field = None
                self.is_slider_dragging = False
                self.slide_before_doc = None
                self.is_dirty = True
                return True
            if event.is_move:
                px, _ = event.screen_pos
                dx = px - self.slide_start_pos[0]
                if abs(dx) > 3.0:
                    self.is_slider_dragging = True

                if self.is_slider_dragging and self.selection.piece_ids:
                    shift_mod = 0.2 if event.shift else 1.0
                    for pid in self.selection.piece_ids:
                        if pid not in self.doc.pieces:
                            continue
                        pl = self._get_or_create_placement(pid)
                        if self.sliding_field == "bone_t":
                            # 200px で 0.0〜1.0 (感度: 0.005)
                            new_val = self.slide_start_val + dx * 0.005 * shift_mod
                            pl.bone_t = max(0.0, min(1.0, new_val))
                        elif self.sliding_field == "bone_angle_deg":
                            # 1px で 1.0度 (感度: 1.0)
                            new_val = self.slide_start_val + dx * 1.0 * shift_mod
                            pl.bone_angle_deg = normalize_angle_deg(new_val)
                        elif self.sliding_field == "pattern_up_angle_deg":
                            # 1px で 1.0度 (感度: 1.0)
                            new_val = self.slide_start_val + dx * 1.0 * shift_mod
                            pl.pattern_up_angle_deg = normalize_angle_deg(new_val)
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
                    first_pid = self.selection.get_active_piece_id()
                    if first_pid and first_pid in self.doc.pieces:
                        pl = self._get_or_create_placement(first_pid)
                        self.editing_field = field
                        if field == "bone_t":
                            self.editing_buffer = f"{int(pl.bone_t * 100)}"
                        elif field == "bone_angle_deg":
                            self.editing_buffer = f"{int(pl.bone_angle_deg)}"
                        elif field == "pattern_up_angle_deg":
                            self.editing_buffer = f"{int(round(pl.pattern_up_angle_deg))}"
                        elif field == "surface_offset":
                            self.editing_buffer = f"{m_to_cm(pl.surface_offset):.1f}"
                        self.is_dirty = True
                    return True

        # コンパスドラッグ中のマウス移動・マウス解放・キャンセル処理
        if self.is_dragging_compass:
            if (event.is_press and event.button == "RIGHT") or (
                event.key and not event.is_release and event.key.upper() in ("ESCAPE", "ESC")
            ):
                self.is_dragging_compass = False
                if self.drag_compass_before_doc is not None:
                    self.doc = self.drag_compass_before_doc
                    self.drag_compass_before_doc = None
                self.is_dirty = True
                return True
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
                    for pid in self.selection.piece_ids:
                        if pid in self.doc.pieces:
                            pl = self._get_or_create_placement(pid)
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
            self.status_message = ""
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

            is_piece_card_mode = self._is_piece_card_mode()

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
                for pid in self.selection.piece_ids:
                    if pid in self.doc.pieces:
                        pl = self._get_or_create_placement(pid)
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
                first_pid = self.selection.get_active_piece_id()
                p = self.doc.pieces.get(first_pid) if first_pid else None
                if p and first_pid:
                    pl = self._get_or_create_placement(first_pid)

                    for field_name in PIECE_CARD_FIELDS:
                        f_rect = layout.card_fields.get(field_name)
                        if not f_rect or not f_rect.contains(px, py):
                            continue
                        if field_name == "target_bone":
                            # ボーン直接選択（Pick Bone）モードを開始
                            self.picking_bone = True
                            self.is_dirty = True
                            return True

                        elif field_name in ("bone_t", "bone_angle_deg", "surface_offset", "pattern_up_angle_deg"):
                            # スライダー操作の準備（ドラッグでスライド調整、3px以内クリックでインライン直接入力）
                            self.sliding_field = field_name
                            self.slide_start_pos = event.screen_pos
                            self.is_slider_dragging = False
                            self.slide_before_doc = self.doc.clone()
                            if field_name == "bone_t":
                                self.slide_start_val = pl.bone_t
                            elif field_name == "bone_angle_deg":
                                self.slide_start_val = pl.bone_angle_deg
                            elif field_name == "pattern_up_angle_deg":
                                self.slide_start_val = pl.pattern_up_angle_deg
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

            elif self.selection.seam_ids and layout.info_card.contains(px, py):
                for f_name in (
                    "seam_match_a_to_b",
                    "seam_match_b_to_a",
                    "seam_match_mode",
                    "seam_match_apply_endpoint_offset",
                    "seam_match_flip_curvature",
                    "seam_match_ease",
                    "seam_routing",
                ):
                    f_rect = layout.card_fields.get(f_name)
                    if f_rect and f_rect.contains(px, py):
                        if f_name == "seam_match_a_to_b":
                            self.match_selected_seam(source_side="A")
                        elif f_name == "seam_match_b_to_a":
                            self.match_selected_seam(source_side="B")
                        elif f_name == "seam_match_mode":
                            modes = ["SIMILAR", "LENGTH_ADJUST_BULGE", "LENGTH_MOVE_END"]
                            cur_idx = modes.index(self.shape_match_mode) if self.shape_match_mode in modes else 0
                            self.shape_match_mode = modes[(cur_idx + 1) % len(modes)]
                            self.is_dirty = True
                        elif f_name == "seam_match_apply_endpoint_offset":
                            self.shape_match_apply_endpoint_offset = not self.shape_match_apply_endpoint_offset
                            self.is_dirty = True
                        elif f_name == "seam_match_flip_curvature":
                            self.shape_match_flip_curvature = not self.shape_match_flip_curvature
                            self.is_dirty = True
                        elif f_name == "seam_match_ease":
                            self.editing_field = "seam_match_ease"
                            if self.shape_match_ease_is_ratio:
                                self.editing_buffer = f"{self.shape_match_ease_value * 100.0:+.1f}%"
                            else:
                                self.editing_buffer = f"{self.shape_match_ease_value * 100.0:+.2f}"
                            self.is_dirty = True
                        elif f_name == "seam_routing":
                            self.toggle_selected_seams_routing()
                        return True

            elif self.selection.edge_ranges and layout.info_card.contains(px, py):
                for f_name in (
                    "edge_subdivide",
                    "edge_unfold",
                    "edge_mirror_dup",
                    "edge_copy_shape",
                    "edge_paste_shape",
                    "shape_match_mode",
                    "shape_match_apply_endpoint_offset",
                    "shape_match_reverse",
                    "shape_match_flip_curvature",
                    "shape_match_ease",
                    "edge_match_seam",
                ):
                    f_rect = layout.card_fields.get(f_name)
                    if f_rect and f_rect.contains(px, py):
                        if f_name == "edge_subdivide":
                            self.subdivide_selected_edge()
                        elif f_name == "edge_unfold":
                            self.unfold_selected()
                        elif f_name == "edge_mirror_dup":
                            self.mirror_duplicate_selected()
                        elif f_name == "edge_copy_shape":
                            self.copy_selected_edge_shape()
                        elif f_name == "edge_paste_shape":
                            self.paste_edge_shape()
                        elif f_name == "edge_match_seam":
                            self.match_selected_seam(source_side="A")
                        elif f_name == "shape_match_mode":
                            modes = ["SIMILAR", "LENGTH_ADJUST_BULGE", "LENGTH_MOVE_END"]
                            cur_idx = modes.index(self.shape_match_mode) if self.shape_match_mode in modes else 0
                            self.shape_match_mode = modes[(cur_idx + 1) % len(modes)]
                            self.is_dirty = True
                        elif f_name == "shape_match_apply_endpoint_offset":
                            self.shape_match_apply_endpoint_offset = not self.shape_match_apply_endpoint_offset
                            self.is_dirty = True
                        elif f_name == "shape_match_reverse":
                            self.shape_match_reverse = not self.shape_match_reverse
                            self.is_dirty = True
                        elif f_name == "shape_match_flip_curvature":
                            self.shape_match_flip_curvature = not self.shape_match_flip_curvature
                            self.is_dirty = True
                        elif f_name == "shape_match_ease":
                            self.editing_field = "shape_match_ease"
                            if self.shape_match_ease_is_ratio:
                                self.editing_buffer = f"{self.shape_match_ease_value * 100.0:+.1f}%"
                            else:
                                self.editing_buffer = f"{self.shape_match_ease_value * 100.0:+.2f}"
                            self.is_dirty = True
                        return True

            elif self._is_multi_node_align_mode() and layout.info_card.contains(px, py):
                for axis_k in ("x", "y"):
                    for mode_k in ("min", "center", "avg", "max"):
                        f_rect = layout.card_fields.get(f"align_{axis_k}_{mode_k}")
                        if f_rect and f_rect.contains(px, py):
                            self.align_selected_nodes(axis=axis_k.upper(), mode=mode_k.upper())
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

        # クリップボード・形状転写・複製ショートカット
        if event.ctrl:
            k = event.key.upper()
            if event.shift:
                if k == "C":
                    if self.copy_selected_edge_shape():
                        return True
                    self.status_message = trans("Please select an edge first")
                    self.is_dirty = True
                    return True
                elif k == "V":
                    if self.paste_edge_shape():
                        return True
                    if not self.edge_clipboard:
                        self.status_message = trans("Clipboard is empty (Copy edge shape first)")
                    else:
                        self.status_message = trans("Please select an edge first")
                    self.is_dirty = True
                    return True
                elif k == "D":
                    if self.duplicate_selected_pieces(mirror_horizontal=True):
                        return True
                    self.status_message = trans("Please select a piece first")
                    self.is_dirty = True
                    return True
                elif k == "M":
                    if self.match_selected_seam(source_side="B"):
                        return True
                    self.status_message = trans("Please select a seam first")
                    self.is_dirty = True
                    return True
            else:
                if k == "D":
                    if self.duplicate_selected_pieces():
                        return True
                    self.status_message = trans("Please select a piece first")
                    self.is_dirty = True
                    return True
                elif k == "C":
                    # 辺が選択されている場合は辺形状をコピー、そうでなければピースをコピー
                    if self.selection.edge_ranges:
                        if self.copy_selected_edge_shape():
                            return True
                    elif self.copy_selected_pieces():
                        return True
                    self.status_message = trans("Please select an edge or piece first")
                    self.is_dirty = True
                    return True
                elif k == "V":
                    # 辺が選択されている場合は辺形状をペースト、そうでなければピースをペースト
                    if self.selection.edge_ranges:
                        if self.paste_edge_shape():
                            return True
                        if not self.edge_clipboard:
                            self.status_message = trans("Clipboard is empty (Copy edge shape first)")
                        else:
                            self.status_message = trans("Please select an edge first")
                        self.is_dirty = True
                        return True
                    elif self.paste_copied_pieces():
                        return True
                    if self.edge_clipboard:
                        self.status_message = trans("Please select a target edge to paste shape")
                    else:
                        self.status_message = trans("Clipboard is empty (Copy piece or edge shape first)")
                    self.is_dirty = True
                    return True
                elif k == "M":
                    if self.match_selected_seam(source_side="A"):
                        return True
                    self.status_message = trans("Please select a seam first")
                    self.is_dirty = True
                    return True

        # グローバルショートカット（キー押下時のみ反応、RELEASEは無視）
        if event.key and not event.is_release:
            k_up = event.key.upper()

            # 選択削除 (X または Delete)
            if k_up in ("X", "DELETE"):
                if self.delete_selected():
                    return True

            # 縫合反転 (Fキーに一本化)
            if not event.ctrl and not event.alt and k_up == "F":
                if self.flip_selected_seams():
                    return True

            # モーダル変形開始 (G: 移動 / R: 回転 / S: 拡大縮小)
            if not event.ctrl and not event.alt and not event.shift and k_up in ("G", "R", "S"):
                if not self._is_active_tool_busy():
                    mode_map = {"G": "MOVE", "R": "ROTATE", "S": "SCALE"}
                    if self.start_modal_transform(mode=mode_map[k_up], cursor_world=event.world_pos):
                        return True
                    if self.status_message:
                        return True

            # 全選択 (A) / 全選択解除 (Alt+A)
            if not event.ctrl and not event.shift and k_up == "A":
                if event.alt:
                    self.selection.clear_all()
                    self.is_dirty = True
                    return True
                if (
                    self.selection.node_ids
                    and not self.selection.piece_ids
                    and not self.selection.edge_ranges
                ):
                    target_pids = {
                        pid
                        for pid, piece in self.doc.pieces.items()
                        if any(
                            nd.id in self.selection.node_ids
                            for path in piece.paths.values()
                            for nd in path.nodes
                        )
                    }
                    for pid in target_pids:
                        piece = self.doc.pieces.get(pid)
                        if not piece:
                            continue
                        for path in piece.paths.values():
                            for nd in path.nodes:
                                self.selection.select_node(pid, nd.id, add=True)
                    self.is_dirty = True
                    return True
                self.selection.select_all_pieces(self.doc.pieces.keys())
                self.is_dirty = True
                return True

            # 寸法線トグル (L)
            if not event.ctrl and not event.alt and k_up == "L":
                self.show_lengths = not self.show_lengths
                self.is_dirty = True
                return True

            # 縫合ボーン巻付けトグル / ボーン選択モードトグル (B)
            if not event.ctrl and not event.alt and k_up == "B":
                if self.selection.seam_ids:
                    if self.toggle_selected_seams_routing():
                        return True
                elif self.selection.piece_ids:
                    self.picking_bone = not self.picking_bone
                    self.is_dirty = True
                    return True

            # パターン上方向 ±90°回転 (T / Shift+T)
            if not event.ctrl and not event.alt and k_up == "T":
                if self.selection.piece_ids:
                    delta_deg = -90.0 if event.shift else 90.0
                    if self.rotate_selected_pieces_pattern_up(delta_deg=delta_deg):
                        return True

            # 形状合わせ方モード切替 (O / Shift+O)
            if not event.ctrl and not event.alt and k_up == "O":
                if self.selection.edge_ranges or self.selection.seam_ids:
                    if event.shift:
                        self.shape_match_apply_endpoint_offset = not self.shape_match_apply_endpoint_offset
                        self.status_message = f"{trans('Apply End Offset')}: {trans('ON') if self.shape_match_apply_endpoint_offset else trans('OFF')}"
                    else:
                        modes = ["SIMILAR", "LENGTH_ADJUST_BULGE", "LENGTH_MOVE_END"]
                        cur_idx = modes.index(self.shape_match_mode) if self.shape_match_mode in modes else 0
                        self.shape_match_mode = modes[(cur_idx + 1) % len(modes)]
                        self.status_message = f"{trans('Match Mode')}: {self.shape_match_mode}"
                    self.is_dirty = True
                    return True

        tool = self.active_tool
        if not tool:
            return False

        consumed = False
        if event.key and not event.is_release:
            consumed = tool.on_key_press(self, event)
            if not consumed and event.key.upper() in ("ESCAPE", "ESC") and self._is_active_tool_busy():
                tool.cancel(self)
                consumed = True
        elif event.is_press:
            consumed = tool.on_mouse_down(self, event)
        elif event.is_move:
            consumed = tool.on_mouse_move(self, event)
        elif event.is_release:
            consumed = tool.on_mouse_up(self, event)

        if consumed:
            self.is_dirty = True
        return consumed

    def start_modal_transform(
        self,
        *,
        mode: str,
        cursor_world: tuple[float, float] = (0.0, 0.0),
    ) -> bool:
        """G (MOVE) / R (ROTATE) / S (SCALE) のモーダル変形セッションを開始する."""
        mode_u = mode.upper()
        if mode_u not in ("MOVE", "ROTATE", "SCALE"):
            return False

        ok, err_msg = can_start_modal_transform(self.doc, self.selection, mode=mode_u)
        if not ok:
            if err_msg:
                self.status_message = trans(err_msg)
                self.is_dirty = True
            return False

        if self.transform_session is not None:
            self._cancel_transform_session()

        snapshot = capture_transform_snapshot(self.doc, self.selection)
        snap_ref = find_nearest_snap_reference(snapshot, cursor_world)
        label_map = {
            "MOVE": "Move Elements",
            "ROTATE": "Rotate Elements",
            "SCALE": "Scale Elements",
        }
        self.history.begin_transaction(self.doc, label=label_map[mode_u])
        self.transform_session = TransformModalSession(
            mode=mode_u,
            snapshot=snapshot,
            start_cursor_world=cursor_world,
            current_cursor_world=cursor_world,
            snap_reference_world=snap_ref,
        )
        self._update_transform_status_message(self.transform_session)
        self.is_dirty = True
        return True

    def _handle_transform_modal_event(self, event: EditorEvent) -> bool:
        """モーダル変形セッション実行中のイベントをすべて優先処理する."""
        session = self.transform_session
        if session is None:
            return False

        if event.is_move:
            session.current_cursor_world = event.world_pos
            if not session.numeric_buffer:
                self._apply_cursor_transform(session, shift=bool(event.shift))
            self._update_transform_status_message(session)
            self.is_dirty = True
            return True

        if event.is_press:
            if event.button == "LEFT":
                self._commit_transform_session()
                return True
            if event.button == "RIGHT":
                self._cancel_transform_session()
                return True
            return True

        if event.is_release:
            return True

        if event.key or event.unicode:
            k = event.key.upper()
            if k in ("ENTER", "NUMPAD_ENTER", "RET", "RETURN"):
                self._commit_transform_session()
                return True
            if k in ("ESCAPE", "ESC"):
                self._cancel_transform_session()
                return True
            if k in ("BACK_SPACE", "BACKSPACE"):
                if session.numeric_buffer:
                    session.numeric_buffer = session.numeric_buffer[:-1]
                    self._reapply_modal_transform(session, shift=bool(event.shift))
                    self._update_transform_status_message(session)
                    self.is_dirty = True
                return True

            if not event.ctrl and not event.alt:
                if k == "X" and session.mode in ("MOVE", "SCALE"):
                    session.axis_lock = None if session.axis_lock == "X" else "X"
                    self._reapply_modal_transform(session, shift=bool(event.shift))
                    self._update_transform_status_message(session)
                    self.is_dirty = True
                    return True
                if k == "Y" and session.mode in ("MOVE", "SCALE"):
                    session.axis_lock = None if session.axis_lock == "Y" else "Y"
                    self._reapply_modal_transform(session, shift=bool(event.shift))
                    self._update_transform_status_message(session)
                    self.is_dirty = True
                    return True
                if (
                    k == "N"
                    and session.mode == "MOVE"
                    and not session.numeric_buffer.lower().endswith("i")
                    and not session.snapshot.piece_only_mode
                    and session.snapshot.active_edge_normal is not None
                ):
                    order: list[Optional[str]] = (
                        [None, "NORMAL", "NORMAL_ACTIVE"]
                        if len(self.selection.edge_ranges) >= 2
                        else [None, "NORMAL"]
                    )
                    cur = session.axis_lock if session.axis_lock in order else None
                    session.axis_lock = order[(order.index(cur) + 1) % len(order)]
                    self._reapply_modal_transform(session, shift=bool(event.shift))
                    self._update_transform_status_message(session)
                    self.is_dirty = True
                    return True

                char = _extract_event_char(event, allowed_chars="0123456789.-+%cminCMIN")
                if char is not None:
                    if char.isalpha():
                        char = char.lower()
                    session.numeric_buffer += char
                    self._reapply_modal_transform(session, shift=bool(event.shift))
                    self._update_transform_status_message(session)
                    self.is_dirty = True
                    return True
        return True

    def _reapply_modal_transform(self, session: TransformModalSession, *, shift: bool) -> None:
        if session.numeric_buffer:
            self._apply_numeric_transform(session)
        else:
            self._apply_cursor_transform(session, shift=shift)

    def _apply_cursor_transform(self, session: TransformModalSession, *, shift: bool) -> None:
        if session.mode == "MOVE":
            delta = (
                session.current_cursor_world[0] - session.start_cursor_world[0],
                session.current_cursor_world[1] - session.start_cursor_world[1],
            )
            eff = apply_translation(
                self.doc,
                session.snapshot,
                delta_world=delta,
                view=self.view,
                snap_settings=self.snap_settings,
                snap_reference_world=session.snap_reference_world,
                axis_lock=session.axis_lock,
                clamp_ortho=shift,
            )
            session.current_delta = eff
        elif session.mode == "ROTATE":
            px, py = session.snapshot.pivot_world
            v0x = session.start_cursor_world[0] - px
            v0y = session.start_cursor_world[1] - py
            v1x = session.current_cursor_world[0] - px
            v1y = session.current_cursor_world[1] - py
            if math.hypot(v0x, v0y) < 1e-6 or math.hypot(v1x, v1y) < 1e-6:
                d_angle = 0.0
            else:
                a0 = math.atan2(v0y, v0x)
                a1 = math.atan2(v1y, v1x)
                d_angle = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))
            if shift:
                step = math.radians(5.0)
                d_angle = round(d_angle / step) * step
            apply_rotation(self.doc, session.snapshot, angle_rad=d_angle)
            session.current_angle_rad = d_angle
        elif session.mode == "SCALE":
            px, py = session.snapshot.pivot_world
            r0 = math.hypot(
                session.start_cursor_world[0] - px,
                session.start_cursor_world[1] - py,
            )
            r1 = math.hypot(
                session.current_cursor_world[0] - px,
                session.current_cursor_world[1] - py,
            )
            raw_s = 1.0 if r0 < 1e-5 else max(0.01, r1 / r0)
            if shift:
                raw_s = max(0.05, round(raw_s / 0.05) * 0.05)
            sx = raw_s if session.axis_lock in (None, "X") else 1.0
            sy = raw_s if session.axis_lock in (None, "Y") else 1.0
            eff_sx, eff_sy = apply_scale(self.doc, session.snapshot, sx=sx, sy=sy)
            session.current_scale = (eff_sx, eff_sy)

    def _apply_numeric_transform(self, session: TransformModalSession) -> None:
        buf = session.numeric_buffer.strip()
        if not buf:
            self._apply_cursor_transform(session, shift=False)
            return

        if session.mode == "MOVE":
            if buf in ("-", "+", ".", "-.", "+."):
                eff = apply_translation(
                    self.doc,
                    session.snapshot,
                    delta_world=(0.0, 0.0),
                    axis_lock=session.axis_lock,
                )
                session.current_delta = eff
                return
            try:
                dist_m = parse_length_input(buf, default_unit="cm")
            except ValueError:
                return
            if session.axis_lock == "X":
                raw_delta = (dist_m, 0.0)
            elif session.axis_lock == "Y":
                raw_delta = (0.0, dist_m)
            elif (
                session.axis_lock in ("NORMAL", "NORMAL_ACTIVE")
                and session.snapshot.active_edge_normal is not None
            ):
                nx, ny = session.snapshot.active_edge_normal
                raw_delta = (nx * dist_m, ny * dist_m)
            else:
                cdx = session.current_cursor_world[0] - session.start_cursor_world[0]
                cdy = session.current_cursor_world[1] - session.start_cursor_world[1]
                clen = math.hypot(cdx, cdy)
                ux, uy = (cdx / clen, cdy / clen) if clen > 1e-6 else (1.0, 0.0)
                raw_delta = (ux * dist_m, uy * dist_m)
            eff = apply_translation(
                self.doc,
                session.snapshot,
                delta_world=raw_delta,
                view=self.view,
                snap_settings=None,
                snap_reference_world=None,
                axis_lock=session.axis_lock,
                clamp_ortho=False,
            )
            session.current_delta = eff
        elif session.mode == "ROTATE":
            clean = buf.replace("°", "").replace("deg", "").strip()
            if clean in ("", "-", "+", ".", "-.", "+."):
                apply_rotation(self.doc, session.snapshot, angle_rad=0.0)
                session.current_angle_rad = 0.0
                return
            try:
                deg = float(clean)
            except ValueError:
                return
            angle_rad = math.radians(deg)
            apply_rotation(self.doc, session.snapshot, angle_rad=angle_rad)
            session.current_angle_rad = angle_rad
        elif session.mode == "SCALE":
            is_pct = buf.endswith("%")
            num_str = buf[:-1].strip() if is_pct else buf
            if num_str in ("", "-", "+", ".", "-.", "+."):
                return
            try:
                val = float(num_str)
            except ValueError:
                return
            if is_pct:
                val /= 100.0
            sx = val if session.axis_lock in (None, "X") else 1.0
            sy = val if session.axis_lock in (None, "Y") else 1.0
            eff_sx, eff_sy = apply_scale(self.doc, session.snapshot, sx=sx, sy=sy)
            session.current_scale = (eff_sx, eff_sy)

    def _commit_transform_session(self) -> None:
        session = self.transform_session
        if session is None:
            return
        if session.mode == "SCALE":
            sx, sy = session.current_scale
            if abs(sx) <= 1.001e-4 or abs(sy) <= 1.001e-4:
                self._cancel_transform_session()
                self.status_message = trans("Scale cannot be zero")
                self.is_dirty = True
                return
            finalize_scale_orientation(self.doc, session.snapshot, sx=sx, sy=sy)

        self.transform_session = None
        self.status_message = ""
        self.history.commit_transaction(self.doc)
        self.is_dirty = True

    def _cancel_transform_session(self) -> None:
        session = self.transform_session
        if session is None:
            return
        self.transform_session = None
        self.status_message = ""
        restore_snapshot(self.doc, session.snapshot)
        self.history.cancel_transaction(self.doc)
        self.is_dirty = True

    def _update_transform_status_message(self, session: TransformModalSession) -> None:
        axis_map = {
            "X": " [X]",
            "Y": " [Y]",
            "NORMAL": f" [{trans('Normal Offset')}]",
            "NORMAL_ACTIVE": f" [{trans('Active Normal')}]",
        }
        axis_label = axis_map.get(session.axis_lock or "", "")
        if session.mode == "MOVE":
            if session.numeric_buffer:
                self.status_message = f"{trans('Move')}{axis_label}: {session.numeric_buffer}"
            else:
                dx, dy = session.current_delta
                self.status_message = f"{trans('Move')}{axis_label}: ({m_to_cm(dx):+.2f}, {m_to_cm(dy):+.2f}) cm"
        elif session.mode == "ROTATE":
            if session.numeric_buffer:
                self.status_message = f"{trans('Rotate')}: {session.numeric_buffer}°"
            else:
                deg = math.degrees(session.current_angle_rad)
                self.status_message = f"{trans('Rotate')}: {deg:+.1f}°"
        elif session.mode == "SCALE":
            if session.numeric_buffer:
                self.status_message = f"{trans('Scale')}{axis_label}: {session.numeric_buffer}"
            else:
                sx, sy = session.current_scale
                self.status_message = f"{trans('Scale')}{axis_label}: ({sx:.2f}, {sy:.2f})"

    def _commit_inline_edit(self) -> None:
        """インライン入力バッファの内容を選択中ピースまたはノッチへ適用する."""
        if not self.editing_field or not self.editing_buffer:
            return

        if self.editing_field in ("shape_match_ease", "seam_match_ease"):
            buf = self.editing_buffer.strip()
            if buf.endswith("%"):
                try:
                    pct = float(buf[:-1])
                    self.shape_match_ease_value = pct / 100.0
                    self.shape_match_ease_is_ratio = True
                    self.is_dirty = True
                except ValueError:
                    pass
            else:
                from .units import parse_length_input
                try:
                    val_m = parse_length_input(buf, default_unit="cm")
                    self.shape_match_ease_value = val_m
                    self.shape_match_ease_is_ratio = False
                    self.is_dirty = True
                except ValueError:
                    pass
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
                    pl.bone_angle_deg = normalize_angle_deg(val_f)
                    modified = True
                except ValueError:
                    pass
            elif self.editing_field == "pattern_up_angle_deg":
                try:
                    clean_buf = self.editing_buffer.replace("°", "").replace("deg", "").strip()
                    val_f = float(clean_buf)
                    pl.pattern_up_angle_deg = normalize_angle_deg(val_f)
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

    def rotate_selected_pieces_pattern_up(self, *, delta_deg: float = 90.0) -> bool:
        """選択中ピースのボーン配置上方向角度 (pattern_up_angle_deg) を加算回転する."""
        if not self.selection.piece_ids:
            return False

        before_doc = self.doc.clone()
        last_angle = 0.0
        modified = False
        for pid in self.selection.piece_ids:
            if pid not in self.doc.pieces:
                continue
            pl = self.doc.placements.get(pid)
            if not pl:
                pl = Placement(piece_id=pid)
                self.doc.placements[pid] = pl
            pl.pattern_up_angle_deg = normalize_angle_deg(pl.pattern_up_angle_deg + delta_deg)
            last_angle = pl.pattern_up_angle_deg
            modified = True

        if not modified:
            return False

        self.history.record_change(
            description="Rotate Pattern Up",
            before_doc=before_doc,
            after_doc=self.doc,
        )
        self.status_message = f"{trans('Pattern Up')}: {int(round(last_angle))}°"
        self.is_dirty = True
        return True

    def toggle_selected_seams_routing(self) -> bool:
        """選択中縫合のルーティング方式 (Direct / Bone Wrap) をトグルする."""
        target_seams = [self.doc.seams[sid] for sid in self.selection.seam_ids if sid in self.doc.seams]
        if not target_seams:
            return False

        before_doc = self.doc.clone()
        new_routing = (
            SeamRouting.BONE_WRAP
            if any(s.routing != SeamRouting.BONE_WRAP for s in target_seams)
            else SeamRouting.DIRECT
        )
        has_any_bone = False
        for seam in target_seams:
            seam.routing = new_routing
            pl_a = self.doc.placements.get(seam.a.piece_id)
            pl_b = self.doc.placements.get(seam.b.piece_id)
            if (pl_a and pl_a.target_bone) or (pl_b and pl_b.target_bone):
                has_any_bone = True

        self.history.record_change(
            description="Toggle Seam Routing",
            before_doc=before_doc,
            after_doc=self.doc,
        )
        if new_routing == SeamRouting.BONE_WRAP:
            if has_any_bone:
                self.status_message = f"{trans('Routing')}: {trans('Bone Wrap')}"
            else:
                self.status_message = f"{trans('Routing')}: {trans('Bone Wrap')} ({trans('No target bone')})"
        else:
            self.status_message = f"{trans('Routing')}: {trans('Direct')}"
        self.is_dirty = True
        return True

    def delete_selected(self) -> bool:
        """現在選択中の要素（ノッチ、縫合、ノード、内部線、またはピース）を優先順位に従って削除し履歴に記録する.

        優先順位:
        1. ノッチが選択されている場合 -> ノッチのみを削除
        2. 縫合が選択されている場合 -> 縫合のみを削除
        3. ノードが選択されている場合 -> ノードのみを削除
        4. 辺が選択されている場合 -> 内部線のみ削除（外周辺は誤削除防止のため保護）
        5. 上記がなくピースが選択されている場合 -> ピース全体を削除
        """
        if (
            not self.selection.piece_ids
            and not self.selection.node_ids
            and not self.selection.notch_ids
            and not self.selection.seam_ids
            and not self.selection.edge_ranges
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

        # 4. 辺削除（内部線のみ削除可、外周辺選択時は親ピースの誤削除を防止）
        if self.selection.edge_ranges:
            has_outer = False
            internal_segs: dict[tuple[str, str], set[int]] = {}
            for er in self.selection.edge_ranges:
                piece = self.doc.pieces.get(er.piece_id)
                if not piece:
                    continue
                path = piece.paths.get(er.path_id)
                if not path:
                    continue
                if path.role == PathRole.OUTER:
                    has_outer = True
                    break
                if path.role == PathRole.INTERNAL:
                    res = path.resolve_edge_range(er)
                    if res:
                        seg_idx, _, _ = res
                        internal_segs.setdefault((piece.id, path.id), set()).add(seg_idx)

            if has_outer:
                self.status_message = trans(
                    "Outer boundary edges cannot be deleted (Delete nodes instead)"
                )
                self.is_dirty = True
                return False

            if internal_segs:
                removed_seg_keys: set[tuple[str, str, str]] = set()
                removed_path_keys: set[tuple[str, str]] = set()
                for (pid, path_id), seg_indices in internal_segs.items():
                    piece = self.doc.pieces.get(pid)
                    if not piece:
                        continue
                    path = piece.paths.get(path_id)
                    if not path:
                        continue
                    n_nodes = len(path.nodes)
                    total_segs = n_nodes if path.closed else max(0, n_nodes - 1)
                    for s_idx in seg_indices:
                        if 0 <= s_idx < n_nodes:
                            removed_seg_keys.add((pid, path_id, path.nodes[s_idx].id))

                    if len(seg_indices) >= total_segs or n_nodes <= 2:
                        del piece.paths[path_id]
                        removed_path_keys.add((pid, path_id))
                        modified = True
                    else:
                        kept_runs: list[list[Node]] = []
                        cur_run: list[Node] = []
                        for i in range(total_segs):
                            if i not in seg_indices:
                                if not cur_run:
                                    cur_run.append(path.nodes[i])
                                cur_run.append(path.nodes[(i + 1) % n_nodes])
                            else:
                                if len(cur_run) >= 2:
                                    kept_runs.append(cur_run)
                                cur_run = []
                        if len(cur_run) >= 2:
                            kept_runs.append(cur_run)

                        del piece.paths[path_id]
                        for r_idx, run_nodes in enumerate(kept_runs):
                            target_id = path_id if r_idx == 0 else new_id()
                            piece.paths[target_id] = Path(
                                id=target_id,
                                role=PathRole.INTERNAL,
                                nodes=[
                                    Node(
                                        id=nd.id,
                                        position=nd.position,
                                        kind=nd.kind,
                                        in_handle=nd.in_handle,
                                        out_handle=nd.out_handle,
                                    )
                                    for nd in run_nodes
                                ],
                                closed=False,
                            )
                        modified = True

                dead_notches = [
                    nid
                    for nid, n in self.doc.notches.items()
                    if (n.piece_id, n.path_id) in removed_path_keys
                    or (n.piece_id, n.path_id, n.segment_start_id) in removed_seg_keys
                ]
                for nid in dead_notches:
                    del self.doc.notches[nid]

                dead_seams = [
                    sid
                    for sid, s in self.doc.seams.items()
                    if any(
                        (s_er.piece_id, s_er.path_id) in removed_path_keys
                        or (s_er.piece_id, s_er.path_id, s_er.start_node_id) in removed_seg_keys
                        for s_er in s.iter_unique_edge_ranges()
                    )
                ]
                for sid in dead_seams:
                    del self.doc.seams[sid]

                self.selection.edge_ranges.clear()
                if modified:
                    self.history.record_change(
                        description="Delete Internal Edge",
                        before_doc=before_doc,
                        after_doc=self.doc,
                    )
                    self.is_dirty = True
                    return True
            return False

        # 5. ピース削除（明示的にピースが選択されている場合のみ）
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

    def subdivide_selected_edge(self, *, ratio: float = 0.5) -> bool:
        """選択中の全辺を実寸比率 ratio で一括分割し、新挿入ノード群を選択状態にする."""
        targets = list(self.selection.edge_ranges)
        if not targets:
            return False
        before_doc = self.doc.clone()
        inserted: list[tuple[str, str]] = []
        for er in targets:
            new_node = self.doc.subdivide_edge(er, ratio=ratio)
            if new_node:
                inserted.append((er.piece_id, new_node.id))
        if not inserted:
            return False
        self.selection.clear()
        for pid, nid in inserted:
            self.selection.select_node(pid, nid, add=True)
        desc = "Subdivide Edge" if len(inserted) == 1 else "Subdivide Edges"
        self.history.record_change(
            description=desc,
            before_doc=before_doc,
            after_doc=self.doc,
        )
        self.is_dirty = True
        return True

    def subdivide_selected_edges(self, *, ratio: float = 0.5) -> bool:
        """選択中の全辺を一括分割する (`subdivide_selected_edge` の複数対応エイリアス)."""
        return self.subdivide_selected_edge(ratio=ratio)

    def align_selected_nodes(self, *, axis: str, mode: str) -> bool:
        """選択中の複数頂点をワールド座標 X / Y で整列し履歴に記録する."""
        before_doc = self.doc.clone()
        if not apply_node_alignment(self.doc, self.selection, axis=axis, mode=mode):
            return False
        axis_u = axis.upper()
        mode_u = mode.upper()
        self.history.record_change(
            description=f"Align Nodes {axis_u} ({mode_u.title()})",
            before_doc=before_doc,
            after_doc=self.doc,
        )
        self.is_dirty = True
        return True

    def _resolve_selected_or_min_x_edge(self) -> Optional[tuple[str, EdgeRange]]:
        """現在選択中の辺、または選択ピースの外周で中点X座標が最小の辺 (piece_id, EdgeRange) を返す."""
        er = self.selection.get_active_edge_range()
        pid = er.piece_id if er else (next(iter(self.selection.piece_ids)) if self.selection.piece_ids else None)
        if not pid:
            return None
        piece = self.doc.pieces.get(pid)
        if not piece:
            return None
        outer = piece.get_outer_path()
        if not outer:
            return None
        if not er:
            nodes = outer.nodes
            n_orig = len(nodes)
            if n_orig < 2:
                return None
            best_i = min(
                range(n_orig),
                key=lambda i: 0.5 * (nodes[(i + 1) % n_orig].position[0] + nodes[i].position[0]),
            )
            er = EdgeRange(
                piece_id=pid,
                path_id=outer.id,
                start_node_id=nodes[best_i].id,
                end_node_id=nodes[(best_i + 1) % n_orig].id,
            )
        return (pid, er)

    def unfold_selected(self) -> bool:
        """選択中の辺（未選択時は中点X最小辺）を軸に対称展開（わ展開）する."""
        resolved = self._resolve_selected_or_min_x_edge()
        if not resolved:
            return False
        pid, er = resolved
        before_doc = self.doc.clone()
        if self.doc.unfold_piece_along_edge(er):
            self.selection.clear()
            self.selection.select_piece(pid)
            self.history.record_change(
                description="Unfold Piece",
                before_doc=before_doc,
                after_doc=self.doc,
            )
            self.is_dirty = True
            return True
        return False

    def mirror_duplicate_selected(self) -> bool:
        """選択中の辺（未選択時は中点X最小辺）を軸に対称な別ピースを生成・自動縫合する."""
        resolved = self._resolve_selected_or_min_x_edge()
        if not resolved:
            return False
        _, er = resolved
        before_doc = self.doc.clone()
        new_piece = self.doc.mirror_duplicate_across_edge(er, create_seam=True)
        if new_piece:
            self.selection.clear()
            self.selection.select_piece(new_piece.id)
            self.history.record_change(
                description="Symmetric Duplicate Piece",
                before_doc=before_doc,
                after_doc=self.doc,
            )
            self.is_dirty = True
            return True
        return False

    def duplicate_selected_pieces(self, *, mirror_horizontal: bool = False) -> bool:
        """選択中の全ピースを複製（mirror_horizontal=Trueで左右反転複製）する."""
        if not self.selection.piece_ids:
            return False
        before_doc = self.doc.clone()
        new_pids: list[str] = []
        for pid in list(self.selection.piece_ids):
            orig_piece = self.doc.pieces.get(pid)
            orig_name = orig_piece.name if orig_piece else ""
            new_p = self.doc.duplicate_piece(pid)
            if new_p:
                if mirror_horizontal:
                    self.doc.flip_piece(new_p.id, direction="HORIZONTAL", flip_name=False)
                    if orig_name:
                        new_p.name = self.doc._symmetric_name(orig_name)
                new_pids.append(new_p.id)
        if new_pids:
            self.selection.clear()
            for npid in new_pids:
                self.selection.select_piece(npid, add=True)
            desc = "Duplicate Mirrored Piece" if mirror_horizontal else "Duplicate Piece"
            self.history.record_change(
                description=desc,
                before_doc=before_doc,
                after_doc=self.doc,
            )
            self.is_dirty = True
            return True
        return False

    def copy_selected_pieces(self) -> bool:
        """選択中のピースをクリップボードにコピーする."""
        if not self.selection.piece_ids:
            return False
        from dataclasses import asdict
        self.piece_clipboard = []
        for pid in self.selection.piece_ids:
            piece = self.doc.pieces.get(pid)
            if piece:
                item = {
                    "id": pid,
                    "piece": piece.to_dict(),
                    "notches": [asdict(n) for n in self.doc.notches.values() if n.piece_id == pid],
                    "anchors": [asdict(a) for a in self.doc.anchors.values() if a.piece_id == pid],
                    "placement": asdict(self.doc.placements[pid]) if pid in self.doc.placements else None,
                }
                self.piece_clipboard.append(item)
        return bool(self.piece_clipboard)

    def paste_copied_pieces(self) -> bool:
        """クリップボードのスナップショットからピースを新規IDで復元ペーストする."""
        if not self.piece_clipboard:
            return False
        from dataclasses import replace
        from .document import Anchor, AnchorKind, Node, Notch, NotchKind, Path, Piece, Placement

        before_doc = self.doc.clone()
        new_pids: list[str] = []
        for item in self.piece_clipboard:
            if not isinstance(item, dict):
                continue
            pdata = item.get("piece", item)
            snap_piece = Piece.from_dict(pdata)
            new_pid = new_id()
            path_map: dict[str, str] = {}
            node_map: dict[str, str] = {}
            new_paths: dict[str, Path] = {}

            for old_path_id, p in snap_piece.paths.items():
                new_path_id = new_id()
                path_map[old_path_id] = new_path_id
                new_nodes: list[Node] = []
                for nd in p.nodes:
                    new_nid = new_id()
                    node_map[nd.id] = new_nid
                    new_nodes.append(Node(
                        id=new_nid,
                        position=nd.position,
                        kind=nd.kind,
                        in_handle=nd.in_handle,
                        out_handle=nd.out_handle,
                    ))
                new_paths[new_path_id] = Path(
                    id=new_path_id,
                    role=p.role,
                    nodes=new_nodes,
                    closed=p.closed,
                )

            new_name = self.doc._unique_piece_name(f"{snap_piece.name}_Copy")
            new_transform = replace(snap_piece.transform, tx=snap_piece.transform.tx + 0.05)
            new_outer_id = path_map.get(snap_piece.outer_path_id, "")
            new_p = Piece(
                id=new_pid,
                name=new_name,
                paths=new_paths,
                outer_path_id=new_outer_id,
                transform=new_transform,
                target_edge_length=snap_piece.target_edge_length,
                layer_id=snap_piece.layer_id,
            )
            self.doc.pieces[new_pid] = new_p

            # ノッチ復元（新しい path_id / segment_start_id へマッピング）
            for ndata in item.get("notches", []):
                old_pid_path = ndata.get("path_id", "")
                old_seg_start = ndata.get("segment_start_id", "")
                if old_pid_path in path_map and old_seg_start in node_map:
                    nid = new_id()
                    raw_kind = ndata.get("kind", "V")
                    n_kind = raw_kind if isinstance(raw_kind, NotchKind) else NotchKind(str(raw_kind))
                    self.doc.notches[nid] = Notch(
                        id=nid,
                        piece_id=new_pid,
                        path_id=path_map[old_pid_path],
                        segment_start_id=node_map[old_seg_start],
                        t=float(ndata.get("t", 0.5)),
                        kind=n_kind,
                        depth=float(ndata.get("depth", 0.005)),
                        width=float(ndata.get("width", 0.002)),
                    )

            # アンカー復元
            for adata in item.get("anchors", []):
                aid = new_id()
                raw_akind = adata.get("kind", "PIN")
                a_kind = raw_akind if isinstance(raw_akind, AnchorKind) else AnchorKind(str(raw_akind))
                pos = adata.get("position", (0.0, 0.0))
                self.doc.anchors[aid] = Anchor(
                    id=aid,
                    piece_id=new_pid,
                    position=(float(pos[0]), float(pos[1])),
                    kind=a_kind,
                    target_object=str(adata.get("target_object", "")),
                    target_bone=str(adata.get("target_bone", "")),
                    weight=float(adata.get("weight", 1.0)),
                )

            # 配置復元
            if item.get("placement"):
                pld = item["placement"]
                off3d = pld.get("offset_3d", (0.0, 0.0, 0.0))
                eul3d = pld.get("euler_3d", (0.0, 0.0, 0.0))
                self.doc.placements[new_pid] = Placement(
                    piece_id=new_pid,
                    offset_3d=(float(off3d[0]), float(off3d[1]), float(off3d[2])),
                    euler_3d=(float(eul3d[0]), float(eul3d[1]), float(eul3d[2])),
                    preset=str(pld.get("preset", "CHEST_FRONT")),
                    target_bone=str(pld.get("target_bone", "")),
                    bone_t=float(pld.get("bone_t", 0.5)),
                    bone_angle_deg=normalize_angle_deg(float(pld.get("bone_angle_deg", 0.0))),
                    surface_offset=float(pld.get("surface_offset", 0.08)),
                    pattern_up_angle_deg=normalize_angle_deg(float(pld.get("pattern_up_angle_deg", 0.0))),
                )
            new_pids.append(new_pid)

        if new_pids:
            self.selection.clear()
            for npid in new_pids:
                self.selection.select_piece(npid, add=True)
            self.history.record_change(
                description="Paste Piece",
                before_doc=before_doc,
                after_doc=self.doc,
            )
            self.is_dirty = True
            return True
        return False

    def flip_selected_pieces(self, *, direction: str = "HORIZONTAL") -> bool:
        """選択中の全ピースをバウンディングボックス中心基準で反転する."""
        if not self.selection.piece_ids:
            return False
        before_doc = self.doc.clone()
        modified = False
        for pid in list(self.selection.piece_ids):
            if self.doc.flip_piece(pid, direction=direction):
                modified = True
        if modified:
            desc = "Flip Piece Horizontal" if direction == "HORIZONTAL" else "Flip Piece Vertical"
            self.history.record_change(
                description=desc,
                before_doc=before_doc,
                after_doc=self.doc,
            )
            self.is_dirty = True
            return True
        return False

    def copy_selected_edge_shape(self) -> bool:
        """選択中の単一または連続した複数辺の形状と弧長をエッジクリップボードに保存する."""
        if not self.selection.edge_ranges:
            logger.debug("[Pattern] copy_selected_edge_shape: no edges selected")
            return False

        first_er = self.selection.edge_ranges[0]
        piece = self.doc.pieces.get(first_er.piece_id)
        if not piece:
            logger.debug(f"[Pattern] copy_selected_edge_shape: piece '{first_er.piece_id}' not found")
            return False
        path = piece.paths.get(first_er.path_id)
        if not path:
            logger.debug(f"[Pattern] copy_selected_edge_shape: path '{first_er.path_id}' not found")
            return False

        from .document import order_edge_chain, extract_edge_chain_data
        ordered_chain = order_edge_chain(path, self.selection.edge_ranges)
        if not ordered_chain:
            logger.debug("[Pattern] copy_selected_edge_shape: selected edges do not form a continuous chain")
            return False

        chain_data = extract_edge_chain_data(piece, path, chain_ranges=ordered_chain)
        if not chain_data:
            logger.debug("[Pattern] copy_selected_edge_shape: failed to extract chain data")
            return False

        from .units import format_cm
        chain_data["edge_range"] = ordered_chain[0]
        chain_data["edge_ranges"] = ordered_chain
        chain_data["length"] = chain_data["total_length"]
        self.edge_clipboard = chain_data

        k = chain_data["num_segments"]
        if k > 1:
            self.status_message = f"{trans('Edge chain copied')} ({k} {trans('edges')}): {format_cm(chain_data['total_length'])}"
        else:
            self.status_message = f"{trans('Edge shape copied')}: {format_cm(chain_data['total_length'])}"

        logger.debug(f"[Pattern] copy_selected_edge_shape: SUCCESS (segments={k}, length={format_cm(chain_data['total_length'])})")
        self.is_dirty = True
        return True

    def paste_edge_shape(self) -> bool:
        """エッジクリップボードの形状・弧長を選択中の辺群へ適用する."""
        if not self.edge_clipboard or not self.selection.edge_ranges:
            logger.debug(f"[Pattern] paste_edge_shape: early return (has_clipboard={bool(self.edge_clipboard)}, edge_ranges={len(self.selection.edge_ranges)})")
            return False

        first_dst = self.selection.edge_ranges[0]
        dst_piece = self.doc.pieces.get(first_dst.piece_id)
        if not dst_piece:
            return False
        dst_path = dst_piece.paths.get(first_dst.path_id)
        if not dst_path:
            return False

        from .document import order_edge_chain
        ordered_dst = order_edge_chain(dst_path, self.selection.edge_ranges)
        if not ordered_dst:
            logger.debug("[Pattern] paste_edge_shape: target edges do not form a continuous chain")
            return False

        before_doc = self.doc.clone()
        ease = (self.edge_clipboard["total_length"] * self.shape_match_ease_value) if self.shape_match_ease_is_ratio else self.shape_match_ease_value

        ok = self.doc.match_edge_chain_shape(
            ordered_dst,
            self.edge_clipboard,
            mode=self.shape_match_mode,
            reverse=self.shape_match_reverse,
            flip_curvature=self.shape_match_flip_curvature,
            apply_endpoint_offset=self.shape_match_apply_endpoint_offset,
            ease_offset=ease,
        )

        if ok:
            mode_desc = f"{self.shape_match_mode}"
            if self.shape_match_apply_endpoint_offset:
                mode_desc += f", {trans('Offset')}"
            self.status_message = f"{trans('Edge shape pasted')} ({mode_desc})"
            self.history.record_change(
                description="Paste Edge Shape",
                before_doc=before_doc,
                after_doc=self.doc,
            )
            self.is_dirty = True
            logger.debug("[Pattern] paste_edge_shape: SUCCESS")
            return True

        logger.debug("[Pattern] paste_edge_shape: match_edge_chain_shape returned False")
        return False

    def match_selected_seam(self, *, source_side: str = "A") -> bool:
        """選択中の Seam に対し、source_side の形状と弧長を反対側へ反映する.

        縫合線が直接選択されていない場合でも、選択中の辺 (edge_ranges) に接続された
        縫合線を自動特定して形状を反映する。
        """
        target_seams: list[tuple[str, str]] = []  # (seam_id, resolved_side)

        if self.selection.seam_ids:
            for sid in self.selection.seam_ids:
                target_seams.append((sid, source_side))
        elif self.selection.edge_ranges:
            for er in self.selection.edge_ranges:
                for sid, seam in self.doc.seams.items():
                    # A側に一致するか判定
                    is_in_a = any(
                        s_er.piece_id == er.piece_id
                        and s_er.path_id == er.path_id
                        and {s_er.start_node_id, s_er.end_node_id} == {er.start_node_id, er.end_node_id}
                        for s_er in seam.effective_a()
                    )
                    if is_in_a:
                        target_seams.append((sid, "A"))
                        break
                    # B側に一致するか判定
                    is_in_b = any(
                        s_er.piece_id == er.piece_id
                        and s_er.path_id == er.path_id
                        and {s_er.start_node_id, s_er.end_node_id} == {er.start_node_id, er.end_node_id}
                        for s_er in seam.effective_b()
                    )
                    if is_in_b:
                        target_seams.append((sid, "B"))
                        break

        if not target_seams:
            logger.debug(f"[Pattern] match_selected_seam: no seams or connected edges found (edges={len(self.selection.edge_ranges)}, seams={len(self.selection.seam_ids)})")
            return False

        before_doc = self.doc.clone()
        modified = False
        for sid, side in target_seams:
            ok = self.doc.match_seam_shape_and_length(
                sid,
                source_side=side,
                mode=self.shape_match_mode,
                flip_curvature=self.shape_match_flip_curvature,
                apply_endpoint_offset=self.shape_match_apply_endpoint_offset,
                ease_value=self.shape_match_ease_value,
                ease_is_ratio=self.shape_match_ease_is_ratio,
            )
            logger.debug(f"[Pattern] match_seam_shape_and_length: sid={sid}, side={side} -> ok={ok}")
            if ok:
                modified = True
        if modified:
            self.status_message = trans("Seam shape matched")
            self.history.record_change(
                description="Match Seam Shape",
                before_doc=before_doc,
                after_doc=self.doc,
            )
            self.is_dirty = True
            logger.debug("[Pattern] match_selected_seam: SUCCESS")
            return True
        logger.debug("[Pattern] match_selected_seam: modified is False")
        return False
