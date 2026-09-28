"""
モーダル操作イベント判定ヘルパー (taremin_cloth.utils.modal_event)

Blenderモーダルオペレーターにおける左クリック押下 (PRESS) / 解放 (RELEASE) /
確定 (CONFIRM) / 取消 (CANCEL) の判定と、クリック / ドラッグの区別を一元化する。
bpy に依存しないため、Blender非起動の単体検証で利用できる。

座標空間の注意:
    ヘルパーは押下時と解放時で同一空間の (x, y) を受け取ることを前提とする。
    呼び出し側は window 相対 (mouse_x/y) か region 相対 (mouse_region_x/y) かを
    一つのジェスチャー内で統一すること。混在させた場合は距離が正しく求まらない。
"""

from typing import Any, Optional, Tuple

#: クリックとみなす移動距離の既定閾値 (px)。二乗距離で比較する。
CLICK_DRAG_THRESHOLD_PX: float = 5.0

#: 確定操作として扱うイベント種別
CONFIRM_TYPES = frozenset({'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'})

#: 取消操作として扱うイベント種別
CANCEL_TYPES = frozenset({'RIGHTMOUSE', 'ESC'})


def _event_str(event: Any, name: str) -> Optional[str]:
    try:
        value = getattr(event, name, None)
    except Exception:
        return None
    return value if isinstance(value, str) else None


def _event_num(event: Any, name: str) -> Optional[float]:
    try:
        value = getattr(event, name, None)
    except Exception:
        return None
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def event_pos(event: Any) -> Optional[Tuple[float, float]]:
    """イベントから (x, y) を取得する。window 相対を優先し、なければ region 相対を使う。"""
    if event is None:
        return None
    x = _event_num(event, "mouse_x")
    y = _event_num(event, "mouse_y")
    if x is not None and y is not None:
        return (x, y)
    rx = _event_num(event, "mouse_region_x")
    ry = _event_num(event, "mouse_region_y")
    if rx is not None and ry is not None:
        return (rx, ry)
    return None


def is_left_press(event: Any) -> bool:
    """左ボタン押下 (PRESS) の場合に True。"""
    return _event_str(event, "type") == 'LEFTMOUSE' and _event_str(event, "value") == 'PRESS'


def is_left_release(event: Any) -> bool:
    """左ボタン解放 (RELEASE) の場合に True。"""
    return _event_str(event, "type") == 'LEFTMOUSE' and _event_str(event, "value") == 'RELEASE'


def is_confirm(event: Any) -> bool:
    """確定操作 (LMB/Enter) の押下の場合に True。"""
    return _event_str(event, "type") in CONFIRM_TYPES and _event_str(event, "value") == 'PRESS'


def is_cancel(event: Any) -> bool:
    """取消操作 (RMB/Esc) の押下の場合に True。"""
    return _event_str(event, "type") in CANCEL_TYPES and _event_str(event, "value") == 'PRESS'


class PressDragTracker:
    """押下追跡によるクリック / ドラッグ識別器 (O(1))。

    使い方:
        tracker.on_press(x, y)      # PRESS時
        tracker.on_move(x, y)       # MOVE時。True=閾値超過 (ドラッグ中)
        tracker.on_release(x, y)    # RELEASE時。"click" | "drag" | None
    """

    def __init__(self, threshold_px: float = CLICK_DRAG_THRESHOLD_PX) -> None:
        try:
            threshold = float(threshold_px)
        except (TypeError, ValueError):
            threshold = CLICK_DRAG_THRESHOLD_PX
        if threshold < 0.0:
            threshold = 0.0
        self._threshold_sq: float = threshold * threshold
        self._pressed: bool = False
        self._dragging: bool = False
        self._press_x: float = 0.0
        self._press_y: float = 0.0

    @property
    def pressed(self) -> bool:
        """指が離れていない (PRESS済み・RELEASE未済み) 場合に True。"""
        return self._pressed

    @property
    def dragging(self) -> bool:
        """閾値を超過した移動を検出済みの場合に True。"""
        return self._dragging

    @property
    def press_pos(self) -> Optional[Tuple[float, float]]:
        """押下座標。未押下時は None。"""
        if not self._pressed:
            return None
        return (self._press_x, self._press_y)

    def reset(self) -> None:
        """追跡状態を破棄する。"""
        self._pressed = False
        self._dragging = False

    def on_press(self, x: Any, y: Any) -> bool:
        """押下座標を記録する。数値以外は (0.0, 0.0) 扱いとする。"""
        try:
            px, py = float(x), float(y)
        except (TypeError, ValueError):
            px, py = 0.0, 0.0
        self._press_x, self._press_y = px, py
        self._pressed = True
        self._dragging = False
        return True

    def _dist_sq(self, x: float, y: float) -> float:
        dx = x - self._press_x
        dy = y - self._press_y
        return dx * dx + dy * dy

    def on_move(self, x: Any, y: Any) -> bool:
        """移動で閾値を超過したら True。未押下時は常に False。"""
        if not self._pressed:
            return False
        try:
            fx, fy = float(x), float(y)
        except (TypeError, ValueError):
            return self._dragging
        if self._dist_sq(fx, fy) > self._threshold_sq:
            self._dragging = True
        return self._dragging

    def on_release(self, x: Any, y: Any) -> Optional[str]:
        """解放時に "click" / "drag" を返す。未押下時は None。呼出後に状態を破棄する。"""
        if not self._pressed:
            return None
        try:
            fx, fy = float(x), float(y)
        except (TypeError, ValueError):
            result = "drag" if self._dragging else "click"
            self.reset()
            return result
        if self._dragging or self._dist_sq(fx, fy) > self._threshold_sq:
            result = "drag"
        else:
            result = "click"
        self.reset()
        return result

    def on_press_event(self, event: Any) -> bool:
        """イベントから座標を抽出して on_press する。座標不明時は (0.0, 0.0) 扱いとする。"""
        pos = event_pos(event)
        if pos is None:
            return self.on_press(0.0, 0.0)
        return self.on_press(pos[0], pos[1])

    def on_move_event(self, event: Any) -> bool:
        """イベントから座標を抽出して on_move する。座標不明時は現状を維持する。"""
        pos = event_pos(event)
        if pos is None:
            return self._dragging if self._pressed else False
        return self.on_move(pos[0], pos[1])

    def on_release_event(self, event: Any) -> Optional[str]:
        """イベントから座標を抽出して on_release する。"""
        pos = event_pos(event)
        if pos is None:
            return self.on_release(self._press_x, self._press_y)
        return self.on_release(pos[0], pos[1])
