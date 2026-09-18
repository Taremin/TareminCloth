"""
パネル描画テスト用ユーティリティ (panel_test_utils)
Blenderのバックグラウンド実行（headless）環境でも、安全にPanel.drawメソッドを
テストし、レイアウト構築・プロパティ参照・オペレーター参照の妥当性を検証します。
"""

import unittest
from typing import Any, Dict, List, Optional, Tuple

try:
    import bpy
except ImportError:
    bpy = None


class MockOperatorProperties:
    """オペレータープロパティ設定をエミュレート・記録するクラス"""

    def __init__(
        self,
        idname: str,
        text: str = "",
        icon: str = "",
        allowed_props: Optional[List[str]] = None,
        strict: bool = True,
    ):
        self._idname = idname
        self._text = text
        self._icon = icon
        self._allowed_props = set(allowed_props) if allowed_props is not None else None
        self._strict = strict
        self._props: Dict[str, Any] = {}

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            if self._strict and self._allowed_props is not None:
                if name not in self._allowed_props:
                    raise AttributeError(
                        f"[MockOperatorProperties] オペレーター '{self._idname}' にプロパティ '{name}' は定義されていません"
                    )
            self._props[name] = value

    def __getattr__(self, name: str) -> Any:
        if name in self._props:
            return self._props[name]
        raise AttributeError(f"MockOperatorProperties '{self._idname}' has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"<MockOperatorProperties idname='{self._idname}' props={self._props}>"


class MockLayoutItem:
    """レイアウトツリーの単一要素"""

    def __init__(self, item_type: str, details: Dict[str, Any]):
        self.item_type = item_type
        self.details = details

    def __repr__(self) -> str:
        return f"<MockLayoutItem {self.item_type}: {self.details}>"


class MockLayout:
    """
    bpy.types.UILayout のエミュレータ。
    バックグラウンド環境で安全に UI 構築をシミュレートし、
    プロパティ名ミスや未登録オペレーターの検知を行います。
    """

    def __init__(
        self,
        layout_type: str = "root",
        parent: Optional["MockLayout"] = None,
        strict_props: bool = True,
        strict_operators: bool = True,
        strict_menus: bool = True,
    ):
        self.layout_type = layout_type
        self.parent = parent
        self.strict_props = strict_props
        self.strict_operators = strict_operators
        self.strict_menus = strict_menus

        # レイアウト状態
        self.enabled = True
        self.active = True
        self.alert = False
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.alignment = 'EXPAND'

        # 子要素リスト
        self.items: List[Any] = []

    # -------------------------------------------------------------------------
    # コンテナ・階層メソッド
    # -------------------------------------------------------------------------

    def row(self, align: bool = False) -> "MockLayout":
        sub = MockLayout(
            layout_type="row",
            parent=self,
            strict_props=self.strict_props,
            strict_operators=self.strict_operators,
            strict_menus=self.strict_menus,
        )
        self.items.append(sub)
        return sub

    def column(self, align: bool = False) -> "MockLayout":
        sub = MockLayout(
            layout_type="column",
            parent=self,
            strict_props=self.strict_props,
            strict_operators=self.strict_operators,
            strict_menus=self.strict_menus,
        )
        self.items.append(sub)
        return sub

    def box(self) -> "MockLayout":
        sub = MockLayout(
            layout_type="box",
            parent=self,
            strict_props=self.strict_props,
            strict_operators=self.strict_operators,
            strict_menus=self.strict_menus,
        )
        self.items.append(sub)
        return sub

    def split(self, factor: float = 0.0, align: bool = False) -> "MockLayout":
        sub = MockLayout(
            layout_type="split",
            parent=self,
            strict_props=self.strict_props,
            strict_operators=self.strict_operators,
            strict_menus=self.strict_menus,
        )
        self.items.append(sub)
        return sub

    # -------------------------------------------------------------------------
    # UI要素構築メソッド
    # -------------------------------------------------------------------------

    def prop(
        self,
        data: Any,
        property: str,
        text: str = "",
        icon: str = 'NONE',
        expand: bool = False,
        slider: bool = False,
        toggle: bool = False,
        icon_only: bool = False,
        event: bool = False,
        full_event: bool = False,
        emboss: bool = True,
        index: int = -1,
        invert_checkbox: bool = False,
    ):
        """プロパティUIウィジェットの追加"""
        if self.strict_props:
            if data is None:
                raise ValueError(f"[MockLayout] prop() に None データが渡されました (property='{property}')")
            if not hasattr(data, property):
                raise AttributeError(
                    f"[MockLayout] オブジェクト '{data}' にプロパティ '{property}' が存在しません"
                )

        item = MockLayoutItem(
            "prop",
            {
                "data": data,
                "property": property,
                "text": text,
                "icon": icon,
            },
        )
        self.items.append(item)

    def operator(
        self,
        operator: str,
        text: str = "",
        text_ctxt: str = "",
        translate: bool = True,
        icon: str = 'NONE',
        emboss: bool = True,
        depress: bool = False,
        icon_value: int = 0,
    ) -> Optional[MockOperatorProperties]:
        """オペレーターボタンの追加"""
        # オペレーターの登録状態を検証
        allowed_props = None
        if self.strict_operators:
            parts = operator.split(".", 1)
            if len(parts) == 2:
                module_name, op_name = parts
                mod = getattr(bpy.ops, module_name, None)
                is_valid = False
                if mod is not None and hasattr(mod, op_name):
                    try:
                        op_func = getattr(mod, op_name)
                        if hasattr(op_func, "get_rna_type"):
                            rna_type = op_func.get_rna_type()
                            allowed_props = list(rna_type.properties.keys())
                            is_valid = True
                        else:
                            is_valid = op_name in dir(mod)
                    except (KeyError, AttributeError):
                        is_valid = False
                if not is_valid:
                    raise NameError(
                        f"[MockLayout] 未登録のオペレーターが呼び出されました: '{operator}'"
                    )
            else:
                raise ValueError(f"[MockLayout] 不正なオペレーター識別名です: '{operator}'")

        op_props = MockOperatorProperties(
            operator,
            text=text,
            icon=icon,
            allowed_props=allowed_props,
            strict=self.strict_props,
        )
        item = MockLayoutItem(
            "operator",
            {
                "operator": operator,
                "text": text,
                "icon": icon,
                "properties": op_props,
            },
        )
        self.items.append(item)
        return op_props

    def menu(
        self,
        menu: str,
        text: str = "",
        text_ctxt: str = "",
        translate: bool = True,
        icon: str = 'NONE',
        icon_value: int = 0,
    ):
        """メニュー呼び出しボタンの追加"""
        if self.strict_menus:
            if not hasattr(bpy.types, menu):
                raise NameError(f"[MockLayout] 未登録のメニューが呼び出されました: '{menu}'")

        item = MockLayoutItem(
            "menu",
            {
                "menu": menu,
                "text": text,
                "icon": icon,
            },
        )
        self.items.append(item)

    def label(
        self,
        text: str = "",
        text_ctxt: str = "",
        translate: bool = True,
        icon: str = 'NONE',
        icon_value: int = 0,
    ):
        """ラベルの追加"""
        item = MockLayoutItem(
            "label",
            {
                "text": text,
                "icon": icon,
            },
        )
        self.items.append(item)

    def separator(self, factor: float = 1.0):
        """セパレータの追加"""
        item = MockLayoutItem("separator", {"factor": factor})
        self.items.append(item)

    def template_list(
        self,
        listtype_name: str,
        list_id: str,
        dataptr: Any,
        propname: str,
        active_dataptr: Any,
        active_propname: str,
        rows: int = 5,
        maxrows: int = 5,
        type: str = 'DEFAULT',
        columns: int = 9,
    ):
        """UIListテンプレートの追加"""
        if self.strict_props:
            if dataptr is not None and not hasattr(dataptr, propname):
                raise AttributeError(
                    f"[MockLayout] template_list のデータポインタにプロパティ '{propname}' が存在しません"
                )
            if active_dataptr is not None and not hasattr(active_dataptr, active_propname):
                raise AttributeError(
                    f"[MockLayout] template_list のアクティブデータポインタにプロパティ '{active_propname}' が存在しません"
                )

        item = MockLayoutItem(
            "template_list",
            {
                "listtype_name": listtype_name,
                "list_id": list_id,
                "dataptr": dataptr,
                "propname": propname,
                "active_dataptr": active_dataptr,
                "active_propname": active_propname,
            },
        )
        self.items.append(item)

    def prop_search(
        self,
        data: Any,
        property: str,
        search_data: Any,
        search_property: str,
        text: str = "",
        text_ctxt: str = "",
        translate: bool = True,
        icon: str = 'NONE',
    ):
        """prop_searchウィジェットの追加"""
        if self.strict_props:
            if not hasattr(data, property):
                raise AttributeError(
                    f"[MockLayout] prop_search: データに '{property}' が存在しません"
                )
            if not hasattr(search_data, search_property):
                raise AttributeError(
                    f"[MockLayout] prop_search: 検索データに '{search_property}' が存在しません"
                )

        item = MockLayoutItem(
            "prop_search",
            {
                "data": data,
                "property": property,
                "search_property": search_property,
            },
        )
        self.items.append(item)

    # -------------------------------------------------------------------------
    # 走査・検証ヘルパー
    # -------------------------------------------------------------------------

    def get_all_items(self) -> List[MockLayoutItem]:
        """再帰的に全ての子アイテムをフラットリストとして取得"""
        result = []
        for item in self.items:
            if isinstance(item, MockLayout):
                result.extend(item.get_all_items())
            elif isinstance(item, MockLayoutItem):
                result.append(item)
        return result

    def get_operators(self) -> List[str]:
        """描画されたすべてのオペレーターの idname 一覧を取得"""
        return [
            item.details["operator"]
            for item in self.get_all_items()
            if item.item_type == "operator"
        ]

    def get_props(self) -> List[Tuple[Any, str]]:
        """描画されたすべての (data, property) タプル一覧を取得"""
        return [
            (item.details["data"], item.details["property"])
            for item in self.get_all_items()
            if item.item_type == "prop"
        ]

    def get_labels(self) -> List[str]:
        """描画されたすべてのラベルテキスト一覧を取得"""
        return [
            item.details["text"]
            for item in self.get_all_items()
            if item.item_type == "label"
        ]

    def get_menus(self) -> List[str]:
        """描画されたすべてのメニュー idname 一覧を取得"""
        return [
            item.details["menu"]
            for item in self.get_all_items()
            if item.item_type == "menu"
        ]

    def find_operator_props(self, idname: str) -> Optional[MockOperatorProperties]:
        """指定したオペレーターのプロパティオブジェクトを検索"""
        for item in self.get_all_items():
            if item.item_type == "operator" and item.details["operator"] == idname:
                return item.details.get("properties")
        return None


class _DummyPanelInstance:
    """drawメソッド呼び出し用のダミーパネルインスタンス"""

    def __init__(self, layout: MockLayout):
        self.layout = layout


def render_panel(
    panel_class: type,
    context: Optional[Any] = None,
    layout: Optional[MockLayout] = None,
    strict: bool = True,
) -> MockLayout:
    """
    パネルクラスの draw メソッドを安全に呼び出してテストを行うユーティリティ関数。

    Args:
        panel_class: テスト対象の Panel クラス (例: TAREMIN_CLOTH_PT_collider_panel)
        context: Blenderコンテキスト (未指定時は bpy.context)
        layout: 既存の MockLayout (未指定時は自動生成)
        strict: True の場合、未登録のオペレーターや存在しないプロパティの参照で例外を発生させる

    Returns:
        描画結果が記録された MockLayout
    """
    if bpy is None:
        raise unittest.SkipTest("[panel_test_utils] Blender (bpy) が利用できない環境のためスキップします")

    if context is None:
        context = bpy.context

    if layout is None:
        layout = MockLayout(
            strict_props=strict,
            strict_operators=strict,
            strict_menus=strict,
        )

    dummy_panel = _DummyPanelInstance(layout)

    try:
        panel_class.draw(dummy_panel, context)
    except Exception as e:
        raise RuntimeError(
            f"[render_panel] '{panel_class.__name__}.draw()' の実行中にエラーが発生しました: {e}"
        ) from e

    return layout
