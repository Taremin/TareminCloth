"""
Taremin Cloth 国際化（i18n）完全性検証テストスイート
- Blender非依存の辞書整合性テスト
- 静的RNAプロパティ・Enum選択肢・オペレーター・パネル網羅テスト（未翻訳検知）
- デッドキー（未使用翻訳）検知テスト
- Blender環境下での動的UI描画インターセプト（MockLayout）およびpgettext_iface翻訳解決検証
"""

import ast
import inspect
import os
import sys
import types
import unittest

# プロジェクトルートおよび python/ ディレクトリをモジュール検索パスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
python_dir = os.path.join(project_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from taremin_cloth import i18n, panels, properties, preferences, ops, presets

try:
    import bpy
    HAS_BPY = True
    IS_REAL_BLENDER = (
        hasattr(bpy, "context")
        and bpy.context is not None
        and hasattr(bpy.context, "preferences")
        and bpy.context.preferences is not None
    )
except ImportError:
    HAS_BPY = False
    IS_REAL_BLENDER = False


class MockOperatorProps:
    """オペレータープロパティ代入を受け付けるモック"""
    def __setattr__(self, name, value):
        pass


class MockLayout:
    """bpy.types.UILayout をエミュレートしてUI描画テキストを収集するモック"""

    def __init__(self, context_name=None):
        self.context_name = context_name or i18n.CONTEXT
        self.collected = []  # list of (context, text, source)
        self.alert = False
        self.enabled = True
        self.active = True
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.alignment = 'EXPAND'
        self.use_property_split = False
        self.use_property_decorate = False

    def row(self, *args, **kwargs):
        return self

    def column(self, *args, **kwargs):
        return self

    def column_flow(self, *args, **kwargs):
        return self

    def box(self, *args, **kwargs):
        return self

    def split(self, *args, **kwargs):
        return self

    def grid_flow(self, *args, **kwargs):
        return self

    def separator(self, *args, **kwargs):
        pass

    def separator_spacer(self, *args, **kwargs):
        pass

    def label(self, text="", text_ctxt="", icon="NONE", icon_value=0):
        ctxt = text_ctxt or self.context_name
        if text:
            self.collected.append((ctxt, text, "label"))

    def prop(self, data, property, text=None, text_ctxt="", icon="NONE", expand=False, slider=False, toggle=False, icon_only=False, event=False, full_event=False, emboss=True, index=-1, icon_value=0):
        ctxt = text_ctxt or self.context_name
        if text is not None:
            if text:
                self.collected.append((ctxt, text, "prop_custom_text"))
        else:
            if hasattr(data, "rna_type"):
                prop_def = data.rna_type.properties.get(property)
                if prop_def and prop_def.name:
                    self.collected.append((ctxt, prop_def.name, "prop_name"))

    def props_enum(self, data, property):
        pass

    def prop_menu_enum(self, data, property, text=None, text_ctxt="", icon="NONE"):
        ctxt = text_ctxt or self.context_name
        if text:
            self.collected.append((ctxt, text, "prop_menu_enum"))

    def prop_search(self, data, property, search_data, search_property, text=None, text_ctxt="", icon="NONE"):
        ctxt = text_ctxt or self.context_name
        if text:
            self.collected.append((ctxt, text, "prop_search"))

    def operator(self, operator, text=None, text_ctxt="", icon="NONE", emboss=True, icon_value=0, depress=False):
        ctxt = text_ctxt or self.context_name
        if text:
            self.collected.append((ctxt, text, "operator_custom_text"))
        return MockOperatorProps()

    def operator_menu_enum(self, operator, property, text=None, text_ctxt="", icon="NONE"):
        ctxt = text_ctxt or self.context_name
        if text:
            self.collected.append((ctxt, text, "operator_menu_enum"))
        return MockOperatorProps()

    def menu(self, menu, text=None, text_ctxt="", icon="NONE", icon_value=0):
        ctxt = text_ctxt or self.context_name
        if text:
            self.collected.append((ctxt, text, "menu"))

    def template_list(self, *args, **kwargs):
        pass


class TestI18nDictionary(unittest.TestCase):
    """Blender非依存の辞書整合性テスト"""

    def test_context_name(self):
        """デフォルト翻訳コンテキスト名がTareminClothであることを確認"""
        self.assertEqual(i18n.CONTEXT, "TareminCloth")

    def test_translations_dict_structure(self):
        """翻訳辞書の構造検証"""
        self.assertIn("ja_JP", i18n.TRANSLATIONS_DICT)
        self.assertIn("en_US", i18n.TRANSLATIONS_DICT)

        ja_dict = i18n.TRANSLATIONS_DICT["ja_JP"]
        self.assertGreater(len(ja_dict), 100, "日本語翻訳エントリが十分に登録されていること")

        for key, value in ja_dict.items():
            self.assertIsInstance(key, tuple, f"キーはタプルである必要があります: {key}")
            self.assertEqual(len(key), 2, f"キーは(context, msgid)の2要素である必要があります: {key}")
            self.assertIsInstance(key[0], str)
            self.assertIsInstance(key[1], str)
            self.assertIsInstance(value, str, f"値は文字列である必要があります: {value}")
            self.assertTrue(len(value.strip()) > 0, f"空の翻訳値が存在します: {key}")

    def test_trans_fallback_without_bpy(self):
        """bpyが無い環境でのtrans()関数のフォールバック動作"""
        if not HAS_BPY:
            result = i18n.trans("Physical Properties")
            self.assertEqual(result, "Physical Properties")

    def test_registered_dict_purity(self):
        """Blender登録辞書がアドオン固有コンテキスト'TareminCloth'単一に純化され、'*'や'Operator'を一切汚染しないことを検証"""
        reg_dict = i18n._build_registered_dict()
        self.assertIn("ja_JP", reg_dict)
        ja_entries = reg_dict["ja_JP"]

        star_entries = [k for k in ja_entries.keys() if k[0] == "*"]
        operator_entries = [k for k in ja_entries.keys() if k[0] == "Operator"]
        taremin_entries = [k for k in ja_entries.keys() if k[0] == i18n.CONTEXT]

        self.assertEqual(
            len(star_entries), 0,
            f"一般コンテキスト '*' に登録されているエントリが存在します ({len(star_entries)}件): {star_entries[:5]}"
        )
        self.assertEqual(
            len(operator_entries), 0,
            f"Blender標準コンテキスト 'Operator' に登録されているエントリが存在します ({len(operator_entries)}件): {operator_entries[:5]}"
        )
        self.assertGreater(
            len(taremin_entries), 100,
            f"TareminCloth コンテキストに十分なエントリが登録されていること ({len(taremin_entries)}件)"
        )

    def test_json_files_syntax(self):
        """translations/*.json ファイルが有効なJSONであることを検証"""
        raw_dict = i18n.get_raw_translations()
        self.assertIn("en_US", raw_dict, "英語マスター en_US.json が読み込まれていること")
        self.assertIn("ja_JP", raw_dict, "日本語辞書 ja_JP.json が読み込まれていること")
        for locale, entries in raw_dict.items():
            self.assertIsInstance(entries, dict, f"{locale} の辞書が辞書型であること")
            self.assertGreater(len(entries), 100, f"{locale} のエントリ数が十分であること")

    def test_en_us_master_consistency(self):
        """英語マスター en_US.json と日本語 ja_JP.json のキーが 1:1 で完全一致していることを検証"""
        raw_dict = i18n.get_raw_translations()
        en_keys = set(raw_dict.get("en_US", {}).keys())
        ja_keys = set(raw_dict.get("ja_JP", {}).keys())

        missing_in_ja = en_keys - ja_keys
        extra_in_ja = ja_keys - en_keys

        self.assertEqual(
            len(missing_in_ja), 0,
            f"英語マスターにあるが日本語訳に存在しないキーが {len(missing_in_ja)} 件あります:\n" +
            "\n".join(f"  {k!r}" for k in sorted(missing_in_ja)[:20])
        )
        self.assertEqual(
            len(extra_in_ja), 0,
            f"日本語訳にあるが英語マスターに存在しないキーが {len(extra_in_ja)} 件あります:\n" +
            "\n".join(f"  {k!r}" for k in sorted(extra_in_ja)[:20])
        )


class TestI18nRNACoverage(unittest.TestCase):
    """静的RNAプロパティ・Enum選択肢・オペレーター・パネルの辞書網羅テスト"""

    def _collect_all_static_terms(self):
        """アドオン定義の全静的用語を収集"""
        terms = set()

        # 1. パネル bl_label
        panel_classes = [
            cls for _, cls in inspect.getmembers(panels, inspect.isclass)
            if hasattr(cls, "bl_label")
        ]
        for p_cls in panel_classes:
            ctx = getattr(p_cls, "bl_translation_context", None) or "*"
            lbl = getattr(p_cls, "bl_label", None)
            if lbl and isinstance(lbl, str):
                terms.add((ctx, lbl.strip()))

        # 2. RNA プロパティ & Enum選択肢
        rna_classes = [
            properties.TareminClothElasticGroup,
            properties.TareminClothObjectSettings,
            properties.TareminClothColliderAnimSettings,
            properties.TareminClothColliderSettings,
            preferences.TareminClothPreferences,
        ]
        for rna_cls in rna_classes:
            if hasattr(rna_cls, "__annotations__"):
                for prop_name, prop_def in rna_cls.__annotations__.items():
                    kw = getattr(prop_def, "keywords", {})
                    name = kw.get("name")
                    desc = kw.get("description")
                    if name and isinstance(name, str):
                        terms.add(("*", name.strip()))
                    if desc and isinstance(desc, str):
                        terms.add(("*", desc.strip()))
                    items = kw.get("items")
                    if items and isinstance(items, (list, tuple)):
                        for itm in items:
                            if isinstance(itm, (list, tuple)) and len(itm) >= 3:
                                if itm[1] and isinstance(itm[1], str):
                                    terms.add(("*", itm[1].strip()))
                                if itm[2] and isinstance(itm[2], str):
                                    terms.add(("*", itm[2].strip()))

        # 3. オペレーター & メニュー (ops, presets)
        op_modules = [ops.basic, ops.gui, ops.interactive, ops.pose, ops.tools, presets]
        for mod in op_modules:
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                is_op = (
                    (issubclass(cls, (bpy.types.Operator, bpy.types.Menu)) if HAS_BPY else getattr(cls, "bl_idname", None))
                    if isinstance(cls, type) else False
                )
                if is_op:
                    lbl = getattr(cls, "bl_label", None)
                    desc = getattr(cls, "bl_description", None)
                    if lbl and isinstance(lbl, str):
                        terms.add(("*", lbl.strip()))
                    if desc and isinstance(desc, str):
                        terms.add(("*", desc.strip()))
                    if hasattr(cls, "__annotations__"):
                        for prop_name, prop_def in cls.__annotations__.items():
                            kw = getattr(prop_def, "keywords", {})
                            p_name = kw.get("name")
                            p_desc = kw.get("description")
                            if p_name and isinstance(p_name, str):
                                terms.add(("*", p_name.strip()))
                            if p_desc and isinstance(p_desc, str):
                                terms.add(("*", p_desc.strip()))
                            items = kw.get("items")
                            if items and isinstance(items, (list, tuple)):
                                for itm in items:
                                    if isinstance(itm, (list, tuple)) and len(itm) >= 3:
                                        if itm[1] and isinstance(itm[1], str):
                                            terms.add(("*", itm[1].strip()))
                                        if itm[2] and isinstance(itm[2], str):
                                            terms.add(("*", itm[2].strip()))

        return terms

    def test_all_rna_terms_are_translated(self):
        """全RNAプロパティ・オペレーターが辞書に漏れなく登録されていることを検証"""
        static_terms = self._collect_all_static_terms()
        ja_dict = i18n.TRANSLATIONS_DICT["ja_JP"]

        missing = []
        for ctxt, text in sorted(static_terms):
            if not text:
                continue
            # 専用コンテキストまたは一般コンテキストで辞書に存在するか
            if (ctxt, text) not in ja_dict and ("*", text) not in ja_dict:
                missing.append((ctxt, text))

        self.assertEqual(
            len(missing), 0,
            f"辞書に未登録の用語が {len(missing)} 件あります:\n" +
            "\n".join(f"  ({c!r}, {t!r})" for c, t in missing[:20])
        )

    def test_all_operator_classes_have_translation_context(self):
        """全オペレータークラス・メニュークラスがアドオン固有コンテキスト bl_translation_context を保持していることを検証"""
        all_classes = list(ops.OPERATOR_CLASSES)
        # presets モジュール内のオペレーター・メニュー
        for _, cls in inspect.getmembers(presets, inspect.isclass):
            if cls.__name__.startswith("TAREMIN_CLOTH_") and (issubclass(cls, (bpy.types.Operator, bpy.types.Menu)) if HAS_BPY else getattr(cls, "bl_idname", None)):
                if cls not in all_classes:
                    all_classes.append(cls)

        missing_context = []
        for cls in all_classes:
            t_ctx = getattr(cls, "bl_translation_context", None)
            if t_ctx != i18n.CONTEXT:
                missing_context.append((cls.__name__, t_ctx))

        self.assertEqual(
            len(missing_context), 0,
            f"bl_translation_context が正しく設定されていないクラスが {len(missing_context)} 件あります:\n" +
            "\n".join(f"  {name}: {ctx!r} (期待値: {i18n.CONTEXT!r})" for name, ctx in missing_context)
        )


class TestI18nDeadKeys(unittest.TestCase):
    """辞書内のデッドキー（未使用の孤立翻訳エントリ）検知テスト"""

    def test_no_dead_keys_in_dictionary(self):
        """辞書内の全エントリがアドオンコード内で実際に使用されていることを検証"""
        addon_dir = os.path.abspath(os.path.join(project_root, "python", "taremin_cloth"))
        code_strings = set()

        for root, _, files in os.walk(addon_dir):
            for f in files:
                if f.endswith(".py") and f != "i18n.py":
                    filepath = os.path.join(root, f)
                    with open(filepath, "r", encoding="utf-8") as fp:
                        try:
                            tree = ast.parse(fp.read(), filename=filepath)
                            for node in ast.walk(tree):
                                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                                    code_strings.add(node.value.strip())
                        except Exception as e:
                            self.fail(f"ASTパースエラー: {filepath}: {e}")

        ja_dict = i18n.TRANSLATIONS_DICT["ja_JP"]
        dead_keys = []
        for (ctxt, msgid), msgstr in ja_dict.items():
            if msgid.strip() not in code_strings:
                dead_keys.append((ctxt, msgid))

        self.assertEqual(
            len(dead_keys), 0,
            f"コードベースに存在しないデッドキーが {len(dead_keys)} 件あります:\n" +
            "\n".join(f"  ({c!r}, {m!r})" for c, m in dead_keys[:20])
        )


@unittest.skipUnless(IS_REAL_BLENDER, "Blender (bpy) 環境でのみ実行される実利用突合テスト")
class TestI18nStrictBlenderUsage(unittest.TestCase):
    """Blender環境下で全RNA/UI定義から抽出した用語集合と辞書を突合し、デッドキーが0件であることを厳密検証"""

    @classmethod
    def setUpClass(cls):
        import taremin_cloth
        taremin_cloth.register()

    def test_all_dictionary_entries_are_strictly_used_by_blender(self):
        """辞書の全エントリがBlenderのUI/RNAで100%実際に使われていることを厳密に検証"""
        all_terms = set()

        def record(term):
            if term and isinstance(term, str):
                t = term.strip()
                if t:
                    all_terms.add(t)

        # 1. 全登録クラス (Panel, Operator, Menu) の bl_label, bl_description
        for mod in [panels, presets, preferences, ops.basic, ops.gui, ops.interactive, ops.pose, ops.tools]:
            for name, cls in inspect.getmembers(mod, inspect.isclass):
                if issubclass(cls, (bpy.types.Panel, bpy.types.Operator, bpy.types.Menu)):
                    lbl = getattr(cls, "bl_label", None)
                    if lbl:
                        record(lbl)
                    desc = getattr(cls, "bl_description", None)
                    if desc:
                        record(desc)

        # 2. 全オペレータープロパティ (bpy.ops.taremin_cloth)
        if hasattr(bpy.ops, "taremin_cloth"):
            for op_name in dir(bpy.ops.taremin_cloth):
                if op_name.startswith("_"):
                    continue
                op_func = getattr(bpy.ops.taremin_cloth, op_name)
                if hasattr(op_func, "get_rna_type"):
                    op_rna = op_func.get_rna_type()
                    for prop in op_rna.properties:
                        if prop.identifier == "rna_type":
                            continue
                        record(prop.name)
                        record(prop.description)
                        if prop.type == 'ENUM':
                            for itm in prop.enum_items:
                                record(itm.name)
                                record(itm.description)

        # 3. 全PropertyGroup
        for p_cls in [
            properties.TareminClothObjectSettings,
            properties.TareminClothColliderSettings,
            properties.TareminClothColliderAnimSettings,
            properties.TareminClothElasticGroup,
            preferences.TareminClothPreferences,
        ]:
            if hasattr(p_cls, "bl_rna"):
                for prop in p_cls.bl_rna.properties:
                    if prop.identifier == "rna_type":
                        continue
                    record(prop.name)
                    record(prop.description)
                    if prop.type == 'ENUM':
                        for itm in prop.enum_items:
                            record(itm.name)
                            record(itm.description)

        # 4. Scene に動的追加されたプロパティ
        for prop_id in ["taremin_cloth_fast_playback", "taremin_cloth_ui_mode", "taremin_cloth_active_config_preset", "taremin_cloth_config_presets_json"]:
            prop = bpy.types.Scene.bl_rna.properties.get(prop_id)
            if prop:
                record(prop.name)
                record(prop.description)
                if prop.type == 'ENUM':
                    for itm in prop.enum_items:
                        record(itm.name)
                        record(itm.description)

        # 5. 動的Enum
        dummy_pref = types.SimpleNamespace(gpu_backend='AUTO')
        try:
            items = preferences._get_gpu_device_items(dummy_pref, bpy.context)
            for itm in items:
                record(itm[1])
                record(itm[2])
        except Exception:
            pass

        # 6. panels.py およびアドオンコード内のUIテキスト・i18n.trans() 引数
        addon_py_dir = os.path.join(project_root, "python", "taremin_cloth")
        for root, _, files in os.walk(addon_py_dir):
            for fname in files:
                if not fname.endswith(".py") or fname == "i18n.py":
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read(), filename=fpath)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            for kw in node.keywords:
                                if kw.arg == "text" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                    record(kw.value.value)
                            if isinstance(node.func, ast.Attribute):
                                if node.func.attr == "label" and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                                    record(node.args[0].value)
                                elif node.func.attr in ("trans", "pgettext", "pgettext_iface") and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                                    record(node.args[0].value)
                            elif isinstance(node.func, ast.Name):
                                if node.func.id == "trans" and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                                    record(node.args[0].value)
                except Exception:
                    pass

        # 7. panels.py 内の動的フォールバックタイトル
        record("Config Preset")
        record("Material Preset")
        record("Quality Preset")
        record("Bone SDF (Character Body)")
        record("Mesh SDF (Mannequin/Rigid)")
        record("Plane (Floor/Ground)")
        record("Sphere")
        record("Capsule")
        record("Mesh (Simple)")
        record("Taremin Cloth")

        # 8. インタラクティブモード実行時メッセージ (ステータスバー、ピン操作、開始/停止/ベンチマーク)
        record("Taremin Cloth: [Left Drag] Move Vertex | [P] Toggle Pin | [Right Click / ESC] Exit")
        record("Unpinned vertex #%d")
        record("Pinned vertex #%d")
        record("Interactive Simulation Started (Press ESC / RightClick or Click Stop to exit)")
        record("Interactive Simulation Stopped (Paused)")
        record("Benchmarking FPS... (will complete in ~2 seconds)")
        record("Debug recording saved: %s (%d frames)")

        # 辞書との突合
        ja_dict = i18n.TRANSLATIONS_DICT["ja_JP"]
        dict_keys = set(msgid.strip() for (ctxt, msgid) in ja_dict.keys())

        unused_keys = dict_keys - all_terms
        self.assertEqual(
            len(unused_keys), 0,
            f"辞書に登録されていますが、BlenderのUI/RNAで実際に使われていないエントリが {len(unused_keys)} 件あります:\n" +
            "\n".join(f"  {k!r} -> {ja_dict.get(('*', k))!r}" for k in sorted(unused_keys))
        )


@unittest.skipUnless(IS_REAL_BLENDER, "Blender (bpy) 環境でのみ実行される結合テスト")
class TestI18nDynamicUIDraw(unittest.TestCase):
    """全パネルの draw() をモックLayoutで実行し、UI描画テキストをインターセプト・翻訳検証"""

    @classmethod
    def setUpClass(cls):
        import taremin_cloth
        taremin_cloth.register()
        # テスト用オブジェクトの作成
        cls.mesh = bpy.data.meshes.new("TestI18nMesh")
        cls.obj = bpy.data.objects.new("TestI18nClothObj", cls.mesh)
        bpy.context.collection.objects.link(cls.obj)
        bpy.context.view_layer.objects.active = cls.obj
        cls.obj.taremin_cloth.is_cloth = True
        cls.obj.taremin_cloth_collider.is_collider = True

    @classmethod
    def tearDownClass(cls):
        import taremin_cloth
        if hasattr(cls, "obj") and cls.obj and cls.obj.name in bpy.data.objects:
            bpy.data.objects.remove(cls.obj)
        if hasattr(cls, "mesh") and cls.mesh and cls.mesh.name in bpy.data.meshes:
            bpy.data.meshes.remove(cls.mesh)
        taremin_cloth.unregister()

    def test_dynamic_ui_draw_interception_and_translation(self):
        """全パネルのdraw()を実行し、露出する全テキストが日本語翻訳されることを検証"""
        pref_view = bpy.context.preferences.view
        orig_lang = getattr(pref_view, "language", "DEFAULT")
        orig_trans = getattr(pref_view, "use_translate_interface", True)

        def contains_japanese(s: str) -> bool:
            """文字列に日本語（ひらがな、カタカナ、漢字、全角文字）が含まれるか判定"""
            for ch in s:
                cp = ord(ch)
                if (
                    0x3040 <= cp <= 0x309F  # ひらがな
                    or 0x30A0 <= cp <= 0x30FF  # カタカナ
                    or 0x4E00 <= cp <= 0x9FFF  # 漢字
                    or 0x3400 <= cp <= 0x4DBF  # CJK統合漢字拡張A
                    or 0xFF00 <= cp <= 0xFFEF  # 全角ASCII・半角カタカナ
                ):
                    return True
            return False

        try:
            pref_view.language = "ja_JP"
            pref_view.use_translate_interface = True

            panel_classes = [
                cls for _, cls in inspect.getmembers(panels, inspect.isclass)
                if issubclass(cls, bpy.types.Panel) and hasattr(cls, "draw")
            ]

            dynamic_prefixes = (
                "Active:",
                "Cloth オブジェクト (",
                "Collider オブジェクト (",
                "選択中:",
                "タイプ:",
                "シーン重力:",
                "ボーン数(",
                "SDFテクスチャ:",
                "解像度:",
                "上限(",
                "自動最適化:",
                "未適用のスケール:",
                "● 接続中",
                "素材:",
                "品質:",
                "コライダー:",
                "アーマチュア:",
                "登録エッジ数:",
                "Bones:",
                "FPS:",
                "Memory:",
                "Step:",
            )

            all_collected_texts = []

            # 状態網羅マトリクス
            ui_states = [
                # 1. オブジェクト未選択
                {"active": False},
                # 2. 布・コライダー無効（Enable Cloth, Enable Collider ボタン）
                {"active": True, "is_cloth": False, "is_col": False, "mode": 'SIMPLE'},
                # 3. 布有効・簡単モード（縫合・自己衝突も有効）
                {"active": True, "is_cloth": True, "is_col": False, "mode": 'SIMPLE', "sewing": True, "self_col": True},
                # 4. 布有効・詳細モード（全サブパネル展開状態）
                {"active": True, "is_cloth": True, "is_col": False, "mode": 'ADVANCED', "sewing": True, "self_col": True, "adaptive": True, "buffering": True, "fps": True},
                # 5. コライダー有効・簡単モード（SPHERE, BONE_SDF, MESH）
                {"active": True, "is_cloth": False, "is_col": True, "mode": 'SIMPLE', "col_type": 'SPHERE'},
                {"active": True, "is_cloth": False, "is_col": True, "mode": 'SIMPLE', "col_type": 'BONE_SDF'},
                {"active": True, "is_cloth": False, "is_col": True, "mode": 'SIMPLE', "col_type": 'MESH'},
                # 6. コライダー有効・詳細モード（各コライダータイプ、片面衝突、アニメーション駆動）
                {"active": True, "is_cloth": False, "is_col": True, "mode": 'ADVANCED', "col_type": 'SPHERE'},
                {"active": True, "is_cloth": False, "is_col": True, "mode": 'ADVANCED', "col_type": 'CAPSULE'},
                {"active": True, "is_cloth": False, "is_col": True, "mode": 'ADVANCED', "col_type": 'PLANE'},
                {"active": True, "is_cloth": False, "is_col": True, "mode": 'ADVANCED', "col_type": 'MESH', "single_sided": True, "cluster": True},
                {"active": True, "is_cloth": False, "is_col": True, "mode": 'ADVANCED', "col_type": 'BONE_SDF', "joint_mesh": True, "anim": True, "anim_type": 'SHAPE_KEY'},
                {"active": True, "is_cloth": False, "is_col": True, "mode": 'ADVANCED', "col_type": 'BONE_SDF', "anim": True, "anim_type": 'POSE_BLEND'},
                {"active": True, "is_cloth": False, "is_col": True, "mode": 'ADVANCED', "col_type": 'BONE_SDF', "anim": True, "anim_type": 'ACTION'},
                {"active": True, "is_cloth": False, "is_col": True, "mode": 'ADVANCED', "col_type": 'MESH_SDF'},
            ]

            scene = bpy.context.scene

            for st in ui_states:
                if not st.get("active"):
                    bpy.context.view_layer.objects.active = None
                else:
                    bpy.context.view_layer.objects.active = self.obj
                    if scene:
                        scene.taremin_cloth_ui_mode = st.get("mode", "SIMPLE")
                    self.obj.taremin_cloth.is_cloth = st.get("is_cloth", False)
                    self.obj.taremin_cloth.enable_sewing = st.get("sewing", False)
                    self.obj.taremin_cloth.enable_self_collision = st.get("self_col", False)
                    self.obj.taremin_cloth.enable_adaptive_substep = st.get("adaptive", False)
                    self.obj.taremin_cloth.enable_frame_buffering = st.get("buffering", False)
                    self.obj.taremin_cloth.show_fps_overlay = st.get("fps", False)

                    col_s = self.obj.taremin_cloth_collider
                    col_s.is_collider = st.get("is_col", False)
                    if col_s.is_collider:
                        col_s.collider_type = st.get("col_type", "SPHERE")
                        col_s.single_sided = st.get("single_sided", False)
                        col_s.enable_cluster_culling = st.get("cluster", False)
                        col_s.enable_joint_mesh = st.get("joint_mesh", False)
                        anim = getattr(col_s, "anim", None)
                        if anim:
                            anim.enabled = st.get("anim", False)
                            anim.target_type = st.get("anim_type", "SHAPE_KEY")

                for p_cls in panel_classes:
                    # poll チェック
                    if hasattr(p_cls, "poll") and not p_cls.poll(bpy.context):
                        continue
                    ctx = getattr(p_cls, "bl_translation_context", i18n.CONTEXT)
                    mock = MockLayout(ctx)
                    dummy_self = types.SimpleNamespace(layout=mock)
                    try:
                        p_cls.draw(dummy_self, bpy.context)
                        for ctxt, text, src in mock.collected:
                            all_collected_texts.append((ctxt, text))
                    except Exception as e:
                        self.fail(f"パネル {p_cls.__name__} (state: {st}) の描画中に例外が発生しました: {e}")

            # 後片付け（アクティブオブジェクト復元）
            bpy.context.view_layer.objects.active = self.obj

            # 3. 英語の動的接頭辞がそのままUI描画に残っていないかを検証
            forbidden_english_prefixes = (
                "Cloth Objects (",
                "Collider Objects (",
                "Selected:",
                "Material:",
                "Quality:",
                "Unapplied Scale:",
            )
            leaked_prefixes = []
            for ctxt, text in all_collected_texts:
                for f_pfx in forbidden_english_prefixes:
                    if text.startswith(f_pfx):
                        leaked_prefixes.append((ctxt, text))

            self.assertEqual(
                len(leaked_prefixes), 0,
                f"日本語環境下で未翻訳の動的接頭辞が検出されました:\n" +
                "\n".join(f"  ({c!r}, {t!r})" for c, t in leaked_prefixes)
            )

            # 4. 収集された全テキストの翻訳解決を検証
            untranslated = []
            test_obj_name = self.obj.name
            allowed_english = {
                "Taremin Cloth", "GPU Cloth", "L0", "DirectX 12", "Vulkan", "AMD Radeon",
                "0.0", "1.0", "Auto", "Auto (Auto)", "Bone SDF", "Mesh SDF",
            }

            for ctxt, text in all_collected_texts:
                clean = text.strip()
                if not clean or clean == test_obj_name or clean in allowed_english:
                    continue
                if any(clean.startswith(p) for p in dynamic_prefixes):
                    continue

                # 既に日本語が含まれている場合（事前翻訳済みテキスト等）は合格
                if contains_japanese(clean):
                    continue

                # まだ英語の場合、Blenderの翻訳APIを通して翻訳を試みる
                translated = bpy.app.translations.pgettext_iface(clean, ctxt)
                if contains_japanese(translated):
                    continue

                # 翻訳後も日本語が含まれず、かつ英字アルファベットが含まれている場合は未翻訳として検出
                if any('a' <= c.lower() <= 'z' for c in clean):
                    untranslated.append((ctxt, clean))

            self.assertEqual(
                len(untranslated), 0,
                f"UI描画時に未翻訳の文言が {len(untranslated)} 件検出されました:\n" +
                "\n".join(f"  ({c!r}, {t!r})" for c, t in sorted(set(untranslated)))
            )

        finally:
            pref_view.language = orig_lang
            pref_view.use_translate_interface = orig_trans


@unittest.skipUnless(IS_REAL_BLENDER, "Blender (bpy) 環境でのみ実行される結合テスト")
class TestI18nBlenderIntegration(unittest.TestCase):
    """Blender環境下での翻訳API基本結合テスト"""

    @classmethod
    def setUpClass(cls):
        i18n.register()

    @classmethod
    def tearDownClass(cls):
        i18n.unregister()

    def test_register_unregister_idempotent(self):
        """多重登録・多重解除がエラーを起こさないことを検証"""
        i18n.register()
        i18n.register()
        i18n.unregister()
        i18n.unregister()
        i18n.register()

    def test_translation_lookup_with_context(self):
        """bpy.app.translations.pgettext_iface によるコンテキスト別翻訳検証"""
        pref_view = bpy.context.preferences.view
        orig_lang = getattr(pref_view, "language", "DEFAULT")
        orig_trans = getattr(pref_view, "use_translate_interface", True)

        try:
            pref_view.language = "ja_JP"
            pref_view.use_translate_interface = True

            # アドオン固有UI文字列の日本語翻訳検証
            diag_trans = bpy.app.translations.pgettext_iface("GPU & Diagnostics", i18n.CONTEXT)
            self.assertEqual(diag_trans, "GPU & 診断情報")

            pin_trans = bpy.app.translations.pgettext_iface("Attachment & Pinning", i18n.CONTEXT)
            self.assertEqual(pin_trans, "追従 & ピン留め")

            # i18n.trans() ヘルパーの動作検証
            helper_trans = i18n.trans("GPU & Diagnostics")
            self.assertEqual(helper_trans, "GPU & 診断情報")

            # 英語（en_US）に切り替えた場合は英語のままであること
            pref_view.language = "en_US"
            en_diag = bpy.app.translations.pgettext_iface("GPU & Diagnostics", i18n.CONTEXT)
            self.assertEqual(en_diag, "GPU & Diagnostics")

        finally:
            pref_view.language = orig_lang
            pref_view.use_translate_interface = orig_trans

    def test_interactive_mode_translations(self):
        """インタラクティブモードのステータスバー・通知文言の多言語化検証"""
        pref_view = bpy.context.preferences.view
        orig_lang = getattr(pref_view, "language", "DEFAULT")
        try:
            # 日本語環境
            pref_view.language = "ja_JP"
            status_ja = i18n.trans("Taremin Cloth: [Left Drag] Move Vertex | [P] Toggle Pin | [Right Click / ESC] Exit")
            self.assertIn("左ドラッグ", status_ja)
            self.assertIn("ピン留め", status_ja)

            pin_ja = i18n.trans("Pinned vertex #%d") % 42
            self.assertEqual(pin_ja, "頂点 #42 をピン留めしました")

            unpin_ja = i18n.trans("Unpinned vertex #%d") % 42
            self.assertEqual(unpin_ja, "頂点 #42 のピン留めを解除しました")

            start_ja = i18n.trans("Interactive Simulation Started (Press ESC / RightClick or Click Stop to exit)")
            self.assertEqual(start_ja, "インタラクティブシミュレーション開始 (ESC / 右クリック または 停止ボタンで終了)")

            stop_ja = i18n.trans("Interactive Simulation Stopped (Paused)")
            self.assertEqual(stop_ja, "インタラクティブシミュレーション停止 (一時停止中)")

            # 英語環境
            pref_view.language = "en_US"
            status_en = i18n.trans("Taremin Cloth: [Left Drag] Move Vertex | [P] Toggle Pin | [Right Click / ESC] Exit")
            self.assertIn("Left Drag", status_en)

            pin_en = i18n.trans("Pinned vertex #%d") % 42
            self.assertEqual(pin_en, "Pinned vertex #42")
        finally:
            pref_view.language = orig_lang

    def test_screenshot_ui_terms_resolved(self):
        """UIスクリーンショットで未翻訳として報告された全UI要素の翻訳解決を検証"""
        pref_view = bpy.context.preferences.view
        orig_lang = getattr(pref_view, "language", "DEFAULT")
        orig_trans = getattr(pref_view, "use_translate_interface", True)

        try:
            pref_view.language = "ja_JP"
            pref_view.use_translate_interface = True

            # 1. オペレーターボタン群（TareminCloth コンテキストで解決されること）
            operator_terms = [
                ("Disable Cloth", "布シミュレーション無効化"),
                ("Interactive Mode (Grab/Drag)", "インタラクティブモード (掴み/ドラッグ)"),
                ("Create Seam (Select 2 Verts)", "縫合線を作成 (2頂点選択)"),
                ("Auto Fit", "自動最適化"),
                ("Clear Cache", "キャッシュを消去"),
                ("Reset All", "全てリセット"),
                ("All ON", "全て有効"),
                ("All OFF", "全て一時停止"),
                ("Solo", "選択中のみ有効 (Solo)"),
            ]
            for msgid, expected in operator_terms:
                trans_tc = bpy.app.translations.pgettext_iface(msgid, i18n.CONTEXT)
                self.assertEqual(
                    trans_tc, expected,
                    f"TareminClothコンテキストでの翻訳失敗: {msgid!r} -> 期待: {expected!r}, 実際: {trans_tc!r}"
                )
                # Operator コンテキストには登録されていないこと（名前空間非汚染の検証）
                trans_op = bpy.app.translations.pgettext_iface(msgid, "Operator")
                self.assertEqual(
                    trans_op, msgid,
                    f"Operatorコンテキストが汚染されています: {msgid!r} -> {trans_op!r}"
                )

            # 2. 接頭辞・ラベル群（アドオン固有コンテキストで解決されること）
            label_prefixes = [
                ("Cloth Objects", "Cloth オブジェクト"),
                ("Collider Objects", "Collider オブジェクト"),
                ("Selected:", "選択中:"),
                ("Material:", "素材:"),
                ("Quality:", "品質:"),
                ("Unapplied Scale:", "未適用のスケール:"),
                ("Scene Gravity:", "シーン重力:"),
                ("Registered Edges:", "登録エッジ数:"),
                ("Connected", "接続中"),
                ("Type:", "タイプ:"),
            ]
            for msgid, expected in label_prefixes:
                trans_lbl = bpy.app.translations.pgettext_iface(msgid, i18n.CONTEXT)
                self.assertEqual(
                    trans_lbl, expected,
                    f"TareminClothコンテキストでの翻訳失敗: {msgid!r} -> 期待: {expected!r}, 実際: {trans_lbl!r}"
                )
        finally:
            pref_view.language = orig_lang
            pref_view.use_translate_interface = orig_trans


if __name__ == "__main__":
    unittest.main()

