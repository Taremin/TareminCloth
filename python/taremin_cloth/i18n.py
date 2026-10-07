"""
Taremin Cloth 国際化（i18n）管理モジュール

Blender標準の bpy.app.translations API を利用し、英語（デフォルト）および
日本語などの多言語UI表示をサポートします。

辞書データは translations/*.json に分離管理され、英語マスター（en_US.json）を
基準として各言語ファイル（ja_JP.json 等）が配置されます。
プロパティ名・オペレーター・UIラベル等の用語はアドオン専用コンテキスト "TareminCloth" に登録して
名前空間汚染を防ぎます。
一方、ツールチップ説明文（description）については、Blender本体のC++側がホバー描画時に
RNAコンテキストを渡さず "*" で検索する仕様に対応するため、"TareminCloth" に加えて
一般コンテキスト "*" にも登録されます。
"""

import json
import os
from pathlib import Path
from typing import Dict, Set, Any

try:
    import bpy
    HAS_BPY = True
except ImportError:
    HAS_BPY = False

# アドオン固有コンテキスト（他アドオン・Blender標準との名前空間衝突を防止）
CONTEXT = "TareminCloth"

# 翻訳データディレクトリのパス
TRANSLATIONS_DIR = Path(__file__).resolve().parent / "translations"
DESCRIPTIONS_FILE = TRANSLATIONS_DIR / "descriptions.json"


def load_translations() -> Dict[str, Dict[str, str]]:
    """translations/ ディレクトリ内の言語別 *.json を読み込み、言語コードごとの辞書を返す"""
    translations = {}
    if not TRANSLATIONS_DIR.is_dir():
        return translations

    for json_file in sorted(TRANSLATIONS_DIR.glob("*.json")):
        # descriptions.json はメタデータのため言語辞書としてはスキップ
        if json_file.name == "descriptions.json":
            continue
        locale = json_file.stem
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                translations[locale] = json.load(f)
        except Exception as e:
            print(f"[TareminCloth] i18n JSON読み込みエラー ({json_file.name}): {e}")
    return translations


def load_descriptions() -> Set[str]:
    """translations/descriptions.json からツールチップ用説明文キーのセットを読み込む"""
    if not DESCRIPTIONS_FILE.is_file():
        return set()
    try:
        with open(DESCRIPTIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return set(data)
    except Exception as e:
        print(f"[TareminCloth] i18n descriptions JSON読み込みエラー: {e}")
    return set()


# モジュールロード時に辞書および説明文セットをキャッシュ
_RAW_TRANSLATIONS = load_translations()
_DESCRIPTIONS = load_descriptions()

# 互換性レイヤー: 従来の (ctx, msgid) タプルキー形式の辞書ビュー（既存テストおよび外部互換用）
TRANSLATIONS_DICT: Dict[str, Dict[tuple, str]] = {}
for locale, entries in _RAW_TRANSLATIONS.items():
    TRANSLATIONS_DICT[locale] = {}
    for msgid, msgstr in entries.items():
        TRANSLATIONS_DICT[locale][(CONTEXT, msgid)] = msgstr
        if msgid in _DESCRIPTIONS:
            TRANSLATIONS_DICT[locale][("*", msgid)] = msgstr


def get_raw_translations() -> Dict[str, Dict[str, str]]:
    """ロード済みの生辞書データ {locale: {msgid: msgstr}} を取得"""
    return _RAW_TRANSLATIONS


def get_descriptions() -> Set[str]:
    """ロード済みのツールチップ説明文キーセットを取得"""
    return _DESCRIPTIONS


def reload_translations():
    """辞書ファイルおよびdescription定義を再読み込み（ホットリロード用）"""
    global _RAW_TRANSLATIONS, _DESCRIPTIONS, TRANSLATIONS_DICT
    _RAW_TRANSLATIONS = load_translations()
    _DESCRIPTIONS = load_descriptions()
    TRANSLATIONS_DICT.clear()
    for locale, entries in _RAW_TRANSLATIONS.items():
        TRANSLATIONS_DICT[locale] = {}
        for msgid, msgstr in entries.items():
            TRANSLATIONS_DICT[locale][(CONTEXT, msgid)] = msgstr
            if msgid in _DESCRIPTIONS:
                TRANSLATIONS_DICT[locale][("*", msgid)] = msgstr


def _build_registered_dict() -> Dict[str, Dict[tuple, str]]:
    """Blender の bpy.app.translations.register 用に辞書を構築

    【重要・Blenderのツールチップ翻訳仕様に関する注意書き】
    BlenderのUIエンジン（C++側）は、ボタンやプロパティのホバー時に表示される
    ツールチップ（RNA property description / bl_description / enum_description）を検索する際、
    RNAプロパティに設定された translation_context を渡さず、汎用コンテキスト "*" で
    BLF_pgettext_tip を呼び出す仕様（または不具合）となっています。
    そのため、アドオン固有コンテキスト ("TareminCloth") のみに登録していると、
    ツールチップが辞書にヒットせず英語のまま表示されてしまいます。
    他アドオンやBlender標準との名前空間汚染を防ぐため、プロパティ名（name）や
    UIラベルなどの短い単語は "TareminCloth" 専用コンテキストのみに登録を維持し、
    衝突の恐れがない説明文（description）に限定して一般コンテキスト "*" にも登録します。
    """
    res = {}
    for locale, entries in _RAW_TRANSLATIONS.items():
        res[locale] = {}
        for msgid, msgstr in entries.items():
            # アドオン固有コンテキストには全エントリを登録
            res[locale][(CONTEXT, msgid)] = msgstr
            # ツールチップ（description）はBlender本体の検索仕様に合わせて一般コンテキスト "*" にも登録
            if msgid in _DESCRIPTIONS:
                res[locale][("*", msgid)] = msgstr
    return res


def register():
    """Blender翻訳辞書を登録"""
    if not HAS_BPY:
        return
    try:
        # 再登録時のエラー防止のため既存登録を例外安全に解除
        try:
            bpy.app.translations.unregister(__name__)
        except Exception:
            pass
        registered_dict = _build_registered_dict()
        bpy.app.translations.register(__name__, registered_dict)
    except Exception as e:
        print(f"[TareminCloth] i18n register error: {e}")


def unregister():
    """Blender翻訳辞書の登録解除"""
    if not HAS_BPY:
        return
    try:
        bpy.app.translations.unregister(__name__)
    except Exception as e:
        print(f"[TareminCloth] i18n unregister error: {e}")


def trans(msgid: str, context: str = CONTEXT) -> str:
    """UI文字列を現在のBlender言語設定に合わせて翻訳するヘルパー関数"""
    if HAS_BPY and hasattr(bpy, "app") and hasattr(bpy.app, "translations"):
        pget = getattr(bpy.app.translations, "pgettext_iface", bpy.app.translations.pgettext)
        res = pget(msgid, context)
        if isinstance(res, str):
            return res
    return msgid


