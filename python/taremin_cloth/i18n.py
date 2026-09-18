"""
Taremin Cloth 国際化（i18n）管理モジュール

Blender標準の bpy.app.translations API を利用し、英語（デフォルト）および
日本語などの多言語UI表示をサポートします。

辞書データは translations/*.json に分離管理され、英語マスター（en_US.json）を
基準として各言語ファイル（ja_JP.json 等）が配置されます。
すべてのアドオン固有用語・プロパティ・オペレーターはアドオン専用コンテキスト "TareminCloth" のみで登録され、
一般コンテキスト "*" および "Operator" への登録・汚染は行いません。
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

try:
    import bpy
    HAS_BPY = True
except ImportError:
    HAS_BPY = False

# アドオン固有コンテキスト（他アドオン・Blender標準との名前空間衝突を防止）
CONTEXT = "TareminCloth"

# 翻訳データディレクトリのパス
TRANSLATIONS_DIR = Path(__file__).resolve().parent / "translations"


def load_translations() -> Dict[str, Dict[str, str]]:
    """translations/ ディレクトリ内の全 *.json を読み込み、言語コードごとの辞書を返す"""
    translations = {}
    if not TRANSLATIONS_DIR.is_dir():
        return translations

    for json_file in sorted(TRANSLATIONS_DIR.glob("*.json")):
        locale = json_file.stem
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                translations[locale] = json.load(f)
        except Exception as e:
            print(f"[TareminCloth] i18n JSON読み込みエラー ({json_file.name}): {e}")
    return translations


# モジュールロード時に辞書をキャッシュ
_RAW_TRANSLATIONS = load_translations()

# 互換性レイヤー: 従来の (ctx, msgid) タプルキー形式の辞書ビュー（既存テストおよび外部互換用）
TRANSLATIONS_DICT: Dict[str, Dict[tuple, str]] = {}
for locale, entries in _RAW_TRANSLATIONS.items():
    TRANSLATIONS_DICT[locale] = {}
    for msgid, msgstr in entries.items():
        TRANSLATIONS_DICT[locale][("*", msgid)] = msgstr
        TRANSLATIONS_DICT[locale][(CONTEXT, msgid)] = msgstr


def get_raw_translations() -> Dict[str, Dict[str, str]]:
    """ロード済みの生辞書データ {locale: {msgid: msgstr}} を取得"""
    return _RAW_TRANSLATIONS


def reload_translations():
    """辞書ファイルを再読み込み（ホットリロード用）"""
    global _RAW_TRANSLATIONS, TRANSLATIONS_DICT
    _RAW_TRANSLATIONS = load_translations()
    TRANSLATIONS_DICT.clear()
    for locale, entries in _RAW_TRANSLATIONS.items():
        TRANSLATIONS_DICT[locale] = {}
        for msgid, msgstr in entries.items():
            TRANSLATIONS_DICT[locale][("*", msgid)] = msgstr
            TRANSLATIONS_DICT[locale][(CONTEXT, msgid)] = msgstr


def _build_registered_dict() -> Dict[str, Dict[tuple, str]]:
    """Blender の bpy.app.translations.register 用に辞書を構築

    英語マスター（en_US）以外の各言語について、
    アドオン固有コンテキスト ('TareminCloth', msgid): msgstr のみで登録します。
    一般コンテキスト '*' や 'Operator' への登録は行いません。
    """
    res = {}
    for locale, entries in _RAW_TRANSLATIONS.items():
        res[locale] = {}
        for msgid, msgstr in entries.items():
            res[locale][(CONTEXT, msgid)] = msgstr
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

