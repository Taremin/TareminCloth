# -*- coding: utf-8 -*-
"""型紙ドキュメントとBlenderシーンプロパティの相互同期ブリッジ (bridge.py)."""

from __future__ import annotations

from typing import Optional
import bpy

from .document import Document


def load_document_from_scene(
    scene: Optional[bpy.types.Scene] = None,
) -> Document:
    """BlenderシーンのJSONプロパティからDocumentを復元する.
    
    プロパティが存在しないか空文字列の場合は新規Documentを作成して返す.
    """
    sc = scene or bpy.context.scene
    raw_json = getattr(sc, "taremin_pattern_document", "")
    if not raw_json or not raw_json.strip():
        doc = Document.new(name="Scene Pattern")
        return doc

    try:
        return Document.from_json(raw_json)
    except Exception as e:
        print(f"[Taremin Cloth] Failed to restore pattern document from scene: {e}")
        return Document.new(name="Scene Pattern")


def save_document_to_scene(
    doc: Document,
    *,
    scene: Optional[bpy.types.Scene] = None,
    push_undo: bool = True,
    undo_message: str = "Pattern Edit",
) -> None:
    """DocumentをJSON文字列としてBlenderシーンプロパティに保存する.
    
    push_undo が True の場合は Blender 本体の Undo スタックへ記録する.
    """
    sc = scene or bpy.context.scene
    sc.taremin_pattern_document = doc.to_json()

    if push_undo and hasattr(bpy.ops.ed, "undo_push"):
        try:
            bpy.ops.ed.undo_push(message=undo_message)
        except Exception:
            pass
