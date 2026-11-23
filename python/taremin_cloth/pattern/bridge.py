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
    JSONが破損している場合は空ドキュメントに `_load_failed = True` 保護フラグを付与し、
    自動保存による元データの上書き消失を防止する.
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
        fallback_doc = Document.new(name="Scene Pattern")
        fallback_doc._load_failed = True
        return fallback_doc


def save_document_to_scene(
    doc: Document,
    *,
    scene: Optional[bpy.types.Scene] = None,
    push_undo: bool = True,
    undo_message: str = "Pattern Edit",
    force_overwrite: bool = False,
) -> None:
    """DocumentをJSON文字列としてBlenderシーンプロパティに保存する.
    
    push_undo が True の場合は Blender 本体の Undo スタックへ記録する.
    """
    if getattr(doc, "_load_failed", False) and not force_overwrite:
        print("[Taremin Cloth] Skipping save_document_to_scene because loaded JSON was corrupted (_load_failed=True).")
        return

    sc = scene or bpy.context.scene
    sc.taremin_pattern_document = doc.to_json()

    if push_undo and hasattr(bpy.ops.ed, "undo_push"):
        try:
            bpy.ops.ed.undo_push(message=undo_message)
        except Exception:
            pass
