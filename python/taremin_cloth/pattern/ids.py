# -*- coding: utf-8 -*-
"""型紙エンティティ用の一意識別子（UUID4 hex）生成および検証."""

import uuid


def new_id() -> str:
    """32文字の16進数文字列（UUID4 hex、ハイフンなし）を生成して返す."""
    return uuid.uuid4().hex


def is_valid_id(val: object) -> bool:
    """指定された値が32文字の小文字16進文字列であるかを判定する."""
    if not isinstance(val, str) or len(val) != 32:
        return False
    try:
        int(val, 16)
        return val.islower() or val.isdigit()
    except (ValueError, TypeError):
        return False
