import logging
import sys

# アドオン用ロガーの取得
logger = logging.getLogger("taremin_cloth")

# ハンドラーが未設定の場合のみ構成（二重登録防止）
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[TareminCloth][%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def set_log_level(level_name: str):
    """
    ロガーの出力レベルを変更する ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)
