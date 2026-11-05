"""
関数の引数設計に関するガードおよび除外マーカー定義モジュール。
"""

from typing import TypeVar, Callable

F = TypeVar("F", bound=Callable[..., object])


def allow_positional(fn: F) -> F:
    """AST監査ツール（tools/audit_function_args.py）によるキーワード専用引数ルールの強制を除外するマーカーデコレータ。

    数学演算（lerp, clamp, dot 等）や、呼び出し順序が本質的に自明な低レイヤー関数において、
    位置引数を意図的に維持したい場合に付与します。

    実行時オーバーヘッドは 0 ナノ秒（恒等関数としてそのまま返却）です。

    使用例:
        @allow_positional
        def lerp(a: float, b: float, t: float) -> float:
            return a + (b - a) * t
    """
    return fn

