from datetime import date
from typing import TypeAlias

__all__ = [
    "IntTuple",
    "DatesInterval"
]


IntTuple: TypeAlias = tuple[int, ...]

DatesInterval: TypeAlias = tuple[date, date]
