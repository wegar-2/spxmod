from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "ewm_zscore",
    "rolling_zscore"
]


def rolling_zscore(
        data: pd.Series,
        window: int,
        min_window: Optional[int] = None
) -> pd.Series:
    if min_window is None:
        min_window = window // 2

    mu = data.rolling(window=window, min_periods=min_window).mean()
    sigma = data.rolling(window=window, min_periods=min_window).std(ddof=0)

    return (data - mu) / sigma


def ewm_zscore(
        data: pd.Series,
        hl: int,
        min_periods: Optional[int] = None
):
    if min_periods is None:
        min_periods = hl
    mu = data.ewm(halflife=hl, min_periods=min_periods, adjust=False).mean()
    sigma = data.ewm(halflife=hl, min_periods=min_periods, adjust=False).std()
    return (data - mu) / sigma


def atr(
        h: pd.Series,
        l: pd.Series,
        c: pd.Series,
        hl: int
) -> pd.Series:

    prev_close = c.shift(1)

    tr = pd.concat([
        (h - prev_close).abs(),
        (l - prev_close).abs(),
        (h - l).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(halflife=hl).mean()


def rsi():
    pass
