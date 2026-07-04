from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "ewm_atr",
    "ewm_rsi",
    "ewm_vol",
    "ewm_zscore",
    "rolling_zscore",
    "tr",
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
        min_periods = max(5, hl // 2)
    mu = data.ewm(halflife=hl, min_periods=min_periods, adjust=False).mean()
    sigma = data.ewm(halflife=hl, min_periods=min_periods, adjust=False).std()
    return (data - mu) / sigma


def tr(data: pd.DataFrame) -> pd.Series:

    h = data["high"]
    l = data["low"]
    c = data["close"]
    prev_close = c.shift(1)

    return pd.concat([
        (h - prev_close).abs(),
        (l - prev_close).abs(),
        (h - l).abs()
    ], axis=1).max(axis=1)


def ewm_atr(
        data: pd.Series[],
        hl: int,
        min_periods: Optional[int] = None
) -> pd.Series:

    if min_periods is None:
        min_periods = max(5, hl // 2)

    tr_: pd.Series = tr(data)
    atr = tr_.ewm(halflife=hl, adjust=False, min_periods=min_periods).mean()

    return atr


def ewm_rsi(
        close: pd.Series,
        hl: int
) -> pd.Series:

    min_periods = max(5, hl // 2)

    ups = (
        close
        .clip(lower=0)
        .ewm(
            halflife=hl,
            min_periods=min_periods,
            adjust=False
        ).mean()
    )
    downs = (
        (-close)
        .clip(down=0)
        .ewm(
            halflife=hl,
            min_periods=min_periods,
            adjust=False
        ).mean()
    )

    rs = ups / downs

    return 100 - (100 / (1 + rs))


def bb(
        close: pd.Series,
        window: int = 20,
        k: int = 2
) -> pd.DataFrame:

    std_ = close.rolling(window=window, min_periods=window).std()
    middle = close.rolling(window=window, min_periods=window).mean()

    return pd.DataFrame(data={
        f"bb_{window}d_mid": middle,
        f"bb_{window}d_lower_{k}stds": middle - k * std_,
        f"bb_{window}d_upper_{k}stds": middle + k * std_,
    }, index=close.index)


def ewm_vol(
        data: pd.Series,
        hl: int,
        annualize: bool = True
):

    out = data.ewm(
        halflife=hl,
        adjust=False,
        min_periods=max(5, hl // 2)
    ).std(bias=False)

    if annualize:
        out = out * np.sqrt(252)

    return out
