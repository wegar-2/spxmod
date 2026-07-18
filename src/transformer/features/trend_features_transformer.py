from itertools import product
import logging
from typing import Final

import pandas as pd

from src.common.utils import ewm_zscore
from src.common.aliases import IntTuple

logger = logging.getLogger(__name__)

__all__ = ["TrendFeaturesTransformer"]



class TrendFeaturesTransformer:

    def __init__(
            self,
            ma_windows: IntTuple = (20, 50, 100, 200),
            zscore_half_lives: IntTuple = (20, 60, 250),
            ma_slope_lags: IntTuple = (5, 10, 20)
    ):
        self._ma_windows: Final[IntTuple] = ma_windows
        self._zscore_half_lives: Final[IntTuple] = zscore_half_lives
        self._ma_slope_lags: Final[IntTuple] = ma_slope_lags

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        logger.info("------- TREND features -------")

        logger.info("TREND 1/3. Calculating MAs and MAs-based features")
        for w in self._ma_windows:

            logger.info(f"MA of close {w} days")
            data[f"close_ma_{w}D"] = (
                data["close"].rolling(window=w).mean())

            logger.info("Ratio of the MA to close")
            data[f"ratio_close_to_ma_{w}D"] = (
                data["close"] / data[f"close_ma_{w}D"])

            for hl in self._zscore_half_lives:
                logger.info(f"EWM Z-scores of the MA using half-life {hl}")
                data[f"close_ma_{w}D_zscore_hl{hl}D"] = ewm_zscore(
                    data=data[f"close_ma_{w}D"],
                    hl=hl,
                    min_periods=max(5, hl // 2)
                )

        logging.info("TREND 2/3. MA slopes")
        for ma_wind, ma_slope_lag in product(
                self._ma_windows,
                self._ma_slope_lags
        ):
            logger.info(f"MA {ma_wind}D slope change over {ma_slope_lag}D")
            data[f"ma_{ma_wind}D_slope_growth_{ma_slope_lag}D"] = (
                data[f"close_ma_{ma_wind}D"] /
                data[f"close_ma_{ma_wind}D"].shift(ma_slope_lag)
            ) - 1

        logger.info("TREND 3/3. Classic crossovers between MAs and/or close price")
        data["MA20_gt_MA50"] = (
            data["close"].rolling(window=20).mean() >
            data["close"].rolling(window=50).mean()
        ).astype(int)
        data["MA50_gt_MA200"] = (
            data["close"].rolling(window=50).mean() >
            data["close"].rolling(window=200).mean()
        ).astype(int)
        data["close_above_MA200"] = (
                data["close"] > data["close"].rolling(window=200).mean()
        ).astype(int)

        return data
