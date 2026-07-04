import logging

import pandas as pd

from src.transformer.features.utils import ewm_zscore

logger = logging.getLogger(__name__)

__all__ = ["TrendFeaturesTransformer"]


class TrendFeaturesTransformer:

    def __init__(
            self,
            ma_windows: tuple[int, ...] = (20, 50, 100, 200),
            zscore_half_lives: tuple[int, ...] = (20, 60, 250)
    ):
        self._ma_windows: tuple[int, ...] = ma_windows
        self._zscore_half_lives: tuple[int, ...] = zscore_half_lives

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        logger.info("Adding TREND features")

        logger.info("Calculating MAs and MAs-based features")
        for win in self._ma_windows:

            logger.info(f"MA of close {win} days")
            data[f"close_ma_{win}D"] = (
                data["close"].rolling(window=win).mean())

            logger.info("Ratio of the MA to close")
            data[f"ratio_close_to_ma_{win}D"] = (
                data["close"] / data[f"close_ma_{win}D"])

            for hl in self._zscore_half_lives:
                logger.info(f"EWM Z-scores of the MA using half-life {hl}")
                data[f"close_ma_{win}D_zscore_hl_{hl}D"] = ewm_zscore(
                    data=data[f"close_ma_{win}D"],
                    hl=hl,
                    min_periods=max(5, hl // 2)
                )

        logging.info("Calculating MA slopes")



        return data
