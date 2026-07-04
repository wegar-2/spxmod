import logging

import pandas as pd

from src.common.utils import bb, ewm_rsi, ewm_zscore
from src.common.aliases import IntTuple

logger = logging.getLogger(__name__)

__all__ = ["MeanReversionFeaturesTransformer"]


class MeanReversionFeaturesTransformer:

    def __init__(
            self,
            price_zscore_half_lives: IntTuple = (20, 60, 250),
            bb_params: tuple[int, int] = (20, 2),
            bb_half_lives: IntTuple = (20, 60, 250),
            rsi_half_lives: IntTuple = (2, 5, 14)
    ):
        self._price_zscore_half_lives: IntTuple = price_zscore_half_lives
        self._bb_params: tuple[int, int] = bb_params
        self._bb_half_lives: IntTuple = bb_half_lives
        self._rsi_half_lives: IntTuple = rsi_half_lives

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        logger.info("------- MEAN REVERSION features -------")

        logger.info("MEAN REVERSION 1/3. Price zscore")
        for phl in self._price_zscore_half_lives:
            logger.info(f"ewma of price using hl{phl}")
            data[f"price_ewm_zscore_hl{phl}"] = ewm_zscore(
                data["close"], hl=phl)

        logger.info("MEAN REVERSION 2/3. Bollinger bands")
        bb_window, bb_k = self._bb_params
        bb_data = bb(close=data["close"], window=bb_window, k=bb_k)

        logger.info("Relative position to bollinger bands")
        data["close_bbs_relative_pos"] = (
            data["close"] / bb_data[f"bb_{bb_window}d_mid"]
        ) / (
            bb_data[f"bb_{bb_window}d_upper_{bb_k}stds"] -
            bb_data[f"bb_{bb_window}d_lower_{bb_k}stds"]
        )
        data["bb_lower_relative_pos"] = (
                data["close"] / bb_data[f"bb_{bb_window}d_lower_{bb_k}stds"]
        ) - 1
        data["bb_lower_relative_pos"] = (
                data["close"] / bb_data[f"bb_{bb_window}d_upper_{bb_k}stds"]
        ) - 1

        logger.info("EWMA smoothing of the position of close withing bbs")
        for bb_hl in self._bb_half_lives:
            data[f"close_bbs_relative_pos_hl_{bb_hl}"] = (
                ewm_zscore(data=data["close_bbs_relative_pos"], hl=bb_hl))

        logger.info("MEAN REVERSION 3/3. Relative strength index")
        for rsi_hl in self._rsi_half_lives:
            data[f"rsi_hl{rsi_hl}"] = ewm_rsi(close=data["close"], hl=rsi_hl)

        return data