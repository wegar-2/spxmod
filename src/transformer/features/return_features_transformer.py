import logging
from itertools import product

import numpy as np
import pandas as pd

from src.common.utils import ewm_zscore

logger = logging.getLogger(__name__)

__all__ = ["ReturnFeaturesTransformer"]


class ReturnFeaturesTransformer:

    def __init__(
            self,
            return_horizons: tuple[int, ...] = (2, 5, 10, 20, 60, 120),
            hls: tuple[int, ...] = (5, 20, 60, 252)
    ):
        self._half_lives = hls
        self._return_horizons = return_horizons

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        logger.info("------- RETURN features -------")

        logger.info("RETURN 1/3. Calculating daily return")
        data["log_ret"] = np.log(data["close"] / data["close"].shift(1))

        logger.info("RETURN 2/3. Calculating returns over longer horizons")
        for rh in self._return_horizons:
            logger.info(f"Adding log-return over last {rh}D")
            data[f"log_ret_{rh}D"] = np.log(
                data["close"] / data["close"].shift(rh))

        logger.info("RETURN 3/3. Calculating EWM-zscored returns")
        for rh, hl in product(self._return_horizons, self._half_lives):
            logger.info(f"Smoothing returns over {rh}D using half-life {hl}")
            data[f"log_ret_{rh}D_"] = ewm_zscore(data[f"log_ret_{rh}D"], hl=hl)

        return data
