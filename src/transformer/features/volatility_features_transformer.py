import logging

import pandas as pd

from src.common.aliases import IntTuple

logger = logging.getLogger(__name__)

__all__ = ["VolatilityFeaturesTransformer"]


class VolatilityFeaturesTransformer:

    def __init__(
            self,
            realized_vol_windows: IntTuple = (5, 20, 60, 250)
    ):
        self._realized_vol_windows: IntTuple = realized_vol_windows

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        logger.info("Realized volatility")
        

        return data
