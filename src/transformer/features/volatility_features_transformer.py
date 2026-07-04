import logging

import pandas as pd

from src.common.aliases import IntTuple
from src.common.utils import ewm_zscore

logger = logging.getLogger(__name__)

__all__ = ["VolatilityFeaturesTransformer"]


class VolatilityFeaturesTransformer:

    def __init__(
            self,
            realized_vol_windows: IntTuple = (5, 20, 60, 250),
            zscore_half_lives: IntTuple = (20, 60, 250)
    ):
        self._realized_vol_windows: IntTuple = realized_vol_windows
        self._zscore_hal_lives: IntTuple = zscore_half_lives

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        logger.info("------- VOLATILITY features -------")

        logger.info("VOLATILITY 1/3. Realized volatility")
        for rv_hl in self._realized_vol_windows:

            logger.info(f"Realized vol over half life {rv_hl}")
            data[f"ewm_realized_vol_hl{rv_hl}"] = data["log_ret"].ewm(
                adjust=False, halflife=rv_hl).std()

            for hl in self._zscore_hal_lives:
                logger.info(f"zscoring realized ewm vol hl {rv_hl} using hl {hl}")
                data[f"ewm_realized_vol_hl{hl}"] = ewm_zscore(
                    data=data[f"ewm_realized_vol_hl{rv_hl}"], hl=hl)

        logger.info("VOLATILITY 2/3. Average true range")

        logger.info("VOLATILITY 3/3. Daily range")



        return data
