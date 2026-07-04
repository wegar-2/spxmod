import logging

import numpy as np
import pandas as pd

from src.common.aliases import IntTuple
from src.common.utils import ewm_zscore

logger = logging.getLogger(__name__)

__all__ = ["VolumeFeaturesTransformer"]


class VolumeFeaturesTransformer:

    def __init__(
            self,
            volume_half_lives: IntTuple = (20, 60, 250),
            obv_lag: int = 20,
            obv_change_half_lives: IntTuple = (20, 60, 250)
    ):
        self._volume_half_lives: IntTuple = volume_half_lives
        self._obv_lag: int = obv_lag
        self._obv_change_half_lives: IntTuple = obv_change_half_lives

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        logger.info("------- VOLUME features -------")

        logger.info("VOLUME 1/2. Volume stats")

        logger.info("Log of volume")
        data["log_volume"] = np.log(data["volume"])

        for hl in self._volume_half_lives:

            logger.info(f"Volume above or below EWM hl{hl} of volume")
            data[f"ratio_volume_to_ewm_hl{hl}"] = (
                data["volume"] /
                data["volume"].ewm(
                    halflife=hl,
                    adjust=False,
                    min_periods=max(hl // 2, 5)
                ).mean()
            ) - 1

            logger.info(f"LogVolume zscore half life {hl}")
            data[f"log_volume_zscore_hl{hl}"] = (
                ewm_zscore(data=data["log_volume"], hl=hl))


        logger.info("VOLUME 2/2. On-balance volume (OBV)")

        logger.info("On-balance volume (OBV)")
        signed_volume = np.sign(data["log_return"]) * data["volume"]
        data["obv"] = signed_volume.cumsum()

        logger.info("Change of on-balance volume")
        data[f"obv_change_{self._obv_lag}D"] = (
            data["obv"] / data["obv"].shift(self._obv_lag)) - 1

        logger.info("Calculating zscores of obv changes")
        for hl in self._obv_change_half_lives:
            data[f"obv_change_{self._obv_lag}D_zscore_hl{hl}"] = ewm_zscore(
                data=data[f"obv_change_{self._obv_lag}D"], hl=hl)

        return data
