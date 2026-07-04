import logging

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["VolumeFeaturesTransformer"]


class VolumeFeaturesTransformer:

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        logger.info("------- VOLUME features -------")

        return data
