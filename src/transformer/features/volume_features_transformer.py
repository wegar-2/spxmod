import logging

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["VolumeFeaturesTransformer"]


class VolumeFeaturesTransformer:

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        logger.info("------- VOLUME features -------")

        logger.info("VOLUME 1/2. Volume stats")

        logger.info("VOLUME 2/2. On-balance volume")

        return data
