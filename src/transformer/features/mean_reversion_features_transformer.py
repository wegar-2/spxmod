import logging

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["MeanReversionFeaturesTransformer"]


class MeanReversionFeaturesTransformer:

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        logger.info("------- MEAN REVERSION features -------")

        return data