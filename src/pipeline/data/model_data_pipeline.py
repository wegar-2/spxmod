import logging
from pathlib import Path

from moddata import load_data

from src.transformer.features.return_features_transformer import (
    ReturnFeaturesTransformer)
from src.transformer.features.trend_features_transformer import (
    TrendFeaturesTransformer)
from src.transformer.features.volatility_features_transformer import (
    VolatilityFeaturesTransformer)
from src.transformer.features.mean_reversion_features_transformer import (
    MeanReversionFeaturesTransformer)
from src.transformer.features.volume_features_transformer import (
    VolumeFeaturesTransformer)
from src.transformer.features.calendar_features_transformer import (
    CalendarFeaturesTransformer)

logger = logging.getLogger(__name__)

__all__ = ["ModelDataPipeline"]


class ModelDataPipeline:

    def __init__(self):
        self._data_path: Path = (
                Path(__file__).parent.parent.parent.parent /
                "data" /
                "model_data.parquet"
        )

        self._transformers = [
            ReturnFeaturesTransformer(),
            TrendFeaturesTransformer(),
            VolatilityFeaturesTransformer(),
            MeanReversionFeaturesTransformer(),
            VolumeFeaturesTransformer(),
            CalendarFeaturesTransformer()
        ]

    def run(self) -> None:

        data = load_data("spx_1901-2025")
        data = data["2000-01-01":]

        for transformer in self._transformers:
            data = transformer.transform(data)

        logger.info("Keeping only rows without NAs")
        data = data.loc[~data.isna().any(axis=1), :]

        logger.info("Saving model dataset to file")
        data.to_parquet(self._data_path, engine="pyarrow")
