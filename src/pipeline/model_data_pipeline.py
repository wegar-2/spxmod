from moddata import load_data
import pandas as pd

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


class ModelDataPipeline:

    def run(self) -> pd.DataFrame:

        data = load_data("spx_1901-2025")
        data = data["2000-01-01":]

        data = ReturnFeaturesTransformer().transform(data)
        data = TrendFeaturesTransformer().transform(data)
        data = VolatilityFeaturesTransformer().transform(data)
        data = MeanReversionFeaturesTransformer().transform(data)
        data = VolumeFeaturesTransformer().transform(data)
        data = CalendarFeaturesTransformer().transform(data)

        data = data.loc[~data.isna().any(axis=1), :]
        return data
