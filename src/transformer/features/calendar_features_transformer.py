import pandas as pd

__all__ = ["CalendarFeaturesTransformer"]


class CalendarFeaturesTransformer:

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        data["day_of_week"] = data.index.wee

        return data
