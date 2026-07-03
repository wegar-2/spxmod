import pandas as pd


class CalendarFeaturesTransformer:

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        data["day_of_week"] = data.index.wee

        return data
