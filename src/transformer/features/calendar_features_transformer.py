import logging

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["CalendarFeaturesTransformer"]


class CalendarFeaturesTransformer:

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        logger.info("------- CALENDAR features -------")

        logger.info("CALENDAR 1/2. Adding day-of-week, months, quaretrs, etc")
        new_cols = {}
        new_cols["day_of_week"] = data.index.weekday

        logger.info("Adding month & quarter")
        new_cols["month"] = data.index.month
        new_cols["quarter"] = data.index.quarter

        logger.info("Adding binary variable for Monday and Friday")
        new_cols["is_monday"] = (new_cols["day_of_week"] == 0).astype(int)
        new_cols["is_friday"] = (new_cols["day_of_week"] == 4).astype(int)

        data = pd.concat([data, pd.DataFrame(new_cols, index=data.index)], axis=1)

        logger.info("CALENDAR 2/2. first and last trading day in month flag")

        dt = data.reset_index(drop=False)[["date"]].copy(deep=True)
        dt.index = dt["date"]
        dt = dt.groupby(pd.Grouper(freq="ME"))[["date"]].min()
        dt.index = dt["date"]
        dt["date"] = 1
        dt = dt.rename(columns={"date": "is_month_start"})
        data = pd.merge(data, dt, left_index=True, right_index=True, how="left")
        data["is_month_start"] = data["is_month_start"].fillna(0)
        data["is_month_start"] = data["is_month_start"].astype(int)

        dt = data.reset_index(drop=False)[["date"]].copy(deep=True)
        dt.index = dt["date"]
        dt = dt.groupby(pd.Grouper(freq="ME"))[["date"]].max()
        dt.index = dt["date"]
        dt["date"] = 1
        dt = dt.rename(columns={"date": "is_month_end"})
        data = pd.merge(data, dt, left_index=True, right_index=True, how="left")
        data["is_month_end"] = data["is_month_end"].fillna(0)
        data["is_month_end"] = data["is_month_end"].astype(int)

        return data
