import logging

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["CalendarFeaturesTransformer"]


class CalendarFeaturesTransformer:

    def _add_month_first_day(self):
        pass

    def _add_month_last_day(self):
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:

        logger.info("Adding calendar-based features")

        logger.info("Adding day-of-week")
        data["day_of_week"] = data.index.weekday

        logger.info("Adding month & quarter")
        data["month"] = data.index.month
        data["quarter"] = data.index.quarter

        logger.info("Adding binary variable for Monday and Friday")
        data["is_monday"] = (data["day_of_week"] == 0).astype(int)
        data["is_friday"] = (data["day_of_week"] == 4).astype(int)


        data["is_month_start"] = data.index.is_month_start.astype(int)
        data["is_month_end"] = data.index.is_month_end.astype(int)

        return data
