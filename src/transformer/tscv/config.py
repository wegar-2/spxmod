from dateutil.relativedelta import relativedelta

from pydantic import BaseModel, ConfigDict

from src.common.aliases import DatesInterval

__all__ = ["TSCVTransformerConfig"]


class TSCVTransformerConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    test_dis: list[DatesInterval]
    embargo_num_obs: int
    purge_num_obs: int
    train_data_offset: relativedelta
