from typing import Any

from pydantic import BaseModel

from src.common.aliases import DatesInterval

__all__ = ["LogRegModelPipelineConfig"]


class LogRegModelPipelineConfig(BaseModel):
    hyperparams: dict[Any, Any]
    train_test_dis: list[DatesInterval]
