from typing import Any

from pydantic import BaseModel

from src.common.aliases import DatesInterval

__all__ = ["LogRegModelPipelineConfig"]


class LogRegModelPipelineConfig(BaseModel):
    hyperparams: dict[Any, Any]
    indep_vars: list[str]
    dep_var: str
    train_di: DatesInterval
    test_di: DatesInterval
