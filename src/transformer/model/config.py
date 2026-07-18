from typing import Any

from pydantic import BaseModel
from src.common.aliases import DatesInterval


class LogRegModelTransformerConfig(BaseModel):
    hyperparams: dict[Any, Any]
    variables: list[str]
    train_di: DatesInterval
    test_di: DatesInterval
