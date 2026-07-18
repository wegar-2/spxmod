import pandas as pd

from src.common.aliases import DatesInterval
from src.transformer.tscv.config import TSCVTransformerConfig

__all__ = ["TSCVTransformer"]


class TSCVTransformer:

    def __init__(self, config: TSCVTransformerConfig):
        self._config: TSCVTransformerConfig = config

    def transform(
            self,
            data: pd.DataFrame
    ) -> list[tuple[DatesInterval, DatesInterval]]:
        out: list[tuple[DatesInterval, DatesInterval]] = []

        for test_di in self._config.test_dis:
            idx = data.index

        return out
