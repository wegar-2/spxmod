import logging
import pandas as pd

from src.extractor.model_data_extractor import ModelDataExtractor
from src.transformer.model.log_reg_model_transformer import (
    LogRegModelTransformer)
from src.pipeline.model.config import LogRegModelPipelineConfig

logger = logging.getLogger(__name__)

__all__ = ["LogRegModelPipeline"]


class LogRegModelPipeline:

    def __init__(self, config: LogRegModelPipelineConfig):
        self._config: LogRegModelPipelineConfig = config

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:

        return data

    def run(self):

        data: pd.DataFrame = ModelDataExtractor().extract()
        data = self._preprocess_data(data)

        x_train = data.loc[
            self._config.train_di[0]:self._config.train_di[1],
            self._config.indep_vars
        ]
        x_test = data.loc[
            self._config.test_di[0]:self._config.test_di[1],
            self._config.indep_vars
        ]
        y_train = data.loc[
            self._config.train_di[0]:self._config.train_di[1],
            [self._config.dep_var]
        ]
        y_test = data.loc[
            self._config.test_di[0]:self._config.test_di[1],
            [self._config.dep_var]
        ]

        LogRegModelTransformer(
            hyperparams=self._config.hyperparams
        ).transform(
            train_data=(x_train, y_train),
            test_data=(x_test, y_test)
        )
