import logging
import pandas as pd

from src.extractor.model_data_extractor import ModelDataExtractor
from src.transformer.model.log_reg_model_transformer import (
    LogRegModelTransformer)

logger = logging.getLogger(__name__)

__all__ = ["LogRegModelPipeline"]


class LogRegModelPipeline:

    def __init__(self):
        pass

    def run(self):
        data: pd.DataFrame = ModelDataExtractor().extract()
        LogRegModelTransformer()
        