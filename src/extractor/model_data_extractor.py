import logging
from pathlib import Path
import pandas as pd

from src.common.utils import ensure_data_directory
from src.pipeline.model_data_pipeline import ModelDataPipeline

logger = logging.getLogger(__name__)

__all__ = ["ModelDataExtractor"]


class ModelDataExtractor:

    def __init__(self):
        self._data_folder_path: Path = Path(__file__).parent.parent.parent / "data"

    def extract(self) -> pd.DataFrame:
        ensure_data_directory()
        data_file_path: Path = self._data_folder_path / "model_data.parquet"
        if not data_file_path.exists():
            logger.info("Model data file does not exist; making it")
            ModelDataPipeline().run()
        return pd.read_parquet(data_file_path)
