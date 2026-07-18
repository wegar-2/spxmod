from datetime import date

from src.pipeline.model.log_reg_model_pipeline import LogRegModelPipeline
from src.pipeline.model.config import LogRegModelPipelineConfig


if __name__ == "__main__":
    LogRegModelPipeline(
        config=LogRegModelPipelineConfig(
            hyperparams={},
            indep_vars=[],
            dep_var="",
            train_di=(date(2021, 1, 1), date(2023, 11, 30)),
            test_di=(date(2024, 1, 1), date(2024, 12, 31))
        )
    ).run()
