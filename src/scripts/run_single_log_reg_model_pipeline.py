from datetime import date

from src.pipeline.model.log_reg_model_pipeline import LogRegModelPipeline
from src.pipeline.model.config import LogRegModelPipelineConfig


if __name__ == "__main__":
    LogRegModelPipeline(
        config=LogRegModelPipelineConfig(
            hyperparams={},
            indep_vars=[],
            dep_var="",
            train_di=(date(), date()),
            test_di=(date(), date())
        )
    ).run()
