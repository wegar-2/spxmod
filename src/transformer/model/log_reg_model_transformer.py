from typing import Any

import pandas as pd

from sklearn.linear_model import LogisticRegression

__all__ = ["LogRegModelTransformer"]


class LogRegModelTransformer:

    def __init__(self, hyperparams: dict[Any, Any]):
        self._model = LogisticRegression(**hyperparams)

    def transform(
            self,
            train_data: tuple[pd.DataFrame, pd.DataFrame],
            test_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        (x_train, y_train), (x_test, y_test) = train_data, test_data
        self._model.fit(x_train, x_test)
        return self._model.predict(x_test), self._model.predict_proba(x_test)
