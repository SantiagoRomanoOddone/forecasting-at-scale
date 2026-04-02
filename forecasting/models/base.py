"""Abstract base class for all forecasting models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Interface that every model must implement.

    This guarantees a consistent API across all model types:
    GluonTS, boosting, baselines, or any future model.
    """

    def __init__(self, prediction_length: int, context_length: int, **kwargs):
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.params = kwargs
        self._fitted = False

    @abstractmethod
    def fit(
        self,
        train_data: Any,
        val_data: Any = None,
    ) -> "BaseModel":
        """Train the model.

        Args:
            train_data: Training dataset (format depends on implementation).
            val_data: Validation dataset for early stopping.

        Returns:
            self
        """

    @abstractmethod
    def predict(self, test_data: Any) -> pd.DataFrame:
        """Generate forecasts.

        Args:
            test_data: Test dataset.

        Returns:
            DataFrame with columns: date, store, prediction.
        """

    @abstractmethod
    def get_hyperparameter_space(self) -> dict:
        """Return the Optuna hyperparameter search space definition.

        Returns:
            Dict mapping param names to tuples of (type, *args).
            Example: {"lr": ("float", 1e-4, 1e-2, True)}
        """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def is_fitted(self) -> bool:
        return self._fitted
