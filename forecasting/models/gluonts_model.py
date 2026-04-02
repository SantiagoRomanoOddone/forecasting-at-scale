"""GluonTS-based models: DeepAR, SimpleFeedforward, TFT, etc."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation import make_evaluation_predictions

from forecasting.features.temporal import create_temporal_features
from forecasting.models.base import BaseModel
from forecasting.models.registry import register

logger = logging.getLogger(__name__)


def build_list_dataset(
    pivot: pd.DataFrame,
    start_date: pd.Timestamp,
    frequency: str,
    use_temporal_features: bool = True,
) -> ListDataset:
    """Convert a pivoted DataFrame into a GluonTS ListDataset.

    Each column (store) becomes a separate time series.

    Args:
        pivot: DataFrame with index=dates, columns=store_ids, values=target.
        start_date: Start date of the series.
        frequency: Pandas frequency string.
        use_temporal_features: Whether to add calendar features.

    Returns:
        GluonTS ListDataset.
    """
    series_list = []
    dates = pivot.index

    temporal_features = None
    if use_temporal_features:
        temporal_features = create_temporal_features(dates)

    for idx, store_id in enumerate(pivot.columns):
        entry = {
            FieldName.TARGET: pivot[store_id].values.astype(np.float32),
            FieldName.START: start_date,
            FieldName.FEAT_STATIC_CAT: [idx],
        }
        if temporal_features is not None:
            entry[FieldName.FEAT_DYNAMIC_REAL] = temporal_features

        series_list.append(entry)

    return ListDataset(series_list, freq=frequency)


class GluonTSModel(BaseModel):
    """Wrapper for any GluonTS estimator.

    Subclasses only need to define _build_estimator() and
    get_hyperparameter_space().
    """

    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        num_stores: int = 1,
        frequency: str = "D",
        max_epochs: int = 50,
        **kwargs,
    ):
        super().__init__(prediction_length, context_length, **kwargs)
        self.num_stores = num_stores
        self.frequency = frequency
        self.max_epochs = max_epochs
        self._predictor = None

    def _build_estimator(self, params: dict) -> Any:
        """Build the GluonTS estimator with given hyperparameters.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def fit(self, train_data: ListDataset, val_data: ListDataset = None):
        """Train the model.

        Args:
            train_data: GluonTS ListDataset for training.
            val_data: GluonTS ListDataset for validation.

        Returns:
            self
        """
        estimator = self._build_estimator(self.params)

        if val_data is not None:
            self._predictor = estimator.train(
                training_data=train_data,
                validation_data=val_data,
            )
        else:
            self._predictor = estimator.train(training_data=train_data)

        self._fitted = True
        logger.info("%s training complete.", self.name)
        return self

    def predict(self, test_data: ListDataset) -> pd.DataFrame:
        """Generate predictions.

        Args:
            test_data: GluonTS ListDataset (full series including test period).

        Returns:
            DataFrame with columns: store_idx, date, prediction.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_data,
            predictor=self._predictor,
        )

        forecasts = list(forecast_it)
        actuals = list(ts_it)

        rows = []
        for idx, (forecast, actual) in enumerate(zip(forecasts, actuals)):
            start = forecast.start_date
            if hasattr(start, "to_timestamp"):
                start = start.to_timestamp()

            dates = pd.date_range(
                start=start,
                periods=self.prediction_length,
                freq=self.frequency,
            )
            pred_values = forecast.mean
            actual_values = actual.values[-self.prediction_length:].flatten()

            for date, pred_val, actual_val in zip(
                dates, pred_values, actual_values,
            ):
                rows.append({
                    "store_idx": idx,
                    "date": date,
                    "prediction": float(pred_val),
                    "actual": float(actual_val),
                })

        return pd.DataFrame(rows)


@register("simple_feedforward")
class SimpleFeedforwardModel(GluonTSModel):
    """GluonTS Simple Feedforward (MLP) model."""

    def _build_estimator(self, params: dict):
        from gluonts.torch.model.simple_feedforward import (
            SimpleFeedForwardEstimator,
        )

        return SimpleFeedForwardEstimator(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            hidden_dimensions=params.get("hidden_dimensions", [40, 40]),
            batch_size=params.get("batch_size", 32),
            trainer_kwargs={"max_epochs": self.max_epochs},
        )

    def get_hyperparameter_space(self) -> dict:
        return {
            "hidden_dim": ("categorical", [[20, 20], [40, 40], [64, 64]]),
            "batch_size": ("categorical", [32, 64, 128]),
            "context_length_multiplier": ("int", 2, 5),
        }


@register("deepar")
class DeepARModel(GluonTSModel):
    """GluonTS DeepAR model."""

    def _build_estimator(self, params: dict):
        from gluonts.torch.model.deepar import DeepAREstimator

        return DeepAREstimator(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            num_layers=params.get("num_layers", 2),
            hidden_size=params.get("hidden_size", 40),
            freq=self.frequency,
            trainer_kwargs={"max_epochs": self.max_epochs},
        )

    def get_hyperparameter_space(self) -> dict:
        return {
            "num_layers": ("int", 1, 4),
            "hidden_size": ("categorical", [20, 40, 64, 128]),
            "context_length_multiplier": ("int", 2, 5),
        }
