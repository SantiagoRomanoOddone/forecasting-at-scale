"""XGBoost model for demand forecasting."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from forecasting.models.base import BaseModel
from forecasting.models.registry import register

logger = logging.getLogger(__name__)

# Columns that are NOT features for the model
NON_FEATURE_COLS = [
    "fecha_comercial", "codigo_barras_sku", "nome_sku",
    "imp_vta", "cant_vta", "vol_vta", "stock",
    "pdv_anonimizado", "qtd_conteudo_sku",
]


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return columns to use as features (exclude metadata + target)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


@register("xgboost")
class XGBoostModel(BaseModel):
    """XGBoost regression model for tabular demand data.

    Expects flat DataFrames with temporal features and one-hot stores.
    """

    def __init__(self, prediction_length: int, context_length: int, **kwargs):
        super().__init__(prediction_length, context_length, **kwargs)
        self.model = None
        self._feature_cols = None
        self._target_col = kwargs.get("target_col", "cant_vta")
        self._date_col = kwargs.get("date_col", "fecha_comercial")
        self._store_col = kwargs.get("store_col", "pdv_anonimizado")

    @property
    def data_format(self) -> str:
        return "tabular"

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame = None):
        """Train XGBoost on tabular data.

        Args:
            train_data: Training DataFrame with features and target.
            val_data: Validation DataFrame for early stopping.

        Returns:
            self
        """
        self._feature_cols = _get_feature_cols(train_data)

        X_train = train_data[self._feature_cols]
        y_train = train_data[self._target_col]

        xgb_params = {
            "n_estimators": self.params.get("n_estimators", 500),
            "max_depth": self.params.get("max_depth", 6),
            "learning_rate": self.params.get("learning_rate", 0.05),
            "gamma": self.params.get("gamma", 0),
            "min_child_weight": self.params.get("min_child_weight", 1),
            "subsample": self.params.get("subsample", 0.8),
            "colsample_bytree": self.params.get("colsample_bytree", 0.8),
            "objective": "reg:squarederror",
            "random_state": 42,
        }

        self.model = xgb.XGBRegressor(**xgb_params)

        fit_params = {}
        if val_data is not None:
            X_val = val_data[self._feature_cols]
            y_val = val_data[self._target_col]
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        self.model.fit(X_train, y_train, **fit_params)
        self._fitted = True
        logger.info("XGBoost training complete.")
        return self

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions on tabular test data.

        Args:
            test_data: DataFrame with same feature columns as training.

        Returns:
            DataFrame with columns: store_idx, date, prediction, actual.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        X_test = test_data[self._feature_cols]
        y_pred = self.model.predict(X_test)
        y_pred = np.clip(y_pred, 0, None).astype(float)

        # Build store index mapping
        stores = sorted(test_data[self._store_col].unique())
        store_to_idx = {s: i for i, s in enumerate(stores)}

        results = pd.DataFrame({
            "store_idx": test_data[self._store_col].map(store_to_idx).values,
            "date": test_data[self._date_col].values,
            "prediction": y_pred,
            "actual": test_data[self._target_col].values,
        })

        return results

    def get_hyperparameter_space(self) -> dict:
        return {
            "n_estimators": ("int", 50, 1000),
            "max_depth": ("int", 3, 15),
            "learning_rate": ("float", 0.001, 0.3),
            "gamma": ("float", 0.0, 5.0),
            "min_child_weight": ("int", 1, 10),
            "subsample": ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
        }
