"""Main pipeline: orchestrates data loading, training, and evaluation."""

import json
import logging
from pathlib import Path

import pandas as pd

from forecasting.config.schema import ForecastConfig
from forecasting.data.loader import load_data
from forecasting.data.preprocessor import prepare_series, prepare_tabular
from forecasting.evaluation.metrics import evaluate
from forecasting.models.gluonts_model import build_list_dataset
from forecasting.models.registry import get_model_class
from forecasting.training.splitter import split_by_date, split_tabular_by_date
from forecasting.training.tuner import tune

logger = logging.getLogger(__name__)


class ForecastRunner:
    """Runs the full forecasting pipeline for a given config.

    Supports three granularity levels:
    - "store_product": one model per (store, product) pair.
    - "product": one model per product, all stores as separate series.
    - "category": one model per category, all products+stores as series.

    Automatically picks the right data format (pivoted vs tabular)
    based on the model's data_format property.
    """

    def __init__(self, config: ForecastConfig):
        self.config = config
        self.df = None
        self.results = {}

    def _get_groups(self) -> list[dict]:
        """Determine what groups to iterate over based on granularity."""
        granularity = self.config.granularity
        product_col = self.config.data.product_col

        if granularity == "product":
            products = self.df[product_col].unique()
            return [{"product_filter": p} for p in products]

        elif granularity == "category":
            cat_col = self.config.data.category_col
            if cat_col is None:
                raise ValueError(
                    "category_col must be set in DataConfig for "
                    "granularity='category'"
                )
            categories = self.df[cat_col].unique()
            return [{"category_filter": c} for c in categories]

        elif granularity == "store_product":
            store_col = self.config.data.store_col
            pairs = (
                self.df[[product_col, store_col]]
                .drop_duplicates()
                .to_dict("records")
            )
            return [
                {"product_filter": r[product_col]}
                for r in pairs
            ]

        else:
            raise ValueError(f"Unknown granularity: {granularity}")

    def _prepare_gluonts_data(self, group: dict):
        """Prepare pivoted data + GluonTS ListDatasets."""
        pivot = prepare_series(self.df, self.config.data, **group)
        train_pivot, val_pivot, test_pivot = split_by_date(
            pivot, self.config.model.prediction_length
        )

        start_date = pivot.index.min()
        num_stores = len(pivot.columns)
        freq = self.config.data.frequency

        train_ds = build_list_dataset(train_pivot, start_date, freq)
        val_ds = build_list_dataset(val_pivot, start_date, freq)
        test_ds = build_list_dataset(test_pivot, start_date, freq)

        return train_ds, val_ds, test_ds, num_stores, pivot.columns

    def _prepare_tabular_data(self, group: dict):
        """Prepare flat tabular data for tree-based models."""
        tabular = prepare_tabular(self.df, self.config.data, **group)
        train, val, test = split_tabular_by_date(
            tabular,
            self.config.data.date_col,
            self.config.model.prediction_length,
        )

        stores = sorted(tabular[self.config.data.store_col].unique())
        return train, val, test, len(stores), stores

    def run(self) -> pd.DataFrame:
        """Execute the full pipeline.

        Returns:
            DataFrame with all predictions and metrics.
        """
        self.df = load_data(self.config.data)
        groups = self._get_groups()
        all_results = []

        logger.info(
            "Running pipeline: %d groups x %d models",
            len(groups),
            len(self.config.models_to_train),
        )

        for group in groups:
            group_label = str(group)
            logger.info("Processing group: %s", group_label)

            for model_name in self.config.models_to_train:
                logger.info("Training model: %s", model_name)

                model_class = get_model_class(model_name)
                data_format = model_class(
                    prediction_length=self.config.model.prediction_length,
                    context_length=self.config.model.context_length,
                ).data_format

                try:
                    if data_format == "tabular":
                        train, val, test, num_stores, store_ids = (
                            self._prepare_tabular_data(group)
                        )
                    else:
                        train, val, test, num_stores, store_ids = (
                            self._prepare_gluonts_data(group)
                        )
                except ValueError as e:
                    logger.warning("Skipping group %s: %s", group_label, e)
                    continue

                # Tune
                model_kwargs = {
                    "num_stores": num_stores,
                    "frequency": self.config.data.frequency,
                    "max_epochs": self.config.model.max_epochs,
                }
                if data_format == "tabular":
                    model_kwargs["target_col"] = self.config.data.target_col
                    model_kwargs["date_col"] = self.config.data.date_col
                    model_kwargs["store_col"] = self.config.data.store_col

                best = tune(
                    model_class=model_class,
                    train_data=train,
                    val_data=val,
                    prediction_length=self.config.model.prediction_length,
                    context_length=self.config.model.context_length,
                    n_trials=self.config.model.n_trials,
                    **model_kwargs,
                )

                # Train final model with best params
                model = model_class(
                    prediction_length=self.config.model.prediction_length,
                    context_length=self.config.model.context_length,
                    **model_kwargs,
                    **best["best_params"],
                )
                model.fit(train, val)

                # Predict and evaluate
                predictions = model.predict(test)
                predictions["model"] = model_name
                predictions["group"] = group_label
                metrics = evaluate(predictions)
                metrics["model"] = model_name
                metrics["group"] = group_label

                logger.info(
                    "%s | %s | avg RMSE=%.4f",
                    group_label,
                    model_name,
                    metrics["rmse"].mean(),
                )

                all_results.append(predictions)
                self._save_params(group_label, model_name, best)

        if not all_results:
            logger.warning("No results produced.")
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)

    def _save_params(self, group: str, model: str, best: dict) -> None:
        """Save best hyperparameters to disk."""
        out_dir = Path(self.config.output_dir) / model
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_group = group.replace("/", "_").replace(" ", "_")
        path = out_dir / f"best_params_{safe_group}.json"

        with open(path, "w") as f:
            json.dump(best, f, indent=2, default=str)
