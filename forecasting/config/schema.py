"""Configuration schema for the forecasting framework."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """Defines how to read and interpret the client's data."""

    path: str
    date_col: str
    target_col: str
    product_col: str
    store_col: str
    sep: str = ","
    frequency: str = "D"
    category_col: Optional[str] = None


@dataclass
class ModelConfig:
    """Defines model training parameters."""

    prediction_length: int = 30
    context_length_multiplier: int = 3
    n_trials: int = 10
    max_epochs: int = 50
    early_stopping_patience: int = 5
    batch_size: int = 32

    @property
    def context_length(self) -> int:
        return self.prediction_length * self.context_length_multiplier


@dataclass
class ForecastConfig:
    """Top-level configuration that ties everything together."""

    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    models_to_train: list[str] = field(
        default_factory=lambda: ["simple_feedforward"]
    )
    granularity: str = "product"  # "store_product", "product", "category"
    output_dir: str = "results"
