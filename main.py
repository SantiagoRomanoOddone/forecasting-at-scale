"""Entry point for the forecasting framework."""

import logging

from forecasting.config.schema import DataConfig, ForecastConfig, ModelConfig
from forecasting.pipeline.runner import ForecastRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    config = ForecastConfig(
        data=DataConfig(
            path="data/extraccion_cerveza_refrigerantes.csv",
            date_col="fecha_comercial",
            target_col="cant_vta",
            product_col="codigo_barras_sku",
            store_col="pdv_anonimizado",
            sep=";",
        ),
        model=ModelConfig(
            prediction_length=30,
            context_length_multiplier=3,
            n_trials=5,
            max_epochs=20,
        ),
        models_to_train=["simple_feedforward"],
        granularity="product",
    )

    runner = ForecastRunner(config)
    results = runner.run()

    print(f"\nDone. Total predictions: {len(results)}")
    print(results.head(10))


if __name__ == "__main__":
    main()
