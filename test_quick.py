"""Quick test: one product, one run, no tuning."""

import logging
import os
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

from forecasting.config.schema import DataConfig
from forecasting.data.loader import load_data
from forecasting.data.preprocessor import prepare_series
from forecasting.training.splitter import split_by_date
from forecasting.models.gluonts_model import build_list_dataset, SimpleFeedforwardModel
from forecasting.evaluation.metrics import evaluate

# 1. Load data
data_config = DataConfig(
    path="data/extraccion_cerveza_refrigerantes.csv",
    date_col="fecha_comercial",
    target_col="cant_vta",
    product_col="codigo_barras_sku",
    store_col="pdv_anonimizado",
    sep=";",
)

df = load_data(data_config)
df = df[['pdv_anonimizado', 'fecha_comercial', 'codigo_barras_sku', 'cant_vta']]

# 2. Pick a product with high sales in test period
product = 7897395060107
pivot = prepare_series(df, data_config, product_filter=product)
print(f"\nPivot shape: {pivot.shape} (days x stores)")
print(pivot.head())

# 3. Split
prediction_length = 30
train, val, test = split_by_date(pivot, prediction_length)

# 4. Build GluonTS datasets
start = pivot.index.min()
train_ds = build_list_dataset(train, start, "D")
val_ds = build_list_dataset(val, start, "D")
test_ds = build_list_dataset(test, start, "D")

# 5. Train (no tuning, just defaults, few epochs)
model = SimpleFeedforwardModel(
    prediction_length=prediction_length,
    context_length=prediction_length * 3,
    num_stores=len(pivot.columns),
    frequency="D",
    max_epochs=5,
)
model.fit(train_ds, val_ds)

# 6. Predict
results = model.predict(test_ds)

# 7. Map store_idx back to store IDs
store_map = {idx: col for idx, col in enumerate(pivot.columns)}
results["store_id"] = results["store_idx"].map(store_map)
results["product"] = product

# 8. Evaluate
metrics = evaluate(results)
metrics["store_id"] = metrics["store_idx"].map(store_map)
print("\nMetrics per store:")
print(metrics.to_string(index=False))

# 9. Save results
os.makedirs("results", exist_ok=True)
results.to_csv("results/test_predictions.csv", index=False)
metrics.to_csv("results/test_metrics.csv", index=False)
print("\nSaved: results/test_predictions.csv")
print("Saved: results/test_metrics.csv")
