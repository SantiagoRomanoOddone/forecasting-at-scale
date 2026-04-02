"""Quick test: XGBoost on one product."""

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
from forecasting.data.preprocessor import prepare_tabular
from forecasting.training.splitter import split_tabular_by_date
from forecasting.models.boosting_model import XGBoostModel
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

# 2. Prepare tabular data for one product
product = 7897395060107
tabular = prepare_tabular(df, data_config, product_filter=product)
print(f"\nTabular shape: {tabular.shape}")
print(f"Feature columns: {[c for c in tabular.columns if c not in ['fecha_comercial', 'codigo_barras_sku', 'nome_sku', 'imp_vta', 'cant_vta', 'vol_vta', 'stock', 'pdv_anonimizado', 'qtd_conteudo_sku']]}")

# 3. Split
prediction_length = 30
train, val, test = split_tabular_by_date(tabular, "fecha_comercial", prediction_length)

# 4. Train XGBoost (defaults, no tuning)
model = XGBoostModel(
    prediction_length=prediction_length,
    context_length=prediction_length * 3,
    target_col="cant_vta",
    date_col="fecha_comercial",
    store_col="pdv_anonimizado",
)
model.fit(train, val)

# 5. Predict
results = model.predict(test)
print(f"\nPredictions shape: {results.shape}")
print(results.head(10))

# 6. Map store_idx back
stores = sorted(test["pdv_anonimizado"].unique())
idx_to_store = {i: s for i, s in enumerate(stores)}
results["store_id"] = results["store_idx"].map(idx_to_store)
results["product"] = product

# 7. Evaluate
metrics = evaluate(results)
metrics["store_id"] = metrics["store_idx"].map(idx_to_store)
print("\nMetrics per store:")
print(metrics.to_string(index=False))

# 8. Save
os.makedirs("results", exist_ok=True)
results.to_csv("results/test_xgboost_predictions.csv", index=False)
metrics.to_csv("results/test_xgboost_metrics.csv", index=False)
print("\nSaved: results/test_xgboost_predictions.csv")
print("Saved: results/test_xgboost_metrics.csv")
