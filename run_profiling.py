"""Run data profiling on the raw dataset."""

import logging
import os

import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

from forecasting.validation.profiler import profile_series

df = pd.read_csv("data/extraccion_cerveza_refrigerantes.csv", sep=";")
df.columns = df.columns.str.lower()
df = df[["pdv_anonimizado", "fecha_comercial", "codigo_barras_sku", "cant_vta"]]

profile = profile_series(
    df,
    date_col="fecha_comercial",
    store_col="pdv_anonimizado",
    product_col="codigo_barras_sku",
    target_col="cant_vta",
)

os.makedirs("results", exist_ok=True)
profile.to_csv("results/series_profile.csv", index=False)

print(f"\nTotal: {len(profile)}")
print(profile["series_class"].value_counts().sort_index().to_string())
print(f"\nSaved: results/series_profile.csv")
