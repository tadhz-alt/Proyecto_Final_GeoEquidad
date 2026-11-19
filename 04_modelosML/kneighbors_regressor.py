import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_validate

df = pd.read_csv("01_datos/base_limpia.csv")
y = df["ingreso_inferior_lpi_porcentaje"]
X = df[[
    "carencia_vivienda_porcentaje",
    "carencia_servicios_basicos_porcentaje",
    "carencia_alimentacion_porcentaje",
    "rezago_educativo_porcentaje",
    "carencia_salud_porcentaje",
    "carencia_seguridad_social_porcentaje",
    "ind_urbano_grande",
    "ind_semiurbano_medio"
]]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Hiperparametros: Numeros de vecinos, el parámetro de potencia para la métrica de Minkowski (p=1 Manhattan)
modelo = KNeighborsRegressor(n_neighbors=20, p=1)

scores = cross_validate(modelo, X, y, cv=kf, scoring=["r2", 'neg_mean_absolute_error'])

r2_promedio = scores['test_r2'].mean()
mae_promedio = -scores['test_neg_mean_absolute_error'].mean()

print("=" * 60)
print("RESULTADOS DEL KNeighbors Regressor (CON VALIDACIÓN CRUZADA)")
print("=" * 60)
print(f"R² Promedio:  {r2_promedio:.4f}")
print(f"MAE Promedio: {mae_promedio:.4f}")
print("=" * 60)