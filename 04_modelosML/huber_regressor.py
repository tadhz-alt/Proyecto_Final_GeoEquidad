import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 1. Cargar los datos (Ajusta la ruta si es necesario)
df = pd.read_csv("01_datos/base_limpia.csv")

# Definir X e Y (exactamente como lo tienes)
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

# 2. Configurar la Validación Cruzada (Misma que tu compañero)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. Definir los modelos a comparar
modelos = {
    "1. Lineal Clásica (Base)": LinearRegression(),
    "2. Huber Robusta (Anti-Outliers)": make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35)),
    "3. Gradient Boosting (No Lineal)": HistGradientBoostingRegressor(
        loss='absolute_error', # Ignora los outliers extremos
        max_depth=3,           # Evita sobreajuste
        max_iter=100, 
        learning_rate=0.05, 
        random_state=42
    )
}

# 4. Evaluar y comparar
resultados = []
print("=" * 90)
print("COMPARATIVA DE MODELOS CON VALIDACIÓN CRUZADA (5-FOLD, SEMILLA 42)")
print("=" * 90)

for nombre, modelo in modelos.items():
    # Evaluamos R2 y MAE al mismo tiempo
    scores = cross_validate(modelo, X, y, cv=kf, scoring=['r2', 'neg_mean_absolute_error'])
    
    r2_mean = scores['test_r2'].mean()
    mae_mean = -scores['test_neg_mean_absolute_error'].mean() # Negativo porque sklearn lo devuelve así
    
    resultados.append({
        "Modelo": nombre,
        "R2_CV_Promedio": round(r2_mean, 4),
        "MAE_CV_Promedio": round(mae_mean, 4)
    })

df_resultados = pd.DataFrame(resultados)
print(df_resultados.to_string(index=False))
print("=" * 90)

# Guardar los resultados para el reporte
# df_resultados.to_csv("03_resultados/tabla_21_comparativa_modelos_robustos.csv", index=False)