import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, cross_val_predict, KFold

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

modelo = LinearRegression(fit_intercept=True)

# 2. Configura K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. Define y evalúa las métricas de regresión
metricas = ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']
resultados = cross_validate(modelo, X, y, cv=kf, scoring=metricas)

mae = -resultados['test_neg_mean_absolute_error'].mean()
rmse = -resultados['test_neg_root_mean_squared_error'].mean()
r2 = resultados['test_r2'].mean()

print("--- Regresión Lineal Múltiple ---")
print(f"MAE promedio: {mae:.4f}")
print(f"RMSE promedio: {rmse:.4f}")
print(f"R² promedio: {r2:.4f}")

# 4. Obtén las predicciones fuera de muestra usando el MISMO objeto 'kf'
predicciones_cv = cross_val_predict(modelo, X, y, cv=kf)
residuos_cv = y - predicciones_cv

# 5. Genera el gráfico Q-Q
fig, ax = plt.subplots(figsize=(6, 4))
stats.probplot(residuos_cv, dist="norm", plot=ax)

ax.set_title("Gráfico Q-Q de Residuos (Regresión Lineal Múltiple)")
ax.set_xlabel("Cuantiles teóricos (Normal)")
ax.set_ylabel("Cuantiles de los residuos")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
