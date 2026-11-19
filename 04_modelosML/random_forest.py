import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
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

# 1. Configurar la Validación Cruzada (Misma semilla)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 2. Crear el Random Forest (Bosque Aleatorio)
bosque = RandomForestRegressor(
    criterion='absolute_error', # Sigue siendo robusto a los outliers (Mezquitic)
    n_estimators=100,           # 100 árboles (expertos) votando
    max_depth=4,                # Mantenemos los árboles simples para que no memoricen
    min_samples_leaf=5,         # Mínimo 5 municipios por hoja
    random_state=42,            # Para que siempre dé el mismo resultado
    n_jobs=-1                   # Usa todos los núcleos de tu procesador (es rápido)
)

# 3. Evaluar con Validación Cruzada
scores = cross_validate(bosque, X, y, cv=kf, scoring=['r2', 'neg_mean_absolute_error'])

r2_promedio = scores['test_r2'].mean()
mae_promedio = -scores['test_neg_mean_absolute_error'].mean()

print("=" * 60)
print("RESULTADOS DEL RANDOM FOREST (CON VALIDACIÓN CRUZADA)")
print("=" * 60)
print(f"R² Promedio:  {r2_promedio:.4f}")
print(f"MAE Promedio: {mae_promedio:.4f}")
print("=" * 60)

# 4. Entrenar con todos los datos para ver la Importancia de las Variables
bosque.fit(X, y)

# Extraer y ordenar la importancia de las variables
importancias = bosque.feature_importances_
df_importancia = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=True) # Orden ascendente para graficar

# 5. Graficar la importancia
plt.figure(figsize=(10, 6))
plt.barh(df_importancia['Variable'], df_importancia['Importancia'], color='forestgreen', edgecolor='black')
plt.title("Importancia de las Variables según Random Forest", fontsize=14, fontweight='bold')
plt.xlabel("Nivel de Importancia (0 a 1)")
plt.ylabel("Variables Predictoras")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Imprimir la tabla en consola
print("\nIMPORTANCIA DE LAS VARIABLES:")
print(df_importancia.sort_values(by='Importancia', ascending=False).to_string(index=False))