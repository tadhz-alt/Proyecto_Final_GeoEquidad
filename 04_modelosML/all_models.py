import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 1. Cargar datos
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

# 2. Definir el diccionario con TODOS los modelos que probaste
modelos = {
    "SVR (Lineal)": make_pipeline(StandardScaler(), SVR(kernel="linear")),
    "HuberRegressor": make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35)),
    "Regresión Lineal (OLS)": LinearRegression(),
    "Random Forest": RandomForestRegressor(criterion='absolute_error', n_estimators=100, max_depth=4, min_samples_leaf=5, random_state=42),
    "KNN (n=20)": KNeighborsRegressor(n_neighbors=20, p=1),
    "Árbol de Decisión": DecisionTreeRegressor(criterion='absolute_error', max_depth=4, min_samples_leaf=5, random_state=42)
}

# 3. Definir las métricas a evaluar
# Nota: Scikit-learn devuelve las métricas de error en negativo para que "más alto sea mejor", 
# por eso luego las multiplicamos por -1.
metricas = {
    'r2': 'r2',
    'mae': 'neg_mean_absolute_error',
    'rmse': 'neg_root_mean_squared_error',
    'mape': 'neg_mean_absolute_percentage_error'
}

resultados = []

# 4. Evaluar todos los modelos
print("Calculando métricas... (esto puede tardar unos segundos por el Random Forest)")
for nombre, modelo in modelos.items():
    scores = cross_validate(modelo, X, y, cv=kf, scoring=list(metricas.values()))
    
    # Extraer promedios y desviaciones estándar
    r2_mean = scores['test_r2'].mean()
    r2_std = scores['test_r2'].std()
    
    mae_mean = -scores['test_neg_mean_absolute_error'].mean()
    mae_std = scores['test_neg_mean_absolute_error'].std()
    
    rmse_mean = -scores['test_neg_root_mean_squared_error'].mean()
    rmse_std = scores['test_neg_root_mean_squared_error'].std()
    
    mape_mean = -scores['test_neg_mean_absolute_percentage_error'].mean() * 100 # Lo pasamos a porcentaje real
    
    resultados.append({
        "Modelo": nombre,
        "R2_Promedio": round(r2_mean, 4),
        "R2_Desv_Std": round(r2_std, 4),
        "MAE_Promedio": round(mae_mean, 4),
        "MAE_Desv_Std": round(mae_std, 4),
        "RMSE_Promedio": round(rmse_mean, 4),
        "RMSE_Desv_Std": round(rmse_std, 4),
        "MAPE_Porcentaje": round(mape_mean, 2)
    })

# 5. Crear DataFrame, ordenarlo por el mejor R2 y guardarlo
df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values(by="R2_Promedio", ascending=False).reset_index(drop=True)

# Agregar una columna de "Posición" (Ranking)
df_resultados.index = df_resultados.index + 1
df_resultados.index.name = "Ranking"

# Guardar en CSV
df_resultados.to_csv("03_resultados/tabla_final_comparativa_modelos.csv", encoding="utf-8-sig")

print("\n" + "=" * 100)
print("TABLA FINAL COMPARATIVA DE MODELOS")
print("=" * 100)
print(df_resultados.to_string())
print("=" * 100)
print("Archivo guardado exitosamente en: 03_resultados/tabla_final_comparativa_modelos.csv")