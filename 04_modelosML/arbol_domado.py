import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
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

# 1. Configurar la Validación Cruzada (Misma semilla para comparar justo)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 2. Crear el Árbol Domesticado
arbol_domado = DecisionTreeRegressor(
    criterion='absolute_error', 
    max_depth=4, 
    min_samples_leaf=5, 
    random_state=42
)

# 3. Evaluar el modelo usando Validación Cruzada
scores = cross_validate(arbol_domado, X, y, cv=kf, scoring=['r2', 'neg_mean_absolute_error'])

r2_promedio = scores['test_r2'].mean()
mae_promedio = -scores['test_neg_mean_absolute_error'].mean()

print("=" * 60)
print("RESULTADOS DEL ÁRBOL DE DECISIÓN (CON VALIDACIÓN CRUZADA)")
print("=" * 60)
print(f"R² Promedio:  {r2_promedio:.4f}")
print(f"MAE Promedio: {mae_promedio:.4f}")
print("=" * 60)

# 4. Entrenar el árbol con TODOS los datos para poder dibujarlo
arbol_domado.fit(X, y)

# 5. ¡El extra! Graficar el árbol para ver cómo tomó sus decisiones
plt.figure(figsize=(20, 10))
plot_tree(
    arbol_domado, 
    feature_names=X.columns, 
    filled=True, 
    rounded=True, 
    fontsize=10,
    impurity=False # Para que el gráfico no se vea tan saturado
)
plt.title("Visualización del Árbol de Decisión (Domado)", fontsize=16, fontweight='bold')
plt.show()