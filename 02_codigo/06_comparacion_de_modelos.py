import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar resultados
ruta_csv = "03_resultados/tabla_final_comparativa_modelos.csv"
df = pd.read_csv(ruta_csv, index_col=0) 

sns.set_theme(style="whitegrid")
colores = sns.color_palette("viridis", len(df))

# ==============================================================================
# GRÁFICO 1: COMPARATIVA DE R²
# ==============================================================================
plt.figure(figsize=(10, 6))

# Ordenar datos de mayor a menor
df_sorted = df.sort_values(by="R2_Promedio", ascending=False)

# Generar barras horizontales con desviación estándar
plt.barh(df_sorted['Modelo'], df_sorted['R2_Promedio'], 
            xerr=df_sorted['R2_Desv_Std'], capsize=5, color=colores, edgecolor='black')

# Invertir eje Y para que el valor más alto aparezca en la parte superior
plt.gca().invert_yaxis()

plt.title("Comparativa de Modelos: R² en Validación Cruzada", fontsize=14, fontweight='bold')
plt.xlabel("R² (Coeficiente de Determinación)", fontsize=11)
plt.ylabel("Modelos", fontsize=11)
plt.axvline(0, color='black', linewidth=1.2) 

# Añadir anotación al modelo con mejor rendimiento
mejor_modelo = df_sorted.iloc[0]['Modelo']
mejor_r2 = df_sorted.iloc[0]['R2_Promedio']

plt.text(mejor_r2 + 0.02, 0, f"Mejor modelo:\n{mejor_modelo}", 
            va='center', color='darkgreen', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig("03_resultados/figura_comparativa_r2.png", dpi=300)
plt.show()

# ==============================================================================
# GRÁFICO 2: COMPARATIVA DE ERRORES
# ==============================================================================
plt.figure(figsize=(12, 6))
x = np.arange(len(df_sorted))
width = 0.35 

# Graficar MAE y RMSE
plt.bar(x - width/2, df_sorted['MAE_Promedio'], width, label='MAE', color='skyblue', edgecolor='black')
plt.bar(x + width/2, df_sorted['RMSE_Promedio'], width, label='RMSE', color='salmon', edgecolor='black')

plt.title("Comparativa de Errores (MAE y RMSE)", fontsize=14, fontweight='bold')
plt.xlabel("Modelos (Ordenados por R²)", fontsize=11)
plt.ylabel("Error (Puntos Porcentuales)", fontsize=11)
plt.xticks(x, df_sorted['Modelo'], rotation=15)
plt.legend()

plt.tight_layout()
plt.savefig("03_resultados/figura_comparativa_errores.png", dpi=300)
plt.show()