# GeoEquidad: Análisis Estadístico y Modelado de Machine Learning de la Vulnerabilidad por Ingresos y Carencias Sociales en los Municipios de Jalisco.

Este repositorio contiene el proyecto **GeoEquidad**, desarrollado en Python para analizar los factores que influyen en la falta de ingresos en los **125 municipios de Jalisco** utilizando datos oficiales del **INEGI** y del **CONEVAL**. El sistema organiza los municipios según su tamaño de población, revisa qué necesidades básicas pesan más en la economía de las familias y compara distintos modelos de predicción para ayudar a dirigir los apoyos sociales hacia donde más se necesitan.

## Contenido del Repositorio

A continuación, se describen las carpetas y programas principales incluidos en esta entrega:

1. **Preparación y Limpieza de Datos** (`02_codigo/01_exportar_y_verificar_datos.py`, `01_datos/datos_crudos.csv`, `01_datos/base_limpia.csv`, `data/`)
   Reúne los datos oficiales de los 125 municipios de Jalisco, revisa que no haya información faltante o mal escrita y guarda una base limpia lista para analizarse.

2. **Análisis Descriptivo y Clasificación de Municipios** (`02_codigo/02_estadistica_descriptiva.py`)
   Calcula los promedios y datos básicos de cada indicador y divide los 125 municipios en grupos según su población (rurales, medios y urbanos grandes) para que las ciudades grandes no alteren las comparaciones.

3. **Relaciones y Selección de Variables** (`02_codigo/03_correlaciones.py`, `02_codigo/04_seleccion_variables_vif.py`)
   Revisa qué tanto se relacionan las seis carencias básicas con el porcentaje de personas de bajos ingresos y evita repetir variables que miden casi lo mismo.

4. **Pruebas Estadísticas y Comparación de Modelos** (`02_codigo/05_regresion_lineal_y_diagnosticos.py`, `02_codigo/06_comparacion_de_modelos.py`, `02_codigo/diagnostico_env.py`)
   Comprueba que la relación entre carencias e ingresos sea real y evalúa los modelos repitiendo las pruebas en distintos grupos de municipios para medir qué tan bien aciertan y cuánto se equivocan.

5. **Modelos de Predicción** (`04_modelosML/all_models.py`, `04_modelosML/support_vector_regressor.py`, `04_modelosML/huber_regressor.py`, `04_modelosML/random_forest.py`, `04_modelosML/kneighbors_regressor.py`, `04_modelosML/arbol_decision_regresion.py`, `04_modelosML/arbol_domado.py`)
   Entrena y compara distintos programas de predicción para estimar el porcentaje de población con bajos ingresos a partir de las carencias sociales.

6. **Resultados, Gráficas y Reporte Final** (`03_resultados/`, `05_reporte/Reporte Final Proyecto Integrador GeoEquidad.pdf`, `05_reporte/verificacion_versiones.png`)
   Guarda las tablas de resultados, las gráficas comparativas en imagen, las capturas de revisión y el documento final en PDF del proyecto.

---

## Resultados Principales

Al probar los modelos en los 125 municipios de Jalisco, los programas más estables y precisos fueron:

* **SVR Lineal (modelo más preciso):** Explica el **33.35%** del problema de bajos ingresos con un error promedio de **4.89%** y error en casos extremos de **6.74%**.
* **HuberRegressor:** Explica el **31.18%** con un error promedio de **4.96%** y error en casos extremos de **6.89%**.
* **Random Forest:** Explica el **12.50%** con un error promedio de **5.54%** y error en casos extremos de **7.76%**.
* **KNN (20 vecinos):** Explica el **10.20%** con un error promedio de **5.70%** y error en casos extremos de **8.01%**.
* **Árbol de Decisión:** Obtiene **-12.50%** con un error promedio de **6.13%** y error en casos extremos de **8.54%**.

Los resultados muestran que la **mala calidad y falta de espacio en la vivienda** y la **falta de acceso a la alimentación** son las dos carencias que más influyen en la falta de ingresos municipales, y que modelos como **SVR Lineal** y **HuberRegressor** funcionan mejor que los modelos basados en árboles cuando se trabaja con 125 municipios.

---

## Herramientas e Infraestructura

* **Lenguaje:** Python 3.10 o superior.
* **Librerías principales:** `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn` y `matplotlib` (listadas en `requirements.txt`).
* **Equipo y entorno de trabajo en el que se trabajo:** Computadora con Windows, Windows PowerShell, Visual Studio Code, Git y GitHub.
* **Archivos excluidos del repositorio:** La carpeta del entorno virtual (`venv/`) y el archivo de variables locales (`.env`) se mantienen fuera de GitHub mediante `.gitignore`.

---

## Cómo Ejecutar el Proyecto

1. **Clonar el repositorio y entrar a la carpeta:**
   ```powershell
   git clone [https://github.com/tadhz-alt/Proyecto_Final_GeoEquidad.git](https://github.com/tadhz-alt/Proyecto_Final_GeoEquidad.git)
   cd Proyecto_Final_GeoEquidad
