import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import (
    cross_validate, cross_val_score, KFold, GridSearchCV
)
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path

print("=== cwd ===")
print(os.getcwd())

print("\n=== ¿Qué archivos ve Python aquí? ===")
for f in os.listdir():
    print(repr(f))          # repr() muestra espacios/unicode ocultos

print("\n=== ¿Existe el nombre que escribiste? ===")
ruta = Path("pobreza_municipal_jalisco_2020_limpia.csv").resolve()
print("Ruta absoluta:", ruta)
print("¿Existe?:", ruta.exists())
print("¿Es archivo?:", ruta.is_file())