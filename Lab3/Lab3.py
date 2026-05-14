import sklearn, numpy as np, pandas as pd, matplotlib.pyplot as plt
print("scikit-learn:", sklearn.__version__)

from sklearn import datasets, model_selection, linear_model, metrics
print("Moduli importati correttamente")

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

for model in [LinearRegression(), DecisionTreeRegressor(), KNeighborsRegressor()]:
    print(type(model).__name__, "→", model.get_params())
    print()
