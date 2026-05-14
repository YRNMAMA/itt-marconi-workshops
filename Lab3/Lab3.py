# ============================================================
# LAB 3 — Regressione con Boston Housing
# Codice completo fino all'esercizio 2.7
# ============================================================

# ============================================================
# 1. IMPORT LIBRERIE
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("scikit-learn version:", sklearn.__version__)


# ============================================================
# ESERCIZIO 1 — HELLO WORLD REGRESSION
# ============================================================

# ------------------------------------------------------------
# 1.2 — Estimator di regressione
# ------------------------------------------------------------

print("\n================ ESTIMATOR =================\n")

models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    KNeighborsRegressor()
]

for model in models:
    print(type(model).__name__)
    print(model.get_params())
    print()


# ------------------------------------------------------------
# 1.3 — Dataset giocattolo
# ------------------------------------------------------------

print("\n================ DATASET GIOCATTOLO =================\n")

X_train_small = np.array([
    [50, 2],
    [80, 3],
    [100, 4],
    [60, 2],
    [120, 5],
    [90, 3]
])

y_train_small = np.array([
    150,
    220,
    290,
    170,
    350,
    250
])

reg = LinearRegression()

reg.fit(X_train_small, y_train_small)

nuovo = np.array([[75, 3]])

print("Prezzo stimato:", reg.predict(nuovo))

print("Coefficienti:", reg.coef_)

print("Intercetta:", reg.intercept_)

# Altri esempi

print("\nNuova casa [110, 4]")
print(reg.predict([[110, 4]]))

print("\nNuova casa [45, 1]")
print(reg.predict([[45, 1]]))


# ============================================================
# ESERCIZIO 2 — BOSTON HOUSING
# ============================================================

# ------------------------------------------------------------
# 2.1 — Caricare dataset
# ------------------------------------------------------------

print("\n================ LOAD DATASET =================\n")

boston = fetch_openml(
    name="boston",
    version=1,
    as_frame=True,
    parser="auto"
)

df = boston.frame.copy()

# colonne maiuscole
df.columns = [c.upper() for c in df.columns]

print("Shape dataset:", df.shape)

print("\nPrime 5 righe:\n")
print(df.head())

print("\nValori mancanti:\n")
print(df.isnull().sum())

print("\nStatistiche:\n")
print(df.describe().round(2))

print("\nMEDV min:", df["MEDV"].min())
print("MEDV max:", df["MEDV"].max())


# ------------------------------------------------------------
# 2.2 — Visualizzazione dati
# ------------------------------------------------------------

print("\n================ VISUALIZZAZIONE =================\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Istogramma target
axes[0].hist(
    df["MEDV"],
    bins=30,
    edgecolor="white"
)

axes[0].set_title("Distribuzione MEDV")
axes[0].set_xlabel("Valore case (k$)")
axes[0].set_ylabel("Conteggio")

# Heatmap correlazioni
corr = df.corr(numeric_only=True)

sns.heatmap(
    corr[["MEDV"]].sort_values("MEDV"),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    ax=axes[1]
)

axes[1].set_title("Correlazione con MEDV")

plt.tight_layout()
plt.show()


# Scatter plot RM e LSTAT

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, feat in zip(axes, ["RM", "LSTAT"]):

    ax.scatter(
        df[feat],
        df["MEDV"],
        alpha=0.5,
        s=20
    )

    ax.set_xlabel(feat)
    ax.set_ylabel("MEDV")
    ax.set_title(f"{feat} vs MEDV")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 2.3 — Train/Test Split
# ------------------------------------------------------------

print("\n================ TRAIN TEST SPLIT =================\n")

FEATURES = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "LSTAT"
]

X = df[FEATURES].astype(float).values

y = df["MEDV"].astype(float).values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\nMedia train:", y_train.mean())
print("Media test :", y_test.mean())


# ------------------------------------------------------------
# 2.4 — Regressione Lineare
# ------------------------------------------------------------

print("\n================ LINEAR REGRESSION =================\n")

lr = LinearRegression()

lr.fit(X_train, y_train)

print("R² training:", lr.score(X_train, y_train))

coef_df = pd.DataFrame({
    "Feature": FEATURES,
    "Coefficiente": lr.coef_
})

coef_df = coef_df.sort_values("Coefficiente")

print("\nCoefficienti:\n")
print(coef_df)

print("\nIntercetta:", lr.intercept_)


# ------------------------------------------------------------
# 2.5 — Predizioni e residui
# ------------------------------------------------------------

print("\n================ PREDIZIONI =================\n")

y_pred = lr.predict(X_test)

comparison = pd.DataFrame({
    "Reale": y_test[:10].round(1),
    "Predetto": y_pred[:10].round(1)
})

comparison["Errore"] = (
    comparison["Predetto"] - comparison["Reale"]
).round(1)

print(comparison)


# Scatter reale vs predetto

plt.figure(figsize=(6, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5,
    s=20
)

plt.plot(
    [5, 50],
    [5, 50],
    "r--",
    lw=2,
    label="Predizione perfetta"
)

plt.xlabel("Valore reale")
plt.ylabel("Valore predetto")

plt.title("Reale vs Predetto")

plt.legend()

plt.show()


# Residui

residui = y_pred - y_test

plt.figure(figsize=(8, 4))

plt.hist(
    residui,
    bins=25,
    edgecolor="white"
)

plt.axvline(
    0,
    color="red",
    linestyle="--"
)

plt.title("Distribuzione residui")
plt.xlabel("Residuo")

plt.show()


# ------------------------------------------------------------
# 2.6 — Metriche di valutazione
# ------------------------------------------------------------

print("\n================ METRICHE =================\n")

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print(f"MAE  : {mae:.2f}")

print(f"MSE  : {mse:.2f}")

print(f"RMSE : {rmse:.2f}")

print(f"R²   : {r2:.4f}")


# Metriche training

y_train_pred = lr.predict(X_train)

print("\n--- TRAINING ---")

print(
    "MAE:",
    mean_absolute_error(y_train, y_train_pred)
)

print(
    "RMSE:",
    np.sqrt(mean_squared_error(y_train, y_train_pred))
)

print(
    "R²:",
    r2_score(y_train, y_train_pred)
)


# ------------------------------------------------------------
# 2.7 — Ridge e Lasso
# ------------------------------------------------------------

print("\n================ RIDGE E LASSO =================\n")

results = []

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:

    for ModelClass in [Ridge, Lasso]:

        model = ModelClass(alpha=alpha)

        model.fit(X_train, y_train)

        y_pred_model = model.predict(X_test)

        results.append({
            "Modello": type(model).__name__,
            "alpha": alpha,
            "R² train": round(
                model.score(X_train, y_train),
                4
            ),
            "R² test": round(
                model.score(X_test, y_test),
                4
            ),
            "RMSE test": round(
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        y_pred_model
                    )
                ),
                3
            )
        })

results_df = pd.DataFrame(results)

print(results_df)


# ------------------------------------------------------------
# Decision Tree Regressor
# ------------------------------------------------------------

print("\n================ DECISION TREE =================\n")

tree_results = []

for depth in [2, 3, 5, 7, 10, None]:

    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    y_tree_pred = tree.predict(X_test)

    tree_results.append({
        "max_depth": str(depth),
        "R² train": round(
            tree.score(X_train, y_train),
            4
        ),
        "R² test": round(
            tree.score(X_test, y_test),
            4
        ),
        "RMSE test": round(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    y_tree_pred
                )
            ),
            3
        )
    })

tree_df = pd.DataFrame(tree_results)

print(tree_df)


# ------------------------------------------------------------
# Grafico train/test R²
# ------------------------------------------------------------

depths = [2, 3, 5, 7, 10, 15]

train_r2 = []
test_r2 = []

for d in depths:

    model = DecisionTreeRegressor(
        max_depth=d,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_r2.append(
        model.score(X_train, y_train)
    )

    test_r2.append(
        model.score(X_test, y_test)
    )

plt.figure(figsize=(8, 5))

plt.plot(
    depths,
    train_r2,
    "o-",
    label="R² training"
)

plt.plot(
    depths,
    test_r2,
    "s-",
    label="R² test"
)

plt.xlabel("max_depth")

plt.ylabel("R²")

plt.title("Decision Tree Regressor")

plt.legend()

plt.grid(True, alpha=0.3)

plt.show()


# ============================================================
# FINE LAB
# ============================================================

print("\nLab completato ")
