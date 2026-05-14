# =========================================================
# LAB 2 — IRIS DATASET WITH SCIKIT-LEARN
# Complete workflow:
# load -> explore -> visualize -> split -> train
# -> predict -> evaluate -> hyperparameter tuning
# =========================================================

# =========================
# IMPORT LIBRARIES
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

print("scikit-learn version:", sklearn.__version__)


# =========================================================
# TASK 1.1 — IMPORT TEST
# =========================================================

from sklearn import datasets, model_selection, neighbors, metrics

print("Moduli importati correttamente")


# =========================================================
# TASK 1.2 — ESTIMATORS
# =========================================================

knn = KNeighborsClassifier(n_neighbors=3)

logreg = LogisticRegression()

tree = DecisionTreeClassifier()

print("\n=== KNN ===")
print(knn)
print(knn.get_params())

print("\n=== LOGISTIC REGRESSION ===")
print(logreg)
print(logreg.get_params())

print("\n=== DECISION TREE ===")
print(tree)
print(tree.get_params())


# =========================================================
# TASK 1.3 — FRUIT CLASSIFICATION
# =========================================================

X_train_fruit = np.array([
    [150, 7],
    [170, 7.5],
    [140, 6.8],
    [200, 8.5],
    [220, 9],
    [180, 8],
])

y_train_fruit = np.array([0, 0, 0, 1, 1, 1])

clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train_fruit, y_train_fruit)

nuovo = np.array([[160, 7.2]])

predizione = clf.predict(nuovo)

print("\nPredizione frutto:", predizione)

nuovo2 = np.array([[210, 8.7]])
print("Predizione:", clf.predict(nuovo2))

nuovo3 = np.array([[180, 7.5]])
print("Predizione:", clf.predict(nuovo3))


# =========================================================
# TASK 2.1 — LOAD IRIS DATASET
# =========================================================

iris = load_iris()

print("\nChiavi disponibili:")
print(iris.keys())

print("\nFeature names:")
print(iris.feature_names)

print("\nTarget names:")
print(iris.target_names)

print("\nShape data:")
print(iris.data.shape)

print("\nShape target:")
print(iris.target.shape)

print("\nPrime 5 righe:")
print(iris.data[:5])

print("\nPrime 10 etichette:")
print(iris.target[:10])

unique, counts = np.unique(
    iris.target,
    return_counts=True
)

print("\nClassi e conteggi:")
print(unique)
print(counts)


# =========================================================
# TASK 2.2 — DATAFRAME + PAIRPLOT
# =========================================================

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = [
    iris.target_names[t]
    for t in iris.target
]

print("\nDataFrame:")
print(df.head())

print("\nConteggio specie:")
print(df["species"].value_counts())

sns.pairplot(
    df,
    hue="species",
    height=2.0
)

plt.suptitle(
    "Iris — Pair Plot",
    y=1.02
)

plt.show()


# =========================================================
# SCATTER PLOT
# =========================================================

colors = {
    "setosa": "red",
    "versicolor": "green",
    "virginica": "blue"
}

plt.figure(figsize=(8, 6))

for specie in df["species"].unique():

    subset = df[df["species"] == specie]

    plt.scatter(
        subset["petal length (cm)"],
        subset["petal width (cm)"],
        label=specie
    )

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")

plt.title("Petal Length vs Petal Width")

plt.legend()

plt.grid(True)

plt.show()


# =========================================================
# TASK 2.3 — TRAIN TEST SPLIT
# =========================================================

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0,
    stratify=y
)

print("\nX_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

train_unique, train_counts = np.unique(
    y_train,
    return_counts=True
)

test_unique, test_counts = np.unique(
    y_test,
    return_counts=True
)

print("\nTrain class distribution:")
print(train_unique, train_counts)

print("\nTest class distribution:")
print(test_unique, test_counts)


# =========================================================
# TASK 2.4 — TRAIN KNN
# =========================================================

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

print("\nModello addestrato:")
print(knn)

print("Numero vicini:", knn.n_neighbors)

print("Classi apprese:", knn.classes_)

train_accuracy = knn.score(X_train, y_train)

print("Training accuracy:", train_accuracy)

knn5 = KNeighborsClassifier(n_neighbors=5)

knn5.fit(X_train, y_train)

print("Training accuracy k=5:")
print(knn5.score(X_train, y_train))


# =========================================================
# TASK 2.5 — PREDICT NEW FLOWERS
# =========================================================

fiore_nuovo = np.array([[5.0, 2.9, 1.0, 0.2]])

prediction = knn.predict(fiore_nuovo)

species = iris.target_names[prediction]

print("\nNuovo fiore:")
print(fiore_nuovo)

print("Classe predetta:")
print(prediction[0], "->", species[0])

flowers = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.0, 5.2, 2.3],
    [5.7, 2.8, 4.1, 1.3]
])

preds = knn.predict(flowers)

print("\nPredizioni multiple:")

for flower, pred in zip(flowers, preds):

    print(
        flower,
        "->",
        iris.target_names[pred]
    )

print("\nProbabilità:")
print(knn.predict_proba(flowers))


# =========================================================
# TASK 2.6 — EVALUATION
# =========================================================

y_pred = knn.predict(X_test)

print("\nAccuracy manuale:")
print(np.mean(y_pred == y_test))

print("\nAccuracy sklearn:")
print(accuracy_score(y_test, y_pred))

print("\nAccuracy score:")
print(knn.score(X_test, y_test))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

report = classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
)

print("\nClassification Report:")
print(report)

disp = ConfusionMatrixDisplay.from_estimator(
    knn,
    X_test,
    y_test,
    display_labels=iris.target_names,
    cmap="Blues"
)

plt.title("Confusion Matrix — kNN")

plt.show()


# =========================================================
# TASK 2.7 — HYPERPARAMETER TUNING
# =========================================================

k_values = range(1, 31)

train_scores = []
test_scores = []

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(X_train, y_train)

    train_scores.append(
        model.score(X_train, y_train)
    )

    test_scores.append(
        model.score(X_test, y_test)
    )

plt.figure(figsize=(10, 5))

plt.plot(
    k_values,
    train_scores,
    "o-",
    label="Training Accuracy"
)

plt.plot(
    k_values,
    test_scores,
    "s-",
    label="Test Accuracy"
)

plt.xlabel("Numero di vicini (k)")

plt.ylabel("Accuracy")

plt.title("Effetto di k sul modello k-NN")

plt.legend()

plt.grid(True, alpha=0.3)

plt.show()

best_k = k_values[
    int(np.argmax(test_scores))
]

print("\nMiglior k trovato:")
print(best_k)

print("Migliore accuracy:")
print(max(test_scores))