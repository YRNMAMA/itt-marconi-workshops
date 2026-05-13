import numpy as np

# =========================================================
# 1.1 — Creare il primo array
# =========================================================

# Da lista Python a ndarray NumPy
a = np.array([1, 2, 3, 4, 5])

print("=== ARRAY A ===")
print("a =", a)
print("shape:", a.shape)
print("dtype:", a.dtype)
print("ndim :", a.ndim)

# Array di 10 zeri
b = np.zeros(10)

# Array da 0 a 19
c = np.arange(20)

# 5 valori equispaziati tra 0 e 1
d = np.linspace(0, 1, 5)

print("\n=== ARRAY B ===")
print(b)
print("shape:", b.shape)
print("dtype:", b.dtype)

print("\n=== ARRAY C ===")
print(c)
print("shape:", c.shape)
print("dtype:", c.dtype)

print("\n=== ARRAY D ===")
print(d)
print("shape:", d.shape)
print("dtype:", d.dtype)

# Checkpoint richiesti
print("\n=== CHECKPOINT ===")
print("c[5] =", c[5])
print("d[0] =", d[0])
print("d[-1] =", d[-1])

# =========================================================
# 1.2 — Operazioni vettorializzate
# =========================================================

prezzi = np.array([10.0, 15.5, 7.2, 22.0, 9.9])

# IVA al 22%
prezzi_con_iva = prezzi * 1.22

print("\n=== PREZZI CON IVA ===")
print(prezzi_con_iva)

# Somma
somma = np.sum(prezzi)

# Media
media = np.mean(prezzi)

# Deviazione standard
dev_std = np.std(prezzi)

# Max e min
massimo = np.max(prezzi)
minimo = np.min(prezzi)

# Conversione euro -> dollari
prezzi_dollari = prezzi * 1.08

# Sconto del 30%
sconto_30 = prezzi * 0.70

print("\n=== STATISTICHE ===")
print("Somma:", somma)
print("Media:", media)
print("Deviazione standard:", dev_std)
print("Massimo:", massimo)
print("Minimo:", minimo)

print("\n=== EURO -> DOLLARI ===")
print(prezzi_dollari)

print("\n=== SCONTO 30% ===")
print(sconto_30)

# =========================================================
# 1.3 — Slicing e indicizzazione
# =========================================================

arr = np.arange(20)

print("\n=== ARRAY COMPLETO ===")
print(arr)

# Elementi dal 5° al 9°
print("\nElementi 5:10")
print(arr[5:10])

# Elementi > 10
print("\nElementi > 10")
print(arr[arr > 10])

# Primi 5 elementi
print("\nPrimi 5 elementi")
print(arr[:5])

# Ultimi 5 elementi
print("\nUltimi 5 elementi")
print(arr[-5:])

# Elementi pari
print("\nElementi pari")
print(arr[arr % 2 == 0])

# Creazione matrice 4x5
matrice = arr.reshape(4, 5)

print("\n=== MATRICE 4x5 ===")
print(matrice)

# Shape matrice
print("\nShape matrice:")
print(matrice.shape)

# Seconda riga
print("\nSeconda riga:")
print(matrice[1])

# Terza colonna
print("\nTerza colonna:")
print(matrice[:, 2])