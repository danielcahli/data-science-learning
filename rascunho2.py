import numpy as np
from scipy.linalg import lu

# Gerar uma matriz aleatória 3x3 com valores entre 0 e 1
A = np.random.rand(3, 3)
print("Matriz A:")
print(A)

# Decomposição LU: PA = LU
P, L, U = lu(A)

print("\nMatriz U (triangular superior):")
print(U)

# Pivôs estão na diagonal principal de U
pivots = np.abs(np.diag(U))

print("\nPivôs (valores absolutos da diagonal de U):")
print(pivots)

# Comparando o primeiro pivô com abs(A[0, 0])
print(f"\nabs(A[0,0]) = {abs(A[0,0]):.4f}")
print(f"Primeiro pivô (|U[0,0]|) = {pivots[0]:.4f}")