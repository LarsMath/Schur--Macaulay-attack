from sage.all import matrix, random_matrix, zero_matrix
import math
from itertools import combinations_with_replacement

def random_system(k, n, m):
    fs, _ = random_isotropic_system(k, n, m, 0)
    return fs

def full_isotropic_system(k, n, o):
    return random_isotropic_system(k, n, math.comb(n+1, 2) - math.comb(o+1, 2), o)
  
# Returns a random o-dimensional subspace A of F^n together with m independent random generators of I_2(A) (in upper triangular matrix form)
def random_isotropic_system(F, n, m, o):
    assert m <= math.comb(n+1, 2) - math.comb(o+1, 2)

    I2A = [matrix(F, n, n, lambda i2, j2: 1 if i == i2 and j == j2 else 0) for i in range(n) for j in range(max(o, i), n)]
    A = matrix(F, n, o, lambda i, j: 1 if i == j else 0)

    while not (S := random_matrix(F, n, n)).is_invertible(): continue
    I2A = [S.transpose() * f * S for f in I2A]
    I2A = [f + matrix(F, n, n, lambda i, j: -f[i][j] if i > j else (f[j][i] if j > i else 0)) for f in I2A]
    A = S.inverse() * A

    assert all(A.transpose() * (f + f.transpose()) * A == zero_matrix(F, o, o) for f in I2A)
    assert all(B[i,i] == 0 for f in I2A for B in [A.transpose() * f * A] for i in range(o))

    # Pick random subspace of I2A
    while not (B := random_matrix(F, m, len(I2A))).rank() == m: continue
    I2 = [sum(B[i][j] * f for j, f in enumerate(I2A)) for i in range(m)]

    return I2, A

def rewrite(fs, n):
    return [{ij: f[ij[0],ij[1]] for ij in combinations_with_replacement(range(n), 2)} for f in fs]
