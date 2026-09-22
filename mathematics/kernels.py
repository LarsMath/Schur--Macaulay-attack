from sage.all import matrix

import math
from itertools import combinations

# ==========================================================================================

def construct_wedge_kernel(A, a):
    F, n, o = A.base_ring(), A.nrows(), A.ncols()

    return matrix(F, math.comb(o, a), math.comb(n, a), [[A.matrix_from_rows_and_columns(s2, s1).det() for s2 in combinations(range(n), a)] for s1 in combinations(range(o), a)])

def construct_schur_kernel(A, a):
    F, n, o = A.base_ring(), A.nrows(), A.ncols()

    A = A.transpose()[:, ::-1].echelon_form()[:, ::-1].transpose()

    components = [construct_wedge_kernel(A.matrix_from_rows_and_columns(range(n-o+a_i), range(o-a_i, o)), a_i) for a_i in a]

    result = components[0]
    for M in components[1:]:
        result = result.tensor_product(M)

    return result