import math
from functools import cache
from .algorithms import signed_sort, _broadcast_n
from itertools import combinations

def dim_ext(n, a):
    return math.comb(n, a) if a >= 0 else 0

def dim_exts(n, a):
    n = _broadcast_n(n, a)
    return math.prod(map(dim_ext, n, a))

@cache
def Jacobi_Trudi_series(n, d):
    if any(d_i < 0 for d_i in d): return 0

    l = len(d)
    if l == 0: return 1
    if l == 1: return dim_ext(n[0], d[0])

    result = 0
    for i in range(min(l, d[0]+1)):
        multiplier = (-1)**(l-1-i) * dim_ext(n[0], d[0]-i)
        for s in combinations(range(l-1), i):
            D = tuple(d_k - (0 if x in s else 1) for x, d_k in enumerate(d[1:]))
            n_new, d_new, s = signed_sort(n[1:], D)
            if s != 0:
                result += s * multiplier * Jacobi_Trudi_series(n_new, d_new)

    return result

def dim_schur(n, a):
    n = _broadcast_n(n, a)
    n, d, s = signed_sort(n, tuple(x + (len(a) - 1 - i) for i, x in enumerate(a)))
    return s * Jacobi_Trudi_series(n, d)

