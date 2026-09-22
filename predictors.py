import math
from functools import cache
from itertools import combinations

from mathematics.combinatorics import *
from mathematics.algorithms import signed_sort, bounded_sequences, _broadcast_n

# ==================================== Wedges =====================================

@cache
def wedge_prediction(n, o, m, divided_powers=True):
    if divided_powers:
        return sum((-1)**i * math.comb(m, i) * dim_ext(n, o - 2*i) for i in range(o//2 + 1))
    else:
        return sum((-1)**i * math.comb(m + i - 1, i) * dim_ext(n, o - 2*i) for i in range(o//2 + 1))


# ======================================== Schur ===============================

@cache
def schur_series(n, d, m, ODD=False):
    if any(d_i < 0 for d_i in d): return 0

    l = len(d)
    if l == 0: return 1
    if l == 1: return dim_ext(n[0], d[0]) if ODD else wedge_prediction(n[0], d[0], m)

    result = 0
    for i in range(min(l, d[0]+1)):
        for j in range(d[0] - i + 1):
            multiplier = (-1)**(l-1-i+j) * (dim_ext(n[0], d[0] - i - j) if ODD else wedge_prediction(n[0], d[0] - i - j, m))
            for s in combinations(range(l-1), i):
                D = tuple(d_k - (0 if x in s else 1) for x, d_k in enumerate(d[1:]))
                for t in bounded_sequences(D, j):
                    d_new = tuple(d_k - t_k for d_k, t_k in zip(D, t))
                    n_new, d_new, s = signed_sort(n[1:], d_new)
                    if s != 0:
                        result += s * multiplier * math.prod(math.comb(m, y) for y in t) * schur_series(n_new, d_new, m, ODD=ODD)
    return result

def schur_prediction(n, a, m, ODD=False):
    n = _broadcast_n(n, a)
    n, d, s = signed_sort(n, tuple(x + (len(a) - 1 - i) for i, x in enumerate(a)))
    return max(0, s * schur_series(n, d, m, ODD=ODD))

# ======================================== SNOVA ===============================

# We do not consider Schur modules, just divided powers
@cache
def snova_series(n, d, m, divided_powers):
    assert len(n) == len(d)
    if (l := len(d)) == 0: return 1

    if l == 1: return wedge_prediction(n[0], d[0], m, divided_powers=divided_powers)

    result = 0
    for j in range(d[0] + 1): # Amount of degrees assigned to cross-factors
        multiplier = (-1)**j * wedge_prediction(n[0], d[0] - j, m, divided_powers = divided_powers)
        for t in bounded_sequences(d[1:], j):
            d_new = tuple(d_k - t_k for d_k, t_k in zip(d[1:], t))
            n_new, d_new, _ = signed_sort(n[1:], d_new)
            if divided_powers:
                result += multiplier * math.prod(math.comb(2*m, y) for y in t) * snova_series(n_new, d_new, m, divided_powers)
            else:
                result += multiplier * math.prod(math.comb(2*m + y - 1, y) for y in t) * snova_series(n_new, d_new, m, divided_powers)

    return result
