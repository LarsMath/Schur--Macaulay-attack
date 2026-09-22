from itertools import combinations, combinations_with_replacement
from sage.all import matrix
from mathematics.combinatorics import *

# ======================================================================================

# All constructions are assumed to be in char 2!
# Signs are not necessarily correct...

# ==================================== Helper functions ========================================

def sym_basis(n, a):
    return list(combinations_with_replacement(range(n), a))

def wedge_basis(n, b):
    return list(combinations(range(n), b))

def tensor_index(*bases):
    dicts = [{label: pos for pos, label in enumerate(b)} for b in bases]
    dimensions = [len(b) for b in bases]
    def index(*labels):
        idx = 0
        for dic, dimension, label in zip(dicts, dimensions, labels):
            idx = idx * dimension + dic[label]
        return idx
    return index

# ====================================== Divided square ==============================================

def index_splits(n, a):
    return [(s, tuple(i for i in range(n) if i not in s)) for s in combinations(range(n), a)]

def tuple_slice(I, s):
    return tuple(I[i] for i in s)

# Returns 2d-form
# Only works for exterior powers
def divided_square(w, n, d):
    assert d % 2 == 0
    splits = index_splits(2*d-1, d-1)
    return {s: sum(w[(s[0],) + tuple_slice(s[1:], s1)] * w[tuple_slice(s[1:], s2)] for s1, s2 in splits) for s in combinations(range(n), 2*d)}

# ====================================== GL-equivariant maps ===========================================

"""
    θ_l: Λ^{a+l} V ⊗ Λ^{b-l} V → Λ^a V ⊗ Λ^b V
"""
def theta_l(n, a, b, l, F):

    domain_index = tensor_index(wedge_basis(n, a + l), wedge_basis(n, b - l))
    codomain_index = tensor_index(wedge_basis(n, a), wedge_basis(n, b))

    entries = {}
    for I in wedge_basis(n, a+l):
        for J in wedge_basis(n, b-l):
            column_index = domain_index(I, J)
            entries.update({(codomain_index(tuple(i for i in I if i not in K), tuple(sorted(J + K))), column_index): 1 for K in combinations([i for i in I if i not in J], l)})

    return matrix(F, dim_exts((n, n), (a, b)), dim_exts((n, n), (a+l, b-l)), entries)

"""
    ∂: Sym^{a-1} V ⊗ Λ^{b+1} V → Sym^a V ⊗ Λ^b V
"""
def koszul_differential(n, a, b, F):

    domain_index = tensor_index(sym_basis(n, a-1), wedge_basis(n, b+1))
    codomain_index = tensor_index(sym_basis(n, a), wedge_basis(n, b))

    entries = {}
    for I in sym_basis(n, a-1):
        for J in wedge_basis(n, b+1):
            column_index = domain_index(I, J)
            entries.update({(codomain_index(tuple(sorted(I + (j,))), J[:idx] + J[idx+1:]), column_index): 1 for idx, j in enumerate(J)})

    return matrix(F, dim_K(n, a, b), dim_K(n, a-1, b+1), entries)


# ====================================== multiplication maps ==============================================

# The multiplication maps are bundles for several fs to amortize costs

"""
    input: 
    fs: dicts wedge_basis(n, deg_f) -> F

    output:
    m_fs: ⨁_{f ∈ fs} Λ^k V → Λ^{k + deg_f} V
"""
def wedge_multiplication_map(n, k, deg_f, fs):

    domain_index = tensor_index(wedge_basis(n, k), list(range(len(fs))))
    splits = index_splits(k + deg_f, deg_f)

    entries = {(r, domain_index(tuple_slice(I, s2), l)): f[tuple_slice(I, s1)] for r, I in enumerate(wedge_basis(n, k + deg_f)) for s1, s2 in splits for l, f in enumerate(fs)}
    return matrix(dim_ext(n, k + deg_f), len(fs) * dim_ext(n, k), entries)

"""
    m_fs: ⨁_{f ∈ fs} Λ^{a-1} V ⊗ Λ^{b-1} V → Λ^a V ⊗ Λ^b V
"""
def bilinear_wedge_multiplication(n, a, b, fs):

    domain_index = tensor_index(wedge_basis(n, a-1), wedge_basis(n, b-1), list(range(len(fs))))
    codomain_index = tensor_index(wedge_basis(n, a), wedge_basis(n, b))

    entries = {}
    for I in wedge_basis(n, a):
        for J in wedge_basis(n, b):
            for l, f in enumerate(fs):
                entries.update({(codomain_index(I, J), domain_index(I[:idx] + I[idx+1:], J[:idy] + J[idy+1:], l)): f[x,y] + f[y,x] for idx, x in enumerate(I) for idy, y in enumerate(J)})

    return matrix(dim_exts(n, (a, b)), len(fs) * dim_exts(n, (a - 1, b - 1)), entries)

