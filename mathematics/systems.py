from mathematics.multilinear_algebra import *
from sage.all import zero_matrix, identity_matrix, block_matrix
from .isotropic_system import rewrite
import itertools

"""
    γ: ⨁_{i >= 1, fs} Λ^{a-2^i} V → Λ^a V
"""
def gamma_all(fs, n, a, pre_echolonize=True):
    fs_squares = [fs]
    for i in range(2, a.bit_length()):
        fs_squares += [[divided_square(f, n, 2**(i-1)) for f in fs_squares[-1]]]

    if pre_echolonize:
        return block_matrix([[wedge_multiplication_map(n, a - 2**(i+1), 2**(i+1), fs) for i, fs in enumerate(fs_squares)]]).transpose().echelon_form().transpose()
    else:
        return block_matrix([[wedge_multiplication_map(n, a - 2**(i+1), 2**(i+1), fs) for i, fs in enumerate(fs_squares)]])

"""
    θ: ⨁_{i >= 0} Λ^{a+2^i} V ⊗ Λ^{b-2^i} V → Λ^a V ⊗ Λ^b V
"""
def theta_all(n, a, b, F, pre_echolonize=True):
    if b == 0: return zero_matrix(F, dim_exts(n, (a, b)), 0)
    p = len(F.prime_subfield())
    if pre_echolonize:
        return block_matrix([[theta_l(n, a, b, p**i, F) for i in range(b.bit_length()) if p**i <= b]]).transpose().echelon_form().transpose()
    else:
        return block_matrix([[theta_l(n, a, b, p**i, F) for i in range(b.bit_length()) if p**i <= b]])

# =================================================================================================

def construct_wedge_matrix(fs, n, a, F):
    fs = rewrite(fs, n)
    M = gamma_all(fs, n, a)

    return M

def construct_schur_matrix(fs, v, a, F, projected=True):
    n = v + a[0]
    fs = rewrite(fs, n)

    M = gamma_all(fs, n, a[0])
    M = M.tensor_product(identity_matrix(F, dim_exts(n, a[1:])))

    for i in range(1, len(a)):
        thetas = theta_all(n, a[i-1], a[i], F)
        id_left = identity_matrix(F, dim_exts(n, a[:i-1]))
        id_right = identity_matrix(F, dim_exts(n, a[i+1:]))
        thetas_tensored = id_left.tensor_product(thetas).tensor_product(id_right)
        M = block_matrix([[M, thetas_tensored]])
        del thetas, id_left, id_right, thetas_tensored

    if projected:
        full_bases = [wedge_basis(n, a_i) for a_i in a]
        proj_bases = [wedge_basis(v + a_i, a_i) for a_i in a]

        valid_columns = []

        idx = tensor_index(*full_bases)

        for i, bases in enumerate(itertools.product(*full_bases)):
            assert i == idx(*bases), (i, idx(bases))
            if all(basis in proj_bases[j] for j, basis in enumerate(bases)):
                valid_columns.append(i)

        M = M.matrix_from_rows(valid_columns)

    return M
