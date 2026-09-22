"""End-to-end MAYO key recovery on toy parameters.

    sage mayo_attack_demo.py          # every configuration below
    sage mayo_attack_demo.py A        # just configuration A

Three configurations, smallest first.  All three have m/v = 1 exactly, as MAYO's primary
sets do; what separates them is v/o, the ratio that actually resists (see below).
  A  n=12 o=4 m=8    495 rows,  ~1 s      v/o = 2.00   divided powers strictly necessary
  B  n=13 o=4 m=9   9295 rows,  ~35 s     v/o = 2.25   a second pillar strictly necessary
  C  n=19 o=5 m=14 11628 rows,  ~1.5 min  v/o = 2.80   as A, at the best v/o reachable

MAYO's whipping is only needed for *signing*.  The secret is a single oil space O of
dimension o with P|_O = 0, so key recovery is exactly the (v, o, m)-UOV problem for
MAYO's own (n, o, m), and the whipping factor k plays no role.  The script therefore
does keygen and then key recovery from the expanded public key.

Every configuration has lambda_1 >= 4, so the divided-power tower of Section 3 is
actually exercised: gamma is the ordinary wedge multiplication by the m forms together
with a divided square of degree 2^i for each 2 <= i < bitlength(lambda_1) -- for all
three configurations that is exactly one extra block, the divided square of degree 4.
Each attack is run twice, once with the plain wedge block alone and once with the full
tower, so the rank the divided powers contribute is visible.

Fidelity of the toy parameters.  MAYO's four primary sets, the ones show_reference tabulates, all
have v = m exactly, o/m between 0.085 and 0.266, and v/o between 3.8 and 11.8.  The 33
alternative parameter sets are looser: three of them have v != m,
and across all 33 o/m reaches 0.516 while v/o falls to 2.03.  So the toy parameters below, at
o/m 0.36-0.50 and v/o 2.0-2.8, sit inside the ranges spanned by the full proposed family.
"""
import itertools
import math
import sys
import time

from sage.all import (GF, block_matrix, identity_matrix, matrix, random_matrix, vector, zero_matrix)

from mathematics.combinatorics import dim_ext, dim_exts, dim_schur
from mathematics.multilinear_algebra import (divided_square, index_splits,
                                             tensor_index, tuple_slice, wedge_basis)
from predictors import schur_prediction

# configurations
# label, n, o, m, shapes to try, m for the Hilbert-series probe (below saturation)
# Listed in the order they run: A is checkable by hand, B adds the second pillar, C pushes
# v/o as far as m = v allows.  All three have m/v = 1 exactly.
CONFIGS = {
    "A": ("small instance where the divided powers are needed at all",
          12, 4, 8, ((4,),), 6),
    "B": ("second pillar required: one pillar fails even with the tower",
          13, 4, 9, ((4,), (4, 1)), 7),
    "C": ("best v/o reachable at m = v; plain wedge fails, divided powers succeed",
          19, 5, 14, ((5,),), 10),
}

q = 16                      # MAYO's field, for every one of its proposed parameter sets
F = GF(q)

MAYO_REFERENCE = (("MAYO_1", 86, 78, 8), ("MAYO_2", 81, 64, 17),
                  ("MAYO_3", 118, 108, 10), ("MAYO_5", 154, 142, 12))


# matrix assembly
# The library builders hand a dict to matrix(), which yields a SPARSE matrix; rank over
# GF(2^e) is then orders of magnitude slower, and both sparse->dense conversion and
# matrix(..., dict, sparse=False) cost one allocation per CELL rather than per nonzero.
# So we generate the nonzeros of each block ourselves and assign them into a single
# pre-allocated dense matrix, applying the tensor offsets by hand instead of calling
# tensor_product.  

def wedge_block(n, k, deg_f, fs):
    """Nonzeros of m_fs: (+)_{f in fs} Lambda^k V -> Lambda^{k + deg_f} V."""
    col = tensor_index(wedge_basis(n, k), list(range(len(fs))))
    splits = index_splits(k + deg_f, deg_f)
    entries = {}
    for r, I in enumerate(wedge_basis(n, k + deg_f)):
        for s1, s2 in splits:
            tail = tuple_slice(I, s2)
            for l, f in enumerate(fs):
                v = f[tuple_slice(I, s1)]
                if v:
                    entries[(r, col(tail, l))] = v
    return entries, dim_ext(n, k + deg_f), len(fs) * dim_ext(n, k)


def theta_block(n, a, b, l):
    """Nonzeros of theta_l: Lambda^{a+l} (x) Lambda^{b-l} -> Lambda^a (x) Lambda^b."""
    col = tensor_index(wedge_basis(n, a + l), wedge_basis(n, b - l))
    row = tensor_index(wedge_basis(n, a), wedge_basis(n, b))
    entries = {}
    for I in wedge_basis(n, a + l):
        for J in wedge_basis(n, b - l):
            c = col(I, J)
            for K in itertools.combinations([i for i in I if i not in J], l):
                entries[(row(tuple(i for i in I if i not in K),
                             tuple(sorted(J + K))), c)] = 1
    return entries, dim_exts(n, (a, b)), dim_exts(n, (a + l, b - l))


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def macaulay(fs, n, lam, divided_powers=True):
    """[ gamma on the first pillar | theta towers between adjacent pillars ].

    Each block B acts on a contiguous run of pillars and enters as I_left (x) B (x) I_right.
    """
    dims = [dim_ext(n, a) for a in lam]

    blocks = []                                     # (entries, brows, bcols, left, right)
    right = _prod(dims[1:])
    blocks.append(wedge_block(n, lam[0] - 2, 2, fs) + (1, right))
    if divided_powers:
        ds = fs
        for i in range(2, lam[0].bit_length()):
            ds = [divided_square(f, n, 2 ** (i - 1)) for f in ds]
            blocks.append(wedge_block(n, lam[0] - 2 ** i, 2 ** i, ds) + (1, right))
    for i in range(1, len(lam)):
        left, right = _prod(dims[:i - 1]), _prod(dims[i + 1:])
        for l in (2 ** j for j in range(lam[i].bit_length())):
            blocks.append(theta_block(n, lam[i - 1], lam[i], l) + (left, right))

    nrows = _prod(dims)
    ncols = sum(left * bcols * right for _, _, bcols, left, right in blocks)
    mat = matrix(F, nrows, ncols, sparse=False)

    offset = 0
    for entries, brows, bcols, left, right in blocks:
        assert left * brows * right == nrows
        for (i, j), v in entries.items():
            for a in range(left):
                ri, ci = (a * brows + i) * right, (a * bcols + j) * right
                for b in range(right):
                    mat[ri + b, offset + ci + b] = v
        offset += left * bcols * right
    return mat


# Hilbert series
# Two candidate series for a single pillar of degree o.  Plain wedge multiplication by an
# alternating form f is a BOOLEAN operator in characteristic 2 (f wedge f = 0), so each
# form contributes a periodic syzygy and a factor 1/(1+t^2).  The divided-power tower
# replaces that by prod_{j>=1} 1/(1+t^{2^j}) = 1 - t^2. 

def series_plain(n, o, m):
    """[t^o] (1+t)^n / (1+t^2)^m -- boolean operators, no divided powers."""
    return sum((-1) ** i * math.comb(m + i - 1, i) * math.comb(n, o - 2 * i)
               for i in range(o // 2 + 1))


def series_divided(n, o, m):
    """[t^o] (1+t)^n (1-t^2)^m -- the full divided-power tower."""
    return sum((-1) ** i * math.comb(m, i) * math.comb(n, o - 2 * i)
               for i in range(o // 2 + 1))


# key generation
def _upper(mat):
    """Upper-triangular representative of the quadratic form y -> y^T mat y."""
    d = mat.nrows()
    out = zero_matrix(F, d, d)
    for j in range(d):
        out[j, j] = mat[j, j]
        for l in range(j + 1, d):
            out[j, l] = mat[j, l] + mat[l, j]
    return out


def keygen(n, o, m):
    """P^(1) and P^(2) random; P^(3) solved for so that P vanishes on O."""
    v = n - o
    o_prime = random_matrix(F, v, o)
    basis_o = block_matrix([[o_prime], [identity_matrix(F, o)]])       # n x o
    pks = []
    for _ in range(m):
        p1 = _upper(random_matrix(F, v, v))
        p2 = random_matrix(F, v, o)
        p3 = _upper(o_prime.transpose() * p1 * o_prime + o_prime.transpose() * p2)
        pks.append(block_matrix([[p1, p2], [zero_matrix(F, o, v), p3]]))
    return pks, basis_o


def public_map(pks, x):
    """P(x) = (x^T P_i x)_i."""
    return vector(F, [x * p * x for p in pks])


def polar_map(pks, x, y):
    """P'(x, y) = P(x+y) - P(x) - P(y), i.e. the alternating forms P_i + P_i^T."""
    return vector(F, [x * (p + p.transpose()) * y for p in pks])


def vanishes_on(pks, basis):
    """Two independent checks: the quadratic forms, and the polar forms."""
    quad = all(public_map(pks, x) == 0 for x in basis)
    pol = all(polar_map(pks, x, y) == 0 for x in basis for y in basis)
    return quad, pol


# recovery
def recover(sols, n, lam):
    """Contract the later pillars away, then contract Lambda^{lam_0} down into O."""
    first = wedge_basis(n, lam[0])
    dims = [dim_ext(n, a) for a in lam]
    pos = {s: i for i, s in enumerate(first)}

    def flat(idxs):
        f = 0
        for d, i in zip(dims, idxs):
            f = f * d + i
        return f

    vecs = []
    for r in range(sols.nrows()):
        w = sols.row(r)
        for rest in itertools.product(*[range(d) for d in dims[1:]]):
            alpha = [w[flat((i,) + rest)] for i in range(dims[0])]
            if not any(alpha):
                continue
            for j in itertools.combinations(range(n), lam[0] - 1):
                sj = set(j)
                u = [F(0)] * n
                for i in range(n):
                    if i not in sj:
                        u[i] = alpha[pos[tuple(sorted(sj | {i}))]]
                if any(u):
                    vecs.append(u)
    return matrix(F, vecs).row_space() if vecs else None


#  driver
def show_reference(configs):
    print(f"{'set':10}{'n':>5}{'v':>5}{'o':>4}{'m':>5} | {'m/v':>6} {'o/m':>6} {'v/o':>6}")
    for name, n, m, o in MAYO_REFERENCE:
        print(f"{name:10}{n:5d}{n-o:5d}{o:4d}{m:5d} | "
              f"{m/(n-o):6.3f} {o/m:6.3f} {(n-o)/o:6.2f}")
    for key in configs:
        _, n, o, m, _, _ = CONFIGS[key]
        print(f"{'toy ' + key:10}{n:5d}{n-o:5d}{o:4d}{m:5d} | "
              f"{m/(n-o):6.3f} {o/m:6.3f} {(n-o)/o:6.2f}   <- q = {q}")


def run(key):
    label, n, o, m, shapes, _ = CONFIGS[key]
    print(f"\n{'='*78}\nconfiguration {key}: {label}")
    print(f"  q = {q}, n = {n}, o = {o}, m = {m}  (v = {n-o})")

    pks, basis_o = keygen(n, o, m)
    quad, pol = vanishes_on(pks, basis_o.columns())
    assert quad and pol
    print("  keygen: P vanishes on the planted O, and O is totally isotropic")
    truth = basis_o.transpose().row_space()

    for lam in shapes:
        target = round(dim_schur(o, lam))
        print(f"\n  lambda = {lam}   ({len(lam)} pillar{'s' if len(lam) > 1 else ''}), "
              f"dim L_lambda(F_q^o) = {target}, "
              f"predicted cokernel = {max(schur_prediction(n, lam, m), target)}")
        cokernels, kernel = {}, None
        for dp in (False, True):
            t = time.time()
            mat = macaulay(pks, n, lam, divided_powers=dp)
            if dp:                      # one elimination gives both rank and kernel
                kernel = mat.left_kernel().basis_matrix()
                cokernels[dp] = kernel.nrows()
            else:
                cokernels[dp] = mat.nrows() - mat.rank()
            tag = "with divided powers   " if dp else "plain wedge only      "
            print(f"    {tag} {mat.nrows():6d} x {mat.ncols():6d}  "
                  f"rank {mat.nrows() - cokernels[dp]:6d}  "
                  f"cokernel {cokernels[dp]:6d}   ({time.time()-t:.0f}s)")
        gained = cokernels[False] - cokernels[True]
        if cokernels[False] == cokernels[True] == target:
            print("    both runs are already at the rank cap (cokernel = dim L_lambda(O)), "
                  "so the tower cannot show here -- see the probe below")
        else:
            print(f"    divided powers contribute {gained} extra dimension"
                  f"{'' if gained == 1 else 's'} of rank")
        if cokernels[True] == target and cokernels[False] != target:
            print("    ==> the divided powers are what make this attack work: the plain "
                  "wedge attack does not saturate here")

        coker = cokernels[True]
        if coker != target:
            print(f"    NOT saturated: {coker - target} spurious dimensions, "
                  f"no oil space recovered")
            continue
        print("    saturated: the kernel is exactly L_lambda(O)")
        rec = recover(kernel, n, lam)
        quad, pol = vanishes_on(pks, rec.basis())
        print(f"    recovered dimension {rec.dimension()} (want {o}); "
              f"equals the planted O: {rec == truth}")
        print(f"    independent check -- P vanishes: {quad}, polar forms vanish: {pol}")


def hilbert_probe(key):
    """Below saturation, check the measured cokernel against both series."""
    _, n, o, _, _, probe_m = CONFIGS[key]
    lam = (o,)
    print(f"\n  Hilbert-series probe at m = {probe_m} (deliberately below saturation, "
          f"lambda = {lam})")
    pks, _ = keygen(n, o, probe_m)
    for dp, name, predicted in (
            (False, "plain wedge only     ", series_plain(n, o, probe_m)),
            (True,  "with divided powers  ", series_divided(n, o, probe_m))):
        t = time.time()
        mat = macaulay(pks, n, lam, divided_powers=dp)
        coker = mat.nrows() - mat.rank()
        flag = "matches" if coker == predicted else f"DIFFERS from {predicted}"
        print(f"    {name} {mat.nrows():6d} x {mat.ncols():6d}  cokernel {coker:6d}  "
              f"series predicts {predicted:6d}  -> {flag}   ({time.time()-t:.0f}s)")
    gap = series_plain(n, o, probe_m) - series_divided(n, o, probe_m)
    dpcols = probe_m * sum(math.comb(n, o - 2 ** l) for l in range(2, o.bit_length()))
    print(f"    gap {gap} = number of divided-power columns ({dpcols}): "
          f"the tower contributes full rank")


if __name__ == "__main__":
    chosen = [a.upper() for a in sys.argv[1:]] or list(CONFIGS)
    bad = [c for c in chosen if c not in CONFIGS]
    if bad:
        sys.exit(f"unknown configuration(s) {bad}; available: {list(CONFIGS)}")
    show_reference(chosen)
    for key in chosen:
        run(key)
        hilbert_probe(key)
