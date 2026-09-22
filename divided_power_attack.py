import math
from sage.all import GF
from mathematics.isotropic_system import random_isotropic_system
from mathematics.divided_powers import (divided_square, divided_power_tower,
                                        tower_generators, multiplication_matrix,
                                        plucker_rows, recover_subspace)

# ======================================================================================
# Full divided-power exterior attack: recover the oil space O from the public forms.
#
# Solution space at degree d = annihilator of I_d = LEFT null space of the multiplication
# matrix.  It always contains Lambda^d O (since q|_O = 0 => q^{(2^j)}|_O = 0), and the
# attack SUCCEEDS exactly when it equals Lambda^d O, i.e. when its dimension is C(o, d).
# Any excess is the spurious tail of the Hilbert series.
# ======================================================================================

def predicted(n, m, d):
    """Froberg-truncated coefficient of (1+t)^n (1-t^2)^m = (1+t)^{n+m} (1-t)^m.

    This is the GENERIC solution dimension (no planted oil space).  The attack is exact
    exactly once this drops to 0, because then the only solutions left are Lambda^d O.
    Verified against experiment: n=12 d=4 gives 114, 54, 0, 0 for m=6,7,8,9 and the
    measured solution dimensions were 114, 54, 1, 1."""
    def raw(e):
        return sum((-1)**k * math.comb(m, k) * math.comb(n + m, e - k)
                   for k in range(min(m, e) + 1))
    for e in range(d + 1):
        if raw(e) <= 0: return 0
    return raw(d)


def attack(fs, n, d, F):
    """Returns (solution space basis as Plucker rows, multiplication matrix)."""
    gens = tower_generators(fs, d)
    M = multiplication_matrix(n, d, gens, F)
    return M.left_kernel().basis_matrix(), M


def run(n, m, o, d, F, level=3):
    fs, A = random_isotropic_system(F, n, m, o)
    sols, M = attack(fs, n, d, F)
    expected = plucker_rows(A, d)

    # 1. the planted oil space really is a solution
    composition = (expected * M == 0)
    # 2. the solution space is no bigger than Lambda^d O  (no spurious solutions)
    exact = (sols.nrows() == math.comb(o, d))
    out = (f"{n}\t{m}\t{o}\t{d}\t{composition}\t\t{predicted(n, m, d):6d}\t"
           f"{sols.nrows()}/{math.comb(o, d)}")

    # 3. recover O and compare with the planted subspace.  Only meaningful when the
    #    solution space IS Lambda^d O -- otherwise say so rather than printing a bare "-".
    if level >= 3 and exact:
        rec = recover_subspace(sols, n, d)
        ok = (rec.row_space() == A.transpose().row_space())
        out += f"\t\t{ok}"
    elif not exact:
        out += "\t\tskipped: m too small, not saturated"
    else:
        out += "\t\t(level<3)"
    return out, composition, exact


if __name__ == "__main__":
    F = GF(16)

    # --- sanity: the divided square really does build the tower ---
    print("=== divided-power tower sanity (char 2) ===")
    from mathematics.divided_powers import alternating_form
    fs0, _ = random_isotropic_system(F, 9, 3, 3)
    qq = alternating_form(fs0[0])
    tower = divided_power_tower(qq, 16)
    print("  generator degrees:", [deg for deg, _ in tower],
          " support sizes:", [len(g) for _, g in tower])
    print("  q^(4) via iterated square == degree 8 elt:",
          all(len(mon) == 8 for mon in divided_square(divided_square(qq))))

    print()
    print("=== oil-space recovery ===")
    # m must be at or above the saturation threshold, else the solution space is bigger
    # than Lambda^d O and there is nothing to recover.  Measured thresholds (d = o):
    #   n=11,o=4: m>=7    n=12,o=4: m>=8    n=13,o=4: m>=10
    #   n=13,o=5: m>=6    n=14,o=5: m>=7
    # Also note d>=4 is needed for the divided powers to contribute at all (deg q^(2) = 4).
    print("n\tm\to\td\tComposition\tpred\tSolDim/C(o,d)\tRecovered")
    for n, m, o, d in [(12, 6, 4, 4),      # deliberately BELOW threshold, for contrast
                       (11, 7, 4, 4),
                       (12, 8, 4, 4),      # divided powers are what make this one work
                       (12, 9, 4, 4),
                       (13, 10, 4, 4),
                       (13, 6, 5, 5),
                       (14, 7, 5, 5)]:
        line, comp, exact = run(n, m, o, d, F)
        print(line)
