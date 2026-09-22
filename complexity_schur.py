import math
from mathematics.combinatorics import dim_exts, dim_ext
from parameters import uov_parameters, mayo_parameters, qr_uov_parameters
from predictors import schur_prediction, wedge_prediction
from mathematics.algorithms import decreasing_sequences

# ============================== User input ===================================

# Flag for using intersection attack instead
intersection = False

# Choose a parameter set
parameters = [(64,4, 61,1024)]#mayo_parameters()

# Maximum amount of gradings for the exterior algebra
MAX_PARTITION = 12

# Heuristically assumes that lambda = [o, ..., o, o']
FAST_SEARCH = False

# =============================================================================

def fast_search_list(o, k):
    return [(o,) * i + (b,) for i in range(k) for b in range(1, o+1)]

# The average density of a single row in the Macaulay matrix
def average_density(n, a, m):
    wedge_densities = [math.comb(n[0] - (a[0] - 2**i), 2**i) for i in range(1, a[0].bit_length())]
    wedge_rows = [m * dim_exts(n, (a[0] - 2**i,) + a[1:]) for i in range(1, a[0].bit_length())]

    theta_densities = [math.comb(a[i] + 2**j, 2**j) for i in range(len(a) - 1) for j in range(a[i+1].bit_length())]
    theta_rows = [dim_exts(n, tuple(a[ii] + (2**j if i == ii else (-2**j if i+1 == ii else 0)) for ii in range(len(a)))) for i in range(len(a) - 1) for j in range(a[i+1].bit_length())]

    total_density = sum(d * r for d, r in zip(wedge_densities, wedge_rows)) + sum(d * r for d, r in zip(theta_densities, theta_rows))
    total_rows = sum(wedge_rows) + sum(theta_rows)

    return total_density / total_rows, total_rows


# The average density of a single row in the Macaulay matrix
def average_density_odd(n, a, m, p):
    bilinear_densities = [math.comb(v + p**i, p**i)**2 for i in range(a[0].bit_length())]
    bilinear_rows = [sum(2 * m * dim_ext(v + a[j1], a[j1] - p**i) * dim_ext(v + a[j2], a[j2] - p**i) * math.prod(dim_ext(v, a[j]) for j in range(len(a)) if j not in [j1, j2]) for j1 in range(len(a)) for j2 in range(j1+1, len(a))) for i in range(a[0].bit_length())]
    
    theta_densities = [math.comb(a[i] + p**j, p**j) for i in range(len(a) - 1) for j in range(a[i+1].bit_length())]
    theta_rows = [dim_exts(n, tuple(a[ii] + (p**j if i == ii else (-p**j if i+1 == ii else 0)) for ii in range(len(a)))) for i in range(len(a) - 1) for j in range(a[i+1].bit_length())]

    total_density = sum(d * r for d, r in zip(bilinear_densities, bilinear_rows)) + sum(d * r for d, r in zip(theta_densities, theta_rows))
    total_rows = sum(bilinear_rows) + sum(theta_rows)

    return total_density / total_rows, total_rows

def base_prime(n):
    return next(p for p in range(2, n + 1) if n % p == 0)

for v, o, m, q in parameters:

    field_op_cost = 2 * math.log2(q)**2 + math.log2(q)
    odd = (q % 2) == 1

    if intersection:
        v, o, m = (2*v - o, 2*o - v, 3*m - 2)
        if o < 2: continue # MQ territory

    if not odd:
        old_o = next((oo for oo in range(2, o) if wedge_prediction(v+oo, oo, m, divided_powers=False) <= 1), None)
        if old_o is not None:
            old_cost = math.log2(3 * math.comb(v+2, 2) * dim_ext(v+old_o, old_o)**2 * field_op_cost)
    else:
        old_o = next((oo for oo in range(2, o) if schur_prediction((v+oo, v+oo), (oo, oo), m, ODD=True) <= 1), None)
        if old_o is not None:
            old_cost = math.log2(3 * (v+1)**2 * dim_exts((v+old_o, v+old_o), (old_o, old_o))**2 * field_op_cost)

    best = math.inf
    best_hypers = None

    a_list = fast_search_list(o, MAX_PARTITION) if FAST_SEARCH else decreasing_sequences(o, MAX_PARTITION)

    for a in a_list:

        if odd and len(a) < 2: continue
        if (a[0] - 1) * m < 2 * v: continue # Spurious oil spaces

        n = tuple(v + a_i for a_i in a)
        columns = dim_exts(n, a)
        cost = math.log2(3) + math.log2(columns) + math.log2(columns) + math.log2(field_op_cost)
        if cost > best: continue

        density, rows = average_density(n, a, m) if not odd else average_density_odd(n, a, m, base_prime(q))
        cost += math.log2(density)
        if cost > best: continue

        memory = math.log2(columns)

        if schur_prediction(n, a, m, ODD=odd) <= 1:
            best = cost
            best_hypers = (a, cost, memory, math.log2(density))
            if FAST_SEARCH: break


    result = f"v:{v:3d} o:{o:2d} m:{m:3d} q:{q:15d}"
    result += f" | N/A N/A" if old_o is None else f" | {int(old_o):3d} {int(old_cost):3d}"

    if best_hypers:
        (a, cost, memory, density) = best_hypers
        result += f" | time:{int(cost):3d} memory:{int(memory):3d} density:{int(density):3d}/{int(math.log2(v*v))} | {a}"
    else:
        result += f" | time:N/A memory:N/A density:N/A | N/A"

    print(result)

 


