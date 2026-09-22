import math
from mathematics.combinatorics import dim_exts, dim_ext
from parameters import snova_parameters
from mathematics.algorithms import decreasing_sequences
from predictors import snova_series

# ============================== User input ===================================

# Choose a parameter set
parameters = snova_parameters()

# Choose whether to include divided powers
divided_powers = True

# =============================================================================

# The average density of a single row in the Macaulay matrix
def average_density(v, o, m, divided_powers):
    if divided_powers:
        densities_wedges = [math.comb(v + 2**i, 2**i) for i in range(1, o[0].bit_length())]
        rows_wedges = [sum(m * dim_ext(v + o[j1], o[j1] - 2**i) * math.prod(dim_ext(v, o[j]) for j in range(len(a)) if j != j1) for j1 in range(len(o))) for i in range(1, o[0].bit_length())]
        densities_bilinears = [math.comb(v + 2**i, 2**i)**2 for i in range(o[0].bit_length())]
        rows_bilinears = [sum(2 * m * dim_ext(v + o[j1], o[j1] - 2**i) * dim_ext(v + o[j2], o[j2] - 2**i) * math.prod(dim_ext(v, o[j]) for j in range(len(a)) if j not in [j1, j2]) for j1 in range(len(o)) for j2 in range(j1+1, len(o))) for i in range(o[0].bit_length())]
    else:
        densities_wedges = [math.comb(v + 2, 2)]
        rows_wedges = [sum(m * dim_ext(v + o[j1], o[j1] - 2) * math.prod(dim_ext(v, o[j]) for j in range(len(a)) if j != j1) for j1 in range(len(o)))]
        densities_bilinears = [math.comb(v + 1, 1)**2]
        rows_bilinears = [sum(2 * m * dim_ext(v + o[j1], o[j1] - 1) * dim_ext(v + o[j2], o[j2] - 1) * math.prod(dim_ext(v, o[j]) for j in range(len(a)) if j not in [j1, j2]) for j1 in range(len(o)) for j2 in range(j1+1, len(o)))]

    total_density = sum(d * r for d, r in zip(densities_wedges, rows_wedges)) + sum(d * r for d, r in zip(densities_bilinears, rows_bilinears))
    total_rows = sum(rows_wedges) + sum(rows_bilinears)

    return total_density / total_rows

# =============================================================================

for v, o, m, q, l in parameters:

    field_op_cost = 2 * math.log2(q)**2 + math.log2(q)

    best = math.inf
    best_hypers = None

    for a in decreasing_sequences(o, l):
        
        if (sum(a)**2 - sum(math.comb(a_i + 1, 2) for a_i in a)) * m < v * sum(a): continue # Spurious oil spaces
        if not divided_powers and len(a) == 1 and 2*m < v: continue

        n = tuple(v + a_i for a_i in a)
        columns = dim_exts(n, a)
        density = average_density(v, a, m, divided_powers)

        cost = math.log2(3 * density) + 2*math.log2(columns) + math.log2(field_op_cost)
        memory = math.log2(columns)

        if cost >= best: continue

        if snova_series(n, a, m, divided_powers) <= 1:
            if cost <= best:
                best = cost
                best_hypers = (a, cost, memory, math.log2(density))


    if best_hypers:
        (a, cost, memory, density) = best_hypers
        print(f"v:{v:3d} o:{o:2d} m:{m:3d} l:{l:1d} q:{q:7d} | time:{int(cost):3d} memory:{int(memory):3d} density:{int(density):3d} | {a}")
    else:
        print(f"v:{v:3d} o:{o:2d} m:{m:3d} l:{l:1d} q:{q:7d} | time:N/A memory:N/A density:N/A | N/A")

 
