import math

from sage.all import GF

from mathematics.systems import construct_schur_matrix
from mathematics.isotropic_system import random_system, rewrite
from predictors import schur_prediction

def prediction(v, a, m):
    return max(0, schur_prediction(tuple(v + a_i for a_i in a), a, m))

def experiment(v, a, m, q):
    F = GF(q)

    fs = rewrite(random_system(F, n, m), n)

    M = construct_schur_matrix(fs, v, a, F)

    M = M.transpose()
    M.echelonize()

    return M.ncols() - M.rank()

# ================================================================ 

# Only even characteristic is implemented
FIELD_SIZE = 256

for n in range(1, 10):
    for a in range(2, n):
        min_m = int(2 * (n-a) / (a-1)) + 1 # Spurious solutions
        max_m = math.comb(n, 2) # If there is a true isotropic subspace it is lower

        if min_m > max_m: continue 
        v = n - a

        ai_list = [(a,)] if a != 2 else [(2,1)]
        while ai_list:
            ai, ai_list = ai_list[0], ai_list[1:]

            instant = False
            for m in range(min_m, max_m + 1):
                pred = prediction(v, ai, m)
                exp  = experiment(v, ai, m, FIELD_SIZE)

                result = f"n:{n:2d} m:{m:2d} | pred:{pred:6d} exp:{exp:6d} | {ai}" + (" | !!! " if pred != exp else " | correct")
                print(result)

                if exp == 0:
                    instant = (m == min_m)
                    break

            if not instant:
                ai_list += [ai + (1,)]
                if len(ai) >= 2 and ai[-1] < ai[-2]:
                    ai_list += [ai[:-1] + (ai[-1] + 1,)]
                
                



