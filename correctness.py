from sage.all import GF

from mathematics.isotropic_system import full_isotropic_system
from mathematics.combinatorics import dim_exts, dim_ext, dim_schur
from mathematics.systems import construct_wedge_matrix, construct_schur_matrix
from mathematics.kernels import construct_wedge_kernel, construct_schur_kernel

def print_header(name, correctness_level):
    header = f"{name}:\nn\to\ta\tb\tComposition"
    if correctness_level >= 2: header += "\tDimension"
    if correctness_level >= 3: header += "\tGenerates"
    print(header)

"""
For correctness we test 3 things
1. We construct the matrix based on the public key fs and the cokernel based on the oil space O, these should compose to 0
2. The dimension of the constructed cokernel is of the expected dimension, i.e. the dimension of the hook in O
3. The constructed cokernel generated the actual cokernel. I.e. no other solutions exist. We likely need a>=2 or b>=2 for this

Since the public key will always be a subspace of I2(O), we just test on the entire I2(O), this also limits spurious solutions.
"""
correctness_level = 3
max_n = 7
F = GF(16)

print_header("Wedges", correctness_level)
for n in range(1, max_n+1):
    for o in range(n + 1):
        for a in range(2, o+1):
            fs, O = full_isotropic_system(F, n, o)
            M = construct_wedge_matrix(fs, n, a, F)   
            ker = construct_wedge_kernel(O, a)
            test = f"{n}\t{o}\t{a}\t"
            if correctness_level >= 1: test += f"{ker * M == 0}\t\t"
            if correctness_level >= 2: test += f"{ker.rank() == dim_ext(o, a)}\t\t"
            if correctness_level >= 3: test += f"{dim_ext(o, a) == dim_ext(n, a) - M.rank()}\t{dim_ext(o, a)}\t{dim_ext(n, a) - M.rank()}"
            print(test)


print_header("Schur", correctness_level)
for n in range(1, max_n+1):
    for o in range(n):
        for a in range(2, o+1):
            for b in range(1,a+1):
                fs, O = full_isotropic_system(F, n, o)
                v = n - o
                M = construct_schur_matrix(fs, v, (a, b), F)   
                ker = construct_schur_kernel(O, (a, b))
                test = f"{n}\t{o}\t{a}\t{b}\t"
                if correctness_level >= 1: test += f"{ker * M == 0}\t\t"
                if correctness_level >= 2: test += f"{ker.rank() == 1}\t\t"
                if correctness_level >= 3: test += f"{1 == dim_exts((v+a, v+b), (a, b)) - M.rank()}"
                print(test)
