from itertools import accumulate


def _broadcast_n(n, a):
    n = (n,) * len(a) if isinstance(n, int) else n
    if len(n) != len(a):
        raise ValueError(f"len(n)={len(n)} != len(a)={len(a)}")
    return n

# Let N = len(B)
# Returns sequences x_1, ..., x_N so that 
# 0 <= x_i <= D_i and sum_i x_i = j
def bounded_sequences(B, j, cap=None):
    if cap is None:
        cap = list(accumulate(reversed(B)))[::-1]
    if len(B) == 0:
        if j == 0: 
            yield tuple()
    elif cap[0] >= j:
        for i in range(min(B[0], j) + 1):
            for x in bounded_sequences(B[1:], j-i, cap=cap[1:]):
                yield (i,) + x

# Sorts n and d simultanuously according to (d_i, n_i) lexicographically
# Additionally returns the sign of the permutation or 0 on repeated elements
def signed_sort(n, d):
    n, d = list(n), list(d)
    s = 1
    for i in range(1, len(d)):
        x, y = d[i], n[i]; j = i
        while j and (d[j-1], n[j-1]) > (x, y):
            d[j], n[j] = d[j-1], n[j-1]; j -= 1
        if (i - j) & 1: s = -s
        if j and (d[j-1], n[j-1]) == (x, y): s = 0
        d[j], n[j] = x, y
    return tuple(n), tuple(d), s

# Returns a list of tuples of length at most k consisting of positive integers at most N
# Returns shorter sequences first, then lexicographic
def decreasing_sequences(N, k):
    out, layer = [], [()]
    for _ in range(k):
        layer = [a + (i,) for a in layer for i in range(1, (a[-1] if a else N) + 1)]
        out += layer
    return out
