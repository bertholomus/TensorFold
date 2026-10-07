"""numpy's float64 exp / log / pairwise sum against libm and plain loops (the served sampler's host arithmetic)."""
import ctypes, ctypes.util, json, sys
import numpy as np
libm = ctypes.CDLL(ctypes.util.find_library("m"))
libm.exp.restype = ctypes.c_double; libm.exp.argtypes = [ctypes.c_double]
libm.log.restype = ctypes.c_double; libm.log.argtypes = [ctypes.c_double]
rng = np.random.default_rng(7)
out = {"numpy": np.__version__, "machine": __import__("platform").machine()}
# exp over the sampler's range: scaled - max in [-60, 0]; log over (0, 1) uniforms and -log(u) in (0, 40)
xs = np.concatenate([-rng.random(200000) * 60.0, -rng.random(50000) * 1e-3, np.array([0.0, -1e-300, -745.0])])
e_np = np.exp(xs)
e_lm = np.array([libm.exp(float(x)) for x in xs])
out["exp_differ"] = int((e_np.view(np.uint64) != e_lm.view(np.uint64)).sum())
us = (rng.integers(0, 2**53, 200000, dtype=np.uint64) >> np.uint64(0)).astype(np.float64) * 2.0 ** -53 + 2.0 ** -54
l_np = np.log(us)
l_lm = np.array([libm.log(float(u)) for u in us])
out["log_differ"] = int((l_np.view(np.uint64) != l_lm.view(np.uint64)).sum())
ys = -l_np
l2_np = np.log(ys)
l2_lm = np.array([libm.log(float(y)) for y in ys])
out["log2_differ"] = int((l2_np.view(np.uint64) != l2_lm.view(np.uint64)).sum())
# sum over the last axis of [rows, 20] (probs.sum(axis=-1)) against numpy's pairwise sum written out
def pairwise(a):
    n = len(a)
    if n < 8:
        r = 0.0
        for v in a:
            r += v
        return r
    r = list(a[:8])
    i = 8
    while i < n - (n % 8):
        for j in range(8):
            r[j] += a[i + j]
        i += 8
    res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
    while i < n:
        res += a[i]
        i += 1
    return res
m = rng.random((20000, 20)) ** 8
s_np = m.sum(axis=-1)
s_pw = np.array([pairwise(list(row)) for row in m])
s_pw0 = np.array([0.0 + pairwise(list(row)) for row in m])
s_seq = np.array([sum(list(row)) for row in m])
out["sum_vs_pairwise"] = int((s_np.view(np.uint64) != s_pw.view(np.uint64)).sum())
out["sum_vs_sequential"] = int((s_np.view(np.uint64) != s_seq.view(np.uint64)).sum())
c_np = np.cumsum(m, axis=-1)
c_seq = np.array([np.array([sum(list(row[:j + 1])) for j in range(20)]) for row in m[:2000]])
out["cumsum_vs_sequential"] = int((c_np[:2000].view(np.uint64) != c_seq.view(np.uint64)).sum())
print(json.dumps(out))
