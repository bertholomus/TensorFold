"""First difference of two binary files as element arrays: python bincmp.py A B ELEM_BYTES [ROW_ELEMS]"""
import sys
a, b = open(sys.argv[1], "rb").read(), open(sys.argv[2], "rb").read()
es = int(sys.argv[3])
row = int(sys.argv[4]) if len(sys.argv) > 4 else 0
if len(a) != len(b):
    print(f"sizes differ: {len(a)} vs {len(b)}")
n = min(len(a), len(b)) // es
diff = [i for i in range(n) if a[i * es:(i + 1) * es] != b[i * es:(i + 1) * es]]
if not diff:
    print(f"equal: {n} elements")
else:
    i = diff[0]
    where = f" (row {i // row}, col {i % row})" if row else ""
    print(f"{len(diff)} of {n} differ; first {i}{where}: {a[i * es:(i + 1) * es].hex()} vs {b[i * es:(i + 1) * es].hex()}")
