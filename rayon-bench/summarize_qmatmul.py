import csv, collections, sys
# rows: variant,threads,rep,qmatmul,dtype,label,m,k,n,ms,gflops
d = {}
for r in csv.reader(open(sys.argv[1])):
    d[(r[0], int(r[1]), r[4], int(r[6]), r[5])] = float(r[10])
vs = sorted({k[0] for k in d}, key=["v0102-base", "fork-base", "fork-native"].index)
ts = sorted({k[1] for k in d})
labels = ["q_proj", "kv_proj", "gate_up", "down"]
for dtype in ["Q4K", "Q6K", "Q8_0"]:
    print(f"\n{dtype}: GFLOP/s summed over the 4 layer shapes (time-weighted)")
    print("| variant | m | " + " | ".join(f"T={t}" for t in ts) + " | T4/T1 |\n|---|---|" + "---|" * (len(ts) + 1))
    for v in vs:
        for m in [1, 16, 128, 512]:
            def agg(t):
                # total flops / total time over shapes
                fl = tm = 0
                for l, (k, n) in zip(labels, [(1536, 1536), (1536, 256), (1536, 8960), (8960, 1536)]):
                    g = d[(v, t, dtype, m, l)]; f = 2 * m * k * n
                    fl += f; tm += f / (g * 1e9)
                return fl / tm / 1e9
            row = [agg(t) for t in ts]
            print(f"| {v} | {m} | " + " | ".join(f"{x:.1f}" for x in row) + f" | {row[ts.index(4)] / row[0]:.2f}x |")
