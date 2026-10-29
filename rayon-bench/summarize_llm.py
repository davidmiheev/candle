import csv, collections, sys
d = collections.defaultdict(list)
for r in csv.reader(open(sys.argv[1])):
    v, t = r[0], int(r[1])
    kv = dict(x.split('=') for x in r[4:8])
    d[(v, t)].append((float(kv['prefill_tok_s']), float(kv['decode_tok_s']), float(kv['load_s'])))
vs = [v for v in ["v0102-base", "v0102-native", "fork-base", "fork-native", "fork-ctx", "anth-ctx"] if (v, 1) in d]
ts = sorted({t for _, t in d})
for i, name in [(0, "prefill tok/s (best of reps)"), (1, "decode tok/s (best of reps)")]:
    print(f"\n{name}\n| variant | " + " | ".join(f"T={t}" for t in ts) + " |\n|---|" + "---|" * len(ts))
    for v in vs:
        print(f"| {v} | " + " | ".join(f"{max(x[i] for x in d[(v, t)]):.1f}" for t in ts) + " |")
spread = max(100 * (max(a) - min(a)) / max(a) for k in d for a in list(zip(*d[k]))[:2])
print(f"\nmax rep-to-rep spread: {spread:.0f}%")
