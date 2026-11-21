"""Summarise tools/fp4_soak.py samples: per clip and stage (GPU temperature, power, SM clock, ACPI zone, MemAvailable,
throttle reasons) and a per-minute trend.   python tools/soak_summary.py ~/ghost-studio/out/soak1"""
import json, sys, glob, collections
rows=[]
for f in sorted(glob.glob(sys.argv[1]+"/samples-*.jsonl")):
    rows += [json.loads(l) for l in open(f) if l.strip()]
t0=rows[0]["t"]
by=collections.OrderedDict()
for r in rows:
    k=(r.get("clip"), r.get("stage") if r.get("stage") in ("denoising","decoding the video","gap","loading") else "other")
    by.setdefault(k,[]).append(r)
print("samples", len(rows), "span_s", round(rows[-1]["t"]-t0))
for (clip,stage),rs in by.items():
    g=[r.get("gpu_c") for r in rs if r.get("gpu_c") is not None]; p=[r.get("power_w") for r in rs if r.get("power_w") is not None]
    s=[r.get("sm_mhz") for r in rs if r.get("sm_mhz") is not None]; z=[r.get("zone_c") for r in rs if r.get("zone_c") is not None]
    reasons=collections.Counter(r.get("reasons") for r in rs)
    if not g: continue
    print(f"clip {clip} {stage:18s} n={len(rs):3d} gpu {g[0]:.0f}->{g[-1]:.0f} max {max(g):.0f}C | power avg {sum(p)/len(p):.0f} max {max(p):.0f}W | sm avg {sum(s)/len(s):.0f} min {min(s):.0f} MHz | zone max {max(z):.1f} | avail min {min(r['MemAvailable'] for r in rs)/1024:.1f}GB | reasons {dict(reasons.most_common(3))}")
# per-minute trend
print("per minute: t, gpu max, power avg, sm avg, zone max")
for m in range(0, int(rows[-1]["t"]-t0)+1, 60):
    rs=[r for r in rows if m <= r["t"]-t0 < m+60 and r.get("gpu_c") is not None]
    if rs: print(m//60, max(r["gpu_c"] for r in rs), round(sum(r["power_w"] for r in rs)/len(rs)), round(sum(r["sm_mhz"] for r in rs)/len(rs)), max(r["zone_c"] for r in rs))
