# CBG memory footprint — per-worker RSS and `-j` sizing for the octant sweep

**Date:** 2026-05-28
**Trigger:** First run of the 6-ASN octant_weighted_cbg sweep on the new
[dual-channel memory instrumentation](../scripts/benchmark/v2/instrument.py) —
wanted to size `-j` for the next run.
**Data:** `scripts/benchmark/v2/outputs/*_octant_sweep/summary.parquet`
(6 ASNs × 10 weight thresholds × 5 folds = 300 combos summarized).

---

## TL;DR

- Per-worker RSS peaks at **~304 MB** (AS7922, the largest VP corpus at 213
  probes) and bottoms at ~223 MB (AS16509/AS31898, 30–31 probes).
- Baseline (Python + libs + inputs loaded, before any combo work) is
  **163 MB across all ASNs** — identical, as expected for `getrusage` after
  imports settle.
- The **CBG-attributable delta** (peak − baseline) scales linearly with VP
  count: 60 MB at 30 VPs → 141 MB at 213 VPs.
- **CPU, not memory, is the bottleneck**. Memory permits 30+ parallel workers
  even worst-case; the host only has 16 cores. **`-j 12` to `-j 14`** is the
  sweet spot. Expected speedup over the just-completed `-j 4` run: ~3×.

---

## Per-ASN RSS (one row per snakemake `run_combo` worker, aggregated across 50 combos/ASN)

| ASN | probes | baseline (MB) | peak max (MB) | peak p95 (MB) | peak med (MB) | CBG-delta max (MB) | CBG-delta med (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| north_america_as7922 | 213 | 163 | **304** | 304 | 298 | 141 | 135 |
| europe_as3209        | 164 | 163 | 279 | 278 | 277 | 116 | 114 |
| europe_as3215        | 141 | 163 | 272 | 272 | 267 | 110 | 104 |
| north_america_as7018 | 125 | 163 | 270 | 269 | 262 | 107 |  99 |
| global_as31898       |  31 | 163 | 224 | 224 | 223 |  61 |  61 |
| global_as16509       |  30 | 163 | 223 | 223 | 223 |  61 |  60 |

The peak is captured via `resource.getrusage(RUSAGE_SELF).ru_maxrss` at the end
of `run_one_combo` ([runner.py:120](../scripts/benchmark/v2/runner.py#L120));
baseline is the same call before fit + per-target work
([runner.py:80](../scripts/benchmark/v2/runner.py#L80)). Monotonic kernel
high-water mark, so `peak ≥ baseline` is guaranteed.

---

## Honest-MTL channel — surprise: tracemalloc is fine at this scale

`mtl_rss_peak_bytes_p95` (median across this ASN's combos) vs
`mtl_alloc_peak_bytes_p95` (tracemalloc, Python-only), in **KB**:

| ASN | mtl_rss (KB) | mtl_alloc (KB) |
|---|---:|---:|
| europe_as3215        | 39 | 207 |
| north_america_as7018 |  8 | 154 |
| north_america_as7922 |  4 | 131 |
| europe_as3209        |  4 |  86 |
| global_as16509       |  8 |  68 |
| global_as31898       | 64 |  67 |

The new RSS sampler is reporting **smaller** numbers than tracemalloc for the
MTL stage on most ASNs. Initial hypothesis (before the run) was the opposite —
that tracemalloc would massively undercount because Shapely/GEOS allocates
in C. Both can be true at the same time:

- Shapely's C allocations for a 125–213-polygon intersection are tiny in
  absolute terms (well under 1 MB) and **transient** — finish well inside a
  single 5 ms sample window of the background RSS sampler
  ([instrument.py:55](../scripts/benchmark/v2/instrument.py#L55)). The
  sampler simply misses them.
- Tracemalloc still picks up the Python-side wrappers (`Polygon` objects,
  coordinate arrays, `shapely.geometry.intersect_all` glue) that linger long
  enough to register.

So at *this* workload size, MTL is not memory-bound — the actual heavy
consumer is the CTR Monte Carlo Sobol grid (`ctr_rss_peak_bytes` in the
~25 MB range per target, matching the pre-rename `ctr_peak_bytes`).

If we ever push to MTL configurations with thousands of polygons or hold
intersections in lists across many targets, the channel ordering might
invert. The right way to confirm is re-run with a tighter sampler interval
(1 ms instead of 5 ms) and see if `mtl_rss_peak` jumps. For the current
sweep grade, **either channel is acceptable for the MTL story**.

---

## `-j` sizing

Host: 16 cores, 16 GB total. With IDE + Claude Code + browser running,
~10 GB free for the sweep.

Memory ceiling:
- Worst-case worker = 304 MB (AS7922).
- `-j 14`: 14 × 304 MB = **4.3 GB** — comfortable.
- `-j 16`: 16 × 304 MB = 4.9 GB — fine memory-wise but oversubscribes CPU
  and hurts interactive responsiveness.
- Memory alone would allow `-j 32`, but you can't reach that on 16 cores.

CPU ceiling:
- 16 physical cores. `-j 12–14` reserves 2–4 cores for OS + IDE.
- Within a single ASN, parallelism is across (5 folds × 10 combos) = 50 jobs
  per ASN — well over `-j 14`, so workers stay busy until late in the DAG.
- The two cloud ASNs (AS16509/AS31898 at 30/31 VPs) finish each combo so
  fast that `-j 14` will likely under-utilize cores for those — but they
  also take very little total time, so the cost is small.

Wall-time projection (vs. the `-j 4` run that took ~2 h 50 min from launch
at 17:23 to last summary at 20:12 — see `scripts/sweeps/octant_sweep.log`):

| `-j` | projected wall time | CPU util |
|---|---|---|
| 4 (measured) | 170 min | low — wasted cores |
| 12 | ~58 min | 75% — reserves 4 cores |
| 14 | ~50 min | 87% — reserves 2 cores |
| 16 | ~50 min | 100% but contended — IDE feels slow |

Linear-scaling assumption holds reasonably well within each ASN's 50-job
DAG; the orchestrator
([run_octant_sweep.py](../scripts/sweeps/run_octant_sweep.py)) runs ASNs
serially so cross-ASN parallelism isn't on the table without re-architecting.

**Recommendation: `-j 14` if you can step away from the machine; `-j 12` if
you want to keep using the IDE comfortably.**

---

## Aggregation snippet (reproduce)

```python
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

frames = []
for sp in sorted(Path('scripts/benchmark/v2/outputs').glob('*_octant_sweep/summary.parquet')):
    df = pq.read_table(sp).to_pandas()
    df['asn_run'] = sp.parent.name
    frames.append(df)
all_df = pd.concat(frames, ignore_index=True)

all_df['delta_mb'] = (all_df['run_peak_rss_bytes'] - all_df['run_baseline_rss_bytes']) / 1024**2
all_df['peak_mb'] = all_df['run_peak_rss_bytes'] / 1024**2

print(all_df.groupby('asn_run').agg(
    peak_mb_max=('peak_mb','max'),
    peak_mb_med=('peak_mb','median'),
    delta_mb_med=('delta_mb','median'),
).round(0))
```

---

## Related

- Instrumentation rationale + design: commit `c7ee30a` ("bench: dual-channel
  memory stats — tracemalloc + RSS sampler, true peak via getrusage")
- Sweep orchestrator: [scripts/sweeps/run_octant_sweep.py](../scripts/sweeps/run_octant_sweep.py)
- Schema: [scripts/benchmark/v2/schema.py](../scripts/benchmark/v2/schema.py) —
  `SUMMARY_METRICS` lists both `*_alloc_peak_bytes` (tracemalloc) and
  `*_rss_peak_bytes` (sampler) per stage.
