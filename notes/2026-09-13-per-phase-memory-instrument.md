# Per-phase memory instrumentation: the RSS channel was measuring nothing

**Date:** 2026-09-13
**Trigger:** Reviewing the benchmark walkthrough surfaced the README's claim
that per-stage `tracemalloc` overhead is "on the same order as the stage
runtime". Checking that claim turned up a much larger problem in the other
channel.
**Supersedes the diagnosis in:** [2026-05-28-cbg-memory-footprint.md](2026-05-28-cbg-memory-footprint.md)

---

## TL;DR

- The per-stage **RSS channel never measured anything**. On a real
  `octant_cbg_hull` fold it took *two* distinct values across 80 LTD
  measurements and was a flat 4096 B for MTL apart from two warmup artifacts.
- The cause is **glibc heap reuse**, not sampler resolution. The 1 ms re-run
  recommended by the 2026-05-28 note could not have fixed it.
- Replaced by a `mallinfo2`-sampled **libc heap-in-use** channel
  (`*_heap_peak_bytes`), which is reuse-immune and sees the GEOS allocations
  `tracemalloc` is structurally blind to.
- The README's tracemalloc-overhead caveat was **wrong**: measured 819 ns per
  cycle and ~1.00× slowdown on NumPy-heavy stages. The real per-stage cost is
  the sampler *thread* (~45 µs).
- Sampler interval dropped **5 ms → 1 ms** after the first real run showed a
  bimodal CTR reading; validated against an exact malloc interposer at
  98.4% (GEOS) / 100.0% (CTR) recovery of the true peak.

---

## The bug

glibc's mmap threshold is **dynamic**: once it observes frees of mmap'd blocks
it raises the threshold, after which allocations come from the reused heap and
never grow RSS. A per-stage RSS delta therefore answers "did this stage raise
the process high-water mark" — which after the first couple of targets is
structurally *no*.

Five identical 16 MB stage allocations:

| iteration | RSS delta | `uordblks + hblkhd` delta |
|---|---:|---:|
| 0 | 16.00 MB | 16.00 MB |
| 1 | 15.91 MB | 16.00 MB |
| 2 | 0.00 MB | 16.00 MB |
| 3 | 0.00 MB | 16.00 MB |
| 4 | 0.00 MB | 16.00 MB |

`uordblks` tracks *logical* bytes in use rather than resident pages, so it is
immune. Both fields are required: a large NumPy buffer served by mmap lands in
`hblkhd` only, and `uordblks` alone reported 0.00 MB for it.

### Why the 2026-05-28 diagnosis was wrong

That note saw `mtl_rss` = 4 KB against `mtl_alloc` = 131 KB and attributed it to
the 5 ms sampler missing transient Shapely allocations, recommending a 1 ms
re-run. Two independent refutations:

1. A 43–317 ms MTL stage gets 9–60 samples at 5 ms. A resolution problem cannot
   produce an **exact one-page constant**.
2. Measured directly: 50 µs, 500 µs and 5 ms sampling all return ~1000–1050 KB
   on the GEOS workload. For that workload the interval was never the variable.

**But the interval does matter for *transient* allocations — see "Sampler
interval" below.** My first pass concluded "5 ms is fine, a finer interval buys
nothing"; that was right for held allocations and wrong for transient ones, and
real data caught it.

---

## The two channels are complementary

Neither dominates. They must never be combined — summing double-counts
malloc-backed NumPy, `max` discards the pymalloc side.

| workload | tracemalloc | mallinfo2 sampled peak |
|---|---:|---:|
| pure-Python 1 MB list | **1028 KB** | 6.9 KB |
| NumPy 32 MB `np.ones` | 32.14 MB | 32.16 MB |
| Shapely/GEOS intersection (~30 ms) | 15.6 KB | **1015 KB** (~64×) |

`mallinfo2` is blind to pymalloc (CPython takes 1 MB obmalloc arenas via raw
mmap); `tracemalloc` is blind to GEOS.

Sampling is essential, not incidental: the *net* `mallinfo2` delta across a
Shapely stage is ~2.5 KB because the allocation is transient, while the
*sampled peak* is ~1015 KB.

---

## Before / after on `as01-260728-260802-mesh`, fold_0, `octant_cbg_hull`

80 targets, all SUCCESS. `p50` in bytes, `nunique` out of 80:

| column | before p50 | nunique | after p50 | nunique |
|---|---:|---:|---:|---:|
| `ltd_rss_peak_bytes` | 4,096 | **2** | NULL | — |
| `mtl_rss_peak_bytes` | 4,096 | **10** | NULL | — |
| `ctr_rss_peak_bytes` | 19,685,376 | 45 | NULL | — |
| `ltd_heap_peak_bytes` | — | — | 82,032 | 50 |
| `mtl_heap_peak_bytes` | — | — | **1,618,600** | **80** |
| `ctr_heap_peak_bytes` | — | — | 25,246,920 | 60 |
| `mtl_alloc_peak_bytes` | 159,145 | 80 | 138,267 | 80 |
| `ctr_alloc_peak_bytes` | 25,256,565 | 52 | 25,221,717 | 50 |

Three things worth noting:

1. **`mtl_heap` reads 11.7× `mtl_alloc`** (11.2–14.3× across the five folds) — GEOS dominates MTL, exactly as the
   blind-spot table predicts, and the old channel reported none of it.
2. **`ctr_heap / ctr_alloc` = 1.0010.** On the NumPy-bound CTR the two
   independent channels agree to 0.1%. This is the cross-check that the
   `mallinfo2` reader is correct, not merely self-consistent.
3. **The 22 MB warmup artifact is gone.** Under `reduce="max"` (which
   `v3/modules/cost.py` uses) that single target set the entire combo's MTL
   memory figure. `mtl_heap` max/p50 is 26.1×, matching `mtl_alloc`'s 20.6× —
   i.e. genuine target-to-target variation, not an artifact.

Any previously published per-stage memory figure that used `memory_rss` should
be regenerated.

---

## Sampler interval: 5 ms → 1 ms (found by the real run, not by theory)

The first end-to-end run at 5 ms produced a **visibly bimodal**
`ctr_heap_peak_bytes` on fold_1: 43 targets reading ~21 MB and 34 reading
~25 MB for identical work, with the per-target `heap/alloc` ratio taking
exactly two values (0.8347 and 1.0010). The gap was exactly **4.000 MiB**, and
`ctr_alloc_peak_bytes` was stable across both clusters — so it was the heap
channel intermittently *missing* a block, not a real difference in demand.

Not a duration problem: CTR stages run 145–1900 ms, i.e. 29–380 windows at
5 ms. The culprit is the Monte-Carlo CTR's rejection-sampling loop, which
briefly holds an extra 4 MiB batch. Catch rate for a 2 ms spike inside a 250 ms
stage, 20 trials:

| interval | caught |
|---|---:|
| 5.0 ms | **14/20** |
| 1.0 ms | 20/20 |
| 0.2 ms | 20/20 |

Default is now **1 ms**. Affordable precisely because `mallinfo2` is 32× cheaper
to read than psutil RSS: at 1 ms that is a ~0.05% duty cycle. glibc arena-lock
contention is the reason not to go lower still.

After the change, the bimodality is gone — 98–99% of targets fall within 10%
on every fold (fold_1 was 44%).

## Ground truth: exact malloc interposition

`heaptrack` **segfaults** on this workload (dies after ~78 K allocations,
before the combo runs), so the cross-check uses a purpose-built `LD_PRELOAD`
interposer instead. It counts every `malloc`/`calloc`/`realloc`/`free` via
`malloc_usable_size` — no bookkeeping map, hence no allocation recursion — and
exports `hp_reset_peak()` / `hp_get_peak()` so the *exact*, unsampled peak can
be read per stage from Python. That is precisely the quantity `HeapSampler`
estimates, so any gap is sampling error and nothing else.

| workload | exact peak | sampled peak | recovery |
|---|---:|---:|---:|
| MTL-like (GEOS, 160 buffers) | 1,059 KB | 1,042 KB | **98.4%** |
| CTR-like (21 MB held + 4 MiB transient) | 24,418 KB | 24,417 KB | **100.0%** |

Six trials each, at the 1 ms default. The residual ~1.6% on the GEOS path is
sub-millisecond allocation churn; the CTR spike that 5 ms missed is now caught
in every trial. Source: [assets/heappeak.c](assets/heappeak.c) — a validation tool, not wired
into the benchmark. Build and use:

```bash
gcc -shared -fPIC -O2 -o libheappeak.so notes/assets/heappeak.c -ldl
LD_PRELOAD=./libheappeak.so .venv/bin/python your_check.py
```

---

## Run-level numbers were already correct

`peak_rss_bytes()` uses `getrusage(RUSAGE_SELF).ru_maxrss` — a monotonic
*kernel* high-water mark (the README's "psutil RSS" label was wrong). Checked
against cgroup v2 under an isolated `systemd-run --user --scope -p
MemoryAccounting=yes` on a 320 MB allocation: `memory.peak` = 342 MB vs
`ru_maxrss` = 355 MB, within 4%. Cgroups add nothing for a single-process
`run_combo`, and no kernel facility can attribute memory to a phase at all —
LTD/MTL/CTR are millisecond events inside one process.

What *was* wrong is the baseline mark. `run_baseline_rss_bytes` is sampled
before the input parquets load, so `peak − baseline` contains input loading and
the fit. `run.json` now carries four ordered marks; on this fold:

```
baseline 172.8 -> after_inputs 225.3 -> after_fit 225.3 -> peak 305.8  (MB)
sweep-only delta = peak - after_fit = 80.4 MB
old "CBG-attributable" = peak - baseline = 133.0 MB   (65% overstated)
```

(The 2026-05-28 note's observation that the baseline was "163 MB identical
across all six ASNs" was corroborating evidence for this: an ASN-independent
number cannot include ASN-sized inputs.)

---

## Overhead, corrected

| operation | cost |
|---|---:|
| `tracemalloc` start/reset_peak/get/stop cycle | 819 ns |
| tracemalloc slowdown, NumPy-heavy stage | ~1.00× |
| `psutil` RSS read | 16,667 ns |
| `mallinfo2` read | **517 ns** (32× cheaper) |
| sampler `Thread.start()` + `join()` | **~45 µs** ← the real cost |

Against stage p50s of 17–530 ms the thread is ~0.03%. A shared
process-lifetime sampler thread would save it but needs a lock; deliberately
rejected at this ratio.

Consequence: the README caveat telling readers not to trust per-target
tracemalloc numbers was unfounded and has been removed.

---

## Also fixed

`instrument.py` claimed `tracemalloc.start()` is refcounted and therefore
nest-safe. It is **not** — `start()` does not stack, and a single `stop()` tears
tracing down, after which an outer session's `get_traced_memory()` returns
`(0, 0)`. Now guarded with `is_tracing()`.

That guard forced a second fix in the same commit: the old code read the
*absolute* peak, which was delta-equivalent only because a fresh `start()`
begins at `current ≈ 0`. Under an outer session `reset_peak()` sets
`peak := current`, which includes the caller's live set — a 1 MB stage under a
4 MB live set would have reported 5 MB. The recorded value is now
`peak − entry_current`.

The inactive channel is written as **NULL, never 0** — a 0 is
indistinguishable from "this stage used no memory", which is the class of bug
this whole change removes.

---

## Related

- Instrument: [../scripts/benchmark/v2/instrument.py](../scripts/benchmark/v2/instrument.py)
- Schema: [../scripts/benchmark/v2/schema.py](../scripts/benchmark/v2/schema.py)
- Cost model: [../scripts/analysis/v3/modules/cost.py](../scripts/analysis/v3/modules/cost.py)
- Superseded diagnosis: [2026-05-28-cbg-memory-footprint.md](2026-05-28-cbg-memory-footprint.md)
