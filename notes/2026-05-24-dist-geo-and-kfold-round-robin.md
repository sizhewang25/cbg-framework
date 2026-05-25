# DIST-GEO VP selection + DistGeo K-fold round-robin

Walk-through of the dist_geo strategy in [scripts/vp_selection/strategies.py](../scripts/vp_selection/strategies.py)
and how it composes with the balanced round-robin in
[scripts/processing/ripe_atlas/stratification.py](../scripts/processing/ripe_atlas/stratification.py).

---

## Part 1 — DIST-GEO sequence strategy

**What it's trying to do.**
Pick an ordered list of VPs so the first K of them are as **geodesically
spread out** as possible. Concretely: maximize the sum of pairwise
great-circle distances among the K chosen VPs (Σ pair-geodesic). It's a
greedy Prim-style approximation to "max-sum diversification."

**Three phases** ([strategies.py:54-109](../scripts/vp_selection/strategies.py#L54-L109)):

1. **Pick the starting node** — random endpoint of the heaviest edge
   (`_max_edge_start`, [strategies.py:126-132](../scripts/vp_selection/strategies.py#L126-L132)).
2. **Seed the weights table** — `weights[v] = dist(v, start)`
   ([strategies.py:89](../scripts/vp_selection/strategies.py#L89)).
3. **Greedy continue** — at each step pick `argmax(weights)`, then for every
   still-unselected neighbor `v`, do `weights[v] += dist(v, picked)`
   ([strategies.py:201-235](../scripts/vp_selection/strategies.py#L201-L235)).

**Key invariant** ([strategies.py:191-192](../scripts/vp_selection/strategies.py#L191-L192)):

> `weights[v]` accumulates Σ dist(v, s) over s in `selected`.

So `argmax(weights)` = "the unselected VP whose addition increases
Σ pair-geodesic the most." Picking it is the locally optimal step, which is
why the update on
[strategies.py:232-235](../scripts/vp_selection/strategies.py#L232-L235) is `+=`,
not a recomputation.

### Worked example — 5 globally-spread VPs

Pool = {NYC, LON, TOK, SYD, JNB}, rough km distances:

|       | NYC    | LON    | TOK    | SYD    | JNB    |
| ----- | ------ | ------ | ------ | ------ | ------ |
| NYC   | —      | 5,570  | 10,840 | 15,990 | 12,840 |
| LON   | 5,570  | —      | 9,560  | 16,990 | 9,070  |
| TOK   | 10,840 | 9,560  | —      | 7,820  | 13,540 |
| SYD   | 15,990 | 16,990 | 7,820  | —      | 11,040 |
| JNB   | 12,840 | 9,070  | 13,540 | 11,040 | —      |

**Phase 1.** Heaviest edge = LON–SYD (16,990). `rng.choice` picks one
endpoint; say it returns **LON**.

```
selected = {LON}
order    = [LON]
```

**Phase 2.** Seed `weights[v] = dist(LON, v)`:

```
weights = { NYC: 5570, TOK: 9560, SYD: 16990, JNB: 9070 }
```

**Phase 3 — greedy loop.**

Step 2. `argmax` = **SYD** (16,990). Update remaining by `+ dist(SYD, v)`:
```
weights[NYC] = 5570  + 15990 = 21560
weights[TOK] = 9560  +  7820 = 17380
weights[JNB] = 9070  + 11040 = 20110
order = [LON, SYD]
```
Invariant check: `weights[NYC] = dist(NYC,LON) + dist(NYC,SYD) = 5570 + 15990 = 21560`. ✓

Step 3. `argmax` = **NYC** (21,560). Update by `+ dist(NYC, v)`:
```
weights[TOK] = 17380 + 10840 = 28220
weights[JNB] = 20110 + 12840 = 32950
order = [LON, SYD, NYC]
```

Step 4. `argmax` = **JNB** (32,950). Update TOK:
```
weights[TOK] = 28220 + 13540 = 41760
order = [LON, SYD, NYC, JNB]
```

Step 5. Only TOK left.

**Final order:** `[LON, SYD, NYC, JNB, TOK]`

Callers slice `order[:K]`:

- K=1 → `[LON]`
- K=2 → `[LON, SYD]` — the two endpoints of the longest edge
- K=3 → `[LON, SYD, NYC]`
- K=4 → `[LON, SYD, NYC, JNB]`
- K=5 → whole pool

TOK comes last because it's closest to SYD (7,820 km) and adds the least
pair-distance.

### Subtleties

- **Seed determinism.** Only the start-side coin flip
  ([strategies.py:132](../scripts/vp_selection/strategies.py#L132)) consumes
  randomness. Different seeds can only flip the start between the two
  endpoints of the heaviest edge.
- **Ties.** `max(weights, key=weights.get)`
  ([strategies.py:226](../scripts/vp_selection/strategies.py#L226)) breaks
  ties by first-inserted dict key. Distances are floats so ties are rare.
- **Disconnected fallback**
  ([strategies.py:202-214](../scripts/vp_selection/strategies.py#L202-L214))
  — defensive code; never fires on a complete pairwise distance graph.
- **`h1_as` / `h1_city`.** Same loop, but before pure argmax prefers any
  candidate from a not-yet-covered cluster
  ([strategies.py:216-224](../scripts/vp_selection/strategies.py#L216-L224)).
  dist_geo is the no-cluster baseline.
- **`h2_as`** ([strategies.py:143-176](../scripts/vp_selection/strategies.py#L143-L176))
  replaces the single max-edge start with 100 random seeds. The weights
  initialization at
  [strategies.py:158-162](../scripts/vp_selection/strategies.py#L158-L162) is
  the multi-seed analogue: `Σ dist(v, s)` over all 100 seeds — same
  invariant, multiple selected nodes from the start.

---

## Part 2 — Balanced round-robin after dist_geo

The loop at
[stratification.py:581-593](../scripts/processing/ripe_atlas/stratification.py#L581-L593)
runs **inside the per-ASN-bucket pass**. For each bucket we have `ordered`
(dist_geo applied to that bucket's anchors) and want to spread them across
K folds. `fold_sizes` is a **running counter across all buckets processed
so far**, not just the current one.

Two goals in tension:

1. **Within a bucket, give each fold a geographically spread slice.**
2. **Across buckets, keep the K fold totals balanced.**

### The two-goal trick

The dist_geo `ordered` list has a useful property: position 0 is the
most-spread anchor, position 1 the next, etc. If you assign positions
`0, K, 2K, …` to fold 0, `1, K+1, 2K+1, …` to fold 1, and so on
(`preferred = i % K`), each fold ends up with an **evenly-spaced slice of
the diversity ordering**. Consecutive (similar) anchors split across folds
rather than stack.

That's goal #1. Pure `i % K` would handle it perfectly within a bucket, but
ignores `fold_sizes`, so unlucky buckets could drift the totals apart.
Goal #2 is the override:

```python
if fold_sizes[preferred] <= min_size + 1:
    fold = preferred                  # honor diversity ordering
else:
    fold = _smallest_fold(fold_sizes) # break the tie toward balance
```

**Why `min_size + 1`, not `min_size`?** Mid-distribution it's normal for
some folds to be one ahead of others — the natural sawtooth from a
round-robin. Allowing slack of 1 means the `i % K` pattern survives most of
the time. Only when preferred is **2+ ahead** (a real, structural
imbalance) does the override kick in.

### Trace 1 — first bucket, K=3, 7 anchors

`ordered = [v0, v1, v2, v3, v4, v5, v6]` (most-spread first).
`fold_sizes` starts `{0:0, 1:0, 2:0}`.

| i | ip | preferred | min_size | `fold_sizes[pref] ≤ min+1`? | fold | sizes after |
|---|----|-----------|----------|----------------------------|------|-------------|
| 0 | v0 | 0         | 0        | 0 ≤ 1 ✓                    | 0    | `{1,0,0}`   |
| 1 | v1 | 1         | 0        | 0 ≤ 1 ✓                    | 1    | `{1,1,0}`   |
| 2 | v2 | 2         | 0        | 0 ≤ 1 ✓                    | 2    | `{1,1,1}`   |
| 3 | v3 | 0         | 1        | 1 ≤ 2 ✓                    | 0    | `{2,1,1}`   |
| 4 | v4 | 1         | 1        | 1 ≤ 2 ✓                    | 1    | `{2,2,1}`   |
| 5 | v5 | 2         | 1        | 1 ≤ 2 ✓                    | 2    | `{2,2,2}`   |
| 6 | v6 | 0         | 2        | 2 ≤ 3 ✓                    | 0    | `{3,2,2}`   |

Result:
- **fold 0** = {v0, v3, v6}    ← positions 0, 3, 6 of the diversity ordering
- **fold 1** = {v1, v4}        ← positions 1, 4
- **fold 2** = {v2, v5}        ← positions 2, 5

Each fold gets a stride-K slice — geographically spread within the bucket,
and 7 anchors split as 3/2/2 (perfectly balanced for 7 mod 3).

### Trace 2 — override kicks in

Suppose state coming in is `{0:5, 1:2, 2:2}` (fold 0 is 3 ahead). Bucket
has 3 anchors.

| i | preferred | min_size | `fold_sizes[pref] ≤ min+1`? | fold | sizes after |
|---|-----------|----------|----------------------------|------|-------------|
| 0 | 0         | 2        | 5 ≤ 3 **✗**                 | **1** (smallest_fold, lowest-index tiebreak between 1 and 2) | `{5,3,2}`   |
| 1 | 1         | 2        | 3 ≤ 3 ✓                    | 1    | `{5,4,2}`   |
| 2 | 2         | 2        | 2 ≤ 3 ✓                    | 2    | `{5,4,3}`   |

The first anchor would have preferred fold 0, but the global override
redirected it to drain the imbalance instead of compounding it. Anchors at
positions 1 and 2 then resume normal `i % K` placement. A single bucket
can "spend" itself patching up imbalance from earlier buckets without
abandoning diversity ordering for its later items.

### Singleton bucket path

The branch at
[stratification.py:561-566](../scripts/processing/ripe_atlas/stratification.py#L561-L566)
handles `len(bucket_anchors) ≤ 1` — drops the lone anchor into
`_smallest_fold(fold_sizes)`. Same `else` branch as the main loop; with one
anchor there's no diversity ordering to honor, so we go straight to
balance. Both paths share the same `fold_sizes` counter, so a long tail of
tiny ASN buckets quietly fills whichever folds are running behind.

### End property

Per the docstring at
[stratification.py:533-534](../scripts/processing/ripe_atlas/stratification.py#L533-L534):

> Each fold has ~1/K of each ASN bucket with intra-bucket spatial spread.

That's the property the eval protocol needs: a held-out fold for
cross-validation that isn't biased toward a single ASN or a single region
within an ASN.
