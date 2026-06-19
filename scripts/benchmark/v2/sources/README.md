# Adding a new DataSource

A `DataSource` adapts an upstream measurement collection (CSV, ClickHouse,
parquet dump, REST API, …) into the iterators the benchmark's
`materialize-inputs` step writes to disk. Once your source is in the
`SOURCES` registry, every existing combo + Snakemake rule works against
it unchanged — the parquets the runner reads are schema-stable.

This README is the contract you need to satisfy. Use
[vultr_csv.py](vultr_csv.py) (file-only, smallest surface) as the reference
implementation; [ripe_atlas.py](ripe_atlas.py) shows the ClickHouse-backed
pattern.

## Skipping the work: GenericCSVSource

If your data is already a flat CSV (one row per (vp, target, RTT) observation),
you don't need to write a class at all. [generic_csv.py](generic_csv.py)
implements the contract against a fixed canonical schema:

| column | type | required? |
| --- | --- | --- |
| `vp_id`, `vp_lat`, `vp_lon` | str, float, float | yes |
| `target_id`, `target_lat`, `target_lon` | str, float, float | yes |
| `rtt_ms` | float (> 0) | yes |
| `vp_asn`, `target_asn` | int | no |
| `vp_country`, `vp_continent`, `vp_region`, `vp_city` | str | no |
| `target_country`, `target_continent`, `target_region`, `target_city` | str | no |

Steps:

1. Write your CSV with those headers.
2. Edit the `CONFIG` block at the top of [generic_csv.py](generic_csv.py)
   to set `DEFAULT_CSV` to your file's path.
3. `python -m scripts.benchmark.v2.cli materialize-inputs --source generic_csv --slice all`

Slices: `all` (everything) or `head<k>` (first k targets after a
deterministic sort — a cheap smoke slice). Both setups (probes_to_anchors
and anchors_to_probes) work out of the box; the same columns play the VP
role under one setup and the target role under the other.

To benchmark several CSVs side-by-side, subclass `GenericCSVSource` and
give each subclass a distinct `name` — that's the only thing that
distinguishes them in the on-disk tree and the `SOURCES` registry.

Read on if your data shape doesn't fit this schema (e.g. ClickHouse-backed,
landmark/probe-coords come from separate files, custom slicing logic).

## 1. Implement the ABC

Subclass [`DataSource`](base.py) and provide the five abstract methods.
All four iterators stream — yield rows, don't materialize a list.

```python
from scripts.benchmark.v2.sources.base import DataSource, EvalTarget, VpConfig
from scripts.framework.v2 import FitSample
from scripts.framework.v2.types import Coord, Latency, VpId


class MySource(DataSource):
    name = "my_source"                # must be unique across SOURCES

    def __init__(self, slice: str, setup: str = DataSource.PROBES_TO_ANCHORS):
        if setup not in DataSource.ALLOWED_SETUPS:
            raise ValueError(f"unknown setup {setup!r}")
        self._slice = slice
        self._setup = setup

    def slice_id(self) -> str: return self._slice
    def setup_id(self) -> str: return self._setup

    def iter_vp_configs(self) -> Iterator[VpConfig]: ...
    def iter_fit_samples(self) -> Iterator[FitSample]: ...
    def iter_eval_targets(self) -> Iterator[EvalTarget]: ...
```

### `iter_vp_configs`

One [`VpConfig`](base.py) per *unique* VP your source exposes. Drives
`vp_configs.parquet`. `asn` / `country` / `continent` / `region` / `city`
are optional (None when absent).
`vp_id` must be stable — a probe ID, anchor IP, or a deterministic
synthetic string. Two rows with the same `vp_id` later in
`iter_fit_samples` / `iter_eval_targets` are assumed to refer to the same
VP.

### `iter_fit_samples`

Flat stream of [`FitSample(vp_id, vp_coord, probe_coord, latency)`](../../../framework/v2/ltd/base.py).
One row per (VP, target-with-known-coord, RTT) triple. Used by
`LTDModel.fit(samples)` to learn the RTT→distance relation.

- `vp_coord` is the VP's location.
- `probe_coord` is the *other* point's location — the "training target".
- `latency > 0` (the type checker rejects ≤ 0 via [`_validate_latency`](../../../framework/v2/ltd/base.py)).

The same (vp, target) pair contributes to both fit and eval — the source
doesn't need to partition the data. LTD models that need a fit/eval split
do it internally.

### `iter_eval_targets`

One [`EvalTarget(target_id, true_coord, obs)`](base.py) per evaluation
target, where `obs` is `list[tuple[VpId, Coord, Latency]]` of *every*
VP→target observation available. Drives `eval_observations.parquet` and
seeds `CBGModel.geolocate(obs)`.

- `target_id` must be stable and unique within this source+slice+setup.
- `true_coord` is the geolocation ground truth used to score the
  prediction (haversine in the runner). If you don't have hard GT, you
  can't use this benchmark — the source layer is *not* the place to
  invent a label.
- `obs` empty is allowed but the runner will mark that row
  `INSUFFICIENT_DATA` immediately, so usually filter upstream.

### `slice_id` and `setup_id`

Both become directory levels in the on-disk layout (see §3). Don't return
free-form strings — pick a small, fixed vocabulary:

- `slice_id`: e.g. `"all_us"`, `"top1"`, `"n723"`. Used to keep different
  subsets of the same source from clobbering each other's parquets.
- `setup_id`: exactly one of `DataSource.PROBES_TO_ANCHORS` or
  `DataSource.ANCHORS_TO_PROBES`. See §2 — every source must support both,
  even if one is degenerate, so the analysis grid stays orthogonal.

## 2. The two setups

The CBG framework is symmetric in (probe, anchor) — the only question is
which side acts as the *vantage point* (VP). Your source must produce both
configurations from the same underlying measurement table.

| setup_id | VP is | Target is | When to use |
| --- | --- | --- | --- |
| `probes_to_anchors` | probe | anchor | Canonical IMC 2023 direction. Anchors are hard GT; many probes per anchor. |
| `anchors_to_probes` | anchor | probe | Role-swap pressure test. Far more targets, fewer VPs, more challenging. |

The right place for the branch is inside each iterator — gate on
`self._setup`, swap which column becomes `vp_id` / `vp_coord` vs
`target` / `probe_coord`. [vultr_csv.py](vultr_csv.py) shows the pattern:
the same DataFrame is interpreted two different ways depending on
`setup_id`, with no duplication of the loader.

## 3. On-disk layout the source produces

[`materialize_inputs`](../inputs.py) calls your iterators and writes:

```
inputs/
  <source.name>/
    <setup_id>/
      <slice_id>/
        vp_configs.parquet      # one row per VP (VP_CONFIGS_SCHEMA)
        fit_samples.parquet     # one row per FitSample (FIT_SAMPLES_SCHEMA)
        eval_observations.parquet  # one row per (target, vp) obs (EVAL_OBSERVATIONS_SCHEMA)
        manifest.json           # counts + generated_at
```

The schemas are pinned in [schema.py](../schema.py). You don't write
parquet yourself — just yield well-formed `VpConfig` / `FitSample` /
`EvalTarget` objects and the writer maps them to columns. Schema drift is
caught at parquet-write time, not silently absorbed.

## 4. Register in `SOURCES`

[`__init__.py`](__init__.py) keeps the name → class registry:

```python
from scripts.benchmark.v2.sources.my_source import MySource

SOURCES: dict[str, type[DataSource]] = {
    VultrCSVSource.name: VultrCSVSource,
    RipeAtlasSource.name: RipeAtlasSource,
    MySource.name: MySource,
}
```

The CLI's `--source` flag looks the name up here. `name` must match what
you set on the class.

## 5. Validation checklist before you ship

- [ ] `iter_vp_configs` yields each `vp_id` at most once.
- [ ] Every `vp_id` referenced in fit-samples / eval-obs also appears in
      `iter_vp_configs` (same string, exactly).
- [ ] `latency > 0` everywhere; NaN coords filtered out at the source.
- [ ] Both `setup_id` values produce non-empty fit + eval streams.
- [ ] `slice_id` is parameterised and at least one "small" slice exists
      (≤ a few hundred targets) — fast `materialize-inputs` makes
      iteration painless.
- [ ] Iterators are pure: calling them twice yields identical sequences
      in the same order. (The materializer reads each iterator once, but
      tests will call them repeatedly.)
- [ ] Smoke-test the round-trip:
      `python -m scripts.benchmark.v2.cli materialize-inputs --source my_source --slice <small>`
      then read the resulting parquets and confirm counts match
      `manifest.json`.

## 6. Things you probably don't need to touch

- Parquet writers, schema definitions, runner, instrumentation, Snakefile.
  All of these are source-agnostic once the iterators are right.
- The framework's `CBGModel.from_config(...)` API. Don't pre-fit or
  pre-filter — the runner owns that lifecycle.
