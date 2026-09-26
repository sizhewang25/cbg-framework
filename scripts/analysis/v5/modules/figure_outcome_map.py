"""Where a cohort's predictions actually landed: one map per method x dataset.

The outcome bars say *how many* predictions fell in the right serving region.
They cannot say **where**, and on these meshes that is the interesting half:
two methods can hold the same `cell_label` share while one is right about the
coasts and wrong about the interior and the other is the reverse. This module
draws the predictions themselves, one panel per `(method, dataset)`, rows =
method and columns = dataset, so a reader compares down a column.

## One cohort per figure

`correct` and `wrong` are drawn as two separate figures rather than two marker
shapes on one. Overlaying them doubles the ink in exactly the metros where the
map is already unreadable (see the known limitation below), and the two read
differently anyway: on the `correct` map a cell's number is "predictions that
landed here and belonged here", on the `wrong` map it is "predictions that
should have landed here and did not". Same keying, opposite sense; one legend
cannot carry both. Rejected: one figure with `o`/`X` overlaid, and a third
`unanswered` cohort -- an unanswered row has no coordinate, so it has nothing
to draw.

## The frame is shared, and it is an argument

`extent` defaults to `DEFAULT_EXTENT`, wider than `mapping.US_MAINLAND_EXTENT`
because SPO's predictions run into Canada and out over the Gulf. It is one
frame for all eighteen panels on purpose: a per-panel `mapping.auto_extent`
would rescale every panel to its own data, and a reader comparing OCT-H's
column against SPO's would be comparing two different maps at two different
scales. A prediction outside the frame is **not dropped** -- it is drawn as a
dark-red caret pinned at the edge and counted in the title (as01/SPO: 20 of
354 `correct`), because silently cropping the worst predictions is how a map
flatters a method.

## solved_mask, and the 107 rows it removes

A FALLBACK row carries the S-P baseline's coordinate, not the method's, so it
is filtered by `status.solved_mask` before anything is drawn or counted.
Without that mask as01/VAN's `correct` count reads 270 instead of 163 -- 107
rows of the baseline's accuracy credited to the variant. The excluded count is
printed in each panel title (`N FB excl.`) rather than left implicit, so the
denominator is visible where the number is.

## Counts key on `tg_seed_id`

The number drawn by a seed is the count of *that cohort's TGs whose own seed
is this one*, never `pred_seed_id`. On the `correct` map the two agree by
definition. On the `wrong` map they do not, and `tg_seed_id` is the one an
operator asks for: "how many of this region's targets did the method lose?"
Keying on `pred_seed_id` would answer "how many strays landed here", which is
a different figure and one the arrows already show.

## Colour is a discrete ramp, not continuous

`pred_dist_to_tg_grid` on `turbo` binned at `OFFSET_BOUNDS`
`[0, 1, 2, 3, 5, 8, 12, 20, 40]`. Discrete because a continuous ramp cannot be
read back to a number off a 3-inch panel, and the first three bins are the
ring tiers `classify.summarize` already counts, so the map and the bars agree
on their vocabulary. Caveat: an offset above 40 takes turbo's over-colour and
is indistinguishable from the 20-40 bin; the true maximum is in every panel
title and in the CSV twin (the observed maximum on as01-03 is 35, as01/SPO).

## Known limitation: labels are repelled from labels, not from markers

`place_labels` is a spring-repulsion relaxation between the count labels. It
knows nothing about the prediction markers, so in the dense north-east cluster
(NYC / Philadelphia / DC / Boston, four seeds inside ~2 degrees) a count can
come to rest on top of a marker, and two leader lines can cross. The numbers
are still correct -- they are in the CSV twin -- but the drawing is locally
ambiguous. Not fixed here. The candidates, in the order they are worth trying:

1. add the markers to the repulsion as fixed obstacles, so a label is pushed
   off a prediction the same way it is pushed off another label;
2. allocate labels by angle around the cluster centroid instead of relaxing
   them independently, which makes non-crossing leaders a guarantee rather
   than an outcome;
3. a callout column beside the panel for the densest cluster, with the map
   carrying only a tick -- the only one of the three that scales past ~6 seeds
   in a degree.

One deliberate deviation from the prototype this was ported from:
`place_labels` there left **exactly coincident** anchors coincident forever
(their separation direction is 0/0, so no push exists). Real seeds are
distinct, so the figure is unchanged, but the routine is now total: a
degenerate pair gets a deterministic antisymmetric direction (`_TIE_AXIS`).

Command: `plot-outcome-map`. Needs `build-answer-space` and `classify` on
every run. Writes `_cross/outcome-map/<datasets>[@<arm>]/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402

from scripts.analysis.v5.modules import cells as CL  # noqa: E402
from scripts.analysis.v5.modules import classify as C  # noqa: E402
from scripts.analysis.v5.modules import cross  # noqa: E402
from scripts.analysis.v5.modules import grid as G  # noqa: E402
from scripts.analysis.v5.modules import mapping as M  # noqa: E402
from scripts.analysis.v5.modules.map_answer_space import load_rung  # noqa: E402
from scripts.analysis.v5.modules.methods import (  # noqa: E402
    method_label,
    method_order,
    method_term_table,
)
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths  # noqa: E402
from scripts.analysis.v5.modules.status import solved_mask  # noqa: E402

#: `_cross/<KIND>/<datasets>[@<arm>]/`.
KIND = "outcome-map"

#: Which rung's answer space and `*_tgs.parquet` are drawn. v5 runs one
#: resolution, so this selects a file rather than a variant -- but it is read
#: off the ladder, not spelled, so a second rung cannot silently desync the
#: seeds from the labels.
SOURCE_NSIDE = G.NSIDE_LADDER[0]

#: The cohorts a panel can draw: the two *graded* cell labels. `unanswered` is
#: excluded by construction -- it has no coordinate.
COHORTS: tuple[str, ...] = C.GRADED_CELL_LABELS

#: One frame for every panel. Wider than `M.US_MAINLAND_EXTENT` on all four
#: sides: SPO answers into Canada and over the Gulf, and a frame that cropped
#: them would hide the failures this figure exists to show.
DEFAULT_EXTENT: tuple[float, float, float, float] = (-128.0, -63.0, 21.0, 55.0)

#: `pred_dist_to_tg_grid` bins. The first three are `classify`'s ring tiers, so
#: the map and the outcome bars name the same things the same way; past that
#: the bins widen because the difference between 21 and 39 grids out is not a
#: difference an operator acts on.
OFFSET_BOUNDS: tuple[int, ...] = (0, 1, 2, 3, 5, 8, 12, 20, 40)

#: `cohort -> (marker, marker area in points^2)`. A filled circle for the good
#: outcome and a cross for the bad one, the cross slightly larger because its
#: strokes carry less fill and so less of the offset colour.
COHORT_MARKER: dict[str, tuple[str, float]] = {"correct": ("o", 15.0), "wrong": ("X", 24.0)}

#: Off-frame predictions. A hue of its own -- nothing else on the map is red --
#: because the caret's *position* is a lie (it is pinned at the frame edge) and
#: it must not be mistaken for a prediction that landed there.
OFF_MAP_INK = "#8b0000"

#: Seed dots. Half `M.MARKER_AREA`: eighteen panels across 17.5 inches leaves
#: each map ~3 in wide, and the answer-space map's marker fills a cell here.
SEED_AREA = 7.0

#: Prediction -> its target. Thin and pale: it is context for the marker, and
#: at 350 marks a panel a heavier line becomes the panel.
JOIN_PT = 0.35
JOIN_ALPHA = 0.55

#: A count is joined back to its seed only once it has been displaced further
#: than this, in degrees. Below it the label is unambiguously its seed's and a
#: leader line is one more mark for nothing.
LEADER_MIN_DEG = 1.3

#: The label relaxation. `MIN_SEP_DEG` is roughly two digits' width at the
#: drawn scale; `SPRING` is weak enough that a crowded run spreads before it is
#: pulled back, strong enough that an isolated label stays on its seed.
LABEL_OFFSET_DEG = (0.7, 0.55)
MIN_SEP_DEG = 2.6
FRAME_PAD_DEG = 1.0
RELAX_ITERS = 400
SPRING = 0.03

#: Below this separation two labels have no direction to push apart along.
_COINCIDENT_EPS = 1e-9

#: The direction a degenerate (exactly coincident) pair separates along, used
#: antisymmetrically so the pair moves apart rather than drifting together.
_TIE_AXIS = (1.0, 0.0)

PNG_NAME = "outcome_map.{cohort}.png"
CSV_NAME = "outcome_map.{cohort}.csv"
MANIFEST_NAME = "outcome_map.{cohort}.manifest.json"

#: Per-seed offset percentiles in the CSV twin. `max` is reported beside them
#: because it is the bound a panel title quotes.
SEED_QUANTILES = (0.5, 0.9)

#: Columns `draw_panel` and the CSV twin need. Read explicitly so a parquet
#: that has drifted fails here rather than three frames later.
_TG_COLUMNS = (
    "tg_id", "tg_lat", "tg_lon", "pred_lat", "pred_lon", "status",
    "site_id", "tg_seed_id", "pred_dist_to_tg_grid", "cell_label",
)

#: Panel geometry. The width is the prototype's 17.5 in over three datasets,
#: kept per-column so a two- or four-dataset call keeps the same panel size.
_PANEL_W = 17.5 / 3
_PANEL_H = 3.15

#: Drawn bottom-up. Cells and seeds sit above the basemap, the joins above
#: them, the markers above those, the carets above the markers, and the labels
#: on top of everything -- a count that a marker could hide would be unreadable.
_Z_CELLS = 4.0
_Z_SEEDS = 4.5
_Z_JOIN = 5.0
_Z_MARKER = 7.0
_Z_OFF_MAP = 9.0
_Z_LEADER = 10.0
_Z_LABEL = 11.0

#: A white stroke under the count labels and their leaders, so a number over a
#: cell boundary or a marker is still legible. It does not *solve* the
#: collision documented above; it makes the digits survive it.
_HALO = [pe.withStroke(linewidth=2.4, foreground="white")]
_LEADER_HALO = [pe.withStroke(linewidth=1.8, foreground="white")]


def offset_norm() -> tuple["matplotlib.colors.Colormap", BoundaryNorm]:
    """The discrete `turbo` ramp over `OFFSET_BOUNDS`, and its norm."""
    cmap = plt.get_cmap("turbo")
    return cmap, BoundaryNorm(list(OFFSET_BOUNDS), cmap.N)


def validate_cohort(cohort: str) -> str:
    if cohort not in COHORTS:
        raise ValueError(
            f"unknown cohort {cohort!r}; pick from {list(COHORTS)}. "
            f"{C.UNANSWERED!r} has no prediction to draw."
        )
    return cohort


def output_dir(run_ids: list[str], *, analysis_root: Path | None = None) -> Path:
    """`_cross/outcome-map/<datasets>[@<arm>]/`, created."""
    return cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)


# -- the label relaxation -------------------------------------------------


def place_labels(
    anchors,
    extent: tuple[float, float, float, float] = DEFAULT_EXTENT,
    *,
    min_sep: float = MIN_SEP_DEG,
    pad: float = FRAME_PAD_DEG,
    iters: int = RELAX_ITERS,
    spring: float = SPRING,
) -> np.ndarray:
    """Push overlapping count labels apart, keeping each near its own seed.

    `anchors` are `(lon, lat)` seed positions; the return is the drawn label
    position for each, in the same order and in the same units.

    Each label starts at its anchor plus `LABEL_OFFSET_DEG`, is pushed out of
    any neighbour within `min_sep`, is pulled back towards that start by
    `spring`, and is clipped `pad` inside the frame. It repels labels from
    labels only -- see the module docstring for what that costs and what would
    fix it.

    The diagonal is masked by setting each self-distance to `min_sep` exactly,
    which is never "too close" and never divides by zero. Filling it with `inf`
    instead (as the prototype did) leaves `inf * 0` in the push term: the
    result is identical because `np.where` discards it, but numpy warns on
    every call, and a figure module that cannot be run under `-W error` is a
    figure module whose real warnings nobody reads.
    """
    a = np.asarray(anchors, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"anchors must be (n, 2) lon/lat, got {a.shape}")
    home = a + np.array(LABEL_OFFSET_DEG)
    pos = home.copy()
    n = len(a)
    # Deterministic, antisymmetric fallback direction for coincident labels.
    idx = np.arange(n)
    tie = np.zeros((n, n, 2))
    tie[..., 0] = np.sign(idx[:, None] - idx[None, :]) * _TIE_AXIS[0]
    tie[..., 1] = np.sign(idx[:, None] - idx[None, :]) * _TIE_AXIS[1]

    for _ in range(iters):
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(d, axis=-1)
        np.fill_diagonal(dist, min_sep)
        too_close = dist < min_sep
        if too_close.any():
            safe = np.where(dist > _COINCIDENT_EPS, dist, 1.0)[..., None]
            unit = np.where(dist[..., None] > _COINCIDENT_EPS, d / safe, tie)
            gap = np.where(too_close, (min_sep - dist) / 2, 0.0)
            pos += (gap[..., None] * unit).sum(axis=1)
        pos += spring * (home - pos)
        pos[:, 0] = np.clip(pos[:, 0], extent[0] + pad, extent[1] - pad)
        pos[:, 1] = np.clip(pos[:, 1], extent[2] + pad, extent[3] - pad)
    return pos


# -- loading --------------------------------------------------------------


@dataclass(frozen=True)
class MapData:
    """Everything the figure draws, loaded once and reused by both cohorts.

    `frames` is keyed `(run_id, method)` and holds the *whole* scored frame,
    unmasked: the panel title needs the FALLBACK count, which `cohort_rows`
    has by then removed.
    """

    run_ids: list[str]
    methods: list[str]
    frames: dict[tuple[str, str], pd.DataFrame]
    seeds: dict[str, pd.DataFrame]
    nside: int


def scored_methods(
    run: RunPaths, nside: int = SOURCE_NSIDE, *, analysis_root: Path | None = None
) -> list[str]:
    """Methods with a `*_tgs.parquet` at this rung. Read from disk, not a config."""
    suffix = C.TGS_PARQUET.format(method="")
    d = run.classify_dir(nside, root=analysis_root)
    return sorted(p.name[: -len(suffix)] for p in d.glob("*" + suffix))


def load(
    runs: list[RunPaths],
    *,
    methods: list[str] | None = None,
    nside: int = SOURCE_NSIDE,
    analysis_root: Path | None = None,
) -> MapData:
    """The scored frames and the seeds, for every `(run, method)` drawn.

    Method coverage is strict, as everywhere else in `_cross/`: a method
    missing from one run would leave a hole in one column of the grid and the
    reader would have to guess whether it was unscored or empty.
    """
    available = {
        r.run_id: set(scored_methods(r, nside, analysis_root=analysis_root)) for r in runs
    }
    if not any(available.values()):
        raise MissingArtifactError(
            f"no method is scored at nside={nside} under "
            f"{[str(r.classify_dir(nside, root=analysis_root)) for r in runs]}; "
            f"run `classify` first."
        )
    chosen = cross.guard_common_methods(
        available, remedy="drop the run that does not carry them."
    )
    if methods:
        missing = sorted(set(methods) - set(chosen))
        if missing:
            raise ValueError(
                f"{missing} are not scored in every run "
                f"({ {r: sorted(m) for r, m in available.items()} })."
            )
        chosen = [m for m in chosen if m in methods]
    chosen = method_order(chosen)

    frames: dict[tuple[str, str], pd.DataFrame] = {}
    seeds: dict[str, pd.DataFrame] = {}
    for run in runs:
        seeds[run.run_id] = load_rung(run, nside, analysis_root=analysis_root).seeds
        for m in chosen:
            path = run.classify_dir(nside, root=analysis_root) / C.TGS_PARQUET.format(method=m)
            if not path.exists():
                raise MissingArtifactError(
                    f"{path} missing; run `classify --run-id {run.run_id}` first"
                )
            frames[(run.run_id, m)] = pd.read_parquet(path, columns=list(_TG_COLUMNS))
    return MapData(
        run_ids=[r.run_id for r in runs],
        methods=chosen,
        frames=frames,
        seeds=seeds,
        nside=int(nside),
    )


# -- the numbers ----------------------------------------------------------


def cohort_rows(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """The rows one panel draws: this cohort's label, solved, with a coordinate.

    `solved_mask` is not optional -- see the module docstring. `pred_lat`
    non-null is belt and braces: a solved row always carries one, and a frame
    where it does not is a frame this figure should not silently thin.
    """
    validate_cohort(cohort)
    return frame[
        (frame["cell_label"] == cohort)
        & solved_mask(frame).to_numpy()
        & frame["pred_lat"].notna()
    ]


def off_frame(rows: pd.DataFrame, extent: tuple[float, float, float, float]) -> np.ndarray:
    """Boolean mask of predictions outside the drawn frame."""
    lon_min, lon_max, lat_min, lat_max = extent
    return (
        (rows["pred_lon"] < lon_min)
        | (rows["pred_lon"] > lon_max)
        | (rows["pred_lat"] < lat_min)
        | (rows["pred_lat"] > lat_max)
    ).to_numpy()


def panel_counts(
    frame: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    run_id: str,
    method: str,
    cohort: str,
    extent: tuple[float, float, float, float],
) -> dict:
    """The numbers a panel title carries, and the manifest and CSV repeat.

    `n_fallback_excluded` is counted over the **whole** frame, not the cohort:
    a FALLBACK row has no cell label of the method's own, so it belongs to
    neither cohort and the figure has to say so once per panel.
    """
    offsets = rows["pred_dist_to_tg_grid"]
    return {
        "run_id": run_id,
        "dataset": cross.short_dataset(run_id),
        "method": method,
        "method_label": method_label(method),
        "cohort": cohort,
        "n_cohort_tgs": int(len(rows)),
        "n_cohort_sites": int(rows["site_id"].nunique()),
        "n_cohort_cells": int(rows["tg_seed_id"].nunique()),
        "n_off_map": int(off_frame(rows, extent).sum()),
        "n_fallback_excluded": int((frame["status"] == "FALLBACK").sum()),
        "cohort_offset_p50": float(offsets.median()) if len(rows) else float("nan"),
        "cohort_offset_max": float(offsets.max()) if len(rows) else float("nan"),
    }


def seed_rows(rows: pd.DataFrame, seeds: pd.DataFrame, counts: dict) -> pd.DataFrame:
    """One row per `(run, method, cohort, seed)`: the drawn count and its spread.

    Keyed on `tg_seed_id`, which is the number the panel prints. A site belongs
    to exactly one seed, so `n_sites` sums over the rows to the panel's own
    site count and the title is reconstructible from the CSV alone.

    A panel whose cohort is empty still emits one row, with a null `seed_id`
    and `n_tgs` 0, so its title -- in particular `n_fallback_excluded` -- does
    not vanish from the twin.
    """
    panel = {k: v for k, v in counts.items()}
    pos = seeds.set_index("seed_id")[["seed_lat", "seed_lon"]]
    if not len(rows):
        return pd.DataFrame([{**panel, "seed_id": pd.NA, "seed_lat": np.nan,
                              "seed_lon": np.nan, "n_tgs": 0, "n_sites": 0,
                              **{f"offset_p{int(q * 100)}": np.nan for q in SEED_QUANTILES},
                              "offset_max": np.nan}])
    by_seed = rows.groupby("tg_seed_id", sort=True)
    out = pd.DataFrame(
        {
            "n_tgs": by_seed.size(),
            "n_sites": by_seed["site_id"].nunique(),
            "offset_max": by_seed["pred_dist_to_tg_grid"].max().astype(float),
        }
    )
    for q in SEED_QUANTILES:
        out[f"offset_p{int(q * 100)}"] = by_seed["pred_dist_to_tg_grid"].quantile(q)
    out = out.rename_axis("seed_id").reset_index()
    out = out.join(pos, on="seed_id")
    for k, v in reversed(panel.items()):
        out.insert(0, k, v)
    return out


def csv_columns() -> list[str]:
    """The CSV twin's columns, in order: what the panel says, then what each
    drawn number is made of."""
    return [
        "run_id", "dataset", "method", "method_label", "cohort",
        "n_cohort_tgs", "n_cohort_sites", "n_cohort_cells", "n_off_map",
        "n_fallback_excluded", "cohort_offset_p50", "cohort_offset_max",
        "seed_id", "seed_lat", "seed_lon", "n_tgs", "n_sites",
        *[f"offset_p{int(q * 100)}" for q in SEED_QUANTILES],
        "offset_max",
    ]


def build_csv(
    data: MapData, cohort: str, extent: tuple[float, float, float, float]
) -> tuple[pd.DataFrame, list[dict]]:
    """The CSV twin and the panel counts, in drawing order (row-major)."""
    frames, counts = [], []
    for method in data.methods:
        for run_id in data.run_ids:
            frame = data.frames[(run_id, method)]
            rows = cohort_rows(frame, cohort)
            c = panel_counts(
                frame, rows, run_id=run_id, method=method, cohort=cohort, extent=extent
            )
            counts.append(c)
            frames.append(seed_rows(rows, data.seeds[run_id], c))
    table = pd.concat(frames, ignore_index=True)
    return table.reindex(columns=csv_columns()), counts


# -- drawing --------------------------------------------------------------


def panel_title(counts: dict) -> str:
    """Two lines: what the panel holds, then how far off it was."""

    def _q(key: str) -> str:
        v = counts[key]
        return "—" if not np.isfinite(v) else f"{v:.0f}"

    t = (
        f"{counts['method_label']} · {counts['dataset']} · "
        f"{counts['cohort']}={counts['n_cohort_tgs']} "
        f"({counts['n_cohort_sites']} sites, {counts['n_cohort_cells']} cells)\n"
        f"offset p50={_q('cohort_offset_p50')} max={_q('cohort_offset_max')}"
    )
    if counts["n_off_map"]:
        t += f" · {counts['n_off_map']} off-map"
    if counts["n_fallback_excluded"]:
        t += f" · {counts['n_fallback_excluded']} FB excl."
    return t


def draw_panel(
    ax,
    rows: pd.DataFrame,
    seeds: pd.DataFrame,
    polygons: dict,
    counts: dict,
    *,
    cohort: str,
    extent: tuple[float, float, float, float],
) -> dict:
    """One `(method, dataset)` map. Returns the counts it drew.

    Basemap, cells, seeds, then per prediction a join line to its target and a
    marker coloured by its grid offset; predictions outside `extent` become
    carets at the edge. Finally the per-cell counts, relaxed apart.
    """
    import cartopy.crs as ccrs

    cmap, norm = offset_norm()
    marker, area = COHORT_MARKER[validate_cohort(cohort)]

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    M.draw_basemap(ax)
    M.draw_cells(ax, polygons, zorder=_Z_CELLS)
    ax.scatter(
        seeds["seed_lon"], seeds["seed_lat"], s=SEED_AREA, c=M.INK, marker="o",
        linewidths=0, transform=ccrs.PlateCarree(), zorder=_Z_SEEDS,
    )

    off = off_frame(rows, extent)
    visible = rows[~off]
    for r in visible.itertuples():
        ax.plot(
            [r.tg_lon, r.pred_lon], [r.tg_lat, r.pred_lat], lw=JOIN_PT, color=M.MUTED,
            alpha=JOIN_ALPHA, transform=ccrs.Geodetic(), zorder=_Z_JOIN,
        )
    ax.scatter(
        visible["pred_lon"], visible["pred_lat"], c=visible["pred_dist_to_tg_grid"],
        cmap=cmap, norm=norm, s=area, marker=marker, linewidths=0.25, edgecolors="#222",
        transform=ccrs.PlateCarree(), zorder=_Z_MARKER,
    )
    # Pinned at the frame edge: the caret says "further out this way", which is
    # why it takes a hue no in-frame mark uses.
    for r in rows[off].itertuples():
        ax.plot(
            np.clip(r.pred_lon, extent[0] + 0.6, extent[1] - 0.6),
            np.clip(r.pred_lat, extent[2] + 0.6, extent[3] - 0.6),
            marker="v", ms=6, color=OFF_MAP_INK,
            transform=ccrs.PlateCarree(), zorder=_Z_OFF_MAP,
        )

    draw_counts(ax, rows, seeds, extent=extent)
    ax.set_title(panel_title(counts), fontsize=8.0, linespacing=1.3)
    return counts


def draw_counts(
    ax, rows: pd.DataFrame, seeds: pd.DataFrame, *, extent: tuple[float, float, float, float]
) -> int:
    """The per-cell counts, keyed on `tg_seed_id`. Returns how many were drawn.

    A seed the answer space does not carry is skipped rather than drawn at an
    invented position; `classify` assigns every TG a seed from this same
    answer space, so the filter should never fire and is a guard, not a policy.
    """
    import cartopy.crs as ccrs

    if not len(rows):
        return 0
    n_by_seed = rows.groupby("tg_seed_id").size()
    pos = seeds.set_index("seed_id")
    ids = [k for k in n_by_seed.index if k in pos.index]
    if not ids:
        return 0
    anchors = np.c_[
        pos.loc[ids, "seed_lon"].to_numpy(), pos.loc[ids, "seed_lat"].to_numpy()
    ]
    placed = place_labels(anchors, extent)
    for (ax0, ay0), (lx, ly), seed_id in zip(anchors, placed, ids):
        if np.hypot(lx - ax0, ly - ay0) > LEADER_MIN_DEG:
            ax.plot(
                [ax0, lx], [ay0, ly], lw=0.55, color=M.INK_2, alpha=0.9,
                transform=ccrs.PlateCarree(), zorder=_Z_LEADER,
                path_effects=_LEADER_HALO,
            )
        ax.text(
            lx, ly, str(int(n_by_seed[seed_id])), fontsize=7.4, fontweight="bold",
            color=M.INK, ha="center", va="center", transform=ccrs.PlateCarree(),
            zorder=_Z_LABEL, path_effects=_HALO,
        )
    return len(ids)


_SUBJECT = {
    "correct": (
        "Predictions scored CORRECT on the cell axis (solved rows only) · "
        "● = prediction, grey line to its target · small dot = seed, thin line "
        "to its count · ▼ = off-map"
    ),
    "wrong": (
        "Predictions scored WRONG on the cell axis (solved rows only) · "
        "✕ = prediction, grey line to its target · small dot = seed, thin line "
        "to the count that SHOULD have landed there · ▼ = off-map"
    ),
}

_CBAR_LABEL = (
    "grid offset (pred_dist_to_tg_grid) — 0 = prediction in the target's own grid"
)


def render(
    data: MapData,
    out_png: Path,
    *,
    cohort: str,
    extent: tuple[float, float, float, float] = DEFAULT_EXTENT,
    dpi: int = 125,
) -> tuple[Path, list[dict]]:
    """Rows = method (`methods.TERM_ORDER`), columns = dataset. `(png, counts)`.

    The cell polygons are built once per run and shared by that run's column:
    they belong to the answer space, not to a method, and the planar Voronoi is
    the slowest thing on the page.
    """
    import cartopy.crs as ccrs

    validate_cohort(cohort)
    polygons = {rid: CL.cell_polygons(data.seeds[rid], extent) for rid in data.run_ids}
    nrows, ncols = len(data.methods), len(data.run_ids)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(_PANEL_W * ncols, _PANEL_H * nrows),
        squeeze=False, subplot_kw={"projection": ccrs.PlateCarree()},
    )
    counts = []
    for i, method in enumerate(data.methods):
        for j, run_id in enumerate(data.run_ids):
            frame = data.frames[(run_id, method)]
            rows = cohort_rows(frame, cohort)
            c = panel_counts(
                frame, rows, run_id=run_id, method=method, cohort=cohort, extent=extent
            )
            counts.append(
                draw_panel(
                    axes[i, j], rows, data.seeds[run_id], polygons[run_id], c,
                    cohort=cohort, extent=extent,
                )
            )

    cmap, norm = offset_norm()
    cb = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes, orientation="horizontal",
        fraction=0.018, pad=0.02, ticks=list(OFFSET_BOUNDS), aspect=70,
    )
    cb.set_label(_CBAR_LABEL, fontsize=9)
    fig.suptitle(_SUBJECT[cohort], fontsize=12, y=0.995)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png, counts


def _manifest(
    data: MapData,
    counts: list[dict],
    *,
    cohort: str,
    extent: tuple[float, float, float, float],
    png_name: str,
    csv_name: str,
) -> str:
    body = {
        "figure": png_name,
        "csv": csv_name,
        "cohort": cohort,
        "datasets": cross.dataset_slug(data.run_ids),
        "run_ids": data.run_ids,
        "arm": cross.arm(data.run_ids),
        "grid": G.describe(data.nside),
        "source_nside": data.nside,
        "extent": dict(zip(("lon_min", "lon_max", "lat_min", "lat_max"), extent)),
        "row_order": [method_label(m) for m in data.methods],
        "column_order": [cross.short_dataset(r) for r in data.run_ids],
        "method_terms": method_term_table(data.methods),
        "panels": counts,
        "encoding": {
            "prediction": (
                f"marker {COHORT_MARKER[cohort][0]!r}, filled by "
                f"pred_dist_to_tg_grid on a discrete turbo ramp at {list(OFFSET_BOUNDS)}"
            ),
            "join": f"{M.MUTED} {JOIN_PT} pt from each prediction to its target, geodetic",
            "seed": f"{M.INK} dot, one per seed of the run's answer space",
            "cell": f"{M.CELL_EDGE} boundary: Voronoi cell of a seed, unbounded",
            "count": (
                "per-cell count of this cohort's TGs, keyed on tg_seed_id, "
                f"repelled apart and joined back to its seed past {LEADER_MIN_DEG} deg"
            ),
            "off_map": f"{OFF_MAP_INK} caret pinned at the frame edge",
        },
        "policy": {
            "solved_mask": (
                "status.solved_mask is applied before anything is drawn or "
                "counted. A FALLBACK row carries the shortest-ping baseline's "
                "coordinate, so counting it credits the variant with the "
                "baseline's answer: as01/vanilla_cbg reads 270 correct without "
                "the mask and 163 with it. n_fallback_excluded is per panel and "
                "over the whole frame, not the cohort."
            ),
            "count_key": (
                "tg_seed_id, never pred_seed_id. On the wrong map the number "
                "over a cell is 'predictions that should have landed here', "
                "which is the operator's question; pred_seed_id would answer "
                "'strays that landed here' instead."
            ),
            "extent": (
                "One frame for every panel, so the columns are comparable. "
                "Predictions outside it are drawn as carets at the edge and "
                "counted in n_off_map, never dropped."
            ),
            "colour_bins": (
                "Discrete: the first three bins are classify's ring tiers. An "
                "offset above the last boundary takes turbo's over-colour and "
                "is indistinguishable from the top bin; cohort_offset_max "
                "carries the true bound."
            ),
        },
        "known_limitation": (
            "place_labels repels labels from labels, not from the prediction "
            "markers, so in the dense north-east cluster (NYC / Philadelphia / "
            "DC / Boston) a count can land on a marker and leader lines can "
            "cross. The numbers are in the CSV twin. Candidate fixes, unimplemented: "
            "treat the markers as obstacles in the repulsion; allocate labels by "
            "angle around a cluster centroid so leaders cannot cross; or a callout "
            "column for the densest cluster."
        ),
    }
    return json.dumps(body, indent=2) + "\n"


def build_for_runs(
    runs: list[RunPaths],
    *,
    cohorts: list[str] | None = None,
    methods: list[str] | None = None,
    extent: tuple[float, float, float, float] = DEFAULT_EXTENT,
    nside: int = SOURCE_NSIDE,
    analysis_root: Path | None = None,
) -> list[Path]:
    """PNG, CSV twin and manifest per cohort. Returns the PNGs."""
    wanted = list(dict.fromkeys(cohorts or COHORTS))
    for c in wanted:
        # Before the parquets are read: loading three meshes costs seconds.
        validate_cohort(c)
    nside = G.validate_nside(nside)
    data = load(runs, methods=methods, nside=nside, analysis_root=analysis_root)
    out_dir = output_dir(data.run_ids, analysis_root=analysis_root)
    written: list[Path] = []
    for cohort in wanted:
        names = {k: v.format(cohort=cohort) for k, v in
                 zip(("png", "csv", "man"), (PNG_NAME, CSV_NAME, MANIFEST_NAME))}
        table, _ = build_csv(data, cohort, extent)
        table.to_csv(out_dir / names["csv"], index=False)
        png, counts = render(data, out_dir / names["png"], cohort=cohort, extent=extent)
        (out_dir / names["man"]).write_text(
            _manifest(
                data, counts, cohort=cohort, extent=extent,
                png_name=names["png"], csv_name=names["csv"],
            )
        )
        written.append(png)
    return written
