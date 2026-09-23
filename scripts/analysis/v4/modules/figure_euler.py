"""The cross-dataset Euler diagram: which methods place the *same* targets.

An accuracy table says Octant-Hull places 24% of as02's targets in the right
cell and the baseline 18%. It cannot say whether those are the same targets.
This figure is that question, and only that question: circle area is a
method's in-cell share, shared area is the share **both** place, a method whose
set sits inside another's is drawn inside it, and two that never agree are
drawn apart.

## One resolution, three tolerances

Pooled across datasets there is one grid (`--nside`, default **128** — 50.9 km
cells) and a `top_n` that names a rung of the containment ladder:

| `top_n` | correct means | printed as |
|---|---|---|
| 1 | `ring == 0` — the truth's own cell | in the cell |
| 2 | `ring <= 1` — that cell or its 8 neighbours | within 1 ring |
| 3 | `ring <= 2` — out to the second ring | within 2 rings |

That is the whole change from v3, whose `top_n` ranked *class seeds* by
distance to the prediction. A Voronoi partition over K seeds labels every point
on Earth, so v3's rank-1 set contained a prediction 2,360 km from its truth.
Here `top_n` is a **containment** tolerance and is local by construction: no
arrangement of far-away cells can make a distant cell adjacent.

The resolution sweep is deliberately **not** repeated here. The rung is the
tolerance dial for a bar chart that has room for four panels; an Euler diagram
is one fitted layout per population, so sweeping nside would write four figures
whose circles cannot be compared by eye anyway. `accuracy_by_resolution.csv`
and `plot-outcome-bars` carry the ladder; this figure picks one rung and varies
the ring instead — which is the axis that changes *which targets are in which
set*, and therefore the only one that changes the overlaps.

## Empty sets have no circle

`spotter_cbg` places **zero** targets in the truth's own cell on all three
meshes, so at `top_n=1` its set is empty. It gets no circle: a zero-radius one
is a dot a reader takes for "very small" rather than "never".

Dropping it is lossless, not a convenience. A set nothing belongs to appears in
no region of the diagram — every intersection involving it is empty and every
other region's count is unchanged — so the figure drawn without it is the same
figure. The method stays in the **denominator** (it is one of the methods the
population was scored over), stays in `intersections.csv` and `pairwise.csv`
with its zeroes, and is named in the figure's footnote and the manifest, so its
absence cannot read as an arbitrary exclusion. Only if fewer than two sets
survive is the figure skipped outright.

## Pooling

Runs are **stacked**: their target sets are disjoint, so a target belongs to
exactly one run and pooling is a concat, re-keyed `<run_id>::<target_id>`.
Every share is therefore a micro-average — a target counts the same whichever
dataset it came from — and, like the pooled outcome bars, the result is each
method's agreement on *this* target mix rather than in general. Coverage is
strict: a method absent from any input run is refused rather than pooled over
the runs that carry it.

## What lands on disk

Under `_cross/cls-accuracy/<dataset-set>/`, one set per `top_n`:

    euler.healpix-128.top1.png              the figure
    euler.healpix-128.top1.fit.csv          observed vs drawn, per region
    euler.healpix-128.top1.intersections.csv  exact counts, per combination
    euler.healpix-128.top1.pairwise.csv     both / a-only / b-only / neither
    euler.healpix-128.top1.membership.csv   the boolean matrix itself
    euler.healpix-128.top1.manifest.json

The slug carries the grid because this directory is keyed by dataset set alone,
and `.top<N>` because a figure is a function of the tolerance — without it a
top-3 pass would silently overwrite the top-1 files.

`fit.csv` is load-bearing rather than an extra. Circles are overdetermined past
two sets, so some combination is always misdrawn; that table is the row-by-row
audit of which, and the figure prints its own `placed` share and worst pair
error pointing at it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.analysis.v4.modules import cross
from scripts.analysis.v4.modules import healpix as H
from scripts.analysis.v4.modules.euler import layout as L
from scripts.analysis.v4.modules.euler import membership as M
from scripts.analysis.v4.modules.euler import tables as T
from scripts.analysis.v4.modules.euler.plot import plot_euler
from scripts.analysis.v4.modules.methods import method_label
from scripts.analysis.v4.modules.paths import (
    RunPaths,
    grid_slug,
)

#: The rung this figure is drawn at unless told otherwise. 50.9 km cells — fine
#: enough that "the right cell" is a real claim about a metro area, coarse
#: enough that every method has a non-empty set by one ring out.
DEFAULT_NSIDE = 128

#: Every rung of the tolerance ladder, written by default. They read as a
#: sequence — each set can only grow — and they share one membership build per
#: run, so the second and third cost only their fits.
DEFAULT_TOP_NS: tuple[int, ...] = (1, 2, 3)

#: Where cross-dataset figures land. The same directory `plot-outcome-bars`
#: and `plot-error-cdf` write to, keyed by the dataset set, so one comparison's
#: artifacts sit together regardless of which figure produced them — which is
#: why the rule lives in `cross` and is re-exported here.
CROSS_KIND = cross.CROSS_KIND

#: `{slug}` is `healpix-<nside>`, `{n}` the top-N.
STEM = "euler.{slug}.top{n}"
ARTIFACTS = {
    "png": "{stem}.png",
    "fit": "{stem}.fit.csv",
    "intersections": "{stem}.intersections.csv",
    "pairwise": "{stem}.pairwise.csv",
    "membership": "{stem}.membership.csv",
    "manifest": "{stem}.manifest.json",
}


dataset_slug = cross.dataset_slug
cross_dir = cross.cross_dir
short_dataset = cross.short_dataset


def drop_empty_sets(membership: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split the columns into the ones with a circle and the ones without.

    A set no target belongs to appears in **no region** of the diagram: every
    intersection involving it is empty, and removing it leaves every other
    region's count exactly as it was. So the drawn figure is the same figure,
    minus a circle that could not have been drawn — `fit_euler_layout` refuses
    a zero radius, and a dot at the origin would read as "very small" rather
    than "never".

    The dropped names are returned rather than discarded: they go on the
    figure's footnote and into the manifest, because an unexplained missing
    circle reads as an arbitrary exclusion. The method keeps its place in the
    denominator and in both count tables.
    """
    order = list(membership.columns)
    live = [m for m in order if bool(membership[m].any())]
    return live, [m for m in order if m not in live]


def tolerance_text(top_n: int) -> str:
    """How this rung reads in a title: `1` -> `"in the cell"`."""
    return M.TOLERANCE_LABELS[M.validate_top_n(top_n)]


def _manifest(
    *,
    run_ids: list[str],
    membership: pd.DataFrame,
    origin: pd.Series,
    nside: int,
    top_n: int,
    order: list[str],
    empty: list[str],
    fit: L.EulerLayout | None,
    png: str | None,
) -> str:
    per_run = origin.value_counts().sort_index()
    methods = list(membership.columns)
    any_correct = membership.any(axis=1)
    body = {
        "runs": sorted(run_ids),
        "arm": cross.arm(run_ids),
        "grid": H.describe(nside),
        "top_n": int(top_n),
        "correctness": (
            f"top_n={top_n} means ring <= {top_n - 1}: the prediction is "
            f"{tolerance_text(top_n)}. Cumulative, so the set at top_n={top_n} "
            f"contains the set at every smaller one."
        ),
        "n_targets": int(len(membership)),
        "n_targets_per_run": {r: int(c) for r, c in per_run.items()},
        "methods": methods,
        "labels": {m: method_label(m) for m in methods},
        "n_correct_per_method": {m: int(membership[m].sum()) for m in methods},
        "n_targets_none_correct": int((~any_correct).sum()),
        "share_any_correct": round(float(any_correct.mean()), 4),
        "pooling": {
            "rule": (
                "micro-average: the runs' targets are concatenated and re-keyed "
                "<run_id>::<target_id>, so a target counts the same whichever "
                "dataset it came from"
            ),
            "largest_share": (
                round(int(per_run.max()) / len(membership), 4) if len(membership) else None
            ),
            "reading_caveat": (
                "these are the methods' agreement on THIS target mix, not in "
                "general — the pooled shares are dominated by whichever input "
                "run is largest"
            ),
            "coverage": (
                "strict — a method absent from any input run is refused rather "
                "than pooled over the runs that carry it"
            ),
        },
        "drawn": fit is not None,
        "circles": order,
        "region_letters": (
            {v: method_label(k) for k, v in T.letter_map(order).items()}
            if order else {}
        ),
        # Named rather than left as a gap: these methods are in the population
        # and in both count tables, they simply have no circle.
        "empty_sets": empty or None,
        "empty_sets_note": (
            "a set no target belongs to appears in no region, so dropping its "
            "circle changes nothing else on the figure; it stays in the "
            "denominator and in intersections.csv / pairwise.csv"
        ),
        "figure": png,
        # The figure's own honesty numbers. Circles are overdetermined past two
        # sets, so some share of targets is always drawn in the wrong region.
        "euler_placed_share": None if fit is None else round(fit.placed, 4),
        "euler_max_pair_error": None if fit is None else round(fit.pair_error(), 4),
        "encoding": (
            "area-proportional: circle area IS the set share and shared area IS "
            "the intersection share, both to the same scale as the dashed "
            "'None' circle. No region carries a printed number — the exact "
            "counts are in intersections.csv."
        ),
    }
    return json.dumps(body, indent=2) + "\n"


def build_one(
    runs: list[RunPaths],
    *,
    nside: int = DEFAULT_NSIDE,
    top_n: int = 1,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
    restarts: int = L.EULER_RESTARTS,
) -> dict[str, Path]:
    """One tolerance's artifact set. Returns the written paths by kind.

    The PNG is absent from the result when fewer than two sets survive
    `drop_empty_sets` — there is no overlap left to draw — and the manifest
    records `drawn: false` with the reason readable from `empty_sets`.
    """
    top_n = M.validate_top_n(top_n)
    nside = H.validate_nside(nside)
    run_ids = [r.run_id for r in runs]
    out_dir = cross_dir(run_ids, analysis_root=analysis_root)

    membership, origin = M.pooled_membership(
        runs, nside, top_n=top_n, methods=methods, analysis_root=analysis_root
    )
    n = len(membership)
    stem = STEM.format(slug=grid_slug(nside), n=top_n)

    def path(kind: str) -> Path:
        return out_dir / ARTIFACTS[kind].format(stem=stem)

    written: dict[str, Path] = {}

    labelled = membership.rename(columns=method_label)
    labelled.insert(0, "run_id", origin)
    labelled.to_csv(path("membership"))
    written["membership"] = path("membership")

    T.intersection_table(membership).to_csv(path("intersections"), index=False)
    written["intersections"] = path("intersections")
    T.pairwise_table(membership).to_csv(path("pairwise"), index=False)
    written["pairwise"] = path("pairwise")

    order, empty = drop_empty_sets(membership)
    fit = None
    if len(order) >= 2:
        fit = L.fit_euler_layout(membership, order, restarts=restarts)
        L.euler_fit_table(fit, n).to_csv(path("fit"), index=False)
        written["fit"] = path("fit")

        names = " + ".join(short_dataset(r) for r in sorted(run_ids))
        written["png"] = plot_euler(
            fit,
            path("png"),
            title=(
                f"{names} — which targets the methods place {tolerance_text(top_n)} "
                f"({n} targets)"
            ),
            subtitle=(
                " · ".join(
                    f"{short_dataset(r)} {int(c)}"
                    for r, c in origin.value_counts().sort_index().items()
                )
                + f" · HEALPix nside {nside}, {H.nominal_cell_km(nside):.1f} km cells"
            ),
            fit_ref=ARTIFACTS["fit"].format(stem=stem),
            empty_sets=empty,
        )

    path("manifest").write_text(
        _manifest(
            run_ids=run_ids,
            membership=membership,
            origin=origin,
            nside=nside,
            top_n=top_n,
            order=order,
            empty=empty,
            fit=fit,
            png=path("png").name if fit is not None else None,
        )
    )
    written["manifest"] = path("manifest")
    return written


def build_for_runs(
    runs: list[RunPaths],
    *,
    nside: int = DEFAULT_NSIDE,
    top_ns: tuple[int, ...] = DEFAULT_TOP_NS,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
    restarts: int = L.EULER_RESTARTS,
) -> list[dict[str, Path]]:
    """One artifact set per tolerance, at a single resolution."""
    wanted = sorted({M.validate_top_n(n) for n in top_ns})
    return [
        build_one(
            runs,
            nside=nside,
            top_n=n,
            methods=methods,
            analysis_root=analysis_root,
            restarts=restarts,
        )
        for n in wanted
    ]
