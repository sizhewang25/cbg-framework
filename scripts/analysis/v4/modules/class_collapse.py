"""What the grid costs before any method is asked: sites the rung cannot separate.

Every accuracy figure in v4 asks "did the prediction land in the right class?".
This table asks the prior question — **how many of the operator's own sites are
still distinguishable at this rung?** — and the answer is not "all of them".

At nside-16 the 65 sites of the three mesh runs occupy 52 classes: 25 sites
share a cell with another, collapsing into 12 cells, one of which holds three.
Those sites are not separable **even in principle** at that rung, by any
estimator. So coarsening does not merely relax the tolerance a method is graded
against; it destroys class distinctions. An operator choosing a coarse
granularity is choosing to stop telling some of their own facilities apart —
fine for regional traffic monitoring, fatal for troubleshooting.

It is also why containment and proximity can disagree on a direct hit at coarse
rungs: when two real sites share a cell, landing in the right cell can still
leave the prediction nearest the wrong site's seed.

## Classes are counted within a run, and the total is a sum

`n_classes` sums each run's own class count rather than counting distinct
`cell_id`s over the concatenated frame, and the difference is large: the union
of cell ids across the three runs is 26 / 33 / 38 / 40 at nside 16 / 32 / 64 /
128, against the 52 / 60 / 63 / 63 this table reports. The runs genuinely share
cells, because they share facilities.

Summing is the correct operation for the same reason the site key carries the
run id (see `sites.py`): operators geolocate ASN by ASN, so a cell occupied in
two runs is two classes in two separate problems, not one class in a merged
one. Taking the union would silently answer a question nobody asked — "how many
cells would one merged operator have?" — and would change every row of this
table. It is spelled out here and in the manifest so that a later reader who
notices the collision does not "fix" it.

## The quantizer merges sites that no rung recovers

as01 holds 20 sites but never exceeds 18 classes, at any rung including the
finest. Two pairs sit close enough to share a cell all the way up the ladder, so
its ceiling is 90% separability rather than 100%. The ladder is a dial on the
cost, not a switch that removes it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.analysis.v4.modules import cross
from scripts.analysis.v4.modules import healpix as H
from scripts.analysis.v4.modules import sites as S
from scripts.analysis.v4.modules.answer_space import ASSIGNMENTS_CSV
from scripts.analysis.v4.modules.paths import MissingArtifactError, RunPaths

#: The artifact triple. No `{slug}` in the name: this table's whole subject is
#: the ladder, so one file carries every rung as rows rather than four files
#: each carrying one.
TABLE_CSV = "class_collapse.csv"
TABLE_MANIFEST = "class_collapse.manifest.json"

CSV_COLUMNS = (
    "nside",
    "cell_km",
    "n_targets",
    "n_sites",
    "n_classes",
    "n_sites_merged",
    "n_cells_holding_merges",
    "max_sites_per_cell",
    "share_sites_separable",
)


def load_assignments(
    run: RunPaths, nside: int, *, analysis_root: Path | None = None
) -> pd.DataFrame:
    """One run's target->cell assignment at one rung."""
    path = run.answer_space_dir(nside, root=analysis_root) / ASSIGNMENTS_CSV
    if not path.exists():
        raise MissingArtifactError(
            f"{path} missing; run `build-answer-space --run-id {run.run_id}` first"
        )
    return pd.read_csv(path)


def _rung_row(frames: dict[str, pd.DataFrame], nside: int) -> dict:
    """One row of the table: the ladder rung, summed over runs."""
    n_targets = n_sites = n_classes = n_merged = n_cells = 0
    max_per_cell = 0
    for frame in frames.values():
        distinct = frame.drop_duplicates(list(S.SITE_COLUMNS))
        per_cell = distinct.groupby("cell_id").size()
        n_targets += len(frame)
        n_sites += len(distinct)
        n_classes += int(per_cell.size)
        n_merged += int(per_cell[per_cell > 1].sum())
        n_cells += int((per_cell > 1).sum())
        max_per_cell = max(max_per_cell, int(per_cell.max()) if per_cell.size else 0)

    return {
        "nside": nside,
        "cell_km": round(H.nominal_cell_km(nside), 1),
        "n_targets": n_targets,
        "n_sites": n_sites,
        "n_classes": n_classes,
        "n_sites_merged": n_merged,
        "n_cells_holding_merges": n_cells,
        "max_sites_per_cell": max_per_cell,
        # Sites that keep a cell to themselves. The complement of the headline:
        # at nside-16 this is 0.615, so 38.5% of the operator's sites are no
        # longer separable even in principle.
        "share_sites_separable": round((n_sites - n_merged) / n_sites, 4)
        if n_sites
        else float("nan"),
    }


def _load_all(
    runs: list[RunPaths], rungs: tuple[int, ...], analysis_root: Path | None
) -> dict[int, dict[str, pd.DataFrame]]:
    """Every rung's assignments, loaded once and shared by table and manifest.

    The manifest reports measurements the table does not carry (the unioned
    class count, the per-run quantizer floor), and both must come from the same
    read -- a manifest that asserts a number the CSV was not built from is the
    failure this whole task exists to remove.
    """
    out: dict[int, dict[str, pd.DataFrame]] = {}
    for nside in rungs:
        frames = {
            r.run_id: load_assignments(r, nside, analysis_root=analysis_root)
            for r in runs
        }
        cross.guard_disjoint_targets(
            {k: set(v["target_id"]) for k, v in frames.items()},
            remedy="Pass one --run-id, which scores that dataset on its own.",
        )
        out[nside] = frames
    return out


def collapse_table(
    runs: list[RunPaths],
    *,
    nsides: tuple[int, ...] = H.NSIDE_LADDER,
    analysis_root: Path | None = None,
) -> pd.DataFrame:
    """The table, coarsest rung last -- the direction the collapse runs in."""
    rungs = tuple(sorted({H.validate_nside(n) for n in nsides}, reverse=True))
    loaded = _load_all(runs, rungs, analysis_root)
    return pd.DataFrame(
        [_rung_row(loaded[n], n) for n in rungs], columns=list(CSV_COLUMNS)
    )


def _unioned_classes(loaded: dict[int, dict[str, pd.DataFrame]]) -> dict[str, int]:
    """Distinct cell ids over the concatenated runs, per rung.

    Measured, not asserted: this is the number `class_counting` warns against
    using, so it is published beside the warning rather than quoted in it.
    """
    return {
        str(nside): int(pd.concat(frames.values())["cell_id"].nunique())
        for nside, frames in sorted(loaded.items(), reverse=True)
    }


def _quantizer_floor(loaded: dict[int, dict[str, pd.DataFrame]]) -> dict:
    """Per run at the finest rung: sites, classes, and the gap between them.

    A run whose class count never reaches its site count holds sites the grid
    merges at every rung of the ladder, so its separability is capped below 1.0
    by the quantizer rather than by any method.
    """
    finest = max(loaded)
    per_run = {}
    for run_id, frame in sorted(loaded[finest].items()):
        distinct = frame.drop_duplicates(list(S.SITE_COLUMNS))
        n_sites = int(len(distinct))
        n_classes = int(distinct["cell_id"].nunique())
        per_run[run_id] = {
            "n_sites": n_sites,
            "n_classes": n_classes,
            "sites_merged_at_every_rung": n_sites - n_classes,
            "separability_ceiling": round(n_classes / n_sites, 4) if n_sites else None,
        }
    capped = {k: v for k, v in per_run.items() if v["sites_merged_at_every_rung"]}
    return {
        "nside": finest,
        "by_run": per_run,
        "note": (
            "Measured at the finest rung built. A non-zero "
            "sites_merged_at_every_rung means the grid merges those sites at "
            "every rung of the ladder, so the ladder is a dial on the "
            "quantization cost rather than a switch that removes it."
            + (
                f" Capped here: {', '.join(sorted(capped))}."
                if capped
                else " No run is capped here: every site has its own class."
            )
        ),
    }


def _manifest(
    table: pd.DataFrame,
    loaded: dict[int, dict[str, pd.DataFrame]],
    *,
    run_ids: list[str],
) -> str:
    coarsest = int(table["nside"].min())
    row = table.loc[table["nside"] == coarsest].iloc[0]
    body = {
        "csv": TABLE_CSV,
        "runs": run_ids,
        "arm": cross.arm(run_ids),
        "rungs": [H.describe(int(n)) for n in table["nside"]],
        # Rung-free: a site is a coordinate, and the grid has no say in how
        # many there are. Taken once at the finest rung rather than per row.
        "sites": S.site_diagnostics(loaded[max(loaded)]),
        "n_classes_if_unioned": _unioned_classes(loaded),
        "class_counting": (
            "n_classes SUMS each run's own class count. The union over the "
            "concatenated frame is smaller -- see n_classes_if_unioned -- "
            "because the runs share facilities and therefore share cells. "
            "Summing is correct here for the same reason the site key carries "
            "the run id: operators geolocate ASN by ASN, so a cell occupied in "
            "two runs is two classes in two separate problems. Do not 'fix' "
            "this into a union; it would change every row."
        ),
        "merge_definition": (
            "n_sites_merged counts sites sharing a cell with at least one "
            "other site; n_cells_holding_merges counts the cells they collapse "
            f"into. At nside-{coarsest} that is {int(row['n_sites_merged'])} "
            f"sites -> {int(row['n_cells_holding_merges'])} cells, the largest "
            f"holding {int(row['max_sites_per_cell'])}."
        ),
        "reading": (
            f"At nside-{coarsest} ({row['cell_km']} km cells) "
            f"{100 * (1 - row['share_sites_separable']):.1f}% of the operator's "
            "own sites are not separable even in principle, by any estimator. "
            "Coarsening destroys class distinctions, not just tolerance."
        ),
        "quantizer_floor": _quantizer_floor(loaded),
        "source": (
            f"target-answer-space/healpix-<n>/{ASSIGNMENTS_CSV}, deduplicated "
            "on the site key. Derived from the answer space itself, so it "
            "describes the question every method was asked rather than any "
            "method's answer -- no scoring is read and no method appears."
        ),
    }
    return json.dumps(body, indent=2) + "\n"


def build_for_runs(
    runs: list[RunPaths],
    *,
    nsides: tuple[int, ...] = H.NSIDE_LADDER,
    analysis_root: Path | None = None,
) -> Path:
    """Write the CSV and its manifest into the cross-dataset directory."""
    rungs = tuple(sorted({H.validate_nside(n) for n in nsides}, reverse=True))
    loaded = _load_all(runs, rungs, analysis_root)
    table = pd.DataFrame(
        [_rung_row(loaded[n], n) for n in rungs], columns=list(CSV_COLUMNS)
    )
    run_ids = [r.run_id for r in runs]
    out_dir = cross.cross_dir(run_ids, analysis_root=analysis_root)
    table.to_csv(out_dir / TABLE_CSV, index=False)
    (out_dir / TABLE_MANIFEST).write_text(_manifest(table, loaded, run_ids=run_ids))
    return out_dir / TABLE_CSV
