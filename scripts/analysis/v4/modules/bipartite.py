"""Targets and vantage points on one grid, at every rung.

Both sides are quantized by **one call, with one nside**, which is the property
worth engineering for: if the two were binned separately, a drift in either
would silently change what "the VP and the target are in the same cell" means,
and nothing downstream would notice. `co_quantize` takes both point sets
together and there is no code path that bins one without the other.

## What the curve says

For each rung, how many distinct cells the targets occupy, how many the VPs
occupy, and how much they overlap. Read across the ladder it answers a question
a single resolution cannot: **at what scale does this dataset's geometry
actually separate?**

A target set whose cell count collapses from 128 to 16 was never 18 distinct
places — it was a handful of metros that a fine grid split. A VP set that
occupies far fewer cells than the targets cannot resolve them no matter what the
latency model does, because the constraints all come from the same few
directions. That is a property of the *dataset*, measurable before any method
runs, and it bounds what any method can achieve.

`dispersion` is the effective number of occupied cells,
`exp(H)` on the cell-occupancy distribution — a count that discounts cells
holding one point. Reported beside the raw count because they diverge exactly
when occupancy is lopsided, which is when the raw count most overstates the
geometry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v4.modules import healpix as H
from scripts.analysis.v4.modules.answer_space import load_targets
from scripts.analysis.v4.modules.paths import MissingArtifactError, RunPaths

TG_CELLS_CSV = "target_cells.csv"
VP_CELLS_CSV = "vp_cells.csv"
OCCUPANCY_CSV = "cell_occupancy.csv"
META_JSON = "meta.json"

#: Merged across rungs, one level above the rung directories.
BY_RESOLUTION_CSV = "occupancy_by_resolution.healpix.csv"


def load_vps(run: RunPaths) -> pd.DataFrame:
    """`vps.csv` — the run's vantage points, shared across every fold."""
    path = run.setup_dir / "vps.csv"
    if not path.exists():
        raise MissingArtifactError(f"{path} missing; cannot quantize the VP side")
    vps = pd.read_csv(path)
    need = {"vp_id", "vp_lat", "vp_lon"}
    missing = need - set(vps.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    return vps


def dispersion(counts) -> float:
    """Effective number of occupied cells: `exp(-sum p log p)`.

    Equals the raw count when occupancy is uniform and falls toward 1 as it
    concentrates, so the gap between the two is a direct read on how lopsided
    the geometry is. One point in each of 18 cells gives 18; 17 points in one
    cell and 1 in another gives ~1.24.
    """
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


@dataclass(frozen=True)
class CoQuantized:
    """Targets and VPs binned at one nside, plus the occupancy summary."""

    nside: int
    targets: pd.DataFrame
    vps: pd.DataFrame
    occupancy: pd.DataFrame
    meta: dict

    def write(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.targets.to_csv(out_dir / TG_CELLS_CSV, index=False)
        self.vps.to_csv(out_dir / VP_CELLS_CSV, index=False)
        self.occupancy.to_csv(out_dir / OCCUPANCY_CSV, index=False)
        (out_dir / META_JSON).write_text(json.dumps(self.meta, indent=2) + "\n")
        return out_dir


def co_quantize(
    targets: pd.DataFrame, vps: pd.DataFrame, nside: int
) -> CoQuantized:
    """Bin both sides at one nside. The only way either side gets a cell.

    Taking both frames in one call is the whole point: a separate
    `quantize_targets` / `quantize_vps` pair would let the two be called with
    different nsides, and "same cell" would quietly stop meaning anything.
    """
    nside = H.validate_nside(nside)

    tg = targets[["target_id", "target_lat", "target_lon"]].copy()
    vp = vps[["vp_id", "vp_lat", "vp_lon"]].copy()
    tg["cell_id"] = H.ang2pix(tg["target_lat"], tg["target_lon"], nside)
    vp["cell_id"] = H.ang2pix(vp["vp_lat"], vp["vp_lon"], nside)

    tg_counts = tg["cell_id"].value_counts()
    vp_counts = vp["cell_id"].value_counts()
    tg_cells = set(int(c) for c in tg_counts.index)
    vp_cells = set(int(c) for c in vp_counts.index)
    shared = tg_cells & vp_cells

    occupancy = (
        pd.DataFrame(
            {
                "cell_id": sorted(tg_cells | vp_cells),
            }
        )
        .assign(
            n_targets=lambda d: d["cell_id"].map(tg_counts).fillna(0).astype(int),
            n_vps=lambda d: d["cell_id"].map(vp_counts).fillna(0).astype(int),
        )
    )
    centres = H.pix2ang(occupancy["cell_id"].to_numpy(), nside)
    occupancy["cell_lat"] = centres[:, 0]
    occupancy["cell_lon"] = centres[:, 1]
    occupancy["shared"] = occupancy["cell_id"].isin(shared)

    meta = {
        "grid": H.describe(nside),
        "n_targets": int(len(tg)),
        "n_vps": int(len(vp)),
        "n_target_cells": len(tg_cells),
        "n_vp_cells": len(vp_cells),
        "n_shared_cells": len(shared),
        # Overlap normalised by the target side, because the question is how
        # much of the answer space a VP actually sits inside.
        "share_of_target_cells_with_a_vp": (
            round(len(shared) / len(tg_cells), 4) if tg_cells else None
        ),
        "target_dispersion": round(dispersion(tg_counts.to_numpy()), 3),
        "vp_dispersion": round(dispersion(vp_counts.to_numpy()), 3),
        "targets_per_occupied_cell": {
            "min": int(tg_counts.min()) if len(tg_counts) else 0,
            "max": int(tg_counts.max()) if len(tg_counts) else 0,
            "mean": round(float(tg_counts.mean()), 3) if len(tg_counts) else 0.0,
        },
        "vps_per_occupied_cell": {
            "min": int(vp_counts.min()) if len(vp_counts) else 0,
            "max": int(vp_counts.max()) if len(vp_counts) else 0,
            "mean": round(float(vp_counts.mean()), 3) if len(vp_counts) else 0.0,
        },
    }
    return CoQuantized(
        nside=nside, targets=tg, vps=vp, occupancy=occupancy, meta=meta
    )


def occupancy_row(q: CoQuantized) -> dict:
    """One row of `occupancy_by_resolution.healpix.csv` — the geometry curve."""
    m = q.meta
    return {
        "grid": "healpix",
        "nside": q.nside,
        "cell_km": round(H.nominal_cell_km(q.nside), 1),
        "n_targets": m["n_targets"],
        "n_vps": m["n_vps"],
        "n_target_cells": m["n_target_cells"],
        "n_vp_cells": m["n_vp_cells"],
        "n_shared_cells": m["n_shared_cells"],
        "share_of_target_cells_with_a_vp": m["share_of_target_cells_with_a_vp"],
        "target_dispersion": m["target_dispersion"],
        "vp_dispersion": m["vp_dispersion"],
    }


def build_for_run(
    run: RunPaths,
    *,
    nsides: tuple[int, ...] = H.NSIDE_LADDER,
    analysis_root: Path | None = None,
) -> list[CoQuantized]:
    """Co-quantize both sides at every rung, write artifacts and the curve."""
    targets = load_targets(run)
    vps = load_vps(run)
    out: list[CoQuantized] = []
    for nside in sorted({H.validate_nside(n) for n in nsides}, reverse=True):
        q = co_quantize(targets, vps, nside)
        q.write(run.bipartite_dir(nside, root=analysis_root))
        out.append(q)
    if len(out) > 1:
        pd.DataFrame([occupancy_row(q) for q in out]).to_csv(
            run.analysis_dir("bipartite-graph", root=analysis_root)
            / BY_RESOLUTION_CSV,
            index=False,
        )
    return out
