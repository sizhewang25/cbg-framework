"""min-RTT against the propagation delay of a two-leg VP → PNI → target path.

A quick experimental scatter, not a paper figure. The question it answers is
whether routing a VP's path through a named intermediate point (a PNI, an IXP,
a first-hop router — anything with a coordinate) explains the measured RTT
better than the straight VP → target line does. The x axis is therefore the
theoretical minimum RTT of the *bent* path,

    x = THEORETICAL_SLOPE · (d(VP, PNI) + d(PNI, TG))

with `THEORETICAL_SLOPE = 0.01 ms/km` — the round-trip delay at 2/3 c, the same
constant the CBG models are anchored on (`scripts/libs/cbg/rtt_model.py`). The
y axis is the observed min-RTT of the (VP, TG) pair.

Reading the figure: the `y = x` line is the speed-of-internet floor, so points
below it are physically impossible *for that path* — either the RTT is wrong or
the traffic did not go through this PNI. Points on it are paths the bent
geometry explains exactly; the vertical gap above it is the inflation left
unexplained. A tight cloud parallel to `y = x` means the PNI detour accounts
for the bulk of the delay, which is the correlation this script is meant to
show or fail to show.

Two fits are printed and drawn, both on the same axes:

* **OLS** on (x, y) — the average relationship, what a correlation claim rests
  on. Its slope is reported in ms/ms, so 1.0 means the detour is fully priced
  in and >1 means RTT grows faster than the bent path length.
* **Straight-line control** — the same OLS against
  `THEORETICAL_SLOPE · d(VP, TG)`, ignoring the PNI. Printed alongside so the
  PNI's contribution is a comparison rather than an assertion; if r² does not
  improve, the intermediate point is not buying anything.

Input is one canonical CSV, one row per (VP, PNI, TG) triple:

    vp_id,vp_lat,vp_lon,pni_id,pni_lat,pni_lon,tg_id,tg_lat,tg_lon,min_rtt

Column prefixes and the RTT column are options, so a CSV that calls its
intermediate `ixp_*` needs `--pni-prefix ixp` rather than a rename. Rows with a
missing coordinate or a non-positive RTT are dropped and counted, since a
0 ms or -1 ms RTT is a measurement sentinel, not a fast path.

    python -m scripts.analysis.v3.cli plot-pni-delay --csv pni_pairs.csv

Writes `<csv stem>_pni_delay.png` and `<csv stem>_pni_delay_points.csv` (the
input plus the computed distance/delay/residual columns) next to the input,
unless `--out-dir` says otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules.diagram.common.draw import plt
from scripts.analysis.v3.modules.diagram.common.palette import (
    _C_AXIS,
    _C_GRID,
    _C_INK,
    _C_INK_2,
    _C_MUTED,
)
from scripts.libs.cbg.rtt_model import THEORETICAL_SLOPE, haversine_distance

#: Marker colour for the scatter. One series, so this is a plain accent rather
#: than a draw from the categorical palette (which is keyed by method id).
_C_POINT = "#3b6ea5"
_C_FIT = "#b5482a"


def _haversine_km(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> np.ndarray:
    """Vectorised great-circle distance, in km.

    `haversine_distance` is already numpy-based, so it vectorises unchanged
    over the columns; calling it keeps this module on the same earth radius and
    formula as the production RTT model rather than a second copy.
    """
    return haversine_distance(
        lat1.to_numpy(dtype=float),
        lon1.to_numpy(dtype=float),
        lat2.to_numpy(dtype=float),
        lon2.to_numpy(dtype=float),
    )


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """`(slope, intercept, pearson_r, r2)` for `y ~ x`.

    Returns NaNs rather than raising when x has no spread (a single distinct
    value, or fewer than two rows), so a degenerate input still produces a
    figure and a report that says so.
    """
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return (float("nan"),) * 4
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return float(slope), float(intercept), r, r * r


def _truthy(col: pd.Series) -> pd.Series:
    """A boolean column, whether pandas inferred it as one or left it a string.

    `to_csv` writes `True`/`False`, and a round trip infers `bool` only when the
    column has no missing values; one NaN makes it `object` and the naive
    `df[col]` mask then selects the string "False" as truthy.
    """
    if col.dtype == bool:
        return col
    return col.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "t"})


def load_points(
    csv: Path,
    *,
    vp_prefix: str = "vp",
    pni_prefix: str = "pni",
    tg_prefix: str = "tg",
    rtt_col: str = "min_rtt",
    where: str | None = None,
) -> tuple[pd.DataFrame, int]:
    """Read the canonical CSV and add the distance/delay columns.

    Returns the usable rows and the number dropped, so the caller can label the
    figure with what it is not showing. `where` names a boolean column to
    restrict to first; rows it excludes are *not* counted as dropped, since they
    are a deliberate subset rather than unusable data.
    """
    df = pd.read_csv(csv)

    needed = [rtt_col] + [
        f"{p}_{axis}" for p in (vp_prefix, pni_prefix, tg_prefix) for axis in ("lat", "lon")
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise typer.BadParameter(
            f"{csv} is missing {missing}. Expected the canonical layout "
            f"<prefix>_lat/<prefix>_lon per role plus {rtt_col!r}; use "
            f"--vp-prefix/--pni-prefix/--tg-prefix/--rtt-col if the names differ. "
            f"Found: {list(df.columns)}"
        )

    if where is not None:
        if where not in df.columns:
            raise typer.BadParameter(
                f"--where {where!r} is not a column of {csv}. Found: {list(df.columns)}"
            )
        df = df[_truthy(df[where])]
        if df.empty:
            raise typer.BadParameter(f"--where {where!r} selected no rows of {csv}")

    n_in = len(df)
    df = df.dropna(subset=needed)
    df = df[pd.to_numeric(df[rtt_col], errors="coerce") > 0].copy()

    df["d_vp_pni_km"] = _haversine_km(
        df[f"{vp_prefix}_lat"], df[f"{vp_prefix}_lon"],
        df[f"{pni_prefix}_lat"], df[f"{pni_prefix}_lon"],
    )
    df["d_pni_tg_km"] = _haversine_km(
        df[f"{pni_prefix}_lat"], df[f"{pni_prefix}_lon"],
        df[f"{tg_prefix}_lat"], df[f"{tg_prefix}_lon"],
    )
    df["d_vp_tg_km"] = _haversine_km(
        df[f"{vp_prefix}_lat"], df[f"{vp_prefix}_lon"],
        df[f"{tg_prefix}_lat"], df[f"{tg_prefix}_lon"],
    )
    df["d_via_pni_km"] = df["d_vp_pni_km"] + df["d_pni_tg_km"]

    # Round-trip propagation delay at 2/3 c for each path length.
    df["prop_rtt_via_pni_ms"] = df["d_via_pni_km"] * THEORETICAL_SLOPE
    df["prop_rtt_direct_ms"] = df["d_vp_tg_km"] * THEORETICAL_SLOPE
    df["min_rtt_ms"] = df[rtt_col].astype(float)
    # Positive = RTT above the bent path's floor; negative = below it, i.e. this
    # PNI cannot be on the path.
    df["residual_ms"] = df["min_rtt_ms"] - df["prop_rtt_via_pni_ms"]

    return df.reset_index(drop=True), n_in - len(df)


def plot(
    df: pd.DataFrame,
    out_png: Path,
    *,
    n_dropped: int = 0,
    log_axes: bool = False,
    title: str | None = None,
) -> dict[str, float]:
    """Draw the scatter and return the fit statistics it reports."""
    x = df["prop_rtt_via_pni_ms"].to_numpy()
    y = df["min_rtt_ms"].to_numpy()
    xd = df["prop_rtt_direct_ms"].to_numpy()

    slope, intercept, r, r2 = _ols(x, y)
    d_slope, d_intercept, d_r, d_r2 = _ols(xd, y)

    fig, ax = plt.subplots(figsize=(6.0, 5.4), dpi=200)
    ax.scatter(x, y, s=14, alpha=0.45, color=_C_POINT, edgecolors="none", zorder=3)

    lo = float(min(x.min(), y.min())) if len(df) else 0.0
    hi = float(max(x.max(), y.max())) if len(df) else 1.0
    span = np.array([max(lo, 1e-3) if log_axes else 0.0, hi * 1.05])
    ax.plot(span, span, ls="--", lw=1.1, color=_C_INK_2, zorder=2,
            label="y = x  (2/3 c floor of the VP→PNI→TG path)")
    if np.isfinite(slope):
        ax.plot(span, slope * span + intercept, lw=1.4, color=_C_FIT, zorder=4,
                label=f"OLS: y = {slope:.2f}x + {intercept:.2f} ms  (r={r:.3f}, r²={r2:.3f})")

    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("propagation delay of d(VP,PNI) + d(PNI,TG) at 2/3 c  [ms]")
    ax.set_ylabel("observed min-RTT (VP → TG)  [ms]")
    ax.set_title(title or "min-RTT vs two-leg propagation delay", color=_C_INK)
    ax.grid(True, color=_C_GRID, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_C_AXIS)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)

    below = int((df["residual_ms"] < 0).sum())
    note = (
        f"n = {len(df)} triples"
        + (f"  ({n_dropped} dropped: missing coords or rtt ≤ 0)" if n_dropped else "")
        + f"\n{below} below the floor ({below / max(len(df), 1):.1%}) — PNI not on that path"
        + f"\nstraight-line control (ignoring PNI): r={d_r:.3f}, r²={d_r2:.3f}, slope={d_slope:.2f}"
    )
    ax.annotate(note, xy=(0.98, 0.02), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=7, color=_C_MUTED)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    return {
        "n": float(len(df)),
        "n_dropped": float(n_dropped),
        "n_below_floor": float(below),
        "via_pni_slope": slope,
        "via_pni_intercept_ms": intercept,
        "via_pni_pearson_r": r,
        "via_pni_r2": r2,
        "direct_slope": d_slope,
        "direct_intercept_ms": d_intercept,
        "direct_pearson_r": d_r,
        "direct_r2": d_r2,
        "median_residual_ms": float(np.median(df["residual_ms"])) if len(df) else float("nan"),
    }


def register(app: typer.Typer) -> None:
    @app.command("plot-pni-delay")
    def plot_pni_delay_cmd(
        csv: Path = typer.Option(
            ..., "--csv", exists=True, dir_okay=False,
            help="Canonical CSV, one row per (VP, PNI, TG) triple: "
                 "vp_lat/vp_lon, pni_lat/pni_lon, tg_lat/tg_lon, min_rtt.",
        ),
        rtt_col: str = typer.Option("min_rtt", "--rtt-col", help="min-RTT column, in ms."),
        vp_prefix: str = typer.Option("vp", "--vp-prefix", help="Column prefix for the VP."),
        pni_prefix: str = typer.Option("pni", "--pni-prefix", help="Column prefix for the PNI."),
        tg_prefix: str = typer.Option("tg", "--tg-prefix", help="Column prefix for the target."),
        where: str = typer.Option(
            None,
            "--where",
            help="Restrict to rows where this boolean column is true, e.g. "
            "`--where is_sping_vp` for one point per target on the VP the "
            "shortest-ping baseline picked.",
        ),
        log_axes: bool = typer.Option(
            False, "--log-axes", help="Log-log axes, for RTTs spanning orders of magnitude."
        ),
        title: str = typer.Option(None, "--title", help="Override the figure title."),
        out_dir: Path = typer.Option(
            None, "--out-dir", help="Where to write. Default: alongside the input CSV."
        ),
    ) -> None:
        """Scatter min-RTT against the VP→PNI→TG propagation delay.

        Writes <stem>_pni_delay.png and <stem>_pni_delay_points.csv, and prints
        the OLS fit plus a straight-line (no-PNI) control for comparison.
        """
        df, n_dropped = load_points(
            csv, vp_prefix=vp_prefix, pni_prefix=pni_prefix,
            tg_prefix=tg_prefix, rtt_col=rtt_col, where=where,
        )
        if df.empty:
            raise typer.BadParameter(
                f"{csv} has no usable rows ({n_dropped} dropped for missing "
                f"coordinates or {rtt_col} <= 0)."
            )

        dest = out_dir or csv.parent
        dest.mkdir(parents=True, exist_ok=True)
        suffix = f"_pni_delay{'.' + where if where else ''}"
        out_png = dest / f"{csv.stem}{suffix}.png"
        out_csv = dest / f"{csv.stem}{suffix}_points.csv"

        stats = plot(df, out_png, n_dropped=n_dropped, log_axes=log_axes, title=title)
        df.to_csv(out_csv, index=False)

        typer.echo(f"wrote {out_png}")
        typer.echo(f"wrote {out_csv}")
        for key, value in stats.items():
            typer.echo(f"  {key:24s} {value:.4g}")
