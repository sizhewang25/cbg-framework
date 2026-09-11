"""min-RTT against the propagation delay of a two-leg VP → PNI → target path.

A quick experimental scatter, not a paper figure. The question it answers is
whether routing a VP's path through a named intermediate point (a PNI, an IXP,
a first-hop router — anything with a coordinate) explains the measured RTT
better than the straight VP → target line does. The x axis is therefore the
theoretical minimum RTT of the *bent* path,

    x = THEORETICAL_SLOPE · (d(VP, PNI) + d(PNI, TG))

with `THEORETICAL_SLOPE = 0.01 ms/km` — the **round-trip** delay at 2/3 c (the
factor 2 is already in the constant), the same constant the CBG models are
anchored on (`scripts/libs/cbg/rtt_model.py`). The y axis is the observed
min-RTT of the (VP, TG) pair.

`--x-unit` chooses whether x carries that path's **distance** (the default, in
km) or its delay. Distance is the axis a linearity claim is actually about, and
it keeps the fitted slope in ms/km so its reciprocal reads as an implied speed;
the 2/3 c floor is then the sloped line `y = 0.01 x` rather than `y = x`. The
delay axis is the same points rescaled, with the floor at 45°. Neither changes
the fit quality or which points fall below the floor.

`--x-axis` chooses *which* path: `via-pni` for d(VP,PNI) + d(PNI,TG), or
`direct` for d(VP,TG), which ignores the PNI and so is the control.

Reading the figure: the floor line is the speed-of-internet limit, so points
below it are physically impossible *for that path*. On the `direct` axis that is
a statement about the data — a point below it means the RTT or a coordinate is
wrong. On the `via-pni` axis it is a statement about the *assignment*: the pair
cannot have crossed the site it was given. Since `via >= direct` always, a
via-PNI violation means exactly `detour_ratio > air_inflation`. The vertical gap
above the floor is the inflation left unexplained; a tight cloud parallel to the
floor means the detour accounts for the bulk of the delay, which is the
correlation this script is meant to show or fail to show.

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

from scripts.analysis.v3.modules.bipartite import ols, truthy
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

    Delegates to `bipartite.ols`. Shared rather than local because
    `compare-pni-linearity` tabulates the same fit this figure draws, and the
    paper quotes one and shows the other — they cannot be allowed to drift.
    """
    return ols(x, y)


def _truthy(col: pd.Series) -> pd.Series:
    """A boolean column, whether pandas inferred it as one or left it a string.

    Delegates to `bipartite.truthy`; kept as a name here because `--where` is
    this module's flag and its tests pin the behaviour. Three modules now read
    `is_sping_vp` back off a CSV, so the definition moved to a matplotlib-free
    module rather than being imported out of a figure.
    """
    return truthy(col)


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


#: The two path definitions the x axis can carry. Each names the km column, the
#: 2/3 c-delay column, the path in words, and how the distance reads in a label.
#: Both fits are always computed and reported under fixed `via_pni_*` /
#: `direct_*` keys, so switching the axis changes what is *drawn*, never what a
#: stat is called.
X_AXES = {
    "via-pni": ("d_via_pni_km", "prop_rtt_via_pni_ms", "VP→PNI→TG",
                "d(VP,PNI) + d(PNI,TG)"),
    "direct": ("d_vp_tg_km", "prop_rtt_direct_ms", "direct VP→TG", "d(VP,TG)"),
}

#: `km` puts the physical quantity on x, which is what a linearity claim is
#: about; the 2/3 c floor is then the sloped line `y = THEORETICAL_SLOPE * x`
#: rather than `y = x`, and the fitted slope is in ms/km so its reciprocal is an
#: implied speed. `ms` converts x at 2/3 c first, which makes the floor `y = x`
#: and the slope dimensionless. Same points, same below-floor set, same fit
#: quality -- only the units and the reference line's angle move.
X_UNITS = ("km", "ms")


def plot(
    df: pd.DataFrame,
    out_png: Path,
    *,
    n_dropped: int = 0,
    log_axes: bool = False,
    title: str | None = None,
    x_axis: str = "via-pni",
    x_unit: str = "km",
) -> dict[str, float]:
    """Draw the scatter and return the fit statistics it reports.

    `x_axis` selects the path that goes on x. `direct` ignores the PNI entirely,
    which makes the same figure the control: the great-circle line every CBG
    variant's latency-to-distance model assumes. `x_unit` selects whether x
    carries that path's distance or its 2/3 c round-trip delay.
    """
    if x_axis not in X_AXES:
        raise typer.BadParameter(
            f"--x-axis must be one of {sorted(X_AXES)} (got {x_axis!r})"
        )
    if x_unit not in X_UNITS:
        raise typer.BadParameter(
            f"--x-unit must be one of {list(X_UNITS)} (got {x_unit!r})"
        )
    km_col, ms_col, floor_name, dist_label = X_AXES[x_axis]
    other = "direct" if x_axis == "via-pni" else "via-pni"
    o_km, o_ms, other_name, _ = X_AXES[other]

    if x_unit == "km":
        x_col, other_col = km_col, o_km
        # The floor is the 2/3 c round trip over that distance.
        floor_slope = THEORETICAL_SLOPE
        x_label = f"{dist_label}  [km]"
        floor_label = f"y = {THEORETICAL_SLOPE} x   (2/3 c round trip over the {floor_name} path)"
    else:
        x_col, other_col = ms_col, o_ms
        floor_slope = 1.0
        x_label = f"propagation delay of {dist_label} at 2/3 c  [ms]"
        floor_label = f"y = x  (2/3 c floor of the {floor_name} path)"

    x = df[x_col].to_numpy()
    y = df["min_rtt_ms"].to_numpy()
    xd = df[other_col].to_numpy()

    slope, intercept, r, r2 = _ols(x, y)
    d_slope, d_intercept, d_r, d_r2 = _ols(xd, y)

    fig, ax = plt.subplots(figsize=(6.0, 5.4), dpi=200)
    ax.scatter(x, y, s=14, alpha=0.45, color=_C_POINT, edgecolors="none", zorder=3)

    lo = float(min(x.min(), y.min() / max(floor_slope, 1e-12))) if len(df) else 0.0
    hi = float(max(x.max(), y.max() / max(floor_slope, 1e-12))) if len(df) else 1.0
    span = np.array([max(lo, 1e-3) if log_axes else 0.0, hi * 1.05])
    ax.plot(span, floor_slope * span, ls="--", lw=1.1, color=_C_INK_2, zorder=2,
            label=floor_label)
    if np.isfinite(slope):
        unit = " ms/km" if x_unit == "km" else ""
        fmt = f"{slope:.4f}" if x_unit == "km" else f"{slope:.2f}"
        ax.plot(span, slope * span + intercept, lw=1.4, color=_C_FIT, zorder=4,
                label=f"OLS: y = {fmt}{unit} · x + {intercept:.2f} ms"
                      f"  (r={r:.3f}, r²={r2:.3f})")

    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(x_label)
    ax.set_ylabel("observed min-RTT (VP → TG)  [ms]")
    ax.set_title(title or "min-RTT vs two-leg propagation delay", color=_C_INK)
    ax.grid(True, color=_C_GRID, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_C_AXIS)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)

    below = int((y < floor_slope * x).sum())
    below_note = (
        "PNI not on that path" if x_axis == "via-pni" else "RTT beats 2/3 c on the geodesic"
    )
    note = (
        f"n = {len(df)} triples"
        + (f"  ({n_dropped} dropped: missing coords or rtt ≤ 0)" if n_dropped else "")
        + f"\n{below} below the floor ({below / max(len(df), 1):.1%}) — {below_note}"
        + f"\ncontrol ({other_name}): r={d_r:.3f}, r²={d_r2:.3f}"
        + (f"\nimplied speed {2 / slope:,.0f} km/ms vs 2/3 c = 200 km/ms"
           if x_unit == "km" and np.isfinite(slope) and slope > 0 else "")
    )
    ax.annotate(note, xy=(0.98, 0.02), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=7, color=_C_MUTED)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    primary = {
        "slope": slope, "intercept_ms": intercept, "pearson_r": r, "r2": r2,
    }
    control = {
        "slope": d_slope, "intercept_ms": d_intercept, "pearson_r": d_r, "r2": d_r2,
    }
    keyed = {x_axis.replace("-", "_"): primary, other.replace("-", "_"): control}
    return {
        "n": float(len(df)),
        "n_dropped": float(n_dropped),
        "n_below_floor": float(below),
        "x_axis": x_axis,
        "x_unit": x_unit,
        **({"implied_km_per_ms": 2 / slope} if x_unit == "km" and slope else {}),
        **{f"via_pni_{k}": v for k, v in keyed["via_pni"].items()},
        **{f"direct_{k}": v for k, v in keyed["direct"].items()},
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
        x_axis: str = typer.Option(
            "via-pni",
            "--x-axis",
            help="Path whose propagation delay goes on x: `via-pni` for "
            "d(VP,PNI)+d(PNI,TG), or `direct` for d(VP,TG) — the same figure as "
            "the no-PNI control. Both fits are reported either way.",
        ),
        x_unit: str = typer.Option(
            "km",
            "--x-unit",
            help="`km` puts the path's distance on x and draws the 2/3 c floor "
            "as a sloped line (fitted slope in ms/km, so 2/slope is an implied "
            "speed). `ms` converts x at 2/3 c first, making the floor y = x.",
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
        suffix = f"_pni_delay.{x_axis}.{x_unit}{'.' + where if where else ''}"
        out_png = dest / f"{csv.stem}{suffix}.png"
        out_csv = dest / f"{csv.stem}{suffix}_points.csv"

        stats = plot(
            df, out_png, n_dropped=n_dropped, log_axes=log_axes, title=title,
            x_axis=x_axis, x_unit=x_unit,
        )
        df.to_csv(out_csv, index=False)

        typer.echo(f"wrote {out_png}")
        typer.echo(f"wrote {out_csv}")
        for key, value in stats.items():
            typer.echo(
                f"  {key:24s} {value}" if isinstance(value, str)
                else f"  {key:24s} {value:.4g}"
            )
