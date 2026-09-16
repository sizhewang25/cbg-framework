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

`--x-max` / `--y-max` and the matching `--x-tick` / `--y-tick` set the view and
nothing else, the contract `plot-distance-rtt` states at length: every fit, the
below-floor count and the median residual are computed over all rows, and the
off-view count is annotated beside them. Restricting x compresses its range,
which attenuates Pearson r on its own, so a clipped-subset fit would not be
comparable with an unclipped one.

Pinning them matters twice over. `create_analysis_artifacts.sh` draws one panel
per assignment rule, and `vp_nearest` bends paths further than `argmin` does, so
data-driven axes would give the three panels three scales. And this figure is
read beside `plot-distance-rtt`, whose x is the same VP-to-target line
*unbent* — the two are cut and ruled alike (5,000 km / 100 ms, stepped 1,000 /
20) so that the bend is the only difference between them. That command defaults
to the frame; here it is per-run in `configs/*.yaml`, since a peer with
intercontinental paths needs a wider one. Unset, the axes end at the data.

Reading the figure: the floor line is the speed-of-internet limit, so points
below it are physically impossible *for that path*. On the `direct` axis that is
a statement about the data — a point below it means the RTT or a coordinate is
wrong. On the `via-pni` axis it is a statement about the *assignment*: the pair
cannot have crossed the site it was given. Since `via >= direct` always, a
via-PNI violation means exactly `detour_ratio > air_inflation`. The vertical gap
above the floor is the inflation left unexplained; a tight cloud parallel to the
floor means the detour accounts for the bulk of the delay, which is the
correlation this script is meant to show or fail to show.

The two floors nest, which is why a point below the via-PNI line need not be a
broken row. `via >= direct` puts the bent floor *above* the straight one, so a
row can clear 2/3 c on the geodesic — no `min_rtt - 0.01·d(VP,TG)` violation
anywhere — and still fall under the bent floor. The band between the two floors
is exactly the rows whose detour is longer than their RTT has inflation to pay
for, i.e. `detour_ratio > air_inflation`, and both ratios are emitted per row so
a listed point carries its own arithmetic.

What such a row rules out is the *triple*: this pair cannot have crossed this
site. It does not say which of the three inputs is at fault, and this module
does not try to attribute it — the pair, the assignment rule
(`build-pni-graph --strategy`) and the site list itself are all candidates, and
a list that is approximate or short of the peer's real footprint produces
exactly these rows. `below_floor()` therefore just lists them: the ids, the
three coordinates' pairwise distances and the RTT, which is what is needed to go
look. `build-pni-feasibility` is the command that tests a site list by
exclusion rather than by eye.

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

`<prefix>_country` / `_region` / `_city` are read where present and merged into
one `<prefix>_loc` for the below-floor listing; roles that carry none -- every
VP and target in this tree -- get theirs from the coordinate instead, and the
command says which is which.

Column prefixes and the RTT column are options, so a CSV that calls its
intermediate `ixp_*` needs `--pni-prefix ixp` rather than a rename. Rows with a
missing coordinate or a non-positive RTT are dropped and counted, since a
0 ms or -1 ms RTT is a measurement sentinel, not a fast path.

    python -m scripts.analysis.v3.cli plot-pni-delay --csv pni_pairs.csv

Writes `<csv stem>_pni_delay.png` and `<csv stem>_pni_delay_points.csv` (the
input plus the computed distance/delay/residual columns) next to the input,
unless `--out-dir` says otherwise. When any row sits below the drawn floor, a
third file `<csv stem>_pni_delay_below_floor.csv` holds just those rows, worst
first, and `--show-below` prints the head of it — the (VP, TG) pair, the site it
was assigned, and every pairwise distance among the three, which is what an
outlier below the line has to be chased with.
"""

from __future__ import annotations

import contextlib
import io
import shutil
from functools import reduce
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
    # The same margin against the geodesic. Negative here is a broken input --
    # an RTT or a coordinate -- rather than a refuted PNI assignment.
    df["residual_direct_ms"] = df["min_rtt_ms"] - df["prop_rtt_direct_ms"]

    # The identity a via-PNI violation reduces to: the detour's length overhead
    # versus the inflation the RTT has to spend on it. Carried per row so a
    # below-floor listing explains itself instead of being a bare complaint.
    # Colocated VP and target divide by zero; inf is the honest answer (any
    # detour at all is unaffordable) and is left to propagate.
    with np.errstate(divide="ignore", invalid="ignore"):
        df["detour_ratio"] = df["d_via_pni_km"] / df["d_vp_tg_km"]
        df["air_inflation"] = df["min_rtt_ms"] / df["prop_rtt_direct_ms"]

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

#: Numbers a below-floor listing carries after the ids. All three pairwise
#: distances among (VP, PNI, TG) are here, not just the two path totals, because
#: which leg is long is the whole of reading one of these rows -- a far VP and a
#: far site are different pictures with the same `d_via_pni_km`. The ratio pair
#: follows, since its inequality *is* the below-floor condition, and
#: `below_by_ms` is how far under the floor the row sits and what it sorts on.
_BELOW_COLS = (
    "min_rtt_ms",
    "d_vp_tg_km",
    "d_vp_pni_km",
    "d_pni_tg_km",
    "d_via_pni_km",
    "detour_ratio",
    "air_inflation",
    "residual_direct_ms",
    "below_by_ms",
)


#: The three place parts, in the order they read as one label.
_LOC_PARTS = ("country", "region", "city")


def loc_provenance(df: pd.DataFrame, prefix: str) -> str:
    """Whether a role's `_loc` is the input's own label or a lookup.

    A column check, not a recompute, so a caller can footnote the distinction
    without re-deriving anything.
    """
    return (
        "declared"
        if any(f"{prefix}_{part}" in df.columns for part in _LOC_PARTS)
        else "derived"
    )


def _join_nonempty(left: pd.Series, right: pd.Series) -> pd.Series:
    """`a-b` elementwise, or whichever side is non-empty.

    Vectorised rather than a row-wise `apply`, which on an EMPTY frame returns
    a DataFrame instead of a Series and so cannot be assigned to a column. That
    is not a corner case here: a run whose assignment rule puts nothing below
    the floor -- argmin, routinely -- lists zero rows.
    """
    sep = pd.Series("-", index=left.index).where((left != "") & (right != ""), "")
    return left + sep + right


def _geocoded_loc(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """`country-region-city` of the nearest GeoNames cities1000 entry.

    This is a NEIGHBOURHOOD, not a claim about where the host is registered:
    the nearest populated place to a VP in an outer suburb is the suburb's
    name. It is here because a coordinate pair is not something a reader can
    place at a glance, and placing the row is the whole point of the listing.

    Deduplicated on the coordinate: the operator datasets hold ~20 IP replicas
    per facility, so a few hundred listed rows carry a handful of distinct
    positions. `mode=1` for the reason `benchmark/v2/geo_eval.py` uses it --
    the multiprocess path forks, and a figure command should not. The library
    announces its kdtree load on stdout, which is swallowed here so it cannot
    land in the middle of a printed table.
    """
    try:
        import reverse_geocoder as rg
    except ImportError:  # pragma: no cover - a declared dependency
        return pd.Series([""] * len(lat), index=lat.index, dtype=object)

    coords = pd.DataFrame({"lat": lat.astype(float), "lon": lon.astype(float)})
    if coords.empty:
        return pd.Series([], index=coords.index, dtype=object)
    uniq = [tuple(row) for row in coords.drop_duplicates().to_numpy()]
    with contextlib.redirect_stdout(io.StringIO()):
        hits = rg.search(uniq, mode=1)
    label = {
        key: "-".join(
            str(hit[k]) for k in ("cc", "admin1", "name") if hit.get(k)
        )
        for key, hit in zip(uniq, hits)
    }
    return pd.Series(
        [label[tuple(row)] for row in coords.to_numpy()],
        index=coords.index,
        dtype=object,
    )


def add_loc_columns(df: pd.DataFrame, *prefixes: str) -> pd.DataFrame:
    """Add `<prefix>_loc`, one readable place per role, in place on a copy.

    Declared parts win where the input has them: `pni_edges.csv` carries
    `sel_pni_country/_region/_city` straight off the operator's site list, and
    an operator's own label for its own facility is not something to
    second-guess with a lookup. Nothing upstream carries the same for a VP or a
    target -- the canonical CSV has `vp_country` at most, and the target has
    nothing -- so those are derived from the coordinates. `loc_provenance` says
    which a column is.
    """
    out = df.copy()
    for prefix in prefixes:
        present = [
            f"{prefix}_{part}" for part in _LOC_PARTS if f"{prefix}_{part}" in out.columns
        ]
        if present:
            parts = [out[c].fillna("").astype(str).str.strip() for c in present]
            out[f"{prefix}_loc"] = reduce(_join_nonempty, parts)
        else:
            out[f"{prefix}_loc"] = _geocoded_loc(
                out[f"{prefix}_lat"], out[f"{prefix}_lon"]
            )
    return out


def _label_cols(
    df: pd.DataFrame, vp_prefix: str, pni_prefix: str, tg_prefix: str
) -> list[str]:
    """Columns that name and place the three roles of a row.

    `<prefix>_id` when the input has one, else the coordinate pair — which
    `load_points` has already required, so a row in a listing is always
    identifiable even for a CSV that carries no ids. Each id is followed by its
    `_loc`, since an id alone is not something a reader can place.

    Role order is the measured pair first and the assigned site last: the row
    says "this pair, between these two places, was given that site". The
    VP→PNI→TG path order would put the thing under question in the middle of
    the thing it is being questioned against.
    """
    cols: list[str] = []
    for prefix in (vp_prefix, tg_prefix, pni_prefix):
        if f"{prefix}_id" in df.columns:
            cols.append(f"{prefix}_id")
        else:
            cols += [f"{prefix}_lat", f"{prefix}_lon"]
        if f"{prefix}_loc" in df.columns:
            cols.append(f"{prefix}_loc")
    return cols


def below_floor(
    df: pd.DataFrame,
    *,
    x_axis: str = "via-pni",
    vp_prefix: str = "vp",
    pni_prefix: str = "pni",
    tg_prefix: str = "tg",
) -> pd.DataFrame:
    """The rows under `x_axis`'s 2/3 c floor, worst first.

    The mask is recomputed from the delay column the floor is drawn from rather
    than read off `residual_ms`, so the listing cannot disagree with the count
    the figure annotates itself with even if a caller has rewritten `min_rtt_ms`
    in place. Ids, legs and the ratio pair are moved to the front; every other
    input column is kept, since the point of the file is to chase the row back
    to whatever produced it.
    """
    if x_axis not in X_AXES:
        raise typer.BadParameter(
            f"--x-axis must be one of {sorted(X_AXES)} (got {x_axis!r})"
        )
    _, ms_col, _, _ = X_AXES[x_axis]

    out = df[df["min_rtt_ms"] < df[ms_col]].copy()
    out["below_by_ms"] = out[ms_col] - out["min_rtt_ms"]
    # Only the listed rows are placed. Reverse geocoding every point would load
    # a kdtree to label 20,000 rows nobody is going to read.
    out = add_loc_columns(out, vp_prefix, tg_prefix, pni_prefix)

    lead = _label_cols(out, vp_prefix, pni_prefix, tg_prefix)
    lead += [c for c in _BELOW_COLS if c in out.columns and c not in lead]
    rest = [c for c in out.columns if c not in lead]
    return (
        out[lead + rest]
        .sort_values("below_by_ms", ascending=False)
        .reset_index(drop=True)
    )


def below_floor_table(
    below: pd.DataFrame,
    *,
    n: int = 10,
    vp_prefix: str = "vp",
    pni_prefix: str = "pni",
    tg_prefix: str = "tg",
    width: int | None = None,
) -> str:
    """The head of a `below_floor()` frame as a plain-text table.

    Wrapped into column blocks at `width` (default: the terminal, floor 120)
    rather than printed flat. Three ids, three places and nine numbers is wider
    than a terminal, and a table the emulator soft-wraps mid-row is unreadable
    in a way a stacked one is not.
    """
    if below.empty:
        return "no rows below the floor"
    cols = _label_cols(below, vp_prefix, pni_prefix, tg_prefix)
    cols += [c for c in _BELOW_COLS if c in below.columns and c not in cols]
    if width is None:
        width = max(shutil.get_terminal_size((160, 24)).columns, 120)
    return below.head(n)[cols].to_string(
        index=False, float_format=lambda v: f"{v:,.3f}", line_width=width
    )


def _fit_title(fig, ax) -> None:
    """Shrink the title if it overruns the axes width.

    `--title` is free text, and matplotlib neither wraps nor complains: an
    overlong one is clipped at BOTH ends, so it loses a word at each side and
    still looks deliberate. The per-strategy titles
    `create_analysis_artifacts.sh` passes are long enough to hit this.

    Measured against the drawn axes rather than capped at a character count,
    which would be a guess about the string's own letters and about the figure
    size. Called after `tight_layout`, since that is what fixes the width being
    measured against.
    """
    title = ax.title
    if not title.get_text():
        return
    fig.canvas.draw()
    available = ax.get_window_extent().width
    used = title.get_window_extent().width
    if used > available:
        title.set_fontsize(title.get_fontsize() * available / used)


def plot(
    df: pd.DataFrame,
    out_png: Path,
    *,
    n_dropped: int = 0,
    log_axes: bool = False,
    title: str | None = None,
    x_axis: str = "via-pni",
    x_unit: str = "km",
    x_max: float | None = None,
    y_max: float | None = None,
    x_tick: float | None = None,
    y_tick: float | None = None,
) -> dict[str, float]:
    """Draw the scatter and return the fit statistics it reports.

    `x_axis` selects the path that goes on x. `direct` ignores the PNI entirely,
    which makes the same figure the control: the great-circle line every CBG
    variant's latency-to-distance model assumes. `x_unit` selects whether x
    carries that path's distance or its 2/3 c round-trip delay.

    `x_max` / `y_max` clip the VIEW and nothing else, as on
    `plot-distance-rtt`: every fit, the below-floor count and the median
    residual are computed over all rows, and the share left off-view is
    reported beside them. Unset, the axes end at the data. `x_tick` / `y_tick`
    fix the gridline spacing, which is what makes two pinned panels read off the
    same ruler rather than merely ending at the same number.
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

    # The two reference lines span the x DATA, not the y range. Extending them
    # to where the floor reaches `y.max()` -- which is what this did -- pushes
    # the autoscaled x limit out to `y.max() / floor_slope`, and since min-RTT
    # is inflated well above the floor that is roughly twice the widest path in
    # the input: half the panel ends up empty and the cloud is squeezed into
    # the rest.
    lo = float(x.min()) if len(df) else 0.0
    hi = float(x_max) if x_max is not None else (float(x.max()) * 1.02 if len(df) else 1.0)
    span = np.array([max(lo, 1e-3) if log_axes else 0.0, hi])
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
        if x_max is not None:
            ax.set_xlim(right=float(x_max))
        if y_max is not None:
            ax.set_ylim(top=float(y_max))
    else:
        # Left edge at 0 rather than at the smallest path: a co-located pair is
        # the interesting end of this axis and an origin is what makes the
        # floor line's slope readable. Same for the y origin, which is where
        # the floor line starts and where an RTT axis belongs.
        ax.set_xlim(0.0, hi)
        ax.set_ylim(0.0, float(y_max) if y_max is not None else ax.get_ylim()[1])
        if x_tick or y_tick:
            # Fixed spacing, not just a fixed top: matplotlib picks the step
            # from the range it is given, so two panels pinned to the same
            # limits could still be read off different rulers.
            from matplotlib.ticker import MultipleLocator

            if x_tick:
                ax.xaxis.set_major_locator(MultipleLocator(float(x_tick)))
            if y_tick:
                ax.yaxis.set_major_locator(MultipleLocator(float(y_tick)))

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
    # Outside the drawn view. `y > y_max` is counted only inside the x window so
    # a point past both cuts is one point, not two -- the same convention as
    # `figure_distance_rtt.series_stats`.
    outside = 0
    if len(df):
        if x_max is not None:
            outside += int((x > float(x_max)).sum())
        if y_max is not None:
            in_x = x <= float(x_max) if x_max is not None else np.ones(len(x), dtype=bool)
            outside += int(((y > float(y_max)) & in_x).sum())
    clip_unit = "km" if x_unit == "km" else "ms"
    clipped = [
        f"x <= {float(x_max):,.0f} {clip_unit}" if x_max is not None else "",
        f"y <= {float(y_max):,.0f} ms" if y_max is not None else "",
    ]
    clip_note = ", ".join(c for c in clipped if c)
    note = (
        f"n = {len(df)} triples"
        + (f"  ({n_dropped} dropped: missing coords or rtt ≤ 0)" if n_dropped else "")
        + f"\n{below} below the floor ({below / max(len(df), 1):.1%}) — {below_note}"
        + f"\ncontrol ({other_name}): r={d_r:.3f}, r²={d_r2:.3f}"
        + (f"\nimplied speed {2 / slope:,.0f} km/ms vs 2/3 c = 200 km/ms"
           if x_unit == "km" and np.isfinite(slope) and slope > 0 else "")
        + (f"\naxes clipped to {clip_note}; every statistic is over ALL triples"
           # The count, not only the share: 6 of 20,608 prints as 0.0%, and a
           # reader cannot tell that from nothing being hidden at all.
           f"\n{outside} off-view ({outside / max(len(df), 1):.1%})"
           if clip_note else "")
    )
    ax.annotate(note, xy=(0.98, 0.02), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=7, color=_C_MUTED)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    _fit_title(fig, ax)
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
        "n_outside_view": float(outside),
        "share_outside_view": outside / max(len(df), 1),
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
        x_max: float = typer.Option(
            None,
            "--x-max",
            help="Cut the x axis here, in the unit --x-unit selects. Clips the "
            "VIEW only: every fit and the below-floor count stay over all rows, "
            "and the off-view share is annotated. Pin it to compare panels.",
        ),
        y_max: float = typer.Option(
            None,
            "--y-max",
            help="Cut the y axis here, in ms. Clips the VIEW only, as --x-max.",
        ),
        x_tick: float = typer.Option(
            None,
            "--x-tick",
            help="x gridline spacing, in the unit --x-unit selects. Ignored on "
            "--log-axes.",
        ),
        y_tick: float = typer.Option(
            None,
            "--y-tick",
            help="y gridline spacing, in ms. Pin it alongside --y-max so two "
            "panels are read off the same ruler, not merely cut at the same "
            "number. Ignored on --log-axes.",
        ),
        x_unit: str = typer.Option(
            "km",
            "--x-unit",
            help="`km` puts the path's distance on x and draws the 2/3 c floor "
            "as a sloped line (fitted slope in ms/km, so 2/slope is an implied "
            "speed). `ms` converts x at 2/3 c first, making the floor y = x.",
        ),
        show_below: int = typer.Option(
            10,
            "--show-below",
            help="Print this many below-floor rows, worst first, with the (VP, "
            "TG) pair and the PNI each was assigned. All of them are written to "
            "<stem>_below_floor.csv regardless; 0 prints none.",
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
        the OLS fit plus a straight-line (no-PNI) control for comparison. Rows
        under the drawn floor also go to <stem>_pni_delay_below_floor.csv.
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
        out_below = dest / f"{csv.stem}{suffix}_below_floor.csv"

        stats = plot(
            df, out_png, n_dropped=n_dropped, log_axes=log_axes, title=title,
            x_axis=x_axis, x_unit=x_unit, x_max=x_max, y_max=y_max,
            x_tick=x_tick, y_tick=y_tick,
        )
        df.to_csv(out_csv, index=False)

        below = below_floor(
            df, x_axis=x_axis, vp_prefix=vp_prefix,
            pni_prefix=pni_prefix, tg_prefix=tg_prefix,
        )

        typer.echo(f"wrote {out_png}")
        typer.echo(f"wrote {out_csv}")
        if not below.empty:
            below.to_csv(out_below, index=False)
            typer.echo(f"wrote {out_below}")
        for key, value in stats.items():
            typer.echo(
                f"  {key:24s} {value}" if isinstance(value, str)
                else f"  {key:24s} {value:.4g}"
            )

        if show_below and not below.empty:
            shown = min(show_below, len(below))
            typer.echo(
                f"\n{len(below)} of {len(df)} rows below the {x_axis} floor "
                f"(worst {shown} shown). residual_direct_ms >= 0 means the row "
                f"is legal on the geodesic and it is the (VP, site, TG) triple "
                f"that 2/3 c rules out, not the RTT:"
            )
            typer.echo(
                below_floor_table(
                    below, n=show_below, vp_prefix=vp_prefix,
                    pni_prefix=pni_prefix, tg_prefix=tg_prefix,
                )
            )
            # Which `_loc` columns the input named and which were looked up.
            # They sit in identical columns and only one of them is a claim
            # about where the host is, so the difference is stated rather than
            # left to be inferred from the schema.
            derived = [
                f"{prefix}_loc" for prefix in (vp_prefix, tg_prefix, pni_prefix)
                if loc_provenance(df, prefix) == "derived"
            ]
            if derived:
                typer.echo(
                    f"  ({', '.join(derived)}: nearest GeoNames city to the "
                    f"coordinate, not a registered location)"
                )
