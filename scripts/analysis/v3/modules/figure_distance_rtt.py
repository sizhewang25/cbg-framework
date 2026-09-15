"""§8.1 figure 1a: min-RTT against d(VP, target), one dataset stacked on another.

The claim the figure is built for: traffic weighting prunes low-traffic *flows*,
which removes most indirect-routing pairs, so a weighted dataset should
concentrate toward the bottom-left of the (distance, RTT) cloud -- short paths
whose RTT sits near the 2/3 c floor.

Two layers, drawn in this order:

* **mesh** -- every measured pair, grey, underneath. Context, not a peer series.
* **traffic-weighted** -- the surviving pairs, red, on top.

The overlay is the point. Both layers are semi-transparent so the weighted
subset reads as density against its own parent population rather than as an
unrelated scatter, and drawing mesh first keeps red from being buried.

## Statistics are computed on ALL pairs, never on the clipped view

`--x-max` / `--y-max` set the axis limits and nothing else. Every number in the
stats file is computed over the full input, and the count of points outside the
view is reported separately.

This is deliberate and it is the whole reason the stats are emitted here rather
than being read off the picture. Restricting x compresses its range, which
attenuates Pearson r **regardless of routing quality** -- so a weighted arm that
legitimately concentrates into short distances would show a *lower* r than the
mesh purely as an artifact of the narrower x. Computing r on the clipped subset
would build that confound into the number.

For the same reason `x_range` and `residual_rmse_ms` ship beside every r. A
lower r with a visibly narrower `x_p1..x_p99` and a comparable or smaller
residual RMSE is range restriction; a lower r with a *larger* residual RMSE is
genuinely worse agreement. Pearson r alone cannot separate the two, and §8.1's
claim is about the second.

## What today's inputs can and cannot show

There is **no real traffic-weighted dataset**. Weights are enterprise-sensitive
and were never collected: no source CSV under `datasets/final/` carries a
`weight` column except the `.randweight` fixture, whose weights are synthetic
and uniform-random.

So `as01-...randweight.traffic-weighted.csv` is the right *shape* -- a strict
subset of mesh pairs, flow-pruned, 41,400 of 53,262 (77.7%), all 399 targets and
all 134 VPs retained -- driven by the wrong *signal*. Pruning at random cannot
concentrate the cloud, so the expected result is red sitting uniformly on grey.
That is this command's validation criterion, not its finding: red concentrating
bottom-left under random weights would mean something is wrong with the code.

Command: `plot-distance-rtt`. Writes `<stem>_distance_rtt.png` and
`<stem>_distance_rtt_stats.json` beside the input CSV, or into `--out-dir`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from scripts.analysis.v3.modules import config as config_mod
from scripts.analysis.v3.modules.answer_space import elementwise_km
from scripts.analysis.v3.modules.diagram.common.draw import plt
from scripts.analysis.v3.modules.diagram.common.palette import (
    _C_AXIS,
    _C_GRID,
    _C_INK,
    _C_INK_2,
    _C_MUTED,
)

#: Round-trip ms per km at 2/3 c -- the same constant the CBG models are
#: anchored on. Drawn as the physical floor; no pair can sit below it.
THEORETICAL_SLOPE = 0.01

#: Grey for the backdrop layer, red for the highlighted subset.
#:
#: Validated with the dataviz skill's `validate_palette.js` against this
#: layer's surface (`#ffffff`): CVD separation dE 11.5 (deutan) / 24.2
#: (tritan), normal-vision dE 21.1, both slots >= 3:1 contrast. The chroma
#: floor FAILs on the grey by construction -- the check exists so two *peer*
#: categorical series are not confused with a disabled state, and this is a
#: highlight-against-context pair where reading as background is the intent.
#: Secondary encoding is present regardless: the legend names both layers and
#: the stats JSON is the table view.
_C_MESH = "#8a8a8a"
_C_TW = "#d32f2f"

STATS_SUFFIX = "_distance_rtt_stats.json"
PNG_SUFFIX = "_distance_rtt.png"


def csvs_from_config(run_id: str) -> tuple[Path, Path | None, dict]:
    """`(mesh_csv, tw_csv, provenance)` read off `configs/<run_id>.yaml`.

    The paths come from **`benchmark.source_kwargs`**, not from an
    `analysis.plot-distance-rtt` block, and that is deliberate.

    `config.py`'s `default_map` only feeds `analysis:` blocks, so these could
    not arrive through the normal defaults mechanism anyway. But the better
    reason is single-sourcing: these two paths are what the *benchmark* was run
    on. Copying them into an `analysis:` block would create a second
    declaration free to drift, and a drifted copy would make this figure
    describe a different dataset than the arm it is captioned for -- silently,
    since both files would exist and parse.

    Two config shapes are recognised:

    * `traffic_weighted_csv` runs declare `mesh_csv_path` **and**
      `weighted_csv_path` -- both layers, which is the figure's purpose.
    * `generic_csv` runs declare a single `csv_path` -- the mesh layer alone,
      which is still the parent population every weighted claim is relative to.

    The published `as0X-260728-260802.yaml` configs carry `benchmark: {}`
    (their CSVs were reconstructed from run outputs, so no config records
    them). Those raise, naming `--mesh-csv` as the way through, rather than
    guessing a path from the run id.
    """
    # Shared with `answer_space`, which resolves the same block to decide a
    # weighted run's class set. One copy of "where the config is" and "how a
    # config path becomes absolute" -- see config.source_kwargs_for_run.
    kwargs, cfg_path = config_mod.source_kwargs_for_run(
        run_id, needed_for="the CSVs to draw"
    )
    _abs = config_mod.resolve_repo_path

    if "mesh_csv_path" in kwargs:
        mesh = _abs(kwargs["mesh_csv_path"])
        tw = _abs(kwargs["weighted_csv_path"]) if kwargs.get("weighted_csv_path") else None
        shape = "traffic_weighted_csv (mesh_csv_path + weighted_csv_path)"
    elif "csv_path" in kwargs:
        mesh, tw = _abs(kwargs["csv_path"]), None
        shape = "generic_csv (csv_path; mesh layer only)"
    else:
        raise typer.BadParameter(
            f"--run-id {run_id}: {cfg_path} declares no benchmark.source_kwargs "
            f"csv path (found keys {sorted(kwargs)}). The published as0X configs "
            f"carry `benchmark: {{}}` because their CSVs were reconstructed from "
            f"run outputs -- pass --mesh-csv explicitly for those."
        )

    missing = [str(q) for q in (mesh, tw) if q is not None and not q.exists()]
    if missing:
        raise typer.BadParameter(
            f"--run-id {run_id}: {cfg_path} points at {missing}, which do not "
            f"exist. For a *-weighted config this is expected until the "
            f"weight-bearing export lands: no .tbweight CSV has been collected "
            f"(weights are enterprise-sensitive), and the config header marks "
            f"both paths `>>> SET THESE TWO PATHS <<<`. Only `.randweight` "
            f"exists, and its weights are synthetic uniform-random -- a "
            f"pipeline fixture, not a traffic-weighted dataset."
        )
    return mesh, tw, {"config": str(cfg_path), "shape": shape}


def load_pairs(
    csv_path: Path,
    *,
    rtt_col: str = "rtt_ms",
    vp_prefix: str = "vp",
    tg_prefix: str = "target",
) -> tuple[pd.DataFrame, int]:
    """`(distance_km, rtt_ms)` per measured pair, plus the dropped count.

    Rows with a missing coordinate or a non-positive RTT are dropped and
    counted, matching `eval_source.load_canonical_csv`'s filter so this figure
    describes the same graph the benchmark ran on.
    """
    need = [f"{vp_prefix}_lat", f"{vp_prefix}_lon", f"{tg_prefix}_lat", f"{tg_prefix}_lon", rtt_col]
    df = pd.read_csv(csv_path)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise typer.BadParameter(
            f"{csv_path} lacks {missing}; have {sorted(df.columns)[:14]}... "
            f"adjust --rtt-col / --vp-prefix / --tg-prefix"
        )
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n_before = len(df)
    df = df[df[need].notna().all(axis=1) & (df[rtt_col] > 0)]
    out = pd.DataFrame(
        {
            "distance_km": elementwise_km(
                df[f"{vp_prefix}_lat"].to_numpy(dtype=float),
                df[f"{vp_prefix}_lon"].to_numpy(dtype=float),
                df[f"{tg_prefix}_lat"].to_numpy(dtype=float),
                df[f"{tg_prefix}_lon"].to_numpy(dtype=float),
            ),
            "rtt_ms": df[rtt_col].to_numpy(dtype=float),
        }
    )
    return out.reset_index(drop=True), n_before - len(out)


def series_stats(
    pairs: pd.DataFrame, *, x_max: float | None, y_max: float | None, n_dropped: int
) -> dict:
    """Fit and spread over **all** pairs, with the out-of-view count beside it.

    `x_range` / `x_p1` / `x_p99` and `residual_rmse_ms` travel with `pearson_r`
    so a lower r can be diagnosed as range restriction rather than misread as
    worse routing -- see the module docstring.
    """
    x = pairs["distance_km"].to_numpy(dtype=float)
    y = pairs["rtt_ms"].to_numpy(dtype=float)
    n = int(len(pairs))
    if n < 2:
        return {"n": n, "n_dropped": int(n_dropped), "note": "fewer than 2 pairs; no fit"}

    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r = float(np.corrcoef(x, y)[0, 1])
    outside = 0
    if x_max is not None:
        outside += int((x > x_max).sum())
    if y_max is not None:
        outside += int(((y > y_max) & (x <= (x_max if x_max is not None else np.inf))).sum())

    return {
        "n": n,
        "n_dropped": int(n_dropped),
        "pearson_r": round(r, 6),
        "r2": round(r * r, 6),
        "ols_slope_ms_per_km": round(float(slope), 8),
        "ols_intercept_ms": round(float(intercept), 6),
        "implied_km_per_ms": round(2 / float(slope), 3) if slope > 0 else None,
        "residual_rmse_ms": round(float(np.sqrt((resid**2).mean())), 6),
        "residual_p50_ms": round(float(np.median(resid)), 6),
        # Range-restriction diagnostics. p1..p99 rather than min..max so one
        # outlying pair cannot make a compressed range look wide.
        "x_min_km": round(float(x.min()), 3),
        "x_p1_km": round(float(np.percentile(x, 1)), 3),
        "x_p50_km": round(float(np.percentile(x, 50)), 3),
        "x_p99_km": round(float(np.percentile(x, 99)), 3),
        "x_max_km": round(float(x.max()), 3),
        "x_sd_km": round(float(x.std(ddof=1)), 3),
        "y_p50_ms": round(float(np.percentile(y, 50)), 3),
        "y_p99_ms": round(float(np.percentile(y, 99)), 3),
        "n_outside_view": outside,
        "share_outside_view": round(outside / n, 6),
    }


def build(
    mesh: pd.DataFrame,
    tw: pd.DataFrame | None,
    *,
    x_max: float | None,
    y_max: float | None,
    mesh_label: str,
    tw_label: str,
    title: str | None,
    mesh_point_size: float,
    tw_point_size: float,
    mesh_alpha: float,
    tw_alpha: float,
    out_png: Path,
    stats: dict,
) -> None:
    """Draw the overlay. Mesh underneath, weighted on top, both translucent."""
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=200)

    m = stats["mesh"]
    ax.scatter(
        mesh["distance_km"], mesh["rtt_ms"],
        s=mesh_point_size, alpha=mesh_alpha, color=_C_MESH, edgecolors="none", zorder=2,
        label=f"{mesh_label}  (n={m['n']:,}, r={m.get('pearson_r', float('nan')):.3f})",
    )
    if tw is not None and len(tw):
        w = stats["traffic_weighted"]
        ax.scatter(
            tw["distance_km"], tw["rtt_ms"],
            s=tw_point_size, alpha=tw_alpha, color=_C_TW, edgecolors="none", zorder=3,
            label=f"{tw_label}  (n={w['n']:,}, r={w.get('pearson_r', float('nan')):.3f})",
        )

    # The physical floor. Anything below it is a coordinate or RTT error.
    hi = x_max if x_max is not None else float(mesh["distance_km"].max())
    span = np.array([0.0, hi])
    ax.plot(span, THEORETICAL_SLOPE * span, ls="--", lw=1.1, color=_C_INK_2, zorder=4,
            label=f"y = {THEORETICAL_SLOPE} x   (2/3 c round trip)")

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("d(VP, target)  [km]")
    ax.set_ylabel("min-RTT  [ms]")
    ax.set_title(title or "min-RTT vs VP-to-target distance", color=_C_INK)
    ax.grid(True, color=_C_GRID, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_C_AXIS)
    # Legend markers are sized and opaque independently of the data marks: at
    # s=2.2 and alpha=0.45 a swatch would be invisible, and identity must never
    # rest on a mark the reader cannot see.
    leg = ax.legend(loc="upper left", fontsize=7.5, frameon=False, scatterpoints=1)
    for handle in leg.legend_handles:
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([28.0])
        handle.set_alpha(1.0)

    note_lines = [
        f"axes clipped to x <= {x_max:,.0f} km, y <= {y_max:,.0f} ms"
        if x_max is not None and y_max is not None else "axes unclipped",
        "every statistic is over ALL pairs, not the clipped view",
    ]
    for key, label in (("mesh", mesh_label), ("traffic_weighted", tw_label)):
        s = stats.get(key)
        if not s or "pearson_r" not in s:
            continue
        note_lines.append(
            f"{label}: x p1-p99 {s['x_p1_km']:,.0f}-{s['x_p99_km']:,.0f} km, "
            f"resid RMSE {s['residual_rmse_ms']:.2f} ms, "
            f"{s['share_outside_view']:.1%} off-view"
        )
    ax.annotate(
        "\n".join(note_lines),
        xy=(0.98, 0.02), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=6.5, color=_C_MUTED,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def register(app: typer.Typer) -> None:
    @app.command("plot-distance-rtt")
    def plot_distance_rtt_cmd(
        run_id: str = typer.Option(
            None, "--run-id",
            help="Take both CSVs from configs/<run-id>.yaml's "
                 "benchmark.source_kwargs -- `mesh_csv_path` + "
                 "`weighted_csv_path` on a *-weighted config, or `csv_path` "
                 "alone on a *-mesh one. Mutually exclusive with --mesh-csv.",
        ),
        mesh_csv: Path = typer.Option(
            None, "--mesh-csv", exists=True, dir_okay=False,
            help="Canonical CSV for the backdrop (grey) layer: vp_lat/vp_lon, "
                 "target_lat/target_lon, rtt_ms. Overrides --run-id.",
        ),
        tw_csv: Path = typer.Option(
            None, "--tw-csv", exists=True, dir_okay=False,
            help="Canonical CSV for the overlaid (red) layer, same schema. "
                 "Omit to draw the backdrop alone. NOTE: no real "
                 "traffic-weighted CSV exists -- the .randweight variant is a "
                 "synthetic uniform fixture, so its overlay validates the code "
                 "path and is not evidence.",
        ),
        x_max: float = typer.Option(4000.0, "--x-max", help="x axis cut, km. Clips the VIEW only."),
        y_max: float = typer.Option(100.0, "--y-max", help="y axis cut, ms. Clips the VIEW only."),
        mesh_label: str = typer.Option("mesh", "--mesh-label"),
        tw_label: str = typer.Option(
            None, "--tw-label",
            help="Overlay legend label. Default: derived from the overlay's "
                 "FILENAME, so a fixture-derived figure says so on its face.",
        ),
        rtt_col: str = typer.Option("rtt_ms", "--rtt-col", help="min-RTT column, in ms."),
        vp_prefix: str = typer.Option("vp", "--vp-prefix", help="Column prefix for the VP."),
        tg_prefix: str = typer.Option("target", "--tg-prefix", help="Column prefix for the target."),
        # The overlay is drawn SMALLER than the backdrop on purpose. A weighted
        # subset is a subset, so at any retention rate the two layers coincide
        # almost everywhere; equal markers let red cover grey completely and the
        # figure then cannot show which pairs were *dropped*, which is the point
        # of the overlay. A smaller red dot inside a larger grey one keeps both
        # readable, and a grey dot with no red centre is a dropped pair.
        mesh_point_size: float = typer.Option(
            11.0, "--mesh-point-size", help="Backdrop marker area in pt^2.",
        ),
        tw_point_size: float = typer.Option(
            2.2, "--tw-point-size", help="Overlay marker area in pt^2. Smaller than the backdrop.",
        ),
        mesh_alpha: float = typer.Option(0.22, "--mesh-alpha", help="Backdrop opacity."),
        tw_alpha: float = typer.Option(0.45, "--tw-alpha", help="Overlay opacity."),
        title: str = typer.Option(None, "--title", help="Override the figure title."),
        out_dir: Path = typer.Option(
            None, "--out-dir", help="Where to write. Default: alongside --mesh-csv.",
        ),
    ) -> None:
        """§8.1 figure 1a: min-RTT vs d(VP, target), weighted stacked on mesh.

        Writes <stem>_distance_rtt.png and <stem>_distance_rtt_stats.json.
        Statistics are computed over every pair; --x-max/--y-max clip the view
        only, because restricting x attenuates Pearson r regardless of routing
        quality and that confound must not be baked into the number.
        """
        # An explicit --mesh-csv wins over --run-id, as the help states. This
        # is NOT interchangeable with a strict XOR: `--config` injects the
        # config's top-level `run_id` as a default into every command that
        # declares one, so under `--config X plot-distance-rtt --mesh-csv Y`
        # both arrive and a strict XOR would reject a perfectly clear request.
        if mesh_csv is not None:
            run_id = None
        elif run_id is None:
            raise typer.BadParameter(
                "pass --mesh-csv, or --run-id to read both CSVs from "
                "configs/<run-id>.yaml's benchmark.source_kwargs"
            )

        provenance: dict = {}
        if run_id is not None:
            mesh_csv, from_cfg_tw, provenance = csvs_from_config(run_id)
            # An explicit --tw-csv still wins, so a config's weighted path can
            # be overridden without editing it.
            if tw_csv is None:
                tw_csv = from_cfg_tw

        mesh, mesh_dropped = load_pairs(
            mesh_csv, rtt_col=rtt_col, vp_prefix=vp_prefix, tg_prefix=tg_prefix
        )
        stats: dict = {
            "mesh": series_stats(mesh, x_max=x_max, y_max=y_max, n_dropped=mesh_dropped),
        }
        tw = None
        if tw_csv is not None:
            tw, tw_dropped = load_pairs(
                tw_csv, rtt_col=rtt_col, vp_prefix=vp_prefix, tg_prefix=tg_prefix
            )
            stats["traffic_weighted"] = series_stats(
                tw, x_max=x_max, y_max=y_max, n_dropped=tw_dropped
            )

        # Derived here rather than by the caller, so the safeguard holds at
        # every entry point. A figure captioned "traffic-weighted" would be
        # read as evidence of real weights; `.randweight` carries synthetic
        # uniform-random ones. This used to live in
        # create_analysis_artifacts.sh, which left `--run-id
        # as01-randweight-precomputed` mislabelling its own output.
        if tw_label is None:
            tw_label = (
                "randweight fixture (SYNTHETIC weights)"
                if tw_csv is not None and "randweight" in Path(tw_csv).name
                else "traffic-weighted"
            )

        stem = mesh_csv.stem
        target_dir = Path(out_dir) if out_dir else mesh_csv.parent
        out_png = target_dir / f"{stem}{PNG_SUFFIX}"
        out_stats = target_dir / f"{stem}{STATS_SUFFIX}"

        build(
            mesh, tw,
            x_max=x_max, y_max=y_max,
            mesh_label=mesh_label, tw_label=tw_label, title=title,
            mesh_point_size=mesh_point_size, tw_point_size=tw_point_size,
            mesh_alpha=mesh_alpha, tw_alpha=tw_alpha,
            out_png=out_png, stats=stats,
        )

        payload = {
            "inputs": {
                "run_id": run_id,
                "resolved_from": provenance or {"source": "--mesh-csv / --tw-csv"},
                "mesh_csv": str(mesh_csv),
                "tw_csv": str(tw_csv) if tw_csv else None,
                "rtt_col": rtt_col,
                "vp_prefix": vp_prefix,
                "tg_prefix": tg_prefix,
            },
            "view": {"x_max_km": x_max, "y_max_km": None, "y_max_ms": y_max,
                     "scale": "linear on both axes"},
            "policy": (
                "every statistic is computed over ALL pairs; --x-max/--y-max clip "
                "the view only. Restricting x compresses its range, which "
                "attenuates Pearson r regardless of routing quality, so x_range "
                "and residual_rmse_ms travel with every r to separate range "
                "restriction from genuinely worse agreement."
            ),
            "series": stats,
        }
        out_stats.parent.mkdir(parents=True, exist_ok=True)
        out_stats.write_text(json.dumps(payload, indent=2) + "\n")

        typer.echo(f"wrote {out_png}")
        typer.echo(f"wrote {out_stats}")
        for key in ("mesh", "traffic_weighted"):
            s = stats.get(key)
            if not s or "pearson_r" not in s:
                continue
            typer.echo(
                f"  {key:17s} n={s['n']:>7,}  r={s['pearson_r']:.4f}  "
                f"resid RMSE={s['residual_rmse_ms']:.3f} ms  "
                f"x p1-p99={s['x_p1_km']:,.0f}-{s['x_p99_km']:,.0f} km  "
                f"off-view={s['share_outside_view']:.1%}"
            )
