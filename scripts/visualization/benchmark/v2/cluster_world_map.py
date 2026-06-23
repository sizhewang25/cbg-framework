"""Centroid-aware world-map viewer — the cluster-classification mirror of
`mtl_world_map.py`.

`mtl_world_map.py` ranks targets by error-to-ground-truth (`error_km`) and draws
the raw MTL geometry. The benchmark now scores targets as a *classification* over
a centroid answer space (a prediction is correct iff it snaps to the truth's
centroid). This viewer reuses the MTL geometry payload (VP rings, feasible
region, prediction, truth) and overlays the **answer space** so a failure can be
read directly:

  - all centroids as faint dots (the answer cells);
  - the truth's cell  — gold ring of radius R around the truth's centroid;
  - the prediction's cell — crimson ring around the predicted centroid;
  - outcome (MATCH / WRONG / GIVE_UP) and the attributed **failure mechanism**
    (NO_PROXIMITY / ERRONEOUS_CONTAINMENT / RTT_INFLATION / OTHER) in the title +
    meta line, with the deciding features (nearest-VP km, RTT inflation, blocker
    fraction) so the analysis from `characterize_failures.py` can be confirmed
    case by case.

Targets are sorted **failures first** (give-ups, then largest centroid error), and
filterable by outcome and mechanism, so the bad cases are front and centre.

Outcome + mechanism are joined from the attribution table written by
`scripts.analysis.partvp.characterize_failures` (per_target_failures.parquet);
run that first. Centroid coordinates come from the same answer space
(`build_answer_space`, R = 50 km by default).

CLI:
    python -m scripts.visualization.benchmark.v2.cluster_world_map \\
        --config scripts/analysis/partvp/cfg_textbook/north_america_as7018_final_na.yaml \\
        --combo vanilla_cbg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
import yaml

from scripts.analysis.plot_cluster_cdf import build_answer_space
from scripts.visualization.benchmark.v2 import mtl_world_map as base
from scripts.visualization.cluster.voronoi import (
    clipped_voronoi_cells,
    resolve_landmass,
)

REPO_ROOT = base.REPO_ROOT
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_HTML_TEMPLATE_PATH = _TEMPLATE_DIR / "cluster_world_map.html"
_JS_TEMPLATE_PATH = _TEMPLATE_DIR / "cluster_world_map.js"

DEFAULT_ATTRIBUTION = (REPO_ROOT / "scripts" / "analysis" / "partvp" / "outputs"
                       / "analysis_fail" / "per_target_failures.parquet")
_OUTPUTS_CANDIDATES = ("outputs_partvp", "outputs")
_MECH_LABEL = {
    "NO_PROXIMITY": "no proximity",
    "ERRONEOUS_CONTAINMENT": "containment",
    "RTT_INFLATION": "RTT inflation",
    "OTHER": "other",
}


def _resolve_outputs_root(run_id: str, source: str, setup: str, combo: str) -> Path:
    """Pick the outputs root that actually holds this run's combo (regional
    partvp runs live under outputs_partvp/, the global run under outputs/)."""
    for name in _OUTPUTS_CANDIDATES:
        root = REPO_ROOT / "scripts" / "benchmark" / "v2" / name
        base_dir = root / run_id / source / setup
        if base_dir.is_dir() and any(
            (p / combo / "targets.parquet").exists()
            for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")
        ):
            return root
    raise SystemExit(
        f"no outputs root holds {run_id}/{source}/{setup}/*/{combo}/targets.parquet "
        f"(looked under {_OUTPUTS_CANDIDATES})")


def _resolve_clusters_dir(run_id: str, source: str, setup: str) -> Path | None:
    """Canonical cluster-eval dir for the run (GT-derived; combo-independent).
    Prefer outputs/ (where cluster-eval writes), then outputs_partvp/."""
    for name in ("outputs", "outputs_partvp"):
        cand = (REPO_ROOT / "scripts" / "benchmark" / "v2" / name / run_id
                / source / setup / "clusters")
        if (cand / "clusters.csv").exists():
            return cand
    return None


def _voronoi_cell_rings(
    lats: np.ndarray, lons: np.ndarray, landmass: str, *, buffer_deg: float = 0.3
) -> tuple[list[list[list[float]]], str, int]:
    """Nearest-centroid (Voronoi) partition of the answer space, clipped to a
    named landmass — the "nearest hub" snapping fences.

    Mirrors `scripts.visualization.cluster.voronoi`: resolve the landmass, seed
    from the centroids inside it (buffered for coastal slack), clip cells to the
    boundary. Returns ``(rings, label, n_seeds)`` where each ring is a list of
    ``[lat, lon]`` (a MultiPolygon cell contributes one ring per part), ready to
    drop into a Plotly ``fill:"toself"`` trace.
    """
    boundary, label = resolve_landmass(landmass)
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    region = boundary.buffer(buffer_deg) if buffer_deg else boundary
    inside = shapely.contains_xy(region, lons, lats)
    cells = clipped_voronoi_cells(lons[inside], lats[inside], boundary)

    rings: list[list[list[float]]] = []
    for geom in cells.geometry:
        for poly in getattr(geom, "geoms", [geom]):
            ext = getattr(poly, "exterior", None)
            if ext is None:
                continue
            rings.append([[round(la, 4), round(lo, 4)] for lo, la in ext.coords])
    return rings, label, int(inside.sum())


def enrich_payload(payload: dict[str, Any], *, run_dir: Path, clusters_dir: Path | None,
                   radius_km: float, attribution: pd.DataFrame, run_id: str,
                   combo: str, landmass: str | None = None) -> dict[str, Any]:
    """Add the answer-space layer + per-target outcome/mechanism to a base
    `mtl_world_map` payload, and re-sort targets failures-first.

    When `landmass` is given, also attach the landmass-clipped Voronoi partition
    of the centroids (`payload["voronoi_cells"]`) as the nearest-hub snapping
    layer."""
    index, n_centroids, n_targets = build_answer_space(
        run_dir, None, None, radius_km, clusters_dir=clusters_dir)

    attr = attribution[(attribution["run_id"] == run_id)
                       & (attribution["combo_id"] == combo)].set_index("target_id")

    payload["centroids"] = [[round(float(la), 4), round(float(lo), 4)]
                            for la, lo in zip(index.lat, index.lon)]
    payload["radius_km"] = radius_km
    payload["n_centroids"] = int(n_centroids)
    payload["mech_labels"] = _MECH_LABEL

    if landmass:
        rings, vlabel, n_seed = _voronoi_cell_rings(
            np.asarray(index.lat), np.asarray(index.lon), landmass)
        payload["voronoi_cells"] = rings
        payload["voronoi_label"] = vlabel
        print(f"voronoi: {len(rings)} cell rings over {vlabel} "
              f"from {n_seed}/{n_centroids} centroids")

    for t in payload["targets"]:
        true = t["true"]
        ti = int(index.query([true[0]], [true[1]])[0][0])
        t["truth_centroid"] = [round(float(index.lat[ti]), 4), round(float(index.lon[ti]), 4)]
        if t["pred"] is not None:
            pi = int(index.query([t["pred"][0]], [t["pred"][1]])[0][0])
            t["pred_centroid"] = [round(float(index.lat[pi]), 4), round(float(index.lon[pi]), 4)]
            etc = index.distance_to([t["pred"][0]], [t["pred"][1]], np.array([ti]))[0]
            t["error_to_centroid_km"] = base._safe_float(etc)
            recomputed_match = bool(pi == ti)
        else:
            t["pred_centroid"] = None
            t["error_to_centroid_km"] = None
            recomputed_match = False

        tid = t["target_id"]
        if tid in attr.index:
            r = attr.loc[tid]
            t["match"] = bool(r["match"])
            t["outcome"] = str(r["outcome"])
            t["mechanism"] = str(r["mechanism"]) if r["mechanism"] else ""
            t["feat"] = {
                "avail_min_vp_km": base._safe_float(r["avail_min_vp_km"]),
                "part_min_infl": base._safe_float(r["part_min_infl"]),
                "frac_blockers": base._safe_float(r["frac_blockers"]),
                "nearest_other_centroid_km": base._safe_float(r["nearest_other_centroid_km"]),
            }
        else:
            t["match"] = recomputed_match
            t["outcome"] = ("MATCH" if recomputed_match
                            else "WRONG" if t["pred"] is not None else "GIVE_UP")
            t["mechanism"] = ""
            t["feat"] = {}

    # Failures first: matched last; among the rest, give-ups (no centroid error)
    # then largest centroid error.
    def _key(t: dict[str, Any]) -> tuple[int, float]:
        e = t.get("error_to_centroid_km")
        return (1 if t["match"] else 0, -(e if e is not None else 1e9))

    payload["targets"].sort(key=_key)
    return payload


def render_html(payload: dict[str, Any]) -> str:
    title = f"{payload['run_id']} · {payload['combo_id']}"
    html = _HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    js = _JS_TEMPLATE_PATH.read_text(encoding="utf-8")
    html = html.replace("__SCRIPT__", js).replace("__TITLE__", title)
    blob = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    return html.replace("__PAYLOAD__", blob)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True,
                        help="Benchmark/cfg_textbook YAML (provides run_id/source/setup).")
    parser.add_argument("--combo", type=str, default="vanilla_cbg")
    parser.add_argument("--radius-km", type=float, default=50.0)
    parser.add_argument(
        "--landmass", type=str, default=None,
        help="Overlay the nearest-hub (Voronoi) snapping partition of the "
             "centroids, clipped to this landmass: a continent ('Europe', "
             "'North America') or a country code/name ('US', 'USA', 'France').")
    parser.add_argument("--attribution", type=Path, default=DEFAULT_ATTRIBUTION,
                        help="per_target_failures.parquet from characterize_failures.")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs_cluster",
                        help="Directory under which <run_id>/<combo>_cluster_map.html is written.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    run_id = cfg["run_id"]
    source = cfg["source"]
    setup = cfg.get("setup", "probes_to_anchors")

    outputs_root = _resolve_outputs_root(run_id, source, setup, args.combo)
    clusters_dir = _resolve_clusters_dir(run_id, source, setup)
    run_dir = outputs_root / run_id
    print(f"run_id={run_id} combo={args.combo} outputs_root={outputs_root.name} "
          f"clusters={'yes' if clusters_dir else 'in-process'}")

    if not args.attribution.exists():
        raise SystemExit(
            f"attribution table {args.attribution} not found — run "
            "`python -m scripts.analysis.partvp.characterize_failures` first.")
    attribution = pd.read_parquet(args.attribution)

    run_out = args.out_dir / run_id
    run_out.mkdir(parents=True, exist_ok=True)
    static_dir = run_out / "static" / args.combo

    payload = base.build_payload(args.config, args.combo, static_dir=static_dir,
                                 outputs_root=outputs_root)
    payload = enrich_payload(payload, run_dir=run_dir, clusters_dir=clusters_dir,
                             radius_km=args.radius_km, attribution=attribution,
                             run_id=run_id, combo=args.combo, landmass=args.landmass)

    out_path = run_out / f"{args.combo}_cluster_map.html"
    out_path.write_text(render_html(payload), encoding="utf-8")
    n_fail = sum(1 for t in payload["targets"] if not t["match"])
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path} ({size_mb:.2f} MB) — {len(payload['targets'])} targets, "
          f"{n_fail} failures, {payload['n_centroids']} centroids")


if __name__ == "__main__":
    main()
