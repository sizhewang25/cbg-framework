"""Refine by-ASN slicing with (probe-country, anchor-country) granularity.

Builds on spotter_normality_by_asn.py. The ASN-pair result was partial:
some pairs recovered Spotter's sigma_z ~ 1.0, others did not because the
ASN spans multiple regions (e.g., AS7922 Comcast US -> AS202422 G-Core
EU/Asia anchors) and routing diversity remains. Hypothesis follow-up: if
we *also* fix (probe-country, anchor-country) within each ASN pair, the
remaining intra-slice routing should be uniform enough that sigma_z snaps
to 1.0 consistently.

Pipeline:
  - Pick top-K (probe-ASN, anchor-ASN) pairs.
  - Within each, pick top-N (probe-country, anchor-country) sub-slices.
  - Re-fit mu(d), sigma(d) per sub-slice and emit Spotter's three panels.
  - Summary chart compares baselines, ASN-only slices, and ASN+country slices.

Run:
    python -m scripts.libs.cbg_feasibility.spotter_normality_by_asn_country
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ujson

from default import (
    CLICKHOUSE_DB,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_USER,
)
from scripts.libs.cbg_feasibility.spotter_normality_check import (
    fit_mu_sigma,
    haversine_vec,
    plot_panel_a,
    plot_panel_b,
    plot_panel_c,
    standardize,
)
from scripts.utils.clickhouse import Clickhouse

PROBES_FILE = Path(
    "datasets/reproducibility_datasets/atlas/reproducibility_probes_and_anchors.json"
)
DEFAULT_OUT_DIR = Path(
    "scripts/libs/cbg_feasibility/outputs/spotter_normality_by_asn_country"
)

BASELINE_UNSLICED = {"mu_z": 0.027, "sigma_z": 0.894, "label": "all probes\n-> all anchors"}
BASELINE_MESHED = {"mu_z": 0.071, "sigma_z": 0.964, "label": "anchors -> anchors"}

# Top-5 ASN-only baselines from spotter_normality_by_asn.py (re-used here for
# direct comparison in the summary chart).
ASN_ONLY_BASELINES = [
    ("AS7922 -> AS396982", 0.016, 0.978),
    ("AS7922 -> AS20473", -0.100, 0.892),
    ("AS7922 -> AS202422", -0.001, 0.867),
    ("AS3320 -> AS20473", 0.022, 1.004),
    ("AS12322 -> AS20473", 0.052, 1.087),
]


def load_info(probes_file: Path):
    """Return dict ip -> (lat, lon, asn_v4, country_code)."""
    with open(probes_file) as f:
        probes = ujson.load(f)
    info = {}
    for p in probes:
        ip = p.get("address_v4")
        asn = p.get("asn_v4")
        country = p.get("country_code")
        geom = p.get("geometry") or {}
        c = geom.get("coordinates")
        if ip is None or asn is None or country is None or c is None:
            continue
        lon, lat = c
        info[ip] = (lat, lon, asn, country)
    return info


def fetch_all(info, table: str, max_rtt_ms: float):
    ch = Clickhouse(
        host=CLICKHOUSE_HOST, database=CLICKHOUSE_DB,
        user=CLICKHOUSE_USER, password=CLICKHOUSE_PASSWORD,
    )
    query = f"""
        WITH arrayMin(groupArray(`min`)) AS min_rtt
        SELECT IPv4NumToString(dst), IPv4NumToString(src), min_rtt
        FROM {CLICKHOUSE_DB}.{table}
        WHERE `min` > 0 AND `min` < {max_rtt_ms} AND dst != src
        GROUP BY dst, src
    """
    dsts, srcs, rtts = [], [], []
    for dst, src, rtt in ch.client.execute_iter(query):
        if dst in info and src in info:
            dsts.append(dst); srcs.append(src); rtts.append(rtt)
    ch.client.disconnect()

    rtts = np.asarray(rtts, dtype=np.float64)
    src_lat = np.array([info[s][0] for s in srcs])
    src_lon = np.array([info[s][1] for s in srcs])
    dst_lat = np.array([info[d][0] for d in dsts])
    dst_lon = np.array([info[d][1] for d in dsts])
    dist = haversine_vec(src_lat, src_lon, dst_lat, dst_lon)
    src_asn = np.array([info[s][2] for s in srcs], dtype=np.int64)
    dst_asn = np.array([info[d][2] for d in dsts], dtype=np.int64)
    src_cc = np.array([info[s][3] for s in srcs])
    dst_cc = np.array([info[d][3] for d in dsts])
    dsts_arr = np.asarray(dsts)
    return rtts, dist, dsts_arr, src_asn, dst_asn, src_cc, dst_cc


def top_asn_pairs(src_asn, dst_asn, k, min_count):
    counter = Counter(zip(src_asn.tolist(), dst_asn.tolist()))
    out = []
    for (pa, aa), c in counter.most_common():
        if c < min_count:
            break
        out.append((int(pa), int(aa), int(c)))
        if len(out) >= k:
            break
    return out


def top_country_pairs(src_cc, dst_cc, mask, k, min_count):
    counter = Counter(zip(src_cc[mask].tolist(), dst_cc[mask].tolist()))
    out = []
    for (pc, ac), c in counter.most_common():
        if c < min_count:
            break
        out.append((pc, ac, int(c)))
        if len(out) >= k:
            break
    return out


def run_slice(rtt, dist, dsts, out_dir, n_anchors=5):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(rtt)
    n_unique_anchors = int(len(np.unique(dsts)))
    n_bins = max(8, min(40, n // 200))
    min_per_bin = max(5, n // (n_bins * 5))
    try:
        p_mu, p_sigma, _, _, _ = fit_mu_sigma(
            rtt, dist, n_bins=n_bins, min_per_bin=min_per_bin
        )
    except (np.linalg.LinAlgError, ValueError) as e:
        return {"n_pairs": n, "n_anchors": n_unique_anchors,
                "mu_z": None, "sigma_z": None, "error": str(e)}

    plot_panel_a(rtt, dist, p_mu, p_sigma, out_dir / "fig3a_scatter.png")
    z = standardize(rtt, dist, p_mu, p_sigma)
    mu_z, sigma_z = plot_panel_b(z, out_dir / "fig3b_standardized.png")
    if n_unique_anchors >= 2:
        plot_panel_c(rtt, dist, dsts, p_mu, p_sigma,
                     out_dir / "fig3c_qq.png",
                     n_anchors=min(n_anchors, n_unique_anchors))
    return {
        "n_pairs": n, "n_anchors": n_unique_anchors, "n_bins": n_bins,
        "mu_z": float(mu_z), "sigma_z": float(sigma_z),
    }


def plot_summary(all_results, out_path):
    """Bar chart grouped by ASN pair, showing ASN-only vs country-restricted sigma_z."""
    labels = [BASELINE_UNSLICED["label"], BASELINE_MESHED["label"]]
    sigmas = [BASELINE_UNSLICED["sigma_z"], BASELINE_MESHED["sigma_z"]]
    mus = [BASELINE_UNSLICED["mu_z"], BASELINE_MESHED["mu_z"]]
    colors = ["#666", "#999"]

    for entry in all_results:
        asn_label = f"AS{entry['probe_asn']}\n->AS{entry['anchor_asn']}"
        labels.append(asn_label + f"\n(ASN only)")
        sigmas.append(entry["asn_only_sigma_z"])
        mus.append(entry["asn_only_mu_z"])
        colors.append("#cccccc")
        for sub in entry["country_slices"]:
            if sub["sigma_z"] is None:
                continue
            labels.append(
                f"{asn_label}\n{sub['probe_country']}->{sub['anchor_country']}\n(n={sub['n_pairs']/1000:.1f}k)"
            )
            sigmas.append(sub["sigma_z"])
            mus.append(sub["mu_z"])
            colors.append("steelblue")

    fig, ax = plt.subplots(figsize=(max(12, 0.9 * len(labels)), 6))
    x = np.arange(len(labels))
    ax.bar(x, sigmas, color=colors)
    ax.axhline(1.0, ls="--", color="red", lw=1.5,
               label=r"Spotter target $\sigma_z=1$")
    ax.axhline(BASELINE_UNSLICED["sigma_z"], ls=":", color="gray", lw=1,
               label=f"unsliced ({BASELINE_UNSLICED['sigma_z']:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_ylabel(r"Pooled standardized $\sigma_z$")
    y_lo = min(0.78, min(s for s in sigmas if s is not None) - 0.02)
    y_hi = max(1.15, max(s for s in sigmas if s is not None) + 0.04)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(r"$\sigma_z$ by constraint tightness: baseline -> ASN -> ASN+country")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    for i, (s, m) in enumerate(zip(sigmas, mus)):
        ax.text(i, s + 0.005, f"$\\sigma$={s:.3f}", ha="center", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="ping_10k_to_anchors")
    parser.add_argument("--max-rtt", type=float, default=200.0)
    parser.add_argument("--top-asn", type=int, default=5)
    parser.add_argument("--top-country", type=int, default=2,
                        help="Top country pairs to slice within each ASN pair.")
    parser.add_argument("--min-asn-count", type=int, default=2000)
    parser.add_argument("--min-country-count", type=int, default=500)
    parser.add_argument("--n-anchors", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--probes-file", type=Path, default=PROBES_FILE)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading coords + ASN + country from {args.probes_file}")
    info = load_info(args.probes_file)
    print(f"      {len(info)} IPs with full attribution")

    print(f"[2/5] Querying {args.table}")
    rtt, dist, dsts, src_asn, dst_asn, src_cc, dst_cc = fetch_all(
        info, args.table, args.max_rtt
    )
    print(f"      {len(rtt)} pairs")

    print(f"[3/5] Top {args.top_asn} ASN pairs")
    asn_pairs = top_asn_pairs(src_asn, dst_asn, args.top_asn, args.min_asn_count)
    for pa, aa, c in asn_pairs:
        print(f"      AS{pa:>6} -> AS{aa:>6}: {c} pairs")

    print(f"[4/5] Sub-slicing each ASN pair by top {args.top_country} country pairs")
    all_results = []
    for pa, aa, asn_count in asn_pairs:
        mask_asn = (src_asn == pa) & (dst_asn == aa)
        # Fit ASN-only slice for direct comparison.
        asn_dir = args.out_dir / f"as{pa}_to_as{aa}" / "asn_only"
        asn_only = run_slice(rtt[mask_asn], dist[mask_asn], dsts[mask_asn],
                             asn_dir, n_anchors=args.n_anchors)
        country_pairs = top_country_pairs(
            src_cc, dst_cc, mask_asn,
            k=args.top_country, min_count=args.min_country_count,
        )
        print(f"      AS{pa}->AS{aa} (ASN sigma_z={asn_only['sigma_z']:.3f}):")
        country_results = []
        for pc, ac, cc in country_pairs:
            mask = mask_asn & (src_cc == pc) & (dst_cc == ac)
            sub_dir = args.out_dir / f"as{pa}_to_as{aa}" / f"{pc}_to_{ac}"
            r = run_slice(rtt[mask], dist[mask], dsts[mask],
                          sub_dir, n_anchors=args.n_anchors)
            r.update({"probe_country": pc, "anchor_country": ac})
            country_results.append(r)
            if r["sigma_z"] is not None:
                print(f"        {pc}->{ac}: n={r['n_pairs']}, anchors={r['n_anchors']}, "
                      f"mu_z={r['mu_z']:+.3f}, sigma_z={r['sigma_z']:.3f}")
            else:
                print(f"        {pc}->{ac}: fit failed ({r.get('error')})")
        all_results.append({
            "probe_asn": pa,
            "anchor_asn": aa,
            "asn_only_mu_z": asn_only["mu_z"],
            "asn_only_sigma_z": asn_only["sigma_z"],
            "asn_only_n_pairs": asn_only["n_pairs"],
            "country_slices": country_results,
        })

    print(f"[5/5] Summary -> {args.out_dir}")
    plot_summary(all_results, args.out_dir / "summary_sigma_z.png")
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump({
            "table": args.table,
            "max_rtt_ms": args.max_rtt,
            "baselines": {
                "all_probes_to_all_anchors": BASELINE_UNSLICED,
                "anchors_meshed": BASELINE_MESHED,
            },
            "asn_pairs": all_results,
        }, f, indent=2)

    print("\n=== sigma_z by constraint tightness ===")
    print(f"  unsliced:                       sigma_z={BASELINE_UNSLICED['sigma_z']:.3f}")
    print(f"  anchors -> anchors (meshed):    sigma_z={BASELINE_MESHED['sigma_z']:.3f}")
    for entry in all_results:
        print(f"  AS{entry['probe_asn']}->AS{entry['anchor_asn']} (ASN only):  "
              f"sigma_z={entry['asn_only_sigma_z']:.3f}")
        for sub in entry["country_slices"]:
            if sub["sigma_z"] is not None:
                print(f"    +{sub['probe_country']}->{sub['anchor_country']}: "
                      f"sigma_z={sub['sigma_z']:.3f}  mu_z={sub['mu_z']:+.3f}  "
                      f"(n={sub['n_pairs']}, anchors={sub['n_anchors']})")


if __name__ == "__main__":
    main()
