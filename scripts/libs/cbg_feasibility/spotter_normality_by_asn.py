"""Test Spotter's normality claim under (probe-ASN, anchor-ASN) slicing.

Hypothesis: Spotter's "landmark-independent normal" reflects infrastructure
uniformity, not Internet physics. If true, constraining both endpoints to a
single ASN pair (uniform last-mile + uniform anchor connectivity) should
recover the normal distribution that fails on the unsliced probes -> anchors
case.

Pipeline per slice (top-K joint ASN pairs by sample count):
  1. Filter all (src, dst) pairs to one (probe-ASN, anchor-ASN).
  2. Re-fit mu(d), sigma(d) on that slice only.
  3. Emit the same three Spotter figures (scatter, standardized histogram, Q-Q).

Output: per-slice figure subdirs + summary_sigma_z.png comparing pooled
sigma_z across constraint levels (unsliced / anchors-meshed / per-ASN-pair).

Run:
    python -m scripts.libs.cbg_feasibility.spotter_normality_by_asn
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
DEFAULT_OUT_DIR = Path("scripts/libs/cbg_feasibility/outputs/spotter_normality_by_asn")

# Measured baselines from spotter_normality_check.py
BASELINE_UNSLICED = {"mu_z": 0.027, "sigma_z": 0.894, "label": "all probes\n-> all anchors"}
BASELINE_MESHED = {"mu_z": 0.071, "sigma_z": 0.964, "label": "anchors -> anchors\n(meshed)"}


def load_info(probes_file: Path):
    """Return dict ip -> (lat, lon, asn_v4)."""
    with open(probes_file) as f:
        probes = ujson.load(f)
    info = {}
    for p in probes:
        ip = p.get("address_v4")
        asn = p.get("asn_v4")
        geom = p.get("geometry") or {}
        c = geom.get("coordinates")
        if ip is None or asn is None or c is None:
            continue
        lon, lat = c
        info[ip] = (lat, lon, asn)
    return info


def fetch_all(info, table: str, max_rtt_ms: float):
    """Pull all (src, dst, min_rtt), annotate with coords + ASN, compute distance."""
    ch = Clickhouse(
        host=CLICKHOUSE_HOST,
        database=CLICKHOUSE_DB,
        user=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
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
            dsts.append(dst)
            srcs.append(src)
            rtts.append(rtt)
    ch.client.disconnect()

    rtts = np.asarray(rtts, dtype=np.float64)
    src_lat = np.array([info[s][0] for s in srcs])
    src_lon = np.array([info[s][1] for s in srcs])
    dst_lat = np.array([info[d][0] for d in dsts])
    dst_lon = np.array([info[d][1] for d in dsts])
    dist = haversine_vec(src_lat, src_lon, dst_lat, dst_lon)
    src_asn = np.array([info[s][2] for s in srcs], dtype=np.int64)
    dst_asn = np.array([info[d][2] for d in dsts], dtype=np.int64)
    dsts_arr = np.asarray(dsts)
    return rtts, dist, dsts_arr, src_asn, dst_asn


def top_asn_pairs(src_asn, dst_asn, k=5, min_count=2000):
    """Top-K (probe-ASN, anchor-ASN) pairs by sample count, with min threshold."""
    counter = Counter(zip(src_asn.tolist(), dst_asn.tolist()))
    out = []
    for (pa, aa), c in counter.most_common():
        if c < min_count:
            break
        out.append((int(pa), int(aa), int(c)))
        if len(out) >= k:
            break
    return out


def run_slice(rtt, dist, dsts, src_asn, dst_asn, probe_asn, anchor_asn,
              out_dir, n_anchors=5):
    """Run Spotter pipeline on one ASN-pair slice."""
    mask = (src_asn == probe_asn) & (dst_asn == anchor_asn)
    rtt_s = rtt[mask]
    dist_s = dist[mask]
    dsts_s = dsts[mask]
    n = int(mask.sum())
    n_unique_anchors = int(len(np.unique(dsts_s)))

    out_dir.mkdir(parents=True, exist_ok=True)

    # Adaptive bin count: fewer bins on smaller slices.
    n_bins = max(8, min(40, n // 200))
    min_per_bin = max(5, n // (n_bins * 5))

    try:
        p_mu, p_sigma, _, _, _ = fit_mu_sigma(
            rtt_s, dist_s, n_bins=n_bins, min_per_bin=min_per_bin
        )
    except (np.linalg.LinAlgError, ValueError) as e:
        return {
            "probe_asn": probe_asn, "anchor_asn": anchor_asn,
            "n_pairs": n, "n_anchors": n_unique_anchors,
            "mu_z": None, "sigma_z": None, "error": str(e),
        }

    plot_panel_a(rtt_s, dist_s, p_mu, p_sigma, out_dir / "fig3a_scatter.png")
    z = standardize(rtt_s, dist_s, p_mu, p_sigma)
    mu_z, sigma_z = plot_panel_b(z, out_dir / "fig3b_standardized.png")
    if n_unique_anchors >= 2:
        plot_panel_c(
            rtt_s, dist_s, dsts_s, p_mu, p_sigma,
            out_dir / "fig3c_qq.png",
            n_anchors=min(n_anchors, n_unique_anchors),
        )

    return {
        "probe_asn": probe_asn,
        "anchor_asn": anchor_asn,
        "n_pairs": n,
        "n_anchors": n_unique_anchors,
        "n_bins": n_bins,
        "mu_z": float(mu_z),
        "sigma_z": float(sigma_z),
    }


def plot_summary(results, out_path):
    """Bar chart: sigma_z across slice constraints vs baselines."""
    labels = [BASELINE_UNSLICED["label"], BASELINE_MESHED["label"]]
    sigmas = [BASELINE_UNSLICED["sigma_z"], BASELINE_MESHED["sigma_z"]]
    mus = [BASELINE_UNSLICED["mu_z"], BASELINE_MESHED["mu_z"]]
    colors = ["#888", "#bbb"]

    for r in results:
        if r["sigma_z"] is None:
            continue
        labels.append(
            f"AS{r['probe_asn']}\n-> AS{r['anchor_asn']}\n(n={r['n_pairs']/1000:.0f}k)"
        )
        sigmas.append(r["sigma_z"])
        mus.append(r["mu_z"])
        colors.append("steelblue")

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(labels)), 5))
    x = np.arange(len(labels))
    ax.bar(x, sigmas, color=colors)
    ax.axhline(1.0, ls="--", color="red", lw=1.5,
               label=r"Spotter target $\sigma_z = 1$")
    ax.axhline(BASELINE_UNSLICED["sigma_z"], ls=":", color="gray", lw=1,
               label=f"unsliced baseline ({BASELINE_UNSLICED['sigma_z']:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"Pooled standardized $\sigma_z$")
    ax.set_ylim(0.8, 1.1)
    ax.set_title(r"$\sigma_z$ across slice constraints "
                 r"(closer to 1.0 = better fit to Spotter's normal)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    for i, (s, m) in enumerate(zip(sigmas, mus)):
        ax.text(i, s + 0.005, f"$\\sigma$={s:.3f}\n$\\mu$={m:+.3f}",
                ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="ping_10k_to_anchors")
    parser.add_argument("--max-rtt", type=float, default=200.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=2000,
                        help="Minimum pairs per (probe-ASN, anchor-ASN) slice.")
    parser.add_argument("--n-anchors", type=int, default=5,
                        help="Max anchors per slice's Q-Q panel.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--probes-file", type=Path, default=PROBES_FILE)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading coords + ASN from {args.probes_file}")
    info = load_info(args.probes_file)
    print(f"      {len(info)} IPs with coords + ASN")

    print(f"[2/5] Querying {args.table} (0 < min RTT < {args.max_rtt} ms)")
    rtt, dist, dsts, src_asn, dst_asn = fetch_all(info, args.table, args.max_rtt)
    print(f"      {len(rtt)} pairs")

    print(f"[3/5] Picking top {args.top_k} (probe-ASN, anchor-ASN) pairs")
    pairs = top_asn_pairs(src_asn, dst_asn, k=args.top_k, min_count=args.min_count)
    if not pairs:
        print(f"      No ASN pairs with >= {args.min_count} samples. Lower --min-count.")
        return
    for pa, aa, c in pairs:
        print(f"      AS{pa:>6} -> AS{aa:>6}: {c} pairs")

    print("[4/5] Running per-slice pipelines")
    results = []
    for pa, aa, c in pairs:
        slice_dir = args.out_dir / f"as{pa}_to_as{aa}"
        r = run_slice(rtt, dist, dsts, src_asn, dst_asn, pa, aa,
                      slice_dir, n_anchors=args.n_anchors)
        results.append(r)
        if r["sigma_z"] is not None:
            print(f"      AS{pa:>6} -> AS{aa:>6}: n={r['n_pairs']}, "
                  f"anchors={r['n_anchors']}, bins={r['n_bins']}, "
                  f"mu_z={r['mu_z']:+.3f}, sigma_z={r['sigma_z']:.3f}")
        else:
            print(f"      AS{pa:>6} -> AS{aa:>6}: fit failed ({r.get('error')})")

    print(f"[5/5] Writing summary to {args.out_dir}")
    plot_summary(results, args.out_dir / "summary_sigma_z.png")
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "table": args.table,
                "max_rtt_ms": args.max_rtt,
                "baselines": {
                    "all_probes_to_all_anchors": BASELINE_UNSLICED,
                    "anchors_meshed": BASELINE_MESHED,
                },
                "slices": results,
            },
            f, indent=2,
        )

    print("\n=== sigma_z by slice constraint ===")
    print(f"  unsliced (probes -> anchors):   mu_z={BASELINE_UNSLICED['mu_z']:+.3f}  "
          f"sigma_z={BASELINE_UNSLICED['sigma_z']:.3f}")
    print(f"  anchors -> anchors (meshed):    mu_z={BASELINE_MESHED['mu_z']:+.3f}  "
          f"sigma_z={BASELINE_MESHED['sigma_z']:.3f}")
    for r in results:
        if r["sigma_z"] is not None:
            print(f"  AS{r['probe_asn']:>6} -> AS{r['anchor_asn']:>6}: "
                  f"mu_z={r['mu_z']:+.3f}  sigma_z={r['sigma_z']:.3f}  "
                  f"(n={r['n_pairs']}, anchors={r['n_anchors']})")


if __name__ == "__main__":
    main()
