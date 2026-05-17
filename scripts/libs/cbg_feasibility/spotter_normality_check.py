"""Spotter (Laki et al. 2011) normality check on the ping_10k_to_anchors dataset.

Reproduces the three panels of Spotter's Figure 3 on our RIPE Atlas data:
  (a) Delay-distance scatter with fitted mu(d) and mu(d) +/- sigma(d).
  (b) Histogram of standardized distances vs N(0, 1).
  (c) Q-Q plot of several anchors' standardized values vs the overall standardized values.

Spotter's PlanetLab fit on ~40k points reported mu = -0.078, sigma = 1.035.
We pool min-RTT per (src, dst) pairs from ping_10k_to_anchors (anchors = landmarks,
probes = additional known-location sources) and check whether the same single normal
distribution describes our standardized delay-distance data.

Run:
    python -m scripts.libs.cbg_feasibility.spotter_normality_check
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ujson
from scipy.stats import norm

from default import (
    CLICKHOUSE_DB,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_USER,
)
from scripts.utils.clickhouse import Clickhouse

PROBES_FILE = Path(
    "datasets/reproducibility_datasets/atlas/reproducibility_probes_and_anchors.json"
)
DEFAULT_OUT_DIR = Path("scripts/libs/cbg_feasibility/outputs/spotter_normality")


def load_coords(probes_file: Path) -> dict[str, tuple[float, float]]:
    """Return dict ip -> (lat, lon)."""
    with open(probes_file) as f:
        probes = ujson.load(f)
    coords = {}
    for p in probes:
        ip = p.get("address_v4")
        geom = p.get("geometry") or {}
        c = geom.get("coordinates")
        if ip is None or c is None:
            continue
        lon, lat = c
        coords[ip] = (lat, lon)
    return coords


def haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in km (matches scripts.utils.helpers.haversine)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6367.0 * 2.0 * np.arcsin(np.sqrt(a))


def fetch_rtt_distance(coords, table: str, max_rtt_ms: float = 200.0):
    """Query min RTT per (src, dst) and compute great-circle distance vectorized."""
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
        if dst in coords and src in coords:
            dsts.append(dst)
            srcs.append(src)
            rtts.append(rtt)
    ch.client.disconnect()

    rtts = np.asarray(rtts, dtype=np.float64)
    dst_lat = np.array([coords[d][0] for d in dsts])
    dst_lon = np.array([coords[d][1] for d in dsts])
    src_lat = np.array([coords[s][0] for s in srcs])
    src_lon = np.array([coords[s][1] for s in srcs])
    dist_km = haversine_vec(src_lat, src_lon, dst_lat, dst_lon)
    return rtts, dist_km, np.asarray(dsts)


def fit_mu_sigma(rtt, dist, n_bins=40, min_per_bin=30, deg_mu=3, deg_sigma=2):
    """Bin by RTT, compute per-bin mean/std of distance, fit polynomials."""
    edges = np.linspace(rtt.min(), rtt.max(), n_bins + 1)
    centers, mus, sigmas = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (rtt >= lo) & (rtt < hi)
        if mask.sum() < min_per_bin:
            continue
        centers.append(0.5 * (lo + hi))
        mus.append(np.mean(dist[mask]))
        sigmas.append(np.std(dist[mask]))
    centers = np.array(centers)
    mus = np.array(mus)
    sigmas = np.array(sigmas)
    p_mu = np.polyfit(centers, mus, deg_mu)
    p_sigma = np.polyfit(centers, sigmas, deg_sigma)
    return p_mu, p_sigma, centers, mus, sigmas


def standardize(rtt, dist, p_mu, p_sigma):
    mu = np.polyval(p_mu, rtt)
    sig = np.polyval(p_sigma, rtt)
    sig = np.where(sig <= 0, np.nan, sig)
    z = (dist - mu) / sig
    return z[np.isfinite(z)]


def plot_panel_a(rtt, dist, p_mu, p_sigma, out_path):
    grid = np.linspace(rtt.min(), rtt.max(), 200)
    mu_curve = np.polyval(p_mu, grid)
    sig_curve = np.polyval(p_sigma, grid)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(rtt, dist / 1000.0, s=2, alpha=0.05, color="green")
    ax.plot(grid, mu_curve / 1000.0, "r-", lw=2, label=r"$\mu(d)$")
    ax.plot(grid, (mu_curve + sig_curve) / 1000.0, "k--", lw=1,
            label=r"$\mu(d) \pm \sigma(d)$")
    ax.plot(grid, (mu_curve - sig_curve) / 1000.0, "k--", lw=1)
    ax.set_xlabel("round trip time [ms]")
    ax.set_ylabel("great circle distance [1000 km]")
    ax.set_title("(a) Delay-distance plot")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_panel_b(z, out_path):
    z_in = z[(z > -4) & (z < 4)]
    mu_z, sigma_z = float(np.mean(z_in)), float(np.std(z_in))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(z, bins=80, range=(-4, 4), density=True, color="green", alpha=0.5,
            label=fr"data ($\mu$={mu_z:+.3f}, $\sigma$={sigma_z:.3f})")
    xs = np.linspace(-4, 4, 400)
    ax.plot(xs, norm.pdf(xs), "r-", lw=2, label=r"$\mathcal{N}(0,1)$")
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\Phi(z)$ standardized probability density")
    ax.set_title("(b) Standardized delay-distance distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return mu_z, sigma_z


def plot_panel_c(rtt, dist, dsts, p_mu, p_sigma, out_path, n_anchors=5):
    """Q-Q plot: observed (per anchor) vs expected (pooled overall)."""
    z_overall = standardize(rtt, dist, p_mu, p_sigma)
    qs = np.linspace(0.005, 0.995, 199)
    expected = np.quantile(z_overall, qs)

    unique, counts = np.unique(dsts, return_counts=True)
    top = unique[np.argsort(counts)[::-1][:n_anchors]]

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.cm.tab10
    for i, anchor in enumerate(top):
        mask = dsts == anchor
        z = standardize(rtt[mask], dist[mask], p_mu, p_sigma)
        if len(z) < 50:
            continue
        observed = np.quantile(z, qs)
        ax.scatter(observed, expected, s=10, alpha=0.7, color=cmap(i),
                   label=f"{anchor} (n={mask.sum()})")
    ax.plot([-3, 3], [-3, 3], "k-", lw=1)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("observed standardized value (per anchor)")
    ax.set_ylabel("expected standardized value (pooled)")
    ax.set_title("(c) Q-Q plot, top-N anchors vs pooled")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=str, default="ping_10k_to_anchors",
                        help="ClickHouse table in geolocation_replication.")
    parser.add_argument("--max-rtt", type=float, default=200.0,
                        help="Cap on min RTT in ms (default 200; Spotter used ~80 on PlanetLab).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help=f"Output dir. Default: {DEFAULT_OUT_DIR}/<table>")
    parser.add_argument("--probes-file", type=Path, default=PROBES_FILE)
    parser.add_argument("--n-bins", type=int, default=40)
    parser.add_argument("--n-anchors", type=int, default=5,
                        help="Number of top anchors for the Q-Q plot.")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT_DIR / args.table
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading coords from {args.probes_file}")
    coords = load_coords(args.probes_file)
    print(f"      {len(coords)} IPs with coordinates")

    print(f"[2/4] Querying {args.table} (0 < min RTT < {args.max_rtt} ms)")
    rtt, dist, dsts = fetch_rtt_distance(coords, table=args.table, max_rtt_ms=args.max_rtt)
    print(f"      {len(rtt)} (src, dst) pairs after geo + RTT filter")
    print(f"      unique anchors (dst): {len(np.unique(dsts))}")

    print("[3/4] Fitting mu(d), sigma(d) over RTT bins")
    p_mu, p_sigma, centers, mus, sigmas = fit_mu_sigma(rtt, dist, n_bins=args.n_bins)
    print(f"      mu(d) poly coeffs (highest deg first):    {p_mu}")
    print(f"      sigma(d) poly coeffs (highest deg first): {p_sigma}")
    print(f"      RTT bins used: {len(centers)}")

    print(f"[4/4] Plotting -> {args.out_dir}")
    plot_panel_a(rtt, dist, p_mu, p_sigma, args.out_dir / "fig3a_scatter.png")
    z = standardize(rtt, dist, p_mu, p_sigma)
    mu_z, sigma_z = plot_panel_b(z, args.out_dir / "fig3b_standardized.png")
    plot_panel_c(rtt, dist, dsts, p_mu, p_sigma,
                 args.out_dir / "fig3c_qq.png", n_anchors=args.n_anchors)

    print("\n=== Spotter normality check ===")
    print(f"Our dataset ({args.table}):  mu = {mu_z:+.4f}   sigma = {sigma_z:.4f}")
    print(f"Spotter (PlanetLab, ~40k pts):  mu = -0.0780       sigma = 1.0350")


if __name__ == "__main__":
    main()
