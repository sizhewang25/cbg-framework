# Public CBG-Related Datasets — Download Log & Characterization

Date: 2026-05-15
Companion to: [2026-05-15-cbg-related-work-datasets.md](./2026-05-15-cbg-related-work-datasets.md)

All downloads landed under [`datasets/external_2026-05-15/`](../datasets/external_2026-05-15/). Total fresh footprint: **718 MB** (RIPE Atlas live 53 MB, DB-IP Lite 85 MB, TrustGeo NYC/LA/Shanghai 580 MB, TMA-2024 vp-ls 65 KB).

---

## What was downloaded

| Dataset | Source | Size | Status |
|---|---|---|---|
| RIPE Atlas anchors (live) | `https://atlas.ripe.net/api/v2/anchors/` | 1.2 MB | ✅ 1,744 anchors |
| RIPE Atlas probes (live) | `https://atlas.ripe.net/api/v2/probes/` | 53 MB | ✅ 58,678 lifetime records (14,299 currently connected) |
| DB-IP Lite (city, 2026-05) | `https://download.db-ip.com/free/dbip-city-lite-2026-05.csv.gz` | 85 MB gz | ✅ 8.06 M rows (3.69 M IPv4 ranges, 4.37 M IPv6) |
| TrustGeo NYC / LA / Shanghai | `github.com/ICDM-UESTC/TrustGeo` (3 ZIPs) | 580 MB unzipped | ✅ 310,870 IPs with VP RTTs + traceroutes |
| Cho TMA-2024 vp-ls | raw URLs on `github.com/grace71/tma24-vp-ls` | 65 KB | ✅ 783 anchors + ISO-3166 lookup (Python scripts skipped to keep download to data only) — **NO RTT, pass** |

Already present in repo (Sept-2023 snapshots, not re-downloaded): MaxMind GeoLite2-City 2023-05-16, Verfploeter hitlist `it102w-20230125`, BGP prefixes, IP-to-ASN `2022-03-28.dat`, IPInfo anchor ground truth, MaxMind anchor ground truth.

---

## Per-dataset characterization

### 1. RIPE Atlas — live anchors (1,744)

| Field | Value |
|---|---|
| Total anchors | 1,744 |
| With IPv4 | 1,744 (100%) |
| With IPv6 | 1,590 (91.2%) |
| Distinct countries | 126 |
| Distinct IPv4 ASes | 1,129 |
| Top-5 countries | US (273), DE (198), NL (104), FR (85), GB (71) |
| Top-5 ASes by anchor count | AS12008 (27), AS15169/Google (20), AS199524/GCore (20), AS16509/AWS (18), AS20473/Choopa (18) |

Object schema (selected keys): `id`, `ip_v4`, `ip_v6`, `as_v4`, `as_v6`, `country`, `city`, `company`, `geometry`, `fqdn`, `date_live`, `date_decommissioned`.

### 2. RIPE Atlas — live probes (14,299 connected of 58,678 lifetime)

| Status bucket | Count |
|---|---|
| Connected | **14,299** |
| Disconnected | 2,329 |
| Abandoned | 24,723 |
| Never Connected | 8,998 |
| Written Off | 8,329 |

Connected-only stats:
- Distinct countries: **183**
- Distinct IPv4 ASes: **4,447**
- IPv6 coverage: 7,851 / 14,299 (**54.9%**)
- Top-5 countries: US (2,257), DE (1,923), FR (1,082), NL (705), GB (651)
- Top-5 ASes: AS3320/DTAG (373), AS12322/Free (314), AS7922/Comcast (247), AS3215/Orange (234), AS3209/Vodafone (193)
- 998 of the 14,299 connected probes are also anchors

### 3. DB-IP Lite (free city DB, 2026-05)

| Field | Value |
|---|---|
| Total rows | 8,061,679 |
| IPv4 ranges | 3,687,766 |
| IPv6 ranges | 4,373,913 |
| IPv4 ranges with non-empty city field | **100.0%** |
| Distinct ISO country codes | 251 |
| Top-5 by # IPv4 ranges | US (2.99 M), DE (526 K), BR (511 K), GB (476 K), FR (303 K) |

CSV columns: `start_ip, end_ip, continent, country, region, city, lat, lon` (no AS, no ISP at the free tier).

### 4. Cho TMA-2024 vp-ls anchor list — **NO RTT, pass**

> Remarks: anchor selection metadata only (per-anchor reachability counts, no RTTs/traceroutes). Cannot be used as a benchmark input on its own — pass for CBG evaluation.


| Field | Value |
|---|---|
| Anchors listed (`anchorSelectionAll.csv`) | **783** (paper text says 780, file is 3 entries larger) |
| Schema | `addr, aid, pid, longitude, latitude, city, country, anchors p, probes p, asn` |
| Distinct countries | 96 |
| Distinct ASes | 536 |
| `anchors p` range | 368 – 15,213 (per-anchor probe-target count at snapshot time) |
| `probes p` range | 1 – 7,580 |
| Top-5 countries | US (120), DE (99), FR (44), NL (41), GB (37) |
| Top-5 ASes | AS12008/NeuStar (26), AS396982/Google Cloud (20), AS20473/Choopa (18), AS16509/AWS (15), AS202422/G-Core (15) |

Cross-checks:
- **97.8% of TMA-2024 anchor IPs are still alive in the May 2026 RIPE Atlas anchor list** (766/783 stable) — reinforces the anchor-population stability finding.
- 648 of the 783 IPs also appear in the IMC-2023 repo anchor snapshot — the three datasets (IMC 2023 / TMA 2024 / live 2026) overlap on a ~648-anchor core.

The accompanying `iso3166.csv` (260 rows) is just an ISO 3166-1 alpha-3 ↔ alpha-2 ↔ country-name lookup table — auxiliary, not measurement data.

### 5. TrustGeo street-level (NYC / LA / Shanghai, KDD-2022 origin)

| City | Targets | Lat range | Lon range | VPs per target | Notes |
|---|---|---|---|---|---|
| Los Angeles | 92,804 | 33.33 – 34.75 | -118.87 – -118.00 | 4 (vp900-903) | All labeled `Los Angeles, America` |
| New York | 91,809 | 40.52 – 40.88 | -74.19 – -73.70 | 4 (vp900-903) | All labeled `New York, America` |
| Shanghai | 126,257 | 30.70 – 31.86 | 120.92 – 121.98 | 1 + 3 (`aiwen`, vp806/808/813) | Includes ISP, port-scan, accuracy=`Street-level` from Aiwen DB |

Per-target fields: `ip`, lat/lon, AS, ISP, city, plus for each VP: ping delay, traceroute (JSON list of {router_ip: delay}), step count, last-router delay, total delay. Shanghai records additionally include port-alive flags for 12 well-known ports — useful for service-discovery-style geolocation.

Bounding boxes confirm the data is metro-area scoped (no rural CA / upstate NY / inland Yangtze targets). VP RTTs include zero-/null-delay records, which downstream models filter or impute.

---

## Cross-dataset comparisons

### A. RIPE Atlas churn 2023 → 2026 (matched by IPv4)

| Subset | 2023 (in repo) | 2026 (live API) | Δ |
|---|---|---|---|
| Anchors with IPv4 | 723 | 1,638 | **+126.6%** (1,021 net) |
| — stable IPs | — | 708 | (98% of 2023 anchors still present) |
| — gone since 2023 | — | 15 | (decommissioned or re-IP'd) |
| — new since 2023 | — | 930 | |
| Anchor countries | 94 | 126 | +32 |
| Probes (connected, IPv4) | 10,135 (any status) | 12,749 | +25.8% |
| — stable IPs | — | 2,187 | (only 22% of 2023 probes still alive at same IP) |
| — gone since 2023 | — | 7,948 | |
| — new since 2023 | — | 10,562 | |

**Takeaway:** Anchor population is very stable (the *infrastructure* hosts persist), but probe population is highly churning — three years of probe data must be re-validated, not assumed stable. This matters for Hu-style "select-K-closest-VP" algorithms that depend on persistent VP–target geometries.

### B. DB-IP Lite accuracy vs RIPE Atlas anchor ground truth (n = 1,744)

| Metric | Value |
|---|---|
| Country match | **92.7%** (1,616 / 1,744) |
| Median geographic error | **7.06 km** |
| p75 | 112 km |
| p90 | 912 km |
| Fraction < 40 km | 69.0% |
| Fraction < 100 km | 74.1% |
| Fraction < 1,000 km | 90.9% |

The country-level accuracy matches what Nur (BalkanCom 2023) reported for free databases; the 7 km median is suspiciously good and likely reflects that **anchors are concentrated in well-known hosting facilities (AWS, DTAG, Google, GCore, Choopa)** that all commercial DBs ground-truth aggressively. The tail (~10% with >1,000 km error) is the load-bearing population for CBG to outperform DBs on.

### C. DB-IP Lite accuracy vs TrustGeo street-level GT

| City | IPs | DB-IP returns target city | Median error (km) | p90 | < 40 km | < 100 km |
|---|---|---|---|---|---|---|
| Los Angeles | 92,804 | 35.0% | 9.70 | 1,340 | 71.5% | 83.7% |
| New York | 91,809 | 68.5% | 10.03 | 255 | 87.7% | 88.6% |
| Shanghai | 126,257 | 73.3% | 18.34 | 52 | 85.4% | 92.2% |

Observations:
- **LA p90 (1,340 km)** is dramatic: DB-IP places ~10% of LA targets on the East Coast or further. This is the classic "registered-to-HQ" failure mode: ASes like Level 3/3356 (which dominates LA's data.csv) are commonly geolocated to AS headquarters.
- **NYC has the best low-error performance** (87.7% < 40 km) but a worse tail than Shanghai — DB-IP frequently confuses NYC metro with Newark/Long Island/Boston cluster.
- **Shanghai's tail is the tightest** (p99 = 1,085 km, p90 = only 52 km). Hypothesis: Chinese ISP prefixes are tightly bound to provincial blocks in DB-IP's source data.
- City-string match rate (35–73%) is much lower than the lat/lon < 40 km rate (72–88%), because DB-IP sometimes picks a nearby suburb (e.g. "Long Beach", "Brooklyn") that's <40 km from the metro centroid but doesn't match the city *string* the dataset uses.

---

## Coverage gaps / what would be worth adding next

Everything below requires sign-up or DUA, so was not auto-downloadable:

| Source | Why add it | Cost |
|---|---|---|
| **MaxMind GeoLite2 (current month)** | Repo only has 2023-05-16; 3-year drift matters | Free, license key |
| **IP2Location LITE DB11 (current)** | Adds 4th DB to compare alongside Nur's evaluation | Free, account |
| **IPInfo Lite (current month)** | Adds 5th DB; repo only has anchor-subset snapshot | Free token |
| **ISI/ANT Verfploeter hitlist (current)** | Repo has it102w-2023-01; modern hitlist evolves rapidly | DUA via ant.isi.edu |
| **CAIDA Ark IPv4 routed /24 topology (1 month)** | Provides modern Alidade-style topology input | Restricted; >1 yr old is public |
| **RouteViews / RIPE RIS BGP table dump** | Repo BGP snapshot is 2023; recent dumps trivial via FTP | Free |
| **Cho et al. TMA-24 VP-LS targets/code** | Needed to reproduce the landmark-selection benchmark | Free git clone (blocked by sandbox here) |

---

## Reproducibility notes

All download commands documented in this note are deterministic from public endpoints, except for the date-bound DB-IP filename. To rebuild from scratch:

```bash
mkdir -p datasets/external_2026-05-15/{ripe_atlas_live,trustgeo,db_ip_lite}

# RIPE Atlas (paginated)
python3 -c "..."  # see commands above

# DB-IP Lite
curl -fsSL -o datasets/external_2026-05-15/db_ip_lite/dbip-city-lite.csv.gz \
  "https://download.db-ip.com/free/dbip-city-lite-$(date +%Y-%m).csv.gz"

# TrustGeo
git clone --depth 1 https://github.com/ICDM-UESTC/TrustGeo.git \
  datasets/external_2026-05-15/trustgeo
cd datasets/external_2026-05-15/trustgeo/datasets && \
  for f in *.zip; do unzip -q -o "$f"; done
```

The characterization scripts (run inline above) are self-contained and depend only on Python 3 stdlib + `bisect`, `csv`, `gzip`, `json`, `ipaddress`, `math`, `statistics`, `collections`.
