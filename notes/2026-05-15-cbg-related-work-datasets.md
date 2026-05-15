# Ground-Truth Datasets Across CBG Related Work — Inventory & Availability

Date: 2026-05-15
Scope: Papers cross-referenced for the CBG benchmark paper (`tasks/20260424-cbg-benchmark-paper/related_work.md`).

---

## ✅ Decision (2026-05-15): datasets for the CBG benchmark paper

**Primary evaluation dataset:** the **IMC 2023 RIPE Atlas snapshot** already shipped in this repo
- VPs / measurement substrate: **12,129 probes** from `datasets/reproducibility_datasets/atlas/reproducibility_probes.json`
- Targets / hard ground truth: **723 anchors** from `datasets/reproducibility_datasets/atlas/reproducibility_anchors.json`
- Measurement data: ClickHouse dumps under `datasets/clickhouse_data/` (anchors-meshed pings, probes-to-prefix pings, etc.; ~2 GB compressed)

**Cross-comparison vs commercial databases (anchor subset only):**
- MaxMind GeoLite2-City IPv4, 2023-05-16 snapshot — `datasets/static_datasets/GeoLite2-City-Blocks-IPv4_20230516.tree` and the labeled subset `datasets/static_datasets/maxmind_free_geo_anchors.json`
- IPInfo — `datasets/static_datasets/ip_info_geo_anchors.json` (anchor subset)

**Rationale:**
- Single matched-vintage snapshot (everything from 2023-05) eliminates temporal drift confounds.
- RIPE Atlas anchors are the field's standard hard ground truth; 723 of them give the headline accuracy numbers.
- The commercial-DB comparison is bounded to the anchor subset because the full DB-IP / IP2Location / IPGeolocationIO May-2023 snapshots are not publicly recoverable (probed all standard archives 2026-05-15; none retained the vintage). MaxMind 2023-05 + IPInfo anchor subset are the only matched-vintage commercial DBs we have.
- Live 2026 RIPE Atlas data, DB-IP Lite 2026-05, TMA-2024 anchors, and the TrustGeo street-level corpus stay available under `datasets/external_2026-05-15/` but are **not** part of the primary evaluation — reserved for sensitivity / temporal-drift analysis only.

**Pressure-test mode (memory / speed / runtime, NOT headline accuracy):** swap the roles of anchors and probes — use the **723 anchors as VPs** and the **12,129 probes as targets**. The probes have known coordinates assigned by their hosts, so ground truth stays hard throughout; no synthetic keys, no soft GT. This gives a ~17× scale-up in the target population from 723 → 12,129 while reusing the same measurement data (the existing `ping_10k_to_anchors` table provides the anchor↔probe RTT matrix needed for the swap). Accuracy will likely be worse in this mode (fewer VPs, less spatial coverage), but that's acceptable — the purpose is runtime/memory at scale, not accuracy.

(See companion task plan in `tasks/20260424-cbg-benchmark-paper/`.)

---

Papers covered:
1. Gueye et al. — Original CBG (IMC 2004 / ToN 2006)
2. Wong et al. — Octant (NSDI 2007)
3. Hu et al. — Million-Scale (IMC 2012)
4. Chandrasekaran et al. — Alidade (Duke CS-TR-2015.001)
5. Darwich et al. — IMC 2023 Replication
6. Nur — Accuracy & Coverage of IP Geo DBs (BalkanCom 2023)
7. Wang et al. — NeighborGeo (Computer Networks 2025)
8. Cho et al. — Selection of Landmarks for Efficient Active Geolocation (TMA 2024)

---

## Per-paper datasets

| Paper | Vantage points / Landmarks | Targets / Ground truth | Geolocation DBs used / compared |
|---|---|---|---|
| Gueye — CBG | RIPE TTM hosts (Western Europe + US sub-datasets) | RIPE TTM hosts (self-known coords) | WHOIS, NetGeo, IP2LL (mentioned) |
| Wong — Octant | 51 PlanetLab nodes (NA) + 53 public traceroute servers | 104 targets (PlanetLab + traceroute servers self-reported) | IP2Location, US Census ZIP DB, WHOIS, `undns` |
| Hu — Million-Scale | 400 PlanetLab nodes as VPs | 25 PlanetLab anchors + /24 blocks from ISI IPv4 census/hitlist; Freebox ADSL ground truth | MaxMind (compared) |
| Chandrasekaran — Alidade | iPlane traceroutes on PlanetLab + CAIDA Ark; web-scraped campus addresses | PlanetLab nodes; six GT sets incl. **EuroGT** (24M IPs from an anonymous European Tier-1, 73 city-level locations), NTP (99 IPs), PLAB; rest via Akamai | EdgeScape, MaxMind GeoCity, MaxMind GeoCity2 Lite, DB-IP, IP2Location, HostIP.Info, IPInfoDB; `undns` + WHOIS |
| Darwich — IMC 2023 | RIPE Atlas (~10K probes, ~500 anchors) | RIPE Atlas anchors; Verfploeter (ISI) IPv4 hitlist (`it102w-20230125`); BGP prefixes | MaxMind GeoLite2-City (`20230516`), IPInfo lite, IP-to-ASN (`2022-03-28.dat`); ClickHouse dumps of 7 measurement tables (~2 GB compressed) on `ftp.iris.dioptra.io`, anonymous FTP, no DUA |
| Nur — DB Accuracy (BalkanCom 2023) | DNS-based ground-truth pipeline | 24,810 RIPE Atlas probe IPs + 1,176 RIPE Atlas anchor IPs + 1,746 M-Lab IPs (≈26,573 unique) + 16,586 router interfaces | MaxMind, DBIP, IP2Location, IPGeolocationIO (all four evaluated) |
| Wang — NeighborGeo | Open-source street-level landmark sets (NYC, LA, Shanghai) — same IPLN/GraphGeo benchmark as Li et al. KDD 2022 | Same NYC/LA/Shanghai street-level test IPs | None as GT; MaxMind / IP2Location commonly cited as baselines |
| Cho — TMA 2024 | RIPE Atlas anchors only (780 anchors, Nov 2022) | 559 commercial VPN endpoints | Uses Darwich et al. CBG implementation; no DB comparison |

---

## What is actually common across all eight

No single named dataset is shared by all eight. The intersections are:

1. **Active RTT measurements (ping/traceroute) from landmarks with known coordinates to unknown targets** — foundational data type in all eight; only the platform differs.
2. **A set of fixed-location "landmarks"/"anchors"/"vantage points"** — RIPE TTM, PlanetLab, RIPE Atlas, or custom street-level landmark IPs.
3. **MaxMind (or equivalent commercial DB)** appears as a baseline / comparison in 6 of 8: Hu, Alidade, Darwich, Nur, NeighborGeo (typical baseline), and Octant uses IP2Location in a similar role. Gueye and Cho TMA-2024 are the two without it.

Closest to a "common dataset" across the modern subset (Darwich 2023, Nur 2023, Cho 2024): **RIPE Atlas anchors/probes**.
Closest across the early subset (Octant 2007, Hu 2012, Alidade 2015): **PlanetLab + traceroute-based measurements (iPlane / Ark)**.
Outliers: Gueye (RIPE TTM, no longer alive) and NeighborGeo (street-level supervised ML benchmarks).

---

## Availability matrix (as of 2026-05)

| # | Source | Paper(s) | Status | Where to get it |
|---|---|---|---|---|
| 1 | RIPE TTM (Test Traffic Measurement) | Gueye | ❌ Discontinued 1 Jul 2014 | [Archived project page](https://www.ripe.net/analyse/archived-projects/ttm/) |
| 2 | PlanetLab (Princeton/global) | Octant, Hu, Alidade (via iPlane) | ❌ Shut down May 2020 | [Static archive](https://planetlab.cs.princeton.edu/status.html); some EU forks survive |
| 3 | Public traceroute servers | Octant | ⚠️ Ad-hoc, no archive | Individual servers come/go |
| 4 | ISI/ANT IPv4 hitlist (incl. Verfploeter) | Hu, Darwich | ✅ Available on request | [ant.isi.edu/datasets/ip_hitlists](https://ant.isi.edu/datasets/ip_hitlists/) (IMPACT DUA) |
| 5 | LANDER geolocation datasets (Hu 2012) | Hu | ✅ Available on request | [ant.isi.edu/datasets/geolocation](https://ant.isi.edu/datasets/geolocation/) — 13 /8 blocks |
| 6 | Freebox ADSL ground truth | Hu | ❌ Proprietary, never released | — |
| 7 | iPlane traceroutes | Alidade | ⚠️ Frozen 2006–2016 | [RIPE Labs mirror](https://labs.ripe.net/datarepository/data-sets/iplane-traceroute-dataset/) (download flaky); original at U.Washington |
| 8 | CAIDA Ark / Archipelago | Alidade | ✅ Available | [publicdata.caida.org/datasets/topology/ark/](https://publicdata.caida.org/datasets/topology/ark/) — public >1yr (IPv4) or all (IPv6); restricted via ark-info@caida.org |
| 9 | Akamai EdgeScape commercial GT | Alidade | ❌ Proprietary | Collaboration-only |
| 9a | EuroGT (anonymous European Tier-1 ISP, ~24M IPs, 73 city-level locations, 2013–14 snapshot) | Alidade | ❌ Proprietary, never released | Provider unnamed; would need fresh data-sharing agreement, or use modern Geofeeds (RFC 9092 / RFC 8805) as a substitute |
| 9b | NTP ground-truth set (99 IPs) | Alidade | ⚠️ Reconstructable | Re-derivable from current NTP pool servers' publicly listed locations |
| 10 | Web-scraped campus addresses | Alidade | ❌ Not released | Generated in-paper |
| 11 | RIPE Atlas probes + anchors | Darwich, Nur, Cho | ✅ Active | [atlas.ripe.net](https://atlas.ripe.net/) — Feb 2026: 13,421 probes, 974 anchors |
| 12 | MaxMind GeoLite2 | Hu, Alidade, Darwich, Nur, + baseline elsewhere | ✅ Free download (license-bound) | [maxmind.com/en/geolite-free-ip-geolocation-data](https://www.maxmind.com/en/geolite-free-ip-geolocation-data) |
| 13 | DB-IP, IP2Location, IPGeolocationIO | Nur + baselines | ✅ Free tiers + commercial | [db-ip.com](https://db-ip.com/), [lite.ip2location.com](https://lite.ip2location.com/), [ipgeolocation.io](https://ipgeolocation.io/) |
| 14 | IPInfo ground truth | Darwich | ✅ Free + commercial tiers | [ipinfo.io/lite](https://ipinfo.io/lite) |
| 15 | BGP prefixes (RouteViews / RIS) | Darwich | ✅ Free | RouteViews, RIPE RIS public dumps |
| 16 | M-Lab (NDT, traceroute, sidestream) | Nur | ✅ Fully open | [measurementlab.net/data](https://www.measurementlab.net/data/) — BigQuery + raw GCS |
| 17 | NYC / LA / Shanghai street-level (KDD-2022 GraphGeo origin) | NeighborGeo, GraphGeo, TrustGeo, ExGeo | ✅ Public on GitHub | [github.com/ICDM-UESTC/TrustGeo](https://github.com/ICDM-UESTC/TrustGeo) — 91,808 NYC / 92,804 LA / 126,258 Shanghai IPs (`data.csv`, `ip.csv`, `last_traceroute.csv`) |
| 18 | 559 commercial VPN endpoints (targets) | Cho TMA-2024 | ✅ Public on GitHub | [github.com/grace71/tma24-vp-ls](https://github.com/grace71/tma24-vp-ls) |
| 19 | IP2Location, US Census ZIP, WHOIS, `undns` | Octant, Alidade | ✅ Available | IP2Location free tier; Census public; WHOIS public; `undns` OSS |

### Summary tally

- ✅ Available now: **13 of 21** — RIPE Atlas, MaxMind, DB-IP, IP2Location, IPGeolocationIO, IPInfo, M-Lab, CAIDA Ark, ISI hitlist (on request), ISI LANDER geo (on request), NYC/LA/Shanghai street-level, Cho's VPN targets, BGP/WHOIS/undns.
- ⚠️ Partial / archived / reconstructable: **3** — iPlane (frozen 2016, mirror flaky); public traceroute servers (no curated archive); Alidade NTP set (re-derivable from current NTP pool).
- ❌ Dead or proprietary: **5** — RIPE TTM, PlanetLab, Freebox ADSL, Akamai EdgeScape ground truth, Alidade **EuroGT** (anonymous European Tier-1, never released) (+ Alidade scraped campus addresses).

### Practical takeaway for CBG benchmarking today

The modern, freely available "stack" — RIPE Atlas (VPs + anchors as GT), ISI/ANT hitlist + LANDER geolocation (target IPs), CAIDA Ark (topology), M-Lab, MaxMind/DB-IP/IP2Location/IPInfo (DB baselines), and NYC/LA/Shanghai street-level — covers every role the older papers needed. Pre-2015 papers' original datasets (RIPE TTM, PlanetLab, Freebox, EdgeScape) cannot be re-obtained, so direct measurement replication of Gueye/Octant/Hu/Alidade is no longer possible — only their *methodology* is reproducible on the current substrate, which is what Darwich et al. did.

---

## Sources

- [Gueye et al. — Constraint-Based Geolocation of Internet Hosts (IMC 2004)](https://www.cs.bu.edu/fac/crovella/paper-archive/imc04-geolocation-full.pdf)
- [Wong et al. — Octant (NSDI 2007)](https://www.cs.cornell.edu/people/egs/papers/octant-nsdi.pdf)
- [Hu et al. — Towards Geolocation of Millions of IP Addresses (IMC 2012)](https://ant.isi.edu/~johnh/PAPERS/Hu12a.pdf)
- [Chandrasekaran et al. — Alidade (Duke CS-TR-2015.001)](https://users.cs.duke.edu/~bmm/assets/pubs/alidade--cs-tr-2015-001.pdf)
- [Darwich et al. — Replication (IMC 2023)](https://dl.acm.org/doi/10.1145/3618257.3624801)
- [Nur — Accuracy and Coverage Analysis of IP Geolocation Databases (BalkanCom 2023)](https://ayasinnur.com/wp-content/uploads/BalkanCom2023.pdf)
- [Wang et al. — NeighborGeo (Computer Networks 2025)](https://www.sciencedirect.com/science/article/abs/pii/S138912862400728X)
- [Cho et al. — Selection of Landmarks for Efficient Active Geolocation (TMA 2024)](https://tma.ifip.org/2024/wp-content/uploads/sites/13/2024/05/tma2024-final40.pdf)
