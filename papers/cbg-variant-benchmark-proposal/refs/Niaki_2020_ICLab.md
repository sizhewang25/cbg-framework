# Niaki et al. 2020 — ICLab: A Global, Longitudinal Internet Censorship Measurement Platform

**Citation:** Niaki, Cho, Weinberg, Hoang, Razaghpanah, Christin, Gill. *ICLab: A Global, Longitudinal Internet Censorship Measurement Platform.* IEEE S&P 2020. Data: https://iclab.org/

## Overview

ICLab is a censorship-measurement platform that uses **commercial VPN endpoints (264)** plus a small set of **volunteer-operated devices (17 VODs, mostly Raspberry Pis)** as vantage points to perform continuous, network-stack-deep measurements of DNS manipulation, TCP packet injection, and block-page delivery. Running since late 2016, it covers 62 countries / 234 ASes and collected **53,906,532 measurements of 45,565 unique URLs between Jan 2017 – Sep 2018**, detecting blocking of 3,602 unique URLs in 60 countries. Not a geolocation paper — but **Appendix B (“VPN Proxy Location Validation”) is the load-bearing piece for CBG benchmarking**: ICLab rejects every VPN endpoint whose RTT to a set of RIPE Atlas landmarks implies a propagation speed above **153 km/ms (≈ 0.5104 c)**. This is exactly a CBG-style speed-of-light filter and the same calibrated speed limit later reused by Cho 2024 (Landmark Selection).

## Architecture

- **Central server / VP / control-node split.** Central server distributes test lists, schedules measurements, collects results; VPs perform per-URL HTTP GETs with full DNS / TCP / TLS / packet-capture instrumentation; *control nodes* (initially 1 academic node in the US, later +3 in Europe/Asia/USA) re-run every test from “uncensored” locations for comparison-based detection.
- **Two VP types:**
  - **VPN-based (264):** commercial VPN proxies. Long-lived, no volunteer recruitment, no physical hardware logistics. ICLab requires ≥2 ASes per country (75% of countries satisfy this).
  - **VOD (17):** physical devices in 13 countries, used where VPN coverage is missing or where ethics demand more transparency.
- **Test lists (three):** Alexa global top-500 (ATL), Citizen Lab + Berkman Klein globally-sensitive list (CBL-G), and per-country sensitive lists (CBL-C). 3,000–5,700 URLs per VP per cycle; all lists refreshed weekly. 47,000 unique URLs tested total.
- **Cycle:** every URL tested per VP **at least every 3 days**, runs of 1–2 hours each.
- **Detection pipeline:** centralized, retrospective, raw observations archived. DNS-manipulation detector (multi-heuristic, false-positive rate ≈ 10⁻⁴ at threshold θ=11 ASes); TCP-packet-injection detector (sequence-number conflict + RST/FIN/block-page evidence); block-page detector with 308 hand-curated regexes plus two **novel ML classifiers** (HTML tag-frequency vectors + locality-sensitive hashing on text) that surfaced **48 previously unknown block-page signatures across 13 countries**.

## Evaluation Setup

- **Scale:** 53,906,532 measurements / 45,565 URLs / 62 countries / 234 ASes / Jan 2017 – Sep 2018 (≈ 21 months).
- **Coverage by continent:** Europe 83 VPNs / 27 countries, N. America 87 / 5, Asia 64 / 14, Oceania 12 / 2, S. America 9 / 5, Africa 9 / 9. Heavy Europe/N.-America skew; *“not free”* countries underrepresented (8 of 62 — Iran, Saudi Arabia, etc.).
- **Freedom-House labels:** 8 NF (not free), 22 PF, 32 F.
- **Ground-truth VP location:** advertised location, validated by RIPE-Atlas-based RTT geolocation (Appendix B).
- **Five most-censoring countries by URL %:** Iran, South Korea, Saudi Arabia, India, Kenya (Turkey/Russia occasionally displace).

## Key Censorship Findings (brief — not the relevance angle)

- **DNS manipulation:** 15,007 events in 56 countries; 98% return NXDOMAIN or non-routable address.
- **TCP packet injection:** 143,225 high-confidence injections in 54 countries on 1,205 URLs; another 15.6M are “probable.”
- **Block pages:** 232,183 across 50 countries / 2,782 URLs; Iran delivers block pages for 24.9% of tested URLs.
- **Longitudinal value:** pinpoints policy shifts in India (net neutrality, 2018) and Turkey (Wikipedia ban). Censorship in PF/F countries is real but uneven.

## Strengths

- **Continuous multi-year operation** — only platform of its kind at this scale.
- **Full-stack capture** (DNS + TCP + HTTP + packet traces) enables retrospective re-analysis as detection improves.
- **Conservative false-positive engineering** at every detector (10⁻⁴ for DNS).
- **Honest VP validation:** the RIPE-Atlas RTT filter rejects ~10% more proxies on average than Weinberg 2018, accepting a smaller VPN footprint to gain location confidence.

## Limitations

- **VPNs are hosted in commercial datacenters** — censorship observed is a lower bound vs. what residential ISPs see (the paper notes 41% of ICLab’s VPN-hosting ASes are CAIDA-classified “content” networks).
- **Country coverage gaps**, especially Africa, S. America, and most “not free” countries (no China VOD; Iran / Syria deemed too risky for VODs).
- **Single ground-truth = VPN advertised country**, even after RTT validation (only confirms country, not city / lat-lon).
- **Test-list bias:** ATL / CBL-G / CBL-C dominate results; “long tail” of nationally blocked content not covered.
- **Control nodes also suffer outages** — early single-node design propagated gaps.

## Relevance to CBG Variant Benchmarking

The censorship findings themselves are off-topic. The relevant content is **methodological**, in roughly descending order of weight:

1. **Appendix B is a deployed CBG variant.** ICLab uses RIPE Atlas anchors as landmarks and applies a one-sided speed-of-light filter at **153 km/ms ≈ 0.5104 c** to reject implausibly-located VPN endpoints. This is a *country-level CBG accept/reject* used in production at IEEE-S&P quality — exactly the algorithm Cho 2024 (also in this refs folder) later subsamples. Citing both papers together gives us: the algorithm in production (ICLab), and an efficiency analysis of it (Cho).
2. **The 0.5104 c speed limit is a calibrated, peer-reviewed number** we can reuse (or compare against) in our CBG variants. It is meaningfully tighter than Katz-Bassett’s 0.44 c (4/9 c) baseline that `default.py` encodes via `SPEED_OF_INTERNET = (2/3) c`.
3. **VPN endpoints as VPs — biases relevant to benchmarking.** ICLab’s VP pool is 264 commercial VPN servers, almost all in datacenter ASes (CAIDA “content” = 41%). This is *complementary* to RIPE Atlas: Atlas is rich in residential/eyeball probes and well-anchored research/IXP anchors, but historically thin in many of the same NF countries (Iran, Saudi Arabia, Syria) where ICLab also struggles. If a CBG variant benchmark wants to test robustness to VP-type skew (datacenter-only vs. mixed), ICLab's published VP list is one of the few large, public, datacenter-heavy VP catalogs.
4. **Ground-truth quality caveat for VPN-based VPs.** ICLab’s own validation step exists *because commercial VPN locations are unreliable*. For any CBG benchmark that considers using public VPN endpoints as either VPs or targets, the appendix is a direct recipe (RTT vs. RIPE Atlas landmarks + 0.5104 c) and a stated rejection rate (~10%+ over prior art). Reuse this as a sanitization stage rather than treating VPN claims as ground truth.
5. **Longitudinal-measurement engineering lessons.** Multi-year operation surfaces issues that single-snapshot CBG papers ignore: VP churn (year-long outage from one provider’s config change), control-node single-points-of-failure, sanctions-driven access loss (Iran, May 2017). If our benchmark wants to claim stability over time, these are the failure modes to instrument for.
6. **Country/AS diversity numbers as a sanity bar.** ICLab’s 62 countries / 234 ASes is a useful *minimum* target for global diversity; the IMC-2023 RIPE dataset (~10K probes / ~500 anchors) easily exceeds it on raw count but is similarly Europe-skewed. Worth quoting when arguing about coverage.

**Connection strength: medium-strong but narrow.** The whole paper is censorship-focused; only Appendix B and the VP-management discussion in §III-B map onto CBG concerns. But that mapping is direct and concrete — Appendix B is, in effect, a small CBG paper hidden inside a censorship paper, and it shares its speed-of-light calibration with Cho 2024.
