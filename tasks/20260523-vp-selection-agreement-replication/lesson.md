# VP Selection — Agreement-Methodology Replication — Lessons

## 2026-05-23

Lessons will be captured here as corrections and discoveries occur during the task.

Initial framing (will likely be revisited):

- **"Agreement" without hard GT is a different question than "accuracy" with hard GT.** Cho had no choice — VPN endpoints don't come with verified physical coordinates, so agreement-with-full-pool was the strongest signal available. We do have hard GT, so we should treat "agreement" as a *generalization check* (does the subset preserve the system's behavior?) and treat "accuracy" as the primary quality signal. Reporting only agreement, with our setup, would be throwing away the better evidence.
- **Pool size matters more than relative percentage.** Cho's "32% of pool" headline number is for a 780-anchor pool. With ~12K probes the diversity tail is much fatter and absolute K — not %-of-pool — is the operational quantity. Plan the sweep on log-K, not log-%.
- **`dist_rtt` is structurally limited in our setup.** No probe↔probe RTT mesh = no full RTT graph over the candidate VP pool. Worth running on the 723-anchor subset as a methodological replication of Cho's negative result, but not as our default candidate.
- **Don't lift Cho's 0.51c speed limit verbatim, and don't recalibrate from probes either.** Cho's number was calibrated against their 780-anchor mesh; we recalibrate on our 723-anchor mesh (post-SOI filter) using the existing `LowEnvelopeLTD` per-anchor LP fit — take the fastest per-anchor envelope's implied one-way speed as $S = 2 / \min_i \text{slope}_i$. Probes are wrong for this step despite being closer to our actual benchmark data: probe-side last-mile + GT noise survives the SOI filter and would inflate $S$. The per-anchor envelope smooths over single-pair RTT noise (Cho's max-over-pairs is fragile to that, especially with noisy GT). When calibrating a *constant* — as opposed to a per-VP curve — favor the cleanest input, even if it's a smaller pool.
