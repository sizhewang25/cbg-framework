# §8.1 PNI Cohort Verification (v3) — Lessons

## 2026-09-14

**Check the unit of analysis before trusting any accuracy number.** as01's "399
targets" are 20 coordinates with ~20 IP replicas each. The tell was a suspicious
result — sping accuracy of exactly 1.000 and exactly 0.000 on two cohorts — plus
metro counts that were almost all exactly 20. Round numbers and perfect scores
are a prompt to check how the data was constructed, not a result.

**Distinguish "the diagnostics contradict each other" from "each measures a
different thing."** ρ(d_pni, closest-VP distance) = −0.304 and
ρ(d_pni, min_inflation) = +0.581 are geometry and routing respectively. They do
not contradict one another, and since the PNI cohort is not traffic weighting,
neither is a verdict on claim 1 at all. They characterize the cohort. Overstating
this was a correction from the user.

**A surrogate's value can be as an optimistic bound, not as a measurement.** The
PNI-co-located cohort is acknowledged fake. It still earns its place: it is the
best case traffic-weighted filtering could deliver, and it lets the whole script
stack be built and validated before real weights exist.

**Prefer the repo's own falsifier over a newly invented one.** The draft proposed
"% of PNI in TG cell should equal sping accuracy," an identity that cannot hold.
`pni_sping.py` already carried the exact identity
(`has_proximate_sping_vp` ≡ `shortest_ping` top-1) and already argued why the
successes cannot test the mechanism. Reading the module docstrings first would
have reached the mediation framing faster.

**A fake cohort earns pipeline validation, not readings.** Distance-binned
statistics off the surrogate (a bound on B, a "cliff" between two bands) were
withdrawn: 20 regions, a fabricated cohort, and thresholds the reporting had
already rejected. Threshold-free statistics on the full mesh survived. When a
fixture exists to exercise code, resist reading it as data.

**Check whether the proposed condition is already sufficient before adding
structure.** The A/B 2×2 collapsed to a 2×1 once it was noticed that, given A,
`correct ⟺ B` exactly — so splitting accuracy by A alone recovers B. The
user's simpler framing was the correct one; the extra column was mine.

**Build the script for the data you will have, not the data you have.** Draft
figures 1a/1b cannot be interpreted on a target-filtered surrogate because real
weighting filters flows. That is a reason to defer *interpretation*, not to
defer *implementation* or to substitute a different metric.

**Synthetic fixtures propagate silently into conclusions.** Every PNI number in
§8.1 traced to a 9-site fabricated list for the wrong peer ASN, and on it the
honest routing axis *lost* to the air control. The `SYNTHETIC-` filename prefix
was doing real work; keep that convention.
**The stated prediction was wrong, and the reason matters more than the miss.**
The plan predicted `singleton_share` would *fall* on the real list, because 14
sites admit larger feasible sets than 9. It rose, 0.633 → 0.837 on sping pairs.
Site count is not the binding constraint: the ellipse's size is set by the RTT,
and shortest-ping RTTs are small enough that adding sites changes nothing. A
prediction about a geometric test should be reasoned from whichever term is
actually binding. Writing the prediction down before running was still right —
it is what made the miss legible.

**Distinguish a synthetic fixture's failure from a real negative result.**
Every as02 PNI failure reversed on as01's real list: `contains_tg_nearest` 0.816
→ 1.000, unexplained-empty 0.136 → 0.000, and `routing_tg_nearest` went from
losing to air (0.795 vs 0.835) to beating both air and the self-fulfilling axis
(0.704 / 0.655 / 0.686). The prior conclusion "the routing story does not hold"
was a statement about `SYNTHETIC-us-carrier-hotels.csv`, not about the mechanism.

**Within-unit and between-unit versions of the same claim can disagree without
either being wrong.** Target-side strategy detection says `direct_no_pni`;
VP-side says `tg_nearest` at 88.2%. Per-target linearity gain is ~0; pooled gain
is +0.049 r². Both pairs resolve the same way: the `tg_nearest` predictor adds a
per-target *constant*, which cannot reorder one target's own VPs but does set
that target's offset relative to others. Check whether a diagnostic's unit of
variation can even see the effect before reading a null as a refutation.

**Report the scope where a test discriminates, not the pooled number.** The
feasibility statistics are 0/200 against the null on sping pairs and
indistinguishable from it (P = 0.23) on all pairs. Quoting the pooled figure
would understate the result; quoting only the sping figure without the pooled
one would hide that the exclusion is weak for distant VPs. Both belong in the
text.

**Clustering changed a conclusion, not just a caveat.** The expectation was
that region-level reporting would widen intervals and add a footnote. The
paired clustered bootstrap instead removed two of three dataset-level claims:
Octant-Hull's lead over Shortest-Ping is +9.2pp on as01 and +7.0pp on as03,
both with intervals spanning zero, and only as02's +29.9pp survives. Point
estimates were unaffected — replica counts are near-equal, so `region_accuracy`
matched `target_accuracy` to a thousandth. Precision was the whole effect.

**Use the paired test before concluding "no separation."** Two overlapping
independent CIs do not establish that two methods tie, because the methods are
scored on the same targets and pairing removes between-target difficulty. The
per-region difference was the right statistic, and it is *more* powerful than
comparing the two intervals — which makes the as01/as03 nulls harder to
dismiss rather than easier.

**A stratum's `n` was hiding its size.** `selection_miss` prints 145 targets
and is 8 sites; `selection_hit` prints 254 and is 13. Adding `n_regions` beside
`n_targets` in the existing tables cost two columns and made every §8.2 stratum
honest at a glance. Emitting the count next to the rate beat writing prose
about the count.

## 2026-09-15

**A robustness defect hid inside a replication artifact.** The ~20 IP replicas
per site were catalogued as a *statistical* problem — effective n, quantized
accuracy, narrow intervals. They are also an unintended repeated-measures
experiment, and it shows that Octant-Spline returns a different class for 15 of
23 as03 sites depending on which of that site's own addresses was probed. The
coordinate, the seed and every VP distance are identical across those
addresses; only the per-address RTT vector varies. Nothing in the paper's
accuracy numbers exposes this, because it is variance rather than bias. When a
dataset turns out to contain repeated measurements of one unit, ask what the
repetition *measures* before treating it purely as a nuisance to correct for.

**`region_accuracy` does not mean the same thing for every method.** At
homogeneity 1.000 it is a mean of verdicts; at 0.348 it is a mean of per-site
success rates. The column is defensible in both cases and is not comparable
across methods without homogeneity printed beside it. A statistic whose
interpretation depends on another statistic must ship with it.

**Cluster means mislead inside a stratum.** `region_rate` weights a region
contributing one address exactly like a region contributing twenty, because
strata subdivide regions. as02's `selection_hit` with SoI reads 0.993 at target
level and 0.889 at region level off a single address at weight 1/9. Whole-run
figures agree closely (max divergence 0.028) precisely because every region
contributes all its addresses there. Found by the subagent, verified, and
documented in the docstring rather than fixed — the estimator is right, the
reading was the risk.

**Check the inputs exist before dispatching work.** The request was to replicate
the full analysis for as02/as03. Half of it — every PNI result — was
unrunnable, because only one real site list exists and it names as01's peer.
Thirty seconds of `ls datasets/pni/` established that, and it turned a task
that would have produced confident fixture artifacts into a scoped one plus a
plainly stated gap.

**Verify a delegate's surprising numbers, not its routine ones.** The
homogeneity table and as03's 0-of-15 result were re-derived independently
before being written down; both reproduced exactly. The flagged ratio anomaly
(4.692 against a 4.462 heuristic) came with a correct explanation — `sqrt(n/R)`
is an approximation, not a bound — which is the kind of self-flagging that
earns the rest of a report more trust, not less.

**`--peer-asn` labels the peering, it does not verify it — and I claimed the
opposite without testing.** The assertion was described as "the guard that makes
a wrong-peer list an error rather than a plausible answer," including in a
config comment. False. The only hard check is that the site list declares one
ASN; `target_asn` was dropped by the parquet reconstruction, so nothing ties
either value to the run's targets and `pni.py` records `run_target_asns: []`
as *unverified rather than silently blessed*. Running as02-mesh against as01's
AS20940 list succeeded and reported "44.1% assigned the target's nearest
site" — precisely the silent-wrong-answer mode the guard was credited with
preventing. Reading the docstring would have been enough; it says so plainly.

**A demonstration intended to prove a safety property destroyed an artifact.**
The command run to show the refusal instead overwrote
`as02-260728-260802-mesh/pni-graph/`, and `outputs/` is gitignored, so there was
no recovery. Restored by rebuilding from the synthetic 9-site list at peer
15169, which reproduced 45.2% and matched the sibling run exactly — good
evidence of the prior state, but evidence rather than proof. A write command is
not a way to test whether a write will be refused; inspect the validation path
first, or run it against a throwaway `--analysis-root`.
