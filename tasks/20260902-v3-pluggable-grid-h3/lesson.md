# v3 Pluggable Answer-Space Grid (HEALPix + H3) — Lessons

## 2026-09-02

**Do not infer a parameter that the data cannot carry.** The first
`Grid.cell_boundaries` recovered HEALPix's `nside` from the cell ids by finding the
smallest grid whose id space contained them. That reasoning is wrong — cell 5 exists at
*every* nside — so a target set whose occupied ids happened to be small would have drawn
completely wrong cells and rendered a plausible-looking map with no error. H3 ids do
self-describe their resolution, which is exactly what made the shortcut tempting. Passing
resolution explicitly costs one parameter and converts a silent-wrong-answer class of bug
into an impossibility.

**A prediction stated in the plan is worth more than one left implicit, including when it
is wrong.** The plan predicted H3 res 4 (finer) would give K equal-or-higher and accuracy
equal-or-lower than HEALPix nside 128. as7018 did the opposite on both counts (K 27 -> 22,
accuracy 0.397 -> 0.500). Because the prediction was written down, the mismatch was
immediately legible as a *result* — boundary alignment dominates resolution at these
scales — rather than being absorbed as an unremarkable number.

**Abstracting revealed a latent bug in the code being abstracted.** Writing
`ring_lonlat` for H3's dateline behaviour showed the pre-existing HEALPix path had the
same defect: `np.where(lon > 180, lon - 360, lon)` wraps individual vertices but does not
keep a *ring* contiguous, so a straddling cell smeared across the map. It had never
surfaced because no target sits near the antimeridian. Generalizing a single-implementation
code path is a cheap way to find the assumptions it was silently relying on.

**Verify library calls, do not read signatures.** Continuing the `venn2_unweighted`
lesson. Three h3 v4 behaviours would each have been a silent bug: `cell_to_boundary`
returns `(lat, lng)` (a transposed ring still renders, just in the wrong hemisphere);
pentagons return 5 vertices to a hexagon's 6, so any rectangular-array assumption breaks
only on the 12 cells per resolution that sit over ocean; and `latlng_to_cell` returns a
hex string, not an int. A ten-line smoke script caught all three before any code depended
on them.

**Verifying a claim for publication is when you find out it is wrong.** The NY-metro
figures had been measured in an earlier session and were reused from notes. Re-running
them before putting them in the paper confirmed the HEALPix numbers exactly *and* turned
up the case that inverted my framing: H3 at 45 km merges the group that HEALPix cannot
merge at 407 km. Had the note been trusted, the paper would have carried "switching
tessellation does not help either" — a sentence contradicted by the repo's own output.

**Attribution is a claim too.** I wrote "on our RIPE data" over numbers drawn from two
different setups whose VP and target roles are reversed (`probes_to_anchors` vs
`anchors_to_probes`), only one of which matches the dataset §7.2 describes. Aggregating
runs into a single unnamed "our data" reads as more evidence than it is and would not
survive a reviewer checking which set produced which figure. Cheaper to name the setup per
number, or state the range across runs.
