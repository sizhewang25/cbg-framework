# 6-Way Ring Venn for Pooled Cross-Run `plot-venn` — Lessons

## 2026-09-03

**Compounding option choices can silently negate the original goal.** The user
picked "skip the area fit" on one question and "concentric bullseye" on the next.
Each was defensible alone; together they removed overlap geometry entirely, so
the result stopped being a Venn diagram — which was the whole request. The
tradeoff was stated on each question separately but never as a combination.
When sequential choices interact, state the combined consequence before building.

**"Venn diagram" is ambiguous at 6 sets, and the ambiguity is load-bearing.** A
mathematical 6-set Venn needs 63 regions and cannot be drawn with circles. The
colloquial/presentation "6-way Venn" is six circles on a ring showing 31 regions.
These are different figures with different fidelity. Asking for a reference image
resolved in one step what three rounds of describing options had not.

**Render the alternatives instead of describing them.** Prose about "stress
0.0034" and "41% region error" did not communicate; one preview image did. For
visual work, build the cheap prototype before asking the user to choose.

**A subset check where equality is meant creates order-dependence.** Validating
each run's method set as a subset of the *first* run's made pooling
`as01 + as7018` behave differently from `as7018 + as01` — error in one order,
silent data loss in the other. Symmetric invariants need symmetric checks; the
test that caught it explicitly ran both orders.

**The template's region set is a clean invariant, and worth pinning as one.**
`n` circles on a ring produce exactly `n*(n-1)+1` regions, and they are exactly
the cyclically-contiguous subsets. Discovering that turned a six-set special case
into a general 3-8 layout and gave the tests something sharper to assert than a
magic 31 — a future edit to `RING_RATIO` that changed the topology would now fail
loudly instead of silently redrawing the figure.

**A figure that cannot show everything must say what it omits in the artifact,
not just the docs.** The ring drops 18.1% of solved targets. The footnote names
the count and the coverage CSV names the combinations, so the omission travels
with the figure. Documenting it only in the README would have left every copy of
the PNG making a claim about the whole population that is false.
