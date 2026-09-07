# §8.1 Figures (v3) — Lessons

## 2026-09-07

- **Pilot the figure before speccing it.** Figure D was ranked fourth of four
  on the reasonable-sounding grounds that it was the only one needing new data
  plumbing. Ten lines of pandas over files already on disk showed it carries a
  crossover, a monotone collapse and a visible confirmation of the
  "SoI-is-the-baseline" finding — three results, more than any other figure in
  the set. Cost of the pilot: one query. Cost of not running it: building the
  most load-bearing figure last, or cutting it.
- **Validate the palette against the surface the chart actually renders on.**
  The dataviz skill defaults to `#fcfcfb`; this repo's `_SURFACE` is `#ffffff`.
  Contrast and lightness-band results are only meaningful against the real
  surface, so the default would have certified numbers no figure here has.
- **A diverging scale is not two sequential ramps with a gray at the join.**
  The first spec ran each arm from the neutral midpoint outward and **failed**
  the ordinal light-end contrast check on both arms (1.79:1 and 1.75:1 against
  a 2:1 floor). The arms have to start one step in from gray, with the neutral
  as a separate band. Running the validator caught this; reasoning about ΔE
  would not have.
- **Ask what the palette is already saying before adding to it.** Blue and red
  are Shortest-Ping and Spotter in this paper. A blue ordinal ramp beside
  variant-hued lines in one panel would give one hue two meanings. `pareto.py`
  had already solved this and written down why — its polylines are neutral
  grey "because hue is taken" — so the rule was inherited rather than invented:
  context gets ink, hue stays with the entity.
- **Check a subagent's structural claims against the source of truth.** An
  exploration pass reported that as7018's `octant_cbg` is "a different
  algorithm" from the operator runs' `octant_cbg_spl` and wanted a footnote.
  Both runs' `run.json` show identical configuration. `SCHEMA.md`'s combo table
  said so too. One query settled it; taking the report at face value would have
  put a false caveat in the paper.
- **A well-built table makes the figure's contract trivial.** Every degenerate
  case is already encoded upstream — `n_targets = 0` with `accuracy = NaN`,
  `zero_variance = True` with `phi = NaN`. So the entire figure-layer
  obligation reduces to one rule: never render a NaN as zero. Had the tables
  emitted `0.0` for an empty stratum, the figure layer would have needed to
  reconstruct the distinction from cell counts.
- **A key that works everywhere else can still be the wrong key here.**
  `short_dataset` is the dataset identity used across the whole v3 layer, so
  pairing a weighted run to its mesh twin with it looked like reuse rather than
  a guess. It strips only an all-numeric trailing tail, so it reduces *none* of
  the plausible weighted run names to their dataset — and the failure mode is a
  row filed under the wrong dataset, not an error. A test written for the happy
  path caught it before any weighted data existed to be mis-filed. The fix was
  to make the caller state the pairing, and to pin the non-inferability so it
  cannot be re-introduced as a convenience.
- **Bolding is a statistical claim, so it needs a stated rule.** "Bold the best
  per row" sounds like formatting. On as03 it would have declared Octant-Hull
  the top-1 winner over Spotter on a 0.028 gap with a 0.023 standard error, and
  on as03 at top-3 it would have hidden that Shortest-Ping and SoI tie for best
  and beat Octant-Hull. A within-one-standard-error band costs three lines and
  changes what the table asserts.
- **Ask what a "top-1 only" table would say before building it.** Octant-Hull
  wins every top-1 row, so the requested table states one conclusion three
  times; top-3 has three different winners. Pivoting the numbers first and
  reading them took one query and turned a formatting request into a scope
  question worth raising.
- **A convention inherited from the reference implementation can be wrong for
  the new data.** The v2 plotter's 1 km log floor was invisible there and
  clipped up to 4.4% of targets here, flattening exactly the curves that had
  earned the left tail. Checking `min()` and counting rows under the floor took
  one query; the figure would otherwise have understated the best variants and
  reported a clamp as a percentile.
- **Two files in one directory must not spell one column two ways.**
  `error_km_p50` appears in both `topn_accuracy.csv` and the CDF's percentile
  CSV. `method="nearest"` had a genuine rationale in v2 (a percentile that
  names a real target) and still had to go: 0.7-1.3 km of disagreement with the
  file the paper's table reads is worse than losing the property. Diffing the
  two files against each other is what caught it — the numbers each looked fine
  alone.
- **Ask what colour already means in this paper before copying a palette.** The
  reference plotter marks 100/500/1000 km with green/orange/red guides. Those
  three hexes are Octant-Hull, Vanilla and Spotter here, so the guides would
  have read as three more series.
- **`ScalarFormatter(set_scientific(False))` renders 0.1 as `0`.** The axis
  claimed the curves started at zero when they started at 100 m. Rendering the
  figure and looking at it is the only thing that catches this class of bug —
  the palette validator checks colour, not axes.
- **Check whether the existing table covers the whole domain before building on
  it.** `confusion_pairs.csv` has a `seeds_crossed` column and looked like the
  obvious input for a scatter against it. Its row-inclusion rule keeps only
  rows *wrong* at the reported top-N, so the entire `y == 0` column — over a
  thousand rows per run, and half the point of the figure — is absent. Reading
  the filter, not the schema, is what surfaced it; and recomputing from the
  same `seed_crossing_matrix` meant the two artifacts could still be checked
  against each other (delta 0 on every shared row).
- **A threshold on a figure needs to come from the data, not from a round
  number.** The two disagreement regions only mean something against a scale
  that says when a coordinate error *should* have changed the label. The answer
  space already publishes it — `seeds.csv`'s `margin_km`, half the distance to
  the nearest other seed — so the rule is read per run (151/172/160 km) rather
  than picked as "100 km looks about right".
- **Two adjacent regions need disjoint comparisons.** `>` for the far side and
  `<=` for the near side, so a point exactly on the margin lands in exactly one
  of them. Both `>=`/`<=` would double-count it and the two counts would no
  longer add up against the total.
- **"Show the % per bin" can silently change the denominator.** Adding the
  share label only works if band 0 equals the reported top-1 accuracy, and that
  forces the denominator to be every target rather than the rows drawn. The two
  differ only on methods that fall back — one of six here — so a spot check on
  any other method would have passed and the figure would have disagreed with
  the paper's own table on Vanilla alone.
- **Alpha accumulation is a density encoding with a ceiling.** It saturates
  near 1/alpha coincident marks, and the data most likely to hit it is the data
  with exactly repeated values — here Shortest-Ping and SoI, whose answer *is*
  a VP coordinate. Stating the ceiling and putting the count in a label costs a
  sentence; discovering it from a reader's question would cost the figure.
- **A right-edge annotation belongs to whichever panel it is nearest.** Placed
  at the axes edge in a multi-panel grid it lands mid-gutter and reads as
  belonging to either neighbour. A twin axis fixed the alignment but not the
  ambiguity; tightening `wspace` after `tight_layout` was the actual fix.
- **Compacting a figure is a physical knob, not a fractional one.** "Make the
  band shorter and the gutter smaller" is unsatisfiable in row fractions —
  they sum to the row, so trading one only moves height to the other. Shrinking
  the row's *inch* height shortens both. Naming that constant (`ROW_INCHES`)
  made the request answerable and the floor statable: at 0.50 in the gutter is
  0.15 in against ~0.10 in of 7 pt text, which is now a test rather than a
  thing to rediscover the next time someone compacts it.
- **A tick labels the data, not the cell it sits in.** Once each row held a band
  *and* a readout gutter, the row's centre stopped being the band's centre.
  Putting the tick at the row's centre would have floated every label above the
  thing it named.
- **An annotation that is also a data mark should be drawn as one.** The median
  rule was 0.04 taller than the band on each side — enough to read as an
  overlay rather than as the middle target of the band. Same height, same
  meaning.
- **Ask what "like the ref image" does not include.** The spectrum chart's rows
  carry *different* x ranges because they continue a sequence; ours share one
  error axis, which is the whole basis of comparing bands. Copying the row
  framing was right, copying the axis structure would have broken the figure.

- **Prove a plotting refactor inert with md5, not with a passing suite.** Pulling
  `_draw_panel` out of `plot_bands` touched every mark on six figures; the tests
  only assert "renders, non-empty". Hashing all six PNGs and the band CSVs before
  and after is the check that the extraction moved code and nothing else, and it
  costs one loop.
- **A cross-dataset figure's orientation is the whole design.** 18 panels lay out
  the same way either way; what changes is which comparison is a straight-line
  read. The question was "does this method's signature survive a change of
  dataset", which is within-method, so methods take the rows. Picking the
  transpose would not have looked wrong — it would just have made the figure's
  own question the hard scan.
- **Keep the shared axis physically the same length across figures that invite
  comparison.** The per-run and cross figures both live in §8.1, so the cross
  grid keeps the 4.9 in panel width rather than fitting six method columns; an
  x position means the same thing in both, and five decades still label without
  collisions.
- **Don't pool runs with disjoint targets of different sizes.** 399 / 412 / 458
  targets means one pooled band share silently weights as03 heaviest and
  describes no dataset. The grid puts three panels side by side instead, each
  with its own `n` printed — and each panel then equals that run's own figure
  exactly, which is a checkable property rather than a claim.
- **"Cross-dataset" is two questions, and picking one silently answers the wrong
  half.** Merging the runs asks "how good is each method on the fleet"; keeping
  them apart asks "does each method behave the same everywhere". I built the
  second and reported it as *the* cross-dataset figure. Both are cheap once the
  loader is shared — the fix was a `--layout` flag and one extra filename stem,
  not a second module.
- **Pooling is legal or illegal depending on the data, so check rather than
  argue.** I had written "a pooled share would describe no dataset" into three
  files; the actual objection to pooling is double-counting, which only happens
  if the runs share targets. `guard_disjoint_targets` turns the worry into an
  assertion, and `weighting_check` turns the remaining caveat (micro vs macro
  average) into a number — max 0.6 pp here.
