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

- **A refactor that unifies two code paths can be the bug.** Rows and pooled rows
  look like the same thing computed at two scales, and computing both from counts
  is the obvious simplification. It moves three of thirty-six printed cells by a
  digit, because a rate rounded to 4 dp and a rate recomputed from its own
  rounded count are not the same number at 3 dp. The invariant worth protecting
  was named in the module's first paragraph — "this table and `table-accuracy`'s
  cannot disagree" — and the unification would have broken it silently, on three
  cells, in a paper table.
- **`groupby` drops null keys by default, and that is a silent row deletion.**
  The new aggregate rows have no dataset, so `groupby(["dataset", "kind"])` would
  have rendered exactly the old table with no error and no warning. Grouping on
  the plan's own index removes the failure mode rather than guarding it, and
  makes the ordering test a one-liner.
- **Check whether an exemption's stated reason still holds after you change the
  thing it was about.** `accuracy_table` skips the setup guard because "this
  table puts each run on its own row where no such averaging can happen". Adding
  a pooled row voided that sentence, and the comment was the only place it was
  written down. Guards are inherited from a premise, not from a module.
- **Pick the guard's input for the dependency graph, not for the read cost.**
  `target_labels.csv` is 5× cheaper to read than a parquet and was the wrong
  choice: it is a `build-proximity` artifact, so a proximity-unrelated check
  would have made the table fail on a classified-but-not-proximity-analyzed run,
  and it is regenerated on a different schedule than the numbers it guards. The
  `classify` parquets sit in the directory the accuracies already come from.
- **An indent needs something to indent under.** `· AS01` reads as subordinate
  beneath a `MESH (3 ASes)` header and as nothing at all without one — and with
  a single dataset the aggregate is suppressed, so two rows both read `· AS01`
  and neither named its campaign. The label had to become a function of whether
  the group header exists.
- **A legend swatch can collide with a data mark.** The `mesh` entry was filled
  with the same neutral as the `fallback` segment, so the legend's dataset-type
  key was pixel-identical to one of its outcome keys. Outline-only kind entries
  fix it, and the general rule is that a legend for a *texture* channel must not
  borrow the colour channel to draw itself.
- **Drop a legend entry for a category that never occurs.** `error` is zero on
  every run and its ink is the lightest step, so its swatch was a near-invisible
  box beside a word — worse than absent, because it implies the reader failed to
  spot it. The footnote still names all four, so the partition is not hidden.
- **`"nan" not in text` is not an assertion about NaN.** It passes and fails on
  the word "unanalyzed". A test for "no cell rendered as NaN" has to look at the
  cells.
- **Two encodings spent means the third is not available.** Hue = method and
  hatch = dataset type left the stack with only lightness, which sounds like a
  constraint and was a gift: the figure needed no new palette entry and so did
  not block on the ordinal-ramp groundwork it would otherwise have waited for.
- **A semantic colour scheme has to displace the old one completely, or it
  reintroduces the collision it was meant to avoid.** With green/red/grey meaning
  the outcome, tinting the method tick labels in the variant hues put
  "Octant-Hull" in green and "Spotter" in red an inch below green and red
  segments — colour meaning outcome inside the bars and identity just outside
  them. The half-measure was worse than either whole one.
- **Red and green is the colour-blind pair, so the fix is lightness, not
  labels.** Monotone L* up the stack (34 / 54 / 73) is what survives greyscale
  and deuteranopia; the per-segment percentages are belt and braces. Picking the
  three by eye would have landed on a green and red of near-identical lightness,
  which is what the first candidate set did (ΔL* 4.6).
- **Choose the label ink per fill, and the fills get chosen on their merits.**
  Fixing on white text forces every fill dark, which killed the light grey the
  stack needed at the top. One luminance test per fill removed the constraint
  entirely.
- **An axis in fractions beside labels in percent makes the reader convert.**
  Adding `%` labels to the segments silently put two units on one figure; the
  axis had to follow.
- **Merge a category into the one it belongs to, not the one it is near.**
  `error` sits next to `fallback` in the count list and next to `wrong` in the
  stack. It is a failure to answer, so it merges into `failed`; merging it into
  `wrong` would have reported a crash as a scoring miss. The CSV keeps the split
  so the merge is a drawing decision rather than a loss.
- **Sorting a small-multiple has to be a property of the figure, not of the
  panel.** Ranking each `compare` panel on its own correct rate puts Octant-Hull
  in a different column per dataset, and the cross-dataset comparison the figure
  exists for becomes a search. One order, computed over every dataset in the
  table, is what makes an x position mean the same thing in all three panels.
- **Two independent encodings need two legends.** Merged into one list,
  `failed` and `mesh` sit as though they were alternatives on one scale, and
  nothing tells the reader a bar is one of each. Splitting them and titling each
  with its channel costs four lines and removes the ambiguity entirely.
- **When prose comes off a figure, check what it was carrying.** The footnote
  was the only place saying the empty outlines were uncollected rather than
  zero. Deleting it silently would have left a ghost bar meaning "scored
  nothing"; the fact moved into the legend label, which is where a reader
  actually looks for it.
- **Sorting a small-multiple has to be a property of the figure, not of the
  panel.** Ranking each `compare` panel on its own correct rate puts Octant-Hull
  in a different column per dataset, and the cross-dataset comparison the figure
  exists for becomes a search. One order, computed over every dataset in the
  table, is what makes an x position mean the same thing in all three panels.
- **Two independent encodings need two legends.** Merged into one list,
  `failed` and `mesh` sit as though they were alternatives on one scale, and
  nothing tells the reader a bar is one of each.
- **A legend shows the mark, not a description of it.** "traffic-weighted (not
  collected)" put a fact about the *data* into the key for the *encoding*. The
  dashed swatch already says the bars are dashed; why they are empty is caption
  material. The swatch matching the mark is the invariant worth keeping.
- **`tight_layout` lays out axes and knows nothing about a figure-level
  legend.** Moving the legends above the axes needed `rect=` to reserve the band
  by hand, and the two constants (`LEGEND_TOP`, `AXES_TOP`) had to be tuned
  against a render — the first pair left a third of the figure blank.
- **A frameless legend still reserves its border padding.** `frameon=False`
  hides the box, not the space, and that space was wide enough to make a title
  look detached from its own first swatch. `borderpad=0, borderaxespad=0`.
- **`tight_layout(rect=...)` reserves a band for the axes *and its
  decorations*, so the axes box lands well below the band's top.** To put the
  bars right under a figure-level legend you set the box directly with
  `subplots_adjust(top=...)`; the `rect` version left a third of the figure
  blank and looked like a legend placement bug.
- **Inline legend titles have to be measured, not offset.** matplotlib puts a
  legend title above its entries with no option to move it, so the title becomes
  separate text and the legend anchors to its right — and the gap depends on the
  rendered width of the word at that figure size. This module draws at two
  widths, so a hard-coded offset would be right in one and overlapping in the
  other. Draw once, measure, then shift.
- **A shared layout constant across two figures is a collision waiting for the
  one with more furniture.** The comparison grid carries per-panel titles the
  pooled figure does not, so one `AXES_TOP` put the legend straight through
  `AS01 · n=399`. Two named bands, and a test pinning that the grid's is the
  wider of the two.
- **A placeholder must not invent the numbers it was not given.** Rates arrived
  without a denominator and without top-3. Back-computing an `n` to make the
  counts line up would have produced a bolded "best" cell resting on a figure
  nobody could reproduce; carrying `n = NaN` instead makes `best_in_row` decline
  on its own, because it cannot form a standard error. Keying the constant on
  the top-N does the same job for the appendix table.
- **Gate a placeholder on the absence of the real thing, not on a flag.**
  `provisional_for` returns `None` the moment the row has a run, so pairing a
  real `--weighted-run-id` supersedes it automatically and deleting the constant
  is a cleanup rather than a switch-over. A `--use-placeholder` flag would have
  been one more thing to remember at exactly the moment it mattered most.
- **Hatch drawn in the surface colour strikes through a label.** The white
  hatch cut across white percentages on the weighted bars. A bbox of the
  segment's own fill behind the text restores the contrast `label_ink` computed
  for — and the hatch itself had to go sparse and half-transparent anyway, since
  dense opaque hatching lightened a green enough to read as a *different colour*,
  which is the one thing an encoding where colour means the outcome may not do.
- **`hatch.linewidth` is an rcParam, not a Patch property.** At the 1.0 default
  `//` renders as densely as `///`, so the pattern constant alone could not make
  the hatch light.
- **When the caveat comes off the figure, say where it went.** Removing the
  provisional note leaves a PNG that cannot be told from a real result. The fact
  survives in three places — the table footnote, the CSV column, the manifest —
  and `provisional_kinds` stays as the single predicate they all query, with a
  docstring saying the figure deliberately does not mark it.

## Phase 0j — the geometry-vs-routing scatter

- **A carried column fails differently from a computed one.** `min_inflation`
  comes from `eval_source` through `build-proximity`'s optional `context`, so a
  run built without a source CSV it could find yields the column full of NaN
  rather than missing. Plotted, that is an empty panel with no complaint; an
  all-NaN axis is now refused by run id, naming `--source-csv`.
- **An uncollected series falls out of every list derived from the data.** The
  drawn kinds come from `points["kind"]`, which by definition cannot contain the
  campaign that was never run — so the legend lost the traffic-weighted entry
  and six grey dots read as the whole comparison. `legend_kinds(drawn, pending)`
  is the union, kept separate from the drawing set on purpose.
- **Set the axis limit *past* a reference line, not to it.** With
  `ylim[0] == 1.0` the speed-of-internet rule landed exactly on the bottom spine
  and vanished, leaving its annotation pointing at nothing. The floor now gets
  4% of the span below it.
- **A per-target threshold must not be drawn as a line.** The seed margin is
  half the distance to the next seed, so it differs per target by a factor of
  three across one dataset. It is drawn as the p25-p75 band with p50 ruled,
  because one number would promise a sharp decision boundary the data has not
  got.
- **Density, not counts, when one series is a subset of the other.** The
  traffic-weighted campaign is the mesh set filtered, so it will always carry
  fewer points; a count marginal would answer "how many survived" when the
  question the figure asks is "where do the survivors sit".
- **A log decade brings eight minor gridlines.** At full strength they read as
  data on a scatter. Kept at a third of the weight, because reading 300 off a
  log axis needs them.
- **An x axis chosen by y is not an independent variable.** The first version
  plotted the *shortest-ping* VP's distance against min-RTT inflation and read
  ρ 0.54 as "geometry and routing co-occur". RTT chooses that VP and inflation
  is what RTT is ranking on, so the axis carried its own y. Against the nearest
  *measured* VP the correlation is -0.02. Both metrics now ship, and the module
  docstring says in as many words that a claim about geometry has to be made on
  `closest`.
- **A near-saturated covariate is a finding, not a boring axis.** 95.3% of mesh
  targets have a VP inside their own seed margin, so the flat scatter is the
  point: the mesh set's difficulty is selection, not opportunity. A figure whose
  x explains nothing can be the one that tells you where to look next.
- **Identity and class are different match rates.** RTT returns the
  geometrically nearest VP by id for 6.5% of targets and one in the right class
  for 47.6%. Quoting the first as a selection rate would understate the baseline
  by a factor of seven; VPs co-locate inside a metro.
