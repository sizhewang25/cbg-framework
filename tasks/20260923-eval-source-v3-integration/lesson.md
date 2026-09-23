# Integrate `eval_source` into the analysis/v3 CLI — Lessons

## 2026-09-23

**A golden baseline has to be generated, not adopted.** The plan said to
snapshot the committed `eval_source/` artifacts and diff against them. That
failed immediately and for a reason worth keeping: those artifacts predate the
CSV reconstruction, so they carry a different row order and a
`n_unique_target_asns` the current CSV cannot produce. Diffing a refactor
against them would have reported ~40 spurious differences and hidden any real
one. The fix — run the *unchanged* code on *today's* input and snapshot that —
is also the only baseline that isolates the refactor from pre-existing drift.

**Row order is not part of the contract; say so in the harness.** The first
diff reported every column as different because the two frames were not
aligned. Sorting on `target_id` before comparing is what makes the check about
values rather than about `groupby` iteration order.

**A layering test needs a negative control.** `test_layering.py` passed the
moment it was written, which proves nothing — the property might have been
untestable as written. Reverting one import to the old form and confirming
6 assertions fail is what established it has teeth. Worth doing for any test
whose job is to prevent a regression that is not currently present.

**Two halves of the same rule need two kinds of check.** The subprocess
import-graph test only sees top-level imports, so an import moved back *inside
a function body* — exactly the form being removed — would pass it. The
source-level grep catches that half. Either check alone is a false sense of
security.

**Measure the argument before accepting it.** The clean-architecture design
rested largely on the claim that the two clusterings disagree, making three
copies of the answer space "two too many". True, but measured at **73 m on one
cluster of as01 and exactly zero on as02/as03** — because the operator datasets
carry near-uniform replica counts per coordinate. That single measurement moved
a whole stage out of scope. The same measurement then *predicted* F6's much
larger `n_members` divergence, which no one had flagged: equal replicas hide
the centroid effect but not the counting effect.

**A shared repo may change under you.** Test counts moved 1287 → 1290 mid-stage
with no test added by me; the cause was concurrent work on
`figure_error_cdf.py` in the same working tree. Explaining an unexpected count
rather than waving it away took one `--collect-only` diff and was worth it —
and it is the reason every commit here stages its own files by name rather
than using `git add -A`.
