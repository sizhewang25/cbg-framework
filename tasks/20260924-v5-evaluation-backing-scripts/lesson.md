# v5 Evaluation Backing Scripts — Lessons

## 2026-09-24

Carried in from the v4 task and the §1b drafting pass, because each of these
cost a wrong number that had already been written down:

- A FALLBACK row carries a real `error_km`. Ranking on it without
  `classify.solved_mask` fills a variant's "most accurate" cohort with give-up
  rows, and the wrong number is the flattering one (Vanilla p5 bound
  69.0 → 33.1 km). Reuse `vp_proximity.cohort_frame`; never rebuild cohorts.
- A ratio metric comparing two quantities that are equal by construction needs
  an explicit margin. `d_sping < error_km` scored Shortest-Ping 100% against
  itself on a 0.006 km floating-point gap.
- Counts over a replica-quantized target set overstate independence. Six
  targets at an identical distance are one observation.
- A max is not a distribution. The 720.7 km bound is one target out of 317;
  quoting it without the count invites a reviewer to read it as typical.
- Check whether the competing explanation also holds for the method you are
  arguing against. Spotter having *more* far-VP reach than Octant is what
  killed the "multilateration is stable" framing.
