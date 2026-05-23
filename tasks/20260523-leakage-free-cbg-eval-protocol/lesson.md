# Leakage-Free CBG Evaluation Protocol — Lessons

## 2026-05-23

Lessons will be captured here as corrections and discoveries occur during the task.

Initial framing (will likely be revisited):

- **LOO is intuitive but wrong for benchmarking deployable systems.** It implies per-query refitting, which is not what any production CBG service does. The right question is "given a model fit on a fixed landmark corpus, how well does it generalize to novel IPs?" — that's K-fold, not LOO.
- **Octant's institutional-exclusion rule was specific to their PlanetLab setup**, where universities had multiple nodes in the same building. RIPE Atlas anchors are clustered differently (a few CDNs hold many anchors in many metros), so blindly copying the rule may not be the right adaptation — the underlying concern (trivially-near landmarks) is different.
- **Probes as eval targets sound appealing for "12K targets" scale**, but probe GT noise contaminates the eval signal in a way per-VP calibration can't absorb. Anchors-only-as-targets is the right call even if it caps eval size at 723.
