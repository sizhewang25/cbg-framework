# CBG Phase Benchmark

Benchmark wall-clock time and memory consumption for each CBG pipeline phase
across `scripts/analysis/cbg_evaluation` combinations.

Primary implementation goal: keep framework code minimally invasive by
instrumenting evaluation-time pipeline method calls instead of editing every
distance, filtering, multilateration, or centroid implementation.
