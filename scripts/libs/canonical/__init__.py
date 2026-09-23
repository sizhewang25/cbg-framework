"""The canonical `(vp, target, rtt_ms)` CSV contract, owned by neither layer.

`scripts/benchmark/v2` builds a run's inputs from these CSVs and
`scripts/analysis/v3` describes and re-reads the same files, so the schema, the
pair reduction and the eval-side filters belong to both and to neither. Keeping
them here is what lets the analysis modules import their own input format
without reaching into the benchmark package — and without dragging
`scripts.framework.v2` along behind it.

Import rule for everything under this package: no `scripts.benchmark`, no
`scripts.analysis`. `scripts.libs.*` only.
"""

from scripts.libs.canonical.eval_filters import (
    apply_eval_target_filters,
    derive_eval_pair_weight_min,
)
from scripts.libs.canonical.pairs import COLOCATED_IDEAL_MS, build_pairs
from scripts.libs.canonical.schema import (
    OPTIONAL_FOR_FILTERS,
    OPTIONAL_META,
    REQUIRED_COLUMNS,
    load_canonical_csv,
    raw_str,
)

__all__ = [
    "COLOCATED_IDEAL_MS",
    "OPTIONAL_FOR_FILTERS",
    "OPTIONAL_META",
    "REQUIRED_COLUMNS",
    "apply_eval_target_filters",
    "build_pairs",
    "derive_eval_pair_weight_min",
    "load_canonical_csv",
    "raw_str",
]
