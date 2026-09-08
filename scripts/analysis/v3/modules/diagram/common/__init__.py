"""What both layouts need: names, colours, membership, counts, primitives.

Split from the figures so a change to how a variant is named or coloured lands
in one place and reaches every figure at once. Nothing here draws a diagram;
`draw` holds only the primitives a diagram is assembled from.
"""

from scripts.analysis.v3.modules.diagram.common.labels import (
    LABELS,
    PREFERRED_ORDER,
    PUBLISHED_METHODS,
    RING_LETTERS,
    artifact_name,
    label_for,
    region_key,
    ring_letter_map,
    short_label,
)
from scripts.analysis.v3.modules.diagram.common.membership import (
    RUN_KEY_SEP,
    available_methods,
    build_membership,
    pooled_membership,
    restrict_to_baseline_failures,
)
from scripts.analysis.v3.modules.diagram.common.palette import method_colors
from scripts.analysis.v3.modules.diagram.common.tables import (
    SET_IDS,
    exact_combination_counts,
    intersection_table,
    pairwise_table,
    venn_spec,
)

__all__ = [
    "LABELS",
    "PREFERRED_ORDER",
    "PUBLISHED_METHODS",
    "RING_LETTERS",
    "RUN_KEY_SEP",
    "SET_IDS",
    "artifact_name",
    "available_methods",
    "build_membership",
    "exact_combination_counts",
    "intersection_table",
    "label_for",
    "method_colors",
    "pairwise_table",
    "pooled_membership",
    "region_key",
    "restrict_to_baseline_failures",
    "ring_letter_map",
    "short_label",
    "venn_spec",
]
