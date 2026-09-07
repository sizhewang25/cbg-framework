"""The vocabulary every figure shares: method names, filenames, region keys.

Nothing here draws. These are the strings a reader matches across a figure, a
CSV and a manifest, so each is defined once and imported rather than formatted
at the call site — which is how a variant ends up named two ways in one run.
"""

from __future__ import annotations

from scripts.analysis.v3.modules.classify import SHORTEST_PING


#: Display labels. Keyed by method id; the Octant-Spline combo id differs by run
#: (`octant_cbg_spl` on the operator runs, `octant_cbg` on the RIPE run).
LABELS: dict[str, str] = {
    SHORTEST_PING: "Shortest-Ping",
    "million_scale_cbg": "SoI CBG",
    "vanilla_cbg": "Vanilla CBG",
    "octant_cbg_hull": "Octant-Hull CBG",
    "octant_cbg_spl": "Octant-Spline CBG",
    "octant_cbg": "Octant-Spline CBG",
    "spotter_cbg": "Spotter CBG",
}


#: Baseline first, then calibration-free, then increasingly fitted.
PREFERRED_ORDER: tuple[str, ...] = (
    SHORTEST_PING,
    "million_scale_cbg",
    "vanilla_cbg",
    "octant_cbg_hull",
    "octant_cbg_spl",
    "octant_cbg",
    "spotter_cbg",
)


#: The six published variants, by id, in §8.1's print order — the method set
#: every paper table and figure reports by default.
#:
#: Distinct from `PREFERRED_ORDER`, which is a *sort key* and therefore lists
#: both spellings of Octant-Spline so either sorts correctly. A method list can
#: only hold one, and this holds the operator runs' spelling: `as7018_us_test01`
#: names it `octant_cbg` and is out of scope until it is re-run with its columns
#: consolidated onto the other AS runs' schema.
PUBLISHED_METHODS: tuple[str, ...] = (
    SHORTEST_PING,
    "million_scale_cbg",
    "vanilla_cbg",
    "octant_cbg_hull",
    "octant_cbg_spl",
    "spotter_cbg",
)


def artifact_name(
    stem: str, ext: str, top_n: int, grid: str | None = None
) -> str:
    """`("overlap_upset", "png", 3)` -> `"overlap_upset.top3.png"`.

    Every overlap artifact is a function of `top_n`, so the suffix is
    mandatory: without it a top-3 run silently overwrites the top-1 files.

    `grid` is the `paths.grid_slug` (`"h3-4"`), and is **optional because the
    two output trees are quantized differently**. Per-run artifacts already live
    under `target-cls-accuracy/<grid>-<res>/`, so repeating it in the filename
    would be noise — and would rename files that already exist on disk. The
    cross-run directory is keyed by dataset set only, so there the slug is the
    sole thing keeping an h3-4 run from overwriting a healpix-128 one.
    """
    grid_part = f"{grid}." if grid else ""
    return f"{stem}.{grid_part}top{top_n}.{ext}"


def label_for(method: str) -> str:
    """Display name, with every CBG variant marked as one.

    Shortest-Ping is the only non-CBG method here, so anything else gets a
    `CBG` suffix — including the ablation arms, which have no entry in `LABELS`
    and would otherwise appear as bare combo ids indistinguishable from the
    baseline at a glance.
    """
    label = LABELS.get(method)
    if label is not None:
        return label
    return method if method == SHORTEST_PING else f"{method} CBG"


def short_label(method: str) -> str:
    """`label_for` with the trailing " CBG" dropped.

    Every method but Shortest-Ping is a CBG variant and the surrounding figure
    or table header says so, so repeating it once per column only makes the
    labels wide enough to collide. Written for `plot-pareto`, which still
    re-exports it; moved here when `table-headline` became the second caller,
    since it is a pure function of `label_for` and pulling it from `pareto`
    would drag matplotlib into a table command.
    """
    label = label_for(method)
    return label[: -len(" CBG")] if label.endswith(" CBG") else label


# ---------------------------------------------------------------------------
# § region keys
# ---------------------------------------------------------------------------
#
# A letter per set, so an intersection can be named in the width a table column
# or a figure label allows. Assigned by **ring position**, which is why they
# live here rather than in `venn/ring.py`: `euler_fit_table` keys its `region`
# column on the same letters, so both packages need them and neither owns them.


#: Single-letter aliases for the ring's circles, in position order from 12
#: o'clock. Eight letters is exactly `RING_MAX_SETS`, and the alphabet is what
#: needed because a set's identity has to fit where it is drawn, and
#: "Octant-Spline CBG" only just does on a circle of its own.
RING_LETTERS = "ABCDEFGH"


def ring_letter_map(order: list[str]) -> dict[str, str]:
    """Method id -> its single letter, assigned by **ring position**.

    Position rather than identity, unlike `method_colors`: the letter is a
    coordinate on this figure ("the circle at 12 o'clock is A"), so it has to
    follow `--ring-order`. Colour is the thing that stays pinned to the variant
    across figures, and the two together let a reader match a key region to its
    circle either way.
    """
    if len(order) > len(RING_LETTERS):
        raise ValueError(
            f"the ring takes at most {len(RING_LETTERS)} methods, got {len(order)}"
        )
    return {method: RING_LETTERS[i] for i, method in enumerate(order)}


def region_key(order: list[str], members) -> str:
    """`{vanilla_cbg, shortest_ping}` -> `"AC"`: the members' letters, in ring order.

    Concatenated rather than joined with `∧`. The five-member regions of a
    six-ring are ~12 pt across, so every separator character is one the label
    cannot afford; the caption states the convention instead.
    """
    letters = ring_letter_map(order)
    return "".join(letters[m] for m in order if m in members)
