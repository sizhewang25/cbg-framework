"""Run discovery and layout resolution for `outputs/benchmark/v2/`.

The v2 benchmark writes each run as

    <root>/<run_id>/
        summary.parquet
        <source>/<setup>/
            vps.csv  targets.csv
            clusters/{clusters,assignments}.csv + meta.json
            fold_<N>/<combo_id>/{targets.parquet,run.json,fit_checkpoint.pkl}
        eval_source/<basename>_*
        eval_dataset/<basename>_*

`(source, setup)` and `<basename>` vary per run, so nothing is hard-coded —
everything is discovered from the tree.

This root is repo-root `outputs/benchmark/v2/`, *not* the legacy
`scripts/benchmark/v2/outputs/` that `scripts/analysis/_v2_io.py` defaults to.
See ../SCHEMA.md for the full schema reference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: Repo root. Public because it is this layer's single source of truth for
#: resolving a relative path — `config.py` imports it rather than recomputing
#: the same `parents[4]`, and `bipartite.py` needs it to resolve the canonical
#: edge CSV path a run records relative to it.
REPO_ROOT = Path(__file__).resolve().parents[4]
_REPO_ROOT = REPO_ROOT

#: Default root holding `<run_id>/` benchmark output directories.
DEFAULT_OUTPUTS_ROOT = _REPO_ROOT / "outputs" / "benchmark" / "v2"

#: Where v3 analysis writes its own artifacts.
DEFAULT_ANALYSIS_ROOT = _REPO_ROOT / "outputs" / "analysis" / "v3"

_NON_SOURCE_DIRS = frozenset({"eval_source", "eval_dataset", "bench_eval"})


class MissingArtifactError(FileNotFoundError):
    """A required artifact is absent, with a hint on how to produce it."""


@dataclass(frozen=True)
class RunPaths:
    """Resolved layout for one benchmark run.

    Build via `resolve_run` / `discover_runs` rather than constructing directly.
    """

    run_id: str
    root: Path
    source: str
    setup: str

    # -- benchmark inputs -------------------------------------------------

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_id

    @property
    def setup_dir(self) -> Path:
        return self.run_dir / self.source / self.setup

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.parquet"

    @property
    def vps_csv(self) -> Path:
        return self.setup_dir / "vps.csv"

    @property
    def targets_csv(self) -> Path:
        return self.setup_dir / "targets.csv"

    @property
    def eval_source_dir(self) -> Path:
        return self.run_dir / "eval_source"

    @property
    def eval_dataset_dir(self) -> Path:
        return self.run_dir / "eval_dataset"

    # -- discovered members ----------------------------------------------

    @property
    def fold_ids(self) -> list[str]:
        """`fold_0` .. `fold_N`, ordered numerically."""
        folds = [p.name for p in self.setup_dir.glob("fold_*") if p.is_dir()]
        return sorted(folds, key=lambda f: int(f.split("_")[1]))

    @property
    def combo_ids(self) -> list[str]:
        """Combo ids holding a `targets.parquet`, unioned across folds."""
        seen: set[str] = set()
        for fold in self.setup_dir.glob("fold_*"):
            for combo in fold.iterdir():
                if combo.is_dir() and (combo / "targets.parquet").exists():
                    seen.add(combo.name)
        return sorted(seen)

    def combo_dir(self, combo_id: str, fold_id: str) -> Path:
        return self.setup_dir / fold_id / combo_id

    # -- eval_* basename --------------------------------------------------

    @property
    def eval_basename(self) -> str:
        """Stem shared by every `eval_source/` + `eval_dataset/` file.

        Derived from the source CSV name (e.g.
        `as01-20260728-20260802.mainland.sanitized`), which does not track
        `run_id` — so it is globbed, never reconstructed.
        """
        for d in (self.eval_source_dir, self.eval_dataset_dir):
            hits = sorted(d.glob("*_eval_per_target.csv"))
            if hits:
                return hits[0].name[: -len("_eval_per_target.csv")]
        raise MissingArtifactError(
            f"no *_eval_per_target.csv under {self.eval_source_dir} or "
            f"{self.eval_dataset_dir}; run `benchmark.v2.cli eval-source` for {self.run_id}"
        )

    def eval_file(self, suffix: str, *, prefer_source: bool = True) -> Path:
        """An `eval_*` artifact by suffix, e.g. `"eval_per_target.csv"`.

        `eval_source/` is a column superset of `eval_dataset/` on the operator
        runs and identical on the RIPE run, so it is preferred — except for
        `dataset_stats.json`, which only exists under `eval_dataset/`.
        """
        name = f"{self.eval_basename}_{suffix}"
        order = [self.eval_source_dir, self.eval_dataset_dir]
        if not prefer_source:
            order.reverse()
        for d in order:
            p = d / name
            if p.exists():
                return p
        raise MissingArtifactError(
            f"{name} not found under {self.eval_source_dir} or {self.eval_dataset_dir}"
        )

    # -- v3 analysis outputs ----------------------------------------------

    def analysis_dir(self, *parts: str, root: Path | None = None) -> Path:
        """`outputs/analysis/v3/<run_id>/<parts...>` (created on demand)."""
        base = (root or DEFAULT_ANALYSIS_ROOT) / self.run_id
        for p in parts:
            base = base / p
        base.mkdir(parents=True, exist_ok=True)
        return base

    def answer_space_dir(
        self, root: Path | None = None, *, grid: str, resolution: int
    ) -> Path:
        return self.analysis_dir(
            "target-answer-space", grid_slug(grid, resolution), root=root
        )

    def cls_accuracy_dir(
        self, root: Path | None = None, *, grid: str, resolution: int
    ) -> Path:
        return self.analysis_dir(
            "target-cls-accuracy", grid_slug(grid, resolution), root=root
        )

    def bipartite_dir(
        self, root: Path | None = None, *, grid: str, resolution: int
    ) -> Path:
        return self.analysis_dir(
            "bipartite-graph", grid_slug(grid, resolution), root=root
        )

    def proximity_dir(
        self, root: Path | None = None, *, grid: str, resolution: int
    ) -> Path:
        """Per-target VP proximity labels — its own tree, not the answer space's.

        `target-answer-space/` is a pure function of (target coordinates, grid),
        which is what lets `classify --answer-space` re-score under a different
        quantization. These labels also depend on the VP roster **and on RTT**,
        so writing them there would make the answer space campaign-dependent and
        leave stale labels behind a re-score. `bipartite-graph/` is likewise out:
        that module's contract is explicitly RTT-free.
        """
        return self.analysis_dir(
            "target-proximity", grid_slug(grid, resolution), root=root
        )

    def pni_graph_dir(self, root: Path | None = None) -> Path:
        """PNI site characterization — the one tree here with no grid slug.

        `grid_slug`'s docstring states the premise every other accessor rests
        on: the answer space "is a *parameter* of every number downstream of
        it", so an artifact is filed under the quantization that produced it.
        That premise is false here. Nothing in `pni-graph/` reads a seed, a cell
        or an answer space; the artifact is a pure function of (canonical CSV,
        PNI CSV). Filing it under `<grid>-<resolution>/` would make `--sweep`
        write N byte-identical copies and invite a consumer to diff two rungs
        expecting a difference.
        """
        return self.analysis_dir("pni-graph", root=root)

    def pni_feasibility_dir(self, root: Path | None = None) -> Path:
        """Speed-of-internet feasibility over the PNI sites — grid-free.

        `pni_graph_dir`'s docstring holds the argument: nothing under here reads
        a seed, a cell or an answer space, so filing it under
        `<grid>-<resolution>/` would make `--sweep` write N byte-identical
        copies and invite a consumer to diff two rungs expecting a difference.
        The feasibility test is a pure function of (RTT, six coordinates, 2/3 c).
        """
        return self.analysis_dir("pni-feasibility", root=root)

    def pni_strategy_dir(self, root: Path | None = None) -> Path:
        """Detected site-selection policy for this run's peer — grid-free.

        Grid-free for `pni_graph_dir`'s reason. Note this tree is *upstream* of
        `pni-graph/` rather than derived from it: the verdict it holds is what
        `build-pni-graph --strategy` consumes, so a stale directory here silently
        parameterizes every artifact below it.
        """
        return self.analysis_dir("pni-strategy", root=root)

    def pni_linearity_dir(self, root: Path | None = None) -> Path:
        """min-RTT-vs-distance linearity across the three x axes — grid-free.

        Grid-free for `pni_graph_dir`'s reason. Note the contrast with
        `cls_accuracy_dir`: an r-squared here is a property of the measurement
        campaign, not of the answer space, so the quantization that scores a
        *prediction* is not a parameter of it.
        """
        return self.analysis_dir("pni-linearity", root=root)

    def mtl_map_dir(
        self, root: Path | None = None, *, grid: str, resolution: int
    ) -> Path:
        """Interactive per-method MTL maps — its own tree, like `proximity_dir`.

        `plot-mtl-map` recomputes the answer space, the seed scoring and the
        proximity labels in-process rather than reading them, so it runs on a
        bare benchmark run. Writing into `target-cls-accuracy/` would imply
        `classify` had already run and make that directory's provenance
        ambiguous.
        """
        return self.analysis_dir("mtl-map", grid_slug(grid, resolution), root=root)


def grid_slug(grid: str, resolution: int) -> str:
    """`("h3", 4)` -> `"h3-4"`; `("healpix", 128)` -> `"healpix-128"`.

    Both analysis outputs are grouped by quantization, because the answer space
    is a *parameter* of every number downstream of it. Without this, a sweep
    would overwrite one `topn_accuracy.csv` once per rung and leave no record of
    which grid produced the surviving one — the same bug class the `.top<N>`
    filename suffix fixes for the top-N axis.

    The scheme name is in the slug, not just the number: `h3` res 4 and HEALPix
    nside 4 are different grids that would otherwise collide on one directory,
    and a bare number cannot say which tessellation produced it.

    `grid` and `resolution` are **required keyword arguments** on both callers
    above, so neither can be defaulted away and silently drift.
    """
    return f"{grid}-{int(resolution)}"


def discover_runs(root: Path | str = DEFAULT_OUTPUTS_ROOT) -> list[RunPaths]:
    """Every run under `root`, sorted by run_id.

    A run is any `<root>/<run_id>/<source>/<setup>/` holding `fold_*/` or
    `clusters/`. A run with several `(source, setup)` pairs yields one
    `RunPaths` each.
    """
    root = Path(root)
    if not root.is_dir():
        raise MissingArtifactError(f"outputs root does not exist: {root}")

    out: list[RunPaths] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for source_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            if source_dir.name in _NON_SOURCE_DIRS:
                continue
            for setup_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
                if any(setup_dir.glob("fold_*")) or (setup_dir / "clusters").is_dir():
                    out.append(
                        RunPaths(
                            run_id=run_dir.name,
                            root=root,
                            source=source_dir.name,
                            setup=setup_dir.name,
                        )
                    )
    return out


def resolve_run(run_id: str, root: Path | str = DEFAULT_OUTPUTS_ROOT) -> RunPaths:
    """The single run matching `run_id`.

    Raises if absent, or if it holds more than one `(source, setup)` pair.
    """
    runs = discover_runs(root)
    hits = [r for r in runs if r.run_id == run_id]
    if not hits:
        known = ", ".join(sorted({r.run_id for r in runs})) or "(none)"
        raise MissingArtifactError(f"no run {run_id!r} under {root}. Known runs: {known}")
    if len(hits) > 1:
        pairs = ", ".join(f"{r.source}/{r.setup}" for r in hits)
        raise ValueError(
            f"run {run_id!r} has multiple (source, setup) pairs: {pairs}. "
            f"Construct RunPaths directly to pick one."
        )
    return hits[0]


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())
