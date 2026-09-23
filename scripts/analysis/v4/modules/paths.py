"""Where v4 reads from and writes to.

Reads the **v2 benchmark** output tree unchanged — v4 replaces the analysis
layer, not the benchmark, so it inherits that tree's population and fold
contract as-is. Writes to its own root so v3's artifacts survive for comparison;
the two are not comparable (nside 128 is 50.9 km against h3-4's 45.2 km, and the
correctness rule differs) and keeping both on disk is what lets the difference be
measured rather than argued about.

## One rung per directory

Artifacts are keyed by `healpix-<nside>`, following v3. The alternative — one
long file with an `nside` column — reads better for plotting, but a rung's
`accuracy.csv` is a complete, self-describing artifact under this layout, and a
sweep cannot overwrite one rung's file with another's. The ladder is joined
instead by the **merged** files that sit one level *above* the rung directories
(`accuracy_by_resolution.healpix.csv`, `grid_sweep.healpix.csv`), which is the
shape v3 already established for `grid_sweep`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: `.../cbg-framework`, four parents up from `v4/modules/paths.py`.
REPO_ROOT = Path(__file__).resolve().parents[4]

#: The v2 benchmark tree v4 scores. Not v4's own.
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs" / "benchmark" / "v2"

#: Where v4 writes. Deliberately beside v3's rather than over it.
DEFAULT_ANALYSIS_ROOT = REPO_ROOT / "outputs" / "analysis" / "v4"

#: Directories under a run that are not a measurement `source`.
_NON_SOURCE_DIRS = frozenset({"eval_source", "eval_dataset", "bench_eval"})


#: The per-run analysis kind holding the scoring artifacts. Named because two
#: directories are built from it: the `healpix-<n>/` rungs, and their rung-free
#: parent, where the grid-independent figures land.
CLS_ACCURACY_KIND = "target-cls-accuracy"


class MissingArtifactError(FileNotFoundError):
    """A required artifact is absent, with a hint on how to produce it."""


def grid_slug(nside: int) -> str:
    """`128` -> `"healpix-128"`.

    Carries the scheme even though v4 only knows one grid: these directories sit
    beside v3's `h3-4/` in spirit and are read by eye, so a bare `128` would be
    ambiguous about which tessellation produced it.
    """
    return f"healpix-{int(nside)}"


@dataclass(frozen=True)
class RunPaths:
    """Resolved layout for one benchmark run. Build via `resolve_run`."""

    run_id: str
    root: Path
    source: str
    setup: str

    # -- benchmark inputs (v2 tree, read-only) ----------------------------

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_id

    @property
    def setup_dir(self) -> Path:
        return self.run_dir / self.source / self.setup

    @property
    def eval_source_dir(self) -> Path:
        return self.run_dir / "eval_source"

    def eval_file(self, suffix: str) -> Path:
        """`eval_source/<basename>_<suffix>` — the dataset-scored sidecar.

        The basename is the canonical CSV's, not the run id, so it is globbed
        rather than constructed. Exactly one match is required: two would mean
        two datasets were scored into one run and picking either silently
        changes the population.
        """
        hits = sorted(self.eval_source_dir.glob(f"*_{suffix}"))
        if not hits:
            raise MissingArtifactError(
                f"no eval_source/*_{suffix} under {self.eval_source_dir}; "
                f"run the benchmark's eval-source stage first"
            )
        if len(hits) > 1:
            raise MissingArtifactError(
                f"{len(hits)} eval_source/*_{suffix} files under "
                f"{self.eval_source_dir}: {[h.name for h in hits]}. Each scores a "
                f"different dataset; keep one."
            )
        return hits[0]

    @property
    def fold_ids(self) -> list[str]:
        """`fold_0` .. `fold_N`, ordered numerically rather than lexically —
        `fold_10` must not sort between `fold_1` and `fold_2`."""
        folds = [p.name for p in self.setup_dir.glob("fold_*") if p.is_dir()]
        return sorted(folds, key=lambda f: int(f.split("_")[1]))

    @property
    def combo_ids(self) -> list[str]:
        """Combo ids holding a `targets.parquet`, unioned across folds.

        Read from the output tree, not from a config: a combo commented out of
        its YAML but still on disk is still scoreable, and a combo in the YAML
        that never ran is not. This is the same rule v3 uses, and the same trap
        — parking an arm means moving its directory, not editing the config.
        """
        if not self.setup_dir.is_dir():
            return []
        seen: set[str] = set()
        for fold in self.setup_dir.glob("fold_*"):
            for combo in fold.iterdir():
                if combo.is_dir() and (combo / "targets.parquet").exists():
                    seen.add(combo.name)
        return sorted(seen)

    def combo_dir(self, combo_id: str, fold_id: str) -> Path:
        return self.setup_dir / fold_id / combo_id

    # -- v4 outputs -------------------------------------------------------

    def analysis_dir(self, kind: str, *, root: Path | None = None) -> Path:
        base = (root or DEFAULT_ANALYSIS_ROOT) / self.run_id / kind
        base.mkdir(parents=True, exist_ok=True)
        return base

    def rung_dir(self, kind: str, nside: int, *, root: Path | None = None) -> Path:
        """`<root>/<run_id>/<kind>/healpix-<nside>/`, created."""
        out = self.analysis_dir(kind, root=root) / grid_slug(nside)
        out.mkdir(parents=True, exist_ok=True)
        return out

    def answer_space_dir(self, nside: int, *, root: Path | None = None) -> Path:
        return self.rung_dir("target-answer-space", nside, root=root)

    def bipartite_dir(self, nside: int, *, root: Path | None = None) -> Path:
        return self.rung_dir("bipartite-graph", nside, root=root)

    def cls_accuracy_dir(self, nside: int, *, root: Path | None = None) -> Path:
        return self.rung_dir(CLS_ACCURACY_KIND, nside, root=root)

    def cls_accuracy_root(self, *, root: Path | None = None) -> Path:
        """The rung-free parent of the `healpix-<n>/` directories.

        Home for anything scoring produces that does **not** vary with the
        grid. `error_km` is the case: it is prediction-to-target, so the error
        CDF is identical at every rung and writing it four times would invite
        a reader to look for a difference that cannot exist.
        """
        return self.analysis_dir(CLS_ACCURACY_KIND, root=root)


def discover_runs(root: Path | str = DEFAULT_OUTPUTS_ROOT) -> list[RunPaths]:
    """Every `<root>/<run_id>/<source>/<setup>/` holding `fold_*`.

    A run with several `(source, setup)` pairs yields one `RunPaths` each.
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
                if any(setup_dir.glob("fold_*")):
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

    Raises on absence, and on ambiguity rather than picking one: a run with two
    `(source, setup)` pairs has two populations, and silently scoring one of them
    would put a number in a table that nobody could reproduce.
    """
    matches = [r for r in discover_runs(root) if r.run_id == run_id]
    if not matches:
        known = sorted({r.run_id for r in discover_runs(root)})
        raise MissingArtifactError(
            f"no run {run_id!r} under {root}. Known: {known}"
        )
    if len(matches) > 1:
        pairs = [f"{r.source}/{r.setup}" for r in matches]
        raise MissingArtifactError(
            f"run {run_id!r} holds several (source, setup) pairs: {pairs}. "
            f"Each is a different population; score them separately."
        )
    return matches[0]
