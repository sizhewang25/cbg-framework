"""TrafficWeightedCSVSource — K-fold on a mesh, evaluated on the traffic-weighted subset.

Subclasses `GenericCSVSource`, so the canonical column contract, the slice
grammar (`all` / `head<k>` / `fold_N`), DistGeo stratification and `min_obs` are
all inherited unchanged. The one thing added is an eval-side mask.

WHAT IT DOES
------------
Fit always sees the **full mesh** at full edge density. Eval is the slice's
targets intersected with the targets that survive traffic filtering, scored on
their surviving flows only. That is the §7.3 protocol: K-fold handles data
scarcity, the fold boundary handles leakage, and traffic restricts only what is
scored.

THE WEIGHTED SUBSET IS A SET OF FLOWS, NOT A THRESHOLD
------------------------------------------------------
Both modes converge on one object -- a set of `(vp_id, target_id)` keys:

  precomputed   `weighted_csv_path` given -> the set is exactly the flows
                present in that file, whatever produced it.
  on-the-fly    no path -> derive a threshold from the mesh via
                `eval_kept_traffic_fraction` / `eval_pair_weight_min`, and the
                set is the flows clearing it.

Membership rather than a re-derived threshold, because §7.3 says "keep only the
flows that appear in the traffic-weighted dataset" -- a set, not a number -- and
because it stays correct for a weighted CSV produced by a rule this code does
not know about.

Two consequences worth knowing:

  * Precomputed mode consults no weights at all, so the mesh needs no `weight`
    column. On-the-fly mode does, and refuses a weightless mesh (see below).
  * The two modes are interchangeable at the same fraction: running
    `derive_traffic_weighted_cbg_data.smk` at 0.95 and passing its output is
    equivalent to passing `eval_kept_traffic_fraction: 0.95`, because both run
    the same keyless whole-mesh derivation. Use precomputed when you also need
    the CSV for plotting or dataset characterisation; use on-the-fly for sweeps
    where materializing a file per fraction is wasteful.

Source kwargs:
  mesh_csv_path      : Path | str   — required; the full mesh (fit corpus).
  weighted_csv_path  : Path | str   — precomputed subset. Mutually exclusive
                       with the two threshold kwargs.
  eval_pair_weight_min : float      — on-the-fly: keep flows with
                       `weight >= this`.
  eval_kept_traffic_fraction : float — on-the-fly: KEYLESS, WHOLE-MESH
                       derivation of that threshold. `total` sums `weight` over
                       every row in the mesh (not just eval-side rows, and not
                       deduped by any key); flows are ranked descending and
                       cumulated, and the smallest prefix reaching
                       `frac * total` fixes the threshold. Whole-mesh
                       normalization is what makes the threshold
                       FOLD-INDEPENDENT -- every `fold_N` of one mesh derives
                       the same number, so folds stay comparable and "survived
                       traffic filtering" is a property of the traffic rather
                       than of the partition. Mesh weights are typically shares
                       of a larger universe (the mesh itself samples top targets
                       per location) and need not sum to 1; renormalizing by the
                       mesh total is what makes the fraction meaningful.
  k / seed / asn_bucket_top_n / min_obs — as `GenericCSVSource`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from scripts.benchmark.v2.sources.base import DataSource, EvalTarget
from scripts.benchmark.v2.sources.generic_csv import GenericCSVSource

logger = logging.getLogger(__name__)

_FLOW_KEY = ("vp_id", "target_id")


class TrafficWeightedCSVSource(GenericCSVSource):
    """Mesh-backed source whose eval side is restricted to traffic-heavy flows.

    See module docstring for the two modes and the fold-independence argument.
    """

    name = "traffic_weighted_csv"

    def __init__(
        self,
        slice: str,
        setup: str = DataSource.ANCHORS_TO_PROBES,
        mesh_csv_path: Optional[Path] = None,
        weighted_csv_path: Optional[Path] = None,
        *,
        k: int = 5,
        seed: int = 42,
        asn_bucket_top_n: int = 20,
        min_obs: Optional[int] = None,
        eval_pair_weight_min: Optional[float] = None,
        eval_kept_traffic_fraction: Optional[float] = None,
    ) -> None:
        if mesh_csv_path is None:
            raise ValueError(
                f"{self.name!r} requires `mesh_csv_path` (the full mesh CSV; it "
                f"is the fit corpus and the universe the weighted subset lives in)"
            )

        n_modes = sum(
            x is not None
            for x in (weighted_csv_path, eval_pair_weight_min, eval_kept_traffic_fraction)
        )
        if n_modes == 0:
            raise ValueError(
                f"{self.name!r} needs the traffic-weighted subset defined exactly "
                f"once: pass `weighted_csv_path` (precomputed) or one of "
                f"`eval_kept_traffic_fraction` / `eval_pair_weight_min` (on-the-fly)"
            )
        if n_modes > 1:
            # A weighted CSV already encodes a cut; re-cutting it would compose
            # two filters and the reported fraction would refer to neither.
            raise ValueError(
                "Pass exactly one of weighted_csv_path, eval_kept_traffic_fraction "
                "or eval_pair_weight_min — they are three ways of naming the same "
                f"subset (got {n_modes})"
            )
        if eval_pair_weight_min is not None and eval_pair_weight_min < 0:
            raise ValueError(
                f"eval_pair_weight_min must be >= 0, got {eval_pair_weight_min}"
            )
        if eval_kept_traffic_fraction is not None and not (
            0 < eval_kept_traffic_fraction <= 1
        ):
            raise ValueError(
                "eval_kept_traffic_fraction must be in (0, 1], "
                f"got {eval_kept_traffic_fraction}"
            )

        super().__init__(
            slice, setup, mesh_csv_path,
            k=k, seed=seed, asn_bucket_top_n=asn_bucket_top_n, min_obs=min_obs,
        )
        self._weighted_csv_path = (
            Path(weighted_csv_path) if weighted_csv_path is not None else None
        )
        self._eval_pair_weight_min = eval_pair_weight_min
        self._eval_kept_traffic_fraction = eval_kept_traffic_fraction
        self._weighted_flows: Optional[set[tuple[str, str]]] = None

    # ---- DataSource API ------------------------------------------------------

    def iter_eval_targets(self) -> Iterator[EvalTarget]:
        # Force the mask to be built before the base class starts grouping, so
        # `_keep_obs` has a populated flow set.
        self._ensure_loaded()
        yield from super().iter_eval_targets()

    # ---- extension hooks -----------------------------------------------------

    def _keep_obs(self, row) -> bool:
        assert self._weighted_flows is not None
        return (str(row.vp_id), str(row.target_id)) in self._weighted_flows

    # ---- internals -----------------------------------------------------------

    def _ensure_loaded(self) -> pd.DataFrame:
        if self._df is None:
            # Loads the mesh, stratifies, applies min_obs. The mask runs after,
            # so a target already pruned by min_obs can never come back.
            super()._ensure_loaded()
            self._build_weighted_flows()
            self._apply_weighted_flow_filter()
        assert self._df is not None
        return self._df

    def _build_weighted_flows(self) -> None:
        if self._weighted_csv_path is not None:
            self._weighted_flows = self._load_weighted_flows()
        else:
            self._weighted_flows = self._derive_weighted_flows()

    def _mesh_flows_from_file(self) -> set[tuple[str, str]]:
        """Flow keys of the mesh *file*, before any slice or `min_obs` drop.

        The subset guard below is about file-pair integrity -- "were these two
        files derived from each other" -- so it must compare against the mesh as
        written. Comparing against `self._df` would flag flows that `min_obs` or
        a `head<k>` slice legitimately removed as if the files were mismatched.
        """
        df = pd.read_csv(self._csv_path, usecols=lambda c: c.lower() in _FLOW_KEY)
        df.columns = df.columns.str.lower()
        return set(zip(df["vp_id"].astype(str), df["target_id"].astype(str)))

    def _load_weighted_flows(self) -> set[tuple[str, str]]:
        """Precomputed mode: the subset is the flows present in the file."""
        assert self._weighted_csv_path is not None
        df = pd.read_csv(self._weighted_csv_path)
        df.columns = df.columns.str.lower()
        missing = [c for c in _FLOW_KEY if c not in df.columns]
        if missing:
            raise ValueError(
                f"weighted CSV {self._weighted_csv_path} missing {missing}; "
                f"columns present: {list(df.columns)}"
            )

        keys = list(zip(df["vp_id"].astype(str), df["target_id"].astype(str)))
        flows = set(keys)
        if len(flows) != len(keys):
            raise ValueError(
                f"weighted CSV {self._weighted_csv_path} is not edge-unique: "
                f"{len(keys) - len(flows)} duplicate (vp_id, target_id) rows"
            )
        if not flows:
            raise ValueError(f"weighted CSV {self._weighted_csv_path} has no rows")

        # A flow the mesh has never seen means the two files do not describe the
        # same measurement campaign — a stale or mismatched pair. Left unchecked
        # it silently yields an eval set that is neither file's.
        orphans = flows - self._mesh_flows_from_file()
        if orphans:
            raise ValueError(
                f"{len(orphans)} flow(s) in {self._weighted_csv_path} are absent "
                f"from the mesh {self._csv_path} (e.g. {sorted(orphans)[:3]}) — "
                f"the weighted CSV must be a subset of the mesh it was derived "
                f"from. Regenerate it, or check the two paths refer to the same "
                f"dataset."
            )
        logger.info(
            "weighted subset: %d flows loaded from %s (precomputed)",
            len(flows), self._weighted_csv_path,
        )
        return flows

    def _derive_weighted_flows(self) -> set[tuple[str, str]]:
        """On-the-fly mode: threshold the mesh's own weights."""
        assert self._df is not None
        if not self._weight_column_present:
            # The synthesized all-1.0 fill would make every threshold <= 1.0
            # retain 100% of flows, so the run would be published as the
            # "weighted" arm while being byte-identical to the unweighted one.
            raise ValueError(
                f"mesh {self._csv_path} has no 'weight' column, but an "
                f"on-the-fly traffic-weighted eval was requested. Either point "
                f"at a weight-bearing mesh, or pass `weighted_csv_path` (which "
                f"needs no weights, since it names the flows directly)."
            )
        if self._eval_kept_traffic_fraction is not None:
            self._derive_eval_weight_min_from_fraction()

        assert self._eval_pair_weight_min is not None
        thr = self._eval_pair_weight_min
        kept = self._df[self._df["weight"] >= thr]
        logger.info(
            "weighted subset: %d of %d flows at weight >= %.12g (on-the-fly)",
            len(kept), len(self._df), thr,
        )
        return set(zip(kept["vp_id"].astype(str), kept["target_id"].astype(str)))

    def _derive_eval_weight_min_from_fraction(self) -> None:
        """Keyless, whole-mesh threshold from a kept-traffic fraction.

        WHOLE-MESH — `total` sums `weight` over every row in the mesh, not just
        the eval-side rows. That is what makes the threshold FOLD-INDEPENDENT:
        every `fold_N` of one mesh derives the same number, so folds stay
        comparable. Normalizing over eval-side rows would give each fold its own
        cut, and "the traffic-weighted dataset" would not be one dataset.
        """
        assert self._df is not None and self._eval_kept_traffic_fraction is not None
        frac = self._eval_kept_traffic_fraction

        weights = self._df["weight"].to_numpy(dtype=float)
        total = float(weights.sum())
        if total <= 0:
            raise ValueError(
                f"eval_kept_traffic_fraction={frac} needs a traffic signal, but "
                f"the total weight over {len(weights)} flows in "
                f"{self._csv_path} is {total} — every flow is weightless"
            )

        weights_sorted = np.sort(weights)[::-1]
        cum = np.cumsum(weights_sorted)
        idx = int(np.searchsorted(cum, frac * total, side="left"))
        # Load-bearing clamp: `cum` sums the sorted array while `total` sums the
        # original, so at frac=1.0 float error can put cum[-1] one ULP below
        # `total` and push searchsorted past the end.
        idx = min(idx, len(weights_sorted) - 1)
        threshold = float(weights_sorted[idx])

        kept = weights >= threshold
        achieved = float(weights[kept].sum() / total)
        self._eval_pair_weight_min = threshold
        logger.info(
            "eval_kept_traffic_fraction=%.3f: derived eval_pair_weight_min=%.12g "
            "over %d whole-mesh flows (total weight %.12g); kept_flows=%d (%.2f%% traffic)",
            frac, threshold, len(weights), total, int(kept.sum()), 100 * achieved,
        )
        if achieved - frac > 0.01:
            logger.warning(
                "  kept traffic overshoots target by %.2f pp: %d flows tie at "
                "the threshold weight %.12g and all are kept (>= semantics)",
                100 * (achieved - frac),
                int((weights == threshold).sum()),
                threshold,
            )

    def _apply_weighted_flow_filter(self) -> None:
        """Intersect the slice's eval targets with the weighted subset.

        Fit targets and fit samples are untouched — training always sees the
        full mesh; only the evaluated view is traffic-restricted."""
        assert self._df is not None and self._weighted_flows is not None
        surviving = {t for _, t in self._weighted_flows}
        base = (
            self._eval_targets
            if self._eval_targets is not None
            else set(self._df["target_id"].astype(str))
        )
        kept = base & surviving
        if not kept:
            raise ValueError(
                f"the traffic-weighted subset left zero eval targets in slice "
                f"{self._slice!r}: none of its {len(base)} targets carries a "
                f"surviving flow"
            )
        logger.info(
            "traffic-weighted eval: targets %d → %d (fit side untouched)",
            len(base), len(kept),
        )
        self._eval_targets = kept
