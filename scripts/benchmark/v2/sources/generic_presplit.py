"""GenericPresplitSource — pre-split train/test files with the canonical schema.

Same canonical column contract as `generic_csv` (one row per
`(vp, target, rtt)` observation; see that module's docstring for the full
column list), but the fit/eval partition is supplied by the caller as two
separate files instead of being derived here:

  * `train_path` — every row feeds `iter_fit_samples()` (LTD training).
  * `test_path`  — every row feeds `iter_eval_targets()` (evaluation).

No fold / wsplit / stratification logic exists in this source — the two
files are assumed to be a clean split. That assumption is enforced: a
`target_id` appearing in BOTH files is a hard error (LTD-leakage guard),
as is a `vp_id` carrying different coordinates across the two files.
VPs themselves may (and usually do) appear in both files.

The legacy `setup` role-swap axis does not apply here: the canonical
columns already fix the roles (`vp_*` = VP, `target_*` = target), so the
`setup` argument is accepted for CLI compatibility but IGNORED, and
`setup_id()` always returns the fixed path label `"vp_to_target"`.
Snakemake configs must set `setup: vp_to_target` so the workflow's
expected paths line up with what the source reports.

Each path may be a `.csv` (read with the same header-lowercasing +
NA-sentinel opt-out as `generic_csv`) or a `.parquet`; the format is
inferred from the extension, independently per file.

Slicing (`--slice`):
  all        — everything.
  head<k>    — keep the k TEST targets that sort first by target_id
               (deterministic smoke slice). The train side is untouched.

Source kwargs:
  train_path : Path | str — required; canonical-schema fit corpus.
  test_path  : Path | str — required; canonical-schema eval corpus.
  min_obs    : int = None — drop targets with fewer observations than this,
               applied per file (sparse fit targets and sparse eval targets
               are both pruned, independently).
  eval_pair_weight_min : float = None — traffic-weighted eval mask on the
               TEST file (after min_obs): an eval target survives iff >= 1
               of its obs has weight >= this value, and surviving targets
               keep only those clearing obs in iter_eval_targets. The train
               side is untouched.
  eval_kept_traffic_fraction : float = None — derive `eval_pair_weight_min`
               from test-side deduped `(vp_id, target_city)` pair weights so
               at least this pair-level traffic fraction is retained, then
               apply the same eval-only mask above.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from scripts.benchmark.v2.sources.base import (
    DataSource,
    EvalTarget,
    TgConfig,
    VpConfig,
)
from scripts.benchmark.v2.sources.generic_csv import (
    _OPTIONAL,
    _OPTIONAL_STR,
    _REQUIRED,
    _opt_col,
    _raw_str,
)
from scripts.framework.v2 import FitSample
from scripts.framework.v2.types import Coord, Latency, VpId
from scripts.processing.ripe_atlas.stratification import normalize_asn

logger = logging.getLogger(__name__)


class GenericPresplitSource(DataSource):
    """Two-file (train, test) source with the canonical schema — no splitting.

    See module docstring for the file contract and slice grammar.
    """

    name = "generic_presplit"

    # Fixed setup label: the canonical columns already assign the VP/target
    # roles, so the legacy role-swap axis is meaningless here. The `setup`
    # constructor argument is accepted (the CLI always passes one) but ignored.
    SETUP_ID = "vp_to_target"

    def __init__(
        self,
        slice: str,
        setup: Optional[str] = None,
        train_path: Optional[Path] = None,
        test_path: Optional[Path] = None,
        *,
        min_obs: Optional[int] = None,
        eval_pair_weight_min: Optional[float] = None,
        eval_kept_traffic_fraction: Optional[float] = None,
    ) -> None:
        if train_path is None or test_path is None:
            raise ValueError(
                f"{self.name!r} requires both `train_path` and `test_path` "
                f"(canonical-schema CSV or Parquet files)"
            )
        if eval_pair_weight_min is not None and eval_pair_weight_min < 0:
            raise ValueError(
                f"eval_pair_weight_min must be >= 0, got {eval_pair_weight_min}"
            )
        if (
            eval_pair_weight_min is not None
            and eval_kept_traffic_fraction is not None
        ):
            raise ValueError(
                "Pass only one of eval_pair_weight_min or "
                "eval_kept_traffic_fraction"
            )
        if (
            eval_kept_traffic_fraction is not None
            and not (0 < eval_kept_traffic_fraction <= 1)
        ):
            raise ValueError(
                "eval_kept_traffic_fraction must be in (0, 1], "
                f"got {eval_kept_traffic_fraction}"
            )
        if slice != "all" and not slice.startswith("head"):
            raise ValueError(
                f"unknown slice {slice!r}; expected 'all' or 'head<k>' "
                f"(the fit/eval partition comes from the two files, not a slice)"
            )

        self._slice = slice
        self._train_path = Path(train_path)
        self._test_path = Path(test_path)
        self._min_obs = min_obs
        self._eval_pair_weight_min = eval_pair_weight_min
        self._eval_kept_traffic_fraction = eval_kept_traffic_fraction

        # Lazily populated by `_ensure_loaded`.
        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._eval_targets: Optional[set[str]] = None

    # ---- DataSource API ------------------------------------------------------

    def slice_id(self) -> str:
        return self._slice

    def setup_id(self) -> str:
        return self.SETUP_ID

    def iter_vp_configs(self) -> Iterator[VpConfig]:
        df = self._combined()
        cols = df.columns
        for _, row in df.drop_duplicates("vp_id").iterrows():
            yield VpConfig(
                vp_id=str(row["vp_id"]),
                lat=float(row["vp_lat"]),
                lon=float(row["vp_lon"]),
                asn=normalize_asn(row.get("vp_asn")) if "vp_asn" in cols else None,
                country=_opt_col(row, "vp_country", cols),
                continent=_opt_col(row, "vp_continent", cols),
                region=_opt_col(row, "vp_region", cols),
                city=_opt_col(row, "vp_city", cols),
            )

    def iter_tg_configs(self) -> Iterator[TgConfig]:
        # Static catalog of every target the source knows about (fit ∪ eval;
        # disjoint by construction — enforced at load time). Matches the
        # convention in generic_csv.iter_tg_configs.
        df = self._combined()
        cols = df.columns
        for _, row in df.drop_duplicates("target_id").iterrows():
            yield TgConfig(
                tg_id=str(row["target_id"]),
                lat=float(row["target_lat"]),
                lon=float(row["target_lon"]),
                asn=normalize_asn(row.get("target_asn")) if "target_asn" in cols else None,
                country=_opt_col(row, "target_country", cols),
                continent=_opt_col(row, "target_continent", cols),
                region=_opt_col(row, "target_region", cols),
                city=_opt_col(row, "target_city", cols),
            )

    def iter_fit_samples(self) -> Iterator[FitSample]:
        train, _ = self._ensure_loaded()
        for row in train.itertuples(index=False):
            yield FitSample(
                vp_id=VpId(str(row.vp_id)),
                vp_coord=Coord(lat=float(row.vp_lat), lon=float(row.vp_lon)),
                probe_coord=Coord(lat=float(row.target_lat), lon=float(row.target_lon)),
                latency=Latency(float(row.rtt_ms)),
            )

    def iter_eval_targets(self) -> Iterator[EvalTarget]:
        _, test = self._ensure_loaded()
        for tg_id, group in test.groupby("target_id", sort=True):
            tg_id_str = str(tg_id)
            if self._eval_targets is not None and tg_id_str not in self._eval_targets:
                continue
            first = group.iloc[0]
            true_coord = Coord(
                lat=float(first["target_lat"]),
                lon=float(first["target_lon"]),
            )
            obs: list[tuple[VpId, Coord, Latency]] = []
            obs_weights: list[float] = []
            for r in group.itertuples(index=False):
                if (
                    self._eval_pair_weight_min is not None
                    and float(r.weight) < self._eval_pair_weight_min
                ):
                    continue
                obs.append((
                    VpId(str(r.vp_id)),
                    Coord(lat=float(r.vp_lat), lon=float(r.vp_lon)),
                    Latency(float(r.rtt_ms)),
                ))
                obs_weights.append(float(r.weight))
            yield EvalTarget(
                target_id=tg_id_str, true_coord=true_coord,
                obs=obs, obs_weights=obs_weights,
            )

    # ---- internals -----------------------------------------------------------

    def _ensure_loaded(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._train_df is None or self._test_df is None:
            train = self._load_frame(self._train_path, role="train")
            test = self._load_frame(self._test_path, role="test")
            self._check_target_overlap(train, test)
            self._check_vp_coord_conflicts(train, test)
            test = self._apply_head_slice(test)
            if self._min_obs is not None:
                train = self._apply_min_obs(train, role="train")
                test = self._apply_min_obs(test, role="test")
            self._train_df = train.reset_index(drop=True)
            self._test_df = test.reset_index(drop=True)
            # Eval weight mask runs LAST: min_obs prunes sparse targets first,
            # and this only shrinks the eval side further. Train is never touched.
            if self._eval_kept_traffic_fraction is not None:
                self._derive_eval_weight_min_from_fraction()
            if self._eval_pair_weight_min is not None:
                self._apply_eval_weight_filter()
        assert self._train_df is not None and self._test_df is not None
        return self._train_df, self._test_df

    def _combined(self) -> pd.DataFrame:
        """Train + test rows in one frame (train first, so on duplicate vp_ids
        the train file's metadata wins in drop_duplicates)."""
        train, test = self._ensure_loaded()
        return pd.concat([train, test], ignore_index=True, sort=False)

    def _load_frame(self, path: Path, role: str) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".csv":
            # Same NA-sentinel opt-out as generic_csv: literal "NA" cells in
            # free-text columns (continent North America, country Namibia)
            # must not parse as NaN. Parquet is typed, so it needs no converters.
            converters = {c: _raw_str for col in _OPTIONAL_STR for c in (col, col.upper())}
            df = pd.read_csv(path, converters=converters)
        else:
            raise ValueError(
                f"{role} file {path}: unsupported extension {suffix!r} "
                f"(expected .csv or .parquet)"
            )
        df.columns = df.columns.str.lower()
        missing = [c for c in _REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(
                f"{role} file {path} missing required columns: {missing}. "
                f"Required: {list(_REQUIRED)}; optional: {list(_OPTIONAL)}."
            )
        df = df.dropna(subset=list(_REQUIRED))
        df = df[df["rtt_ms"] > 0].copy()
        if df.empty:
            raise ValueError(
                f"{role} file {path} has no valid rows after dropping "
                f"NaN-required / non-positive-RTT rows"
            )
        return self._normalize_weight(df, path)

    @staticmethod
    def _normalize_weight(df: pd.DataFrame, path: Path) -> pd.DataFrame:
        """Guarantee a numeric non-negative `weight` column. Same two defaults
        as generic_csv: column absent → 1.0 everywhere ("no weight notion");
        cell NaN in a present column → 0.0 ("unknown traffic = weightless")."""
        if "weight" not in df.columns:
            df["weight"] = 1.0
            return df
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        n_missing = int(df["weight"].isna().sum())
        if n_missing:
            logger.info(
                "%s: weight: %d missing/non-numeric cells defaulted to 0.0 "
                "(unknown traffic = weightless)",
                path, n_missing,
            )
            df["weight"] = df["weight"].fillna(0.0)
        if (df["weight"] < 0).any():
            raise ValueError(
                f"{path}: weight must be >= 0 (found negative values)"
            )
        return df

    def _check_target_overlap(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> None:
        """Hard error on any target_id present in both files — the whole point
        of the pre-split contract is that the eval targets never reached LTD
        training."""
        overlap = (
            set(train["target_id"].astype(str))
            & set(test["target_id"].astype(str))
        )
        if overlap:
            examples = sorted(overlap)[:5]
            raise ValueError(
                f"{len(overlap)} target_id(s) appear in BOTH {self._train_path} "
                f"and {self._test_path} (e.g. {examples}) — the two files must "
                f"be a clean train/test split (LTD-leakage guard)"
            )

    def _check_vp_coord_conflicts(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> None:
        """A vp_id shared across files must carry identical coordinates —
        otherwise the union vp_configs catalog would silently pick one."""
        def first_coords(df: pd.DataFrame) -> pd.DataFrame:
            out = df.drop_duplicates("vp_id")[["vp_id", "vp_lat", "vp_lon"]].copy()
            out["vp_id"] = out["vp_id"].astype(str)
            return out.set_index("vp_id")

        a, b = first_coords(train), first_coords(test)
        shared = a.index.intersection(b.index)
        if shared.empty:
            return
        mismatch = shared[
            (a.loc[shared, "vp_lat"].to_numpy() != b.loc[shared, "vp_lat"].to_numpy())
            | (a.loc[shared, "vp_lon"].to_numpy() != b.loc[shared, "vp_lon"].to_numpy())
        ]
        if len(mismatch):
            examples = sorted(mismatch)[:5]
            raise ValueError(
                f"{len(mismatch)} vp_id(s) have conflicting coordinates between "
                f"{self._train_path} and {self._test_path} (e.g. {examples})"
            )

    def _apply_head_slice(self, test: pd.DataFrame) -> pd.DataFrame:
        """`head<k>` keeps the k test targets that sort first by target_id."""
        if self._slice == "all":
            return test
        try:
            k = int(self._slice.removeprefix("head"))
        except ValueError as e:
            raise ValueError(f"invalid head-k slice: {self._slice!r}") from e
        if k < 1:
            raise ValueError(f"head-k must be >=1, got {k}")
        keep = sorted(test["target_id"].astype(str).unique())[:k]
        return test[test["target_id"].astype(str).isin(keep)].copy()

    def _apply_min_obs(self, df: pd.DataFrame, role: str) -> pd.DataFrame:
        assert self._min_obs is not None
        counts = df.groupby("target_id")["target_id"].transform("count")
        before = df["target_id"].nunique()
        df = df[counts >= self._min_obs].reset_index(drop=True)
        after = df["target_id"].nunique()
        logger.info(
            "min_obs=%d (%s): %d → %d targets", self._min_obs, role, before, after
        )
        if df.empty:
            raise ValueError(
                f"min_obs={self._min_obs} left zero {role} targets "
                f"({self._train_path if role == 'train' else self._test_path})"
            )
        return df

    def _apply_eval_weight_filter(self) -> None:
        """Traffic-weighted eval mask (`eval_pair_weight_min`). An eval target
        survives iff >= 1 of its obs has weight >= the threshold;
        iter_eval_targets then emits only the clearing obs. Train is untouched."""
        assert self._test_df is not None and self._eval_pair_weight_min is not None
        df = self._test_df
        thr = self._eval_pair_weight_min
        base = set(df["target_id"].astype(str))
        kept = set(df.loc[df["weight"] >= thr, "target_id"].astype(str))
        if not kept:
            raise ValueError(
                f"eval_pair_weight_min={thr} left zero eval targets in "
                f"{self._test_path} — the threshold exceeds the data's weights"
            )
        logger.info(
            "eval_pair_weight_min=%s: eval targets %d → %d (train side untouched)",
            thr, len(base), len(kept),
        )
        self._eval_targets = kept

    def _derive_eval_weight_min_from_fraction(self) -> None:
        """Derive eval_pair_weight_min from test-side pair traffic retention:
        dedupe at `(vp_id, target_city)` by per-pair max(weight), then
        descending cumulative sum to the requested kept fraction. Mirrors
        generic_csv._derive_eval_weight_min_from_fraction."""
        assert self._test_df is not None and self._eval_kept_traffic_fraction is not None
        eval_df = self._test_df
        frac = self._eval_kept_traffic_fraction

        if "target_city" not in eval_df.columns:
            raise ValueError(
                "eval_kept_traffic_fraction requires a target_city column "
                f"in the test file ({self._test_path})"
            )
        city = eval_df["target_city"].astype(str).str.strip()
        blank_city = ~eval_df["target_city"].notna() | (city == "")
        if blank_city.any():
            raise ValueError(
                "eval_kept_traffic_fraction requires non-blank target_city "
                f"on every test row ({self._test_path})"
            )

        per_pair = (
            eval_df.groupby(["vp_id", "target_city"], as_index=False)
            .agg(weight=("weight", "max"), distinct_values=("weight", "nunique"))
        )
        inconsistent_pairs = int((per_pair["distinct_values"] > 1).sum())

        weights = per_pair["weight"].to_numpy(dtype=float)
        total = float(weights.sum())
        if total <= 0:
            self._eval_pair_weight_min = 0.0
            logger.info(
                "eval_kept_traffic_fraction=%.3f: all test-side pair weights "
                "are zero; derived eval_pair_weight_min=0.0",
                frac,
            )
            return

        weights_sorted = np.sort(weights)[::-1]
        target = frac * total
        cum = np.cumsum(weights_sorted)
        idx = int(np.searchsorted(cum, target, side="left"))
        idx = min(idx, len(weights_sorted) - 1)
        threshold = float(weights_sorted[idx])

        kept_pairs = int((weights >= threshold).sum())
        achieved = float(weights[weights >= threshold].sum() / total)
        self._eval_pair_weight_min = threshold
        logger.info(
            "eval_kept_traffic_fraction=%.3f: derived "
            "eval_pair_weight_min=%.12g from %d test-side (vp_id,target_city) "
            "pairs; kept_pairs=%d (%.2f%% traffic)",
            frac,
            threshold,
            len(per_pair),
            kept_pairs,
            100 * achieved,
        )
        if inconsistent_pairs:
            logger.info(
                "  note: %d test-side (vp_id,target_city) pairs had "
                "non-identical row weights; used per-pair max(weight)",
                inconsistent_pairs,
            )
