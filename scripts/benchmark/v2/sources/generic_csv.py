"""GenericCSVSource — adapt any CSV with the canonical schema into the v2 benchmark.

Canonical schema (one row per `(vp, target, rtt)` observation):

  required:
    vp_id, vp_lat, vp_lon            str, float, float    — entity acting as VP
    target_id, target_lat, target_lon str, float, float   — entity being geolocated
    rtt_ms                           float (>0)           — strictly positive RTT
  optional:
    vp_asn, vp_country, vp_continent, vp_region, vp_city
    target_asn, target_country, target_continent, target_region, target_city
    weight                      float (>=0)          — traffic weight of the
                                     (vp, target) pair, in whatever unit the
                                     dataset uses (e.g. TB). Column absent →
                                     1.0 everywhere (target weight degenerates
                                     to obs count); column present but cell
                                     NaN → 0.0 (unknown traffic = weightless).

`vp_*` columns **always** supply the VP-role data; `target_*` columns **always**
supply the target-role data. The `setup` flag is descriptive metadata only —
it does not affect column routing. Users canonicalize their CSV per the
desired role assignment before pointing this source at it. For RIPE Atlas /
Vultr-style data (anchors-as-VPs by convention), anchor data goes into the
`vp_*` columns and probe data goes into the `target_*` columns; for
IMC-2023-style data (probes-as-VPs) the mapping is reversed.

Slicing (`--slice`):
  all        — every row, no fit/eval split (smoke-test mode; leaks for stateful LTDs).
  head<k>    — keep the k targets that sort first by target_id (deterministic
               smoke slice; same no-stratification semantics as `all`).
  fold_N     — K-fold partition driven by `DistGeoStratification`. Eval = the
               targets in fold N; fit = the targets in the other K-1 folds.
               Deterministic in (k, seed, asn_bucket_top_n) source_kwargs.
               When `target_asn` is absent / missing, those targets land in
               the `asn_none` bucket and still round-robin into the K folds.
  wsplit<P>  — location-weighted holdout (P in 1..99, e.g. wsplit20). Per
               `target_city` (must be non-blank for every target), rank
               targets by summed weight descending (ties broken by
               target_id) and send the top ceil(P/100 * n) to the eval set;
               the rest go to fit. Every city contributes >= 1 eval target
               (a 1-target city goes entirely to eval and is absent from
               fit) — test-side location coverage is the invariant, and no
               top-weight target ever leaks into the fit corpus.

Source kwargs (defaults match the prior VultrCSVSource):
  csv_path           : Path | str   — required; canonical-schema CSV path.
  k                  : int = 5      — fold count for `fold_N`.
  seed               : int = 42     — DistGeo RNG seed.
  asn_bucket_top_n   : int = 20     — DistGeo bucket cap.
  min_obs            : int = None   — drop targets with fewer VP observations.
  eval_pair_weight_min : float = None — traffic-weighted eval mask, applied
                       AFTER the slice's fit/eval split (and after min_obs):
                       an eval target survives iff >= 1 of its obs has
                       weight >= this value, and surviving targets keep
                       only those clearing obs in iter_eval_targets. The fit
                       side is untouched — training always sees the full
                       mesh; only the evaluated view is traffic-restricted.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from scripts.benchmark.v2.sources.base import (
    DataSource,
    EvalTarget,
    TgConfig,
    VpConfig,
)
from scripts.framework.v2 import FitSample
from scripts.framework.v2.types import Coord, Latency, VpId
from scripts.processing.ripe_atlas.stratification import (
    AnchorInfo,
    DistGeoStratification,
    normalize_asn,
)

logger = logging.getLogger(__name__)


_REQUIRED = (
    "vp_id", "vp_lat", "vp_lon",
    "target_id", "target_lat", "target_lon",
    "rtt_ms",
)

_OPTIONAL = (
    "vp_asn", "vp_country", "vp_continent", "vp_region", "vp_city",
    "target_asn", "target_country", "target_continent", "target_region", "target_city",
)

# Optional free-text columns. Read with an identity converter so pandas does
# NOT apply its default NA-sentinel set to them — otherwise common literal
# codes like the continent "NA" (North America) or country "NA" (Namibia)
# would silently parse as NaN. Converters for columns absent from the CSV are
# ignored by pandas, so passing the full tuple is safe.
_OPTIONAL_STR = (
    "vp_country", "vp_continent", "vp_region", "vp_city",
    "target_country", "target_continent", "target_region", "target_city",
)

_FOLD_SLICE_RE = re.compile(r"^fold_(\d+)$")
_WSPLIT_SLICE_RE = re.compile(r"^wsplit(\d{1,2})$")


class GenericCSVSource(DataSource):
    """CSV-backed source with a fixed canonical schema + DistGeo K-fold stratification.

    See module docstring for the column contract and slice grammar.
    """

    name = "generic_csv"

    def __init__(
        self,
        slice: str,
        setup: str = DataSource.ANCHORS_TO_PROBES,
        csv_path: Optional[Path] = None,
        *,
        k: int = 5,
        seed: int = 42,
        asn_bucket_top_n: int = 20,
        min_obs: Optional[int] = None,
        eval_pair_weight_min: Optional[float] = None,
    ) -> None:
        if setup not in DataSource.ALLOWED_SETUPS:
            raise ValueError(
                f"unknown setup {setup!r}; expected one of {DataSource.ALLOWED_SETUPS}"
            )
        if csv_path is None:
            raise ValueError(
                f"{self.name!r} requires `csv_path` (path to a canonical-schema CSV)"
            )
        if eval_pair_weight_min is not None and eval_pair_weight_min < 0:
            raise ValueError(
                f"eval_pair_weight_min must be >= 0, got {eval_pair_weight_min}"
            )
        fold_match = _FOLD_SLICE_RE.match(slice)
        wsplit_match = _WSPLIT_SLICE_RE.match(slice)
        wsplit_pct = None
        if fold_match is not None:
            fold_index = int(fold_match.group(1))
            if fold_index >= k:
                raise ValueError(
                    f"slice fold index {fold_index} >= k={k} "
                    f"(available: fold_0..fold_{k - 1})"
                )
        elif wsplit_match is not None:
            fold_index = None
            wsplit_pct = int(wsplit_match.group(1))
            if not 1 <= wsplit_pct <= 99:
                raise ValueError(
                    f"wsplit percentage must be in 1..99, got {wsplit_pct} "
                    f"(slice {slice!r})"
                )
        elif slice == "all" or slice.startswith("head"):
            # `head<k>` is fully validated inside _apply_slice when the row
            # parser runs; constructor only confirms the prefix is legal.
            fold_index = None
        else:
            raise ValueError(
                f"unknown slice {slice!r}; expected 'all', 'head<k>', "
                f"'fold_N', or 'wsplit<P>'"
            )

        self._slice = slice
        self._setup = setup
        self._csv_path = Path(csv_path)
        self._fold_index = fold_index
        self._wsplit_pct = wsplit_pct
        self._k = k
        self._seed = seed
        self._asn_bucket_top_n = asn_bucket_top_n
        self._min_obs = min_obs
        self._eval_pair_weight_min = eval_pair_weight_min

        # Lazily populated by `_ensure_loaded`.
        self._df: Optional[pd.DataFrame] = None
        self._eval_targets: Optional[set[str]] = None
        self._fit_targets: Optional[set[str]] = None

    # ---- DataSource API ------------------------------------------------------

    def slice_id(self) -> str:
        return self._slice

    def setup_id(self) -> str:
        return self._setup

    def iter_vp_configs(self) -> Iterator[VpConfig]:
        df = self._ensure_loaded()
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
        # Static catalog of every target the source knows about (eval ∪ fit).
        # The eval/fit split is enforced inside iter_eval_targets /
        # iter_fit_samples. Matches the convention at
        # ripe_atlas_asn_corpora.py:137-158.
        df = self._ensure_loaded()
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
        df = self._ensure_loaded()
        for row in df.itertuples(index=False):
            tg_id = str(row.target_id)
            if self._fit_targets is not None and tg_id not in self._fit_targets:
                continue
            yield FitSample(
                vp_id=VpId(str(row.vp_id)),
                vp_coord=Coord(lat=float(row.vp_lat), lon=float(row.vp_lon)),
                probe_coord=Coord(lat=float(row.target_lat), lon=float(row.target_lon)),
                latency=Latency(float(row.rtt_ms)),
            )

    def iter_eval_targets(self) -> Iterator[EvalTarget]:
        df = self._ensure_loaded()
        for tg_id, group in df.groupby("target_id", sort=True):
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

    def _ensure_loaded(self) -> pd.DataFrame:
        if self._df is None:
            self._load_csv()
            if self._wsplit_pct is not None:
                # min_obs must run BEFORE the weight split: dropping a
                # top-weight target after the split would silently shrink
                # test-side location coverage — the split's invariant.
                if self._min_obs is not None:
                    self._apply_min_obs_filter()
                self._apply_weight_split()
            else:
                if self._fold_index is not None:
                    self._apply_stratification()
                if self._min_obs is not None:
                    self._apply_min_obs_filter()
            # Eval weight mask runs LAST: the split defines the eval set,
            # min_obs prunes sparse targets, and this only shrinks the eval
            # side further. The fit side is never touched.
            if self._eval_pair_weight_min is not None:
                self._apply_eval_weight_filter()
        assert self._df is not None
        return self._df

    def _load_csv(self) -> None:
        df = pd.read_csv(
            self._csv_path,
            converters={c: _raw_str for c in _OPTIONAL_STR},
        )
        missing = [c for c in _REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV {self._csv_path} missing required columns: {missing}. "
                f"Required: {list(_REQUIRED)}; optional: {list(_OPTIONAL)}."
            )
        df = df.dropna(subset=list(_REQUIRED))
        df = df[df["rtt_ms"] > 0].copy()
        df = self._normalize_weight(df)
        df = self._apply_slice(df, self._slice)
        self._df = df.reset_index(drop=True)

    def _normalize_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """Guarantee a numeric non-negative `weight` column.

        Two distinct defaults, deliberately:
          * column absent  → 1.0 everywhere ("this dataset has no weight
            notion"; summed target weight degenerates to obs count, and
            pair-weight-min thresholds <= 1.0 stay no-ops).
          * column present but cell NaN/non-numeric → 0.0 ("traffic unknown
            = weightless"): the pair can never outrank genuinely heavy
            flows in the wsplit ranking nor survive a weighted-eval
            threshold. Filling 1.0 here would masquerade as 1 unit of real
            traffic (e.g. 1 TB) among measured values.
        """
        if "weight" not in df.columns:
            df["weight"] = 1.0
            if self._wsplit_pct is not None:
                logger.info(
                    "CSV %s has no weight column; defaulting to 1.0 — "
                    "wsplit ranks targets by obs count", self._csv_path,
                )
            return df
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        n_missing = int(df["weight"].isna().sum())
        if n_missing:
            logger.info(
                "weight: %d missing/non-numeric cells defaulted to 0.0 "
                "(unknown traffic = weightless)",
                n_missing,
            )
            df["weight"] = df["weight"].fillna(0.0)
        if (df["weight"] < 0).any():
            raise ValueError(
                f"CSV {self._csv_path}: weight must be >= 0 "
                f"(found negative values)"
            )
        return df

    @staticmethod
    def _apply_slice(df: pd.DataFrame, slice_name: str) -> pd.DataFrame:
        """Row-level slice. `fold_N` keeps every row — the eval/fit partition
        lives in cached id sets, not in row drops."""
        if slice_name == "all":
            return df
        if _FOLD_SLICE_RE.match(slice_name) or _WSPLIT_SLICE_RE.match(slice_name):
            return df
        if slice_name.startswith("head"):
            try:
                k = int(slice_name.removeprefix("head"))
            except ValueError as e:
                raise ValueError(f"invalid head-k slice: {slice_name!r}") from e
            if k < 1:
                raise ValueError(f"head-k must be >=1, got {k}")
            keep = sorted(df["target_id"].astype(str).unique())[:k]
            if not keep:
                raise ValueError(
                    f"slice {slice_name!r} selected no targets (empty CSV?)"
                )
            return df[df["target_id"].astype(str).isin(keep)].copy()
        raise ValueError(
            f"unknown slice {slice_name!r}; expected 'all', 'head<k>', "
            f"'fold_N', or 'wsplit<P>'"
        )

    def _apply_stratification(self) -> None:
        """Stratify unique target_ids into K folds via DistGeo and cache
        the eval / fit target-id sets."""
        assert self._df is not None and self._fold_index is not None
        unique = self._df.drop_duplicates("target_id")
        targets: list[AnchorInfo] = []
        for _, row in unique.iterrows():
            asn = row.get("target_asn")
            country = row.get("target_country")
            targets.append(AnchorInfo(
                ip=str(row["target_id"]),
                lat=float(row["target_lat"]),
                lon=float(row["target_lon"]),
                country=str(country) if pd.notna(country) else None,
                asn=normalize_asn(asn),
            ))

        algo = DistGeoStratification(
            k=self._k,
            fold_index=0,  # full assignment is fold-index-independent
            seed=self._seed,
            asn_bucket_top_n=self._asn_bucket_top_n,
        )
        fold_by_id = algo.compute_fold_assignments(targets)

        eval_targets: set[str] = set()
        fit_targets: set[str] = set()
        for tg_id, fold in fold_by_id.items():
            (eval_targets if fold == self._fold_index else fit_targets).add(tg_id)
        self._eval_targets = eval_targets
        self._fit_targets = fit_targets
        logger.info(
            "stratified %d targets into K=%d folds: eval=fold_%d (%d targets), "
            "fit=union of %d other folds (%d targets)",
            len(targets), self._k, self._fold_index,
            len(eval_targets), self._k - 1, len(fit_targets),
        )

    def _apply_weight_split(self) -> None:
        """Location-weighted holdout (`wsplit<P>`). Per target_city, the top
        ceil(P/100 * n) targets by summed weight go to the eval set;
        the rest go to fit. Deterministic: ties break on target_id."""
        assert self._df is not None and self._wsplit_pct is not None
        df = self._df
        if "target_city" not in df.columns:
            raise ValueError(
                f"slice {self._slice!r} requires a target_city column "
                f"(missing from {self._csv_path})"
            )

        unique = df.drop_duplicates("target_id").copy()
        unique["target_id"] = unique["target_id"].astype(str)
        city = unique["target_city"].astype(str).str.strip()
        blank = ~unique["target_city"].notna() | (city == "")
        if blank.any():
            raise ValueError(
                f"slice {self._slice!r} requires a non-blank target_city for "
                f"every target; {int(blank.sum())} of {len(unique)} targets "
                f"have none (e.g. {unique.loc[blank, 'target_id'].head(3).tolist()})"
            )

        weight_by_target = (
            df.assign(target_id=df["target_id"].astype(str))
            .groupby("target_id")["weight"].sum()
        )

        eval_targets: set[str] = set()
        fit_targets: set[str] = set()
        n_cities_without_fit = 0
        for _, group in unique.groupby(city):
            ranked = sorted(
                group["target_id"],
                key=lambda t: (-weight_by_target[t], t),
            )
            n_eval = math.ceil(self._wsplit_pct / 100 * len(ranked))
            eval_targets.update(ranked[:n_eval])
            fit_targets.update(ranked[n_eval:])
            if n_eval == len(ranked):
                n_cities_without_fit += 1

        self._eval_targets = eval_targets
        self._fit_targets = fit_targets
        logger.info(
            "weight-split %d targets over %d cities at %d%%: eval=%d "
            "(top-sum-weight per city), fit=%d (%d cities absent from fit)",
            len(unique), city.nunique(), self._wsplit_pct,
            len(eval_targets), len(fit_targets), n_cities_without_fit,
        )

    def _apply_eval_weight_filter(self) -> None:
        """Traffic-weighted eval mask (`eval_pair_weight_min`). An eval target
        survives iff >= 1 of its obs has weight >= the threshold;
        iter_eval_targets then emits only the clearing obs. Fit targets and
        fit samples are untouched (full-mesh training)."""
        assert self._df is not None and self._eval_pair_weight_min is not None
        df = self._df
        thr = self._eval_pair_weight_min
        surviving = set(df.loc[df["weight"] >= thr, "target_id"].astype(str))
        base = (
            self._eval_targets
            if self._eval_targets is not None
            else set(df["target_id"].astype(str))
        )
        kept = base & surviving
        if not kept:
            raise ValueError(
                f"eval_pair_weight_min={thr} left zero eval targets in slice "
                f"{self._slice!r} — the threshold exceeds the data's weights"
            )
        logger.info(
            "eval_pair_weight_min=%s: eval targets %d → %d (fit side untouched)",
            thr, len(base), len(kept),
        )
        self._eval_targets = kept

    def _apply_min_obs_filter(self) -> None:
        assert self._df is not None and self._min_obs is not None
        counts = self._df.groupby("target_id")["target_id"].transform("count")
        before = self._df["target_id"].nunique()
        self._df = self._df[counts >= self._min_obs].reset_index(drop=True)
        after = self._df["target_id"].nunique()
        surviving = set(self._df["target_id"].astype(str))
        if self._eval_targets is not None:
            self._eval_targets &= surviving
        if self._fit_targets is not None:
            self._fit_targets &= surviving
        logger.info("min_obs=%d: %d → %d targets", self._min_obs, before, after)


def _raw_str(value: str) -> str:
    """Identity converter — keeps a cell's literal text so pandas' default
    NA-sentinel coercion never fires on it (see `_OPTIONAL_STR`)."""
    return value


def _opt_col(row: "pd.Series", col: str, cols: "pd.Index") -> Optional[str]:
    """Stringify an optional column value, or None when the column is absent
    from the CSV, the cell is NaN, or the cell is empty/whitespace (empty
    cells arrive as "" under the `_raw_str` converter, not NaN)."""
    if col not in cols:
        return None
    value = row.get(col)
    if not pd.notna(value):
        return None
    text = str(value).strip()
    return text or None
