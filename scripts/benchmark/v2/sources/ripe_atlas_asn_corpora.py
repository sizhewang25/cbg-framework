"""RipeAtlasASNCorporaSource — per-ASN VP corpus vs shared anchor eval set.

Sibling to `RipeAtlasSource`. The IMC 2023 source pipes all ~10K probes into
every CBG variant — the resulting "5K-VP/anchor median" corpus collapses the
tight-band methods (Octant, Spotter) because their per-VP annuli almost never
intersect at that density. Real deployments don't look like that: an ISP has
a few hundred probes inside its own AS; a CDN has datacenter VPs spread
across the world. This source models that shape.

For one configured (probe_data_dir, probe_asn, anchor_data_dir):
  * VPs = the single-ASN probe corpus produced by
    `scripts/processing/ripe_atlas/select_probes_and_anchors.py`
    (already continent-filtered + city-deduped).
  * Targets = the K-fold-stratified anchor set produced by
    `scripts/processing/ripe_atlas/stratify.py`. Fold N is the eval slice;
    the other K-1 folds are the fit corpus.
  * RTTs come from ClickHouse `ping_10k_to_anchors` via the existing
    `scripts.analysis.analysis.compute_rtts_per_dst_src` helper (same access
    pattern as RipeAtlasSource — lazy-imported, injectable for tests).

Only `setup=probes_to_anchors` is supported. Per-ASN corpora are VP fleets
by construction; transposing them would be meaningless.

Slice grammar: `fold_N` where N indexes a file `anchor_fold_<N>.json` in
`anchor_data_dir`. K is auto-discovered from the file count.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Iterator, Optional

import default

from scripts.benchmark.v2.sources.base import DataSource, EvalTarget, TgConfig, VpConfig
from scripts.framework.v2 import FitSample
from scripts.framework.v2.types import Coord, Latency, VpId

logger = logging.getLogger(__name__)

_FOLD_SLICE_RE = re.compile(r"^fold_(\d+)$")
_FOLD_FILE_RE = re.compile(r"^anchor_fold_(\d+)\.json$")

_DEFAULT_FILTER = ""
_DEFAULT_THRESHOLD = 70


class RipeAtlasASNCorporaSource(DataSource):
    """Single-ASN VP corpus vs K-fold-stratified anchor eval set."""

    name = "ripe_atlas_asn_corpora"

    def __init__(
        self,
        slice: str,
        setup: str = DataSource.PROBES_TO_ANCHORS,
        *,
        probe_data_dir: str | Path,
        probe_asn: int,
        anchor_data_dir: str | Path,
        ping_table: Optional[str] = None,
        threshold: int = _DEFAULT_THRESHOLD,
        filter_clause: str = _DEFAULT_FILTER,
        rtt_query: Optional[Callable[..., dict[str, dict[str, list[float]]]]] = None,
    ) -> None:
        if setup != DataSource.PROBES_TO_ANCHORS:
            raise ValueError(
                f"{self.name!r} only supports setup={DataSource.PROBES_TO_ANCHORS!r}, "
                f"got {setup!r}. Per-ASN corpora are VP fleets by construction."
            )
        match = _FOLD_SLICE_RE.match(slice)
        if not match:
            raise ValueError(
                f"slice must match 'fold_N' for {self.name!r} (got {slice!r}). "
                f"Each fold of the anchor stratification is a separate slice."
            )

        self._slice = slice
        self._setup = setup
        self._fold_index = int(match.group(1))

        self._probe_data_dir = Path(probe_data_dir)
        self._probe_asn = int(probe_asn)
        self._anchor_data_dir = Path(anchor_data_dir)

        self._ping_table = (
            ping_table if ping_table is not None
            else default.PROBES_TO_ANCHORS_PING_TABLE
        )
        self._threshold = threshold
        self._filter_clause = filter_clause
        # Tests inject `rtt_query=`; production keeps it None and we
        # lazy-import compute_rtts_per_dst_src on first iteration so importing
        # this module doesn't require ClickHouse.
        self._rtt_query = rtt_query

        # Lazily populated caches; first iter_*() call triggers loading.
        self._probe_coords: Optional[dict[str, Coord]] = None
        self._probe_asn_by_ip: dict[str, Optional[int]] = {}
        self._probe_country_by_ip: dict[str, Optional[str]] = {}
        self._anchor_coords: Optional[dict[str, Coord]] = None
        self._anchor_asn_by_ip: dict[str, Optional[int]] = {}
        self._anchor_country_by_ip: dict[str, Optional[str]] = {}
        self._eval_anchors: Optional[set[str]] = None
        self._fit_anchors: Optional[set[str]] = None
        # rtts_by_anchor[anchor_ip] = {probe_ip: min_rtt_ms} — already filtered
        # to (anchor_ip ∈ eval ∪ fit, probe_ip ∈ probe_coords).
        self._rtts_by_anchor: Optional[dict[str, dict[str, float]]] = None
        # IPs of probes that appear in at least one RTT row (post-filter). Used
        # to skip orphaned probes in iter_vp_configs.
        self._active_probes: Optional[set[str]] = None

    # ---- DataSource API ------------------------------------------------------

    def slice_id(self) -> str:
        return self._slice

    def setup_id(self) -> str:
        return self._setup

    def iter_vp_configs(self) -> Iterator[VpConfig]:
        self._ensure_loaded()
        assert self._probe_coords is not None and self._active_probes is not None
        for ip in sorted(self._active_probes):
            coord = self._probe_coords.get(ip)
            if coord is None:
                continue
            yield VpConfig(
                vp_id=ip, lat=coord.lat, lon=coord.lon,
                asn=self._probe_asn_by_ip.get(ip),
                country=self._probe_country_by_ip.get(ip),
            )

    def iter_tg_configs(self) -> Iterator[TgConfig]:
        # Only the EVAL fold belongs in tg_configs — fit anchors are training
        # material, not eval targets.
        self._ensure_loaded()
        assert self._anchor_coords is not None and self._eval_anchors is not None
        for ip in sorted(self._eval_anchors):
            coord = self._anchor_coords.get(ip)
            if coord is None:
                continue
            yield TgConfig(
                tg_id=ip, lat=coord.lat, lon=coord.lon,
                asn=self._anchor_asn_by_ip.get(ip),
                country=self._anchor_country_by_ip.get(ip),
            )

    def iter_fit_samples(self) -> Iterator[FitSample]:
        self._ensure_loaded()
        assert (
            self._rtts_by_anchor is not None
            and self._probe_coords is not None
            and self._anchor_coords is not None
            and self._fit_anchors is not None
        )
        for anchor_ip, vp_rtts in self._rtts_by_anchor.items():
            if anchor_ip not in self._fit_anchors:
                continue
            anchor_coord = self._anchor_coords.get(anchor_ip)
            if anchor_coord is None:
                continue
            for probe_ip, rtt in vp_rtts.items():
                probe_coord = self._probe_coords.get(probe_ip)
                if probe_coord is None or rtt <= 0:
                    continue
                # FitSample.probe_coord = "known target coord" (v1 naming quirk
                # preserved). See ripe_atlas.py:245-250 for precedent.
                yield FitSample(
                    vp_id=VpId(probe_ip),
                    vp_coord=probe_coord,
                    probe_coord=anchor_coord,
                    latency=Latency(float(rtt)),
                )

    def iter_eval_targets(self) -> Iterator[EvalTarget]:
        self._ensure_loaded()
        assert (
            self._rtts_by_anchor is not None
            and self._probe_coords is not None
            and self._anchor_coords is not None
            and self._eval_anchors is not None
        )
        for anchor_ip in sorted(self._eval_anchors):
            anchor_coord = self._anchor_coords.get(anchor_ip)
            if anchor_coord is None:
                continue
            vp_rtts = self._rtts_by_anchor.get(anchor_ip, {})
            obs: list[tuple[VpId, Coord, Latency]] = []
            for probe_ip, rtt in vp_rtts.items():
                probe_coord = self._probe_coords.get(probe_ip)
                if probe_coord is None or rtt <= 0:
                    continue
                obs.append((VpId(probe_ip), probe_coord, Latency(float(rtt))))
            if obs:
                yield EvalTarget(target_id=anchor_ip, true_coord=anchor_coord, obs=obs)

    # ---- internals -----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._probe_coords is None:
            self._load_probes()
        if self._anchor_coords is None:
            self._load_anchors()
        if self._rtts_by_anchor is None:
            self._load_rtts()

    def _load_probes(self) -> None:
        probes_file = self._probe_data_dir / f"probes_of_as_{self._probe_asn}.json"
        with probes_file.open() as fh:
            entries = json.load(fh)
        coords: dict[str, Coord] = {}
        for p in entries:
            ip = p.get("address_v4")
            geom = (p.get("geometry") or {}).get("coordinates")
            if not ip or not geom or len(geom) < 2:
                continue
            lon, lat = geom[0], geom[1]
            coords[ip] = Coord(lat=float(lat), lon=float(lon))
            self._probe_asn_by_ip[ip] = p.get("asn_v4")
            self._probe_country_by_ip[ip] = p.get("country_code")
        self._probe_coords = coords
        logger.info(
            "loaded %d probes from %s (AS%d)",
            len(coords), probes_file, self._probe_asn,
        )

    def _load_anchors(self) -> None:
        """Read every anchor_fold_<i>.json in `anchor_data_dir`, partition
        fold N (eval) vs the rest (fit), and build the per-anchor metadata
        dicts. K is the number of fold files found; the slice's fold_index
        must be less than K."""
        fold_files = sorted(
            self._anchor_data_dir.glob("anchor_fold_*.json"),
            key=lambda p: int(_FOLD_FILE_RE.match(p.name).group(1)),  # type: ignore[union-attr]
        )
        if not fold_files:
            raise FileNotFoundError(
                f"no anchor_fold_*.json files found in {self._anchor_data_dir}"
            )
        k = len(fold_files)
        if self._fold_index >= k:
            raise ValueError(
                f"slice fold index {self._fold_index} >= K={k} "
                f"(available: fold_0..fold_{k - 1} in {self._anchor_data_dir})"
            )

        coords: dict[str, Coord] = {}
        eval_anchors: set[str] = set()
        fit_anchors: set[str] = set()
        for fold_path in fold_files:
            m = _FOLD_FILE_RE.match(fold_path.name)
            assert m is not None
            i = int(m.group(1))
            with fold_path.open() as fh:
                entries = json.load(fh)
            target_set = eval_anchors if i == self._fold_index else fit_anchors
            for a in entries:
                ip = a.get("address_v4")
                geom = (a.get("geometry") or {}).get("coordinates")
                if not ip or not geom or len(geom) < 2:
                    continue
                lon, lat = geom[0], geom[1]
                coords[ip] = Coord(lat=float(lat), lon=float(lon))
                self._anchor_asn_by_ip[ip] = a.get("asn_v4")
                self._anchor_country_by_ip[ip] = a.get("country_code")
                target_set.add(ip)
        self._anchor_coords = coords
        self._eval_anchors = eval_anchors
        self._fit_anchors = fit_anchors
        logger.info(
            "loaded %d anchors over K=%d folds: eval=fold_%d (%d anchors), "
            "fit=union of %d other folds (%d anchors)",
            len(coords), k, self._fold_index,
            len(eval_anchors), k - 1, len(fit_anchors),
        )

    def _load_rtts(self) -> None:
        assert (
            self._probe_coords is not None
            and self._eval_anchors is not None
            and self._fit_anchors is not None
        )
        if self._rtt_query is not None:
            query = self._rtt_query
        else:
            from scripts.analysis.analysis import compute_rtts_per_dst_src
            query = compute_rtts_per_dst_src

        raw = query(
            self._ping_table,
            self._filter_clause,
            self._threshold,
            is_per_prefix=False,
        )

        probe_ips = self._probe_coords.keys()
        anchor_ips = self._eval_anchors | self._fit_anchors

        out: dict[str, dict[str, float]] = {}
        active_probes: set[str] = set()
        for anchor_ip, vp_rtts in raw.items():
            if anchor_ip not in anchor_ips:
                continue
            collapsed: dict[str, float] = {}
            for probe_ip, rtt_list in vp_rtts.items():
                if probe_ip not in probe_ips or not rtt_list:
                    continue
                rtt = min(float(r) for r in rtt_list if r is not None)
                if rtt > 0:
                    collapsed[probe_ip] = rtt
                    active_probes.add(probe_ip)
            if collapsed:
                out[anchor_ip] = collapsed
        self._rtts_by_anchor = out
        self._active_probes = active_probes
        logger.info(
            "loaded RTTs: %d anchors with measurements, %d active probes "
            "(out of %d corpus probes)",
            len(out), len(active_probes), len(probe_ips),
        )
