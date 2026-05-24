"""RipeAtlasSource — primary eval path. Reads IMC 2023 RIPE Atlas data from ClickHouse.

Targets are the 723 anchors in `reproducibility_anchors.json` (hard ground truth);
VPs are the ~12K probes in `reproducibility_probes.json`. RTTs come from the
`ping_10k_to_anchors` ClickHouse table via the existing
`scripts.analysis.analysis.compute_rtts_per_dst_src` query helper.

This source reaches out to ClickHouse on first iteration. Cache the materialized
parquets and avoid re-querying. For unit testing, prefer VultrCSVSource (file-only).

The slice mechanism here is intentionally coarse:
  - "all_anchors"       : every anchor with at least one valid (probe, RTT)
  - "n<K>"              : keep the K anchors with the most VP measurements
                          (deterministic by (count desc, anchor_ip asc) tiebreaker)

Optional anchor-city enrichment: if `anchor_city.json` (produced by
`scripts/processing/append_city_to_anchors.py`) is present, each anchor's city
gets attached to its `VpConfig.city` (visible in vp_configs.parquet). Probes
remain city-less. Missing or unreadable file is silently skipped.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import default

from scripts.benchmark.v2.sources.base import DataSource, EvalTarget, TgConfig, VpConfig
from scripts.benchmark.v2.sources.holdout import (
    AnchorInfo,
    HoldoutPolicy,
    compute_fold_assignments,
)
from scripts.framework.v2 import FitSample
from scripts.framework.v2.types import Coord, Latency, VpId

logger = logging.getLogger(__name__)


# Default ClickHouse query parameters. `threshold=70` matches v1/million_scale.py:76.
_DEFAULT_FILTER = ""
_DEFAULT_THRESHOLD = 70
# Sanitization re-runs the paper's SOI-violation removal. threshold=300 matches
# datasets/create_datasets.ipynb (the cell that produced reproducibility_filtered_probes.json).
_DEFAULT_SANITIZE_THRESHOLD = 300

# Nominatim address-component fallback chain. Mirrors the notebook cell that
# produced datasets/static_datasets/population_city_file.json: prefer `city`,
# fall back to village/town/country in that order.
_CITY_KEYS = ("city", "village", "town", "country")


def _extract_city(response: Any) -> Optional[str]:
    """Pull a city-like label out of one Nominatim reverse-geocode response."""
    if not isinstance(response, dict):
        return None
    features = response.get("features") or []
    if not features:
        return None
    props = (features[0] or {}).get("properties") or {}
    address = props.get("address") or {}
    for key in _CITY_KEYS:
        value = address.get(key)
        if value:
            return str(value)
    return None


class RipeAtlasSource(DataSource):
    """RIPE Atlas probes → anchors (IMC 2023 reproducibility dataset)."""

    name = "ripe_atlas"

    def __init__(
        self,
        slice: str = "all_anchors",
        setup: str = DataSource.PROBES_TO_ANCHORS,
        *,
        probes_and_anchors_file: Optional[Path] = None,
        anchor_city_file: Optional[Path] = None,
        ping_table: Optional[str] = None,
        threshold: int = _DEFAULT_THRESHOLD,
        filter_clause: str = _DEFAULT_FILTER,
        rtt_query: Optional[Callable[..., dict[str, dict[str, list[float]]]]] = None,
        sanitize: bool = True,
        anchor_mesh_table: Optional[str] = None,
        sanitize_threshold: int = _DEFAULT_SANITIZE_THRESHOLD,
        holdout: Optional[HoldoutPolicy] = None,
    ) -> None:
        if setup not in DataSource.ALLOWED_SETUPS:
            raise ValueError(
                f"unknown setup {setup!r}; expected one of {DataSource.ALLOWED_SETUPS}"
            )
        self._slice = slice
        self._setup = setup
        self._probes_and_anchors_file = (
            Path(probes_and_anchors_file)
            if probes_and_anchors_file is not None
            else Path(default.REPRO_PROBES_AND_ANCHORS_FILE)
        )
        self._anchor_city_file = (
            Path(anchor_city_file)
            if anchor_city_file is not None
            else Path(default.REPRO_ANCHOR_CITY_FILE)
        )
        self._ping_table = ping_table if ping_table is not None else default.PROBES_TO_ANCHORS_PING_TABLE
        self._threshold = threshold
        self._filter_clause = filter_clause
        # Tests inject a fake here; production keeps it None and we
        # lazy-import compute_rtts_per_dst_src so importing this module
        # doesn't pull in ClickHouse / the v1 analysis module.
        self._rtt_query = rtt_query

        self._sanitize = sanitize
        self._anchor_mesh_table = (
            anchor_mesh_table if anchor_mesh_table is not None
            else default.ANCHORS_MESHED_PING_TABLE
        )
        self._sanitize_threshold = sanitize_threshold
        if holdout is not None and setup == DataSource.ANCHORS_TO_PROBES:
            logger.warning(
                "HoldoutPolicy ignored for setup=anchors_to_probes: the eval targets are "
                "probes (noisy GT, secondary setup) and the anchor-axis holdout does not "
                "apply. Iterators and slice_id() will behave as if holdout=None."
            )
            holdout = None
        self._holdout = holdout

        # Lazily populated caches; first iter_*() call triggers loading.
        self._coords_by_ip: Optional[dict[str, Coord]] = None
        self._anchor_ips: Optional[set[str]] = None
        self._asn_by_ip: dict[str, Optional[int]] = {}
        self._country_by_ip: dict[str, Optional[str]] = {}
        self._city_by_ip: dict[str, Optional[str]] = {}
        # rtts_by_anchor[anchor_ip] = {probe_ip: min_rtt_ms}
        self._rtts_by_anchor: Optional[dict[str, dict[str, float]]] = None
        # IPs removed by SOI sanitization (populated when sanitize=True).
        self._removed_ips: set[str] = set()
        # Populated by _apply_holdout() when holdout is set. None until then.
        self._fold_by_anchor: Optional[dict[str, int]] = None
        self._train_anchors: Optional[set[str]] = None
        self._test_anchors: Optional[set[str]] = None

    # ---- DataSource API ------------------------------------------------------

    def slice_id(self) -> str:
        if self._holdout is None:
            return self._slice
        return f"{self._slice}__{self._holdout.slice_suffix()}"

    def setup_id(self) -> str:
        return self._setup

    def iter_vp_configs(self) -> Iterator[VpConfig]:
        self._ensure_loaded()
        assert self._coords_by_ip is not None and self._rtts_by_anchor is not None
        if self._setup in (DataSource.PROBES_TO_ANCHORS, DataSource.ANCHORS_TO_ANCHORS):
            # VPs are src-side IPs — probes (P2A) or anchors (A2A).
            active_ips: set[str] = set()
            for measurements in self._rtts_by_anchor.values():
                active_ips.update(measurements.keys())
        else:  # ANCHORS_TO_PROBES — VPs are the anchors themselves
            active_ips = set(self._rtts_by_anchor.keys())
        for ip in sorted(active_ips):
            coord = self._coords_by_ip.get(ip)
            if coord is None:
                continue
            yield VpConfig(
                vp_id=ip, lat=coord.lat, lon=coord.lon,
                asn=self._asn_by_ip.get(ip),
                country=self._country_by_ip.get(ip),
                city=self._city_by_ip.get(ip),
            )

    def iter_tg_configs(self) -> Iterator[TgConfig]:
        self._ensure_loaded()
        assert self._coords_by_ip is not None and self._rtts_by_anchor is not None
        if self._setup in (DataSource.PROBES_TO_ANCHORS, DataSource.ANCHORS_TO_ANCHORS):
            # Targets are dst-side IPs — anchors in both P2A and A2A setups.
            target_ips = set(self._rtts_by_anchor.keys())
        else:  # ANCHORS_TO_PROBES — targets are probes that appear in any anchor's RTT row.
            target_ips = set()
            for measurements in self._rtts_by_anchor.values():
                target_ips.update(measurements.keys())
        for ip in sorted(target_ips):
            coord = self._coords_by_ip.get(ip)
            if coord is None:
                continue
            yield TgConfig(
                tg_id=ip, lat=coord.lat, lon=coord.lon,
                asn=self._asn_by_ip.get(ip),
                country=self._country_by_ip.get(ip),
                city=self._city_by_ip.get(ip),
            )

    def iter_fit_samples(self) -> Iterator[FitSample]:
        self._ensure_loaded()
        assert self._rtts_by_anchor is not None and self._coords_by_ip is not None
        # The training pair is (vp_known_coord, target_known_coord, rtt).
        # The setup decides which side gets called the VP.
        for anchor_ip, vp_rtts in self._rtts_by_anchor.items():
            if self._train_anchors is not None and anchor_ip not in self._train_anchors:
                continue
            anchor_coord = self._coords_by_ip.get(anchor_ip)
            if anchor_coord is None:
                continue
            for probe_ip, rtt in vp_rtts.items():
                probe_coord = self._coords_by_ip.get(probe_ip)
                if probe_coord is None or rtt <= 0:
                    continue
                if self._setup in (DataSource.PROBES_TO_ANCHORS, DataSource.ANCHORS_TO_ANCHORS):
                    # vp = src (probe in P2A, anchor in A2A); target = dst (anchor in both).
                    yield FitSample(
                        vp_id=VpId(probe_ip),
                        vp_coord=probe_coord,
                        probe_coord=anchor_coord,
                        latency=Latency(float(rtt)),
                    )
                else:  # ANCHORS_TO_PROBES
                    yield FitSample(
                        vp_id=VpId(anchor_ip),
                        vp_coord=anchor_coord,
                        probe_coord=probe_coord,
                        latency=Latency(float(rtt)),
                    )

    def iter_eval_targets(self) -> Iterator[EvalTarget]:
        self._ensure_loaded()
        assert self._rtts_by_anchor is not None and self._coords_by_ip is not None
        if self._setup in (DataSource.PROBES_TO_ANCHORS, DataSource.ANCHORS_TO_ANCHORS):
            for anchor_ip in sorted(self._rtts_by_anchor.keys()):
                if self._test_anchors is not None and anchor_ip not in self._test_anchors:
                    continue
                anchor_coord = self._coords_by_ip.get(anchor_ip)
                if anchor_coord is None:
                    continue
                obs: list[tuple[VpId, Coord, Latency]] = []
                for probe_ip, rtt in self._rtts_by_anchor[anchor_ip].items():
                    probe_coord = self._coords_by_ip.get(probe_ip)
                    if probe_coord is None or rtt <= 0:
                        continue
                    obs.append((VpId(probe_ip), probe_coord, Latency(float(rtt))))
                if obs:
                    yield EvalTarget(target_id=anchor_ip, true_coord=anchor_coord, obs=obs)
        else:  # ANCHORS_TO_PROBES — transpose: one EvalTarget per probe
            # First build {probe_ip: [(anchor_ip, anchor_coord, rtt), ...]}
            probe_obs: dict[str, list[tuple[str, Coord, float]]] = {}
            for anchor_ip, probe_rtts in self._rtts_by_anchor.items():
                anchor_coord = self._coords_by_ip.get(anchor_ip)
                if anchor_coord is None:
                    continue
                for probe_ip, rtt in probe_rtts.items():
                    if rtt <= 0:
                        continue
                    probe_obs.setdefault(probe_ip, []).append((anchor_ip, anchor_coord, rtt))
            for probe_ip in sorted(probe_obs):
                probe_coord = self._coords_by_ip.get(probe_ip)
                if probe_coord is None:
                    continue
                obs = [
                    (VpId(a), c, Latency(float(r)))
                    for a, c, r in probe_obs[probe_ip]
                ]
                if obs:
                    yield EvalTarget(target_id=probe_ip, true_coord=probe_coord, obs=obs)

    # ---- internals -----------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._coords_by_ip is None:
            self._load_coords()
        if self._rtts_by_anchor is None:
            self._load_rtts()
            self._apply_slice()
            self._apply_holdout()

    def _load_coords(self) -> None:
        # Local load instead of analysis.compute_geo_info() because the latter
        # also reads a precomputed pairwise-distance file we don't need here.
        with open(self._probes_and_anchors_file) as fh:
            probes = json.load(fh)
        city_by_anchor_id = self._load_anchor_cities()
        coords: dict[str, Coord] = {}
        anchor_ips: set[str] = set()
        for p in probes:
            ip = p.get("address_v4")
            geom = (p.get("geometry") or {}).get("coordinates")
            if not ip or not geom or len(geom) < 2:
                continue
            lon, lat = geom[0], geom[1]
            coords[ip] = Coord(lat=float(lat), lon=float(lon))
            self._asn_by_ip[ip] = p.get("asn_v4")
            self._country_by_ip[ip] = p.get("country_code")
            anchor_id = p.get("id")
            if anchor_id is not None:
                city = city_by_anchor_id.get(str(anchor_id))
                if city is not None:
                    self._city_by_ip[ip] = city
            if p.get("is_anchor"):
                anchor_ips.add(ip)
        self._coords_by_ip = coords
        self._anchor_ips = anchor_ips

    def _load_anchor_cities(self) -> dict[str, str]:
        """Load and parse `anchor_city.json` into {anchor_id_str: city_name}.

        Returns an empty dict if the file is missing or unreadable — city
        enrichment is best-effort, not a hard dependency.
        """
        try:
            with open(self._anchor_city_file) as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "could not read anchor city file %s: %s — VPs will have city=None",
                self._anchor_city_file, exc,
            )
            return {}
        out: dict[str, str] = {}
        for anchor_id, response in raw.items():
            city = _extract_city(response)
            if city:
                out[str(anchor_id)] = city
        return out

    def _load_rtts(self) -> None:
        # Imported lazily so importing this module doesn't require ClickHouse
        # to be reachable. Tests pass `rtt_query=` instead.
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

        if self._sanitize:
            self._removed_ips = self._compute_soi_removed_ips(query, raw)
            if self._removed_ips:
                raw = {
                    dst: {s: r for s, r in srcs.items() if s not in self._removed_ips}
                    for dst, srcs in raw.items()
                    if dst not in self._removed_ips
                }

        # raw shape: {dst_ip: {src_ip: [min_rtt]}}. Collapse to one float per pair.
        out: dict[str, dict[str, float]] = {}
        for anchor_ip, vp_rtts in raw.items():
            collapsed: dict[str, float] = {}
            for vp_ip, rtt_list in vp_rtts.items():
                if not rtt_list:
                    continue
                rtt = min(float(r) for r in rtt_list if r is not None)
                if rtt > 0:
                    collapsed[vp_ip] = rtt
            if collapsed:
                out[anchor_ip] = collapsed
        self._rtts_by_anchor = out

    def _compute_soi_removed_ips(
        self,
        query: Callable[..., dict[str, dict[str, list[float]]]],
        rtt_probes: dict[str, dict[str, list[float]]],
    ) -> set[str]:
        """Re-run the IMC 2023 sanitization on the live data.

        Phase 1: query the meshed anchor-anchor table, iteratively remove anchors
        with the most SOI violations until none remain.
        Phase 2: same procedure on the probe→anchor data, with phase-1 anchors
        already excluded.

        Returns the union set of IPs to drop. Stays an empty set if either query
        returns nothing (e.g. test injection that only knows the probe table).
        """
        from scripts.analysis.analysis import compute_remove_wrongly_geolocated_probes
        from scripts.utils.helpers import haversine

        assert self._coords_by_ip is not None
        coords = self._coords_by_ip

        try:
            rtt_anchors = query(
                self._anchor_mesh_table,
                "",
                self._sanitize_threshold,
                is_per_prefix=False,
            )
        except Exception:
            # If the anchor-mesh table is unavailable (e.g. test fakes), skip
            # phase 1 rather than crashing. Phase 2 still runs.
            rtt_anchors = {}

        # Build only the (dst, src) distances the function will actually look up.
        dist: dict[str, dict[str, float]] = {}
        for rtt_dict in (rtt_anchors, rtt_probes):
            for dst, srcs in rtt_dict.items():
                if dst not in coords:
                    continue
                row = dist.setdefault(dst, {})
                d_loc = (coords[dst].lat, coords[dst].lon)
                for src in srcs.keys():
                    if src in row or src not in coords or src == dst:
                        continue
                    s_loc = (coords[src].lat, coords[src].lon)
                    row[src] = float(haversine(d_loc, s_loc))

        removed_anchors = compute_remove_wrongly_geolocated_probes(
            rtt_anchors, coords, dist, set()
        )
        removed_probes = compute_remove_wrongly_geolocated_probes(
            rtt_probes, coords, dist, set(removed_anchors)
        )
        return set(removed_anchors) | set(removed_probes)

    def _apply_slice(self) -> None:
        assert self._rtts_by_anchor is not None
        if self._slice == "all_anchors":
            return
        if not self._slice.startswith("n"):
            raise ValueError(
                f"unknown RIPE Atlas slice {self._slice!r}; expected 'all_anchors' or 'n<K>'"
            )
        try:
            k = int(self._slice.removeprefix("n"))
        except ValueError as e:
            raise ValueError(f"invalid n<K> slice: {self._slice!r}") from e
        if k < 1:
            raise ValueError(f"slice K must be >=1, got {k}")
        ranked = sorted(
            self._rtts_by_anchor.items(),
            key=lambda kv: (-len(kv[1]), kv[0]),
        )[:k]
        self._rtts_by_anchor = {ip: rtts for ip, rtts in ranked}

    def _apply_holdout(self) -> None:
        """Partition anchors into K folds; populate train/test sets.

        Only meaningful for PROBES_TO_ANCHORS and ANCHORS_TO_ANCHORS — the
        held-out axis is anchors in both. ANCHORS_TO_PROBES with a holdout is
        already stripped to `holdout=None` at construction time (see __init__).
        """
        if self._holdout is None:
            return
        assert self._rtts_by_anchor is not None and self._coords_by_ip is not None

        anchor_infos: list[AnchorInfo] = []
        for anchor_ip in self._rtts_by_anchor:
            coord = self._coords_by_ip.get(anchor_ip)
            if coord is None:
                continue
            anchor_infos.append(AnchorInfo(
                ip=anchor_ip,
                lat=coord.lat,
                lon=coord.lon,
                country=self._country_by_ip.get(anchor_ip),
                asn=self._asn_by_ip.get(anchor_ip),
            ))

        self._fold_by_anchor = compute_fold_assignments(anchor_infos, self._holdout)
        self._test_anchors = {
            ip for ip, fold in self._fold_by_anchor.items()
            if fold == self._holdout.fold_index
        }
        self._train_anchors = {
            ip for ip, fold in self._fold_by_anchor.items()
            if fold != self._holdout.fold_index
        }
