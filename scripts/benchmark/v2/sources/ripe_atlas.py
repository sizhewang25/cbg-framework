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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterator, Optional

import default

from scripts.benchmark.v2.sources.base import DataSource, EvalTarget, VpConfig
from scripts.framework.v2 import FitSample
from scripts.framework.v2.types import Coord, Latency, VpId


# Default ClickHouse query parameters. `threshold=70` matches v1/million_scale.py:76.
_DEFAULT_FILTER = ""
_DEFAULT_THRESHOLD = 70


class RipeAtlasSource(DataSource):
    """RIPE Atlas probes → anchors (IMC 2023 reproducibility dataset)."""

    name = "ripe_atlas"

    def __init__(
        self,
        slice: str = "all_anchors",
        setup: str = DataSource.PROBES_TO_ANCHORS,
        *,
        probes_and_anchors_file: Optional[Path] = None,
        ping_table: Optional[str] = None,
        threshold: int = _DEFAULT_THRESHOLD,
        filter_clause: str = _DEFAULT_FILTER,
        rtt_query: Optional[Callable[..., dict[str, dict[str, list[float]]]]] = None,
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
        self._ping_table = ping_table if ping_table is not None else default.PROBES_TO_ANCHORS_PING_TABLE
        self._threshold = threshold
        self._filter_clause = filter_clause
        # Tests inject a fake here; production keeps it None and we
        # lazy-import compute_rtts_per_dst_src so importing this module
        # doesn't pull in ClickHouse / the v1 analysis module.
        self._rtt_query = rtt_query

        # Lazily populated caches; first iter_*() call triggers loading.
        self._coords_by_ip: Optional[dict[str, Coord]] = None
        self._anchor_ips: Optional[set[str]] = None
        self._asn_by_ip: dict[str, Optional[int]] = {}
        self._country_by_ip: dict[str, Optional[str]] = {}
        # rtts_by_anchor[anchor_ip] = {probe_ip: min_rtt_ms}
        self._rtts_by_anchor: Optional[dict[str, dict[str, float]]] = None

    # ---- DataSource API ------------------------------------------------------

    def slice_id(self) -> str:
        return self._slice

    def setup_id(self) -> str:
        return self._setup

    def iter_vp_configs(self) -> Iterator[VpConfig]:
        self._ensure_loaded()
        assert self._coords_by_ip is not None and self._rtts_by_anchor is not None
        if self._setup == DataSource.PROBES_TO_ANCHORS:
            # VPs are probes that appear in at least one RTT row.
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
            )

    def iter_fit_samples(self) -> Iterator[FitSample]:
        self._ensure_loaded()
        assert self._rtts_by_anchor is not None and self._coords_by_ip is not None
        # The training pair is (vp_known_coord, target_known_coord, rtt).
        # The setup decides which side gets called the VP.
        for anchor_ip, vp_rtts in self._rtts_by_anchor.items():
            anchor_coord = self._coords_by_ip.get(anchor_ip)
            if anchor_coord is None:
                continue
            for probe_ip, rtt in vp_rtts.items():
                probe_coord = self._coords_by_ip.get(probe_ip)
                if probe_coord is None or rtt <= 0:
                    continue
                if self._setup == DataSource.PROBES_TO_ANCHORS:
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
        if self._setup == DataSource.PROBES_TO_ANCHORS:
            for anchor_ip in sorted(self._rtts_by_anchor.keys()):
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

    def _load_coords(self) -> None:
        # Local load instead of analysis.compute_geo_info() because the latter
        # also reads a precomputed pairwise-distance file we don't need here.
        with open(self._probes_and_anchors_file) as fh:
            probes = json.load(fh)
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
            if p.get("is_anchor"):
                anchor_ips.add(ip)
        self._coords_by_ip = coords
        self._anchor_ips = anchor_ips

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
