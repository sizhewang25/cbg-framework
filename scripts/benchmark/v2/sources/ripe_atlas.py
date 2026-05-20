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
from typing import Iterator, Optional

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
        *,
        probes_and_anchors_file: Optional[Path] = None,
        ping_table: Optional[str] = None,
        threshold: int = _DEFAULT_THRESHOLD,
        filter_clause: str = _DEFAULT_FILTER,
    ) -> None:
        self._slice = slice
        self._probes_and_anchors_file = (
            Path(probes_and_anchors_file)
            if probes_and_anchors_file is not None
            else Path(default.REPRO_PROBES_AND_ANCHORS_FILE)
        )
        self._ping_table = ping_table if ping_table is not None else default.PROBES_TO_ANCHORS_PING_TABLE
        self._threshold = threshold
        self._filter_clause = filter_clause

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

    def iter_vp_configs(self) -> Iterator[VpConfig]:
        self._ensure_loaded()
        # Only emit VPs that actually appear in at least one RTT row — i.e.
        # ones the eval will actually use. Avoids polluting the parquet with
        # 10k+ probes that contributed no measurements.
        active_vp_ips: set[str] = set()
        for measurements in (self._rtts_by_anchor or {}).values():
            active_vp_ips.update(measurements.keys())
        for ip in sorted(active_vp_ips):
            coord = self._coords_by_ip.get(ip) if self._coords_by_ip else None
            if coord is None:
                continue  # RTT row referenced a probe missing from coords JSON
            yield VpConfig(
                vp_id=ip,
                lat=coord.lat,
                lon=coord.lon,
                asn=self._asn_by_ip.get(ip),
                country=self._country_by_ip.get(ip),
            )

    def iter_fit_samples(self) -> Iterator[FitSample]:
        self._ensure_loaded()
        assert self._rtts_by_anchor is not None and self._coords_by_ip is not None
        for anchor_ip, vp_rtts in self._rtts_by_anchor.items():
            anchor_coord = self._coords_by_ip.get(anchor_ip)
            if anchor_coord is None:
                continue  # anchor coord missing — skip
            for vp_ip, rtt in vp_rtts.items():
                vp_coord = self._coords_by_ip.get(vp_ip)
                if vp_coord is None or rtt <= 0:
                    continue
                yield FitSample(
                    vp_id=VpId(vp_ip),
                    vp_coord=vp_coord,
                    probe_coord=anchor_coord,
                    latency=Latency(float(rtt)),
                )

    def iter_eval_targets(self) -> Iterator[EvalTarget]:
        self._ensure_loaded()
        assert self._rtts_by_anchor is not None and self._coords_by_ip is not None
        for anchor_ip in sorted(self._rtts_by_anchor.keys()):
            anchor_coord = self._coords_by_ip.get(anchor_ip)
            if anchor_coord is None:
                continue
            obs: list[tuple[VpId, Coord, Latency]] = []
            for vp_ip, rtt in self._rtts_by_anchor[anchor_ip].items():
                vp_coord = self._coords_by_ip.get(vp_ip)
                if vp_coord is None or rtt <= 0:
                    continue
                obs.append((VpId(vp_ip), vp_coord, Latency(float(rtt))))
            if not obs:
                continue
            yield EvalTarget(target_id=anchor_ip, true_coord=anchor_coord, obs=obs)

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
        # to be reachable (tests/CI can mock-load fixtures).
        from scripts.analysis.analysis import compute_rtts_per_dst_src

        raw = compute_rtts_per_dst_src(
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
