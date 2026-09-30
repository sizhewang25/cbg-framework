"""Fixtures the contest figures share: one run on disk, two methods over it.

C1-C4 all read the same thing -- two scored `*_tgs.parquet` and a written
answer space -- and differ only in what they draw from it. The run is built
once here so a change to the shape of that input is one edit rather than four,
and so the `solved_mask` trap is present in every figure's fixture by default
rather than by each test remembering it.

Named `contest_*` rather than `run`/`space`: `test_figure_outcome_map` defines
its own fixtures under those names, and a module fixture shadowing a conftest
one silently is worse than two names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import answer_space as A
from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import contest as CT
from scripts.analysis.v5.modules.paths import ANSWER_SPACE_KIND, CLASSIFY_KIND, grid_slug

SEATTLE = (47.449, -122.309)
OMAHA = (41.2565, -95.9345)
CHICAGO = (41.8781, -87.6298)
MIAMI = (25.7932, -80.29)
PLACES = (SEATTLE, OMAHA, CHICAGO, MIAMI)

NSIDE = CT.SOURCE_NSIDE

#: Replicas per site. Four, not the meshes' twenty: the counting rule is the
#: same and the parquets are a quarter the size.
REPLICAS = 4

#: What each method gets right, **by index into `PLACES`**. `alpha` sweeps
#: Seattle and splits Omaha; `beta` loses Seattle outright and takes Omaha,
#: they tie on Chicago, and both answer every Miami replica correctly -- but
#: `beta`'s Miami rows are FALLBACK, so they count for nothing and Miami is a
#: win for `alpha`. That last site is the whole `solved_mask` trap in one
#: contest.
ALPHA_CORRECT = (REPLICAS, 2, 3, REPLICAS)
BETA_CORRECT = (0, REPLICAS, 3, REPLICAS)

#: Index into `PLACES`, **not a `site_id`**. `sites.site_ids` numbers sites in
#: sorted key order, which is not the order `PLACES` lists them: Miami is
#: `PLACES[3]` and `site_id` 0. A test that conflates the two can pass against
#: the wrong site -- one did. Go through `place_to_site`.
BETA_FALLBACK_PLACE = 3


@dataclass(frozen=True)
class FakeRun:
    """The slice of `RunPaths` the contest figures touch."""

    run_id: str
    root: Path
    source: str = "src"
    setup: str = "setup"

    def analysis_dir(self, kind, *, root=None):
        d = (root or self.root) / self.run_id / kind
        d.mkdir(parents=True, exist_ok=True)
        return d

    def answer_space_dir(self, nside, *, root=None):
        d = self.analysis_dir(ANSWER_SPACE_KIND, root=root) / grid_slug(nside)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def classify_dir(self, nside, *, root=None):
        d = self.analysis_dir(CLASSIFY_KIND, root=root) / grid_slug(nside)
        d.mkdir(parents=True, exist_ok=True)
        return d


def offset_point(point, north_km=0.0, east_km=0.0):
    lat, lon = point
    return (lat + north_km / 111.195, lon + east_km / (111.195 * np.cos(np.radians(lat))))


def scored_frame(space, correct_per_site, fallback_sites=()):
    """A scored frame where site `i` gets `correct_per_site[i]` right.

    A correct row answers 3 km from its own site; a wrong one answers at
    Miami, which is another cell for every site but Miami itself -- so Miami's
    wrong rows answer at Seattle instead.
    """
    rows = []
    for i, place in enumerate(PLACES):
        k = correct_per_site[i]
        elsewhere = SEATTLE if place == MIAMI else MIAMI
        for r in range(REPLICAS):
            pred = offset_point(place if r < k else elsewhere, north_km=3)
            rows.append(
                {
                    "tg_id": f"tg-{i}-{r}",
                    "tg_lat": place[0],
                    "tg_lon": place[1],
                    "pred_lat": pred[0],
                    "pred_lon": pred[1],
                    "status": "FALLBACK" if i in fallback_sites else "SUCCESS",
                }
            )
    return C.score_method(pd.DataFrame(rows), space)


@pytest.fixture(scope="session")
def contest_space():
    """Four sites, far enough apart that each keeps its own seed and cell."""
    t = pd.DataFrame(
        {
            "tg_id": [f"tg-{i}-{k}" for i in range(4) for k in range(REPLICAS)],
            "tg_lat": [p[0] for p in PLACES for _ in range(REPLICAS)],
            "tg_lon": [p[1] for p in PLACES for _ in range(REPLICAS)],
        }
    )
    return A.build_answer_space(t, nside=NSIDE, run_id="syn")


def write_run(run: FakeRun, space) -> FakeRun:
    """An answer space and the two methods, on disk under `run`."""
    space.write(run.answer_space_dir(NSIDE))
    d = run.classify_dir(NSIDE)
    scored_frame(space, ALPHA_CORRECT).to_parquet(
        d / C.TGS_PARQUET.format(method="alpha"), index=False
    )
    scored_frame(space, BETA_CORRECT, fallback_sites=(BETA_FALLBACK_PLACE,)).to_parquet(
        d / C.TGS_PARQUET.format(method="beta"), index=False
    )
    return run


@pytest.fixture(scope="session")
def place_to_site(contest_space):
    """`PLACES` index -> `site_id`, the only safe way to name a fixture site.

    `sites.site_ids` assigns ids in sorted key order, so the mapping is a
    permutation and not the identity. See `BETA_FALLBACK_PLACE`.
    """
    return {
        int(tg_id.split("-")[1]): int(site_id)
        for tg_id, site_id in zip(contest_space.tgs["tg_id"], contest_space.tgs["site_id"])
    }


@pytest.fixture(scope="module")
def contest_run(tmp_path_factory, contest_space):
    root = tmp_path_factory.mktemp("v5-contest")
    return write_run(FakeRun("syn1-000000-000000-mesh", root), contest_space)


@pytest.fixture(scope="module")
def contest_two_runs(contest_run, contest_space):
    """The same mesh twice under two run ids, so a layout has a grid to lay.

    Deliberately identical: a layout is a layout and must not depend on what
    the panels hold.
    """
    second = write_run(FakeRun("syn2-000000-000000-mesh", contest_run.root), contest_space)
    return [contest_run, second]


# -- PNI gap: one canonical CSV, one PNI list, three designed groups ---------
#
# Shared by `test_pni_gap` and `test_figure_pni_cluster_rtt`: the RTT figure
# reads the clusters `plot-pni-gap` wrote, so both need the same run on disk.

#: Two PNIs, and five VPs. Two of the VPs sit on the PNIs.
PNI_ROWS = (("pni-a", 40.0, -100.0), ("pni-b", 40.0, -80.0))
PNI_VPS = {
    "vp-a": (40.0, -100.0),
    "vp-b": (40.0, -80.0),
    "vp-west": (35.0, -120.0),
    "vp-east": (45.0, -70.0),
    "vp-south": (30.0, -90.0),
}

#: `(site coordinate, smallest-RTT VP)`, three sites per group.
#: `near`:  on or beside a PNI, latency picks the nearest VP   -> gap ~0.
#: `far`:   far from both PNIs, latency picks a PNI's VP       -> gap ~1,000-2,000 km.
#: `trombone`: beside a PNI, latency picks a VP across the map -> gap ~2,500+ km.
PNI_GROUPS = {
    "near": (((40.0, -100.0), "vp-a"), ((40.0, -80.0), "vp-b"), ((40.2, -100.2), "vp-a")),
    "far": (((30.0, -90.0), "vp-a"), ((35.0, -120.0), "vp-a"), ((30.5, -90.5), "vp-b")),
    "trombone": (((40.1, -100.1), "vp-east"), ((40.1, -80.1), "vp-west"), ((39.9, -80.0), "vp-west")),
}
PNI_REPLICAS = 3


def _haversine(a, b):
    from scripts.analysis.v5.modules.geodesy import haversine_km

    return float(haversine_km([a[0]], [a[1]], [b[0]], [b[1]])[0])


def write_pni_inputs(
    root: Path, *, tg_prefix: str = "tg", pni_rows=PNI_ROWS
) -> tuple[Path, Path, dict[str, str]]:
    """`(edge_csv, pni_csv, tg_group)`. RTT is propagation + 1 ms, except the
    designated VP, which gets 0.5 ms so it is the smallest by construction.

    `tg_prefix` keeps two runs' TG ids disjoint; `pni_rows` lets a second run
    carry a different operator's list."""
    root.mkdir(parents=True, exist_ok=True)
    rows, tg_group = [], {}
    for group, members in PNI_GROUPS.items():
        for s, (site, sp_vp) in enumerate(members):
            for r in range(PNI_REPLICAS):
                tg = f"{tg_prefix}-{group}-{s}-{r}"
                tg_group[tg] = group
                for vp, where in PNI_VPS.items():
                    rtt = 0.5 if vp == sp_vp else _haversine(site, where) / 100 + 1.0
                    rows.append({"vp_id": vp, "vp_lat": where[0], "vp_lon": where[1],
                                 "target_id": tg, "target_lat": site[0], "target_lon": site[1],
                                 "rtt_ms": rtt})
    edge_csv = root / "edges.csv"
    pd.DataFrame(rows).to_csv(edge_csv, index=False)
    pni_csv = root / "test-pni.csv"
    pd.DataFrame(pni_rows, columns=["pni_id", "pni_lat", "pni_lon"]).to_csv(pni_csv, index=False)
    return edge_csv, pni_csv, tg_group


@pytest.fixture
def pni_inputs(tmp_path):
    """A fresh `(run, edge_csv, pni_csv, tg_group, analysis_root)` per test."""
    edge_csv, pni_csv, tg_group = write_pni_inputs(tmp_path)
    return FakeRun("pni-run", tmp_path), edge_csv, pni_csv, tg_group, tmp_path / "analysis"


@pytest.fixture
def pni_two_runs(tmp_path):
    """Two runs at the same coordinates, disjoint TG ids, different PNI lists.

    `run-b`'s operator peers only at `pni-b`, so its `d_pni` differs from
    `run-a`'s for every TG near `pni-a` -- which is how a test can tell that
    pooling measured each run against its own list."""
    a = write_pni_inputs(tmp_path / "a", tg_prefix="a")
    b = write_pni_inputs(tmp_path / "b", tg_prefix="b", pni_rows=PNI_ROWS[1:])
    runs = [FakeRun("run-a", tmp_path), FakeRun("run-b", tmp_path)]
    return {
        "runs": runs,
        "edge_csvs": {"run-a": a[0], "run-b": b[0]},
        "pni_csvs": {"run-a": a[1], "run-b": b[1]},
        "root": tmp_path / "analysis",
    }
