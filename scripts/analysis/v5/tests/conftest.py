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
