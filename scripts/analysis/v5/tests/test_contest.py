"""The per-site contest: the solved mask, the absent threshold, the site key."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import contest as CT

SEATTLE = (47.449, -122.309)
OMAHA = (41.2565, -95.9345)
CHICAGO = (41.8781, -87.6298)
MIAMI = (25.7932, -80.29)
COORDS = {0: SEATTLE, 1: OMAHA, 2: CHICAGO, 3: MIAMI}


def ok(n):
    return [("correct", "SUCCESS")] * n


def bad(n):
    return [("wrong", "SUCCESS")] * n


def declined(n):
    """FALLBACK rows that *look* correct -- the shape `solved_mask` exists for.

    `classify` labels a FALLBACK row on the shortest-ping baseline's
    coordinate, and on a VP-adjacent target that coordinate is trivially in
    the right cell. So these carry `correct` and must still count for nothing.
    """
    return [("correct", "FALLBACK")] * n


def frame(sites: dict, *, seeds: dict | None = None) -> pd.DataFrame:
    """A scored frame from `{site_id: [(cell_label, status), ...]}`."""
    rows = []
    for sid, replicas in sites.items():
        lat, lon = COORDS[sid]
        for i, (label, status) in enumerate(replicas):
            rows.append(
                {
                    "tg_id": f"tg-{sid}-{i}",
                    "tg_lat": lat,
                    "tg_lon": lon,
                    "site_id": sid,
                    "tg_seed_id": (seeds or {}).get(sid, sid),
                    "cell_label": label,
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def contest(a: dict, b: dict, **kw) -> pd.DataFrame:
    return CT.site_contest(
        frame(a, **kw), frame(b, **kw), run_id="as01-x-mesh", method_a="A", method_b="B"
    )


# -- the category, and the threshold that is not there --------------------


def test_a_nine_of_twenty_site_is_a_loss_not_a_tie():
    """The Los Angeles regression.

    A majority rule (`correct_frac >= 0.5`) read Octant-Hull's 9 of 20 here as
    a clean loss and painted the site an *exclusive* Spotter win -- on a site
    where Octant-Hull was the more accurate of the two. The category compares
    counts, so the site is a Spotter win with a 20-to-9 margin on the label,
    and a reader can see the margin.
    """
    row = contest({0: ok(20)}, {0: ok(9) + bad(11)}).iloc[0]
    assert row["category"] == CT.A_WINS
    assert (row["k_a"], row["k_b"], row["n_tgs"]) == (20, 9, 20)


def test_a_single_replica_decides_a_site():
    """No threshold anywhere: 1 of 20 against 0 of 20 is a win.

    Stated as a test because it is the cost of having no threshold, and the
    plan accepts it explicitly -- as03 carries a 20-19. McNemar counts
    direction, never margin.
    """
    assert contest({0: ok(1) + bad(19)}, {0: bad(20)}).iloc[0]["category"] == CT.A_WINS


def test_both_at_zero_is_neither_not_tied():
    """A site neither method reaches says nothing about which is better.

    Folding it into `tied` would inflate the tie count with failures, and the
    tie count is what the reader takes as 'these two agree here'.
    """
    assert contest({0: bad(20)}, {0: bad(20)}).iloc[0]["category"] == CT.NEITHER


def test_equal_and_nonzero_is_tied():
    assert contest({0: ok(7) + bad(13)}, {0: ok(7) + bad(13)}).iloc[0]["category"] == CT.TIED


def test_categorize_is_vectorised_over_the_four_cases():
    got = CT.categorize([20, 0, 9, 5], [9, 0, 20, 5])
    assert list(got) == [CT.A_WINS, CT.NEITHER, CT.B_WINS, CT.TIED]


# -- the solved mask ------------------------------------------------------


def test_a_fallback_row_never_reaches_a_count():
    """Without the mask B reads 20 of 20 and the site is a tie. With it, B
    earned 9 and the site is a win for A -- the 163-against-270 bug."""
    row = contest({0: ok(20)}, {0: ok(9) + declined(11)}).iloc[0]
    assert row["k_b"] == 9
    assert row["n_b"] == 20, "a declined row stays in the denominator"
    assert row["category"] == CT.A_WINS


def test_the_shortest_ping_control_is_wholly_solved():
    """An all-BASELINE frame has no fallback path, so every row counts.

    `status.solved_mask`'s rule, pinned here because a contest is the place it
    would silently divide the control by zero answers.
    """
    baseline = frame({0: ok(5)})
    baseline["status"] = "BASELINE"
    assert CT.correct_mask(baseline).sum() == 5


# -- the site key ---------------------------------------------------------


def test_every_row_carries_its_run():
    """`site_id` is assigned per run and collides across meshes. Pooling on it
    alone reported 20 sites where there were 34."""
    a = contest({0: ok(5), 1: ok(5)}, {0: ok(5), 1: bad(5)})
    b = CT.site_contest(
        frame({0: ok(5), 1: ok(5)}), frame({0: bad(5), 1: ok(5)}),
        run_id="as02-x-mesh", method_a="A", method_b="B",
    )
    pooled = pd.concat([a, b], ignore_index=True)
    assert set(pooled["site_id"]) == {0, 1}
    assert len(pooled.groupby(list(CT.SITE_KEY))) == 4


def test_a_site_reports_its_own_coordinate_and_seed():
    row = contest({2: ok(3)}, {2: bad(3)}).iloc[0]
    assert (row["tg_lat"], row["tg_lon"]) == CHICAGO
    assert row["tg_seed_id"] == 2


# -- the guards -----------------------------------------------------------


def test_two_methods_on_different_rosters_are_refused():
    with pytest.raises(ValueError, match="do not score the same TGs"):
        CT.site_contest(
            frame({0: ok(5)}), frame({0: ok(4)}),
            run_id="as01-x-mesh", method_a="A", method_b="B",
        )


def test_a_repeated_tg_id_is_refused():
    dupe = pd.concat([frame({0: ok(2)})] * 2, ignore_index=True)
    with pytest.raises(ValueError, match="repeats"):
        CT.site_contest(
            dupe, dupe, run_id="as01-x-mesh", method_a="A", method_b="B"
        )


def test_two_answer_spaces_are_refused():
    with pytest.raises(ValueError, match="different answer spaces"):
        CT.site_contest(
            frame({0: ok(3)}), frame({0: ok(3)}, seeds={0: 9}),
            run_id="as01-x-mesh", method_a="A", method_b="B",
        )


def test_a_missing_column_names_the_command():
    thin = frame({0: ok(3)}).drop(columns=["tg_seed_id"])
    with pytest.raises(ValueError, match="classify"):
        CT.site_contest(
            thin, thin, run_id="as01-x-mesh", method_a="A", method_b="B"
        )


# -- McNemar --------------------------------------------------------------


def test_mcnemar_reproduces_the_resolved_mesh():
    got = CT.mcnemar(8, 1)
    assert got["mcnemar_p"] == pytest.approx(0.0390625)
    assert got["mcnemar_n_discordant"] == 9


def test_mcnemar_reports_the_floor_it_could_not_beat():
    """Two discordant sites cannot return below 0.5 however they split.

    A clean sweep -- both of them to the same method -- is the most extreme
    result the sample allows, and it still reads 0.5. Without the floor beside
    it that looks like a null result; it is an absent one.
    """
    got = CT.mcnemar(2, 0)
    assert got["mcnemar_p"] == pytest.approx(0.5)
    assert got["mcnemar_p_floor"] == pytest.approx(0.5)
    assert CT.mcnemar(1, 1)["mcnemar_p"] == pytest.approx(1.0)


def test_no_discordant_sites_is_not_a_test():
    got = CT.mcnemar(0, 0)
    assert got["mcnemar_p"] == 1.0 and got["mcnemar_p_floor"] == 1.0


def test_negative_counts_are_refused():
    with pytest.raises(ValueError, match="non-negative"):
        CT.mcnemar(-1, 2)


def test_the_counts_partition_the_sites():
    table = contest(
        {0: ok(5), 1: ok(5), 2: bad(5), 3: ok(5)},
        {0: ok(2) + bad(3), 1: ok(5), 2: bad(5), 3: bad(5)},
    )
    got = CT.contest_counts(table)
    assert sum(got[f"n_{c}"] for c in CT.CATEGORIES) == got["n_sites"] == 4
    assert got["mcnemar_b"] == got[f"n_{CT.A_WINS}"]
    assert got["mcnemar_c"] == got[f"n_{CT.B_WINS}"]


# -- peripherality --------------------------------------------------------


def test_centroid_km_is_measured_from_the_seed_cloud_not_the_sites():
    """The origin is the answer space's, so a clustered site set cannot move
    the thing it is being measured against."""
    lats = [c[0] for c in (SEATTLE, MIAMI)]
    lons = [c[1] for c in (SEATTLE, MIAMI)]
    seeds_lat = [c[0] for c in COORDS.values()]
    seeds_lon = [c[1] for c in COORDS.values()]
    far = CT.seed_cloud_centroid_km(lats, lons, seeds_lat, seeds_lon)
    near = CT.seed_cloud_centroid_km([OMAHA[0]], [OMAHA[1]], seeds_lat, seeds_lon)
    assert near[0] < far.min(), "Omaha sits inside the cloud; the coasts do not"
    # Moving the *sites* leaves the distances of the ones that stayed alone.
    again = CT.seed_cloud_centroid_km(lats + [OMAHA[0]], lons + [OMAHA[1]], seeds_lat, seeds_lon)
    assert np.allclose(again[:2], far)


def test_a_point_on_the_centre_reads_zero():
    """To a millimetre.

    `elementwise_km` is an arccos of a dot product, which loses precision as
    the separation goes to zero (see `geodesy.haversine_km`'s docstring); the
    residual here is 0.13 mm. It is not worth `haversine_km` for a column
    whose smallest real value on these meshes is 91 km, and `classify`
    measures at this scale the same way.
    """
    lat = [c[0] for c in COORDS.values()]
    lon = [c[1] for c in COORDS.values()]
    clat, clon = CT.seed_cloud_centre(lat, lon)
    assert CT.seed_cloud_centroid_km([clat], [clon], lat, lon)[0] == pytest.approx(0.0, abs=1e-3)


def test_the_centre_is_the_packages_one_notion_of_centre():
    """`seeds` places a seed over its sites with `spherical_centroid`; the
    centre of the seeds is found the same way, not in a projection."""
    from scripts.analysis.v5.modules.geodesy import spherical_centroid

    lat = [c[0] for c in COORDS.values()]
    lon = [c[1] for c in COORDS.values()]
    assert CT.seed_cloud_centre(lat, lon) == spherical_centroid(lat, lon)


def test_the_distance_is_great_circle():
    """Not planar: a projection is for drawing cells, not for measuring how
    far out in the partition a site sits."""
    from scripts.analysis.v5.modules.geodesy import elementwise_km

    lat = [c[0] for c in COORDS.values()]
    lon = [c[1] for c in COORDS.values()]
    clat, clon = CT.seed_cloud_centre(lat, lon)
    got = CT.seed_cloud_centroid_km([SEATTLE[0]], [SEATTLE[1]], lat, lon)
    assert got[0] == pytest.approx(
        elementwise_km([SEATTLE[0]], [SEATTLE[1]], [clat], [clon])[0]
    )
