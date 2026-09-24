"""Tests for the site key and the class-collapse table it backs.

`TestTheKeyDecision` is the one worth keeping honest: it pins the choice that
the run id is *part* of the key, by asserting that the same coordinate in two
runs is two sites. That is a modelling decision, not an implementation detail —
operators geolocate ASN by ASN — and every published site count, ICC and
effective n moves if it is quietly changed.

`TestFallbackIsNeverANumerator` pins the trap that has already produced a wrong
number twice: a FALLBACK row carries a real `ring`, so counting rings without
`solved_mask` credits a method for predictions it declined to make.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.analysis.v4.modules import answer_space as A
from scripts.analysis.v4.modules import class_collapse as CC
from scripts.analysis.v4.modules import classify as C
from scripts.analysis.v4.modules import sites as S
from scripts.analysis.v4.modules.paths import MissingArtifactError, resolve_run

SEATTLE = (47.4490, -122.3090)
OMAHA = (41.2565, -95.9345)
CHICAGO = (41.8781, -87.6298)
MIAMI = (25.7617, -80.1918)
#: 0.3 deg east of Seattle -- its own cell at nside-128, Seattle's from
#: nside-64 down. Found by sweeping offsets, not derived from a distance.
NEAR_SEATTLE = (47.4490, -122.0090)

MESH_RUNS = (
    "as01-260728-260802-mesh",
    "as02-260728-260802-mesh",
    "as03-260728-260802-mesh",
)


def _targets(coords, *, replicas=1, first=0):
    """A per-target frame with `replicas` IP addresses at each coordinate."""
    rows = []
    i = first
    for lat, lon in coords:
        for _ in range(replicas):
            rows.append({"target_id": f"tg-{i:04d}", "target_lat": lat, "target_lon": lon})
            i += 1
    return pd.DataFrame(rows)


class TestTheKeyDecision:
    """The run id is part of the key. This is the decision, pinned."""

    def test_one_coordinate_in_two_runs_is_two_sites(self):
        a = _targets([SEATTLE, OMAHA])
        b = _targets([SEATTLE, MIAMI], first=100)
        assert S.site_key(a, run_id="as01").nunique() == 2
        assert S.site_key(b, run_id="as02").nunique() == 2

        both = pd.concat(
            [a.assign(run_id="as01"), b.assign(run_id="as02")], ignore_index=True
        )
        # Four sites, three places: Seattle is served by two runs, so it poses
        # two geolocation problems rather than one.
        assert S.site_key(both).nunique() == 4
        assert both[list(S.SITE_COLUMNS)].drop_duplicates().shape[0] == 3

    def test_diagnostics_publish_both_counts(self):
        """65-style and 43-style numbers must never appear without each other."""
        d = S.site_diagnostics(
            {"as01": _targets([SEATTLE, OMAHA]), "as02": _targets([SEATTLE, MIAMI], first=100)}
        )
        assert d["n_sites"] == 4
        assert d["n_distinct_coordinates"] == 3
        assert d["coordinates_by_run_count"] == {"1": 2, "2": 1}

    def test_a_missing_run_id_is_refused_not_guessed(self):
        """Defaulting would merge two runs' sites — the collapse the key prevents."""
        with pytest.raises(ValueError, match="no run id"):
            S.site_key(_targets([SEATTLE]))


class TestSiteIds:
    def test_ids_do_not_depend_on_frame_order(self):
        t = _targets([MIAMI, SEATTLE, OMAHA, CHICAGO])
        shuffled = t.sample(frac=1.0, random_state=7)
        a = S.site_ids(t, run_id="as01")
        b = S.site_ids(shuffled, run_id="as01")
        assert (a.sort_index() == b.sort_index()).all()

    def test_replicas_of_one_coordinate_share_an_id(self):
        t = _targets([SEATTLE, OMAHA], replicas=20)
        ids = S.site_ids(t, run_id="as01")
        assert ids.nunique() == 2
        assert ids.value_counts().tolist() == [20, 20]

    def test_a_missing_coordinate_is_one_bucket_and_stays_in_the_denominator(self):
        t = _targets([SEATTLE, OMAHA])
        t.loc[0, "target_lat"] = None
        ids = S.site_ids(t, run_id="as01")
        assert (ids == S.MISSING_SITE).sum() == 1
        assert len(ids) == len(t)  # not dropped

    def test_rounding_is_a_guard_not_a_knob(self):
        """Byte-identical replicas: every rounding from 2 to 6 dp must agree."""
        d = S.site_diagnostics({"as01": _targets([SEATTLE, OMAHA, CHICAGO], replicas=5)})
        assert d["site_count_is_rounding_stable"]
        assert set(d["n_sites_by_decimals"].values()) == {3}

    def test_jittered_replicas_announce_themselves(self):
        """A dataset that *is* rounding-sensitive must not resolve silently."""
        t = _targets([SEATTLE, (SEATTLE[0] + 0.0001, SEATTLE[1])])
        d = S.site_diagnostics({"as01": t})
        assert not d["site_count_is_rounding_stable"]


class TestFallbackIsNeverANumerator:
    """A FALLBACK row carries a real ring, because the fallback is a real point."""

    def test_solved_mask_is_the_re_exported_one(self):
        assert S.solved_mask is C.solved_mask

    def test_a_fallback_row_is_excluded_from_a_ring_numerator(self):
        df = pd.DataFrame(
            {"status": ["SUCCESS", "FALLBACK", "SUCCESS"], "ring": [0, 0, 1]}
        )
        solved = S.solved_mask(df)
        # Denominator is every evaluated target; numerator only solved rows.
        assert len(df) == 3
        assert int((solved & (df["ring"] == 0)).sum()) == 1
        # The trap: without the mask this reads 2.
        assert int((df["ring"] == 0).sum()) == 2

    def test_an_all_baseline_frame_is_wholly_solved(self):
        df = pd.DataFrame({"status": ["BASELINE"] * 3, "ring": [0, 1, 2]})
        assert S.solved_mask(df).all()


class TestNeverSaysVoronoi:
    """The metric is ring-bounded serving-region recovery.

    The old name invites exactly the objection the ring bound exists to defuse,
    so it must not reach a column name, a manifest value or a docstring.
    """

    def test_no_emitted_string_says_voronoi(self):
        blob = json.dumps(
            {
                "diagnostics": S.site_diagnostics({"as01": _targets([SEATTLE, OMAHA])}),
                "columns": list(CC.CSV_COLUMNS) + [S.SITE_COL, S.SITE_KEY_COL],
                "note": S.UNIT_NOTE,
            }
        )
        assert "voronoi" not in blob.lower()

    def test_the_modules_do_not_say_it_either(self):
        for mod in (S, CC):
            assert "voronoi" not in (mod.__doc__ or "").lower()


class TestClassCollapse:
    def test_two_sites_in_one_cell_count_as_one_class(self):
        """The whole point: coarsening destroys class distinctions.

        `NEAR_SEATTLE` sits far enough out to hold its own cell at nside-128
        and close enough to share one from nside-64 down, so the site count
        stays at 3 while the class count falls to 2. Chosen by measurement,
        not by distance: HEALPix boundaries decide this, and a pair 78 km
        apart can straddle one at every rung.
        """
        targets = _targets([SEATTLE, NEAR_SEATTLE, MIAMI])
        seen = []
        for nside in (128, 64, 32, 16):
            space = A.build_answer_space(targets, nside=nside)
            row = CC._rung_row({"as01": space.assignments}, nside)
            assert row["n_sites"] == 3
            seen.append(row["n_classes"])
        assert seen == [3, 2, 2, 2]
        assert row["n_sites_merged"] == 2
        assert row["max_sites_per_cell"] == 2

    def test_class_count_never_rises_as_cells_grow(self):
        """Exact nesting: a coarser cell cannot split what a finer one merged."""
        targets = _targets([SEATTLE, NEAR_SEATTLE, OMAHA, CHICAGO, MIAMI])
        counts = [
            CC._rung_row(
                {"as01": A.build_answer_space(targets, nside=n).assignments}, n
            )["n_classes"]
            for n in (128, 64, 32, 16)
        ]
        assert counts == sorted(counts, reverse=True)

    def test_classes_are_summed_per_run_not_unioned(self):
        """Two runs at one coordinate are two classes, not one.

        Counting distinct cell ids over the concatenated frame would give 1
        here and would change every row of the published table.
        """
        frames = {}
        for i, run in enumerate(("as01", "as02")):
            t = _targets([SEATTLE], first=100 * i)
            frames[run] = A.build_answer_space(t, nside=128).assignments
        row = CC._rung_row(frames, 128)
        assert row["n_classes"] == 2
        assert pd.concat(frames.values())["cell_id"].nunique() == 1


class TestRealRuns:
    """The published numbers, on the data that produced them."""

    @pytest.fixture(scope="class")
    @classmethod
    def runs(cls):
        try:
            return [resolve_run(r) for r in MESH_RUNS]
        except MissingArtifactError as exc:
            pytest.skip(f"mesh runs not available: {exc}")

    @pytest.fixture(scope="class")
    @classmethod
    def frames(cls, runs):
        try:
            return {
                r.run_id: CC.load_assignments(r, 128) for r in runs
            }
        except MissingArtifactError as exc:
            pytest.skip(f"answer space not built: {exc}")

    def test_sixty_five_sites_across_three_meshes(self, frames):
        d = S.site_diagnostics(frames)
        assert d["n_sites"] == 65
        assert sorted(d["n_sites_by_run"].values()) == [20, 22, 23]
        assert d["n_rows"] == 1269

    def test_but_only_forty_three_distinct_places(self, frames):
        """65 counts site-appearances; the companion's 'unique coordinates' is wrong."""
        d = S.site_diagnostics(frames)
        assert d["n_distinct_coordinates"] == 43
        assert d["coordinates_by_run_count"] == {"1": 28, "2": 8, "3": 7}

    def test_replicas_are_byte_identical(self, frames):
        assert S.site_diagnostics(frames)["site_count_is_rounding_stable"]

    def test_the_class_collapse_table(self, runs):
        table = CC.collapse_table(runs)
        got = {
            int(r.nside): (int(r.n_sites), int(r.n_classes), int(r.n_sites_merged),
                           int(r.n_cells_holding_merges), int(r.max_sites_per_cell))
            for r in table.itertuples()
        }
        assert got == {
            128: (65, 63, 4, 2, 2),
            64: (65, 63, 4, 2, 2),
            32: (65, 60, 10, 5, 2),
            16: (65, 52, 25, 12, 3),
        }

    def test_the_unioned_count_is_published_not_quoted(self, runs):
        """The number `class_counting` warns against must be measured beside it."""
        loaded = CC._load_all(runs, (128, 64, 32, 16), None)
        assert CC._unioned_classes(loaded) == {"128": 40, "64": 38, "32": 33, "16": 26}

    def test_as01_never_separates_all_twenty_of_its_sites(self, runs):
        """The quantizer floor: the ladder is a dial on the cost, not a switch."""
        floor = CC._quantizer_floor(CC._load_all(runs, (128,), None))
        as01 = floor["by_run"]["as01-260728-260802-mesh"]
        assert (as01["n_sites"], as01["n_classes"]) == (20, 18)
        assert as01["separability_ceiling"] == 0.9
