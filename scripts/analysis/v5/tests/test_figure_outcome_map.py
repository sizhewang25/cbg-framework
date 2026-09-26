"""The outcome map: the solved mask, the count key, the relaxation, the twin."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import answer_space as A
from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import figure_outcome_map as F
from scripts.analysis.v5.modules.paths import (
    ANSWER_SPACE_KIND,
    CLASSIFY_KIND,
    MissingArtifactError,
    grid_slug,
)

SEATTLE = (47.449, -122.309)
OMAHA = (41.2565, -95.9345)
CHICAGO = (41.8781, -87.6298)
MIAMI = (25.7932, -80.29)

NSIDE = F.SOURCE_NSIDE


def _offset(point, north_km=0.0, east_km=0.0):
    lat, lon = point
    return (lat + north_km / 111.195, lon + east_km / (111.195 * np.cos(np.radians(lat))))


@dataclass(frozen=True)
class _Run:
    """The slice of `RunPaths` the map touches."""

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


@pytest.fixture(scope="module")
def space():
    """Four sites, far enough apart that each keeps its own seed and cell."""
    t = pd.DataFrame(
        {
            "tg_id": [f"tg-{i}" for i in range(4)],
            "tg_lat": [c[0] for c in (SEATTLE, OMAHA, CHICAGO, MIAMI)],
            "tg_lon": [c[1] for c in (SEATTLE, OMAHA, CHICAGO, MIAMI)],
        }
    )
    return A.build_answer_space(t, nside=NSIDE, run_id="syn")


def _scored(space, preds, statuses=None, tgs=None):
    """Score `preds` against the answer space, one row per prediction.

    `tgs` picks which TG each row belongs to, by index into the space's TG
    table; it defaults to the first, as the outcome-bars fixture does.
    """
    idx = tgs if tgs is not None else [0] * len(preds)
    rows = space.tgs.iloc[idx].reset_index(drop=True)
    frame = pd.DataFrame(
        {
            "tg_id": rows["tg_id"],
            "tg_lat": rows["tg_lat"],
            "tg_lon": rows["tg_lon"],
            "pred_lat": [p[0] if p else np.nan for p in preds],
            "pred_lon": [p[1] if p else np.nan for p in preds],
            "status": statuses or ["SUCCESS"] * len(preds),
        }
    )
    return C.score_method(frame, space)


@pytest.fixture(scope="module")
def run(tmp_path_factory, space):
    """One run on disk: an answer space and two scored methods.

    `near` answers beside every TG. `giving_up` answers correctly on two rows
    and FALLBACKs on three more that *also* carry a correct-looking
    coordinate -- the exact shape `solved_mask` exists to catch.
    """
    root = tmp_path_factory.mktemp("v5-outcome-map")
    r = _Run("syn1-000000-000000-mesh", root)
    space.write(r.answer_space_dir(NSIDE))

    near = _scored(
        space,
        [_offset(c, north_km=3) for c in (SEATTLE, OMAHA, CHICAGO, MIAMI)],
        tgs=[0, 1, 2, 3],
    )
    giving_up = _scored(
        space,
        [_offset(SEATTLE, north_km=3)] * 2 + [_offset(OMAHA, north_km=3)] * 3,
        ["SUCCESS"] * 2 + ["FALLBACK"] * 3,
        tgs=[0, 0, 1, 1, 1],
    )
    d = r.classify_dir(NSIDE)
    for name, frame in (("near", near), ("giving_up", giving_up)):
        frame.to_parquet(d / C.TGS_PARQUET.format(method=name), index=False)
    return r


@pytest.fixture(scope="module")
def data(run):
    return F.load([run], nside=NSIDE, analysis_root=run.root)


# -- the solved mask ------------------------------------------------------


def test_a_fallback_row_is_excluded_and_reported(data):
    """A FALLBACK row carries the baseline's coordinate, so it must not count
    as the method's own answer -- the bug worth 107 rows on as01/VAN."""
    frame = data.frames[(data.run_ids[0], "giving_up")]
    assert (frame["status"] == "FALLBACK").sum() == 3
    # Every row of this frame is labelled `correct`: the FALLBACK ones only
    # drop out because of the mask, not because they look wrong.
    assert (frame["cell_label"] == "correct").all()

    rows = F.cohort_rows(frame, "correct")
    assert len(rows) == 2, "the three FALLBACK rows must not be drawn"
    assert set(rows["status"]) == {"SUCCESS"}

    counts = F.panel_counts(
        frame, rows, run_id=data.run_ids[0], method="giving_up", cohort="correct",
        extent=F.DEFAULT_EXTENT,
    )
    assert counts["n_cohort_tgs"] == 2
    assert counts["n_fallback_excluded"] == 3
    assert "3 FB excl." in F.panel_title(counts)


def test_the_fallback_count_spans_the_whole_frame_not_the_cohort(data):
    """`n_fallback_excluded` belongs to neither cohort, so both panels carry
    the same number."""
    frame = data.frames[(data.run_ids[0], "giving_up")]
    per_cohort = {
        c: F.panel_counts(
            frame, F.cohort_rows(frame, c), run_id="r", method="m", cohort=c,
            extent=F.DEFAULT_EXTENT,
        )["n_fallback_excluded"]
        for c in F.COHORTS
    }
    assert set(per_cohort.values()) == {3}


# -- the count key --------------------------------------------------------


def test_counts_key_on_the_tg_seed_never_the_prediction_seed(space):
    """On the `wrong` map a cell's number is "predictions that should have
    landed here", so it is keyed on `tg_seed_id`."""
    stray = _scored(space, [_offset(MIAMI, north_km=3)] * 4, tgs=[0, 0, 0, 0])
    rows = F.cohort_rows(stray, "wrong")
    assert len(rows) == 4

    tg_seed = int(rows["tg_seed_id"].iloc[0])
    pred_seed = int(rows["pred_seed_id"].iloc[0])
    assert tg_seed != pred_seed, "the fixture must actually send them elsewhere"

    counts = F.panel_counts(
        stray, rows, run_id="syn1-000000-000000-mesh", method="stray", cohort="wrong",
        extent=F.DEFAULT_EXTENT,
    )
    table = F.seed_rows(rows, space.seeds, counts)
    assert list(table["seed_id"]) == [tg_seed]
    assert int(table["n_tgs"].iloc[0]) == 4
    assert pred_seed not in set(table["seed_id"])


def test_the_drawn_labels_are_the_seed_rows(space):
    """`draw_counts` and the CSV twin must not diverge: same key, same counts."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    stray = _scored(
        space,
        [_offset(MIAMI, north_km=3)] * 3 + [_offset(SEATTLE, north_km=3)] * 2,
        tgs=[0, 0, 0, 1, 1],
    )
    rows = F.cohort_rows(stray, "wrong")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(F.DEFAULT_EXTENT, crs=ccrs.PlateCarree())
    drawn = F.draw_counts(ax, rows, space.seeds, extent=F.DEFAULT_EXTENT)
    labels = sorted(int(t.get_text()) for t in ax.texts)
    plt.close(fig)

    counts = F.panel_counts(
        stray, rows, run_id="r", method="m", cohort="wrong", extent=F.DEFAULT_EXTENT
    )
    table = F.seed_rows(rows, space.seeds, counts)
    assert drawn == len(table) == 2
    assert labels == sorted(table["n_tgs"].astype(int))


# -- the relaxation -------------------------------------------------------


def test_place_labels_separates_two_coincident_anchors():
    """The prototype left an exactly coincident pair coincident forever: its
    separation direction was 0/0. `_TIE_AXIS` gives the degenerate pair a
    deterministic one, and the labels still sit near their shared anchor."""
    anchor = (-100.0, 40.0)
    pos = F.place_labels([anchor, anchor])
    assert np.linalg.norm(pos[0] - pos[1]) == pytest.approx(F.MIN_SEP_DEG, rel=0.05)
    home = np.array(anchor) + np.array(F.LABEL_OFFSET_DEG)
    assert np.linalg.norm(pos - home, axis=1).max() < F.MIN_SEP_DEG


def test_place_labels_leaves_an_isolated_label_on_its_seed():
    pos = F.place_labels([(-100.0, 40.0), (-80.0, 30.0)])
    home = np.array([(-100.0, 40.0), (-80.0, 30.0)]) + np.array(F.LABEL_OFFSET_DEG)
    assert np.allclose(pos, home)


def test_place_labels_keeps_labels_inside_the_frame():
    lon_min, lon_max, lat_min, lat_max = F.DEFAULT_EXTENT
    corner = (lon_max, lat_max)
    pos = F.place_labels([corner, corner, corner])
    assert (pos[:, 0] <= lon_max - F.FRAME_PAD_DEG + 1e-9).all()
    assert (pos[:, 1] <= lat_max - F.FRAME_PAD_DEG + 1e-9).all()
    assert (pos[:, 0] >= lon_min + F.FRAME_PAD_DEG - 1e-9).all()
    assert (pos[:, 1] >= lat_min + F.FRAME_PAD_DEG - 1e-9).all()


def test_place_labels_raises_no_warning():
    """`inf` on the distance diagonal left `inf * 0` in the push term: the
    result was right and numpy warned on every call. A figure module nobody
    can run under `-W error` is one whose real warnings nobody reads."""
    crowd = [(-100.0, 40.0), (-100.0, 40.0), (-99.5, 40.2), (-80.0, 30.0)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        F.place_labels(crowd)


def test_place_labels_refuses_a_shape_it_cannot_read():
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        F.place_labels([1.0, 2.0, 3.0])


# -- the CSV twin ---------------------------------------------------------


@pytest.mark.parametrize("cohort", F.COHORTS)
def test_csv_columns_are_all_emitted(data, cohort):
    table, counts = F.build_csv(data, cohort, F.DEFAULT_EXTENT)
    assert list(table.columns) == F.csv_columns()
    assert not table.columns.duplicated().any()
    assert len(counts) == len(data.methods) * len(data.run_ids)


def test_the_twin_reproduces_every_panel_title(data):
    """The figure's numbers must come back off the CSV without the parquets."""
    for cohort in F.COHORTS:
        table, counts = F.build_csv(data, cohort, F.DEFAULT_EXTENT)
        drawn = table[table["seed_id"].notna()]
        for c in counts:
            g = drawn[(drawn.run_id == c["run_id"]) & (drawn.method == c["method"])]
            assert int(g["n_tgs"].sum()) == c["n_cohort_tgs"]
            # A site belongs to exactly one seed, so the per-seed counts sum.
            assert int(g["n_sites"].sum()) == c["n_cohort_sites"]
            assert g["seed_id"].nunique() == c["n_cohort_cells"]


def test_an_empty_cohort_still_carries_its_panel_row(data):
    """`near` answers every TG correctly, so its `wrong` panel is empty -- and
    its FALLBACK count would vanish from the twin if the row did."""
    table, _ = F.build_csv(data, "wrong", F.DEFAULT_EXTENT)
    empty = table[(table.method == "near") & (table.run_id == data.run_ids[0])]
    assert len(empty) == 1
    assert empty["seed_id"].isna().all()
    assert int(empty["n_tgs"].iloc[0]) == 0
    assert int(empty["n_cohort_tgs"].iloc[0]) == 0


# -- the frame ------------------------------------------------------------


def test_a_prediction_outside_the_frame_is_counted_not_dropped(space):
    """Cropping the worst predictions is how a map flatters a method."""
    far = _scored(space, [(70.0, -100.0), _offset(SEATTLE, north_km=3)], tgs=[0, 0])
    rows = F.cohort_rows(far, "correct")
    counts = F.panel_counts(
        far, rows, run_id="r", method="m", cohort="correct", extent=F.DEFAULT_EXTENT
    )
    assert F.off_frame(rows, F.DEFAULT_EXTENT).sum() == 1
    assert counts["n_off_map"] == 1
    # Still in the denominator: off-map is a position, not an exclusion.
    assert counts["n_cohort_tgs"] == len(rows)
    assert "1 off-map" in F.panel_title(counts)


# -- wiring ---------------------------------------------------------------


def test_the_cohorts_are_the_graded_cell_labels_only():
    assert F.COHORTS == C.GRADED_CELL_LABELS
    assert C.UNANSWERED not in F.COHORTS


def test_an_unknown_cohort_is_refused():
    with pytest.raises(ValueError, match="unknown cohort"):
        F.validate_cohort(C.UNANSWERED)


def test_the_source_rung_comes_off_the_ladder():
    from scripts.analysis.v5.modules import grid as G

    assert F.SOURCE_NSIDE == G.NSIDE_LADDER[0]


def test_rows_are_ordered_by_term_not_by_arrival(data):
    from scripts.analysis.v5.modules.methods import method_order

    assert data.methods == method_order(data.methods)


def test_a_missing_classify_names_the_command(tmp_path):
    with pytest.raises(MissingArtifactError, match="classify"):
        F.load([_Run("nobody-000000-000000-mesh", tmp_path)], analysis_root=tmp_path)


def test_build_writes_the_triple_per_cohort(run, monkeypatch):
    import matplotlib.pyplot as plt

    captured = []
    real_close = plt.close
    monkeypatch.setattr(plt, "close", captured.append)
    pngs = F.build_for_runs([run], analysis_root=run.root)
    for fig in captured:
        real_close(fig)

    assert [p.name for p in pngs] == [
        F.PNG_NAME.format(cohort=c) for c in F.COHORTS
    ]
    out = pngs[0].parent
    assert out.parent.name == F.KIND
    for cohort in F.COHORTS:
        for template in (F.PNG_NAME, F.CSV_NAME, F.MANIFEST_NAME):
            assert (out / template.format(cohort=cohort)).exists()
        body = json.loads((out / F.MANIFEST_NAME.format(cohort=cohort)).read_text())
        assert body["cohort"] == cohort
        assert body["source_nside"] == NSIDE
        # Unpublished ids sort last and among themselves by term, so the
        # fixture's two land alphabetically -- `methods.method_order`, not
        # the order they came off disk.
        assert body["row_order"] == ["giving up", "near"]
        assert [p["method"] for p in body["panels"]] == ["giving_up", "near"]
        assert "solved_mask" in body["policy"]
        assert "place_labels" in body["known_limitation"]


def test_the_figure_draws_a_panel_per_method_and_run(run, monkeypatch):
    import matplotlib.pyplot as plt

    captured = {}
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.setdefault("fig", fig))
    data = F.load([run], nside=NSIDE, analysis_root=run.root)
    png, counts = F.render(
        data, run.root / "outcome_map.correct.png", cohort="correct"
    )
    fig = captured["fig"]
    real_close(fig)

    assert png.exists() and png.stat().st_size > 10_000
    assert len(counts) == 2
    # Two method panels plus the colorbar's own axes.
    assert len(fig.axes) == len(data.methods) * len(data.run_ids) + 1
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert any("3 FB excl." in t for t in titles)
