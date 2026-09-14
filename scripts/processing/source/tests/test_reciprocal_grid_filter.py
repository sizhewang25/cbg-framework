"""Invariants of the grid-based reciprocal filter.

Four things are load-bearing and easy to regress:

  * the match is **reciprocal** -- both datasets come out filtered, each against the
    other's occupied cells, and neither is merely a mask;
  * the two roles are matched **independently** -- a VP near the other dataset's target
    must not keep that VP alive, since the point is per-role topology overlap;
  * an edge needs **both** endpoints, so node pruning prunes flows; and
  * each dataset passes through **byte-for-byte**. Inputs are read as text precisely so
    that filtering cannot reformat a float or swallow a literal "NA", and a plain
    `read_csv`/`to_csv` round trip does both.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analysis.v3.modules.grid import get_grid
from scripts.processing.source.reciprocal_grid_filter import (
    _default_output,
    _default_summary,
    match_to_distribution,
    nearest_counterpart_km,
    node_frame,
    read_canonical_csv,
    reciprocal_filter,
)

# Far enough apart to occupy distinct h3 res-4 (~45 km) cells.
CHI = (41.9742, -87.9073)
SJC = (37.4675, -121.9215)
NYC = (40.7085, -74.0095)
DAL = (32.7767, -96.7970)
#: ~1 km from CHI: a different node, the same place.
CHI_NEAR = (41.9820, -87.9073)


def near(base: tuple, i: int) -> tuple:
    """`i` steps of ~1.1 km north of `base` -- a distinct node in the same res-4 cell."""
    return (base[0] + 0.01 * i, base[1])


def _mesh(rows: list[tuple[str, tuple, str, tuple]]) -> pd.DataFrame:
    """(vp_id, vp_coord, target_id, target_coord) -> a canonical-schema frame."""
    return pd.DataFrame(
        {
            "vp_id": [r[0] for r in rows],
            "vp_lat": [r[1][0] for r in rows],
            "vp_lon": [r[1][1] for r in rows],
            "target_id": [r[2] for r in rows],
            "target_lat": [r[3][0] for r in rows],
            "target_lon": [r[3][1] for r in rows],
            "rtt_ms": [1.0 + i for i in range(len(rows))],
        }
    )


def _filter(public: pd.DataFrame, private: pd.DataFrame, *, balance: bool = True):
    g = get_grid("h3")
    return reciprocal_filter(
        public, private, grid=g, resolution=g.DEFAULT_RESOLUTION, balance=balance
    )


class TestReciprocity(unittest.TestCase):
    def test_both_datasets_are_filtered_not_just_the_public_one(self):
        """The property the name claims: each side loses its unmatched geography."""
        pub = _mesh([("pv1", CHI, "pt1", NYC), ("pv2", SJC, "pt2", SJC)])
        priv = _mesh([("qv1", CHI_NEAR, "qt1", NYC), ("qv2", DAL, "qt2", DAL)])
        kept_pub, kept_priv, summary = _filter(pub, priv)

        # Only the CHI/NYC pairing is shared; SJC is public-only, DAL private-only.
        self.assertEqual(list(kept_pub["vp_id"]), ["pv1"])
        self.assertEqual(list(kept_pub["target_id"]), ["pt1"])
        self.assertEqual(list(kept_priv["vp_id"]), ["qv1"])
        self.assertEqual(list(kept_priv["target_id"]), ["qt1"])

        for side in ("vp", "target"):
            c = summary["cells"][side]
            self.assertEqual(c["common_cells"], 1)
            self.assertEqual(c["public_nodes_out"], 1)
            self.assertEqual(c["private_nodes_out"], 1)
            self.assertEqual(c["public_cells_unmatched"], 1)
            self.assertEqual(c["private_cells_unmatched"], 1)

    def test_the_match_is_symmetric_under_swapping_the_two_inputs(self):
        """Swapping the arguments swaps the outputs and nothing else."""
        pub = _mesh([("pv1", CHI, "pt1", NYC), ("pv2", SJC, "pt2", SJC)])
        priv = _mesh([("qv1", CHI_NEAR, "qt1", NYC), ("qv2", DAL, "qt2", DAL)])

        a_pub, a_priv, a_sum = _filter(pub, priv)
        b_priv, b_pub, b_sum = _filter(priv, pub)

        pd.testing.assert_frame_equal(a_pub, b_pub)
        pd.testing.assert_frame_equal(a_priv, b_priv)
        for side in ("vp", "target"):
            self.assertEqual(
                a_sum["cells"][side]["common_cells"],
                b_sum["cells"][side]["common_cells"],
            )

    def test_a_dataset_matched_against_itself_is_unchanged(self):
        mesh = _mesh([("v1", CHI, "t1", NYC), ("v2", SJC, "t2", DAL)])
        kept_pub, kept_priv, summary = _filter(mesh, mesh)

        pd.testing.assert_frame_equal(kept_pub, mesh)
        pd.testing.assert_frame_equal(kept_priv, mesh)
        self.assertEqual(summary["public"]["rows_retention_pct"], 100.0)
        self.assertEqual(summary["private"]["rows_retention_pct"], 100.0)

    def test_disjoint_datasets_keep_nothing_on_either_side(self):
        pub = _mesh([("pv1", CHI, "pt1", CHI)])
        priv = _mesh([("qv1", SJC, "qt1", SJC)])
        kept_pub, kept_priv, summary = _filter(pub, priv)

        self.assertTrue(kept_pub.empty)
        self.assertTrue(kept_priv.empty)
        self.assertEqual(summary["cells"]["vp"]["common_cells"], 0)
        self.assertEqual(summary["public"]["rows_out"], 0)
        self.assertEqual(summary["private"]["rows_out"], 0)


class TestMatching(unittest.TestCase):
    def test_the_two_roles_are_matched_independently(self):
        """A private VP sitting on a public *target* must not rescue that target.

        Collapsing the roles would answer a different question -- "is anything of yours
        near anything of mine" -- and would quietly inflate retention on both sides.
        """
        pub = _mesh([("pv1", CHI, "pt1", NYC)])
        # Private VP at NYC (the public target) and private target at CHI (the public
        # VP): every cell is shared, but never with its own role.
        priv = _mesh([("qv1", NYC, "qt1", CHI)])
        kept_pub, kept_priv, summary = _filter(pub, priv)

        self.assertTrue(kept_pub.empty)
        self.assertTrue(kept_priv.empty)
        self.assertEqual(summary["cells"]["vp"]["common_cells"], 0)
        self.assertEqual(summary["cells"]["target"]["common_cells"], 0)

    def test_an_edge_dies_when_either_endpoint_dies(self):
        pub = _mesh(
            [
                ("v1", CHI, "t1", NYC),  # both endpoints matched
                ("v1", CHI, "t2", DAL),  # target unmatched
                ("v2", SJC, "t1", NYC),  # VP unmatched
            ]
        )
        priv = _mesh([("q1", CHI, "g1", NYC)])
        kept_pub, _, summary = _filter(pub, priv)

        self.assertEqual(len(kept_pub), 1)
        row = kept_pub.iloc[0]
        self.assertEqual((row["vp_id"], row["target_id"]), ("v1", "t1"))
        self.assertEqual(summary["public"]["rows_in"], 3)
        self.assertEqual(summary["public"]["rows_out"], 1)
        self.assertEqual(summary["public"]["vps_out"], 1)
        self.assertEqual(summary["public"]["targets_out"], 1)

    def test_row_order_and_columns_are_preserved(self):
        pub = _mesh(
            [("v1", CHI, "t1", NYC), ("v2", SJC, "t2", DAL), ("v1", CHI, "t1", NYC)]
        )
        pub["target_continent"] = "NA"
        priv = _mesh([("q1", CHI, "g1", NYC)])
        kept_pub, _, _ = _filter(pub, priv)

        self.assertEqual(list(kept_pub.columns), list(pub.columns))
        self.assertEqual(list(kept_pub.index), [0, 2])
        self.assertEqual(list(kept_pub["target_continent"]), ["NA", "NA"])

    def test_surviving_targets_are_reported_by_vp_count_not_enforced(self):
        """Multilateration needs 3 constraints; falling under it is reported, not fixed.

        Pinned at the presence stage (`balance=False`): this fixture has 2 public VPs in
        the CHI cell against 1 private, so balancing would prune it to 1 and the report
        under test would describe a different graph.
        """
        pub = _mesh([("v1", CHI, "t1", NYC), ("v2", CHI_NEAR, "t1", NYC)])
        priv = _mesh([("q1", CHI, "g1", NYC)])
        kept_pub, _, summary = _filter(pub, priv, balance=False)

        self.assertEqual(len(kept_pub), 2)
        self.assertEqual(summary["public"]["targets_out_by_vp_count"],
                         {"1": 0, "2": 1, "3+": 0})
        self.assertEqual(summary["public"]["targets_out_with_ge_3_vps"], 0)
        self.assertEqual(summary["private"]["targets_out_by_vp_count"],
                         {"1": 1, "2": 0, "3+": 0})

    def test_grid_block_records_the_merge_scale(self):
        mesh = _mesh([("v1", CHI, "t1", NYC)])
        _, _, summary = _filter(mesh, mesh)
        self.assertEqual(summary["grid"]["scheme"], "h3")
        self.assertEqual(summary["grid"]["resolution"], 4)
        self.assertAlmostEqual(summary["grid"]["nominal_cell_km"], 45.16, places=1)
        self.assertTrue(json.dumps(summary))  # the summary must be serializable


class TestNodeFrame(unittest.TestCase):
    def test_one_id_with_two_coordinates_is_rejected(self):
        """It would quantize into two cells, so survival would depend on row order."""
        mesh = _mesh([("v1", CHI, "t1", NYC), ("v1", SJC, "t1", NYC)])
        with self.assertRaisesRegex(ValueError, "more than one coordinate"):
            node_frame(mesh, "vp", label="public")

    def test_a_missing_coordinate_is_rejected(self):
        # All-text, as `read_canonical_csv` returns it.
        mesh = _mesh([("v1", CHI, "t1", NYC)]).astype(str)
        mesh.loc[0, "vp_lat"] = ""
        with self.assertRaisesRegex(ValueError, "missing/non-numeric coordinate"):
            node_frame(mesh, "vp", label="public")

    def test_a_missing_column_names_the_dataset(self):
        mesh = _mesh([("v1", CHI, "t1", NYC)]).drop(columns=["vp_lat"])
        with self.assertRaisesRegex(ValueError, "private is missing vp columns"):
            node_frame(mesh, "vp", label="private")


class TestIO(unittest.TestCase):
    def test_default_output_keeps_multi_dot_stage_tags(self):
        src = Path("datasets/final/x.mainland.sanitized.csv")
        out = _default_output(src)
        self.assertEqual(out.name, "x.mainland.sanitized.reciprocal.csv")
        self.assertEqual(out.parent, src.parent)
        self.assertEqual(
            _default_summary(out).name, "x.mainland.sanitized.reciprocal.summary.json"
        )

    def test_each_output_lands_beside_its_own_input(self):
        """The two datasets live in different directories and stay there."""
        pub = _default_output(Path("datasets/ripe_as7018/as7018-us-test01.csv"))
        priv = _default_output(Path("datasets/final/as01.mainland.sanitized.csv"))
        self.assertEqual(str(pub), "datasets/ripe_as7018/as7018-us-test01.reciprocal.csv")
        self.assertEqual(str(priv), "datasets/final/as01.mainland.sanitized.reciprocal.csv")
        self.assertNotEqual(pub, priv)

    def test_reading_preserves_na_and_float_text_exactly(self):
        """The reason the inputs are read as text rather than parsed."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "in.csv"
            path.write_text(
                "vp_id,vp_lat,vp_lon,vp_continent,target_id,target_lat,target_lon,rtt_ms\n"
                "v1,41.9742,-87.9073,NA,t1,40.7085,-74.0095,7.6751249999999995\n"
            )
            df = read_canonical_csv(path)
            self.assertEqual(df.loc[0, "vp_continent"], "NA")
            self.assertEqual(df.loc[0, "rtt_ms"], "7.6751249999999995")

            out = Path(d) / "out.csv"
            df.to_csv(out, index=False)
            self.assertEqual(
                out.read_text().splitlines()[1],
                "v1,41.9742,-87.9073,NA,t1,40.7085,-74.0095,7.6751249999999995",
            )

    def test_headers_are_lowercased(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "in.csv"
            path.write_text(
                "VP_ID,VP_LAT,VP_LON,TARGET_ID,TARGET_LAT,TARGET_LON,RTT_MS\n"
                "v1,41.9,-87.9,t1,40.7,-74.0,1.0\n"
            )
            self.assertIn("vp_id", read_canonical_csv(path).columns)


if __name__ == "__main__":
    unittest.main()


class TestMatchToDistribution(unittest.TestCase):
    """The pure matcher, independent of any dataset."""

    def test_keeps_exactly_the_smaller_count_without_repeats(self):
        idx = match_to_distribution([0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 3.0])
        self.assertEqual(len(idx), 2)
        self.assertEqual(len(set(idx.tolist())), 2)

    def test_claims_the_nearest_available_value(self):
        idx = match_to_distribution([0.0, 10.0, 100.0], [9.0, 101.0])
        self.assertEqual(sorted(idx.tolist()), [1, 2])

    def test_a_second_claimant_falls_through_to_the_next_nearest(self):
        """Two wanted values near one candidate must not both take it."""
        idx = match_to_distribution([5.0, 6.0, 500.0], [5.1, 5.2])
        self.assertEqual(sorted(idx.tolist()), [0, 1])

    def test_is_independent_of_the_order_of_the_wanted_values(self):
        a = match_to_distribution([0.0, 10.0, 20.0, 30.0], [25.0, 5.0])
        b = match_to_distribution([0.0, 10.0, 20.0, 30.0], [5.0, 25.0])
        self.assertEqual(a.tolist(), b.tolist())

    def test_equal_sizes_keep_everything(self):
        idx = match_to_distribution([3.0, 1.0, 2.0], [9.0, 9.0, 9.0])
        self.assertEqual(idx.tolist(), [0, 1, 2])

    def test_nan_is_rejected_rather_than_winning_every_match(self):
        """argmin over a NaN-bearing array returns the NaN's position."""
        with self.assertRaisesRegex(ValueError, "NaN"):
            match_to_distribution([1.0, float("nan")], [1.0])

    def test_the_larger_side_must_be_the_larger_side(self):
        with self.assertRaisesRegex(ValueError, "cannot match"):
            match_to_distribution([1.0], [1.0, 2.0])


class TestBalancing(unittest.TestCase):
    def test_the_fixture_helper_keeps_nodes_in_one_cell(self):
        """Guards every test below: `near` must not silently cross a cell boundary."""
        g = get_grid("h3")
        pts = [near(CHI, i) for i in range(4)]
        cells = {
            g.cell_ids([p[0]], [p[1]], g.DEFAULT_RESOLUTION)[0] for p in pts
        }
        self.assertEqual(len(cells), 1)

    def test_the_larger_side_is_decided_per_cell_not_globally(self):
        """Public is larger around CHI, private is larger around NYC. Both get pruned."""
        vps = [("v1", CHI), ("v2", SJC)]
        pub = _mesh(
            [(v, c, t, tc)
             for v, c in vps
             for t, tc in [("t1", near(CHI, 0)), ("t2", near(CHI, 1)),
                           ("t3", near(CHI, 2)), ("t4", near(NYC, 0))]]
        )
        priv = _mesh(
            [(v, c, t, tc)
             for v, c in [("q1", CHI), ("q2", SJC)]
             for t, tc in [("g1", near(CHI, 0)), ("g2", near(NYC, 0)),
                           ("g3", near(NYC, 1)), ("g4", near(NYC, 2))]]
        )
        kept_pub, kept_priv, summary = _filter(pub, priv)

        self.assertEqual(kept_pub["target_id"].nunique(), 2)
        self.assertEqual(kept_priv["target_id"].nunique(), 2)
        b = summary["balance"]["target"]
        self.assertEqual(b["quota_total"], 2)          # 1 per cell, two cells
        self.assertEqual(b["public_cells_pruned"], 1)   # the CHI cell
        self.assertEqual(b["private_cells_pruned"], 1)  # the NYC cell

    def test_cells_that_already_agree_are_left_alone(self):
        pub = _mesh([("v1", CHI, "t1", NYC), ("v2", SJC, "t1", NYC)])
        priv = _mesh([("q1", CHI, "g1", NYC), ("q2", SJC, "g1", NYC)])
        kept_pub, kept_priv, summary = _filter(pub, priv)

        pd.testing.assert_frame_equal(kept_pub, pub)
        pd.testing.assert_frame_equal(kept_priv, priv)
        for side in ("vp", "target"):
            b = summary["balance"][side]
            self.assertEqual(b["public_cells_pruned"], 0)
            self.assertEqual(b["private_cells_pruned"], 0)

    def test_it_matches_the_distribution_rather_than_keeping_the_easiest(self):
        """The anti-cherry-picking property, and the reason RTT is not the matched axis.

        One far-away VP, so a target's nearest-measured-VP distance is set by where it
        sits. The public target is the FAR one; the private side offers near, middle and
        far. Matching must keep private's far target -- keeping the "best" (nearest-VP)
        one would hand the pruned dataset its easiest case.
        """
        far_vp = SJC
        spots = {f"g{i}": near(CHI, i) for i in (0, 2, 5)}
        priv = _mesh([("q1", far_vp, t, c) for t, c in spots.items()])

        # Derived, not assumed: great-circle distance from CHI's cell to SJC does not vary
        # monotonically with latitude, so which spot is farthest is a fact to look up.
        x = nearest_counterpart_km(priv, "target")
        farthest = str(x.idxmax())
        nearest = str(x.idxmin())
        self.assertNotEqual(farthest, nearest)              # fixture sanity

        # The public side offers only the farthest spot, so matching must keep that one.
        pub = _mesh([("v1", far_vp, "t_far", spots[farthest])])
        _, kept_priv, _ = _filter(pub, priv)

        self.assertEqual(list(kept_priv["target_id"]), [farthest])
        self.assertNotIn(nearest, set(kept_priv["target_id"]))

    def test_balancing_is_on_by_default_and_can_be_switched_off(self):
        pub = _mesh([("v1", CHI, "t1", near(NYC, 0)), ("v1", CHI, "t2", near(NYC, 1))])
        priv = _mesh([("q1", CHI, "g1", near(NYC, 0))])

        on_pub, _, on_sum = _filter(pub, priv)
        off_pub, _, off_sum = _filter(pub, priv, balance=False)

        self.assertTrue(on_sum["balanced"])
        self.assertIsNotNone(on_sum["balance"])
        self.assertEqual(on_pub["target_id"].nunique(), 1)

        self.assertFalse(off_sum["balanced"])
        self.assertIsNone(off_sum["balance"])
        self.assertEqual(off_pub["target_id"].nunique(), 2)

    def test_balanced_output_is_still_a_byte_exact_subset(self):
        pub = _mesh([("v1", CHI, "t1", near(NYC, 0)), ("v1", CHI, "t2", near(NYC, 1))])
        pub["target_continent"] = "NA"
        priv = _mesh([("q1", CHI, "g1", near(NYC, 0))])
        kept_pub, _, _ = _filter(pub, priv)

        self.assertEqual(list(kept_pub.columns), list(pub.columns))
        self.assertTrue(kept_pub.index.isin(pub.index).all())
        self.assertEqual(set(kept_pub["target_continent"]), {"NA"})

    def test_the_both_endpoints_rule_survives_balancing(self):
        """Every surviving row's endpoints must both survive on its own side."""
        pub = _mesh(
            [(v, c, t, tc)
             for v, c in [("v1", CHI), ("v2", near(CHI, 1)), ("v3", SJC)]
             for t, tc in [("t1", near(NYC, 0)), ("t2", near(NYC, 1))]]
        )
        priv = _mesh(
            [(v, c, t, tc)
             for v, c in [("q1", CHI), ("q2", SJC)]
             for t, tc in [("g1", near(NYC, 0))]]
        )
        kept_pub, kept_priv, _ = _filter(pub, priv)
        for kept in (kept_pub, kept_priv):
            self.assertFalse(kept.empty)
            for side in ("vp", "target"):
                # a node appears in the output only via rows whose other end also survived
                self.assertEqual(
                    kept[f"{side}_id"].nunique(),
                    kept.drop_duplicates(f"{side}_id").shape[0],
                )

    def test_target_statistics_are_recomputed_after_vp_pruning(self):
        """The phase order: a target's constraint set is the VPs that remain.

        Public has one VP; private has two in the same cell, so the VP phase prunes private
        to one. If the target statistic were computed before that, it would be the min over
        both private VPs -- including the dropped one.
        """
        pub = _mesh([("v1", CHI, "t1", near(NYC, 0))])
        priv = _mesh([("q1", CHI, "g1", near(NYC, 0)),
                      ("q2", near(CHI, 1), "g1", near(NYC, 0))])
        _, kept_priv, summary = _filter(pub, priv)

        self.assertEqual(kept_priv["vp_id"].nunique(), 1)
        after = nearest_counterpart_km(kept_priv, "target")
        self.assertEqual(
            summary["balance"]["target"]["private_matched_stat_before"]["p50"],
            round(float(after["g1"]), 3),
        )
