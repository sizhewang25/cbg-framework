"""The cross-dataset directory rule.

`guard_common_methods` and `guard_disjoint_targets` are exercised through the
figures that raise them, in `test_figure_outcome_bars.py` and
`test_figure_euler.py`. What has no home there is the *naming*: `dataset_slug`
is asserted from both figure modules (it is their label as well as their
directory), but `arm` exists only to keep two directories apart, so its cases
are collected here.
"""

from __future__ import annotations

import pytest

from scripts.analysis.v4.modules import cross


MESH = [
    "as01-260728-260802-mesh",
    "as02-260728-260802-mesh",
    "as03-260728-260802-mesh",
]
WEIGHTED = [r.replace("-mesh", "-weighted") for r in MESH]


class TestArm:
    def test_shared_remainder_is_the_arm(self):
        assert cross.arm(MESH) == "260728-260802-mesh"
        assert cross.arm(WEIGHTED) == "260728-260802-weighted"

    def test_order_independent(self):
        assert cross.arm(MESH) == cross.arm(list(reversed(MESH)))

    def test_mixed_remainders_have_no_arm(self):
        """A set spanning two arms is not an arm, so it keeps the bare name."""
        assert cross.arm([MESH[0], WEIGHTED[1]]) is None

    def test_a_single_run_still_has_its_arm(self):
        assert cross.arm([MESH[0]]) == "260728-260802-mesh"

    def test_a_run_id_with_no_hyphen_has_no_arm(self):
        assert cross.arm(["as7018_us_test01"]) is None

    def test_no_vocabulary_of_arm_names(self):
        """Whatever the run ids share is the arm, date range included."""
        assert cross.arm(["as01-260728-260802", "as02-260728-260802"]) == (
            "260728-260802"
        )


class TestCrossDir:
    def test_the_two_arms_do_not_collide(self, tmp_path):
        """The bug this exists for: both arms used to be `as01+as02+as03`."""
        mesh = cross.cross_dir(MESH, analysis_root=tmp_path)
        weighted = cross.cross_dir(WEIGHTED, analysis_root=tmp_path)
        assert mesh != weighted
        assert mesh.name == "as01+as02+as03@260728-260802-mesh"
        assert weighted.name == "as01+as02+as03@260728-260802-weighted"

    def test_armless_sets_keep_the_bare_dataset_name(self, tmp_path):
        out = cross.cross_dir([MESH[0], WEIGHTED[1]], analysis_root=tmp_path)
        assert out.name == "as01+as02"

    def test_the_directory_is_created(self, tmp_path):
        assert cross.cross_dir(MESH, analysis_root=tmp_path).is_dir()

    def test_order_independent(self, tmp_path):
        a = cross.cross_dir(MESH, analysis_root=tmp_path)
        b = cross.cross_dir(list(reversed(MESH)), analysis_root=tmp_path)
        assert a == b

    def test_the_label_does_not_carry_the_arm(self):
        """`dataset_slug` is printed in figures and CSVs; only the path grew."""
        assert cross.dataset_slug(MESH) == cross.dataset_slug(WEIGHTED)
        assert cross.dataset_slug(MESH) == "as01+as02+as03"


@pytest.mark.parametrize("run_ids", [MESH, WEIGHTED, [MESH[0]]])
def test_the_arm_is_recoverable_from_the_directory_name(run_ids, tmp_path):
    """A figure's provenance can be read off its path."""
    name = cross.cross_dir(run_ids, analysis_root=tmp_path).name
    assert name == f"{cross.dataset_slug(run_ids)}@{cross.arm(run_ids)}"
