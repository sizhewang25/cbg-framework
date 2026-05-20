"""LTD model checkpoint snapshot — pickle a fitted LTDModel for later loading.

Stateless LTDs (e.g. SpeedOfInternetLTD) don't carry any post-fit state, so
pickling them yields the equivalent of constructing the class with no args.
Rather than write a meaningless pickle, this module detects the stateless case
via FittingResult.args (always empty/None for stateless) and writes a sentinel
`.stateless` marker file instead. Downstream tooling reads either form.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from scripts.framework.v2.ltd.base import FittingResult, LTDModel

_STATELESS_MARKER = ".stateless"
_CHECKPOINT_FILE = "fit_checkpoint.pkl"


def save_ltd_checkpoint(
    ltd: LTDModel,
    fit_result: FittingResult,
    *,
    combo_dir: Path,
) -> Path:
    """Write a checkpoint for the fitted LTD into combo_dir.

    Returns the path actually written (either the pickle or the stateless
    marker). combo_dir must already exist.
    """
    combo_dir.mkdir(parents=True, exist_ok=True)
    if _is_stateless(fit_result):
        marker = combo_dir / _STATELESS_MARKER
        marker.write_text(
            f"{type(ltd).__name__} has no fitted state.\n"
            "Construct the class directly to obtain an equivalent instance.\n"
        )
        return marker
    pickle_path = combo_dir / _CHECKPOINT_FILE
    with open(pickle_path, "wb") as fh:
        pickle.dump(ltd, fh)
    return pickle_path


def load_ltd_checkpoint(combo_dir: Path) -> Optional[LTDModel]:
    """Inverse of save_ltd_checkpoint.

    Returns None when combo_dir contains only a stateless marker (the caller
    should construct the LTD class directly in that case)."""
    pickle_path = combo_dir / _CHECKPOINT_FILE
    if pickle_path.exists():
        with open(pickle_path, "rb") as fh:
            return pickle.load(fh)
    if (combo_dir / _STATELESS_MARKER).exists():
        return None
    raise FileNotFoundError(
        f"No checkpoint at {combo_dir} (expected '{_CHECKPOINT_FILE}' or '{_STATELESS_MARKER}')"
    )


def _is_stateless(fit_result: FittingResult) -> bool:
    """A FittingResult with no args (None or empty dict) signals the LTD
    didn't store any per-VP / per-call learned state."""
    args = fit_result.args
    if args is None:
        return True
    if isinstance(args, dict) and not args:
        return True
    return False
