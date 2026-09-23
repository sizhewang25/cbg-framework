"""Scored-frame fixtures shared by the figures' tests.

`classify.score_method`'s output shape, built small enough to reason about.
Both `test_figure_outcome_bars` and `test_figure_error_cdf` need it -- the
bars read the ring columns, the CDF reads `error_km` -- so it lives here
rather than in whichever test file wanted it first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.v4.modules import classify as C


def _cells(
    *,
    ring0=0,
    ring1=0,
    ring2=0,
    beyond=0,
    failed=0,
    errors=None,
    first_id=0,
):
    """Per-target scored rows, the shape `classify.score_method` emits.

    Only the columns `summarize` reads. `beyond` and `failed` both carry
    `ring=-1` and separate on `status`, exactly as the real scorer does:
    "answered and nowhere near" against "never answered".
    """
    rows = []
    tid = first_id
    for ring, count in ((0, ring0), (1, ring1), (2, ring2), (-1, beyond)):
        for _ in range(count):
            rows.append(
                {
                    "target_id": f"t{tid}",
                    "status": "SUCCESS",
                    "ring": ring,
                    "error_km": 100.0,
                    "tg_seed_id": 1,
                    "nearest_seed_id_retired": 1,
                }
            )
            tid += 1
    for _ in range(failed):
        rows.append(
            {
                "target_id": f"t{tid}",
                "status": "FALLBACK",
                "ring": -1,
                "error_km": np.nan,
                "tg_seed_id": 1,
                "nearest_seed_id_retired": -1,
            }
        )
        tid += 1
    df = pd.DataFrame(rows)
    if errors is not None:
        solved = df["status"] == "SUCCESS"
        assert len(errors) == int(solved.sum()), "one error per solved row"
        df.loc[solved, "error_km"] = list(errors)
    return df


def _write_run(root, dataset, spec, *, nside=128):
    """A run on disk with an `accuracy.csv` and one cells parquet per method.

    The summary is produced by `classify.summarize` rather than hand-written, so
    the per-run rows the pooled row gets compared against are the real thing.
    """
    from scripts.analysis.v4.modules.paths import RunPaths

    run = RunPaths(
        run_id=f"{dataset}-260728-260802-mesh",
        root=root,
        source="generic_csv",
        setup="s",
    )
    out = run.cls_accuracy_dir(nside, root=root)
    C.summarize(spec, nside).to_csv(out / C.ACCURACY_CSV, index=False)
    for method, df in spec.items():
        df.to_parquet(out / C.CELLS_PARQUET.format(method=method), index=False)
    return run
