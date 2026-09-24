"""Which rows a method answered -- the one home of that predicate in v5.

Separate from `classify` so `sites` and anything else can import it without
pulling in the scorer. v4 re-exported `solved_mask` from `sites`, which is what
made `sites -> classify -> answer_space` a cycle the moment `answer_space`
needed a site key; v5 has no such edge.

A FALLBACK row carries a real prediction -- the shortest-ping VP's coordinate
-- so filtering on NaN does not drop it. Every count over "answered" rows must
go through `solved_mask`, or a variant is credited with the baseline's answers
exactly where it gave up.
"""

from __future__ import annotations

import pandas as pd

#: The Shortest-Ping control. Not a combo -- it has no LTD/MTL/CTR and no
#: `targets.parquet`; its prediction is the nearest-by-RTT VP's own coordinate.
SHORTEST_PING = "shortest_ping"


def solved_mask(df: pd.DataFrame) -> pd.Series:
    """`SUCCESS` for a CBG arm; every row for an all-`BASELINE` frame.

    The Shortest-Ping control writes `BASELINE` on every row and has no
    fallback path, so it is wholly solved -- otherwise its accuracy would be
    divided by zero answers.
    """
    status = df["status"].astype(str)
    if (status == "BASELINE").all():
        return pd.Series(True, index=df.index)
    return status == "SUCCESS"
