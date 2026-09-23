"""The analysis layer does not import the benchmark layer to read its input.

`scripts/analysis/v3` describes datasets and scores runs; `scripts/benchmark/v2`
produces them. The canonical `(vp, target, rtt_ms)` CSV is read by both, so its
contract lives in `scripts/libs/canonical/` and neither package owns it.

Before that split, five analysis modules imported `load_canonical_csv` /
`build_pairs` from `benchmark.v2.eval_source`, and because that module borrows
`raw_str` from `sources/generic_csv.py`, the import chain reached
`scripts.framework.v2` — every CBG solver the analysis layer exists to
evaluate — just to parse a CSV. The imports were written inside function bodies
to defer that cost, which hid the dependency rather than removing it.

These tests fail if either half comes back: a top-level reach into the
benchmark package, or the framework arriving behind a library import.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

#: The modules that used to carry the deferred import. Each reads the canonical
#: CSV and must now get it from `scripts.libs.canonical`.
CSV_READING_MODULES = (
    "scripts.analysis.v3.modules.bipartite",
    "scripts.analysis.v3.modules.proximity",
    "scripts.analysis.v3.modules.map_mtl",
    "scripts.analysis.v3.modules.pni",
    "scripts.analysis.v3.modules.pni_strategy",
)

#: `figure_ltd_model` reads LTD checkpoints and the inputs root — artifacts the
#: runner writes, so those are genuine benchmark-layer reads rather than a data
#: contract. They stay deferred and are exempt; this test is about the CSV.
EXEMPT = ("scripts.analysis.v3.modules.figure_ltd_model",)


def _imports_in_subprocess(module: str) -> set[str]:
    """`sys.modules` after importing `module` in a clean interpreter."""
    code = (
        f"import {module}, sys, json;"
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m.startswith('scripts.benchmark') or m.startswith('scripts.framework'))))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    import json

    return set(json.loads(out.stdout.strip().splitlines()[-1]))


class TestCanonicalCsvLayering(unittest.TestCase):
    def test_the_library_stands_alone(self):
        """`scripts.libs.canonical` may not reach into either consumer."""
        leaked = _imports_in_subprocess("scripts.libs.canonical")
        self.assertEqual(
            leaked, set(),
            f"scripts.libs.canonical pulled {sorted(leaked)}; the contract has to "
            f"be importable by both layers, so it may depend on neither",
        )

    def test_csv_readers_do_not_pull_the_benchmark(self):
        for module in CSV_READING_MODULES:
            with self.subTest(module=module):
                leaked = _imports_in_subprocess(module)
                self.assertEqual(
                    leaked, set(),
                    f"{module} pulled {sorted(leaked)}. It reads the canonical CSV, "
                    f"so it should import scripts.libs.canonical — reaching into "
                    f"benchmark.v2.eval_source drags scripts.framework.v2 behind it.",
                )

    def test_the_cli_starts_without_the_benchmark_or_the_solvers(self):
        """The whole v3 entry point, not just the five modules."""
        leaked = _imports_in_subprocess("scripts.analysis.v3.cli")
        self.assertEqual(
            leaked, set(),
            f"importing the v3 CLI pulled {sorted(leaked)}",
        )

    def test_no_module_imports_the_csv_contract_from_the_benchmark(self):
        """Source-level, so a *deferred* reintroduction is caught too.

        The subprocess tests above only see top-level imports; an import moved
        back inside a function body would pass them while restoring exactly the
        coupling this package removed.
        """
        offenders = []
        for path in sorted((REPO_ROOT / "scripts/analysis/v3").rglob("*.py")):
            if any(path.match(p) for p in ("*/tests/*",)) or "__pycache__" in str(path):
                continue
            text = path.read_text()
            for name in ("load_canonical_csv", "build_pairs", "apply_eval_target_filters"):
                if f"eval_source import" in text and name in text:
                    rel = path.relative_to(REPO_ROOT)
                    if str(rel).replace("/", ".").removesuffix(".py") in EXEMPT:
                        continue
                    offenders.append(f"{rel} ({name})")
                    break
        self.assertEqual(
            offenders, [],
            f"these import the canonical-CSV contract from the benchmark layer: "
            f"{offenders}. Import from scripts.libs.canonical instead.",
        )


if __name__ == "__main__":
    unittest.main()
