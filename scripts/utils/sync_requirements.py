"""Generate requirements.txt from the ``[project].dependencies`` table in pyproject.toml.

Poetry is the source of truth for dependencies; requirements.txt exists only so
that people without Poetry can ``pip install -r requirements.txt``. Because it is
derived, it must never be hand-edited -- regenerate it instead::

    python -m scripts.utils.sync_requirements

To verify it is up to date without writing anything (exits 1 on drift)::

    python -m scripts.utils.sync_requirements --check

Only ``[project].dependencies`` is exported. If optional-dependency extras or
Poetry dependency groups are ever added, they are skipped with a warning -- pip
users would silently miss them, so teach this script about them at that point.
"""

from __future__ import annotations

import argparse
import difflib
import logging
import re
import sys
import tomllib
from pathlib import Path

try:
    from packaging.requirements import Requirement
except ModuleNotFoundError:  # keep this script runnable in a bare interpreter
    Requirement = None

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_DEFAULT_REQUIREMENTS = _REPO_ROOT / "requirements.txt"

_HEADER = """\
# Generated from pyproject.toml by scripts/utils/sync_requirements.py -- do not edit.
# Poetry is the source of truth; run `python -m scripts.utils.sync_requirements`
# after changing [project].dependencies.
"""

# Poetry writes PEP 508 requirements with the version spec parenthesised, e.g.
# `numpy (>=1.26.4,<2.0.0)`. pip accepts that, but the bare form is conventional.
_PARENTHESISED = re.compile(
    r"""^\s*
    (?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)      # distribution name
    \s*(?P<extras>\[[^\]]*\])?                # optional extras
    \s*\(\s*(?P<spec>[^)]*?)\s*\)             # parenthesised version spec
    \s*(?P<marker>;.*)?                       # optional environment marker
    $""",
    re.VERBOSE,
)


def _same_requirement(left: str, right: str) -> bool:
    """True when two PEP 508 strings mean the same thing.

    ``packaging`` is only a transitive dependency, so treat it as an optional
    safety net -- the rewrite below is stdlib-only and must work without it.
    """
    if Requirement is None:
        return True
    a, b = Requirement(left), Requirement(right)
    return (
        a.name == b.name
        and a.extras == b.extras
        and a.specifier == b.specifier
        and str(a.marker) == str(b.marker)
    )


def normalize_requirement(raw: str) -> str:
    """Strip Poetry's parentheses from a dependency string, preserving spec order.

    ``packaging`` could re-render the requirement for us, but it sorts the
    specifiers (``pillow<12.0.0,>=10.4.0``), which churns the diff against a
    hand-ordered pyproject. So rewrite the text and, when ``packaging`` is
    importable, use it to confirm the rewrite did not change the meaning.
    """
    match = _PARENTHESISED.match(raw)
    if match is None:
        normalized = " ".join(raw.split())
    else:
        name, extras, spec, marker = match.group("name", "extras", "spec", "marker")
        normalized = f"{name}{extras or ''}{''.join(spec.split())}"
        if marker:
            normalized = f"{normalized}{' ' if marker.startswith(';') else ''}{marker.strip()}"

    if not _same_requirement(raw, normalized):
        raise ValueError(f"normalizing {raw!r} changed its meaning: {normalized!r}")
    return normalized


def render_requirements(pyproject_path: Path) -> str:
    """Build the full requirements.txt body for the given pyproject.toml."""
    pyproject = tomllib.loads(pyproject_path.read_text())
    project = pyproject.get("project", {})

    if project.get("optional-dependencies"):
        logger.warning(
            "[project.optional-dependencies] is not exported to requirements.txt"
        )
    if pyproject.get("tool", {}).get("poetry", {}).get("group"):
        logger.warning("[tool.poetry.group] dependencies are not exported to requirements.txt")

    dependencies = project.get("dependencies")
    if not dependencies:
        raise ValueError(f"no [project].dependencies found in {pyproject_path}")

    lines = [normalize_requirement(dep) for dep in dependencies]
    return _HEADER + "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pyproject", type=Path, default=_DEFAULT_PYPROJECT,
                        help="pyproject.toml to read dependencies from.")
    parser.add_argument("--output", type=Path, default=_DEFAULT_REQUIREMENTS,
                        help="requirements.txt to write (or check).")
    parser.add_argument("--check", action="store_true",
                        help="Do not write; exit 1 if the output file is out of sync.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    expected = render_requirements(args.pyproject)

    if args.check:
        actual = args.output.read_text() if args.output.exists() else ""
        if actual == expected:
            logger.info("%s is up to date (%d requirements)",
                        args.output.name, expected.count("\n") - _HEADER.count("\n"))
            return 0
        diff = difflib.unified_diff(
            actual.splitlines(keepends=True), expected.splitlines(keepends=True),
            fromfile=f"{args.output.name} (current)", tofile=f"{args.output.name} (expected)",
        )
        sys.stderr.writelines(diff)
        logger.error("%s is out of sync with %s; run "
                     "`python -m scripts.utils.sync_requirements`",
                     args.output.name, args.pyproject.name)
        return 1

    args.output.write_text(expected)
    logger.info("wrote %s (%d requirements)",
                args.output, expected.count("\n") - _HEADER.count("\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
