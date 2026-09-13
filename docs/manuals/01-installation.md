# 01 — Installation

From a bare Linux/macOS device to a working `geoscale` environment. This covers
the Python environment only; provisioning measurement data and producing figures
is [02 — Accuracy plotting](02-accuracy-plotting.md).

> **Ignore `install.sh` and the install section of the root `README.md`.** Those
> belong to the original geoloc-imc-2023 pipeline — Python 3.9, Docker,
> ClickHouse — and are unrelated to the `scripts/analysis/v3` layer. Nothing in
> this manual needs ClickHouse.

---

## 0. Prerequisites

| Requirement | Why | How to check |
|---|---|---|
| Python **3.12** | `requires-python = ">=3.11,<3.13"`; 3.13 lacks wheels for several pins. `.python-version` pins 3.12 | `python3.12 --version` |
| Poetry ≥ 2.0 | `build-system` requires `poetry-core>=2.0.0` | `poetry --version` |
| git, ~2 GB free | The scientific stack (numpy/scipy/matplotlib/cartopy/rasterio) is most of it | — |

Any `3.12.x` patch level works. For an air-gapped target, stop here and use
[`../../OFFLINE_INSTALL.md`](../../OFFLINE_INSTALL.md) plus
[`../../KICK_START.md`](../../KICK_START.md) instead — that path ships a
pre-installed `.venv/` and needs no network.

---

## 1. Clone and install

```bash
git clone git@github.com:sizhewang25/cbg-framework.git
cd cbg-framework
poetry install
```

`poetry.toml` is tracked and sets `in-project = true`, so the environment lands
in `./.venv/` rather than in Poetry's cache. That is what the rest of this
manual assumes.

---

## 2. Put the venv on `PATH`

```bash
export PATH="$PWD/.venv/bin:$PATH"
```

Not optional, and `poetry run` is not a substitute for it. The Snakemake rules
(`scripts/analysis/*.smk`, `scripts/benchmark/v2/Snakefile`, `scripts/processing/**`)
shell out to a bare `python`, so with a pyenv shim first on `PATH` every rule
dies on a missing `pandas` rather than on anything informative.

---

## 3. Verify

```bash
python -c "import h3, astropy_healpix, matplotlib_venn, upsetplot, yaml; print('v3 deps OK')"
poetry check --lock            # expect: All set!
python -m scripts.analysis.v3.cli --help
```

The five imports are the dependencies the v3 analysis layer added on top of the
benchmark's; they are the ones a stale lock file silently omits. `poetry check
--lock` answering anything other than `All set!` means `pyproject.toml` and
`poetry.lock` have drifted — see §5.

---

## 4. Running the tests

The suites use two different runners, and only one of them is installable from
this repo's dependencies as they stand.

```bash
# unittest-style: scripts/libs/*, scripts/benchmark/v2
python -m unittest discover -s scripts/benchmark/v2/tests -t .   # 214 tests, OK

# pytest-style: all of scripts/analysis/v3/tests (24 files, no unittest import)
pip install pytest
python -m pytest scripts/ -q          # 1333 tests + 104 subtests, ~2 min
```

> **`pytest` is neither a declared nor a locked dependency.** So after a clean
> `poetry install` the v3 tests cannot run at all: `unittest discover` on
> `scripts/analysis/v3/tests` collects **0 tests**, because every file there is
> bare `def test_*` functions rather than `unittest.TestCase` subclasses. The
> comment in `pyproject.toml` — "Tests use stdlib `unittest` — no test-runner
> dep needed" — is accurate for the older suites and stale for v3. Install
> `pytest` by hand until it is added as a dev dependency.

---

## 5. Maintaining the dependency files

Three files describe the dependencies, and only the first is edited by hand.

| File | Role | Regenerate with |
|---|---|---|
| `pyproject.toml` | `[project.dependencies]` — the source of truth | hand-edited |
| `poetry.lock` | Resolved versions + a `content-hash` of the above | `poetry lock` |
| `requirements.txt` | Flat mirror for the offline installer's `pip install` | `./package_offline.sh` (step 1 writes it) |

After adding or changing a dependency, run **both** generators. Two failure
modes to know about, both of which have already happened here:

- **A hash-only lock failure.** Promoting a package from transitive to direct
  (`pyyaml`, which arrived via snakemake) changes `pyproject.toml`'s
  `content-hash` without changing the resolved package set, so `poetry check
  --lock` fails while the lock is in fact complete. On Poetry 2.x, `poetry lock`
  refreshes the hash and holds existing pins; `--regenerate` is the one that
  re-resolves and can move versions. Diff the lock afterwards and expect exactly
  one changed line.
- **A drifted `requirements.txt`.** It is generated, but it is also *tracked*, so
  a committed copy can fall behind `pyproject.toml` and an offline install off
  the tracked file will miss the difference. Confirm with:

```bash
python - <<'PY'
import re, tomllib
n = lambda s: re.split(r"[\s\(\[<>=!;]", s.strip())[0].lower().replace("_", "-")
proj = {n(d) for d in tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]}
req = {n(l) for l in open("requirements.txt") if l.strip() and not l.startswith("#")}
print("missing from requirements.txt:", sorted(proj - req) or "none")
print("extra in requirements.txt   :", sorted(req - proj) or "none")
PY
```

---

## Next

[02 — Accuracy plotting](02-accuracy-plotting.md): the data inputs to copy over
and the command sequence that produces the §8.1 accuracy table and outcome bars.
