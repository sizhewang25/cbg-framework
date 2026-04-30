# CBG Framework

## Unit Test Rules

- Put modular unit tests under a `tests/` folder inside the module being tested.
  For example, RTT-distance tests belong in `scripts/framework/distance/tests/`.
- Split test files by purpose. Do not mix unrelated component behavior in one
  broad test file. For example, keep `speed_of_internet`, `low_envelope`, and
  `bounded_spline` tests in separate files.
- Test real implementations, not fake model stand-ins, when validating framework
  wrappers. Use small deterministic synthetic datasets to fit the real model
  classes offline.
- Use easy-to-interpret synthetic data. Prefer relationships with simple manual
  calculations, such as `RTT = 0.02 * distance + 5` or distance bounds like
  `100 * RTT - 100` and `100 * RTT + 100`.
- Assert manual expected values directly. Do not compute expected values by
  calling the same function or model output being tested.
- Keep shared fixtures and dataset builders in module-local test helpers, such
  as `tests/helpers.py`, when multiple test files need them.
- Keep tests unit-sized: no ClickHouse, no external data files, no network, and
  no long-running benchmark paths.

## Smoke Test Workflow

- Treat `scripts/analysis/cbg_evaluation/evaluate.py` as the end-to-end smoke
  test path for framework integration.
- When the smoke test exposes an error, do not patch implementation code first.
  Read the relevant framework unit tests, reproduce the bug with a focused unit
  test, and add guardrail assertions for the expected behavior.
- After the failing guardrail test is in place, update the implementation until
  the new unit test and existing framework tests pass.
- Keep smoke-test regressions covered at the lowest framework module that owns
  the behavior, rather than adding broad end-to-end assertions for every bug.

Run framework unit tests with:

```bash
python -m unittest discover scripts/framework -p 'test*.py'
```
