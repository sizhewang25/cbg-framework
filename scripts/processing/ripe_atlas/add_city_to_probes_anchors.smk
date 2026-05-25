"""Nominatim reverse-geocoding for the sanitized RIPE Atlas anchor + probe corpora.

Calls scripts/processing/append_city_to_anchors.py once for each side, reading
the sanitized inputs from datasets/ripe_atlas/ and writing {anchor,probe}_city.json
into the same directory. The script doesn't care whether its input is anchors
or probes — both have id-keyed entries with geometry.coordinates, which is all
it touches.

Both rules are resumable: the script preserves prior entries in the output
file and only queries Nominatim for entries not already cached, so re-runs
after a network hiccup fill in the gaps. Each rule's output file is the
cache — delete it to force a full re-fetch.

Nominatim policy (https://operations.osmfoundation.org/policies/nominatim/):
  - Max ~1 req/s — the script sleeps 1.0s between calls by default.
  - >thousands of lookups should go to a self-hosted instance. The probe
    corpus is ~9K entries → expect ~2.5 hours of wall time on the first
    full run; the anchor corpus (~720) is ~12 minutes.

Run from the repo root:
  snakemake -s scripts/processing/ripe_atlas/add_city_to_probes_anchors.smk -j 1

`-j 1` is required: Nominatim's rate limit applies across all callers, so
the two rules cannot run in parallel.
"""

from pathlib import Path

DATASETS = Path("datasets/ripe_atlas")
SCRIPT = "scripts/processing/append_city_to_anchors.py"


rule all:
    input:
        DATASETS / "anchor_city.json",
        DATASETS / "probe_city.json",


rule geocode_anchors:
    input:
        DATASETS / "filtered_anchors.json",
    output:
        DATASETS / "anchor_city.json",
    shell:
        "python {SCRIPT} --input {input} --output {output}"


rule geocode_probes:
    input:
        DATASETS / "filtered_probes.json",
    output:
        DATASETS / "probe_city.json",
    shell:
        "python {SCRIPT} --input {input} --output {output}"
