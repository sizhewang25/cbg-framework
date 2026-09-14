export PATH="$PWD/.venv/bin:$PATH"

PUBLIC=datasets/ripe_as7018/as7018-us-test01.csv
PRIVATE=datasets/final/as02-20260728-20260802.mainland.sanitized.csv
OUT=scripts/processing/source/outputs/as7018-ripe-mesh-x-as02-260728-260802-mesh
mkdir -p "$OUT"

python -m scripts.processing.source.reciprocal_grid_filter \
    --public      "$PUBLIC" \
    --private     "$PRIVATE" \
    --out-public  datasets/ripe_as7018/as7018-us-test01.reciprocal-as02.csv \
    --out-private datasets/final/as02-20260728-20260802.mainland.sanitized.reciprocal-as7018.csv \
    --summary     "$OUT/reciprocal.summary.json"