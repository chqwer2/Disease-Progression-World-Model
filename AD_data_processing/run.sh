#!/usr/bin/env bash
# Stage 0: build the longitudinal pair CSVs from the raw OASIS release.
# Run from this directory. See README.md for the required downloads.
set -euo pipefail

# ---- config (EDIT) ----
DATA_DIR="${RAW_DATA_DIR:?set RAW_DATA_DIR to the directory containing OASISv1.0}/OASISv1.0"
OUT_DIR="${OUT_DIR:-files}"
CONDITION_FILE="${CONDITION_FILE:-configs/OASIS3_patients_condition.txt}"

mkdir -p "$OUT_DIR"

# Step 0: standardize the raw volumes (run the notebook once).
#   jupyter nbconvert --to notebook --execute OASIS-standardize.ipynb

# Step 1: per-visit table (approx. 10 min)
python s1_generate_csv.py \
    --data_dir       "$DATA_DIR" \
    --condition_file "$CONDITION_FILE" \
    --output_file    "$OUT_DIR/s1_OASIS3_ready.csv"

# Step 2: longitudinal pair tables (oasis_train_A/B/C/D.csv)
python s2_gather_csv.py \
    --dataset_csv "$OUT_DIR/s1_OASIS3_ready.csv" \
    --output_path "$OUT_DIR/" \
    --force

# Step 3: drop pairs failing the registration quality check
ORIG_CSV="$OUT_DIR/oasis_train_B.csv" \
OUT_CSV="$OUT_DIR/oasis_train_B_clean.csv" \
    python s3_clean_data.py

# Step 4 (optional): attach prior-visit columns for history conditioning
if [ "${WITH_PRIORS:-1}" = "1" ]; then
    DERIVED_DIR="$OUT_DIR" python build_multiprior.py
fi

echo "Pair CSVs written to $OUT_DIR/"
