#!/usr/bin/env bash
# Stage 0 — build the pair CSV the training stages consume.
#
# Cohort preparation lives in its own top-level package, AD_data_processing/,
# because both projects in this repository share it. This script simply
# delegates there; see AD_data_processing/README.md for the standalone usage.
#
# RAW_DATA_DIR must point at the downloaded cohort (ADNI / AIBL / OASIS).
set -eu
source "$(dirname "$0")/00_env.sh"
: "${RAW_DATA_DIR:?set RAW_DATA_DIR to the raw cohort directory}"

PREP="$(dirname "$0")/../../AD_data_processing"
[ -d "$PREP" ] || { echo "AD_data_processing/ not found at $PREP" >&2; exit 1; }

OUT_DIR="${DERIVED_DIR:-${DATA_DIR:-.}/derived}" bash "$PREP/run.sh"
