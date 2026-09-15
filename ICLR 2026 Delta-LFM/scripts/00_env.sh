#!/usr/bin/env bash
# Edit these two, then source this file. Nothing else in the repo hard-codes a path.
export DATA_DIR="${DATA_DIR:-/path/to/dataset}"   # contains Image/<subject>/<date>/t1.nii.gz and derived/*.csv
export WORK_DIR="${WORK_DIR:-./outputs}"
export RES_SCALE="${RES_SCALE:-0.82}"             # 0.82 -> 112x112x112; overrides image_size in the YAML
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
mkdir -p "$WORK_DIR"
