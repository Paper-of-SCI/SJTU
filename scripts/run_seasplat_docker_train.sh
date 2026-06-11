#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $(basename "$0") <DATASET_PATH> <EXPERIMENT_NAME> [extra train args...]"
  echo "Example: $(basename "$0") /home/leo/datasets/reef exp01 --do_seathru --seathru_from_iter 10000"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_PATH="$1"
EXPERIMENT_NAME="$2"
shift 2

if [ ! -d "$DATASET_PATH" ]; then
  echo "[error] dataset path not found: $DATASET_PATH"
  exit 1
fi

export HOST_UID="$(id -u)"
export SEASPAT_DATASET="$(realpath "$DATASET_PATH")"

COMPOSE_FILE="$ROOT_DIR/methods/seasplat/docker-compose.yml"

docker compose -f "$COMPOSE_FILE" build seasplat

docker compose -f "$COMPOSE_FILE" run --rm \
  seasplat \
  python3.10 train.py -s /home/user/data --exp "$EXPERIMENT_NAME" "$@"

