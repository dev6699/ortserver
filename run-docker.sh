#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$script_dir"

image_name="ortserver:local"
container_name="ortserver"

if docker ps -a --format '{{.Names}}' | grep -qx "$container_name"; then
  docker rm -f "$container_name" >/dev/null
fi

docker run -d \
  --name "$container_name" \
  -p 18500:18500 \
  -p 9090:9090 \
  -e ORTSERVER_MODEL_ROOT=/models \
  -e ORTSERVER_HOST=0.0.0.0 \
  -e ORTSERVER_PORT=18500 \
  -e ORTSERVER_METRICS_HOST=0.0.0.0 \
  -e ORTSERVER_METRICS_PORT=9090 \
  -e ORTSERVER_MAX_MESSAGE_MB=16 \
  -e ORTSERVER_RELOAD_INTERVAL_SECONDS=60 \
  -e ORTSERVER_LOG_PREDICT=false \
  -e ORTSERVER_INTRA_THREADS=2 \
  -e ORTSERVER_INTER_THREADS=2 \
  -v "$repo_root/models:/models:ro" \
  "$image_name"
