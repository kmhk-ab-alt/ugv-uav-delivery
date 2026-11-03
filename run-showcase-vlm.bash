#!/bin/bash
set -eo pipefail
pai_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    key_file="$pai_root/runtime/openrouter-api-key"
    if [ -f "$key_file" ]; then
        OPENROUTER_API_KEY=$(cat "$key_file")
        export OPENROUTER_API_KEY
    fi
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo 'Set OPENROUTER_API_KEY or save the key in runtime/openrouter-api-key before running this demo.' >&2
    exit 1
fi
exec bash "$pai_root/run-gazebo.bash" \
 --physical --lockstep --showcase --scene-perception vlm --perception vlm \
 --base-url https://openrouter.ai/api/v1 \
 --model nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free \
 --real-time-factor 2 --speed-scale 1.25 --meeting-policy eta --operation-limit 180 \
 --budget-policy learned --city-layout blocks --city-seed 75100 --randomize-orders --seed 175100 \
 --ground-count 1 --uav-count 1 --package-count 12 --timeout 1200 \
 --record-dir "$pai_root/runtime/recordings" "$@"
