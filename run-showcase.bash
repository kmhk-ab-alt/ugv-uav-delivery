#!/bin/bash
set -eo pipefail
pai_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$pai_root/run-gazebo.bash" \
 --physical --lockstep --stub --showcase --scene-perception fixture --real-time-factor 2 --speed-scale 1.25 --meeting-policy eta --operation-limit 180 \
 --budget-policy learned --city-layout blocks --city-seed 75100 --randomize-orders --seed 175100 \
 --ground-count 1 --uav-count 1 --package-count 12 --timeout 1200 "$@"
