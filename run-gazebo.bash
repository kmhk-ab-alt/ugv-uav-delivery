#!/bin/bash
set -eo pipefail
pai_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$pai_root/install/setup.bash" ]; then
    echo 'Build the project first: bash build.bash' >&2
    exit 1
fi
source "$pai_root/activate.bash"
cd "$pai_root"
exec python -m pai_ros.gazebo_demo "$@"
