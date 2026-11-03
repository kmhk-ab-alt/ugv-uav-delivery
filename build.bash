#!/bin/bash
set -eo pipefail
pai_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$pai_root/activate.bash"
cd "$pai_root"
cmake -S "$pai_root/plugins" -B "$pai_root/build/plugins" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$pai_root/build/plugins/lib"
cmake --build "$pai_root/build/plugins" -j 2
colcon build --paths "$pai_root" --packages-select pai_ros \
    --build-base "$pai_root/build/ros" --install-base "$pai_root/install"
