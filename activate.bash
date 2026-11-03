#!/bin/bash
# Source this file from bash. All paths are relative to this checkout.
if [ -z "${BASH_VERSION:-}" ]; then
    echo 'Use bash to source activate.bash' >&2
    return 1
fi
pai_activate() {
    local pai_root pai_hook
    pai_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return 1
    if ! command -v pixi >/dev/null 2>&1; then
        echo 'Install Pixi and add it to PATH before continuing.' >&2
        return 1
    fi
    pai_hook="$(pixi shell-hook --manifest-path "$pai_root/pixi.toml" --shell bash)" || return 1
    unset AMENT_ENVIRONMENT_HOOKS AMENT_CURRENT_PREFIX COLCON_CURRENT_PREFIX
    unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH
    export AMENT_SHELL=bash
    shopt -s extglob
    eval "$pai_hook" || return 1
    export PAI_WORKSPACE="$pai_root"
    export ROS_HOME="$pai_root/runtime/ros"
    export ROS_LOG_DIR="$ROS_HOME/log"
    export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
    export GZ_IP=127.0.0.1
    export GZ_PARTITION=pai_delivery_gazebo
    export OGRE2_RESOURCE_PATH="$pai_root/.pixi/envs/default/lib/OGRE-Next"
    if [ -f "$pai_root/install/setup.bash" ]; then
        source "$pai_root/install/setup.bash" || return 1
    fi
    export GZ_SIM_SYSTEM_PLUGIN_PATH="$pai_root/build/plugins/lib${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
}
pai_activate
