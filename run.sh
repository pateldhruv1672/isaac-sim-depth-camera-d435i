#!/usr/bin/env bash
# Launch the D435i-style Isaac Sim publisher.
# Set ISAAC_SIM_PATH to your Isaac Sim release folder (the one containing python.sh).
set -e
cd "$(dirname "$0")"
: "${ISAAC_SIM_PATH:?Set ISAAC_SIM_PATH to the Isaac Sim folder containing python.sh}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-43}"
export ROS_DISTRO="${ROS_DISTRO:-jazzy}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export PYTHONUNBUFFERED=1
unset CONDA_PREFIX CONDA_DEFAULT_ENV
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/fastdds.xml"
export FASTDDS_DEFAULT_PROFILES_FILE="$PWD/fastdds.xml"
if [[ ! -f libalignment.so || alignment.cpp -nt libalignment.so ]]; then
  g++ -O3 -shared -fPIC alignment.cpp -o libalignment.so
fi
exec "$ISAAC_SIM_PATH/python.sh" "$PWD/camera_sim.py" "$@"
