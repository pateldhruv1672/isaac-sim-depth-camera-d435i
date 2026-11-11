#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-43}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/fastdds.xml"
export FASTDDS_DEFAULT_PROFILES_FILE="$PWD/fastdds.xml"
exec rviz2 -d "$PWD/camera.rviz" --ros-args -p use_sim_time:=true
