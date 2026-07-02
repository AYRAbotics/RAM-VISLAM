#!/bin/bash
set -e

echo "=== RAM-VI SLAM Installation ==="
echo "1. Installing Python dependencies..."
pip3 install -r requirements.txt || pip install -r requirements.txt

echo "2. Building ROS 2 package..."
colcon build --symlink-install --packages-select ram_vi_slam

echo "Installation complete!"
echo "To run the launcher, source the workspace first:"
echo "  source install/setup.bash"
echo "  ./run.sh"
