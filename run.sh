#!/bin/bash
if [ -f "install/setup.bash" ]; then
    source install/setup.bash
fi
python3 launcher.py
