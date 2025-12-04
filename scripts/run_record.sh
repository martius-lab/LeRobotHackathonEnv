#!/bin/bash
uv run mjpython record_in_sim_demonstration.py --teleop.type=so101_leader \
--teleop.port="/dev/tty.usbmodem5A460841371" \
--teleop.id=lerobot_leader_arm \
--teleop.use_degrees true \
--teleop.calibration_dir .
