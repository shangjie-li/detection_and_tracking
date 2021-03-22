#!/bin/bash

unset LD_LIBRARY_PATH

source /opt/ros/kinetic/setup.bash
source /home/lishangjie/detection-and-tracking/ros_ws/devel/setup.bash

roslaunch /home/lishangjie/detection-and-tracking/ros_ws/src/detection_and_tracking/launch/dt.launch &
sleep 1

cd /home/lishangjie/detection-and-tracking/ros_ws/src/detection_and_tracking/conf/
rosparam load param.yaml

cd /home/lishangjie/detection-and-tracking/ros_ws/src/detection_and_tracking/scripts/
python3 detection_and_tracking.py
