roscore
roslaunch points_process_test points_process_test.launch
rosparam load param.yaml
python3 detection_and_tracking.py
rosbag play 2020-08-29-12-21-28.bag -l
rviz
