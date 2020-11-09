# detection_and_tracking

ROS package for detection and tracking

## 安装
 - 需要的环境包括Python3和Pip3以及ros-kinetic-desktop-full
 - 安装YOLACT依赖(Pytorch 1.0.1和TorchVision以及一些相关包)
   ```Shell
   sudo pip3 install torch==1.0.1 -f https://download.pytorch.org/whl/cu90/stable
   sudo pip3 install torchvision==0.2.2
   pip3 install cython
   pip3 install opencv-python pillow pycocotools matplotlib
   ```
 - 安装其他依赖
   ```Shell
   # scikit-learn机器学习库
   pip3 install -U scikit-learn
   # Python3与ROS兼容
   pip3 install catkin_tools
   pip3 install rospkg
   # 网络接口数据包捕获函数库
   sudo apt-get install libpcap-dev
   ```
 - 建立工作空间并拷贝这个库
   ```Shell
   mkdir -p ros_ws/src
   cd ros_ws/src
   git clone https://github.com/shangjie-li/detection_and_tracking.git
   cd ..
   catkin_make
   ```

## 参数配置
 - 编写相机及激光雷达标定参数`detection_and_tracking/conf/head_camera.yaml`
   ```Shell
   %YAML:1.0
   ---
   ProjectionMat: !!opencv-matrix
      rows: 3
      cols: 4
      dt: d
      data: [581.921142578125, 0, 605.0637343471462, 0, 0, 604.7725830078125, 332.6973828462578, 0, 0, 0, 1, 0]
   LidarToCameraMat: !!opencv-matrix
      rows: 4
      cols: 4
      dt: d
      data: [0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1]
   RotationAngleX: 0
   RotationAngleY: 0
   RotationAngleZ: 0
   ```
    - `ProjectionMat`：该3x4矩阵为通过相机内参矩阵标定得到的projection_matrix。
    - `LidarToCameraMat`：该4x4矩阵为相机与激光雷达坐标系的转换矩阵，左上角3x3为旋转矩阵，右上角3x1为平移矩阵。
    - `RotationAngleX/Y/Z`：该值是对LidarToCameraMat矩阵进行修正的旋转角度，初始应设置为0，之后根据标定效果进行细微调整，单位为度。
 - 修改目标检测及跟踪算法相关参数`detection_and_tracking/scripts/param.yaml`
   ```Shell
   ...
   
   image_topic: /usb_cam/image_rect_color
   lidar_topic: /velodyne_points
   pub_topic: /targets
   calibration_file_path: /your_path_to/detection_and_tracking/scripts/param.yaml

   is_limit_mode: True
   the_view_number: 1
   the_field_of_view: 100
  
   is_clip_mode: True
   the_sensor_height: 2.0
   the_view_higher_limit: 4.0
   the_view_lower_limit: -2.0
   the_min_distance: 0.5
   the_max_distance: 100
   
   ...
   
   jet_color: 25
   ```
    - `image_topic`指明订阅的相机话题。
    - `lidar_topic`指明订阅的激光雷达话题。
    - `pub_topic`指明发布的检测及跟踪的目标话题。
    - `calibration_file_path`指明标定文件的绝对路径。
    - `the_view_number`为激光雷达视场区域编号，1为x正向，2为y负向，3为x负向，4为y正向。
    - `the_field_of_view`为水平视场角，单位度。
    - `the_sensor_height`指明传感器距地面高度，单位为米。
    - `the_view_higher_limit`和`the_view_lower_limit`指明期望的点云相对地面的限制高度，单位为米。
    - `the_min_distance`和`the_max_distance`指明期望的点云相对传感器的限制距离，单位为米。
    - `jet_color`与点云成像颜色有关。

## 运行
 - 加载参数文件至ROS参数服务器
   ```Shell
   cd detection_and_tracking/scripts
   rosparam load param.yaml
 - 启动`detection_and_tracking`
   ```Shell
   python3 detection_and_tracking.py
   ```
 - 检测及跟踪的目标发布至话题`/targets`，类型为`BoundingBoxArray`，可以通过`rviz`查看

## 附图
   ```Shell
                               \     /    Initial rotation:
                                \ |z/     [0 -1  0]
                                 \|/      [0  0 -1]
                                  █————x  [1  0  0]
                               forward    => (pi/2, -pi/2, 0) Euler angles
                                 cam_1    
  
                                █████
                   |x         ██  |x ██
   [1  0  0]       |         █    |    █                 [-1 0  0]
   [0  0 -1]  z————█ cam_4  █ y———.z    █  cam_2 █————z  [0  0 -1]
   [0  1  0]                 █         █         |       [0 -1  0]
   => (pi/2, 0, 0)            ██     ██          |x      => (-pi/2, 0, pi)
                                █████
                                lidar

                             x————█       [0  1  0]
                                  |       [0  0 -1]
                                  |z      [-1 0  0]
                                 cam_3    => (pi/2, pi/2, 0)
   ```


