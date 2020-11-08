# detection_and_tracking

ROS package for detection and tracking

## 安装
 - 需要的环境包括Python3和Pip3以及ros-kinetic-desktop-full
 - 安装YOLACT依赖(Pytorch 1.0.1和TorchVision以及其他相关包)
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
 - 修改相机及激光雷达标定参数`detection_and_tracking/conf/head_camera.yaml`
   ```Shell
   ProjectionMat: !!opencv-matrix
      rows: 3
      cols: 4
      dt: d
      data: [920.6949462890625, 0, 348.0517667538334, 0, 0, 934.1571044921875, 177.8464786820405, 0, 0, 0, 1, 0]
   LidarToCameraMat: !!opencv-matrix
      rows: 4
      cols: 4
      dt: d
      data: [-0.0601795, -0.998042, -0.0170758, 0.129992, -0.0317277, 0.0190107, -0.999316, 0.0185626, 0.997683, -0.0595965, -0.0328096, 0.0337639, 0, 0, 0, 1]
   RotationAngleX: 0
   RotationAngleY: 0
   RotationAngleZ: 0
   ```
 - 修改目标检测及跟踪算法相关参数`detection_and_tracking/scripts/param.yaml`
   ```Shell
   image_topic: /usb_cam/image_rect_color
   lidar_topic: /velodyne_points
   pub_topic: /targets

   calibration_file_path: ~/ros_workspace/detection_and_tracking_ws/src/detection_and_tracking/conf/head_camera.yaml
   ```
    - `image_topic`指明订阅的相机话题。
    - `lidar_topic`指明订阅的激光雷达话题。
    - `pub_topic`指明发布的检测及跟踪结果话题。

## 运行
 - 加载参数文件至ROS参数服务器
   ```Shell
   cd detection_and_tracking/scripts
   rosparam load param.yaml
 - 启动`detection_and_tracking`
   ```Shell
   python3 detection_and_tracking.py
   ```
 - 检测及跟踪的目标发布至话题`/targets`，类型为`BoundingBoxArray`，可以通过rviz查看



