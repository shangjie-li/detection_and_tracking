# detection_and_tracking

ROS package for detection and tracking

## 安装
 - 需要的环境包括Python3和ros-kinetic-desktop-full
 - 安装YOLACT依赖(Pytorch 1.0.1和TorchVision以及一些相关包)
   ```Shell
   sudo pip3 install torch==1.0.1 -f https://download.pytorch.org/whl/cu90/stable
   sudo pip3 install torchvision==0.2.2
   sudo pip3 install cython
   sudo pip3 install opencv-python pillow pycocotools matplotlib
   ```
 - 安装其他依赖
   ```Shell
   # scikit-learn机器学习库
   sudo pip3 install -U scikit-learn
   # Python3与ROS兼容
   sudo pip3 install catkin_tools
   sudo pip3 install rospkg
   # 网络接口数据包捕获函数库
   sudo apt-get install libpcap-dev
   ```
 - 建立工作空间并拷贝这个库
   ```Shell
   mkdir -p ros_ws/src
   cd ros_ws/src
   git clone https://github.com/shangjie-li/detection_and_tracking.git --recursive
   cd ..
   catkin_make
   ```
 - 下载模型文件[yolact_resnet50_54_800000.pth](https://drive.google.com/file/d/1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0/view?usp=sharing)，并保存至目录`modules/yolact/weights`

## 参数配置
 - 编写相机与激光雷达标定参数`conf/head_camera.yaml`
   ```Shell
   %YAML:1.0
   ---
   ProjectionMat: !!opencv-matrix
      rows: 3
      cols: 4
      dt: d
      data: [920, 0, 348, 0, 0, 934, 177, 0, 0, 0, 1, 0]
   LidarToCameraMat: !!opencv-matrix
      rows: 4
      cols: 4
      dt: d
      data: [0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1]
   RotationAngleX: 0
   RotationAngleY: 0
   RotationAngleZ: 0
   ```
 - 参数含义如下
   ```Shell
   ProjectionMat:
     该3x4矩阵为通过相机内参标定得到的projection_matrix。
   LidarToCameraMat:
     该4x4矩阵为相机与激光雷达坐标系的齐次转换矩阵，左上角3x3为旋转矩阵，右上角3x1为平移矩阵。
     例如：
     Translation = [dx, dy, dz]T
                [0, -1,  0]
     Rotation = [0,  0, -1]
                [1,  0,  0]
     则：
     LidarToCameraMat = [0, -1,  0, dx]
                        [0,  0, -1, dy]
                        [1,  0,  0, dz]
                        [0,  0,  0,  1]
   RotationAngleX/Y/Z:
     该值是对LidarToCameraMat矩阵进行修正的旋转角度，初始应设置为0，之后根据投影效果进行细微调整，单位为度。
   ```
 - 修改目标检测及跟踪算法相关参数`conf/param.yaml`
   ```Shell
   print_time:                         True
   print_objects_info:                 False
   record_objects_info:                True
  
   sub_image_topic:                    /usb_cam/image_rect_color
   sub_point_clouds_topic:             /pandar_points
   pub_marker_topic:                   /objects
   calibration_file:                   head_camera.yaml
  
   display_image_raw:                  False
   display_image_masked:               False
   display_point_clouds_raw:           False
   display_point_clouds_projected:     False
  
   display_detection_result:           False
   display_segmentation_result:        False
   display_2d_modeling_result:         True
   display_3d_modeling_result:         True
  
   processing_mode: 'DT' # D - detection, DT - detection and tracking
   processing_object: 'both' # car, person, both
  
   pc_view_crop:                       True
   area_number:                        1
   fov_angle:                          100
  
   pc_range_crop:                      True
   sensor_height:                      2.0
   higher_limit:                       4.0
   lower_limit:                        -4.0
   min_distance:                       1.5
   max_distance:                       50.0
  
   blind_update_limit:                 5
   frame_rate:                         10
   max_id:                             10000
   ```
    - `sub_image_topic`指明订阅的图像话题。
    - `sub_point_clouds_topic`指明订阅的点云话题。
    - `pub_marker_topic`指明发布的话题。
    - `calibration_file`指明标定文件的名称。
    - `area_number`为激光雷达视场区域编号，1为x正向，2为y负向，3为x负向，4为y正向。
    - `fov_angle`为相机水平视场角，单位度。
    - `sensor_height`指明激光雷达距地面高度，单位为米。
    - `higher_limit`和`lower_limit`指明期望的点云相对地面的限制高度，单位为米。
    - `min_distance`和`max_distance`指明期望的点云相对激光雷达的限制距离，单位为米。

## 运行
 - 加载参数文件至ROS参数服务器
   ```Shell
   cd conf
   rosparam load param.yaml
 - 启动`detection_and_tracking`
   ```Shell
   cd scripts
   python3 detection_and_tracking.py
   ```
 - 发布的话题类型为`MarkerArray`，可以通过`rviz`查看
 - 如果运行时发生下列错误
   ```Shell
   RuntimeError: /pytorch/torch/csrc/jit/fuser/cuda/fused_kernel.cpp:137: a PTX JIT compilation failed
   ```
    - 出错原因可能是使用的Cuda版本与Pytorch版本不匹配，删除Cuda相关的环境变量即可解决。

## 附图
   ```Shell
                               \     /    Rotation:
                                \ |z/     [0 -1  0]
                                 \|/      [0  0 -1]
                                  █————x  [1  0  0]
                               forward    
                                 cam_1    
  
                                █████
                   |x         ██  |x ██
   [1  0  0]       |         █    |    █                 [-1 0  0]
   [0  0 -1]  z————█ cam_4  █ y———.z    █  cam_2 █————z  [0  0 -1]
   [0  1  0]                 █         █         |       [0 -1  0]
                              ██     ██          |x      
                                █████
                                lidar

                             x————█       [0  1  0]
                                  |       [0  0 -1]
                                  |z      [-1 0  0]
                                 cam_3    
   ```


