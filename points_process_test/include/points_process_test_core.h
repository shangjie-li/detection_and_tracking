#pragma once

#include <ros/ros.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

#include <sensor_msgs/PointCloud2.h>

#include <cmath>

#define PI 3.1415926

class PclTestCore
{
  private:
    std::string sub_topic_name;
    std::string pub_topic_name;
    
    bool limit_mode;
    bool clip_mode;
    bool filter_mode;
    bool downsample_mode;
    bool show_points_size;
    
    float field_of_view;

    float sensor_height;
    float view_higher_limit;
    float view_lower_limit;
    float min_distance;
    float max_distance;
    
    float meank;
    float stdmul;
    float leafsize;
    
    ros::Subscriber sub_point_cloud_;
    ros::Publisher pub_point_cloud_processed_;
    
    //视场限制
    void limit(const pcl::PointCloud<pcl::PointXYZI>::Ptr in,
                        const pcl::PointCloud<pcl::PointXYZI>::Ptr out);
    //区域裁剪
    void clip(const pcl::PointCloud<pcl::PointXYZI>::Ptr in,
                       const pcl::PointCloud<pcl::PointXYZI>::Ptr out);
    //回调函数
    void point_cb(const sensor_msgs::PointCloud2ConstPtr& in_cloud);

  public:
    PclTestCore(ros::NodeHandle &nh);
    ~PclTestCore();
    void Spin();
};

