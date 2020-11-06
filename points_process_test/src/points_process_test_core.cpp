#include "points_process_test_core.h"

PclTestCore::PclTestCore(ros::NodeHandle &nh){
    nh.param<std::string>("sub_topic", sub_topic_name, "/pandar_points");
    nh.param<std::string>("pub_topic", pub_topic_name, "/pandar_points_processed");
    
    nh.param<bool>("is_limit_mode", limit_mode, false);
    nh.param<bool>("is_clip_mode", clip_mode, false);
    nh.param<bool>("is_filter_mode", filter_mode, false);
    nh.param<bool>("is_downsample_mode", downsample_mode, false);
    nh.param<bool>("is_show_points_size", show_points_size, false);
    
    //水平视场角，单位度
    nh.param<float>("the_field_of_view", field_of_view, 90);

    //以地面为基准，设置雷达高度、高度裁剪上限、高度裁剪下限
    nh.param<float>("the_sensor_height", sensor_height, 1.8);
    nh.param<float>("the_view_higher_limit", view_higher_limit, 2.0);
    nh.param<float>("the_view_lower_limit", view_lower_limit, 0.2);
    //设置近处裁剪极限、远处裁剪极限
    nh.param<float>("the_min_distance", min_distance, 2.0);
    nh.param<float>("the_max_distance", max_distance, 50.0);
    //设置滤波算法中用来计算平均值的相邻点的数目、标准偏差阈值的乘值
    nh.param<float>("the_meank", meank, 10);
    nh.param<float>("the_stdmul", stdmul, 0.2);
    //设置缩减采样算法中体素voxel的大小
    nh.param<float>("the_leafsize", leafsize, 0.2);
    
    sub_point_cloud_ = nh.subscribe(sub_topic_name, 1, &PclTestCore::point_cb, this);
    pub_point_cloud_processed_ = nh.advertise<sensor_msgs::PointCloud2>(pub_topic_name, 1);
    
    ros::spin();
}

PclTestCore::~PclTestCore(){}

void PclTestCore::Spin(){
    
}

//视场限制
void PclTestCore::limit(const pcl::PointCloud<pcl::PointXYZI>::Ptr in,
                        const pcl::PointCloud<pcl::PointXYZI>::Ptr out)
{
    pcl::ExtractIndices<pcl::PointXYZI> clipper;//创建ExtractIndices对象

    clipper.setInputCloud(in);//输入点云
    pcl::PointIndices indices;//创建索引
//pragma omp for语法是OpenMP的并行化语法，即希望通过OpenMP并行化执行这条语句后的for循环，从而起到加速效果
#pragma omp for
    float alpha = 90 - 0.5 * field_of_view;
    float k = tan(alpha * PI / 180.0f);
    for (size_t i = 0; i < in->points.size(); i++)
    {
        if (in->points[i].x > k * in->points[i].y && in->points[i].x > -k * in->points[i].y)
        {
            continue;
        }
        else
        {
            indices.indices.push_back(i);//记录点的索引
        }
    }
    clipper.setIndices(boost::make_shared<pcl::PointIndices>(indices));//输入索引
    clipper.setNegative(true); //移除索引的点
    clipper.filter(*out);//输出点云
}

//区域裁剪
void PclTestCore::clip(const pcl::PointCloud<pcl::PointXYZI>::Ptr in,
                       const pcl::PointCloud<pcl::PointXYZI>::Ptr out)
{
    pcl::ExtractIndices<pcl::PointXYZI> clipper;//创建ExtractIndices对象

    clipper.setInputCloud(in);//输入点云
    pcl::PointIndices indices;//创建索引
//pragma omp for语法是OpenMP的并行化语法，即希望通过OpenMP并行化执行这条语句后的for循环，从而起到加速效果
#pragma omp for
    for (size_t i = 0; i < in->points.size(); i++)
    {
        if (in->points[i].z < view_higher_limit - sensor_height &&
            in->points[i].z > view_lower_limit - sensor_height &&
            in->points[i].x * in->points[i].x + in->points[i].y * in->points[i].y > min_distance * min_distance &&
            in->points[i].x * in->points[i].x + in->points[i].y * in->points[i].y < max_distance * max_distance)
        {
            continue;
        }
        indices.indices.push_back(i);//记录点的索引
    }
    clipper.setIndices(boost::make_shared<pcl::PointIndices>(indices));//输入索引
    clipper.setNegative(true); //移除索引的点
    clipper.filter(*out);//输出点云
}

//回调函数point_cb
void PclTestCore::point_cb(const sensor_msgs::PointCloud2ConstPtr & in_cloud_ptr){
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_pc_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr limited_pc_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr clipped_pc_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_pc_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_pc_ptr(new pcl::PointCloud<pcl::PointXYZI>);
    
    pcl::fromROSMsg(*in_cloud_ptr, *current_pc_ptr);
    
    if (show_points_size)
    {
        std::cout<<"points_size of current_pc_ptr:"<<current_pc_ptr->points.size()<<std::endl;
    }
    
    //视场限制
    if (limit_mode)
    {
        limit(current_pc_ptr, limited_pc_ptr);
    }
    else
    {
        limited_pc_ptr = current_pc_ptr;
    }
    
    //区域裁剪
    if (clip_mode)
    {
        clip(limited_pc_ptr, clipped_pc_ptr);
    }
    else
    {
        clipped_pc_ptr = limited_pc_ptr;
    }
    
    //PCL提供的统计离群值剔除算法，该算法限定处于平均值附近的一个范围，并剔除偏离平均值太多的点
    //该算法较为耗时，应在裁剪点云后进行
    if (filter_mode)
    {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZI> statFilter;//定义滤波器statFilter
        statFilter.setInputCloud(clipped_pc_ptr);//输入点云
        statFilter.setMeanK(meank);//设置用来计算平均值的相邻点的数目（偏离平均值太多的点将被剔除）
        statFilter.setStddevMulThresh(stdmul);//设置标准偏差阈值的乘值（偏离平均值μ ± σ·stdmul以上的点认为是离群点）
        statFilter.filter(*filtered_pc_ptr);//输出点云
    }
    else
    {
        filtered_pc_ptr = clipped_pc_ptr;
    }
    
    //PCL提供的体素栅格缩减采样算法，该算法将点云分解成体素voxel，并用子云的中心点代替每个体素voxel中包含的所有点
    //以Pandar40为例，原始点云数量约为每帧140000，经过(0.1,0.1,0.1)缩减采样后约为每帧60000，经过(0.2,0.2,0.2)缩减采样后约为每帧40000
    if (downsample_mode)
    {
        pcl::VoxelGrid<pcl::PointXYZI> voxelSampler;//定义滤波器voxelSampler
        voxelSampler.setInputCloud(filtered_pc_ptr);//输入点云
        voxelSampler.setLeafSize(leafsize, leafsize, leafsize);//设置体素voxel的大小
        voxelSampler.filter(*downsampled_pc_ptr);//输出点云
    }
    else
    {
        downsampled_pc_ptr = filtered_pc_ptr;
    }
    
    if (show_points_size)
    {
        std::cout<<"points_size of downsampled_pc_ptr:"<<downsampled_pc_ptr->points.size()<<std::endl;
    }
    
    sensor_msgs::PointCloud2 pub_pc_processed;
    pcl::toROSMsg(*downsampled_pc_ptr, pub_pc_processed);
    
    pub_pc_processed.header = in_cloud_ptr->header;
    pub_point_cloud_processed_.publish(pub_pc_processed);
}

