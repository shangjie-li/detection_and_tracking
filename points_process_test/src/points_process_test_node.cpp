#include "points_process_test_core.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "points_process_test");

    ros::NodeHandle nh("~");

    PclTestCore core(nh);
    return 0;
}

