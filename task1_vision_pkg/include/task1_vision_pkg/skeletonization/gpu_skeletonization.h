// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo

#ifndef _GPU_SKELETONIZATION_H_
#define _GPU_SKELETONIZATION_H_

#include <ros/ros.h>
#include <ros/console.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include <geometry_msgs/PolygonStamped.h>
#include <sensor_msgs/Image.h>
#include <task1_vision_pkg/skeletonization/skeletonization_kernel.h>

class GPUSkeletonization {

    typedef pcl::PointXYZRGB PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
   
 private:
    cv::Mat downsamplePoints(unsigned char *, const cv::Size);
   
 protected:
    void onInit();
    void subscribe();
    void unsubscribe();
  
    ros::NodeHandle pnh_;
    ros::Subscriber sub_image_;
    ros::Publisher pub_image_;
   
 public:
    GPUSkeletonization();
    void callback(const sensor_msgs::Image::ConstPtr &);
};


#endif  // _GPU_SKELETONIZATION_H_
