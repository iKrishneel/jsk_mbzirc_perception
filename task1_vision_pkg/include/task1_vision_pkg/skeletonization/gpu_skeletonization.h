// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo

#ifndef _GPU_SKELETONIZATION_H_
#define _GPU_SKELETONIZATION_H_

#include <ros/ros.h>
#include <ros/console.h>
#include <cv_bridge/cv_bridge.h>

#include <geometry_msgs/PolygonStamped.h>
#include <sensor_msgs/Image.h>

// #include <image_geometry/pinhole_camera_model.h>
#include <task1_vision_pkg/skeletonization/skeletonization_kernel.h>
#include <task1_vision_pkg/Skeletonization.h>

namespace jsk_tasks = task1_vision_pkg;

class GPUSkeletonization {

 private:
      
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
    bool skeletonizationGPUSrv(jsk_tasks::Skeletonization::Request &,
                               jsk_tasks::Skeletonization::Response &);
};


#endif  // _GPU_SKELETONIZATION_H_
