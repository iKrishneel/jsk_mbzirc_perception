// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab,
// The University of Tokyo

#include <task1_vision_pkg/skeletonization/gpu_skeletonization.h>

GPUSkeletonization::GPUSkeletonization() {
    this->onInit();
}

void GPUSkeletonization::onInit() {
    this->subscribe();
    this->pub_image_ = this->pnh_.advertise<sensor_msgs::Image>(
       "/skeletonization/output/image", 1);
}

void GPUSkeletonization::subscribe() {
    this->sub_image_ = this->pnh_.subscribe(
       "input_image", 1, &GPUSkeletonization::callback, this);
}

void GPUSkeletonization::unsubscribe() {
    this->sub_image_.shutdown();
}

void GPUSkeletonization::callback(
    const sensor_msgs::Image::ConstPtr &image_msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
       cv_ptr = cv_bridge::toCvCopy(
          image_msg, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv::Mat image = cv_ptr->image.clone();
    
    //! CHANGE TO CUDA VERSION
    // int morph_size = 4;
    // cv::Mat element = cv::getStructuringElement(
    //    cv::MORPH_ELLIPSE, cv::Size(2*morph_size + 1, 2*morph_size+1),
    //    cv::Point(morph_size, morph_size));
    // cv::Mat dst;
    // cv::dilate(image, dst, element);
    // element = cv::getStructuringElement(
    //    cv::MORPH_ELLIPSE, cv::Size(3*morph_size + 1, 3*morph_size+1),
    //    cv::Point(morph_size, morph_size));
    // cv::erode(dst, image, element);
    // cv::GaussianBlur(image, image, cv::Size(21, 21), 1, 0);
    
    // cv::imshow("dilate", dst);

    const int im_size = image.rows * image.cols;
    unsigned char skel_points[im_size];
    skeletonizationGPU(image, skel_points);

    image = downsamplePoints(skel_points, image.size());
    
    // cv::imshow("image", image);
    // cv::waitKey(3);
    
    cv_ptr->image = image.clone();
    this->pub_image_.publish(cv_ptr->toImageMsg());

    // delete skel_points;
}

cv::Mat GPUSkeletonization::downsamplePoints(
    unsigned char *skel_points, const cv::Size img_size) {
    int icounter = 0;
    PointCloud::Ptr cloud(new PointCloud);

    // TODO: CONVERT TO 3D POINTS HERE
    
    for (int i = 0; i < img_size.height; i++) {
       for (int j = 0; j < img_size.width; j++) {
          int index = j + (i * img_size.width);
          unsigned char pixel = skel_points[index] * 255;
          if (static_cast<int>(pixel) == 255) {
             PointT pt;
             pt.x = j;
             pt.y = i;
             pt.z = 0.0f;
             pt.r = pixel;
             pt.b = pixel;
             pt.g = pixel;
             cloud->push_back(pt);
          }
       }
    }
    
    const float leaf_size = 10.0f;
    pcl::VoxelGrid<PointT> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(leaf_size, leaf_size, 0.0);
    voxel_grid.filter(*cloud);

    cv::Mat image = cv::Mat::zeros(img_size, CV_8UC1);
    for (int i = 0; i < cloud->size(); i++) {
       PointT pt = cloud->points[i];
       image.at<uchar>(pt.y, pt.x) = 255;
    }

    // std::cout << "\n\n NUM POINTS: " << cloud->size()   << "\n";
    
    PointCloud().swap(*cloud);
    // cv::namedWindow("down", cv::WINDOW_NORMAL);
    // cv::imshow("down", image);
    return image.clone();
}


int main(int argc, char *argv[]) {

    ros::init(argc, argv, "gpu_skeletonization");
    GPUSkeletonization gpu_s;
    ros::spin();
    return 0;
}
