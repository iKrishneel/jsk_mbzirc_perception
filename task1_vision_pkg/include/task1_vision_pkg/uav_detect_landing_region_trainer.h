
#pragma once
#ifndef _UAV_DETECT_LANDING_REGION_TRAINER_H_
#define _UAV_DETECT_LANDING_REGION_TRAINER_H_

#include <ros/ros.h>
#include <ros/console.h>
#include <task1_vision_pkg/histogram_of_oriented_gradients.h>
#include <fstream>

#include <boost/thread/mutex.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>

class UAVLandingRegionTrainer {

 private:
    int stride_;
    std::string positive_data_path_;
    std::string negative_data_path_;
    std::string data_directory_;

    cv::Ptr<cv::cuda::HOG> cuda_hog_;

 public:
    UAVLandingRegionTrainer();
    void trainUAVLandingRegionDetector(const std::string, const std::string,
                                       const std::string, const std::string);
    void getTrainingDataset(cv::Mat &, cv::Mat &, const std::string);
    void uploadDataset(const std::string);
    cv::Mat extractFeauture(cv::Mat &);
    void trainSVM(const cv::Mat, const cv::Mat, std::string);
    cv::Mat regionletFeatures(const cv::Mat, const cv::Size);

    boost::shared_ptr<HOGFeatureDescriptor> hog_;
    cv::Ptr<cv::ml::SVM> svm_;
    cv::Size sliding_window_size_;
    std::string svm_save_path_;
};


#endif  // _UAV_DETECT_LANDING_REGION_TRAINER_H_
