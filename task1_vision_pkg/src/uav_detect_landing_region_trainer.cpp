
#include <task1_vision_pkg/uav_detect_landing_region_trainer.h>

UAVLandingRegionTrainer::UAVLandingRegionTrainer() :
    stride_(8) {
    this->hog_ = boost::shared_ptr<HOGFeatureDescriptor>(
       new HOGFeatureDescriptor());

    this->sliding_window_size_ = cv::Size(32, 32);
}

void UAVLandingRegionTrainer::trainUAVLandingRegionDetector(
    const std::string data_directory, const std::string positive_path,
    const std::string negative_path, const std::string svm_path) {
    this->positive_data_path_ = positive_path;
    this->negative_data_path_ = negative_path;
    this->svm_save_path_ = svm_path;
    this->data_directory_ = data_directory;
    
    std::cout << "ABOUT TO TRAIN"  << "\n";
    this->uploadDataset(svm_path);
}

void UAVLandingRegionTrainer::uploadDataset(
    const std::string dpath) {
    /*
    if (!dpath.empty()) {
       ROS_ERROR("PROVIDE DATASET TEXT FILE PATH");
       return;
    }
   */
    // read text file and extract features and labels
    cv::Mat feature_vector;
    cv::Mat labels;
    this->getTrainingDataset(feature_vector, labels, this->positive_data_path_);
    this->getTrainingDataset(feature_vector, labels, this->negative_data_path_);

    cv::Mat labelMD;
    labels.convertTo(labelMD, CV_32S);
    
    // train
    this->trainSVM(feature_vector, labelMD, this->svm_save_path_);
    
    // generate manifest
    
}

void UAVLandingRegionTrainer::getTrainingDataset(
    cv::Mat &feature_vector, cv::Mat &labels, const std::string directory) {
    if (directory.empty()) {
      std::cout << "PROVIDE DATASET TEXT FILE PATH"  << "\n";
      return;
    }
    std::cout << "Reading Training Dataset......" << std::endl;
    char buffer[255];
    std::ifstream in_file;
    in_file.open((directory).c_str(), std::ios::in);
    if (!in_file.eof()) {
       while (in_file.good()) {
          in_file.getline(buffer, 255);
          std::string read_line(buffer);
          if (!read_line.empty()) {
             std::istringstream iss(read_line);
             std::string img_path;
             iss >> img_path;
             std::string l;
             iss >> l;
             cv::Mat img = cv::imread(this->data_directory_ + img_path,
                                      CV_LOAD_IMAGE_GRAYSCALE);
             if (img.data) {
                cv::resize(img, img, this->sliding_window_size_);
                cv::Mat desc = this->extractFeauture(img);
                if (desc.data) {
                   feature_vector.push_back(desc);
                   float lab = std::atof(l.c_str());
                   labels.push_back(lab);
                }
             }
          }
       }
    }
   
    std::cout << "Training Dataset Reading Completed......" << std::endl;
}


cv::Mat UAVLandingRegionTrainer::extractFeauture(
    cv::Mat &image) {
    if (image.empty()) {
      return cv::Mat();
    }
    cv::resize(image, image, this->sliding_window_size_);
    cv::Mat desc = this->hog_->computeHOG(image);
    
    //! regionlets
    cv::Size wsize = cv::Size(image.size().width/2, image.size().height/2);
    cv::Mat region_desc = this->regionletFeatures(image, wsize);
    cv::hconcat(desc, region_desc, desc);
    return desc;
}

cv::Mat UAVLandingRegionTrainer::regionletFeatures(
    const cv::Mat image, const cv::Size wsize) {
    if (image.empty()) {
       return cv::Mat();
    }
    cv::Mat features;
    for (int j = 0; j < image.rows; j+= wsize.height) {
       for (int i = 0; i < image.cols; i+= wsize.width) {
          cv::Rect rect = cv::Rect(i, j, wsize.width, wsize.height);
          cv::Mat roi = image(rect).clone();
          cv::Mat desc = this->hog_->computeHOG(roi);
          features.push_back(desc);
       }
    }
    features = features.reshape(1, 1);
    return features;
}

void UAVLandingRegionTrainer::trainSVM(
    const cv::Mat feature_vector, cv::Mat labels, std::string save_path) {
    if (feature_vector.empty() || feature_vector.rows != labels.rows) {
       ROS_ERROR("TRAINING FAILED DUE TO UNEVEN DATA");
       return;
    }
    this->svm_ = cv::ml::SVM::create();
    this->svm_->setType(cv::ml::SVM::C_SVC);
    this->svm_->setKernel(cv::ml::SVM::INTER);
    this->svm_->setDegree(0.0);
    this->svm_->setGamma(0.90);
    this->svm_->setCoef0(0.70);
    this->svm_->setC(1);
    this->svm_->setNu(0.70);
    this->svm_->setP(1.0);
    // this->svm_->setClassWeights(cv::Mat());
    cv::TermCriteria term_crit  = cv::TermCriteria(
        cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
        static_cast<int>(1e5), FLT_EPSILON);
    this->svm_->setTermCriteria(term_crit);
    cv::Ptr<cv::ml::ParamGrid> param_grid = new cv::ml::ParamGrid();
    param_grid->minVal = 0;
    param_grid->maxVal = 0;
    param_grid->logStep = 1;
    this->svm_->train(feature_vector, cv::ml::ROW_SAMPLE, labels);

    if (!save_path.empty()) {
       this->svm_->save(static_cast<std::string>(save_path));
       ROS_INFO("\033[34mSVM SUCCESSFULLY TRAINED AND SAVED TO\033[0m");
    }
}
