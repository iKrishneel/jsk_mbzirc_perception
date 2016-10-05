
#include <task1_vision_pkg/uav_detect_landing_region_trainer.h>

UAVLandingRegionTrainer::UAVLandingRegionTrainer() :
    stride_(8) {
    this->hog_ = boost::shared_ptr<HOGFeatureDescriptor>(
       new HOGFeatureDescriptor());
    this->sliding_window_size_ = cv::Size(32, 32);
    
    this->d_hog_ = cv::cuda::HOG::create(this->sliding_window_size_,
                                         cv::Size(16, 16),
                                         cv::Size(8, 8),
                                         cv::Size(8, 8), 9);
}

void UAVLandingRegionTrainer::trainUAVLandingRegionDetector(
    ros::NodeHandle phn,
    const std::string data_directory, const std::string caffe_solver_path,
    const std::string train_pos_path, const std::string train_neg_path,
    const std::string test_pos_path, const std::string test_neg_path) {
   
    this->positive_data_path_ = train_pos_path;
    this->negative_data_path_ = train_neg_path;
    this->data_directory_ = data_directory;

    ROS_INFO("\033[33m READING AND EXTRACTING FEATURES\33[0m");
    std::string train_feature_path;
    this->readAndExtractImageFeatures(train_feature_path, "/train_features");

    this->positive_data_path_ = test_pos_path;
    this->negative_data_path_ = test_neg_path;
    std::string test_feature_path;
    this->readAndExtractImageFeatures(test_feature_path, "/test_features");
    
    if (train_feature_path.empty() || test_feature_path.empty()) {
       ROS_ERROR("FILE NOT WRITTEN");
       return;
    }

    //! create hdf5
    ROS_INFO("\033[33m CREATING CAFFE HDF5\33[0m");
    ros::ServiceClient hdf5_client = phn.serviceClient<
       task1_vision_pkg::CaffeHdf5Convertor>("caffe_hdf5_convertor");
    task1_vision_pkg::CaffeHdf5Convertor hdf5_srv;

    std_msgs::String p_str;
    p_str.data = train_feature_path;
    std_msgs::String p_str1;
    p_str1.data = "landing_marker_train_features";

    hdf5_srv.request.feature_file_path = p_str;
    hdf5_srv.request.hdf5_filename = p_str1;
    if (!hdf5_client.call(hdf5_srv)) {
       ROS_ERROR("HDF5 CONVERTOR FOR TRAIN SRV FAILED!");
       return;
    }
    
    p_str.data = test_feature_path;
    p_str1.data = "landing_marker_test_features";

    hdf5_srv.request.feature_file_path = p_str;
    hdf5_srv.request.hdf5_filename = p_str1;
    if (!hdf5_client.call(hdf5_srv)) {
       ROS_ERROR("HDF5 CONVERTOR FOR TEST SRV FAILED!");
       return;
    }
    
    ROS_INFO("\033[33m TRAINING NEURAL NET\33[0m");
    ros::ServiceClient caf_net_client = phn.serviceClient<
       task1_vision_pkg::CaffeNetwork>("caffe_network");
    task1_vision_pkg::CaffeNetwork net_srv;
    net_srv.request.caffe_solver_path.data = caffe_solver_path;
    if (!caf_net_client.call(net_srv)) {
       ROS_ERROR("CAFFE NET SRV FAILED!");
       return;
    }
    
    ROS_INFO("\033[34m TRAINED SUCCESSFULLY\33[0m");
}

void UAVLandingRegionTrainer::readAndExtractImageFeatures(
    std::string &feature_txt_path, const std::string suffix) {
    cv::Mat feature_vector;
    cv::Mat labels;
    this->getTrainingDataset(feature_vector, labels, this->positive_data_path_);
    this->getTrainingDataset(feature_vector, labels, this->negative_data_path_);

    cv::Mat labelMD;
    labels.convertTo(labelMD, CV_32S);

    if (labelMD.rows != feature_vector.rows) {
       ROS_ERROR("LABEL AND FEATURE DIMENSIONS ARE NOT SAME");
       return;
    }
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) == NULL) {
       ROS_ERROR("WORKING DIRECTORY ERROR");
       return;
    }
    
    feature_txt_path = std::string(cwd) + suffix + ".txt";
    std::ofstream outfile(feature_txt_path.c_str(), std::ios::out);
    for (int j = 0; j < feature_vector.rows; j++) {
       for (int i = 0; i < feature_vector.cols; i++) {
          outfile << feature_vector.at<float>(j, i) << " ";
       }
       outfile << labelMD.at<int>(j, 0)  << std::endl;
    }
    outfile.close();
    ROS_INFO("WRITING FEATURES TO FILE");
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
    // if (!in_file.eof()) {
    //    while (in_file.good()) {
    //       in_file.getline(buffer, 255);

    std::string read_line;
    while (std::getline(in_file, read_line)) {
       // std::string read_line(buffer);
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
             } else {
                ROS_ERROR("IMAGE NOT FOUND: ");
                std::cout << this->data_directory_ + img_path  << "\n";
             }
          }
    }

    std::cout << feature_vector.size() << " " << labels.size()  << "\n";
    std::cout << "Training Dataset Reading Completed......" << std::endl;
}


cv::Mat UAVLandingRegionTrainer::extractFeauture(
    cv::Mat &image) {
    if (image.empty()) {
      return cv::Mat();
    }
    cv::resize(image, image, this->sliding_window_size_);
    cv::cuda::GpuMat d_image(image);
    if (image.channels() > 1) {
       cv::cuda::cvtColor(d_image, d_image, CV_BGR2GRAY);
    }
    cv::cuda::GpuMat d_descriptor;
    this->d_hog_->compute(d_image, d_descriptor);

    cv::Mat desc;
    d_descriptor.download(desc);
    
    /*
    cv::Mat desc = this->hog_->computeHOG(image);
    //! regionlets
    cv::Size wsize = cv::Size(image.size().width/2, image.size().height/2);
    cv::Mat region_desc = this->regionletFeatures(image, wsize);
    cv::hconcat(desc, region_desc, desc);
    */
    return desc;
}

cv::cuda::GpuMat UAVLandingRegionTrainer::extractFeauture(
    cv::cuda::GpuMat &d_image, bool is_resize) {
    if (d_image.empty()) {
       return cv::cuda::GpuMat();
    }
    if (d_image.channels() > 1) {
       cv::cuda::cvtColor(d_image, d_image, CV_BGR2GRAY);
    }
    if (is_resize) {
       cv::cuda::resize(d_image, d_image, this->sliding_window_size_);
    }
    cv::cuda::GpuMat d_descriptor;
    this->d_hog_->compute(d_image, d_descriptor);
    return d_descriptor;
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
