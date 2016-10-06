// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab, The University
// of Tokyo, Japan

#include <task1_vision_pkg/uav_detect_landing_region.h>

UAVLandingRegion::UAVLandingRegion() :
    down_size_(1), ground_plane_(0.0), track_width_(3.0f),
    landing_marker_width_(1.1f), min_wsize_(8), nms_thresh_(0.01f),
    icounter_(0), num_threads_(16) {
    this->nms_client_ = this->pnh_.serviceClient<
       jsk_tasks::NonMaximumSuppression>("non_maximum_suppression");
    
    bool is_train;
    this->pnh_.getParam("train_detector", is_train);
    std::string pkg_directory;
    std::string caffe_solver_path;
    this->pnh_.getParam("pkg_directory", pkg_directory);
    this->pnh_.getParam("caffe_solver_path", caffe_solver_path);
    
    if (is_train) {
       std::string train_pos_data_path;
       std::string train_neg_data_path;
       std::string test_pos_data_path;
       std::string test_neg_data_path;
       this->pnh_.getParam("train_positive_dataset", train_pos_data_path);
       this->pnh_.getParam("train_negative_dataset", train_neg_data_path);
       this->pnh_.getParam("test_positive_dataset", test_pos_data_path);
       this->pnh_.getParam("test_negative_dataset", test_neg_data_path);
       
       this->trainUAVLandingRegionDetector(pnh_, pkg_directory,
                                           caffe_solver_path,
                                           train_pos_data_path,
                                           train_neg_data_path,
                                           test_pos_data_path,
                                           test_neg_data_path);
    }
    
    ROS_INFO("\033[34m LOADING PRE-TRANED DETECTOR \033[0m");
    
    std::string caffe_test_network = pkg_directory +
       "/models/uav_landing_region_test.prototxt";
    std::string caffe_model = pkg_directory +
       "/models/uav_landing_region_iter_10000.caffemodel";
    
    caffe::Caffe::SetDevice(0);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    net_.reset(new caffe::Net<float>(caffe_test_network, caffe::TEST));
    net_->CopyTrainedLayersFrom(caffe_model);

    ROS_INFO("\033[34m LOADED CAFFE MODEL \033[0m");

    /*
    cv::Mat image = cv::imread(
       "/home/krishneel/Desktop/mbzirc/track-data/frame0000.jpg");
    cv::resize(image, image, cv::Size(320, 240));    
    this->traceandDetectLandingMarker(image, image,
                                      this->sliding_window_size_);

    
    
    // cv::cuda::GpuMat d_image(image);
    // cv::cuda::GpuMat feat = this->extractFeauture(d_image);
    // this->caffeClassifer(feat);
    cv::waitKey(0);
    */
    
    this->onInit();
}

void UAVLandingRegion::onInit() {
    this->subscribe();
    this->pub_image_ = pnh_.advertise<sensor_msgs::Image>(
       "/uav_landing_region/output/image", sizeof(char));
    this->pub_point_ = pnh_.advertise<geometry_msgs::PointStamped>(
       "/uav_landing_region/output/point", sizeof(char));

    this->pub_cloud_ = pnh_.advertise<sensor_msgs::PointCloud2>(
       "/uav_landing_region/output/cloud", sizeof(char));
    this->pub_pose_ = pnh_.advertise<geometry_msgs::PoseStamped>(
       "/uav_landing_region/output/pose", sizeof(char));
}

void UAVLandingRegion::subscribe() {
    //! debug only

    this->debug_sub_ = this->pnh_.subscribe(
      "input_image", 1, &UAVLandingRegion::imageCBDebug, this);

   
    this->sub_image_.subscribe(this->pnh_, "input_image", 1);
    this->sub_mask_.subscribe(this->pnh_, "input_mask", 1);
    this->sub_imu_.subscribe(this->pnh_, "input_imu", 1);
    this->sub_odom_.subscribe(this->pnh_, "input_odom", 1);
    this->sub_proj_.subscribe(this->pnh_, "input_proj_mat", 1);
    this->sync_ = boost::make_shared<message_filters::Synchronizer<
       SyncPolicy> >(100);
    this->sync_->connectInput(
       this->sub_image_, this->sub_mask_, this->sub_imu_,
       this->sub_odom_, this->sub_proj_);
    this->sync_->registerCallback(
      boost::bind(
         &UAVLandingRegion::imageCB, this, _1, _2, _3, _4, _5));
}

void UAVLandingRegion::unsubscribe() {
    this->sub_image_.unsubscribe();
    this->sub_mask_.unsubscribe();
}

void UAVLandingRegion::imageCB(
    const sensor_msgs::Image::ConstPtr &image_msg,
    const sensor_msgs::Image::ConstPtr &mask_msg,
    const sensor_msgs::Imu::ConstPtr &imu_msg,
    const nav_msgs::Odometry::ConstPtr &odom_msg,
    const jsk_msgs::ProjectionMatrix::ConstPtr &proj_mat_msg) {


    ROS_INFO("\033[033m In callback \033[0m");
   
    cv::Mat image = this->convertImageToMat(image_msg, "bgr8");
    cv::Mat im_mask = this->convertImageToMat(mask_msg, "mono8");
    if (image.empty() || im_mask.empty()) {
       ROS_ERROR("EMPTY IMAGE. SKIP LANDING SITE DETECTION");
       return;
    }

    cv::Size im_downsize = cv::Size(image.cols/this->down_size_,
                                    image.rows/this->down_size_);
    cv::resize(image, image, im_downsize);
    cv::resize(im_mask, im_mask, im_downsize);


    ROS_INFO("\033[033m -- computing window size \033[0m");
    
    cv::Size wsize = this->getSlidingWindowSize(image.size(), *proj_mat_msg);
    if (wsize.width < this->min_wsize_) {
       ROS_WARN("HIGH ALTITUDE. SKIPPING DETECTION");
       return;
    }

    std::cout << "\033[34m Window Size:  \033[0m" << wsize  << "\n";
    
    ROS_INFO("\033[34m DETECTION \033[0m");
    
    cv::Point2f marker_point = this->traceandDetectLandingMarker(
       image, im_mask, wsize);
    if (marker_point.x == -1) {
       return;
    }


    ROS_INFO("\033[033m -- projecting to 3D coords \033[0m");
    
    Point3DStamped ros_point = this->pointToWorldCoords(
       *proj_mat_msg, marker_point.x * this->down_size_,
       marker_point.y * this->down_size_);


    ROS_INFO("\033[033m -- DONE \033[0m");
    
    ros_point.header = image_msg->header;
    this->pub_point_.publish(ros_point);
    
    cv_bridge::CvImagePtr pub_msg(new cv_bridge::CvImage);
    pub_msg->header = image_msg->header;
    pub_msg->encoding = sensor_msgs::image_encodings::BGR8;
    pub_msg->image = image.clone();
    this->pub_image_.publish(pub_msg);

    geometry_msgs::PoseStamped ros_pose;
    ros_pose.pose = odom_msg->pose.pose;
    ros_pose.pose.orientation.x = imu_msg->orientation.x;
    ros_pose.pose.orientation.y = imu_msg->orientation.y;
    ros_pose.pose.orientation.z = imu_msg->orientation.z;
    ros_pose.pose.orientation.w = imu_msg->orientation.w;
    ros_pose.header = image_msg->header;
    this->pub_pose_.publish(ros_pose);
    
    cv::waitKey(5);
}


void UAVLandingRegion::imageCBDebug(
    const sensor_msgs::Image::ConstPtr &image_msg) {

    ROS_WARN("IN CALLBACK");
    cv::Mat image = this->convertImageToMat(image_msg, "bgr8");
    cv::resize(image, image, cv::Size(320, 240));
    cv::Size wsize = cv::Size(20, 20);
    cv::Point2f marker_point = this->traceandDetectLandingMarker(
       image, image, wsize);

    cv::waitKey(5);
}


cv::Point2f UAVLandingRegion::traceandDetectLandingMarker(
    cv::Mat img, const cv::Mat marker, const cv::Size wsize) {
    if (img.empty() || marker.empty() || img.size() != marker.size()) {
        ROS_ERROR("EMPTY INPUT IMAGE FOR DETECTION");
        return cv::Point2f(-1, -1);
    }

    cv::Mat image = img.clone();
    cv::cuda::GpuMat d_image(img);
    
    if (image.type() != CV_8UC1) {
       cv::cuda::cvtColor(d_image, d_image, CV_BGR2GRAY);
    }
    /*
    cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(
       d_image.type(), d_image.type(), cv::Size(5, 5), 1);
    filter->apply(d_image, d_image);

    cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edge =
       cv::cuda::createCannyEdgeDetector(50.0, 100.0, 3, true);
    cv::cuda::GpuMat d_edge;
    canny_edge->detect(d_image, d_edge);
    */
    
    jsk_tasks::NonMaximumSuppression nms_srv;
    int pyramid_level = 1;
    const int stride = 8;
    cv::cuda::GpuMat d_roi;
    cv::cuda::GpuMat d_desc;

    int iter = 0;
    do {
       cv::cuda::GpuMat d_desc1 = this->extractFeauture(d_image, false);
       int y = 0;
       int x = 0;
       for (int i = 0; i < d_desc1.rows; i++) {
          d_desc = d_desc1.row(i);
          float response = this->caffeClassifer(d_desc);
       
          if  (x >= d_image.cols) {
             y += stride;
             x = 0;
          }
          cv::Rect rect = cv::Rect(x, y, this->sliding_window_size_.width,
                                   this->sliding_window_size_.height);
       
          if (response == 1) {
             jsk_msgs::Rect bbox;
             bbox.x = rect.x;
             bbox.y = rect.y;
             bbox.width = rect.width;
             bbox.height = rect.height;
          
             nms_srv.request.rect.push_back(bbox);
             nms_srv.request.probabilities.push_back(response);
          }
          x += stride;
       }

       cv::cuda::pyrUp(d_image, d_image);
       
    } while (iter++ < pyramid_level);

    
    /*

    int icounter = 0;
    for (int j = 0; j < img.rows; j += stride) {
       for (int i = 0; i < img.cols; i += stride) {
          // if (static_cast<int>(img.at<uchar>(j, i)) != 0) {
             cv::Rect rect = cv::Rect(i, j, wsize.width, wsize.height);
             if (rect.x + rect.width < image.cols &&
                 rect.y + rect.height < image.rows) {
                
                d_roi = d_image(rect);
                cv::cuda::resize(d_roi, d_roi, this->sliding_window_size_);
                d_desc = this->extractFeauture(d_roi);
                float response = this->caffeClassifer(d_desc);

                icounter++;
                
                
                // d_roi = d_desc(rect);
                // float response = this->caffeClassifer(d_roi);
                
                if (response == 1) {

                   cv::rectangle(det_img, rect, cv::Scalar(0, 255, 0), 2);
                   
                   jsk_msgs::Rect bbox;
                   bbox.x = rect.x;
                   bbox.y = rect.y;
                   bbox.width = rect.width;
                   bbox.height = rect.height;
                   
                   nms_srv.request.rect.push_back(bbox);
                   nms_srv.request.probabilities.push_back(response);

                }
             }
             // }
       }
    }
    */
    
    nms_srv.request.threshold = this->nms_thresh_;

    //! 2 - non_max_suprresion
    cv::Mat weight = img.clone();
    cv::Point2f center = cv::Point2f(-1, -1);
    if (this->nms_client_.call(nms_srv)) {
       for (int i = 0; i < nms_srv.response.bbox_count; i++) {
          cv::Rect_<int> rect = cv::Rect_<int>(
             nms_srv.response.bbox[i].x,
             nms_srv.response.bbox[i].y,
             nms_srv.response.bbox[i].width,
             nms_srv.response.bbox[i].height);
          
          center.x = rect.x + rect.width / 2;
          center.y = rect.y + rect.height / 2;

          // for viz
          cv::Point2f vert1 = cv::Point2f(center.x, center.y - wsize.width);
          cv::Point2f vert2 = cv::Point2f(center.x, center.y + wsize.width);
          cv::Point2f hori1 = cv::Point2f(center.x - wsize.width, center.y);
          cv::Point2f hori2 = cv::Point2f(center.x + wsize.width, center.y);
          cv::line(weight, vert1, vert2, cv::Scalar(0, 0, 255), 1);
          cv::line(weight, hori1, hori2, cv::Scalar(0, 0, 255), 1);
          cv::rectangle(weight, rect, cv::Scalar(0, 255, 0), 1);
       }
    } else {
       ROS_FATAL("NON-MAXIMUM SUPPRESSION SRV NOT CALLED");
       return cv::Point2f(-1, -1);
    }
    
    // 3 - return bounding box
    // TODO(REMOVE OTHER FALSE POSITIVES): HERE?

    img = weight.clone();
    
    std::string wname = "result";
    cv::namedWindow(wname, cv::WINDOW_NORMAL);
    cv::imshow(wname, weight);

    return center;
}

float UAVLandingRegion::caffeClassifer(cv::cuda::GpuMat d_desc) {
   
    caffe::Blob<float>* data_layer = this->net_->input_blobs()[0];
    const int BYTE = sizeof(float) * data_layer->height();
    /*
    float* input_data = data_layer->mutable_cpu_data();
    cv::Mat feat;
    d_desc.download(feat);
    std::memcpy(input_data, feat.data, BYTE);
    */
    
    float* input_data_gpu = data_layer->mutable_gpu_data();
    cudaMemcpy(input_data_gpu, d_desc.data, BYTE, cudaMemcpyDeviceToDevice);
    
    /*
    for (int i = 0; i < data_layer->height(); ++i) {
       std::cout << input_data[i] << " " <<
                 feat.at<float>(0, i) << "\n";
    }
    */
    
    // LOG(INFO) << "Blob size: "<< net_->has_blob("fc1");
    this->net_->Forward();

    caffe::Blob<float>* output_layer = this->net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    std::vector<float> predict = std::vector<float>(begin, end);
    
    float max_prob = 0.0f;
    int max_idx = -1;
    for (int i = 0; i < predict.size(); i++) {
       if (predict[i] > max_prob) {
          max_prob = predict[i];
          max_idx = i;
       }
       // std::cout << "PREDICT: " << predict[i]  << "\n";
    }
    return max_idx;
}

cv::Size UAVLandingRegion::getSlidingWindowSize(
    const cv::Size im_size,
    const jsk_msgs::ProjectionMatrix projection_matrix) {
    float A[2][2];
    float bv[2];
    
    const int NUM_POINTS = 2;
    const float pixel_lenght = 50;
    float init_point = im_size.height/2;
    cv::Point2f point[NUM_POINTS];
    point[0] = cv::Point2f(init_point, init_point);
    point[1] = cv::Point2f(init_point + pixel_lenght,
                           init_point);

    cv::Point3_<float> world_coords[NUM_POINTS];
    for (int k = 0; k < NUM_POINTS; k++) {
       Point3DStamped point_3d = this->pointToWorldCoords(
          projection_matrix, static_cast<int>(point[k].x),
          static_cast<int>(point[k].y));
       world_coords[k].x = point_3d.point.x;
       world_coords[k].y = point_3d.point.y;
       world_coords[k].z = point_3d.point.z;
    }
    float world_distance = this->EuclideanDistance(world_coords);

#ifdef _DEBUG
    cv::Mat img = cv::Mat::zeros(im_size, CV_8UC3);
    cv::circle(img, point[0], 5, cv::Scalar(0, 255, 0), -1);
    cv::circle(img, point[1], 5, cv::Scalar(255, 0, 0), -1);
    cv::namedWindow("wsize", cv::WINDOW_NORMAL);
    cv::imshow("wsize", img);
    std::cout << "distance: " <<  world_distance  << "\n";
#endif
    
    float wsize = (pixel_lenght * landing_marker_width_) / world_distance;
    return cv::Size(static_cast<int>(wsize), static_cast<int>(wsize));
}

float UAVLandingRegion::EuclideanDistance(
    const cv::Point3_<float> *world_coords) {
    float x = world_coords[1].x - world_coords[0].x;
    float y = world_coords[1].y - world_coords[0].y;
    return std::sqrt((std::pow(x, 2) + (std::pow(y, 2))));
}

cv::Mat UAVLandingRegion::convertImageToMat(
    const sensor_msgs::Image::ConstPtr &image_msg, std::string encoding) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
       cv_ptr = cv_bridge::toCvCopy(image_msg, encoding);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return cv::Mat();
    }
    return cv_ptr->image.clone();
}

UAVLandingRegion::Point3DStamped UAVLandingRegion::pointToWorldCoords(
    const jsk_msgs::ProjectionMatrix projection_matrix,
    const float x, const float y) {
    float A[2][2];
    float bv[2];
    int i = static_cast<int>(y);
    int j = static_cast<int>(x);
    A[0][0] = j * projection_matrix.data.at(8) -
       projection_matrix.data.at(0);
    A[0][1] = j * projection_matrix.data.at(9) -
       projection_matrix.data.at(1);
    A[1][0] = i * projection_matrix.data.at(8) -
       projection_matrix.data.at(4);
    A[1][1] = i * projection_matrix.data.at(9) -
       projection_matrix.data.at(5);
    bv[0] = projection_matrix.data.at(2)*ground_plane_ +
       projection_matrix.data.at(3) - j*projection_matrix.data.at(
             10)*ground_plane_ - j*projection_matrix.data.at(11);
    bv[1] = projection_matrix.data.at(4)*ground_plane_ +
       projection_matrix.data.at(7) - i*projection_matrix.data.at(
          10)*ground_plane_ - i*projection_matrix.data.at(11);
    float dominator = A[1][1] * A[0][0] - A[0][1] * A[1][0];

    Point3DStamped world_coords;
    world_coords.point.x = (A[1][1]*bv[0]-A[0][1]*bv[1]) / dominator;
    world_coords.point.y = (A[0][0]*bv[1]-A[1][0]*bv[0]) / dominator;
    world_coords.point.z = this->ground_plane_;
    return world_coords;
}

void UAVLandingRegion::predictVehicleRegion(
    cv::Point2f &points, const MotionInfo *motion_info) {
    float dx = motion_info[1].veh_position.x - motion_info[0].veh_position.x;
    float dy = motion_info[1].veh_position.y - motion_info[0].veh_position.y;
    // float dz = motion_info[1].veh_position.z - motion_info[0].veh_position.z;
    float dz = 1.0f;

    dx = (dx == 0) ? 1.0f : dx;
    dy = (dy == 0) ? 1.0f : dy;
    dz = (dz == 0) ? 1.0f : dz;
    
    std::cout << "DIFF: " << dx << ", " << dy << ", "<< dz  << "\n";
    std::cout << "DIFF: " << motion_info[1].veh_position.x << ", "
              << motion_info[1].veh_position.y   << "\n";
    
    const int NUM_STATE = 3;
    float dynamics[NUM_STATE][NUM_STATE] = {{1/dx, 0.0f, 0.0f},
                                            {0.0f, 1/dy, 0.0f},
                                            {0.0f, 0.0f, dz}};
    cv::Mat dynamic_model = cv::Mat(NUM_STATE, NUM_STATE, CV_32F);
    for (int j = 0; j < NUM_STATE; j++) {
       for (int i = 0; i < NUM_STATE; i++) {
          dynamic_model.at<float>(j, i) = dynamics[j][i];
       }
    }
    cv::Mat cur_pos = cv::Mat::zeros(NUM_STATE, sizeof(char), CV_32F);
    cur_pos.at<float>(0, 0) = motion_info[1].veh_position.x;
    cur_pos.at<float>(1, 0) = motion_info[1].veh_position.y;
    // cur_pos.at<float>(2, 0) = motion_info[1].veh_position.z;
    cur_pos.at<float>(2, 0) = 1.0f;
    
    cv::Mat trans = (dynamic_model * cur_pos) + cur_pos;

    std::cout <<"\n" << trans   << "\n";
    
    points.x = trans.at<float>(0, 0);
    points.y = trans.at<float>(1, 0);
    // points.z = trans.at<float>(2, 0);
}


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "task1_vision_pkg");
    UAVLandingRegion ulr;
    ros::spin();
    return 0;
}
