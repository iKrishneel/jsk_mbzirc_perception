// Copyright (C) 2016 by Krishneel Chaudhary @ JSK Lab, The University
// of Tokyo, Japan

#include <task1_vision_pkg/uav_detect_landing_region.h>

UAVLandingRegion::UAVLandingRegion() :
    down_size_(2), ground_plane_(0.0), track_width_(3.0f),
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
    // LOG(INFO) << "Blob size: "<< net_->input_blobs().size();
    
    //! prediction

    cv::Mat image = cv::imread(pkg_directory + "/data/positive/frame0000.jpg");
    cv::Mat feat = this->extractFeauture(image);

    std::cout << "TYPE: " << feat.size()  << "\n";
    
    // std::vector<float> test_features;
    // std::copy(std::istream_iterator<float>(in_file),
    //           std::istream_iterator<float>(),
    //           std::back_inserter(test_features));
    // std::cout << test_features.size()  << "\n";

    caffe::Blob<float>* data_layer = net_->input_blobs()[0];

    std::vector<cv::Mat>* input_channels;
    int width = data_layer->width();
    int height = data_layer->height();

    std::cout << "Info: " << width << "\t" << height  << "\n";
    
    float* input_data = data_layer->mutable_cpu_data();
    // float* input_data = data_layer->mutable_gpu_data();
    for (int i = 0; i < data_layer->height(); ++i) {
       // input_data[i] = test_features.at(i);
       input_data[i] = feat.at<float>(0, i);
    }
    
    std::clock_t start;
    start = std::clock();
    
    // LOG(INFO) << "Blob size: "<< net_->has_blob("fc1");
    net_->Forward();

    caffe::Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    std::vector<float> predict = std::vector<float>(begin, end);

    double duration = (std::clock() - start) /
       static_cast<double>(CLOCKS_PER_SEC);
    std::cout <<"printf: " << duration <<'\n';
    
    
    for (int i = 0; i < predict.size(); i++) {
       std::cout << "PREDICT: " << predict[i]  << "\n";
    }

    // this->onInit();
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
    
    /**
     * DEBUG ONLY
     */
    /*
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
       new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointXYZRGB pt;
    pt.x = ros_point.point.x;
    pt.y = ros_point.point.y;
    pt.z = ros_point.point.z;
    pt.r = 0; pt.g = 255; pt.b = 0;
    cloud->push_back(pt);
    
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*cloud, ros_cloud);
    ros_cloud.header = image_msg->header;
    ros_cloud.header.frame_id = "/world";
    this->pub_cloud_.publish(ros_cloud);
    
    //! update motion
    /*
    this->motion_info_[0] = this->motion_info_[1];
    this->motion_info_[1].veh_position = ros_point.point;
    */
    /*
    this->motion_info_[0] = this->motion_info_[1];
    this->motion_info_[1].veh_position = marker_point;
    this->motion_info_[1].time = image_msg->header.stamp;
    
    if (this->icounter_++ > 1) {
       ROS_INFO("\033[34m COMPUTING MOTION \033[0m");


       std::cout << motion_info_[0].time.toSec() << "\t"
                 << motion_info_[1].time.toSec()  << "\n";
       
       cv::Point2f pred_position;
       this->predictVehicleRegion(pred_position, this->motion_info_);
       cv::circle(image, pred_position, 10, cv::Scalar(255, 0, 255), CV_FILLED);

       std::cout << "Points: " << pred_position  << "\n";
       
       std::string wname = "predict";
       cv::namedWindow(wname, cv::WINDOW_NORMAL);
       cv::imshow(wname, image);
    }
    */
    
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

cv::Point2f UAVLandingRegion::traceandDetectLandingMarker(
    cv::Mat img, const cv::Mat marker, const cv::Size wsize) {
    if (img.empty() || marker.empty() || img.size() != marker.size()) {
        ROS_ERROR("EMPTY INPUT IMAGE FOR DETECTION");
        return cv::Point2f(-1, -1);
    }
    // cv::Mat image = marker.clone();
    cv::Mat image = img.clone();
    if (image.type() != CV_8UC1) {
       cv::cvtColor(image, image, CV_BGR2GRAY);
    }

    cv::GaussianBlur(img, img, cv::Size(5, 5), 1, 0);
    
    cv::Mat im_edge;
    cv::Canny(image, im_edge, 50, 100);
    cv::Mat weight = img.clone();

    jsk_tasks::NonMaximumSuppression nms_srv;
    
    //! 1 - detect
#ifdef _OPENMP
#pragma omp parallel for num_threads(this->num_threads_)
#endif
    for (int j = 0; j < im_edge.rows; j += 2) {
       for (int i = 0; i < im_edge.cols; i += 2) {
          if (static_cast<int>(im_edge.at<uchar>(j, i)) != 0) {
             cv::Rect rect = cv::Rect(i, j, wsize.width, wsize.height);
             if (rect.x + rect.width < image.cols &&
                 rect.y + rect.height < image.rows) {
                cv::Mat roi = img(rect).clone();
                cv::resize(roi, roi, this->sliding_window_size_);
                cv::Mat desc = this->extractFeauture(roi);
                
                float response = this->svm_->predict(desc);
                if (response == 1) {
                   jsk_msgs::Rect bbox;
                   bbox.x = rect.x;
                   bbox.y = rect.y;
                   bbox.width = rect.width;
                   bbox.height = rect.height;
#ifdef _OPENMP
#pragma omp critical
#endif
                   {
                      nms_srv.request.rect.push_back(bbox);
                      nms_srv.request.probabilities.push_back(response);
                   }
                   // cv::rectangle(weight, rect, cv::Scalar(0, 255, 0), 1);
                }
            }
          }
       }
    }
    nms_srv.request.threshold = this->nms_thresh_;

    //! 2 - non_max_suprresion
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
