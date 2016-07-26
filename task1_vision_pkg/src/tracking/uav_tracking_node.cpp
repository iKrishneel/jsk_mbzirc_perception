
#include <task1_vision_pkg/uav_tracking.h>

UAVTracker::UAVTracker():
    block_size_(10), tracker_init_(false), object_init_(false),
    down_size_(2), is_type_automatic_(!true) {
    this->onInit();
}

void UAVTracker::onInit() {
    this->subscribe();
    this->pub_image_ = pnh_.advertise<sensor_msgs::Image>(
       "/uav_tracking/output/image", sizeof(char));
    // this->pub_rect_ = pnh_.advertise<jsk_mbzirc_msgs::Rect>(
    //    "/uav_tracking/output/rect", sizeof(char));
    this->pub_rect_ = pnh_.advertise<geometry_msgs::PointStamped>(
       "/uav_tracking/output/point2d", sizeof(char));
}

void UAVTracker::subscribe() {
    if (!is_type_automatic_) {
       this->sub_screen_pt_ = this->pnh_.subscribe(
          "input_screen", 1, &UAVTracker::screenPointCallback, this);
    } else {
       this->init_img_.subscribe(this->pnh_, "init_image", 1);
       this->init_rect_.subscribe(this->pnh_, "input_screen", 1);
       this->sync_ = boost::make_shared<message_filters::Synchronizer<
                                           SyncPolicy> >(100);
       this->sync_->connectInput(this->init_img_, this->init_rect_);
       this->sync_->registerCallback(
          boost::bind(&UAVTracker::screenPointCallbackAuto, this, _1, _2));
    }
    
    this->sub_image_ = pnh_.subscribe(
       "input", 1, &UAVTracker::callback, this);
}

void UAVTracker::unsubscribe() {
    this->sub_image_.shutdown();
}

void UAVTracker::callback(const sensor_msgs::Image::ConstPtr &image_msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
       cv_ptr = cv_bridge::toCvCopy(
          image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
       ROS_ERROR("cv_bridge exception: %s", e.what());
       return;
    }
    if (!object_init_) {
       ROS_WARN_ONCE("PLEASE INITIALIZE THE OBJECT");
       return;
    }
    cv::Mat image = cv_ptr->image;
    cv::resize(image, image, cv::Size(image.cols/this->down_size_,
                                      image.rows/this->down_size_));
    cv::Point2f init_tl = cv::Point2f(this->screen_rect_.x / down_size_,
                                      this->screen_rect_.y / down_size_);
    cv::Point2f init_br = cv::Point2f(
       init_tl.x + (this->screen_rect_.width / this->down_size_),
       init_tl.y + (this->screen_rect_.height / this->down_size_));
    
    cv::Mat im_gray;
    cv::Mat img = image.clone();
    cv::cvtColor(image, im_gray, CV_RGB2GRAY);
    if (!tracker_init_) {
       if (this->is_type_automatic_) {
          cv::cvtColor(this->init_image_, im_gray, CV_RGB2GRAY);
       }
       this->initialise(im_gray, init_tl, init_br);
       this->tracker_init_ = true;
    }
    this->processFrame(im_gray);
    for (int i = 0; i < this->trackedKeypoints.size(); i++) {
       cv::circle(img, this->trackedKeypoints[i].first.pt,
                  3, cv::Scalar(255, 255, 255));
    }
    cv::Scalar color = cv::Scalar(0, 255, 0);
    
    cv::line(img, this->topLeft,  this->topRight, color, 3);
    cv::line(img, this->topRight, this->bottomRight, color, 3);
    cv::line(img, this->bottomRight, this->bottomLeft, color, 3);
    cv::line(img, this->bottomLeft, this->topLeft, color, 3);

    jsk_mbzirc_msgs::Rect jsk_rect;
    jsk_rect.x = this->topLeft.x;
    jsk_rect.y = this->topLeft.y;
    jsk_rect.width = (this->bottomRight.x - jsk_rect.x);
    jsk_rect.height = (this->bottomRight.y - jsk_rect.y);

    geometry_msgs::PointStamped ros_point;
    ros_point.point.x = (jsk_rect.x + (jsk_rect.width / 2));
    ros_point.point.y = (jsk_rect.y + (jsk_rect.height / 2));
    ros_point.header = image_msg->header;
    this->pub_rect_.publish(ros_point);
    // this->pub_rect_.publish(jsk_rect);

    
    
    cv_bridge::CvImagePtr pub_msg(new cv_bridge::CvImage);
    pub_msg->header = image_msg->header;
    pub_msg->encoding = sensor_msgs::image_encodings::BGR8;
    pub_msg->image = img.clone();
    this->pub_image_.publish(pub_msg);
    
    // cv::imshow("image", img);
    // cv::waitKey(3);
}

void UAVTracker::screenPointCallbackAuto(
    const sensor_msgs::Image::ConstPtr & image_msg,
    const geometry_msgs::PolygonStamped::ConstPtr &screen_msg) {
    int x = screen_msg->polygon.points[0].x;
    int y = screen_msg->polygon.points[0].y;
    int width = screen_msg->polygon.points[1].x - x;
    int height = screen_msg->polygon.points[1].y - y;
    this->screen_rect_ = cv::Rect_<int>(x, y, width, height);
    this->object_init_ = false;
    if (width > this->block_size_ && height > this->block_size_) {
       this->object_init_ = true;
       this->tracker_init_ = false;
       try {
          cv_bridge::CvImagePtr cv_ptr;
          cv_ptr = cv_bridge::toCvCopy(
             image_msg, sensor_msgs::image_encodings::BGR8);
          this->init_image_ = cv_ptr->image.clone();
       } catch (cv_bridge::Exception& e) {
          ROS_ERROR("cv_bridge exception: %s", e.what());
          this->object_init_ = false;
          this->tracker_init_ = true;
          return;
       }
       ROS_INFO("OBJECT INTIALIZED. NOW TRACKING..");
    } else {
       ROS_WARN("-- Selected Object Size is too small... Not init tracker");
    }
}

void UAVTracker::screenPointCallback(
    const geometry_msgs::PolygonStamped &screen_msg) {
    int x = screen_msg.polygon.points[0].x;
    int y = screen_msg.polygon.points[0].y;
    int width = screen_msg.polygon.points[1].x - x;
    int height = screen_msg.polygon.points[1].y - y;
    this->screen_rect_ = cv::Rect_<int>(x, y, width, height);
    this->object_init_ = false;
    if (width > this->block_size_ && height > this->block_size_) {
       this->object_init_ = true;
       this->tracker_init_ = false;
       ROS_INFO("OBJECT INTIALIZED. NOW TRACKING..");
    } else {
       ROS_WARN("-- Selected Object Size is too small... Not init tracker");
    }
}

int main(int argc, char *argv[]) {

    ros::init(argc, argv, "jsk_mbzirc_tasks");
    UAVTracker uavt;
    ros::spin();
    return 0;
}


