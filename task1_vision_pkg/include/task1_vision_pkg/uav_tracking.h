
#include <omp.h>

#include <ros/ros.h>
#include <ros/console.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>

#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <jsk_mbzirc_msgs/Rect.h>
#include <jsk_mbzirc_msgs/ProjectionMatrix.h>
#include <task1_vision_pkg/CMT.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/video/background_segm.hpp>

#include <boost/thread/mutex.hpp>

class UAVTracker: public CMT {

 private:

    typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image, geometry_msgs::PolygonStamped> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::Image> init_img_;
    message_filters::Subscriber<geometry_msgs::PolygonStamped> init_rect_;
    boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> >sync_;
   
    cv::Rect_<int> screen_rect_;
    cv::Mat init_image_;  //! image used for init
   
    int block_size_;
    bool tracker_init_;
    bool object_init_;

    bool is_type_automatic_;
    int down_size_;
   
 protected:

    ros::NodeHandle pnh_;
    ros::Publisher pub_image_;
    ros::Publisher pub_rect_;
    ros::Subscriber sub_image_;
    ros::Subscriber sub_screen_pt_;
   
    void onInit();
    void subscribe();
    void unsubscribe();
   
 public:
    UAVTracker();
    virtual void callback(const sensor_msgs::Image::ConstPtr &);
    void screenPointCallbackAuto(
       const sensor_msgs::Image::ConstPtr &,
       const geometry_msgs::PolygonStamped::ConstPtr &);
    void screenPointCallback(const geometry_msgs::PolygonStamped &);
   
};

