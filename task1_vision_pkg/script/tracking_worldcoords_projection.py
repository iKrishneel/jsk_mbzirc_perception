#!/usr/bin/env python

import roslib
roslib.load_manifest('task1_vision_pkg')

import rospy
import numpy as np
import sys
import message_filters

from jsk_mbzirc_msgs.msg import ProjectionMatrix
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pcl2

sub_pmatrix_ = '/projection_matrix'
sub_point2d_ = '/uav_tracking/output/point2d'

class UAVProjectionToWorld:
    def __init__(self):
        rospy.loginfo("running node to project points to world")
        self.proj_matrix = None
        self.ground_z = rospy.get_param('~z_distance', 0.0)
        #self.pub_point3d = rospy.Publisher('/uav_tracking/output/heliport_world_coords', PointStamped, queue_size = 10)
        self.pub_point3d = rospy.Publisher('/uav_landing_region/output/point', PointStamped, queue_size = 10)
        self.pub_cloud = rospy.Publisher('/track_proj', PointCloud2, queue_size = 10)

        self.subscribe()

    def subscribe(self):
        sub_proj = message_filters.Subscriber(sub_pmatrix_, ProjectionMatrix)
        sub_rect = message_filters.Subscriber(sub_point2d_, PointStamped)

        ats = message_filters.ApproximateTimeSynchronizer((sub_proj, sub_rect), 10, 10)
        ats.registerCallback(self.callback)

    def callback(self, proj_msg, point2d_msg):
        self.proj_matrix = np.reshape(proj_msg.data, (3, 4))
        
        point3d = self.projection_to_world_coords(point2d_msg.point.x, point2d_msg.point.y, self.ground_z)

        header = proj_msg.header
        header.frame_id = '/world'

        ros_point = PointStamped()
        ros_point.point.x = point3d[0]
        ros_point.point.y = point3d[1]
        ros_point.point.z = point3d[2]
        ros_point.header = header
        self.pub_point3d.publish(ros_point)

        cloud_points = [[point3d[0], point3d[1], point3d[2]]]
        pcd = pcl2.create_cloud_xyz32(header, cloud_points)
        self.pub_cloud.publish(pcd)

    def projection_to_world_coords(self, x, y, ground_z):
        a00 = x * self.proj_matrix[2, 0] - self.proj_matrix[0, 0]
        a01 = x * self.proj_matrix[2, 1] - self.proj_matrix[0, 1]
        a10 = y * self.proj_matrix[2, 0] - self.proj_matrix[1, 0]
        a11 = y * self.proj_matrix[2, 1] - self.proj_matrix[1, 1]
        bv0 = self.proj_matrix[0, 2] * ground_z + self.proj_matrix[0, 3] -  \
              x * self.proj_matrix[2, 2] * ground_z - x * self.proj_matrix[2, 3]
        bv1 = self.proj_matrix[1, 2] * ground_z + self.proj_matrix[1, 3] -  \
              y * self.proj_matrix[2, 2] * ground_z - y * self.proj_matrix[2, 3]
        denom = a11 * a00 - a01 * a10
        pos_x = (a11 * bv0 - a01 * bv1) / denom
        pos_y = (a00 * bv1 - a10 * bv0) / denom
        return (pos_x, pos_y, ground_z)

def main():
    rospy.init_node('tracking_worldcoords_projection', anonymous=True)
    upw = UAVProjectionToWorld()
    rospy.spin()

if __name__ == "__main__":
    main()
