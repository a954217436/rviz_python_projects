#!/usr/bin/python2

import os
import cv2
import rospy
import numpy as np

from cv_bridge import CvBridge
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import Image, PointCloud2


DATA_PATH = "/home/zhanghao/code/data/kitti_raw/2011_09_26/2011_09_26_drive_0001_sync/"


if __name__ == "__main__":
    frame = 0

    rospy.init_node("kitti_node", anonymous=True)
    cam_pub = rospy.Publisher("kitti_cam", Image, queue_size=10)
    pcl_pub = rospy.Publisher("kitti_point_cloud", PointCloud2, queue_size=10)
    bridge = CvBridge()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        img = cv2.imread(os.path.join(DATA_PATH, "image_02/data/%010d.png"%frame))
        ros_img = bridge.cv2_to_imgmsg(img, "bgr8")
        cam_pub.publish(ros_img)

        point_cloud = np.fromfile(os.path.join(DATA_PATH, 'velodyne_points/data/%010d.bin'%frame),dtype=np.float32).reshape(-1,4)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        pcl_pub.publish(pcl2.create_cloud_xyz32(header, point_cloud[:,:3]))

        
        rospy.loginfo("camera image published")
        rate.sleep()
        frame += 1
        frame %= 108

