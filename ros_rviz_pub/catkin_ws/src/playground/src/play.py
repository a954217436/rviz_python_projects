#!/usr/bin/python3
import pickle
from publish_utils import *

ROS_RATE = 2

if  __name__ == "__main__":

    SEQ_IDX = 0
    DET_PKL_PATH = "/mnt/data/WAYMO_det3d_tiny/train/preds/waymo_centerpoint_pp_two_pfn_stride1_3x/" 


    frame = 0
    bridge = CvBridge()
    rospy.init_node('waymo_node',anonymous=True)
    pcl_pub = rospy.Publisher('waymo_point_cloud', PointCloud2, queue_size=10)
    rate = rospy.Rate(ROS_RATE)

    ################################################################################

    while not rospy.is_shutdown():
        all_dets = pickle.load(open(DET_PKL_PATH + "seq_%d_frame_%d.pkl"%(SEQ_IDX, frame), "rb"))
        
        point_cloud = all_dets['points'][1]
        publish_point_cloud(pcl_pub, point_cloud)

        rospy.loginfo("waymo published")
        rate.sleep()

        frame += 1
        if frame == 198:
            frame = 0


