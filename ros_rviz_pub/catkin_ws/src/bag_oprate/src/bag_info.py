#!/usr/bin/python3

import rosbag
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError



if  __name__ == "__main__":
    print("start...")
    rospy.init_node('bag_node',anonymous=True)
    print("init_node done...")
    
    camera0_path = '/mnt/data/SGData/cam120/'
    # bag_path = "/mnt/data/SGData/2021-11-26-10-09-05.bag"
    bag_path = "/mnt/Public/Rosbag/20211130/2021-11-30-15-15-36-qilin/data.bag"
    topic_name = "/sensor/camera/f120/compressed"
    topic_name_lidar = "/sensor/lidar/fusion"

    bridge = CvBridge()
    print("loading bag...")
    with rosbag.Bag(bag_path, 'r') as bag:  #要读取的bag文件
        print("loading bag success")
        for topic,msg,t in bag.read_messages():
            # print(topic)
            if topic in [topic_name, topic_name_lidar]:

                timestr = msg.header.stamp.to_sec()  #.to_nsec()
                print(timestr)
                
                
