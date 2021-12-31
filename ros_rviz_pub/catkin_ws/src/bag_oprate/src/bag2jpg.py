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
    
    camera0_path = '/mnt/data/SGData/20211224/2021-12-24-11-40-50-3d/cam30/'
    # bag_path = "/mnt/data/SGData/2021-11-26-10-09-05.bag"
    bag_path = "/mnt/data/SGData/20211224/2021-12-24-11-40-50-3d/data.bag"

    topic_name = "/sensor/camera/f30/compressed"

    bridge = CvBridge()
    print("loading bag...")
    with rosbag.Bag(bag_path, 'r') as bag:  #要读取的bag文件
        print("loading bag success")
        for topic,msg,t in bag.read_messages():
            # print(topic)
            if topic == topic_name: #图像的topic
                #print(type(msg))
                #print(msg)
                try:
                    # cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                    cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                except CvBridgeError as e:
                    print(e)
                timestr = msg.header.stamp.to_sec()  #.to_nsec()
                # print(timestr)
                #%.6f表示小数点后带有6位，可根据精确度需要修改；
                #timer = 100000000 * timestr
                image_name = "%.6f" % timestr + ".jpg" #图像命名：时间戳.png
                print(image_name)
                cv2.imwrite(camera0_path + image_name, cv_image)  #保存；
