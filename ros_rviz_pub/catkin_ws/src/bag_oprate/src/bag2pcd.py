# #!/usr/bin/python3

# import rosbag
# import rospy
# import cv2
# import numpy as np
# from sensor_msgs.msg import PointCloud2, PointField
# import sensor_msgs.point_cloud2 as pc2


# def sglidarmsg_to_PointXYZI(src_msg):
#     msg = PointCloud2()
#     msg.header.stamp = src_msg.header.stamp
#     msg.header.frame_id = src_msg.header.frame_id
#     msg.height = src_msg.height
#     msg.width = src_msg.width

#     msg.fields = [
#         PointField('x', 0, PointField.FLOAT32, 1),
#         PointField('y', 4, PointField.FLOAT32, 1),
#         PointField('z', 8, PointField.FLOAT32, 1),
#         PointField('intensity', 12, PointField.FLOAT32, 1)]
#     msg.is_bigendian = False
#     msg.point_step = 16
#     msg.row_step = msg.point_step * msg.width
#     # msg.is_dense = int(np.isfinite(points).all())
#     msg.is_dense = src_msg.is_dense

#     data_arrary = np.empty(
#         [msg.height, msg.width, 4], dtype=np.float32)
#     src_points = pc2.read_points_list(src_msg, field_names=(
#         "x", "y", "z", "intensity"), skip_nans=True)
#     for i in range(msg.height):
#         for j in range(msg.width):
#             src_p = src_points[i*msg.width + j]
#             data_arrary[i][j][0] = src_p[0]
#             data_arrary[i][j][1] = src_p[1]
#             data_arrary[i][j][2] = src_p[2]
#             data_arrary[i][j][3] = src_p[3]

#     # print(data_arrary.shape)
#     msg.data = data_arrary.tostring()
#     return msg, data_arrary


# if  __name__ == "__main__":
#     print("start...")
#     rospy.init_node('bag_node',anonymous=True)
#     print("init_node done...")
    
#     lidar_path = '/mnt/data/SGData/20211224/2021-12-24-11-40-50-3d/lidar2/'
#     bag_path = "/mnt/data/SGData/20211224/2021-12-24-11-40-50-3d/data.bag"

#     topic_name = "/sensor/lidar/fusion"

#     print("loading bag...")
#     with rosbag.Bag(bag_path, 'r') as bag:  #要读取的bag文件
#         print("loading bag success")
#         for topic,msg,t in bag.read_messages():
#             # print(topic)
#             if topic == topic_name:
#                 new_msg, data_arrary = sglidarmsg_to_PointXYZI(msg)
#                 # print("......................")
#                 timestr = msg.header.stamp.to_sec()  #.to_nsec()
#                 print(timestr)
#                 #%.6f表示小数点后带有6位，可根据精确度需要修改；
#                 #timer = 100000000 * timestr
#                 lidar_name = "%.9f" % timestr + "" #命名：时间戳.pcd
#                 save_path = lidar_path + "/" + lidar_name

#                 np.save(save_path, data_arrary.reshape(-1, 4))
#                 # pc.save(save_path, compression='binary_compressed')
