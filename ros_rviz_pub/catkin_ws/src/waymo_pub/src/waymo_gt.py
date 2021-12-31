#!/usr/bin/python3

# from sort_3d import Sort
# from pc3t import Sort

from data_utils import *
from publish_utils import *
import glob


if  __name__ == "__main__":
    ROS_RATE = 10
    SEQ_IDX = 3
    SHOW_Track = True # show gt-det or gt-track  True  False

    D_PATH = '/mnt/data/WAYMO_det3d_tiny/'
    # D_PATH = '/mnt/data/waymo_opensets/'
    LIDAR_PATH = D_PATH + "train/lidar/seq_%s_frame_"%SEQ_IDX     # val test train   
    LABEL_PATH = D_PATH + "train/annos/seq_%s_frame_"%SEQ_IDX     # val test train
    IMAGE_PATH = D_PATH + "train/images/seq_%s_frame_"%SEQ_IDX    # val test train

    frame_nums = len(glob.glob(LIDAR_PATH+"*"))
    print(frame_nums)

    frame = 0
    bridge = CvBridge()
    rospy.init_node('waymo_node',anonymous=True)
    cam_pub = rospy.Publisher('waymo_cam', Image, queue_size=10)
    pcl_pub = rospy.Publisher('waymo_point_cloud', PointCloud2, queue_size=10)
    ego_pub = rospy.Publisher('waymo_ego_car',Marker, queue_size=10)
    box3d_pub = rospy.Publisher('waymo_3dbox',MarkerArray, queue_size=10)

    # track_pub = rospy.Publisher('3dbox_track',MarkerArray,queue_size=10)
    rate = rospy.Rate(ROS_RATE)

    ################################################################################

    while not rospy.is_shutdown():
        # image = get_image_tfrecord(TF_PATH, frame)
        point_cloud = get_lidar_pkl(LIDAR_PATH + "%s.pkl"%frame)
        boxes7d, labels, ids = get_box3d_pkl(LABEL_PATH + "%s.pkl"%frame)

        image = get_cv_image(IMAGE_PATH + "%s.jpg"%frame)
        # publish_camera(cam_pub, bridge, image)
        publish_point_cloud(pcl_pub, point_cloud)
        publish_ego_car(ego_pub)

        corner_3d_velos = []
        for boxes in boxes7d:
            corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6])
            corner_3d_velos.append(np.array(corner_3d_velo).T)
        
        # ## 只发布一个 box，调试使用的
        # boxes = boxes7d[10]
        # corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], -boxes[6])
        # corner_3d_velos.append(np.array(corner_3d_velo).T)
        # np.savetxt("/mnt/data/WAYMO_det3d/train/annos_global/%d.txt"%frame, boxes)


        # print(*zip(labels, ids))
        # publish_3dbox(box3d_pub, corner_3d_velos, texts=ids, types=ids, track_color=True)
        if SHOW_Track:
            publish_3dbox(box3d_pub, corner_3d_velos, texts=ids, types=ids, track_color=True, Lifetime=1.0/ROS_RATE)
        else:
            publish_3dbox(box3d_pub, corner_3d_velos, texts=labels, types=labels, track_color=False, Lifetime=1.0/ROS_RATE)

        rospy.loginfo("waymo published")
        rate.sleep()

        frame += 1
        if frame == (frame_nums-1):
            frame = 0




    ################################################################################
    ## Method 2
    ################################################################################

    # DATA_PATH = '/mnt/data/WAYMO_det3d/tfrecord_training/segment-16102220208346880_1420_000_1440_000_with_camera_labels.tfrecord'
    # DATA_PATH = '/mnt/data/WAYMO_det3d/tfrecord_training/segment-54293441958058219_2335_200_2355_200_with_camera_labels.tfrecord'
    # DATA_PATH = '/mnt/data/WAYMO_det3d/tfrecord_training/segment-15832924468527961_1564_160_1584_160_with_camera_labels.tfrecord'
    # DATA_PATH = '/mnt/data/WAYMO_det3d/tfrecord_training/segment-33101359476901423_6720_910_6740_910_with_camera_labels.tfrecord'
    # DATA_PATH = '/mnt/data/WAYMO_det3d/tfrecord_training/segment-33101359476901423_6720_910_6740_910_with_camera_labels.tfrecord'
    # DATA_PATH = '/mnt/data/WAYMO_det3d/tfrecord_testing/segment-3328513486129168664_2080_000_2100_000_with_camera_labels.tfrecord'

    # for image, point_cloud in get_one_frame_tfrecord(DATA_PATH):
    #     print(len(point_cloud))
    #     publish_camera(cam_pub, bridge, image)
    #     publish_point_cloud(pcl_pub, point_cloud)
    #     publish_ego_car(ego_pub)



