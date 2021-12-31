#!/usr/bin/python3
import cv2
import glob
import rospy
import numpy as np
from cv_bridge import CvBridge

import utils
from publish_utils import *


ROS_RATE = 10
DET_THRESH = 0.45

DRAW_LIDAR_ON_IMAGE = True
DRAW_2D_BBOX_ON_IMAGE = False
DRAW_3D_BBOX_ON_IMAGE = True


Life_time = 1.0 / ROS_RATE
CV_WAIT_TIME = int(100.0 / ROS_RATE)

sp = cv2.imread("/home/zhanghao/sp.png")

if  __name__ == "__main__":
    # LIDAR_PATH = "/mnt/data/SGData/lidar/"
    # IMAGE_PATH = "/mnt/data/SGData/cam120/"
    # DETECTION_PKL = "/mnt/data/SGData/detection/detections.pkl"

    # LIDAR_PATH = "/mnt/data/SGData/1222/2021-12-22-10-31-13-9/lidar/"
    # IMAGE_PATH = "/mnt/data/SGData/1222/2021-12-22-10-31-13-9/cam120/"
    # DETECTION_PKL = "/mnt/data/SGData/1222/2021-12-22-10-31-13-9/detections.pkl"

    LIDAR_PATH = "/mnt/data/SGData/1222/2021-12-22-10-46-24-11/lidar/"
    # IMAGE_PATH = "/mnt/data/SGData/1222/2021-12-22-10-46-24-11/cam120/"
    IMAGE_PATH = "/mnt/data/SGData/1222/2021-12-22-10-46-24-11/cam60/"
    # IMAGE_PATH = "/mnt/data/SGData/1222/2021-12-22-10-46-24-11/cam30/"
    DETECTION_PKL = "/mnt/data/SGData/1222/2021-12-22-10-46-24-11/detections.pkl"

    # LIDAR_PATH = "/mnt/data/SGData/20211224/2021-12-24-11-40-50-3d/lidar/"
    # IMAGE_PATH = "/mnt/data/SGData/20211224/2021-12-24-11-40-50-3d/cam60/"
    # DETECTION_PKL = "/mnt/data/SGData/1222/2021-12-22-10-46-24-11/detections.pkl"
    
    lidar_files = sorted(glob.glob(LIDAR_PATH + "*"))
    image_files = sorted(glob.glob(IMAGE_PATH + "*"))
    frame_nums = len(lidar_files)
    
    # frame = 12000
    frame = 5700
    bridge = CvBridge()
    rospy.init_node('sg_node',anonymous=True)
    cam_pub = rospy.Publisher('sg_cam60', Image, queue_size=10)
    pcl_pub = rospy.Publisher('sg_point_cloud', PointCloud2, queue_size=10)
    box3d_pub = rospy.Publisher('sg_3dbox',MarkerArray, queue_size=10)
    rate = rospy.Rate(ROS_RATE)

    # #######################################################################################
    # # cam 120 第一版本的参数
    # ex = np.array([[ 0.0663, -0.9978,  0.0067, -0.0469],
    #                [-0.0022, -0.0069, -1.0000, -0.1641],
    #                [ 0.9978,  0.0663, -0.0027, -0.1177],
    #                [      0,       0,       0,       1],
    #                ])

    # ix = np.array([[1.0199e3,  -0.7512, 950.1035, 0],
    #                [       0, 1.0240e3, 558.4226, 0],
    #                [       0,        0,        1, 0]
    #                ])
    # # 畸变系数
    # k1, k2, k3, p1, p2 = -0.3423, 0.1192, -0.0193, -2.5420e-04, 7.8463e-04
    # #######################################################################################
    # #######################################################################################
    # # cam 120 第二版本的参数
    # ex = np.array([[ 0.0390, -0.9992,     -2.7217e-04,  -0.0087],
    #                [-0.0082, -4.6110e-05, -1.0000,      -0.1472],
    #                [ 0.9992,  0.0390,     -0.0082,      -0.1585],
    #                [      0,       0,           0,            1],
    #                ])

    # ix = np.array([[1.0109e3,        0, 965.5663, 0],
    #                [       0, 1.0129e3, 559.3838, 0],
    #                [       0,        0,        1, 0]
    #                ])
    # # 畸变系数
    # k1, k2, k3, p1, p2 = -0.3714, 0.1674, -0.0409, -8.1010e-04, 9.5426e-05
    # #######################################################################################
    # #######################################################################################
    # # cam 60 的参数
    # ex = np.array([[-0.0273, -0.9995,     -0.0133,  -0.0389],
    #                [0.0012,   0.0133,     -0.9999,  -0.1625],
    #                [ 0.9996,  -0.0273,     0.0008,  -0.2082],
    #                [      0,       0,           0,        1],
    #                ])

    # ix = np.array([[1.9641e3,   1.0997, 995.8989, 0],
    #                [       0, 1.9643e3, 557.9326, 0],
    #                [       0,        0,        1, 0]
    #                ])
    # # 畸变系数
    # k1, k2, k3, p1, p2 = -0.5638, 0.3916, -0.2709, 5.9609e-04, -0.0035
    # #######################################################################################
    ######################################################################################
    # # cam 60 的第二组参数
    ex = np.array([[-0.0255, -0.9997,     -0.0053,  -0.0279],
                   [ 0.0057,  0.0051,     -1.0000,  -0.1244],
                   [ 0.9997,  -0.0255,     0.0056,  -0.1017],
                   [      0,       0,           0,        1],
                   ])

    ix = np.array([[1.9768e3,   1.1512, 990.2660, 0],
                   [       0, 1.9770e3, 549.7352, 0],
                   [       0,        0,        1, 0]
                   ])
    # 畸变系数
    k1, k2, k3, p1, p2 = -0.5574, 0.3210, -0.1407, 0.0021, -0.0031
    #######################################################################################
    # #######################################################################################
    # #cam 30 的参数
    # ex = np.array([[0.0479,  -0.9990,     -0.0011, -0.2587],
    #                [0.0548,   0.0037,     -0.9985, -0.1472],
    #                [ 0.9973,  0.0477,      0.0549, -0.1785],
    #                [      0,       0,           0,       1],
    #                ])

    # ix = np.array([[3986.8,   0.0, 936.97, 0],
    #                [       0, 3988.4, 564.35, 0],
    #                [       0,        0,        1, 0]
    #                ])
    # # 畸变系数
    # #k1, k2, k3, p1, p2 = -0.499687, 0.828706, 0.000144, 0.007460, 0.000000
    # k1, k2, k3, p1, p2 = -0.3495, 0.1648, 0, 0, 0.000000
    # #######################################################################################


    dist_coef = np.array([k1, k2, p1, p2, k3])
    lidar_to_image = np.matmul(ix, ex)
    
    # all_detections = utils.read_detections("/mnt/data/SGData/detection/detections.pkl")
    # all_detections = utils.read_detections("/mnt/data/SGData/1222/2021-12-22-10-31-13-9/detections.pkl")
    all_detections = utils.read_detections(DETECTION_PKL)
    while not rospy.is_shutdown():
        lidar_file = lidar_files[frame]
        image_file = image_files[frame]

        # 读取点云发布
        point_cloud = utils.read_pcd(lidar_file)
        publish_point_cloud(pcl_pub, point_cloud)

        # 图像去畸变
        image_ori = utils.get_cv_image(image_file)
        image = image_ori.copy()
        cv2.undistort(image_ori, ix[:, :3], dist_coef, image, None)

        # 将点云映射到 Front-camera 上面  
        if DRAW_LIDAR_ON_IMAGE:
            lidar_points, lidar_attrs = point_cloud[..., :3], np.linalg.norm(point_cloud[..., :3], ord=2, axis=1).reshape(-1,1)  
            utils.display_laser_on_image(image, lidar_points, lidar_to_image, lidar_attrs)

        # 读取检测框
        token = lidar_file.split('/')[-1]
        boxes7d, classes, scores = utils.get_dets_fname(all_detections, token, DET_THRESH)

        # 转换到角点
        corner_3d_velos = []
        for bbox in boxes7d:
            # corner_3d_velo = utils.compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6])
            corner_3d_velo = utils.compute_3d_cornors(bbox)
            corner_3d_velos.append(np.array(corner_3d_velo).T)
            # 绘制映射的 2d-box
            if DRAW_LIDAR_ON_IMAGE and DRAW_2D_BBOX_ON_IMAGE:
                corners = utils.get_3d_box_projected_corners(lidar_to_image, bbox[:7])
                # Compute the 2D bounding box of the label
                if corners is not None:
                    bbox = utils.compute_2d_bounding_box((1080, 1920), corners)
                    bbox = np.array(bbox).reshape(-1, 4)
                    image = utils.draw_bbox(image, bbox)
            
            # 绘制映射的 3d-box
            # if DRAW_LIDAR_ON_IMAGE and DRAW_3D_BBOX_ON_IMAGE:
            if DRAW_3D_BBOX_ON_IMAGE:
                utils.display_3dbox_on_image(image, corner_3d_velo.T, lidar_to_image)
                # break
        

        publish_camera(cam_pub, bridge, image)
        publish_3dbox(box3d_pub, corner_3d_velos, texts=scores, types=classes, track_color=False, Lifetime=Life_time)
        
        rospy.loginfo("waymo published [%d]"%frame)
        # rate.sleep()
        
        cv2.imshow("控制窗口", sp)
        if cv2.waitKey(CV_WAIT_TIME) & 0xFF == ord(' '):
            print("space pressed......")
            cv2.waitKey(0)

        frame += 1
        if frame == (frame_nums - 1):
            # frame = 0
            break
        # a=input()

