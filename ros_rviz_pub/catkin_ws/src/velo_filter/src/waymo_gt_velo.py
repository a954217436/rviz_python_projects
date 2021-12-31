#!/usr/bin/python3
import glob
from data_utils import *
from publish_utils import *

'''
Play WAYMO-dataset Seqs(det3d format) Groundtruth with ID & LABEL & BBOX & VELO ......
'''


ROS_RATE = 10
Life_time = 1.0 / ROS_RATE


if  __name__ == "__main__":
    # SEQ_LIST = [0,1,2,3,4,5]
    # SEQ_LIST = [a for a in range(111,210)]
    # SEQ_LIST = [119]
    # SEQ_LIST = [6,7,9,10,119]
    SEQ_LIST = [i for i in range(40, 202)]

    SPLIT = "val"
    DRAW_LIDAR_ON_IMAGE = False

    D_PATH = '/mnt/data/waymo_opensets/'
    # D_PATH = '/mnt/data/WAYMO_det3d_tiny/'
    LIDAR_PATH = D_PATH    + SPLIT + "/lidar/" 
    LABEL_PATH = D_PATH    + SPLIT + "/annos/"
    POSE_PATH = D_PATH     + SPLIT + "/pose/"

    LABEL_MAP  = {0:"Car", 1:"Ped", 2:"Tra", 3:"Cyc"}


    bridge = CvBridge()
    rospy.init_node('waymo_gt_node',anonymous=True)
    pcl_pub = rospy.Publisher('waymo_gt_point_cloud', PointCloud2, queue_size=10)
    ego_pub = rospy.Publisher('waymo_gt_ego_car',Marker, queue_size=10)
    track_pub = rospy.Publisher('waymo_gt_track',MarkerArray, queue_size=10)
    # box3d_pub = rospy.Publisher('waymo_3dbox',MarkerArray, queue_size=10)

    rate = rospy.Rate(ROS_RATE)

    
    for SEQ_IDX in SEQ_LIST:
        frame = 0
        frame_nums = len(glob.glob(LIDAR_PATH + "seq_%d_*"%SEQ_IDX))
        print(frame_nums)
        frame_nums = 40
        # pose0_vehiche_to_global_inv = np.linalg.inv(get_pose_from_anno(LABEL_PATH + "seq_%d_frame_0.pkl"%(SEQ_IDX)))

        while not rospy.is_shutdown():
            TOKEN = "seq_%d_frame_%d.pkl"%(SEQ_IDX, frame)
            point_cloud = get_lidar_pkl(LIDAR_PATH + TOKEN)

            ################################################################################
            publish_point_cloud(pcl_pub, point_cloud)
            publish_ego_car(ego_pub)
            ################################################################################

            if True:  # 全局坐标系
                pose_vehiche_to_global = get_pose_from_anno(LABEL_PATH + TOKEN)
                # if TO_FISRT_VIEW:
                #     pose_vehiche_to_global = np.matmul(pose0_vehiche_to_global_inv, pose_vehiche_to_global)
            else:
                pose_vehiche_to_global = None

            boxes7d, labels, ids, global_speeds = get_box3d_pkl(LABEL_PATH + TOKEN, with_velo=True)
            if boxes7d is None:
                continue
            ################################################################################
            corner_3d_velos = []
            for boxes in boxes7d:
                corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6])
                corner_3d_velos.append(np.array(corner_3d_velo).T)

            if len(global_speeds) > 0:
                local_speeds = rotate_vec(global_speeds, np.arctan2(-pose_vehiche_to_global[1, 0], pose_vehiche_to_global[0, 0]))
            else:
                local_speeds = None
            publish_3dbox(track_pub, corner_3d_velos, texts=ids, types=ids, speeds=local_speeds, track_color=True, Lifetime=Life_time)

            rospy.loginfo("waymo published [%s]"%TOKEN)
            # rospy.loginfo("=*"*20)
            rate.sleep()
            
            frame += 1
            if frame == (frame_nums):
                break

