#!/usr/bin/python3

# from sort_3d import Sort
# from pc3t import Sort

from data_utils import *
from publish_utils import *
import glob

ROS_RATE = 10

if  __name__ == "__main__":

    SEQ_IDX = 250
    DET_THRESH = 0.3

    LIDAR_PATH = '/mnt/data/sg_lidar2/bin3/'
    DET_PKL_PATH = "/mnt/data/sg_lidar2/sg_detections.pkl" 

    all_dets = get_all_det_pkl(DET_PKL_PATH)

    frame_nums = len(glob.glob(LIDAR_PATH+"*"))
    print(frame_nums)

    frame = 0
    bridge = CvBridge()
    rospy.init_node('waymo_node',anonymous=True)
    pcl_pub = rospy.Publisher('waymo_point_cloud', PointCloud2, queue_size=10)
    ego_pub = rospy.Publisher('waymo_ego_car',Marker, queue_size=10)
    box3d_pub = rospy.Publisher('waymo_3dbox',MarkerArray, queue_size=10)
    
    rate = rospy.Rate(ROS_RATE)

    ################################################################################

    while not rospy.is_shutdown():
        pc_name = "seq_%s_frame_%d.bin"%(SEQ_IDX, frame)
        # point_cloud = np.loadtxt(LIDAR_PATH + pc_name)
        point_cloud = np.fromfile(LIDAR_PATH + pc_name, dtype=np.float64).reshape(-1,3)

        publish_point_cloud(pcl_pub, point_cloud)
        publish_ego_car(ego_pub)

        box3d_preds, label_preds, scores = get_dets_fname(all_dets, pc_name, thres=DET_THRESH, keys=['boxes', 'scores', 'classes'])
        corner_3d_velos = []
        for boxes in box3d_preds:
            # print(boxes)
            corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6])
            corner_3d_velos.append(np.array(corner_3d_velo).T)

        publish_3dbox(box3d_pub, corner_3d_velos, texts=scores, types=label_preds, track_color=False, Lifetime=1. / ROS_RATE)

        rospy.loginfo("waymo published")
        rate.sleep()

        frame += 1
        if frame == (frame_nums-1):
            frame = 0


