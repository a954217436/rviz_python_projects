#!/usr/bin/python3

# from sort_3d import Sort
# from pc3t import Sort

from data_utils import *
from publish_utils import *
import glob

ROS_RATE = 1

if  __name__ == "__main__":

    SEQ_IDX = 119
    DET_THRESH = 0.3
    SPLIT = "val"

    # D_PATH = '/mnt/data/WAYMO_det3d_tiny/'
    D_PATH = '/mnt/data/waymo_opensets/'
    LIDAR_PATH = D_PATH    + SPLIT + "/lidar/seq_%s_frame_"%SEQ_IDX     # val test train   
    LABEL_PATH = D_PATH    + SPLIT + "/annos/seq_%s_frame_"%SEQ_IDX     # val test train
    IMAGE_PATH = D_PATH    + SPLIT + "/images/seq_%s_frame_"%SEQ_IDX    # val test train
    # DET_PKL_PATH = D_PATH  + SPLIT + "/preds/waymo_centerpoint_pp_two_pfn_stride1_two_stage_bev_6epoch.pkl" 
    # DET_PKL_PATH = D_PATH  + SPLIT + "/preds/waymo_centerpoint_voxelnet_2sweep_3x_withvelo.pkl" 
    DET_PKL_PATH = D_PATH +"val/preds/voxelnet_3x_epoch36.pkl" 


    all_dets = get_all_det_pkl(DET_PKL_PATH)

    frame_nums = len(glob.glob(LIDAR_PATH+"*"))
    print(frame_nums)

    frame = 0
    bridge = CvBridge()
    rospy.init_node('waymo_node',anonymous=True)
    cam_pub = rospy.Publisher('waymo_cam', Image, queue_size=10)
    pcl_pub = rospy.Publisher('waymo_point_cloud', PointCloud2, queue_size=10)
    ego_pub = rospy.Publisher('waymo_ego_car',Marker, queue_size=10)
    box3d_pub = rospy.Publisher('waymo_3dbox',MarkerArray, queue_size=10)
    
    # center_pub = rospy.Publisher('waymo_centers', MarkerArray, queue_size=10)

    # track_pub = rospy.Publisher('3dbox_track',MarkerArray,queue_size=10)
    rate = rospy.Rate(ROS_RATE)

    ################################################################################

    while not rospy.is_shutdown():
        # image = get_image_tfrecord(TF_PATH, frame)
        point_cloud = get_lidar_pkl(LIDAR_PATH + "%s.pkl"%frame)
        box3d_preds, label_preds, scores = get_dets_fname(all_dets, "seq_%s_frame_%s.pkl"%(SEQ_IDX, frame), thres=DET_THRESH)
        # print(box3d_preds.shape, label_preds.shape, scores.shape)

        # image = get_cv_image(IMAGE_PATH + "%s.jpg"%frame)
        # publish_camera(cam_pub, bridge, image)
        publish_point_cloud(pcl_pub, point_cloud)
        publish_ego_car(ego_pub)

        corner_3d_velos = []
        for boxes in box3d_preds:
            # corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6])
            corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6])
            corner_3d_velos.append(np.array(corner_3d_velo).T)

        publish_3dbox(box3d_pub, corner_3d_velos, texts=scores, types=label_preds, track_color=False, Lifetime=1. / ROS_RATE)

        # centers = box3d_preds[:, :3]
        # centers = np.array([[2,2,1], [3,3,2], [4,4,3]])  #
        # publish_centers(center_pub, centers, Lifetime=1000 / ROS_RATE)

        rospy.loginfo("waymo published")
        rate.sleep()

        frame += 1
        if frame == (frame_nums-1):
            frame = 0


