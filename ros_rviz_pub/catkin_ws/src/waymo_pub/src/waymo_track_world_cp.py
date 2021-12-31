#!/usr/bin/python3
'''
使用 centerpoint 中的跟踪方法，需要使用带 velo 的 prediction 
'''
import glob

from center_trakcer import Tracker
# from pc3t_zh import Sort, STATE_MAP
# from sort_3d_zh import Sort, STATE_MAP

from data_utils import *
from publish_utils import *
# from cp_utils import *
from cp_utils import *
import transform


ROS_RATE = 10

if  __name__ == "__main__":

    # 训练集 ==>>  0:慢速转弯  1:车多夜间街道  2:人多复杂街道  3:车多直路  4:无车弯道  5:慢速直角弯
    # 验证集 ==>>  0:人多复杂街道  1:十字路口

    SPLIT = "val"    # train val test
    SEQ_IDX = 0
    VAL_FILE_LENS =[0, 199, 199+198] 
    DET_THRESH = 0.5
 
    PUB_DET   = True    # True False
    PUB_TRACK = False    # True False
    USE_GT = False
    DRAW_LIDAR_ON_IMAGE = False

    D_PATH = '/mnt/data/WAYMO_det3d_tiny/'
    MATRIX_PATH = '/mnt/data/WAYMO_det3d_tiny/train/matrix/seq_%s'%SEQ_IDX
    LIDAR_PATH = D_PATH    + SPLIT + "/lidar/seq_%s_frame_"%SEQ_IDX     # val test train   
    LABEL_PATH = D_PATH    + SPLIT + "/annos/seq_%s_frame_"%SEQ_IDX     # val test train
    IMAGE_PATH = D_PATH    + SPLIT + "/images_front/seq_%s_frame_"%SEQ_IDX    # val test train
    POSE_PATH = D_PATH     + SPLIT + "/pose/seq_%s_frame_"%SEQ_IDX    # val test train
    DET_PKL_PATH = D_PATH  + SPLIT + "/preds/waymo_centerpoint_voxelnet_2sweep_3x_withvelo.pkl" 

    INFO_PKL_PATH = D_PATH  + "infos_val_02sweeps_filter_zero_gt.pkl" 

    # LABEL_MAP  = {0:"Cyc", 1:"Car", 2:"Ped", 3:"Tra"} if USE_GT else {0:"Car", 1:"Ped", 2:"Tra", 3:"Cyc"}
    LABEL_MAP  = {0:"Car", 1:"Ped", 2:"Tra", 3:"Cyc"}

    # 读取 detect 结果并转换
    all_dets = get_all_det_pkl(DET_PKL_PATH)
    with open(INFO_PKL_PATH, 'rb') as f:
        infos = pickle.load(f)
        infos = reorganize_info(infos)
    global_preds, detection_results = convert_detection_to_global_box(all_dets, infos)
    global_preds = global_preds[VAL_FILE_LENS[SEQ_IDX]:VAL_FILE_LENS[SEQ_IDX+1]]
    print("global_preds: ", len(global_preds))

    frame_nums = len(glob.glob(LIDAR_PATH+"*"))
    print(frame_nums)

    frame = 0
    bridge = CvBridge()
    rospy.init_node('waymo_node',anonymous=True)
    # cam_pub = rospy.Publisher('waymo_cam', Image, queue_size=10)
    pcl_pub = rospy.Publisher('waymo_point_cloud', PointCloud2, queue_size=10)
    ego_pub = rospy.Publisher('waymo_ego_car',Marker, queue_size=10)
    box3d_pub = rospy.Publisher('waymo_3dbox',MarkerArray, queue_size=10)
    # track_pub = rospy.Publisher('waymo_track',MarkerArray, queue_size=10)

    center_pub = rospy.Publisher('waymo_centers', MarkerArray, queue_size=10)

    # track_pub = rospy.Publisher('3dbox_track',MarkerArray,queue_size=10)
    rate = rospy.Rate(ROS_RATE)

    max_dist = {
        'VEHICLE':    0.8,
        'PEDESTRIAN': 0.4,
        'CYCLIST':    0.6
    }
    tracker = Tracker(max_age=3, max_dist=max_dist, score_thresh=0.75)

    while not rospy.is_shutdown():       
        point_cloud = get_lidar_pkl(LIDAR_PATH + "%s.pkl"%frame)
        # publish_camera(cam_pub, bridge, image)
        publish_point_cloud(pcl_pub, point_cloud)

        pred = global_preds[frame]

        # reset tracking after one video sequence
        if pred['frame_id'] == 0 or frame == 0:
            tracker.reset()
            last_time_stamp = pred['timestamp']

        time_lag = (pred['timestamp'] - last_time_stamp) 
        last_time_stamp = pred['timestamp']

        current_det = pred['global_boxs'][:]

        outputs = tracker.step_centertrack(current_det, time_lag)

        centers = np.concatenate([out['translation'][:3] for out in outputs]).reshape(-1, 3)
        tracking_ids = np.array([out['tracking_id'] for out in outputs])
        det_boxes_7d = np.array([out['local_box'] for out in outputs])

        # convert to local
        pose = np.loadtxt(POSE_PATH + "%s.txt"%(frame))
        ##############################################################
        ### 方式一: einsum
        ##############################################################
        center_shift = centers - np.expand_dims(pose[..., 0:3, 3], axis=-2)
        centers_w = np.einsum('...ij,...nj->...ni', pose[..., 0:3, 0:3].T, center_shift)
        
        ##############################################################
        ### 方式二: matmul
        ##############################################################
        # ones = np.ones(shape=(centers.shape[0], 1))
        # centers1 = np.concatenate([centers, ones], -1)
        # centers_w2 = np.matmul(centers1, np.matrix(pose.T).I)
        # ##############################################################   
        
        corner_3d_velos = []
        for boxes in det_boxes_7d:
            # print(boxes)
            # corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], -boxes[6], pose)
            corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], -boxes[6])
            corner_3d_velos.append(np.array(corner_3d_velo).T)
        publish_3dbox(box3d_pub, corner_3d_velos, texts=tracking_ids, types=tracking_ids, track_color=True, Lifetime=1.0/ROS_RATE)

        # ################################################################################
        publish_centers(center_pub, centers_w, radius=1, Lifetime=1.0/ROS_RATE, types=tracking_ids, track_color=True)
        # publish_ego_car(ego_pub)
        # ################################################################################

        rospy.loginfo("waymo published")
        rate.sleep()
        
        frame += 1
        if frame == (frame_nums-1):
            frame = 0



