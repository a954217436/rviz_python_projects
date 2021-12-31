#!/usr/bin/python3
'''
使用 centerpoint 中的跟踪方法，需要使用带 velo 的 prediction 
'''
import glob
from center_trakcer import Tracker

from data_utils import *
from publish_utils import *
from cp_utils import *
import transform


def save_track_results_pkl(pkl_path, track_data):
    dd = {
        "objects" : [{
            "id" : a['tracking_id'],
            "score" : a['score'],
            "label" : a['label_preds'],
            "state" : 0,
            "points_nums" : 0,
            
            "local_box" : a['local_box'],
            "local_box_filtered" : None,

            "world_box" : None,
            "world_box_filtered" : None,
            "world_box_predict" : None,
            
            "output_velocity" : a['velocity'],
            "mesured_center_velocity" : None,
            "mesured_center_acceleration" : None,

            "bary_center" : None,
            "corners" : None,           
        } for a in track_data]
    }
    with open(pkl_path, "wb") as ff:
        pickle.dump(dd, ff)


ROS_RATE = 100
Life_time = 1.0 / ROS_RATE

SPLIT = "val"    # train val
SEQ_LIST = [6,7,9,10]
DET_THRESH = 0.6
NO_SHOW = True

D_PATH = '/mnt/data/waymo_opensets/'
LIDAR_PATH   = D_PATH + SPLIT + "/lidar/" 
LABEL_PATH   = D_PATH + SPLIT + "/annos/"
# IMAGE_PATH   = D_PATH + SPLIT + "/images_front/"


SAVE_PATH = D_PATH + SPLIT + "/track_results/ct_valseq67910_0.6/"
os.makedirs(SAVE_PATH, exist_ok=True)

INFO_PKL_PATH = D_PATH  + "infos_val_02sweeps_filter_zero_gt.pkl" 
LABEL_MAP  = {0:"Car", 1:"Ped", 2:"Tra", 3:"Cyc"}


if  __name__ == "__main__":
    bridge = CvBridge()
    rospy.init_node('waymo_node',anonymous=True)
    pcl_pub = rospy.Publisher('waymo_point_cloud', PointCloud2, queue_size=10)
    ego_pub = rospy.Publisher('waymo_ego_car',Marker, queue_size=10)
    box3d_pub = rospy.Publisher('waymo_3dbox',MarkerArray, queue_size=10)
    center_pub = rospy.Publisher('waymo_centers', MarkerArray, queue_size=10)

    rate = rospy.Rate(ROS_RATE)

    max_dist = {
        'VEHICLE':    0.8,
        'PEDESTRIAN': 0.4,
        'CYCLIST':    0.6
    }
    

    for SEQ_IDX in SEQ_LIST:
        frame = 0
        tracker = Tracker(max_age=3, max_dist=max_dist, score_thresh=DET_THRESH)

        # 需要先使用  python split_preds_seqs.py 将 dist_test 生成的 pkl 转换成多个
        DET_PKL_PATH = D_PATH + SPLIT + "/preds/voxelnet_2sweep_3x_withvelo/seq_%d.pkl"%SEQ_IDX
        # 读取 detect 结果并转换
        all_dets = get_all_det_pkl(DET_PKL_PATH)
        with open(INFO_PKL_PATH, 'rb') as f:
            infos = pickle.load(f)
            infos = reorganize_info(infos)
        global_preds, _ = convert_detection_to_global_box(all_dets, infos)

        frame_nums = len(glob.glob(LIDAR_PATH+"seq_%d_frame_*"%SEQ_IDX))

        while not rospy.is_shutdown():
            TOKEN = "seq_%d_frame_%d.pkl"%(SEQ_IDX, frame)

            if not NO_SHOW:
                point_cloud = get_lidar_pkl(LIDAR_PATH + TOKEN)
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
            # pose = np.loadtxt(POSE_PATH + "%s.txt"%(frame))
            pose = get_pose_from_anno(LABEL_PATH + TOKEN)
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
            
            # 预测的速度, 转换到 local 下
            velocities = np.array([out['velocity'] for out in outputs])
            # velocities = rotate_vec(velocities, np.arctan2(-pose[1, 0], pose[0, 0]))  # 方式一
            velocities = np.matmul(velocities, pose[..., 0:2, 0:2])  # 方式二
            # print(time_lag, velocities)

            save_track_results_pkl(SAVE_PATH + TOKEN, outputs)

            if not NO_SHOW:
                corner_3d_velos = []
                for boxes in det_boxes_7d:
                    # print(boxes)
                    # corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], -boxes[6], pose)
                    corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], -boxes[6])
                    corner_3d_velos.append(np.array(corner_3d_velo).T)
                publish_3dbox(box3d_pub, corner_3d_velos, texts=tracking_ids, types=tracking_ids, track_color=True, Lifetime=Life_time)

                # ################################################################################
                publish_centers(center_pub, centers_w, radius=1, Lifetime=Life_time, types=tracking_ids, speeds=velocities, track_color=True)
                # publish_ego_car(ego_pub)
                # ################################################################################

            rospy.loginfo("waymo published " + TOKEN)
            rate.sleep()
            
            frame += 1
            if frame == (frame_nums-1):
                break
                # frame = 0



