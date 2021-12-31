#!/usr/bin/python3
import glob
import time

# from sort_3d_more import Sort
from mot.mot_model import MOTModel

from data_utils import *
from publish_utils import *
from track_data import TrackData


ROS_RATE = 1
DET_THRESH = 0.5
Life_time = 1.0 / ROS_RATE


def get_pkl_with_barys(pkl_path, thres, timestamp=0.0, pose=None):
    infos = pickle.load(open(pkl_path, "rb"))
    
    scores = infos["scores"] 
    mask = scores > thres  

    box3d_lidar = np.array(infos["box3d_lidar"][mask])  
    label_preds = np.array(infos["label_preds"][mask])
    scores      = np.array(scores[mask])

    barycenter  = np.array(infos["barycenter"])[mask]
    corners     = np.array(infos["corners"])[mask]
    points_nums = np.array(infos["points_nums"])[mask]
    
    if pose is not None:
        world_boxes = veh_to_world_bbox(box3d_lidar, pose)
        barycenter  = veh_to_world_points(barycenter, pose)
        corners     = veh_to_world_points(corners.reshape(-1, 3), pose).reshape(-1, 4, 3)
    else:
        world_boxes = box3d_lidar

    track_datas = []
    for box,label,score,bary,corner,p_num,world_box in \
        zip(box3d_lidar, label_preds, scores, barycenter, corners, points_nums,world_boxes):

        t_data = TrackData( local_box=box,
                            world_box=world_box, 
                            label=label, 
                            score=score, 
                            bary_center=bary, 
                            corners=corner,
                            points_nums=p_num,
                            timestamp=timestamp)
        track_datas.append(t_data)

    return track_datas


def save_track_results_pkl(pkl_path, track_data, pose=None):
    dd = {
        "objects" : [{
            "id" : a.id,
            "score" : a.score,
            "label" : a.label,
            "state" : a.state,
            "points_nums" : a.points_nums,
            
            "local_box" : a.local_box,
            "local_box_filtered" : world_to_veh_bbox(a.world_box_filtered, pose),

            "world_box" : a.world_box,
            "world_box_filtered" : a.world_box_filtered,
            "world_box_predict" : a.world_box_predict,
            
            "output_velocity" : a.output_velocity,
            "mesured_center_velocity" : a.mesured_center_velocity,
            "mesured_center_acceleration" : a.mesured_center_acceleration,

            "bary_center" : a.bary_center,
            "corners" : a.corners,           
        } for a in track_data]
    }
    with open(pkl_path, "wb") as ff:
        pickle.dump(dd, ff)


if  __name__ == "__main__":

    # train ==>>  0:  1:  2:  3:  4:  5:
    # val   ==>>  0:  1:

    SPLIT = "val"    # train val test
    # SEQ_LIST = [0,1,2,3,4,5]
    # SEQ_LIST = [i for i in range(0, 202)]
    # SEQ_LIST = [i for i in range(0, 10)]
    # SEQ_LIST = [6,7,9,10]
    SEQ_LIST = [111]

    PUB_TRACK = True    # True False
    NO_SHOW = False
    USE_GT = False
    TO_FISRT_VIEW = False

    # D_PATH = '/mnt/data/WAYMO_det3d_tiny/'
    D_PATH = '/mnt/data/waymo_opensets/'
    LIDAR_PATH = D_PATH    + SPLIT + "/lidar/" 
    LABEL_PATH = D_PATH    + SPLIT + "/annos/"
    SAVE_PATH = D_PATH     + SPLIT + "/track_results/kfcarot_valseq67910_0.5/"    # save results in this folder
    # DET_PKL_PATH = D_PATH  + SPLIT + "/preds/waymo_centerpoint_pp_two_pfn_stride1_3x/" 
    DET_PKL_PATH = D_PATH  + SPLIT + "/preds/voxelnet_3x_epoch36/" 

    os.makedirs(SAVE_PATH, exist_ok=True)

    # LABEL_MAP  = {0:"Cyc", 1:"Car", 2:"Ped", 3:"Tra"} if USE_GT else {0:"Car", 1:"Ped", 2:"Tra", 3:"Cyc"}
    LABEL_MAP  = {0:"Car", 1:"Ped", 2:"Tra", 3:"Cyc"}

    # all_dets = get_all_det_pkl(DET_PKL_PATH)

    rospy.init_node('waymo_node',anonymous=True)
    pcl_pub = rospy.Publisher('waymo_point_cloud', PointCloud2, queue_size=10)
    ego_pub = rospy.Publisher('waymo_ego_car',Marker, queue_size=10)
    box3d_pub = rospy.Publisher('waymo_3dbox',MarkerArray, queue_size=10)
    track_pub = rospy.Publisher('waymo_track',MarkerArray, queue_size=10)

    rate = rospy.Rate(ROS_RATE)

    
    for SEQ_IDX in SEQ_LIST:
        frame = 0
        frame_nums = len(glob.glob(LIDAR_PATH+"seq_%d_frame_*"%SEQ_IDX))
        print("Start SEQ_IDX : ", SEQ_IDX, ", total : ", frame_nums)
        # vehicle_to_image = get_vehicle_to_image(MATRIX_PATH)
        pose0_vehiche_to_global_inv = np.linalg.inv(get_pose_from_anno(LABEL_PATH + "seq_%d_frame_0.pkl"%(SEQ_IDX)))

        ################################################################################
        # create instance of the MOTModel tracker
        mot_tracker = MOTModel(iou_threshold=0.1) 
        ################################################################################

        while not rospy.is_shutdown():
            TOKEN = "seq_%d_frame_%d.pkl"%(SEQ_IDX, frame)

            if True:  # 转到全局坐标系
                pose_vehiche_to_global = get_pose_from_anno(LABEL_PATH + TOKEN)
                if TO_FISRT_VIEW:
                    pose_vehiche_to_global = np.matmul(pose0_vehiche_to_global_inv, pose_vehiche_to_global)
            else:
                pose_vehiche_to_global = None

            if not NO_SHOW:

                point_cloud = get_lidar_pkl(LIDAR_PATH + TOKEN)
                # point_cloud_world = veh_to_world_points(point_cloud[:,:3], pose_vehiche_to_global)

                ################################################################################
                publish_point_cloud(pcl_pub, point_cloud)
                # publish_point_cloud(pcl_pub, point_cloud_world)
                publish_ego_car(ego_pub)
                ################################################################################

            # box3d_preds, label_preds, scores = get_box3d_pkl(LABEL_PATH + TOKEN)
            # box3d_world = veh_to_world_bbox(box3d_preds, pose_vehiche_to_global)
            # print(box3d_world)
            boxes_my = np.array([[0, 0, 2, 10, 2, 1, 0],
                                [10, 0, 2, 10, 2, 1, np.pi/2.0],
                                [20, 0, 2, 10, 2, 1, -np.pi/2.0],
                                ])
            ################################################################################
            if not NO_SHOW:
                corner_3d_velos = []
                for boxes in boxes_my:
                    corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6])
                    corner_3d_velos.append(np.array(corner_3d_velo).T)

                publish_3dbox(box3d_pub, corner_3d_velos, texts=None, types=None, track_color=False, Lifetime=1.0/ROS_RATE)
            ################################################################################

            rospy.loginfo("waymo published [%s]"%TOKEN)
            # rospy.loginfo("=*"*20)
            rate.sleep()
            
            # frame += 1
            # if frame == (frame_nums):
            #     break

