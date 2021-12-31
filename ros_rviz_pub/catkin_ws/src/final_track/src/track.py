#!/usr/bin/python3
import glob
import time

from mot.mot_model import MOTModel
from mot.mot_model_two_stage import MOTModelTwoStage

from mot.utils.data_utils import *
from mot.utils.publish_utils import *
from mot.track_data import TrackData

ROS_RATE = 10
NO_SHOW = False      # True False

SPLIT = "val"    # train val
# SEQ_LIST = [0,1,2,3,4,5]
SEQ_LIST = [i for i in range(40, 202)]
# SEQ_LIST = [i for i in range(0, 10)]
# SEQ_LIST = [6,7,9,10]
# SEQ_LIST = [6,7,9,10,119]
# SEQ_LIST = [119]    # 119 是高速，对向车道使用 iou 会跟踪失败
# SEQ_LIST = [120]    #

# USE_GT = False
# PUB_TRACK = True    
# TO_FISRT_VIEW = False
DET_THRESH = 0.1

if NO_SHOW:
    ROS_RATE = 1000
Life_time = 1.0 / ROS_RATE


def get_pkl_dets(pkl_path, thres, timestamp=0.0, pose=None):
    infos = pickle.load(open(pkl_path, "rb"))
    
    scores = infos["scores"] 
    mask = scores > thres  

    box3d_lidar = np.array(infos["box3d_lidar"][mask])  
    label_preds = np.array(infos["label_preds"][mask])
    scores      = np.array(scores[mask])

    if pose is not None:
        world_boxes = veh_to_world_bbox(box3d_lidar, pose)
    else:
        world_boxes = box3d_lidar

    track_datas = []
    for box,label,score,world_box in \
        zip(box3d_lidar, label_preds, scores, world_boxes):

        t_data = TrackData( local_box=box,
                            world_box=world_box, 
                            label=label, 
                            score=score, 
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
            
            "local_box" : a.local_box,
            "local_box_filtered" : world_to_veh_bbox(a.world_box_filtered, pose),

            "world_box" : a.world_box,
            "world_box_filtered" : a.world_box_filtered,
            
            "output_velocity" : a.output_velocity,
            "mesured_center_velocity" : a.mesured_center_velocity,
            "mesured_center_acceleration" : a.mesured_center_acceleration,

        } for a in track_data]
    }
    with open(pkl_path, "wb") as ff:
        pickle.dump(dd, ff)


if  __name__ == "__main__":

    D_PATH = '/mnt/data/waymo_opensets/'
    LIDAR_PATH = D_PATH    + SPLIT + "/lidar/" 
    LABEL_PATH = D_PATH    + SPLIT + "/annos/"
    DET_PKL_PATH = D_PATH  + SPLIT + "/preds/voxelnet_3x_epoch36/" 

    USE_GIOU = False
    IOU_THRESH = -0.5 if USE_GIOU else 0.1
    SAVE_PATH = D_PATH     + SPLIT + "/track_results/kfca_67910119_12321_2s3060_iou/"    # save results in this folder
    os.makedirs(SAVE_PATH, exist_ok=True)

    # # LABEL_MAP  = {0:"Cyc", 1:"Car", 2:"Ped", 3:"Tra"} if USE_GT else {0:"Car", 1:"Ped", 2:"Tra", 3:"Cyc"}
    # LABEL_MAP  = {0:"Car", 1:"Ped", 2:"Tra", 3:"Cyc"}

    rospy.init_node('waymo_node',anonymous=True)
    pcl_pub = rospy.Publisher('waymo_point_cloud', PointCloud2, queue_size=10)
    ego_pub = rospy.Publisher('waymo_ego_car',Marker, queue_size=10)
    track_pub = rospy.Publisher('waymo_track',MarkerArray, queue_size=10)
    rate = rospy.Rate(ROS_RATE)

    
    for SEQ_IDX in SEQ_LIST:
        frame = 0
        frame_nums = len(glob.glob(LIDAR_PATH+"seq_%d_frame_*"%SEQ_IDX))
        print("Start SEQ_IDX : ", SEQ_IDX, ", total : ", frame_nums)

        frame_nums = 40
        # vehicle_to_image = get_vehicle_to_image(MATRIX_PATH)
        pose0_vehiche_to_global_inv = np.linalg.inv(get_pose_from_anno(LABEL_PATH + "seq_%d_frame_0.pkl"%(SEQ_IDX)))

        ################################################################################
        # create instance of the MOTModel tracker
        # mot_tracker = MOTModel(iou_threshold=0.1, det_threshold=0.5)       # iou
        # mot_tracker = MOTModel(iou_threshold=IOU_THRESH, use_giou=USE_GIOU)    # giou 
        mot_tracker = MOTModelTwoStage(iou_threshold=IOU_THRESH, high_thres=0.6, low_thres=0.3, use_giou=USE_GIOU) 
        ################################################################################

        while not rospy.is_shutdown():
            TOKEN = "seq_%d_frame_%d.pkl"%(SEQ_IDX, frame)

            pose_vehiche_to_global = get_pose_from_anno(LABEL_PATH + TOKEN)
            # if TO_FISRT_VIEW:
            #     pose_vehiche_to_global = np.matmul(pose0_vehiche_to_global_inv, pose_vehiche_to_global)

            data_infos = get_pkl_dets(DET_PKL_PATH + TOKEN, DET_THRESH, timestamp=0.1*frame, pose=pose_vehiche_to_global)
            track_results = mot_tracker.update(data_infos)
            save_track_results_pkl(SAVE_PATH + TOKEN, track_results, pose=pose_vehiche_to_global)
            
            ################################################################################
            # track boxes 彩色框是追踪框
            if not NO_SHOW:
                corner_3d_velos = []
                for track_res in track_results:
                    # boxes = track_res.local_box

                    # if (track_res.world_box_filtered is not None):
                    #     world_boxes = track_res.world_box_filtered
                    # else:
                    #     world_boxes = track_res.world_box
                    boxes = world_to_veh_bbox(track_res.world_box_filtered, pose_vehiche_to_global)[0]

                    corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6])
                    corner_3d_velos.append(np.array(corner_3d_velo).T)

                
                if len(track_results) > 0:
                    # world_velos = np.array([a.mesured_center_velocity for a in track_results])
                    world_velos = np.array([a.output_velocity for a in track_results])
                    # speeds = rotate_vec(track_results[:,7:9], np.arctan2(-pose_vehiche_to_global[1, 0], pose_vehiche_to_global[0, 0]))
                    speeds = rotate_vec(world_velos, np.arctan2(-pose_vehiche_to_global[1, 0], pose_vehiche_to_global[0, 0]))
                else:
                    speeds=None
                label_texts = [x.plot_string for x in track_results]
                types = [x.id for x in track_results]
                publish_3dbox(track_pub, corner_3d_velos, texts=label_texts, types=types, speeds=speeds, track_color=True, Lifetime=Life_time)
            

                ################################################################################
                point_cloud = get_lidar_pkl(LIDAR_PATH + TOKEN)
                publish_point_cloud(pcl_pub, point_cloud)
                publish_ego_car(ego_pub)
                ################################################################################
            ################################################################################

            rospy.loginfo("waymo published [%s]"%TOKEN)
            rate.sleep()
            
            frame += 1
            if frame == (frame_nums):
                break

