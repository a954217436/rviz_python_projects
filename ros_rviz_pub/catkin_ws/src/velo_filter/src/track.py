#!/usr/bin/python3
import glob
import time

from mot.mot_model import MOTModel
from mot.mot_model_two_stage import MOTModelTwoStage

from data_utils import *
from publish_utils import *
from track_data import TrackData

ROS_RATE = 1
NO_SHOW = False      # True False

SPLIT = "val"    # train val
# SEQ_LIST = [0,1,2,3,4,5]
# SEQ_LIST = [i for i in range(0, 202)]
# SEQ_LIST = [i for i in range(0, 10)]
# SEQ_LIST = [6,7,9,10]
SEQ_LIST = [119]    # 119 是高速，对向车道使用 iou 会跟踪失败

USE_GT = False
PUB_TRACK = True    
TO_FISRT_VIEW = False
DET_THRESH = 0.1

if NO_SHOW:
    ROS_RATE = 1000
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
    # return {
    #     "local_boxes" : box3d_lidar,
    #     "label_preds"       : label_preds,
    #     "scores"            :      scores,
    #     "barycenters"       :  barycenter,
    #     "corners"           :     corners,
    #     "points_nums"       : points_nums,
    # }
    # return np.array(box3d_lidar), np.array(label_preds), np.array(scores), barycenter, corners, points_nums


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

    # D_PATH = '/mnt/data/WAYMO_det3d_tiny/'
    D_PATH = '/mnt/data/waymo_opensets/'
    LIDAR_PATH = D_PATH    + SPLIT + "/lidar/" 
    LABEL_PATH = D_PATH    + SPLIT + "/annos/"
    SAVE_PATH = D_PATH     + SPLIT + "/track_results/kfca_lm/"    # save results in this folder
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
        mot_tracker = MOTModel(iou_threshold=0.1, det_threshold=0.5)       # iou
        # mot_tracker = MOTModel(iou_threshold=-0.5)    # giou 
        # mot_tracker = MOTModelTwoStage(iou_threshold=-0.5, high_thres=0.6, low_thres=0.35)    # giou 
        ################################################################################

        while not rospy.is_shutdown():
            TOKEN = "seq_%d_frame_%d.pkl"%(SEQ_IDX, frame)

            # time2 = time.time()
            # print("time1-2 using : ", time2 - time1)

            if True:  # 转到全局坐标系
                pose_vehiche_to_global = get_pose_from_anno(LABEL_PATH + TOKEN)
                if TO_FISRT_VIEW:
                    pose_vehiche_to_global = np.matmul(pose0_vehiche_to_global_inv, pose_vehiche_to_global)
            else:
                pose_vehiche_to_global = None

            if USE_GT:
                # # 读取 groudtruth 当做 pred，测试跟踪性能, xyzlwhr, labels, ids, 注意，gt 中的 labels 编号 和 det不一样
                box3d_preds, label_preds, scores = get_box3d_pkl(LABEL_PATH + TOKEN)
            else:
                # # # 读取 detect 结果，xyzlwhr, labels, scores
                # box3d_preds, label_preds, scores = get_dets_fname(all_dets, "seq_%s_frame_%s.pkl"%(SEQ_IDX, frame), thres=DET_THRESH)
                data_infos = get_pkl_with_barys(DET_PKL_PATH + TOKEN, DET_THRESH, timestamp=0.1*frame, pose=pose_vehiche_to_global)

            # time3 = time.time()
            # print("time2-3 using : ", time3 - time2)

            track_results = mot_tracker.update(data_infos)
            save_track_results_pkl(SAVE_PATH + TOKEN, track_results, pose=pose_vehiche_to_global)
            
            # # time4 = time.time()
            # # print("time3-4 using : ", time4 - time3)

            ################################################################################
            # track boxes 彩色框是追踪框
            if PUB_TRACK and (not NO_SHOW):
                corner_3d_velos = []
                for track_res in track_results:
                    boxes = track_res.local_box
                    # boxes = world_to_veh_bbox(track_res.world_box_filtered, pose_vehiche_to_global)[0] \
                    #     if (track_res.world_box_filtered is not None) else track_res.local_box
                    # corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6], pose_vehiche_to_global)
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
            
            if not NO_SHOW:
                time0 = time.time()
                # image = get_image_tfrecord(TF_PATH, frame)
                point_cloud = get_lidar_pkl(LIDAR_PATH + TOKEN)

                # time1 = time.time()
                # print("time0-1 using : ", time1 - time0)

                ################################################################################
                publish_point_cloud(pcl_pub, point_cloud)
                publish_ego_car(ego_pub)
                ################################################################################

            ################################################################################
            # # time5 = time.time()
            # # print("time4-5 using : ", time5 - time4)

            rospy.loginfo("waymo published [%s]"%TOKEN)
            # rospy.loginfo("=*"*20)
            rate.sleep()
            
            frame += 1
            if frame == (frame_nums):
                break

