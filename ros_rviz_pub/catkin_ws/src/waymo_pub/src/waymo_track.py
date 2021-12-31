#!/usr/bin/python3

# from sort_3d import Sort
# from pc3t import Sort
from sort_3d_zh import Sort, STATE_MAP, STATE_CHANGE
# from pc3t_zh import Sort, STATE_MAP

from data_utils import *
from publish_utils import *
import transform
import glob, time

ROS_RATE = 4
# LABEL_MAP  = {0:"Car", 1:"Ped", 2:"Tra", 3:"Cyc"}
Life_time = 1.0/ROS_RATE


if  __name__ == "__main__":

    SEQ_IDX = 0
    DET_THRESH = 0.4
    SPLIT = 'train'

    PUB_DET   = True    # True False
    PUB_TRACK = False    # True False
    USE_GT = True
    DRAW_LIDAR_ON_IMAGE = True
    DRAW_3D_BBOX_ON_IMAGE = True
    DRAW_2D_BBOX_ON_IMAGE = False

    D_PATH = '/mnt/data/WAYMO_det3d_tiny/'
    MATRIX_PATH = '/mnt/data/WAYMO_det3d_tiny/%s/matrix/seq_%s'%(SPLIT, SEQ_IDX)
    LIDAR_PATH = D_PATH   + SPLIT + "/lidar/seq_%s_frame_"%SEQ_IDX     # val test train   
    LABEL_PATH = D_PATH   + SPLIT + "/annos/seq_%s_frame_"%SEQ_IDX     # val test train
    IMAGE_PATH = D_PATH   + SPLIT + "/images_front/seq_%s_frame_"%SEQ_IDX    # val test train
    DET_PKL_PATH = D_PATH + SPLIT + "/preds/waymo_centerpoint_pp_two_pfn_stride1_3x.pkl" 
    #DET_PKL_PATH = D_PATH + SPLIT + "/preds/waymo_centerpoint_pp_two_pfn_stride1_3x.pkl" 

    SAVE_GT_TK_RES = False
    if SAVE_GT_TK_RES:
        os.makedirs("./res/gt", exist_ok=True)
        os.makedirs("./res/tk", exist_ok=True)

    # LABEL_MAP  = {0:"Cyc", 1:"Car", 2:"Ped", 3:"Tra"} if USE_GT else {0:"Car", 1:"Ped", 2:"Tra", 3:"Cyc"}
    LABEL_MAP  = {0:"Car", 1:"Ped", 2:"Tra", 3:"Cyc"}

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
    track_pub = rospy.Publisher('waymo_track',MarkerArray, queue_size=10)

    # track_pub = rospy.Publisher('3dbox_track',MarkerArray,queue_size=10)
    rate = rospy.Rate(ROS_RATE)

    ################################################################################
    # create instance of the SORT tracker
    # 如使用 sort_3d_zh.py，前两个参数不再有效，可配置状态转移 map
    mot_tracker = Sort(max_age=5,
                       min_hits=5,
                       iou_threshold=0.2) 
    ################################################################################
    vehicle_to_image = get_vehicle_to_image(MATRIX_PATH)


    while not rospy.is_shutdown():
        # image = get_image_tfrecord(TF_PATH, frame)
        point_cloud = get_lidar_pkl(LIDAR_PATH + "%s.pkl"%frame)

        # 获取 Front-camera 的图像，将点云映射到 image 上面
        image = get_cv_image(IMAGE_PATH + "%s.jpg"%frame)
        if DRAW_LIDAR_ON_IMAGE:
            lidar_points, lidar_attrs = point_cloud[..., :3], np.linalg.norm(point_cloud[..., :3], ord=2, axis=1).reshape(-1,1)  
            display_laser_on_image(image, lidar_points, vehicle_to_image, lidar_attrs)

        ################################################################################
        publish_point_cloud(pcl_pub, point_cloud)
        publish_ego_car(ego_pub)
        ################################################################################

        if USE_GT:
            # # 读取 groudtruth 当做 pred，测试跟踪性能, xyzlwhr, labels, ids, 注意，gt 中的 labels 编号 和 det不一样
            # box3d_preds, label_preds, scores = get_box3d_pkl(LABEL_PATH + "%s.pkl"%frame, with_tra=False)
            box3d_preds, label_preds, scores = get_box3d_pkl(LABEL_PATH + "%s.pkl"%frame)
        else:
            # # 读取 detect 结果，xyzlwhr, labels, scores
            box3d_preds, label_preds, scores = get_dets_fname(all_dets, "seq_%s_frame_%s.pkl"%(SEQ_IDX, frame), thres=DET_THRESH)

        ################################################################################
        # detect boxes, 白色框是检测框
        if PUB_DET:
            corner_3d_velos = []
            for boxes in box3d_preds:
                corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6])
                corner_3d_velos.append(np.array(corner_3d_velo).T)

                # 绘制映射的 3d-box
                if DRAW_LIDAR_ON_IMAGE and DRAW_3D_BBOX_ON_IMAGE:
                    display_3dbox_on_image(image, corner_3d_velo.T, vehicle_to_image)

                if DRAW_LIDAR_ON_IMAGE and DRAW_2D_BBOX_ON_IMAGE:
                    # 绘制映射的 2d-box
                    corners = transform.get_3d_box_projected_corners(vehicle_to_image, boxes[:7])
                    # print(corners)
                    # Compute the 2D bounding box of the label
                    if corners is not None:
                        bbox = transform.compute_2d_bounding_box((1080, 1920), corners)
                        bbox = np.array(bbox).reshape(-1, 4)
                        image = draw_bbox(image, bbox)
            if PUB_TRACK:
                publish_3dbox(box3d_pub, corner_3d_velos, texts=None, types=None, track_color=False, Lifetime=Life_time)
            else:
                publish_3dbox(box3d_pub, corner_3d_velos, texts=scores, types=label_preds, track_color=False, Lifetime=Life_time)
        ################################################################################


        ################################################################################
        # track boxes 彩色框是追踪框
        if PUB_TRACK:
            # 合并检测结果，送到 mot_tracker 中预测
            preds_9d = np.hstack((box3d_preds, scores.reshape(-1,1), label_preds.reshape(-1,1)))
            tracks = mot_tracker.update(preds_9d)

            corner_3d_velos = []
            for boxes in tracks:
                corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6])
                corner_3d_velos.append(np.array(corner_3d_velo).T)

                if DRAW_LIDAR_ON_IMAGE:
                    # 绘制映射的 2d-box
                    corners = transform.get_3d_box_projected_corners(vehicle_to_image, boxes[:7])
                    # Compute the 2D bounding box of the label
                    if corners is not None:
                        bbox = transform.compute_2d_bounding_box((1080, 1920), corners)
                        bbox = np.array(bbox).reshape(-1, 4)
                        image = draw_bbox(image, bbox)

            label_texts = ["%s-%s-%d"%(LABEL_MAP.get(int(x), "Unknown"), STATE_MAP[y], z) for x,y,z in tracks[:,-3:]]
            # label_texts=None
            publish_3dbox(track_pub, corner_3d_velos, texts=label_texts, types=tracks[:,-1], speeds=tracks[:,7:9], track_color=True, Lifetime=1.0/ROS_RATE)
        
        publish_camera(cam_pub, bridge, image)  # image 后发布，因为也可能画 bbox
        ################################################################################

        ################################################################################
        if SAVE_GT_TK_RES:
            box3d_gt, label_gt, ids_gt = get_box3d_pkl(LABEL_PATH + "%s.pkl"%frame, with_tra=False)
            label_gt = label_gt.reshape(-1,1)
            ids_gt = ids_gt.reshape(-1,1)
            ones = np.ones_like(ids_gt)
            gt_towrite = np.hstack((ids_gt, box3d_gt, label_gt, ones))
            np.savetxt("./res/gt/%s.txt"%frame, gt_towrite)

            tracks_towrite = tracks[:, [10,0,1,2,3,4,5,6,8,7,9]][:,:10]
            np.savetxt("./res/tk/%s.txt"%frame, tracks_towrite)
        ################################################################################

        rospy.loginfo("waymo published [%d]"%frame)
        rate.sleep()

        frame += 1
        if frame == (frame_nums-1):
            frame = 0
            # break

        # if frame == 2:
        #     break


    if SAVE_GT_TK_RES:
        import sys
        sys.path.append("/home/zhanghao/code/master/4_TRACK/TrackEval_MOTA3D")
        import eval3D
        
        args_str = "".join(map(str, list(STATE_CHANGE.values())))
        e = eval3D.trackingEvaluation(gt_path     = "./res/gt/",
                                    tk_path     = "./res/tk/",
                                    save_path   = "./res/lc_eval_seq%s_%s_%s_%s/"%(SEQ_IDX, DET_THRESH, args_str, SPLIT),
                                    min_overlap = 0.2)
        e.do_evaluate()
