#!/usr/bin/python3

# from sort_3d import Sort
# from pc3t import Sort

from data_utils import *
from publish_utils import *
import glob

ROS_RATE = 5
THRESH = 0.4

def get_dets_fname_wh(frame_name, prefix="/home/zhanghao/centerpoint_pp_baseline_score0.1_nms0.7_gpuint8/", thr=THRESH):
    print(prefix + frame_name + ".bin.txt")
    box9d = np.loadtxt(prefix + frame_name + ".bin.txt")
    box7ds = box9d[:, [0,1,2,3,4,5,8]]
    classes = box9d[:, -1]
    scores = box9d[:, -2]
    mask = scores > thr
    # print(mask)
    return box7ds[mask], classes[mask], scores[mask]


def get_dets_fname_gt(frame_name, thres=THRESH):
    data = pickle.load(open(frame_name, "rb"))
    box3d_lidar = data['box3d_lidar']    
    scores = data['scores']    
    label_preds = data['label_preds']
    mask = scores>thres

    return np.array(box3d_lidar[mask]), np.array(label_preds[mask]), np.array(scores[mask])


if  __name__ == "__main__":

    SEQ_IDX = 166    # 20:很多人/有卡车

    # LIDAR_PATH = "/home/zhanghao/seq201_to_zh/lidar/seq_%s_frame_"%SEQ_IDX
    LIDAR_PATH = "/mnt/data/waymo_opensets/val/lidar/seq_%s_frame_"%SEQ_IDX

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
        # image = get_image_tfrecord(TF_PATH, frame)
        point_cloud = get_lidar_pkl(LIDAR_PATH + "%s.pkl"%frame)
        # box3d_preds, label_preds, scores = get_dets_fname_wh("seq_%s_frame_%s"%(SEQ_IDX, frame))
        box3d_preds, label_preds, scores = get_dets_fname_gt("/home/zhanghao/code/master/4_TRACK/CenterPointNR/save_dir_tmp/seq_%s_frame_%s.pkl"%(SEQ_IDX, frame))

        publish_point_cloud(pcl_pub, point_cloud)
        publish_ego_car(ego_pub)

        corner_3d_velos = []
        for boxes in box3d_preds:
            # corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6])
            corner_3d_velo = compute_3d_cornors(boxes[0], boxes[1], boxes[2], boxes[3], boxes[4], boxes[5], boxes[6])
            corner_3d_velos.append(np.array(corner_3d_velo).T)

        publish_3dbox(box3d_pub, corner_3d_velos, texts=scores, types=label_preds, track_color=False, Lifetime=1. / ROS_RATE)

        rospy.loginfo("waymo published")
        rate.sleep()

        frame += 1
        if frame == (frame_nums-1):
            frame = 0


