#!/usr/bin/python3

# from sort_3d import Sort
# from pc3t import Sort

from data_utils import *
from publish_utils import *
import glob


def load_txt_pc_txt(txt_path):
    return np.loadtxt(txt_path)[:, :3]


def load_txt_pc(txt_path):
    return np.fromfile(txt_path, dtype=np.float16).reshape(-1, 4)[:, :3]


def get_registration_angle(mat):
    cos_theta = mat[0, 0]
    sin_theta = mat[1, 0]

    if cos_theta < -1:
        cos_theta = -1
    if cos_theta > 1:
        cos_theta = 1

    theta_cos = np.arccos(cos_theta)

    if sin_theta >= 0:
        return theta_cos
    else:
        return 2 * np.pi - theta_cos


def register_bbs(boxes, pose):
    ang = get_registration_angle(pose)
    # t_id = boxes.shape[1] // 7
    ones = np.ones(shape=(boxes.shape[0], 1))

    b_id = 0
    box_xyz = boxes[:, b_id:b_id + 3]
    box_xyz1 = np.concatenate([box_xyz, ones], -1)
    box_world = np.matmul(box_xyz1, pose)
    # print('mean shift , ',((box_world[:,:3]-box_xyz1[:,:3])**2).mean())
    # box_world = box_xyz1

    boxes[:, b_id:b_id + 3] = box_world[:, 0:3]
    boxes[:, b_id + 6] += ang
    return boxes



if  __name__ == "__main__":
    D_PATH = '/mnt/data/WAYMO_det3d/train/lidar_global/'

    frame_nums = len(glob.glob(D_PATH+"seq_1_frame_*"))
    print(frame_nums)

    frame = 0
    pcl_pub = rospy.Publisher('waymo_point_cloud', PointCloud2, queue_size=10)
    box3d_pub = rospy.Publisher('waymo_3dbox',MarkerArray, queue_size=10)

    rospy.init_node('waymo_node',anonymous=True)
    rate = rospy.Rate(10)

    ################################################################################

    while not rospy.is_shutdown():
        point_cloud = load_txt_pc(D_PATH + "seq_1_frame_%s.npy"%frame)
        publish_point_cloud(pcl_pub, point_cloud)

        box3d = np.loadtxt("/mnt/data/WAYMO_det3d/train/annos_global/%d.txt"%frame).reshape(1,-1)
        pose_npy = np.loadtxt("/mnt/data/WAYMO_det3d/train/pose/seq_1_frame_%d.txt"%(frame))

        g_box = register_bbs(box3d, pose_npy)[0]
        print(g_box)
        corner_3d_velos = []
        corner_3d_velo = compute_3d_cornors(g_box[0], g_box[1], g_box[2], g_box[3], g_box[4], g_box[5], g_box[6])
        corner_3d_velos.append(np.array(corner_3d_velo).T)
        publish_3dbox(box3d_pub, corner_3d_velos, track_color=False)
        # print(corner_3d_velos)

        rospy.loginfo("waymo published")
        rate.sleep()

        frame += 1
        if frame == (frame_nums-1):
            frame = 0

