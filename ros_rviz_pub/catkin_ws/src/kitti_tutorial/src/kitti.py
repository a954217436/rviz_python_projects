#!/usr/bin/python3

# from sort_3d import Sort
from pc3t import Sort
from data_utils import *
from publish_utils import *



# DATA_PATH = '/home/wanghao/Desktop/slam_ws/2011_09_30_sync/2011_09_30_sync_drive_0027_sync/'
# PRED_PATH = "/home/wanghao/Desktop/projects/OpenPCDet/tools/demo_pointrcnn_iou_results_track"

LABEL_MAP  = {1:"Car",2:"Pedistrian",3:"Cyclist"}
PRED_PATH = '/home/zhanghao/KITTI/kitti_raw/2011_09_26/2011_09_26_drive_0096_sync/preds_pv'
DATA_PATH = '/home/zhanghao/KITTI/kitti_raw/2011_09_26/2011_09_26_drive_0096_sync/'


# PRED_PATH = '/home/zhanghao/KITTI/kitti_raw/2011_09_29/2011_09_29_drive_0004_sync/preds_pv'
# DATA_PATH = '/home/zhanghao/KITTI/kitti_raw/2011_09_29/2011_09_29_drive_0004_sync/'

if  __name__ == "__main__":
    frame = 0
    bridge = CvBridge()
    rospy.init_node('kitti_node',anonymous=True)
    cam_pub = rospy.Publisher('kitti_cam', Image, queue_size=10)
    pcl_pub = rospy.Publisher('kitti_point_cloud', PointCloud2, queue_size=10)
    ego_pub = rospy.Publisher('kitti_ego_car',Marker, queue_size=10)
#     model_car_pub = rospy.Publisher('kitti_model_car',Marker, queue_size=10)
#     imu_pub = rospy.Publisher('kitti_imu',Imu, queue_size=10)
#     gps_pub = rospy.Publisher('kitti_gps',NavSatFix, queue_size=10)

    box3d_pub = rospy.Publisher('kitti_3dbox',MarkerArray, queue_size=10)
    track_pub = rospy.Publisher('3dbox_track',MarkerArray,queue_size=10)
    rate = rospy.Rate(10)

    ################################################################################
    mot_tracker = Sort(max_age=3, 
                   min_hits=3,
                   iou_threshold=0.3) #create instance of the SORT tracker

    while not rospy.is_shutdown():
        # cv2.imread,np.array
        image = read_camera(os.path.join(DATA_PATH, 'image_02/data/%010d.png'%frame))
        # N x 4, np.array
        point_cloud = read_point_cloud(os.path.join(DATA_PATH, 'velodyne_points/data/%010d.bin'%frame))
        # mask = np.where(point_cloud[:,0]>=0)[0]
        # TODO : select pcls in front of the vehicle
        # point_cloud = point_cloud[point_cloud[:,0]>=0]
        # imu_data = read_imu(os.path.join(DATA_PATH,'oxts/data/%010d.txt'%frame))
        publish_camera(cam_pub, bridge, image)
        publish_point_cloud(pcl_pub, point_cloud)
        publish_ego_car(ego_pub)
        # publish_car_model(model_car_pub)
        # publish_imu(imu_pub, imu_data )
        # publish_gps(gps_pub, imu_data ) #gps rviz cannot visulize, only use rostopic echo

        if 0:
            preds_9d = read_pred(os.path.join(PRED_PATH, '%010d.txt'%frame))    # x,y,z,dx,dy,dz,r,score,class
            preds_9d = preds_9d[preds_9d[:,0]**2+preds_9d[:,1]**2>3]
            preds_9d = preds_9d[preds_9d[:, -2] > 0.4]
            # tracks = mot_tracker.update(preds_9d)
            # tracks = read_pred(os.path.join(PRED_PATH, '%010d.txt'%frame))    # x,y,z,dx,dy,dz,r,score,class

            corner_3d_velos = []
            for box_9d in preds_9d:
                # 3 x 8, 8 corners
                if box_9d[0]**2 + box_9d[1]**2<4:
                    continue
                corner_3d_velo = compute_3d_box_pred2(box_9d[0], box_9d[1], box_9d[2], box_9d[3], box_9d[4], box_9d[5], -box_9d[6])
                corner_3d_velos.append(np.array(corner_3d_velo).T),
                #  labels = [LABEL_MAP.get(int(x),"Unknown") for x in tracks[:,-2]]
            # publish_3dbox(box3d_pub, corner_3d_velos, texts = preds_9d[:,0]**2 + preds_9d[:,1]**2 ) #preds_9d[:,7])
            publish_3dbox(box3d_pub, corner_3d_velos, texts = preds_9d[:,-1], types=preds_9d[:,-1], track_color=False)

        # corner_3d_velos = []
        # for box in tracks:
        #     # 3 x 8, 8 corners
        #    corner_3d_velo = compute_3d_box_pred2(box[0], box[1], box[2], box[3], box[4], box[5], -box[6])
        #    corner_3d_velos.append(np.array(corner_3d_velo).T),
        # # print(corner_3d_velos[0].shape)
        # labels = [LABEL_MAP.get(int(x),"Unknown") for x in tracks[:,-2]]
        # publish_3dbox(track_pub, corner_3d_velos, texts = labels,types = tracks[:,-1],track_color=True)

        rospy.loginfo("kitti published")
        rate.sleep()
        frame += 1
        frame %= 108







