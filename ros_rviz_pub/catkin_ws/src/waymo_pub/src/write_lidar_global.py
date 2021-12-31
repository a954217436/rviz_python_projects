import glob
import tqdm
import pathlib
from data_utils import *

D_PATH = '/mnt/data/WAYMO_det3d/'
ANNOS_LABEL = D_PATH + 'train/annos/seq_1_frame_0.pkl'
TF_PATH = D_PATH + 'tfrecord_training/segment-%s_with_camera_labels.tfrecord'
TF_this_seq = TF_PATH % (get_pkl_info(ANNOS_LABEL))  # 找到对应当前 frame 序列的 tfrecord

GLOBAL_SAVE_PATH = "/mnt/data/WAYMO_det3d/train/lidar_global/"
os.makedirs(GLOBAL_SAVE_PATH, exist_ok=True)

# dataset = tf.data.TFRecordDataset(TF_this_seq, compression_type='')
# for data in dataset:
#     frame = open_dataset.Frame()
#     frame.ParseFromString(bytearray(data.numpy()))
#     break

# transform_matrix = np.reshape(np.array(frame.pose.transform), [4, 4])
lidar_to_vehicle = np.loadtxt(D_PATH + "train/matrix/seq_1_las_top_ex_matrix.txt").reshape(4,4)

lidar_files = glob.glob("/mnt/data/WAYMO_det3d/train/lidar/seq_1_frame_*")
file_nums = len(lidar_files)

for idx in tqdm.tqdm(range(file_nums)):
    ld_file = "/mnt/data/WAYMO_det3d/train/lidar/seq_1_frame_%d.pkl"%idx
    points = get_lidar_pkl(ld_file)

    ones = np.ones(shape=(points.shape[0], 1))
    points_xyz = points[:, :3]
    points_xyz1 = np.concatenate([points_xyz, ones], -1)
    # print(points_xyz1[0])
    transform_matrix = np.loadtxt("/mnt/data/WAYMO_det3d/train/pose/seq_1_frame_%d.txt"%(idx))

    points_xyz1_veh = np.matmul(points_xyz1, lidar_to_vehicle)
    # print(points_xyz1_veh[0])

    points_world = np.matmul(points_xyz1, transform_matrix)
    # print(points_world[0])

    # np.savetxt(GLOBAL_SAVE_PATH + pathlib.Path(ld_file).stem + ".txt", box_world)
    np.save(GLOBAL_SAVE_PATH + pathlib.Path(ld_file).stem + ".npy", points_world.astype(np.float16))

    # break




