import glob
import pathlib
from data_utils import *
from transform import get_image_transform

if __name__ == "__main__":

    OUTPUT_PATH = "/mnt/data/WAYMO_det3d/train/matrix/"
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    ############################################
    #########  保存第一帧 Front-camera   #########
    ############################################
    
    TF_PATH = "/mnt/data/WAYMO_det3d/tfrecord_training/"
    tf_records = glob.glob(TF_PATH + "*.tfrecord")

    for tf_record in tf_records:
        dataset = tf.data.TFRecordDataset(tf_record, compression_type='')
        frame = open_dataset.Frame()

        for idx, data in tqdm.tqdm(enumerate(dataset)):
            frame.ParseFromString(bytearray(data.numpy()))
            break

        ############################################
        #########         保存 matrix       #########
        ############################################
        file_name = pathlib.Path(tf_record).stem
        cam_front_ex_matrix = np.array(frame.context.camera_calibrations[0].extrinsic.transform)
        np.savetxt(OUTPUT_PATH + file_name + "_cam_front_ex_matrix.txt", cam_front_ex_matrix)

        cam_front_in_matrix = np.array(frame.context.camera_calibrations[0].intrinsic)
        np.savetxt(OUTPUT_PATH + file_name + "_cam_front_in_matrix.txt", cam_front_in_matrix)

        las_top_ex_matrix = np.array(frame.context.laser_calibrations[4].extrinsic.transform)
        np.savetxt(OUTPUT_PATH + file_name + "_las_top_ex_matrix.txt", las_top_ex_matrix)
