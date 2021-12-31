import os
import pathlib

from data_utils import *
import waymo_decoder
import transform


sub_dirs = [
            "annos", 
            "images",
            "images_front",
            "lidar",
            "point_cloud",
            "matrix",
            "pose",
            "preds",
            ]

def parse_one_record(tfrecord_path, output_path, seq=0):
    
    os.makedirs(output_path, exist_ok=True)
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(output_path, sub_dir), exist_ok=True)

    # 保存个文件夹名称，记录当前 tfrecord 的名称
    os.makedirs(os.path.join(output_path, pathlib.Path(tfrecord_path).stem + "_%d"%seq), exist_ok=True)
    
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    for idx, data in tqdm.tqdm(enumerate(dataset)):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # 保存 matrix, 只需保存一次即可
        if idx == 0:  
            cam_front_ex_matrix = np.array(frame.context.camera_calibrations[0].extrinsic.transform)
            ex_path = os.path.join(output_path, "matrix", "seq_%d_cam_front_ex_matrix.txt"%seq)
            np.savetxt(ex_path, cam_front_ex_matrix)

            cam_front_in_matrix = np.array(frame.context.camera_calibrations[0].intrinsic)
            in_path = os.path.join(output_path, "matrix", "seq_%d_cam_front_in_matrix.txt"%seq)
            np.savetxt(in_path, cam_front_in_matrix)

            las_top_ex_matrix = np.array(frame.context.laser_calibrations[4].extrinsic.transform)
            las_ex_path = os.path.join(output_path, "matrix", "seq_%d_las_top_ex_matrix.txt"%seq)
            np.savetxt(las_ex_path, las_top_ex_matrix)


        # 保存 pose
        pose_npy = np.reshape(np.array(frame.pose.transform), [4, 4])
        pose_path = os.path.join(output_path, "pose", "seq_%d_frame_%d.txt"%(seq, idx))
        np.savetxt(pose_path, pose_npy)


        # # 保存 5 个 camera 的图片 + label
        # serialize_5image_frame(frame, os.path.join(output_path, "images", "seq_%d_frame_%d.jpg"%(seq, idx)))


        # 保存 front camera 图片
        image_front = tf.image.decode_jpeg(frame.images[0].image)
        cv_image_front = np.array(image_front)[...,::-1]
        cv2.imwrite(os.path.join(output_path, "images_front", "seq_%d_frame_%d.jpg"%(seq, idx)), cv_image_front)


        # 保存点云 npy 文件
        range_images, camera_projections, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                                        frame,
                                        range_images,
                                        camera_projections,
                                        range_image_top_pose)
        point_cloud = np.concatenate(points, axis=0)
        np.save(os.path.join(output_path, "point_cloud", "seq_%d_frame_%d.npy"%(seq, idx)), point_cloud)


        # # 保存 CenterPoint 的 annos 和 lidar pkl 格式文件
        # decoded_frame = waymo_decoder.decode_frame(frame, idx)
        # decoded_annos = waymo_decoder.decode_annos(frame, idx)
        # with open(os.path.join(output_path, "lidar", 'seq_{}_frame_{}.pkl'.format(seq, idx)), 'wb') as f:
        #     pickle.dump(decoded_frame, f)
        # with open(os.path.join(output_path, "annos", 'seq_{}_frame_{}.pkl'.format(seq, idx)), 'wb') as f:
        #     pickle.dump(decoded_annos, f)


if __name__ == "__main__":

    # TF_RECORD_PATH = "/mnt/data/WAYMO_det3d/tfrecord_training/segment-15832924468527961_1564_160_1584_160_with_camera_labels.tfrecord"
    # parse_one_record(TF_RECORD_PATH, "/home/zhanghao/testttt")

    import glob
    all_records = glob.glob("/mnt/data/WAYMO_det3d/tfrecord_testing/*.tfrecord")
    print(all_records)

    for ii, record in enumerate(all_records):
        parse_one_record(record, "/mnt/data/WAYMO_details/test/", ii)


