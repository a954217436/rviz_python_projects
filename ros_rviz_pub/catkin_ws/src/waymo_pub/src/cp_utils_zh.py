#!/usr/bin/python3
import copy
import copy
import pickle

import numpy as np
from tqdm import tqdm


def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 


def reorganize_info(infos):
    new_info = {}

    for info in infos:
        token = info['token']
        new_info[token] = info

    return new_info 


def transform_box(box, pose):
    """
    Transforms 3d upright boxes from one frame to another.
    Args:
        box: [..., N, 7] boxes.
        from_frame_pose: [...,4, 4] origin frame poses.
        to_frame_pose: [...,4, 4] target frame poses.
    Returns:
        Transformed boxes of shape [..., N, 7] with the same type as box.
    """
    transform = pose 
    heading = box[..., -1] + np.arctan2(transform[..., 1, 0], transform[..., 0,
                                                                    0])
    center = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                    box[..., 0:3]) + np.expand_dims(
                        transform[..., 0:3, 3], axis=-2)

    velocity = box[..., [6, 7]] 

    velocity = np.concatenate([velocity, np.zeros((velocity.shape[0], 1))], axis=-1) # add z velocity

    velocity = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                    velocity)[..., [0, 1]] # remove z axis 

    return np.concatenate([center, box[..., 3:6], velocity, heading[..., np.newaxis]], axis=-1)    


def sort_detections(detections):
    indices = [] 

    for det in detections:
        f = det['token']
        seq_id = int(f.split("_")[1])
        frame_id= int(f.split("_")[3][:-4])

        idx = seq_id * 1000 + frame_id
        indices.append(idx)

    rank = list(np.argsort(np.array(indices)))
    detections = [detections[r] for r in rank]
    return detections


def convert_detection_to_global_box(detections, infos):
    ret_list = [] 

    detection_results = {} # copy.deepcopy(detections)

    for token in tqdm(infos.keys()):
        detection = detections[token]
        detection_results[token] = copy.deepcopy(detection)

        info = infos[token]
        # pose = get_transform(info)
        anno_path = info['anno_path']
        ref_obj = get_obj(anno_path)
        pose = np.reshape(ref_obj['veh_to_global'], [4, 4])

        box3d = detection["box3d_lidar"].detach().clone().cpu().numpy() 
        labels = detection["label_preds"].detach().clone().cpu().numpy()
        scores = detection['scores'].detach().clone().cpu().numpy()
        box3d[:, -1] = -box3d[:, -1] - np.pi / 2
        box3d[:, [3, 4]] = box3d[:, [4, 3]]

        box3d = transform_box(box3d, pose)

        frame_id = token.split('_')[3][:-4]

        num_box = len(box3d)

        anno_list =[]
        for i in range(num_box):
            anno = {
                'translation': box3d[i, :3],
                'velocity': box3d[i, [6, 7]],
                'detection_name': label_to_name(labels[i]),
                'score': scores[i], 
                'box_id': i 
            }

            anno_list.append(anno)

        ret_list.append({
            'token': token, 
            'frame_id':int(frame_id),
            'global_boxs': anno_list,
            'timestamp': info['timestamp'] 
        })

    sorted_ret_list = sort_detections(ret_list)

    return sorted_ret_list, detection_results 