# -*- coding:utf-8 -*-

from __future__ import print_function

import os
import shutil
import time
import argparse
import numpy as np
from tqdm import tqdm

from filterpy.kalman import KalmanFilter
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu, boxes_iou3d_gpu
from track_data import TrackData

np.random.seed(0)


STATE_MAP = {0 : "Ne",    # "NEW" 
             1 : "St",    # "STABLE"
             2 : "Lo",    # "LOSE"  
             3 : "De" }   # "DELETE"


STATE_CHANGE = {"NEW2STABLE"    : 1,
                "STABLE2LOSE"   : 2,
                "LOSE2DELETE"   : 2,
                "NEW2DELETE"    : 2,
                "LOSE2STABLE"   : 1}


def read_pred(path):
    preds = np.loadtxt(path)
    # print(path, " has %d results..."%len(preds))
    return preds


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers) == 0):
        # TODO : why np.empty((0,5)) ?
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    if(len(detections) == 0):
        return np.empty((0, 2), dtype=int), np.empty((0, 5), dtype=int), np.arange(len(trackers)), 
    
    # N x M,直接调用pcdet底层iou算法
    # iou_matrix = iou_batch3d(detections, trackers)
    iou_matrix = boxes_bev_iou_cpu(detections[:, :7], trackers)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        # 每个detection最多只配到了一个tracker，或者每个tracker最多只配到了一个detection
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # matched_indices :  (N x 2)
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    # 在分配的基础上必须大于iou阈值才能算配对
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, trackdata, gamma=0.5):
        """
        Initialises a tracker using initial bounding box.
            bbox : (cx, cy, cz, dx, dy, dz, heading, timestamp)
            state: (cx, cy, cz, dx, dy, dz, heading, vx, vy, ax, ay)
        """
        self.delta_time = 0.1
        self.track_data = trackdata
        # self.last_trackdata = trackdata

        # define constant acceleration model
        self.kf = KalmanFilter(dim_x=11,  dim_z=7)
        #
        self.kf.F = np.zeros((11, 11))
        self.kf.H = np.zeros((7, 11))
        

        for i in range(11):
            self.kf.F[i, i] = 1
            if i < 7:
                self.kf.H[i, i] = 1

        # print(self.kf.H)
        self.kf.F[0, 7] = self.kf.F[1, 8]  = self.delta_time
        self.kf.F[7, 9] = self.kf.F[8, 10] = self.delta_time
        self.kf.F[0, 9] = self.kf.F[1, 10] = 0.5 * self.delta_time ** 2
        # print(self.kf.F)

        self.kf.R[:, :] *= 10
        self.kf.R[2:6, 2:6] *= 100
        self.kf.R[6, 6] *= 0.01

        self.kf.P[7:, 7:] *= 1000.  # give high uncertainty to the unobservable initial accelerations
        # self.kf.Q[2:7, 2:7] *= 0.01

        # print("self.kf.R = ", self.kf.R)
        # print("self.kf.P = ", self.kf.P)
        # print("self.kf.Q = ", self.kf.Q)

        self.kf.x[:7, 0] = self.track_data.world_box
        self.kf.x[7:9, 0] = 0.01          # initial velo, acce = 0

        # 目前已经连续多少次未击中
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        self.track_data.id = self.id
        KalmanBoxTracker.count += 1

        self.history = [self.track_data]
        # 目前总共击中了多少次
        self.hits = 0
        # 目前连续击中了多少次，当前step未更新则为0
        self.hit_streak = 0
        # 该tracker生命长度
        self.age = 0
        # add score and class
        self.score = 0
        self.label = 0

        self.gamma = gamma
        # self.last_score = None
        self.track_state = 0  # NEW
        # self.mesured_velo = np.array([0, 0])
        # self.last_box_filtered = None  #(cx, cy, cz, dx, dy, dz, heading, vx, vy, ax, ay)


    # def smooth_average(self, bbox):
    #     # # bbox (x,y,z,H,W,Z,theta, score)
    #     if self.last_score is None:
    #         self.last_score = self.score
    #     else:
    #         self.score = self.gamma * self.last_score + (1-self.gamma) * self.score
    #         self.last_score = self.score

    #     if self.last_box is None:
    #         self.last_box = bbox
    #     else:
    #         bbox[2:6] =  self.gamma * self.last_box[2:6] + (1-self.gamma) * bbox[2:6]
    #         self.last_box = bbox
        
    #     # theta 取最近几次的平均值
    #     if len(self.theta_history) >= self.theta_mean_num:
    #         theta_mean = self.theta_history[-self.theta_mean_num:].mean()
    #     else:
    #         theta_mean = self.kf.x[6][0]
    #     bbox[6] = theta_mean

    #     return bbox


    def update(self, track_data):
        """
        Updates the state vector with observed bbox.
            track_data.world_box : 
                (cx, cy, cz, dx, dy, dz, heading, score, class, timestamp)  
        """
        self.track_data = track_data
        # print("UUUUUUUUUUUUUUpdate with: ", track_data)

        bbox = track_data.world_box
        last_trackdata = self.history[-1]
        last_bbox = last_trackdata.world_box

        time_diff = track_data.timestamp - last_trackdata.timestamp
        if time_diff < 0.01:
            velo_mesured = np.array([0., 0.])
        else:
            velo_mesured = (bbox[:2] - last_bbox[:2]) / time_diff
        # if self.id == 10:
        #     print(bbox[:2], " - ", last_bbox[:2])

        # self.mesured_velo = velo_mesured

        self.track_data.mesured_center_velocity = velo_mesured

        self.kf.update(bbox[:7])
        
        self.track_data.world_box_filtered = self.kf.x.reshape(1, -1)[0][:7]
        self.track_data.is_predict = False
        self.history.append(self.track_data)

        self.track_data.output_velocity = self.kf.x.reshape(1, -1)[0][-4:-2]

        # if self.id == 76:
        #     print(self.track_data)

        # 更新内部状态
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        # 更新 track_state
        if (self.track_state == 0) and (self.hit_streak >= STATE_CHANGE["NEW2STABLE"]):
            self.track_state = 1
        if self.track_state == 2 and self.hit_streak >= STATE_CHANGE["LOSE2STABLE"]:
            self.track_state = 1
        

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
            bbox : (cx, cy, cz, dx, dy, dz, heading, vx, vy)
        """
        # print("before pred: ", self.kf.x)
        self.kf.predict()
        # print("after pred: ", self.kf.x)
        self.age += 1

        # 预测的 bbox
        self.track_data.world_box_predict = self.kf.x.reshape(1, -1)[0][:7]
        self.track_data.is_predict = True
        # self.track_data.world_box_filtered = self.kf.x.reshape(1, -1)[0][:7]

        if(self.time_since_update > 0):  # 也就是连续执行两次 predict
            self.hit_streak = 0
        self.time_since_update += 1

        if self.track_state == 1 and self.time_since_update >= STATE_CHANGE["STABLE2LOSE"]:
            self.track_state = 2
        if self.track_state == 2 and self.time_since_update >= STATE_CHANGE["LOSE2DELETE"]:
            self.track_state = 3
        if self.track_state == 0 and self.time_since_update >= STATE_CHANGE["NEW2DELETE"]:
            self.track_state = 3

        return self.kf.x.reshape(1, -1)[:7]


    def get_state(self):
        """
        Returns the current bounding box estimate.
            return: (cx, cy, cz, dx, dy, dz, heading, vx, vy, score, label, state)
        """
        # print("SSSSSSSSSSSSSSSsstate: ", self.track_state, ": ", self.id)
        self.track_data.id = self.id
        self.track_data.state = self.track_state
        self.track_data.plot_string = "%d-%.2f-%s"%(self.track_data.id, self.track_data.score, self.track_state)
        return self.track_data

        # ret = self.kf.x[:7]
        # # # ret = self.smooth_average(ret)
        # ret = np.append(ret, self.score)            # score
        # ret = np.append(ret, self.label)            # label
        # # ret = np.append(ret, self.track_state)      # track_state

        # # ret = self.last_box[:-1]   #(cx, cy, cz, dx, dy, dz, heading, score, class)
        # # ret = np.append(ret, self.track_state)      # (cx, cy, cz, dx, dy, dz, heading, score, class, track_state)
        # # ret = np.insert(ret, 7, self.kf.x[-3])
        # # ret = np.insert(ret, 7, self.kf.x[-4])
        # ret = np.insert(ret, 7, self.mesured_velo[1])
        # ret = np.insert(ret, 7, self.mesured_velo[0])

        # return ret.reshape(1,-1)    # (cx, cy, cz, dx, dy, dz, heading, vx, vy, score, class, track_state)



class Sort(object):
    def __init__(self, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.trackers = []
        self.frame_count = 0
        self.iou_threshold = iou_threshold

    def update(self, dets=None):
        """
        Params:
            dets: list: [TrackData_1, TrackData_2......]
        Requires: this method must be called once for each frame even with 
                  empty detections (use np.empty((0, 9)) for frames without detections).
        Returns:  a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        self.frame_count += 1

        # 1. 预测阶段
        predict_states = np.zeros((len(self.trackers), 7))
        to_del = []
        for t, trk in enumerate(predict_states):
            # pos : [cx, cy, cz, dx, dy, dz, heading]
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        # 删除 NAN 的预测与 tracker
        predict_states = np.ma.compress_rows(np.ma.masked_invalid(predict_states))
        for t in reversed(to_del):
            self.trackers.pop(t)

        all_det_boxes = np.array([a.world_box for a in dets])
        # 2. 匹配阶段
        matched, unmatched_dets, unmatched_trks = \
            associate_detections_to_trackers(all_det_boxes, predict_states, self.iou_threshold)

        # print(matched)
        # update matched trackers with assigned detections
        for m in matched:
            matched_det_idx = m[0]
            # self.trackers[m[1]].update(data_dict['world_boxes'][m[0], :])
            self.trackers[m[1]].update(dets[matched_det_idx])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            # print(dets[i, :])
            # new_trk = KalmanBoxTracker(data_dict['world_boxes'][i, :])
            new_trk = KalmanBoxTracker(dets[i])
            self.trackers.append(new_trk)

        # 3. 后处理阶段
        tracker_nums = len(self.trackers)
        final_results = []
        for trk in reversed(self.trackers):
            # estimate_state = trk.get_state()[0]
            track_data_final = trk.get_state()
            # if trk.track_state in [1,2]:    # new || stable || lose
            #     # final_results.append(np.append(estimate_state, trk.id+1))
            if trk.track_state in [0,1,2]: 
                final_results.append(track_data_final)
            tracker_nums -= 1
            if trk.track_state == 3:    # delete
                self.trackers.pop(tracker_nums)

        return final_results
        # # (N,13): [cx, cy, cz, dx, dy, dz, heading, vx, vy, score, label, state, track_id]
        # if(len(final_results) > 0):
        #     return np.stack(final_results)
        # return np.empty((0, 10))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display',
                        help='Display online tracker output (slow) [False]', action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.",
                        type=str, default='demo_pointrcnn_iou_results')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", 
                        type=str, default='train')
    parser.add_argument("--iou_threshold",
                        help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    args = parse_args()
    display = args.display
    phase = args.phase
