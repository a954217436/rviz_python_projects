# -*- coding:utf-8 -*-

import copy
import numpy as np
from filterpy.kalman import KalmanFilter
from .angle_correction import acute_angle, correction_angle

np.random.seed(0)


STATE_CHANGE = {"NEW2STABLE"    : 1,
                "STABLE2LOSE"   : 2,
                "LOSE2DELETE"   : 3,
                "NEW2DELETE"    : 2,
                "LOSE2STABLE"   : 1}


class KalmanBoxTrackerCARot(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, trackdata):
        """
        Initialises a tracker using initial bounding box.
            bbox : (cx, cy, cz, dx, dy, dz, heading, timestamp)
            state: (cx, cy, cz, dx, dy, dz, heading, vx, vy, ax, ay, vh)
        """
        self.delta_time = 0.1
        self.track_data = trackdata
        # self.last_trackdata = trackdata

        # define constant acceleration model
        self.kf = KalmanFilter(dim_x=12,  dim_z=7)
        #
        self.kf.F = np.zeros((12, 12))
        self.kf.H = np.zeros((7, 12))

        for i in range(12):
            self.kf.F[i, i] = 1
            if i < 7:
                self.kf.H[i, i] = 1

        # print(self.kf.H)
        self.kf.F[0, 7] = self.kf.F[1, 8]  = self.delta_time
        self.kf.F[7, 9] = self.kf.F[8, 10] = self.delta_time
        self.kf.F[6,11] = self.delta_time
        self.kf.F[0, 9] = self.kf.F[1, 10] = 0.5 * self.delta_time ** 2
        # print(self.kf.F)
        self.original_F = copy.deepcopy(self.kf.F)

        self.kf.R[:, :] *= 10
        self.kf.R[2:6, 2:6] *= 100
        self.kf.R[6, 6] *= 0.01

        self.kf.P[7:, 7:] *= 1000.  # give high uncertainty to the unobservable initial accelerations
        # self.kf.Q[2:7, 2:7] *= 0.01

        self.kf.x[:7, 0] = self.track_data.world_box
        # self.kf.x[7:, 0] = 0.0          # initial velo, acce = 0

        # 目前已经连续多少次未击中
        self.time_since_update = 0
        self.id = KalmanBoxTrackerCARot.count
        self.track_data.id = self.id
        KalmanBoxTrackerCARot.count += 1

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

        # self.gamma = gamma
        # self.last_score = None
        self.track_state = 0  # NEW


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

        # time_diff = track_data.timestamp - last_trackdata.timestamp
        # if time_diff < 0.01:
        #     velo_mesured = np.array([0., 0.])
        # else:
        #     ct_translation = bbox[:2] - last_bbox[:2]
        #     velo_mesured = ct_translation
        
        # # self.mesured_velo = velo_mesured
        # self.track_data.mesured_center_velocity = velo_mesured

        # # For Vehicle
        # if track_data.label == 0:
        #     cur_dir = bbox[-1]
        #     pre_dir = last_bbox[-1]
        #     delta_dir = cur_dir - pre_dir
        #     rot_extend_matrix = np.eye(11)
        #     for i in range(7, 11):
        #         rot_extend_matrix[i, i] = np.cos(delta_dir)
        #     rot_extend_matrix[7,8] = rot_extend_matrix[9,10] = -np.sin(delta_dir)
        #     rot_extend_matrix[8,7] = rot_extend_matrix[10,9] =  np.sin(delta_dir)
        #     self.kf.F = np.matmul(rot_extend_matrix, self.kf.F)
        
        self.kf.x[6] = correction_angle(self.kf.x[6])
        bbox[6] = correction_angle(bbox[6])
        self.kf.x[6] = acute_angle(bbox[6], self.kf.x[6])

        self.kf.update(bbox[:7])
        self.kf.F = copy.deepcopy(self.original_F)

        self.track_data.world_box_filtered = self.kf.x.reshape(1, -1)[0][:7]
        self.track_data.is_predict = False
        self.history.append(self.track_data)

        self.track_data.output_velocity = self.kf.x.reshape(1, -1)[0][-5:-3]

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
        self.kf.x[6] = correction_angle(self.kf.x[6])
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


