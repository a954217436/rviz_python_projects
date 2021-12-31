# -*- coding:utf-8 -*-

import numpy as np
from filterpy.kalman import KalmanFilter
from .angle_correction import acute_angle, correction_angle
from .life_manage import LifeManager

np.random.seed(0)


class KalmanBoxTrackerCA(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, trackdata):
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

        self.kf.x[:7, 0] = self.track_data.world_box
        self.kf.x[7:9, 0] = 0.01          # initial velo, acce = 0


        self.id = KalmanBoxTrackerCA.count
        self.track_data.id = self.id
        KalmanBoxTrackerCA.count += 1

        self.history = [self.track_data]
        # add score and class
        self.score = 0
        self.label = 0

        self.life_manager = LifeManager()
        
        
    def update(self, track_data, asso_score=0.0):
        """
        Updates the state vector with observed bbox.
            track_data.world_box : 
                (cx, cy, cz, dx, dy, dz, heading, score, class, timestamp)  
        """
        if track_data is not None:
            self.track_data = track_data
            # print("UUUUUUUUUUUUUUpdate with: ", track_data)

            bbox = track_data.world_box
            last_trackdata = self.history[-1]
            last_bbox = last_trackdata.world_box

            self.kf.x[6] = correction_angle(self.kf.x[6])
            bbox[6] = correction_angle(bbox[6])
            self.kf.x[6] = acute_angle(bbox[6], self.kf.x[6])

            time_diff = track_data.timestamp - last_trackdata.timestamp
            if time_diff < 0.01:
                velo_mesured = np.array([0., 0.])
            else:
                ct_translation = bbox[:2] - last_bbox[:2]
                velo_mesured = ct_translation

            self.track_data.mesured_center_velocity = velo_mesured

            self.kf.update(bbox[:7])
            
            self.track_data.world_box_filtered = self.kf.x.reshape(1, -1)[0][:7]
            self.track_data.is_predict = False
            self.history.append(self.track_data)

            self.track_data.output_velocity = self.kf.x.reshape(1, -1)[0][-4:-2]

            velo_norm = np.linalg.norm(self.track_data.output_velocity)
            ## print("velo_norm = ", velo_norm)
            if velo_norm > 0.8:
                sin_angle = self.track_data.output_velocity[1] / velo_norm
                cos_angle = self.track_data.output_velocity[0] / velo_norm
                velo_angle1 = correction_angle(-np.pi/2.0 - np.arctan2(sin_angle, cos_angle))

                car_angle = self.kf.x[6]
                diff = correction_angle(velo_angle1 - car_angle)

                if abs(diff) > np.pi/6.0:
                    self.track_data.output_velocity = np.array([0., 0.])
            else:
                self.track_data.output_velocity = np.array([0., 0.])

        self.life_manager.update(asso_score)
        

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
            bbox : (cx, cy, cz, dx, dy, dz, heading, vx, vy)
        """
        # print("before pred: ", self.kf.x)
        self.kf.predict()
        self.kf.x[6] = correction_angle(self.kf.x[6])
        # print("after pred: ", self.kf.x)

        # 预测的 bbox
        self.track_data.world_box_predict = self.kf.x.reshape(1, -1)[0][:7]
        self.track_data.world_box_filtered = self.track_data.world_box_predict
        self.track_data.is_predict = True
        # self.track_data.world_box_filtered = self.kf.x.reshape(1, -1)[0][:7]

        self.life_manager.predict()
        return self.kf.x.reshape(1, -1)[:7]


    def get_state(self):
        """
        Returns the current bounding box estimate.
            return: (cx, cy, cz, dx, dy, dz, heading, vx, vy, score, label, state)
        """
        self.track_data.id = self.id
        self.track_data.state = self.life_manager.state
        self.track_data.plot_string = "%d-%.2f-%s"%(self.track_data.id, self.track_data.score, self.life_manager.state_string())
        return self.track_data


