# -*- coding:utf-8 -*-
# AB3DMOT
import numpy as np
from filterpy.kalman import KalmanFilter

np.random.seed(0)


STATE_CHANGE = {"NEW2STABLE"    : 2,
                "STABLE2LOSE"   : 2,
                "LOSE2DELETE"   : 2,
                "NEW2DELETE"    : 2,
                "LOSE2STABLE"   : 1}


def correction_angle(theta):
    if theta >= np.pi: 
        theta -= np.pi * 2    # make the theta still in the range
    if theta < -np.pi: 
        theta += np.pi * 2
    return theta


def acute_angle(new_theta, predicted_theta):
    correction_theta = predicted_theta

    # if the angle of two theta is not acute angle
    if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:
        correction_theta += np.pi       
        correction_theta = correction_angle(correction_theta)

    # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
    if abs(new_theta - correction_theta) >= np.pi * 3 / 2.0:
        if new_theta > 0: 
            correction_theta += np.pi * 2
        else: 
            correction_theta -= np.pi * 2
    return correction_theta


class KalmanBoxTrackerCVRot(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, trackdata, gamma=0.5):
        """
        Initialises a tracker using initial bounding box.
            bbox : (cx, cy, cz, dx, dy, dz, heading, timestamp)
            state: (cx, cy, cz, dx, dy, dz, heading, vx, vy, vh)
        """
        self.delta_time = 0.1
        self.track_data = trackdata
        # self.last_trackdata = trackdata

        # define constant acceleration model
        self.kf = KalmanFilter(dim_x=10,  dim_z=7)

		# with angular velocity
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        # # state transition matrix       
        self.kf.F = np.array([[1,0,0,0,0,0,0,self.delta_time,0,0],    # cx
		                      [0,1,0,0,0,0,0,0,self.delta_time,0],    # cy
		                      [0,0,1,0,0,0,0,0,0,0],                  # cz
		                      [0,0,0,1,0,0,0,0,0,0],                  # dx
		                      [0,0,0,0,1,0,0,0,0,0],                  # dy
		                      [0,0,0,0,0,1,0,0,0,0],                  # dz
		                      [0,0,0,0,0,0,1,0,0,self.delta_time],                  # theta
		                      [0,0,0,0,0,0,0,1,0,0],
		                      [0,0,0,0,0,0,0,0,1,0],
		                      [0,0,0,0,0,0,0,0,0,1]])     

        self.kf.H = np.array([[1,0,0,0,0,0,0,0,0,0],      # measurement function,
		                      [0,1,0,0,0,0,0,0,0,0],
		                      [0,0,1,0,0,0,0,0,0,0],
		                      [0,0,0,1,0,0,0,0,0,0],
		                      [0,0,0,0,1,0,0,0,0,0],
		                      [0,0,0,0,0,1,0,0,0,0],
		                      [0,0,0,0,0,0,1,0,0,0]])

        # # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
        self.kf.P[7:, 7:] *= 1000. 	# state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
        self.kf.P *= 10.

		# self.kf.Q[-1,-1] *= 0.01    # process uncertainty
        self.kf.Q[7:, 7:] *= 0.01

        self.kf.x[:7, 0] = self.track_data.world_box[:7]
        # self.kf.x[7:9, 0] = 0.01          # initial velo, acce = 0
        # print("Initial self.kf.x = ", self.kf.x)

        # 目前已经连续多少次未击中
        self.time_since_update = 0
        self.id = KalmanBoxTrackerCVRot.count
        self.track_data.id = self.id
        KalmanBoxTrackerCVRot.count += 1

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

        self.kf.x[6] = correction_angle(self.kf.x[6])
        bbox[6] = correction_angle(bbox[6])

        # print("bbox[6] = ", bbox[6])
        # print("self.kf.x[6] = ", self.kf.x[6])
        self.kf.x[6] = acute_angle(bbox[6], self.kf.x[6])
        # print("self.kf.x[6] = ", self.kf.x[6])
        # print("update before self.kf.x = ", self.kf.x)
        # print("update with bbox[:7] = ", bbox[:7])
        self.kf.update(bbox[:7])
        # print("update done  self.kf.x = ",  self.kf.x)
        
        self.track_data.world_box_filtered = self.kf.x.reshape(1, -1)[0][:7]
        self.track_data.is_predict = False
        self.history.append(self.track_data)

        self.track_data.output_velocity = self.kf.x.reshape(1, -1)[0][-3:-1]

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
        # print("before predict, self.kf.x = ", self.kf.x)
        self.kf.predict()
        # print("after  predict, self.kf.x = ", self.kf.x)

        self.kf.x[6] = correction_angle(self.kf.x[6])
        self.age += 1

        # 预测的 bbox
        self.track_data.world_box_predict = self.kf.x.reshape(1, -1)[0][:7]
        self.track_data.is_predict = True
        self.track_data.world_box_filtered = self.kf.x.reshape(1, -1)[0][:7]

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


