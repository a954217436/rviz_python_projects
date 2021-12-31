import os
import shutil
import time
import argparse
import numpy as np
from tqdm import tqdm

from filterpy.kalman import KalmanFilter


class KalmanVelo(object):
    def __init__(self, velo, gamma=0.5, delta_t=0.1):
        """
        Initialises a tracker using initial velo.
        state(velo) : (vx, vy, ax, ay)
        """
        # define constant acceleration model
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.zeros((4, 4))
        self.kf.H = np.zeros((2, 4))

        # typical linear formulation
        for i in range(4):
            self.kf.F[i, i] = 1
            if i < 2:
                self.kf.H[i, i] = 1
        self.kf.F[0, 2] = self.kf.F[1, 3] = delta_t

        # self.kf.R[2:4, 2:4] *= 100  # 10.
        self.kf.R[:2, :2] = 0.05   # for theta, small R : believe in detection
        # self.kf.R[6, 6] = 0.01      # for theta, big  R : believe in prediction

        self.kf.P[3:, 3:] *= 1000.  # give high uncertainty to the unobservable initial acc
        # self.kf.P *= 10.
        # self.kf.P *= 10

        # self.kf.Q[-1,-1] *= 0.01
        #  (x,y,z,H,W,Z,theta)
        self.kf.x[:2, 0] = velo
        # 目前已经连续多少次未击中
        self.time_since_update = 0
        self.history = []
        # 目前总共击中了多少次
        self.hits = 0
        # 目前连续击中了多少次，当前step未更新则为0
        self.hit_streak = 0
        # 该tracker生命长度
        self.age = 0

        self.gamma = gamma
        self.last_velo= None
        self.track_state = 0  # NEW


    def update(self, velo):
        """
        velo : [vx, vy]
        Updates the state vector with observed velo.
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(velo.reshape(-1, 1))


    def predict(self):
        """
        Advances the state vector and returns the predicted velo estimate.
        [vx, vy, ax, ay]
        """
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):  # 也就是连续执行两次 predict
            self.hit_streak = 0
        self.time_since_update += 1
  
        self.history.append(self.kf.x.reshape(1, -1))
        return self.kf.x


    def get_state(self):
        """
        Returns the current velo estimate.
        """
        ret = self.kf.x.reshape(1,-1)
        # ret = self.smooth_average(ret)
        return ret


class MotionFilter(object):
    def __init__(self):
        self.velo_filters = []

    def update(self, velo=np.empty((0, 2))):
        pass


if __name__ == "__main__":
    x = np.array([[0.11, 0.12],
                  [0.20, 0.23],
                  [0.32, 0.31],
                  [0.41, 0.42],
                  [0.52, 0.42],
                  [0.64, 0.62],
                  [0.55, 0.53],
                  [0.41, 0.46],
                  [0.33, 0.33],
                  [0.27, 0.23],
                  [0.12, 0.11],
                  [0.03, 0.02],
                  ])
    
    kv = KalmanVelo(x[0])
    for xx in x[1:]:
        print("xx = ", xx)
        yy = kv.predict()
        print("yy = ", yy)
        kv.update(xx)
        zz = kv.get_state()
        print("zz = ", zz)
        print("-*"*10)




