import numpy as np
from filterpy.kalman import KalmanFilter

from .base_tracker import BaseTracker, TrackState


class KFCATracker(BaseTracker):
    def __init__(self, trackdata):
        """
        Initialises a tracker using initial bounding box.
            bbox : (cx, cy, cz, dx, dy, dz, heading, timestamp)
            state: (cx, cy, cz, dx, dy, dz, heading, vx, vy, ax, ay)
        """
        self.delta_time = 0.1
        self.track_data = trackdata

        ######################################################
        # KF Motion Model
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
        # self.kf.x[7:9, 0] = 0.01          # initial velo, acce = 0
        ######################################################

        



