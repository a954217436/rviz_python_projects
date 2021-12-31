import numpy as np


class TrackData(object):
    def __init__(self,
                 id = -1,
                 local_box = None,
                 world_box = None,
                 score = None,
                 label = None,
                 bary_center = None,
                 corners = None,
                 points_nums = None,
                 timestamp = None):
        self.id = id                       # int
        # self.age = -1                    # int
        self.local_box = local_box       # [x, y, z, dx, dy, dz, heading]
        self.local_box_filtered = None

        self.world_box = world_box       # [x, y, z, dx, dy, dz, heading]
        self.world_box_filtered = None
        self.world_box_predict = None
        
        self.score = score
        self.label = label

        self.bary_center = bary_center
        self.corners = corners
        self.points_nums = points_nums

        self.timestamp = timestamp
        
        self.association_score = 0.
        self.is_static = True           # object is static or moving
        self.is_predict = False          # object is predict or update
        
        # self.center = None
        # self.bary_center = None
        # self.corners = None

        self.mesured_center_velocity = np.zeros(2)
        self.mesured_center_acceleration = None
        self.output_velocity = np.zeros(2)

        self.state = None
        self.mesurement_covarniance = None
        self.state_covarniance = None
        self.output_velocity_uncertainty = None
        self.plot_string = ""

        self.pose = None


    def __str__(self) -> str:
        ret = "<<<<<<\nid=" + \
              str(self.id)          + "\nscore=" + \
              str(self.score)       + "\nlabel=" + \
              str(self.label)       + "\nlocal_box=" + \
              str(self.local_box)   + "\nworld_box=" + \
              str(self.world_box)   + "\nbary_center=" + \
              str(self.bary_center) + "\ncorners=" + \
              str(self.corners)     + "\npoints_nums=" + \
              str(self.points_nums) + "\nmesured_center_velocity=" + \
              str(self.mesured_center_velocity) + "\noutput_velocity=" + \
              str(self.output_velocity) + "\nis_pred=" + \
              str(self.is_predict) + "\ntimestamp=" + \
              str(self.timestamp)   + "\n>>>>>> \n"
            
        return ret
