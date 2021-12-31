import numpy as np
from .utils.data_utils import veh_to_world_bbox

class TrackData(object):
    def __init__(self,
                 id = -1,
                 local_box = None,
                 world_box = None,
                 score = None,
                 label = None,
                 pose = None,
                 timestamp = None):
                
        self.id = id                        # int
        self.local_box = local_box          # [x, y, z, dx, dy, dz, heading]
        self.local_box_filtered = None

        if world_box is None:
            self.world_box = veh_to_world_bbox(np.array([local_box]), pose)[0]
        else:
            self.world_box = world_box          # [x, y, z, dx, dy, dz, heading]
        self.world_box_filtered = world_box
        
        self.score = score
        self.label = label

        self.timestamp = timestamp
        
        self.is_static = True            # object is static or moving
        self.is_predict = False          # object is predict or update

        self.mesured_center_velocity = np.zeros(2)
        self.mesured_center_acceleration = None
        self.output_velocity = np.zeros(2)

        self.state = None
        self.plot_string = ""

        self.pose = pose


    def __str__(self) -> str:
        ret = "<<<<<<\n    id=" + \
                str(self.id)          + "\n    score=" + \
                str(self.score)       + "\n    label=" + \
                str(self.label)       + "\n    local_box=" + \
                str(self.local_box)   + "\n    world_box=" + \
                str(self.world_box)   + "\n    bary_center=" + \
                str(self.mesured_center_velocity) + "\n    output_velocity=" + \
                str(self.output_velocity) + "\n    is_pred=" + \
                str(self.is_predict)  + "\n    timestamp=" + \
                str(self.timestamp)   + "\n>>>>>> \n"
            
        return ret
