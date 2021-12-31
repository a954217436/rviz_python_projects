import numpy as np
from .associations import associate_detections_to_trackers
from .kf_tracker import KalmanBoxTrackerCA as KFModel
from .life_manage import TrackState


class MOTModel(object):
    def __init__(self, iou_threshold=0.3, det_threshold=0.5, use_giou=False):
        """
        Sets key parameters for MOTModel
        """
        self.trackers = []
        self.frame_count = 0
        self.iou_threshold = iou_threshold
        self.det_threshold = det_threshold
        self.use_giou = use_giou

    def update(self, dets=None):
        """
        Params:
            dets: list: [TrackData_1, TrackData_2......]
        Requires: this method must be called once for each frame even with 
                  empty detections (use np.empty((0, 9)) for frames without detections).
        Returns:  a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        dets = [a for a in dets if a.score >= self.det_threshold]
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
            associate_detections_to_trackers(all_det_boxes, predict_states, self.iou_threshold, self.use_giou)

        # print(matched)
        # update matched trackers with assigned detections
        for m in matched:
            matched_det_idx = m[0]
            # self.trackers[m[1]].update(data_dict['world_boxes'][m[0], :])
            self.trackers[m[1]].update(dets[matched_det_idx])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            # print(dets[i, :])
            new_trk = KFModel(dets[i])
            self.trackers.append(new_trk)

        # 3. 后处理阶段
        tracker_nums = len(self.trackers)
        final_results = []
        for trk in reversed(self.trackers):
            # estimate_state = trk.get_state()[0]
            track_data_final = trk.get_state()
            # if trk.track_state in [1,2]:    # new || stable || lose
            #     # final_results.append(np.append(estimate_state, trk.id+1))
            # if trk.track_state in [0,1,2]: 
            if trk.life_manager.state in [TrackState.new, TrackState.stable, TrackState.lose]: 
                final_results.append(track_data_final)
            tracker_nums -= 1
            if trk.life_manager.state == TrackState.delete:    # delete
                self.trackers.pop(tracker_nums)

        return final_results