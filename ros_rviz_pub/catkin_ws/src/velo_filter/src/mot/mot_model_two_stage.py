import numpy as np
from .associations import associate_detections_to_trackers
from .kf_tracker_ca import KalmanBoxTrackerCA as KFModel
from .kf_tracker_carot import KalmanBoxTrackerCARot
from .kf_tracker_cv import KalmanBoxTrackerCV
from .kf_tracker_cvrot import KalmanBoxTrackerCVRot


class MOTModelTwoStage(object):
    def __init__(self, iou_threshold=0.3, high_thres=0.5, low_thres=0.1):
        """
        Sets key parameters for MOTModel
        """
        self.trackers = []
        self.frame_count = 0
        self.iou_threshold = iou_threshold
        self.high_thres = high_thres
        self.low_thres = low_thres

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
        # # 删除 NAN 的预测与 tracker
        # predict_states = np.ma.compress_rows(np.ma.masked_invalid(predict_states))
        # for t in reversed(to_del):
        #     self.trackers.pop(t)

        # 2. Two-stage Association
        det_infos_1st = np.array([a for a in dets if a.score >= self.high_thres])
        det_infos_2nd = np.array([a for a in dets if (self.low_thres <= a.score < self.high_thres)])

        # 2.1 High score association
        det_boxes_1st = np.array([a.world_box for a in det_infos_1st])
        matched, unmatched_dets, unmatched_trks = \
            associate_detections_to_trackers(det_boxes_1st, predict_states, self.iou_threshold)

        # print("unmatched_trks = ", unmatched_trks)

        # print(matched)
        # update matched trackers with assigned detections
        for m in matched:
            matched_det_idx = m[0]
            # self.trackers[m[1]].update(data_dict['world_boxes'][m[0], :])
            self.trackers[m[1]].update(det_infos_1st[matched_det_idx])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            # print(dets[i, :])
            new_trk = KFModel(dets[i])
            self.trackers.append(new_trk)

        # 2.2 Low score association
        remained_trks = [self.trackers[i] for i in unmatched_trks]
        det_boxes_2nd = np.array([a.world_box for a in det_infos_2nd])
        if len(remained_trks) > 0 and len(det_boxes_2nd) > 0:
            r_predict_states = predict_states[unmatched_trks]
            # print("     two-stage association......")
            r_matched, r_unmatched_dets, r_unmatched_trks = \
                associate_detections_to_trackers(det_boxes_2nd, r_predict_states, self.iou_threshold)

            # print("r_matched = ", r_matched)
            for m in r_matched:
                matched_det_idx = m[0]
                # self.trackers[m[1]].update(data_dict['world_boxes'][m[0], :])
                # self.trackers[unmatched_trks[m[1]]].update(det_infos_2nd[matched_det_idx])
                self.trackers[unmatched_trks[m[1]]].update(None)

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