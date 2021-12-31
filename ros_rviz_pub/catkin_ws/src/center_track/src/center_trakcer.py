#!/usr/bin/python3

'''
From CenterTrack (CenterPoint)
'''

import copy
import numpy as np


WAYMO_TRACKING_NAMES = [
    'VEHICLE',
    'PEDESTRIAN',
    'CYCLIST'
]


def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)

    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18  # 本行作废, 不再参与
            matched_indices.append([i, j])

    return np.array(matched_indices, np.int32).reshape(-1, 2)


class Tracker(object):
    def __init__(self, max_age=0, max_dist={}, score_thresh=0.1):
        self.max_age = max_age
        self.WAYMO_CLS_VELOCITY_ERROR = max_dist 

        self.score_thresh = score_thresh 
        self.reset()
  

    def reset(self):
        self.id_count = 0
        self.tracks = []


    def step_centertrack(self, results, time_lag):
        if len(results) == 0:
            self.tracks = []
            return []
        else:
            temp = []
            for det in results:
                # filter out classes not evaluated for tracking 
                if det['detection_name'] not in WAYMO_TRACKING_NAMES:
                    print("filter {}".format(det['detection_name']))
                    continue 

                det['ct'] = np.array(det['translation'][:2])
                # det['local_box'] = np.array(det['local_box'])
                det['tracking'] = np.array(det['velocity'][:2]) * -1 *  time_lag
                det['label_preds'] = WAYMO_TRACKING_NAMES.index(det['detection_name'])
                temp.append(det)

            results = temp

        N = len(results)
        M = len(self.tracks)

        # N X 2 
        if 'tracking' in results[0]:
            dets = np.array([det['ct'] + det['tracking'].astype(np.float32) for det in results], np.float32)
        else:
            dets = np.array([det['ct'] for det in results], np.float32) 

        item_cat = np.array([item['label_preds'] for item in results], np.int32)           # N
        track_cat = np.array([track['label_preds'] for track in self.tracks], np.int32)    # M
        max_diff = np.array([self.WAYMO_CLS_VELOCITY_ERROR[box['detection_name']] for box in results], np.float32)  # N
        tracks = np.array([pre_det['ct'] for pre_det in self.tracks], np.float32)          # M x 2

        if len(tracks) > 0:  # NOT FIRST FRAME
            dist = (((tracks.reshape(1, -1, 2) - dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
            dist = np.sqrt(dist) # absolute distance in meter

            invalid = ((dist > max_diff.reshape(N, 1)) + (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0

            dist = dist + invalid * 1e18
            matched_indices = greedy_assignment(copy.deepcopy(dist))

        else:  # first few frame
            assert M == 0
            matched_indices = np.array([], np.int32).reshape(-1, 2)

        unmatched_dets = [d for d in range(dets.shape[0]) if not (d in matched_indices[:, 0])]
        unmatched_tracks = [d for d in range(tracks.shape[0]) if not (d in matched_indices[:, 1])]
        matches = matched_indices

        ret = []
        for m in matches:
            track = results[m[0]]
            track['tracking_id'] = self.tracks[m[1]]['tracking_id']      
            track['age'] = 1
            track['active'] = self.tracks[m[1]]['active'] + 1
            
            # zhanghao
            cur_box7d = track['local_box']
            track['last_box7d'] = cur_box7d
            # if 'last_box7d' in self.tracks[m[1]]:
            #     last_box7d = self.tracks[m[1]]['last_box7d']

            #     # process theta, make the theta still in the range
            #     if last_box7d[6] >= np.pi:  last_box7d[6] -= np.pi * 2    
            #     if last_box7d[6] < -np.pi:  last_box7d[6] += np.pi * 2
            #     if cur_box7d[6] >= np.pi:  cur_box7d[6] -= np.pi * 2    
            #     if cur_box7d[6] < -np.pi:  cur_box7d[6] += np.pi * 2

            #     # 根据 last_box7d 中的角度，纠正当前检测帧的角度
            #     # 这个方法可以平滑角度抖动，但是可能会出现: 前面帧检测方向反了，导致后面即使检测了180度相反方向，也无法再纠正
            #     # # 若两个角度差不是锐角, 在90~270度之间，则新角度加上 180 度
            #     theta_diff = abs(cur_box7d[6] - last_box7d[6])
            #     if np.pi / 2.0 < theta_diff < np.pi * 3 / 2.0:
            #         cur_box7d[6] += np.pi
            #         if cur_box7d[6] >= np.pi:  cur_box7d[6] -= np.pi * 2    
            #         if cur_box7d[6] < -np.pi:  cur_box7d[6] += np.pi * 2
            #     # 角度差在90~270度之间处理完了，再处理 > 270度的情况
            #     if abs(cur_box7d[6] - last_box7d[6]) >= np.pi * 3 / 2.0:
            #         if last_box7d[6] > 0: 
            #             cur_box7d[6] += np.pi * 2
            #         else: 
            #             cur_box7d[6] -= np.pi * 2

            #     gamma = 0.5
            #     mean_box7d = last_box7d * (1 - gamma) + cur_box7d * gamma
            #     track['local_box'] = mean_box7d

            ret.append(track)

        for i in unmatched_dets:
            track = results[i]
            if track['score'] > self.score_thresh:
                self.id_count += 1
                track['tracking_id'] = self.id_count
                track['age'] = 1
                track['active'] =  1
                ret.append(track)

        # still store unmatched tracks if its age doesn't exceed max_age, however, we shouldn't output 
        # the object in current frame 
        for i in unmatched_tracks:
            track = self.tracks[i]
            if track['age'] < self.max_age:
                track['age'] += 1
                track['active'] = 0
                ct = track['ct']

                # movement in the last second
                if 'tracking' in track:
                    offset = track['tracking'] * -1 # move forward 
                    track['ct'] = ct + offset 
                ret.append(track)

        self.tracks = ret
        return ret
