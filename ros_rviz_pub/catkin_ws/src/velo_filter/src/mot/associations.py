import numpy as np
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu, boxes_iou3d_gpu
from .box_utils import iou2d, iou3d, giou2d, giou3d, iouMatrix

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers) == 0):
        # TODO : why np.empty((0,5)) ?
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    if(len(detections) == 0):
        return np.empty((0, 2), dtype=int), np.empty((0, 5), dtype=int), np.arange(len(trackers)), 
    
    # N x M, 直接调用pcdet底层iou算法
    # iou_matrix = iou_batch3d(detections, trackers)
    iou_matrix = boxes_bev_iou_cpu(detections[:, :7], trackers)
    # iou_matrix = iouMatrix(detections[:, :7], trackers)     # GIoU
    # print(iou_matrix)
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        # 每个detection最多只配到了一个tracker，或者每个tracker最多只配到了一个detection
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # matched_indices :  (N x 2)
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # print("unmatched_trackers = ", len(unmatched_trackers))
    # filter out matched with low IOU
    # 在分配的基础上必须大于iou阈值才能算配对
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)