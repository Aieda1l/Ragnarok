"""Motion-only BoT-SORT / BYTE two-stage tracker core.

Vendored from NirAharon/BoT-SORT (tracker/bot_sort.py BoTSORT.update), MIT
licensed. Modifications:
  * All ReID / FastReID / appearance branches removed.
  * The OpenCV optical-flow CMC (``self.gmc.apply``) replaced by an injected
    ``ego.estimate(frame)`` returning a 2x3 affine; applied via STrack.multi_gmc.
  * ``lap.lapjv`` replaced by ragnarok.tracking.associate.linear_assignment
    (scipy).
  * A Mahalanobis 2-DOF gate (ragnarok.tracking.gate.mahalanobis_gate) added to
    the first (high-score) IoU association step.
  * ``mot20`` flag dropped (score fusion always applied).

numpy + scipy only (scipy reached transitively through associate/gate/kalman).
"""
import numpy as np

from ragnarok.tracking.associate import linear_assignment
from ragnarok.tracking.gate import CHI2_GATE_2DOF, mahalanobis_gate

from .iou import iou_distance
from .kalman_xywh import KalmanFilter
from .basetrack import BaseTrack, TrackState
from .strack import STrack


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def apply_mahalanobis_gate(strack_pool, detections, dists, thresh=CHI2_GATE_2DOF):
    """Gate the IoU cost by the 2-DOF Mahalanobis distance on (x, y) centers."""
    if dists.size == 0 or len(strack_pool) == 0 or len(detections) == 0:
        return dists
    kf = STrack.shared_kalman
    means, covs = [], []
    for t in strack_pool:
        pm, pc = kf.project(t.mean, t.covariance)
        means.append(pm[:2])
        covs.append(pc[:2, :2])
    track_mean_xy = np.asarray(means)
    track_S_xy = np.asarray(covs)
    det_xy = np.asarray([d.xywh[:2] for d in detections])
    return mahalanobis_gate(dists, track_mean_xy, track_S_xy, det_xy, thresh=thresh)


class BoTSORT(object):
    def __init__(self, ego, track_high_thresh=0.6, track_low_thresh=0.1,
                 new_track_thresh=0.7, track_buffer=30, match_thresh=0.8,
                 proximity_thresh=0.5, frame_rate=30):
        self.tracked_stracks = []   # type: list[STrack]
        self.lost_stracks = []      # type: list[STrack]
        self.removed_stracks = []   # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.ego = ego

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.proximity_thresh = proximity_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, frame):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        output_results = np.asarray(output_results, dtype=float).reshape(-1, 6)
        if len(output_results):
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]   # x1y1x2y2
            classes = output_results[:, 5]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]
        else:
            bboxes = np.empty((0, 4))
            scores = np.empty((0,))
            classes = np.empty((0,))
            dets = np.empty((0, 4))
            scores_keep = np.empty((0,))
            classes_keep = np.empty((0,))

        if len(dets) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                          (tlbr, s, c) in zip(dets, scores_keep, classes_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Inject ego-motion (camera/global) compensation.
        warp = self.ego.estimate(frame)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        ious_dists = iou_distance(strack_pool, detections)
        ious_dists = fuse_score(ious_dists, detections)
        # Mahalanobis 2-DOF gate on top of the IoU cost.
        ious_dists = apply_mahalanobis_gate(strack_pool, detections, ious_dists)
        dists = ious_dists

        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.track_high_thresh
            inds_low = scores > self.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = np.empty((0, 4))
            scores_second = np.empty((0,))
            classes_second = np.empty((0,))

        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                                 (tlbr, s, c) in zip(dets_second, scores_second, classes_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track
                             if strack_pool[i].state == TrackState.Tracked]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = iou_distance(unconfirmed, detections)
        ious_dists = fuse_score(ious_dists, detections)
        dists = ious_dists

        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
