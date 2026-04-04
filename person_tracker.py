"""Multi-person tracking module"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter1d


# Get bounding box from keypoints
def get_bbox_from_keypoints(keypoints):
    valid = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
    if len(valid) == 0:
        return None
    return np.array([valid[:, 0].min(), valid[:, 1].min(), valid[:, 0].max(), valid[:, 1].max()])


# Compute IoU between two boxes
def compute_iou(bbox_a, bbox_b):
    xi1 = max(bbox_a[0], bbox_b[0])
    yi1 = max(bbox_a[1], bbox_b[1])
    xi2 = min(bbox_a[2], bbox_b[2])
    yi2 = min(bbox_a[3], bbox_b[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter_area == 0:
        return 0.0
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    return inter_area / (area_a + area_b - inter_area)


# Track people across frames using Hungarian algorithm
def track_people(raw_keypoints, iou_threshold=0.3):
    N, max_people, n_joints, _ = raw_keypoints.shape
    tracked = np.zeros_like(raw_keypoints)
    tracked[0] = raw_keypoints[0]
    prev_boxes = [get_bbox_from_keypoints(raw_keypoints[0, p]) for p in range(max_people)]
    
    for frame_idx in range(1, N):
        frame_kps = raw_keypoints[frame_idx]
        det_boxes = []
        det_indices = []
        for d in range(max_people):
            bbox = get_bbox_from_keypoints(frame_kps[d])
            if bbox is not None:
                det_boxes.append(bbox)
                det_indices.append(d)
        
        if not det_boxes:
            continue
        
        cost_matrix = np.ones((max_people, len(det_boxes)))
        for track_id in range(max_people):
            if prev_boxes[track_id] is None:
                continue
            for det_id, det_bbox in enumerate(det_boxes):
                iou = compute_iou(prev_boxes[track_id], det_bbox)
                cost_matrix[track_id, det_id] = 1.0 - iou
        
        track_ids, det_ids = linear_sum_assignment(cost_matrix)
        assigned_tracks = set()
        assigned_dets = set()
        
        for track_id, det_id in zip(track_ids, det_ids):
            if cost_matrix[track_id, det_id] > (1 - iou_threshold):
                continue
            orig_det_idx = det_indices[det_id]
            tracked[frame_idx, track_id] = frame_kps[orig_det_idx]
            prev_boxes[track_id] = det_boxes[det_id]
            assigned_tracks.add(track_id)
            assigned_dets.add(det_id)
        
        leftover_dets = [det_indices[d] for d in range(len(det_boxes)) if d not in assigned_dets]
        empty_tracks = [t for t in range(max_people) if t not in assigned_tracks]
        
        for track_slot, orig_det_idx in zip(empty_tracks, leftover_dets):
            tracked[frame_idx, track_slot] = frame_kps[orig_det_idx]
            prev_boxes[track_slot] = get_bbox_from_keypoints(frame_kps[orig_det_idx])
    
    return tracked


# Filter short tracks
def filter_short_tracks(tracked_keypoints, min_frames=10):
    N, max_people = tracked_keypoints.shape[:2]
    filtered = tracked_keypoints.copy()
    
    for person_id in range(max_people):
        has_detection = np.array([tracked_keypoints[i, person_id].max() > 0 for i in range(N)])
        in_run = False
        run_start = 0
        
        for i in range(N + 1):
            active = i < N and has_detection[i]
            if active and not in_run:
                run_start = i
                in_run = True
            elif not active and in_run:
                run_length = i - run_start
                if run_length < min_frames:
                    filtered[run_start:i, person_id] = 0.0
                in_run = False
    
    return filtered


# Smooth a single track
def smooth_track(keypoints_track, sigma=2):
    N, n_joints, n_coords = keypoints_track.shape
    smoothed = keypoints_track.copy()
    has_detection = np.array([keypoints_track[i].max() > 0 for i in range(N)])
    
    if has_detection.sum() < 3:
        return smoothed
    
    runs = []
    in_run = False
    run_start = 0
    
    for i in range(N):
        if has_detection[i] and not in_run:
            run_start = i
            in_run = True
        elif not has_detection[i] and in_run:
            runs.append((run_start, i))
            in_run = False
    if in_run:
        runs.append((run_start, N))
    
    for start, end in runs:
        if end - start < 3:
            continue
        for joint_idx in range(n_joints):
            for coord in range(n_coords):
                segment = keypoints_track[start:end, joint_idx, coord].astype(np.float64)
                valid = segment > 0
                if valid.sum() < 2:
                    continue
                indices = np.arange(len(segment))
                filled = np.interp(indices, indices[valid], segment[valid])
                smooth = gaussian_filter1d(filled, sigma=sigma)
                result = smoothed[start:end, joint_idx, coord]
                result[valid] = smooth[valid]
                smoothed[start:end, joint_idx, coord] = result
    
    return smoothed.astype(np.float32)


# Smooth all tracks
def smooth_all_tracks(tracked_keypoints, sigma=2):
    N, max_people = tracked_keypoints.shape[:2]
    smoothed = np.zeros_like(tracked_keypoints)
    for person_id in range(max_people):
        smoothed[:, person_id] = smooth_track(tracked_keypoints[:, person_id], sigma=sigma)
    return smoothed


# Stateful person tracker
class PersonTracker:
    def __init__(self, max_people=10, iou_threshold=0.3):
        self.max_people = max_people
        self.iou_threshold = iou_threshold
        self.prev_boxes = [None] * max_people
        self.frame_count = 0
    
    def update(self, detections):
        if len(detections) == 0 or detections[0] is None:
            self.frame_count += 1
            return np.zeros((self.max_people, 17, 2), dtype=np.float32)
        
        n_joints = detections[0].shape[0] if len(detections) > 0 else 17
        tracked = np.zeros((self.max_people, n_joints, 2), dtype=np.float32)
        
        if self.frame_count == 0:
            for i, det in enumerate(detections[:self.max_people]):
                tracked[i] = np.array(det, dtype=np.float32)
                self.prev_boxes[i] = get_bbox_from_keypoints(tracked[i])
        else:
            det_boxes = []
            for det in detections:
                det_arr = np.array(det, dtype=np.float32)
                bbox = get_bbox_from_keypoints(det_arr)
                det_boxes.append((det_arr, bbox))
            
            cost_matrix = np.ones((self.max_people, len(det_boxes)))
            for track_id in range(self.max_people):
                if self.prev_boxes[track_id] is None:
                    continue
                for det_id, (_, det_bbox) in enumerate(det_boxes):
                    if det_bbox is not None:
                        iou = compute_iou(self.prev_boxes[track_id], det_bbox)
                        cost_matrix[track_id, det_id] = 1.0 - iou
            
            track_ids, det_ids = linear_sum_assignment(cost_matrix)
            assigned_tracks = set()
            assigned_dets = set()
            
            for track_id, det_id in zip(track_ids, det_ids):
                if cost_matrix[track_id, det_id] > (1 - self.iou_threshold):
                    continue
                det_arr, det_bbox = det_boxes[det_id]
                tracked[track_id] = det_arr
                self.prev_boxes[track_id] = det_bbox
                assigned_tracks.add(track_id)
                assigned_dets.add(det_id)
            
            leftover = [(i, det_boxes[i]) for i in range(len(det_boxes)) if i not in assigned_dets]
            empty = [t for t in range(self.max_people) if t not in assigned_tracks]
            
            for slot, (_, (det_arr, det_bbox)) in zip(empty, leftover):
                tracked[slot] = det_arr
                self.prev_boxes[slot] = det_bbox
        
        self.frame_count += 1
        return tracked
    
    def reset(self):
        self.prev_boxes = [None] * self.max_people
        self.frame_count = 0
