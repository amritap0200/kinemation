"""
Multi-Person Tracking Module

Provides IoU-based Hungarian algorithm matching for consistent person IDs across frames.
Based on tracking logic from acmpesuecc/kinemation videopose3d-temporal-smoothing branch.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter1d


def get_bbox_from_keypoints(keypoints):
    """
    Compute bounding box from keypoints.
    
    Args:
        keypoints: array of shape (N_joints, 2) with (x, y) coordinates
        
    Returns:
        bbox: [x_min, y_min, x_max, y_max] or None if no valid keypoints
    """
    valid = keypoints[(keypoints[:, 0] > 0) & (keypoints[:, 1] > 0)]
    if len(valid) == 0:
        return None
    return np.array([
        valid[:, 0].min(),
        valid[:, 1].min(),
        valid[:, 0].max(),
        valid[:, 1].max()
    ])


def compute_iou(bbox_a, bbox_b):
    """
    Compute Intersection over Union between two bounding boxes.
    
    Args:
        bbox_a, bbox_b: arrays [x_min, y_min, x_max, y_max]
        
    Returns:
        iou: float between 0 and 1
    """
    xi1 = max(bbox_a[0], bbox_b[0])
    yi1 = max(bbox_a[1], bbox_b[1])
    xi2 = min(bbox_a[2], bbox_b[2])
    yi2 = min(bbox_a[3], bbox_b[3])
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    if inter_area == 0:
        return 0.0
    
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union_area = area_a + area_b - inter_area
    
    return inter_area / union_area


def track_people(raw_keypoints, iou_threshold=0.3):
    """
    Track people across frames using IoU-based Hungarian algorithm.
    
    Args:
        raw_keypoints: array of shape (N_frames, max_people, n_joints, 2)
        iou_threshold: minimum IoU to consider a match
        
    Returns:
        tracked: array of same shape with consistent person IDs (track slots)
    """
    N, max_people, n_joints, _ = raw_keypoints.shape
    tracked = np.zeros_like(raw_keypoints)
    
    # Initialize first frame
    tracked[0] = raw_keypoints[0]
    prev_boxes = [get_bbox_from_keypoints(raw_keypoints[0, p]) for p in range(max_people)]
    
    for frame_idx in range(1, N):
        frame_kps = raw_keypoints[frame_idx]
        
        # Get detections for current frame
        det_boxes = []
        det_indices = []
        for d in range(max_people):
            bbox = get_bbox_from_keypoints(frame_kps[d])
            if bbox is not None:
                det_boxes.append(bbox)
                det_indices.append(d)
        
        if not det_boxes:
            # No detections in this frame
            continue
        
        # Build cost matrix (1 - IoU) for Hungarian algorithm
        cost_matrix = np.ones((max_people, len(det_boxes)))
        for track_id in range(max_people):
            if prev_boxes[track_id] is None:
                continue
            for det_id, det_bbox in enumerate(det_boxes):
                iou = compute_iou(prev_boxes[track_id], det_bbox)
                cost_matrix[track_id, det_id] = 1.0 - iou
        
        # Run Hungarian algorithm
        track_ids, det_ids = linear_sum_assignment(cost_matrix)
        
        assigned_tracks = set()
        assigned_dets = set()
        
        # Assign matched pairs
        for track_id, det_id in zip(track_ids, det_ids):
            if cost_matrix[track_id, det_id] > (1 - iou_threshold):
                # IoU too low, skip this match
                continue
            orig_det_idx = det_indices[det_id]
            tracked[frame_idx, track_id] = frame_kps[orig_det_idx]
            prev_boxes[track_id] = det_boxes[det_id]
            assigned_tracks.add(track_id)
            assigned_dets.add(det_id)
        
        # Assign leftover detections to empty track slots
        leftover_dets = [det_indices[d] for d in range(len(det_boxes)) if d not in assigned_dets]
        empty_tracks = [t for t in range(max_people) if t not in assigned_tracks]
        
        for track_slot, orig_det_idx in zip(empty_tracks, leftover_dets):
            tracked[frame_idx, track_slot] = frame_kps[orig_det_idx]
            prev_boxes[track_slot] = get_bbox_from_keypoints(frame_kps[orig_det_idx])
    
    return tracked


def filter_short_tracks(tracked_keypoints, min_frames=10):
    """
    Remove short detection runs that are likely false positives.
    
    Args:
        tracked_keypoints: array of shape (N, max_people, n_joints, 2)
        min_frames: minimum number of consecutive frames for valid track
        
    Returns:
        filtered: array with short tracks zeroed out
    """
    N, max_people = tracked_keypoints.shape[:2]
    filtered = tracked_keypoints.copy()
    
    for person_id in range(max_people):
        # Find frames where this person has a detection
        has_detection = np.array([
            tracked_keypoints[i, person_id].max() > 0 for i in range(N)
        ])
        
        # Find contiguous runs
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
                    # Wipe this short run
                    filtered[run_start:i, person_id] = 0.0
                in_run = False
    
    return filtered


def smooth_track(keypoints_track, sigma=2):
    """
    Smooth a single person's keypoint trajectory using Gaussian filter.
    Only smooths within detected frames (no bleeding across gaps).
    
    Args:
        keypoints_track: array of shape (N, n_joints, 2)
        sigma: Gaussian filter sigma
        
    Returns:
        smoothed: array of same shape
    """
    N, n_joints, n_coords = keypoints_track.shape
    smoothed = keypoints_track.copy()
    
    # Find frames with valid detection
    has_detection = np.array([keypoints_track[i].max() > 0 for i in range(N)])
    
    if has_detection.sum() < 3:
        return smoothed
    
    # Find contiguous detection runs
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
    
    # Smooth each run independently
    for start, end in runs:
        if end - start < 3:
            continue
        
        for joint_idx in range(n_joints):
            for coord in range(n_coords):
                segment = keypoints_track[start:end, joint_idx, coord].astype(np.float64)
                valid = segment > 0
                
                if valid.sum() < 2:
                    continue
                
                # Fill small gaps within the run
                indices = np.arange(len(segment))
                filled = np.interp(indices, indices[valid], segment[valid])
                
                # Apply Gaussian smoothing
                smooth = gaussian_filter1d(filled, sigma=sigma)
                
                # Write back only to frames with detections
                result = smoothed[start:end, joint_idx, coord]
                result[valid] = smooth[valid]
                smoothed[start:end, joint_idx, coord] = result
    
    return smoothed.astype(np.float32)


def smooth_all_tracks(tracked_keypoints, sigma=2):
    """
    Smooth all person tracks in the sequence.
    
    Args:
        tracked_keypoints: array of shape (N, max_people, n_joints, 2)
        sigma: Gaussian filter sigma
        
    Returns:
        smoothed: array of same shape
    """
    N, max_people = tracked_keypoints.shape[:2]
    smoothed = np.zeros_like(tracked_keypoints)
    
    for person_id in range(max_people):
        smoothed[:, person_id] = smooth_track(tracked_keypoints[:, person_id], sigma=sigma)
    
    return smoothed


class PersonTracker:
    """
    Stateful person tracker for online processing.
    """
    
    def __init__(self, max_people=10, iou_threshold=0.3):
        self.max_people = max_people
        self.iou_threshold = iou_threshold
        self.prev_boxes = [None] * max_people
        self.frame_count = 0
    
    def update(self, detections):
        """
        Update tracker with new frame detections.
        
        Args:
            detections: list of keypoint arrays, one per detected person
            
        Returns:
            tracked: array of shape (max_people, n_joints, 2) with consistent IDs
        """
        if len(detections) == 0 or detections[0] is None:
            self.frame_count += 1
            return np.zeros((self.max_people, 17, 2), dtype=np.float32)
        
        n_joints = detections[0].shape[0] if len(detections) > 0 else 17
        tracked = np.zeros((self.max_people, n_joints, 2), dtype=np.float32)
        
        if self.frame_count == 0:
            # First frame - assign directly
            for i, det in enumerate(detections[:self.max_people]):
                tracked[i] = np.array(det, dtype=np.float32)
                self.prev_boxes[i] = get_bbox_from_keypoints(tracked[i])
        else:
            # Compute bounding boxes for detections
            det_boxes = []
            for det in detections:
                det_arr = np.array(det, dtype=np.float32)
                bbox = get_bbox_from_keypoints(det_arr)
                det_boxes.append((det_arr, bbox))
            
            # Build cost matrix
            cost_matrix = np.ones((self.max_people, len(det_boxes)))
            for track_id in range(self.max_people):
                if self.prev_boxes[track_id] is None:
                    continue
                for det_id, (_, det_bbox) in enumerate(det_boxes):
                    if det_bbox is not None:
                        iou = compute_iou(self.prev_boxes[track_id], det_bbox)
                        cost_matrix[track_id, det_id] = 1.0 - iou
            
            # Hungarian matching
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
            
            # Assign unmatched detections to empty slots
            leftover = [(i, det_boxes[i]) for i in range(len(det_boxes)) if i not in assigned_dets]
            empty = [t for t in range(self.max_people) if t not in assigned_tracks]
            
            for slot, (_, (det_arr, det_bbox)) in zip(empty, leftover):
                tracked[slot] = det_arr
                self.prev_boxes[slot] = det_bbox
        
        self.frame_count += 1
        return tracked
    
    def reset(self):
        """Reset tracker state."""
        self.prev_boxes = [None] * self.max_people
        self.frame_count = 0
